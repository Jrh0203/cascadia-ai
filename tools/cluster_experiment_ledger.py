#!/usr/bin/env python3
"""Maintain the durable research experiment ledger shown on the dashboard."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_LEDGER = Path("artifacts/cluster/research-experiments-v1.json")
STATUSES = {"planned", "running", "completed", "cancelled"}
OUTCOMES = {"pending", "passed", "failed", "inconclusive", "invalid"}
TONES = {"good", "bad", "warn", "neutral"}


class LedgerError(RuntimeError):
    """Raised when the experiment ledger violates its durable schema."""


def unix_millis() -> int:
    return time.time_ns() // 1_000_000


def empty_ledger(now_ms: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_unix_ms": unix_millis() if now_ms is None else now_ms,
        "experiments": [],
    }


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{field} must be a nonempty string")
    return value


def _unique_strings(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise LedgerError(f"{field} must be a unique list of nonempty strings")
    return value


def validate_experiment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LedgerError("experiment must be an object")
    experiment_id = _nonempty(value.get("id"), "experiment id")
    for field in ("title", "hypothesis", "summary"):
        _nonempty(value.get(field), f"experiment {experiment_id} {field}")
    if value.get("status") not in STATUSES:
        raise LedgerError(f"experiment {experiment_id} has an invalid status")
    if value.get("outcome") not in OUTCOMES:
        raise LedgerError(f"experiment {experiment_id} has an invalid outcome")
    updated = value.get("updated_unix_ms")
    if not isinstance(updated, int) or updated < 0:
        raise LedgerError(
            f"experiment {experiment_id} updated_unix_ms must be nonnegative"
        )
    started = value.get("started_unix_ms")
    completed = value.get("completed_unix_ms")
    if started is not None and (not isinstance(started, int) or started < 0):
        raise LedgerError(
            f"experiment {experiment_id} started_unix_ms must be nonnegative or null"
        )
    if completed is not None and (
        not isinstance(completed, int) or completed < 0
    ):
        raise LedgerError(
            f"experiment {experiment_id} completed_unix_ms must be nonnegative or null"
        )
    if value["status"] == "running" and started is None:
        raise LedgerError(f"running experiment {experiment_id} requires a start time")
    if value["status"] == "completed":
        if completed is None or value["outcome"] == "pending":
            raise LedgerError(
                f"completed experiment {experiment_id} requires time and outcome"
            )
    elif completed is not None:
        raise LedgerError(
            f"non-completed experiment {experiment_id} cannot have a completion time"
        )
    if started is not None and completed is not None and completed < started:
        raise LedgerError(f"experiment {experiment_id} completes before it starts")
    for field in ("hosts", "tags", "task_ids", "notes"):
        _unique_strings(value.get(field, []), f"experiment {experiment_id} {field}")
    metrics = value.get("metrics", [])
    if not isinstance(metrics, list):
        raise LedgerError(f"experiment {experiment_id} metrics must be a list")
    for metric in metrics:
        if not isinstance(metric, dict):
            raise LedgerError(f"experiment {experiment_id} metric must be an object")
        _nonempty(metric.get("label"), f"experiment {experiment_id} metric label")
        _nonempty(metric.get("value"), f"experiment {experiment_id} metric value")
        if metric.get("tone") not in TONES:
            raise LedgerError(
                f"experiment {experiment_id} metric has an invalid tone"
            )
    criteria = value.get("criteria", [])
    if not isinstance(criteria, list):
        raise LedgerError(f"experiment {experiment_id} criteria must be a list")
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise LedgerError(
                f"experiment {experiment_id} criterion must be an object"
            )
        _nonempty(
            criterion.get("label"),
            f"experiment {experiment_id} criterion label",
        )
        if criterion.get("passed") not in (True, False, None):
            raise LedgerError(
                f"experiment {experiment_id} criterion passed must be boolean or null"
            )
        observed = criterion.get("observed")
        if observed is not None:
            _nonempty(
                observed,
                f"experiment {experiment_id} criterion observed",
            )
    artifacts = value.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise LedgerError(f"experiment {experiment_id} artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise LedgerError(
                f"experiment {experiment_id} artifact must be an object"
            )
        _nonempty(
            artifact.get("label"),
            f"experiment {experiment_id} artifact label",
        )
        artifact_path = Path(
            _nonempty(
                artifact.get("path"),
                f"experiment {experiment_id} artifact path",
            )
        )
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise LedgerError(
                f"experiment {experiment_id} artifact path must stay inside the repository"
            )
    return value


def validate_ledger(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LedgerError("experiment ledger must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("unsupported experiment ledger schema")
    updated = value.get("updated_unix_ms")
    if not isinstance(updated, int) or updated < 0:
        raise LedgerError("updated_unix_ms must be nonnegative")
    experiments = value.get("experiments")
    if not isinstance(experiments, list):
        raise LedgerError("experiments must be a list")
    ids: set[str] = set()
    for experiment in experiments:
        validate_experiment(experiment)
        if experiment["id"] in ids:
            raise LedgerError(f"duplicate experiment id {experiment['id']}")
        ids.add(experiment["id"])
    return value


def read_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_ledger()
    return validate_ledger(json.loads(path.read_text()))


def write_ledger(path: Path, value: dict[str, Any]) -> None:
    validate_ledger(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@contextmanager
def locked_ledger(path: Path) -> Iterator[dict[str, Any]]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = read_ledger(path)
        yield value
        value["updated_unix_ms"] = unix_millis()
        write_ledger(path, value)


def upsert(ledger: dict[str, Any], experiment: dict[str, Any]) -> None:
    validate_experiment(experiment)
    for index, existing in enumerate(ledger["experiments"]):
        if existing["id"] == experiment["id"]:
            ledger["experiments"][index] = experiment
            return
    ledger["experiments"].append(experiment)


def summary(ledger: dict[str, Any]) -> dict[str, Any]:
    counts = {status: 0 for status in STATUSES}
    outcomes = {outcome: 0 for outcome in OUTCOMES}
    for experiment in ledger["experiments"]:
        counts[experiment["status"]] += 1
        outcomes[experiment["outcome"]] += 1
    return {
        "schema_version": ledger["schema_version"],
        "updated_unix_ms": ledger["updated_unix_ms"],
        "experiments": len(ledger["experiments"]),
        "statuses": counts,
        "outcomes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init").add_argument("--force", action="store_true")
    upsert_parser = subparsers.add_parser("upsert")
    upsert_parser.add_argument("--spec", type=Path, required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("status")
    args = parser.parse_args()

    if args.command == "init":
        if args.ledger.exists() and not args.force:
            raise LedgerError("experiment ledger already exists; pass --force")
        write_ledger(args.ledger, empty_ledger())
        report = summary(read_ledger(args.ledger))
    elif args.command == "upsert":
        experiment = json.loads(args.spec.read_text())
        with locked_ledger(args.ledger) as ledger:
            upsert(ledger, experiment)
        report = summary(read_ledger(args.ledger))
    elif args.command == "validate":
        validate_ledger(json.loads(args.ledger.read_text()))
        report = {"valid": True, "ledger": str(args.ledger)}
    else:
        report = summary(read_ledger(args.ledger))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
