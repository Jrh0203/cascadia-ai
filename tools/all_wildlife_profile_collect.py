#!/usr/bin/env python3
"""Fail-closed collection for externally sharded wildlife score profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules

EXACT_STATUSES = frozenset({"OPTIMAL", "FEASIBLE", "INFEASIBLE"})
ALLOWED_STATUSES = EXACT_STATUSES | {"UNKNOWN"}
TASK_FIELDS = (
    "task_index",
    "case_index",
    "case_id",
    "profile_index",
    "ruleset",
    "counts",
    "threshold",
    "upper",
    "score_profile",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _validate_witness(result: dict[str, Any], task: dict[str, Any]) -> None:
    tokens = rules.normalized_tokens(result["tokens"])
    counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if counts != tuple(task["counts"]):
        raise ValueError(f"task {task['task_index']}: witness count mismatch")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(rules.components(occupied)) != 1:
        raise ValueError(f"task {task['task_index']}: disconnected witness")
    breakdown = rules.score_tokens(tokens, task["ruleset"])
    if list(breakdown) != result["independent_score_breakdown"]:
        raise ValueError(f"task {task['task_index']}: witness score mismatch")
    if list(breakdown) != task["score_profile"]:
        raise ValueError(f"task {task['task_index']}: witness/profile mismatch")
    if sum(breakdown) < task["threshold"]:
        raise ValueError(f"task {task['task_index']}: witness below threshold")


def _validate_result(result: dict[str, Any], task: dict[str, Any]) -> None:
    for field in TASK_FIELDS:
        if result.get(field) != task[field]:
            raise ValueError(f"task {task['task_index']}: {field} mismatch")
    status = result.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"task {task['task_index']}: invalid status {status!r}")
    for field in ("elapsed_seconds", "branches", "conflicts"):
        value = result.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"task {task['task_index']}: invalid {field}")
    if status in {"OPTIMAL", "FEASIBLE"}:
        if result.get("tokens") is None:
            raise ValueError(f"task {task['task_index']}: feasible result has no witness")
        _validate_witness(result, task)
    elif any(
        result.get(field) is not None
        for field in ("tokens", "independent_score_breakdown", "objective")
    ):
        raise ValueError(f"task {task['task_index']}: non-feasible result has a witness")


def collect(
    taskset_path: Path,
    fleet_path: Path,
    shard_paths: list[Path],
    *,
    known_case_index: int,
    known_max_seconds: float,
    hard_case_indices: list[int],
) -> dict[str, Any]:
    taskset = _read_json(taskset_path)
    fleet = _read_json(fleet_path)
    taskset_sha = _sha256(taskset_path)
    if taskset.get("schema") != "all-wildlife-score-profile-taskset-v1":
        raise ValueError("unexpected taskset schema")
    if fleet.get("schema") != "all-wildlife-score-profile-fleet-v1":
        raise ValueError("unexpected fleet schema")
    if fleet.get("state") not in {"running", "completed"}:
        raise ValueError("fleet ledger is neither running nor completed")
    if fleet.get("taskset_sha256") != taskset_sha:
        raise ValueError("fleet/taskset hash mismatch")
    if taskset.get("task_count") != len(taskset.get("tasks", [])):
        raise ValueError("taskset task count mismatch")
    tasks = {task["task_index"]: task for task in taskset["tasks"]}
    if sorted(tasks) != list(range(len(tasks))):
        raise ValueError("task indices are not contiguous")

    current_hashes = {
        "rules_source_sha256": _sha256(Path("tools/all_wildlife_rules.py")),
        "exact_source_sha256": _sha256(Path("tools/all_wildlife_exact.py")),
        "exact_support_source_sha256": _sha256(Path("tools/cbddb_wildlife_exact.py")),
        "runner_source_sha256": _sha256(Path("tools/all_wildlife_profile_proof.py")),
    }
    fleet_hash_keys = {
        "rules_source_sha256": "rules_source_sha256",
        "exact_source_sha256": "exact_source_sha256",
        "exact_support_source_sha256": "exact_support_sha256",
        "runner_source_sha256": "runner_source_sha256",
    }
    for identity_key, fleet_key in fleet_hash_keys.items():
        if fleet.get(fleet_key) != current_hashes[identity_key]:
            raise ValueError(f"current source does not match fleet {fleet_key}")

    assignment_by_indices: dict[tuple[int, ...], dict[str, Any]] = {}
    assigned: list[int] = []
    for shard in fleet["shards"]:
        indices = tuple(shard["task_indices"])
        if indices in assignment_by_indices:
            raise ValueError("duplicate fleet shard assignment")
        assignment_by_indices[indices] = shard
        assigned.extend(indices)
    if sorted(assigned) != list(range(len(tasks))) or len(assigned) != len(set(assigned)):
        raise ValueError("fleet assignments do not cover the taskset exactly once")

    seen: dict[int, dict[str, Any]] = {}
    shard_rows = []
    seen_hosts = set()
    for path in shard_paths:
        shard = _read_json(path)
        if shard.get("schema") != "all-wildlife-score-profile-shard-v1":
            raise ValueError(f"{path}: unexpected shard schema")
        indices = tuple(shard.get("task_indices", []))
        assignment = assignment_by_indices.get(indices)
        if assignment is None:
            raise ValueError(f"{path}: task assignment mismatch")
        host = assignment["host"]
        if host in seen_hosts:
            raise ValueError(f"{path}: duplicate shard for {host}")
        seen_hosts.add(host)
        identity = shard.get("identity", {})
        expected_identity = {"taskset_sha256": taskset_sha, **current_hashes}
        if identity != expected_identity:
            raise ValueError(f"{path}: identity mismatch")
        configuration = shard.get("configuration", {})
        expected_configuration = {
            "seconds_per_profile": float(
                fleet["configuration"]["seconds_per_profile"]
            ),
            "jobs": fleet["configuration"]["jobs_per_host"],
            "solver_workers": fleet["configuration"]["solver_workers"],
            "connectivity_required": fleet["configuration"]["connectivity_required"],
            "seed": fleet["configuration"]["seed"],
        }
        if configuration != expected_configuration:
            raise ValueError(f"{path}: configuration mismatch")
        results = shard.get("results", [])
        if [row.get("task_index") for row in results] != sorted(indices):
            raise ValueError(f"{path}: result ordering or coverage mismatch")
        for result in results:
            index = result["task_index"]
            if index in seen:
                raise ValueError(f"duplicate result for task {index}")
            _validate_result(result, tasks[index])
            seen[index] = result
        shard_rows.append(
            {
                "host": host,
                "path": str(path),
                "sha256": _sha256(path),
                "task_indices": list(indices),
                "elapsed_seconds": shard["elapsed_seconds"],
            }
        )
    if seen_hosts != {shard["host"] for shard in fleet["shards"]}:
        raise ValueError("missing fleet shard")
    if sorted(seen) != list(range(len(tasks))):
        raise ValueError("collected results do not cover the taskset exactly once")

    case_summaries = []
    by_case: dict[int, list[dict[str, Any]]] = {}
    for index in sorted(seen):
        by_case.setdefault(seen[index]["case_index"], []).append(seen[index])
    taskset_cases = {case["case_index"]: case for case in taskset["cases"]}
    for case_index in sorted(by_case):
        case_results = by_case[case_index]
        case = taskset_cases[case_index]
        if len(case_results) != case["profile_count"]:
            raise ValueError(f"case {case_index}: profile coverage mismatch")
        statuses = Counter(row["status"] for row in case_results)
        case_summaries.append(
            {
                **case,
                "status_counts": dict(sorted(statuses.items())),
                "exact_profiles": sum(
                    row["status"] in EXACT_STATUSES for row in case_results
                ),
                "complete_exact": all(
                    row["status"] in EXACT_STATUSES for row in case_results
                ),
                "max_elapsed_seconds": max(
                    row["elapsed_seconds"] for row in case_results
                ),
                "total_branches": sum(row["branches"] for row in case_results),
                "exact_infeasible_profiles": [
                    row["profile_index"]
                    for row in case_results
                    if row["status"] == "INFEASIBLE"
                ],
            }
        )

    known = next(
        row for row in case_summaries if row["case_index"] == known_case_index
    )
    known_selected = (
        known["status_counts"] == {"INFEASIBLE": known["profile_count"]}
        and known["max_elapsed_seconds"] <= known_max_seconds
    )
    hard_selected = [
        case_index
        for case_index in hard_case_indices
        if next(
            row for row in case_summaries if row["case_index"] == case_index
        )["complete_exact"]
    ]
    selected = known_selected or bool(hard_selected)
    return {
        "schema": "all-wildlife-score-profile-collection-v1",
        "taskset": {
            "path": str(taskset_path),
            "sha256": taskset_sha,
            "task_count": len(tasks),
        },
        "fleet": {
            "path": str(fleet_path),
            "sha256": _sha256(fleet_path),
            "tag": fleet["tag"],
            "launch_utc": fleet["launch_utc"],
            "source_revision": fleet["source_revision"],
        },
        "shards": sorted(shard_rows, key=lambda row: row["host"]),
        "cases": case_summaries,
        "totals": {
            "profiles": len(seen),
            "exact_profiles": sum(
                row["status"] in EXACT_STATUSES for row in seen.values()
            ),
            "unknown_profiles": sum(
                row["status"] == "UNKNOWN" for row in seen.values()
            ),
            "exact_infeasible_profiles": sum(
                row["status"] == "INFEASIBLE" for row in seen.values()
            ),
        },
        "selection": {
            "selected": selected,
            "known_case_index": known_case_index,
            "known_max_seconds": known_max_seconds,
            "known_case_selected": known_selected,
            "hard_case_indices": hard_case_indices,
            "complete_hard_cases": hard_selected,
            "verdict": "SELECT" if selected else "REJECT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--fleet-ledger", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--known-case-index", type=int, required=True)
    parser.add_argument("--known-max-seconds", type=float, required=True)
    parser.add_argument("--hard-case-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = collect(
        args.taskset,
        args.fleet_ledger,
        args.shard,
        known_case_index=args.known_case_index,
        known_max_seconds=args.known_max_seconds,
        hard_case_indices=args.hard_case_index,
    )
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                **payload["totals"],
                **payload["selection"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
