"""Audit the fixed-width frontier-anchored R1200 proposal ceiling."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.graded_oracle_dataset import GradedOracleDataset
from cascadia_mlx.graded_oracle_frontier_anchor import (
    evaluate_frontier_anchored_target_ceiling,
    frontier_anchored_target_ceiling_gates,
)

EXPERIMENT_ID = "complete-action-frontier-anchored-set-ranker-v1"
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
    "john4": "john4",
}


def audit_target_ceiling(dataset_path: str | Path) -> dict[str, Any]:
    """Audit one already-open train or validation split."""
    started = time.perf_counter()
    dataset = GradedOracleDataset(dataset_path)
    if dataset.split not in {"train", "validation"}:
        raise ValueError("frontier-anchored ceiling accepts only train or validation")
    manifest_hash = _checksum(dataset.root / "dataset.json")
    scientific = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "dataset": {
            "dataset_id": dataset.manifest["dataset_id"],
            "split": dataset.split,
            "games": dataset.manifest["completed_games"],
            "seeds": dataset.manifest["seeds"],
            "groups": dataset.group_count,
            "candidates": dataset.candidate_count,
            "manifest_blake3": manifest_hash,
        },
        "ceiling": evaluate_frontier_anchored_target_ceiling(dataset),
        "test_split_opened": False,
    }
    gates = frontier_anchored_target_ceiling_gates(scientific["ceiling"])
    host_name = socket.gethostname().split(".")[0]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit_kind": "frontier-anchored-r1200-target-ceiling",
        "host": HOST_ALIASES.get(host_name, host_name),
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "gates": gates,
        "passed": all(gates.values()),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "test_split_opened": False,
        },
    }


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_target_ceiling(args.dataset)
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
