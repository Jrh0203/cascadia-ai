"""Cross-host validation for the frontier-anchored set proposer."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.graded_oracle_dataset import GradedOracleDataset
from cascadia_mlx.graded_oracle_frontier_anchor import (
    benchmark_frontier_anchored,
    evaluate_frontier_anchored,
    frontier_anchored_validation_gates,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)

EXPERIMENT_ID = "complete-action-frontier-anchored-set-ranker-v1"
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
    "john4": "john4",
}


def evaluate_validation(
    run_dir: str | Path,
    validation_dataset: str | Path,
) -> dict[str, object]:
    """Evaluate one selected replica without opening the sealed test."""
    started = time.perf_counter()
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    _validate_run_kind(run)
    training = run["training"]
    dataset = GradedOracleDataset(validation_dataset)
    if dataset.split != "validation":
        raise ValueError("frontier-anchored evaluation accepts only validation")
    manifest_hash = _checksum(dataset.root / "dataset.json")
    if manifest_hash != run["datasets"]["validation_manifest_blake3"]:
        raise ValueError("validation manifest does not match the training run")

    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
    )
    metrics = evaluate_frontier_anchored(model, dataset, group_batch_size=64)
    performance = benchmark_frontier_anchored(model, dataset)
    host_name = socket.gethostname().split(".")[0]
    host = HOST_ALIASES.get(host_name, host_name)
    gates = frontier_anchored_validation_gates(
        metrics,
        performance_by_host={host: performance},
    )
    scientific = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(checkpoint / "model.safetensors"),
        "source_run_manifest_blake3": _checksum(run_dir / "run.json"),
        "dataset": _dataset_identity(dataset, manifest_hash),
        "metrics": metrics,
        "test_split_opened": False,
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_kind": "validation-cross-host",
        "host": host,
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "performance": performance,
        "gates": gates,
        "passed": all(gates.values()),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "test_split_opened": False,
        },
    }
    _write_json_atomic(run_dir / f"validation-report-{host}.json", report)
    return report


def _validate_run_kind(run: dict[str, Any]) -> None:
    if run.get("kind") != "graded-oracle-frontier-anchored-ranking":
        raise ValueError("run is not a frontier-anchored set-ranking replica")


def _dataset_identity(
    dataset: GradedOracleDataset,
    manifest_hash: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.manifest["dataset_id"],
        "split": dataset.split,
        "games": dataset.manifest["completed_games"],
        "seeds": dataset.manifest["seeds"],
        "groups": dataset.group_count,
        "candidates": dataset.candidate_count,
        "manifest_blake3": manifest_hash,
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_validation(args.run_dir, args.validation_dataset)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
