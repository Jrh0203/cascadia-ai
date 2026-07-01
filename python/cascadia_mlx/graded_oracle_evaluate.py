"""Cross-host validation and performance evaluation for ADR 0081."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import blake3

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.graded_oracle_dataset import GradedOracleDataset
from cascadia_mlx.graded_oracle_metrics import (
    benchmark_graded_oracle,
    evaluate_graded_oracle,
    graded_oracle_validation_gates,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)


def evaluate_validation(
    run_dir: str | Path,
    validation_dataset: str | Path,
) -> dict[str, object]:
    """Evaluate one selected replica without opening the sealed test split."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    training = run["training"]
    dataset = GradedOracleDataset(validation_dataset)
    if dataset.split != "validation":
        raise ValueError("ADR 0081 evaluation accepts only the validation split")
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
    metrics = evaluate_graded_oracle(model, dataset, group_batch_size=64)
    performance = benchmark_graded_oracle(model, dataset)
    host = socket.gethostname().split(".")[0]
    gates = graded_oracle_validation_gates(
        metrics,
        performance_by_host={host: performance},
    )
    report = {
        "schema_version": 1,
        "experiment_id": "complete-action-graded-oracle-ranker-v1",
        "evaluation_kind": "validation-cross-host",
        "host": host,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(checkpoint / "model.safetensors"),
        "source_run_manifest_blake3": _checksum(run_dir / "run.json"),
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": manifest_hash,
        "metrics": metrics,
        "performance": performance,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json_atomic(run_dir / f"validation-report-{host}.json", report)
    return report


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
