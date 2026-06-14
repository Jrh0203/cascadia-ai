"""Evaluate the selected public beam set-ranker checkpoint on sealed test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import blake3

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.public_beam_set_model import (
    PublicBeamSetModelConfig,
    PublicBeamSetRanker,
)
from cascadia_mlx.public_beam_set_train import evaluate_public_beam_set
from cascadia_mlx.public_beam_value_dataset import PublicBeamValueDataset


def validation_gates(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "centered_advantage_correlation": metrics["centered_advantage_correlation"] >= 0.70,
        "top_value_recall": metrics["top_value_recall"] >= 0.40,
        "mean_top_action_regret": metrics["mean_top_action_regret"] <= 0.35,
    }


def evaluate_public_beam_set_test(
    run_dir: Path,
    test_dataset: Path,
    *,
    group_batch_size: int = 8,
) -> dict[str, object]:
    dataset = PublicBeamValueDataset(test_dataset)
    if dataset.split != "test":
        raise ValueError("public beam set advancement requires the sealed test split")
    try:
        run = json.loads((run_dir / "run.json").read_text())
        best = json.loads((run_dir / "best.json").read_text())
        training = run["training"]
        expected_teacher = run["datasets"]["teacher"]
        validation = best["validation"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read public beam set run manifest: {error}") from error
    if run.get("kind") != "public-beam-set-ranking":
        raise ValueError("run is not a public beam set-ranking experiment")
    passed_validation = validation_gates(validation)
    if not all(passed_validation.values()):
        raise ValueError("sealed test access denied because validation gates did not pass")
    if dataset.manifest["teacher"] != expected_teacher:
        raise ValueError("test dataset teacher does not match the frozen training teacher")
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: PublicBeamSetRanker(
            PublicBeamSetModelConfig.from_dict(values)
        ),
    )
    metrics = evaluate_public_beam_set(model, dataset, group_batch_size)
    gates = {
        "centered_advantage_correlation": metrics["centered_advantage_correlation"] >= 0.65,
        "top_value_recall": metrics["top_value_recall"] >= 0.35,
        "mean_top_action_regret": metrics["mean_top_action_regret"] <= 0.45,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "validation": validation,
        "validation_gates": passed_validation,
        "test_dataset": str(test_dataset.resolve()),
        "test_manifest_blake3": _checksum(test_dataset / "dataset.json"),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = run_dir / "test-report.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--group-batch-size", type=int, default=8)
    args = parser.parse_args()
    report = evaluate_public_beam_set_test(
        args.run_dir,
        args.test_dataset,
        group_batch_size=args.group_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
