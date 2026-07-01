"""Evaluate the selected conservative-advantage checkpoint on untouched test data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import blake3

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.conservative_advantage_dataset import ConservativeAdvantageDataset
from cascadia_mlx.conservative_advantage_model import (
    ConservativeAdvantageModel,
    ConservativeAdvantageModelConfig,
)
from cascadia_mlx.conservative_advantage_train import evaluate_conservative_advantage


def evaluate_conservative_advantage_test(
    run_dir: Path,
    test_dataset: Path,
    *,
    group_batch_size: int = 16,
) -> dict[str, object]:
    dataset = ConservativeAdvantageDataset(test_dataset)
    if dataset.split != "test":
        raise ValueError("conservative-advantage advancement requires the untouched test split")
    try:
        run = json.loads((run_dir / "run.json").read_text())
        training = run["training"]
        expected_teacher = run["datasets"]["teacher"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read conservative-advantage run manifest: {error}") from error
    if run.get("kind") != "conservative-advantage":
        raise ValueError("run is not a conservative-advantage experiment")
    if dataset.manifest["teacher"] != expected_teacher:
        raise ValueError("test dataset teacher does not match the frozen training teacher")
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: ConservativeAdvantageModel(
            ConservativeAdvantageModelConfig.from_dict(values)
        ),
    )
    metrics = evaluate_conservative_advantage(model, dataset, group_batch_size)
    gates = {
        "improves_zero_predictor": metrics["mean_squared_error"]
        < metrics["zero_predictor_mean_squared_error"],
        "mean_policy_regret": metrics["mean_policy_regret"] <= 0.20,
        "exact_policy_agreement": metrics["exact_policy_agreement"] >= 0.65,
        "anchor_false_positive_rate": metrics["anchor_false_positive_rate"] <= 0.20,
        "selected_challenger_recall": metrics["selected_challenger_recall"] >= 0.35,
        "lower_bound_correlation": metrics["lower_bound_correlation"] >= 0.50,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
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
    parser.add_argument("--group-batch-size", type=int, default=16)
    args = parser.parse_args()
    report = evaluate_conservative_advantage_test(
        args.run_dir,
        args.test_dataset,
        group_batch_size=args.group_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
