"""Evaluate the selected action-delta checkpoint on an untouched test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import blake3

from cascadia_mlx.action_ranking_dataset import ActionRankingDataset
from cascadia_mlx.action_ranking_model import (
    ActionDeltaRanker,
    ActionRankingModelConfig,
)
from cascadia_mlx.action_ranking_train import action_ranking_adapter
from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.ranking_train import evaluate_ranking


def evaluate_action_ranking_test(
    run_dir: Path,
    test_dataset: Path,
    *,
    group_batch_size: int = 16,
) -> dict[str, object]:
    dataset = ActionRankingDataset(test_dataset)
    if dataset.split != "test":
        raise ValueError("action-ranking advancement requires the untouched test split")
    try:
        run = json.loads((run_dir / "run.json").read_text())
        training = run["training"]
        expected_teacher = run["datasets"]["teacher"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read action-ranking run manifest: {error}") from error
    if run.get("kind") != "action-delta-ranking":
        raise ValueError("run is not an action-delta ranking experiment")
    if dataset.manifest["teacher"] != expected_teacher:
        raise ValueError("test dataset teacher does not match the frozen training teacher")
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: ActionDeltaRanker(ActionRankingModelConfig.from_dict(values)),
    )
    metrics = evaluate_ranking(
        model,
        dataset,
        group_batch_size,
        adapter=action_ranking_adapter(),
    )
    gates = {
        "mean_top1_regret": metrics["mean_top1_regret"] <= 0.75,
        "pairwise_accuracy": metrics["pairwise_accuracy"] >= 0.65,
        "value_difference_correlation": metrics["value_difference_correlation"] >= 0.30,
        "top1_value_recall": metrics["top1_value_recall"] >= 0.45,
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
    report = evaluate_action_ranking_test(
        args.run_dir,
        args.test_dataset,
        group_batch_size=args.group_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
