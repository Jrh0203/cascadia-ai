"""Evaluate the selected imitation checkpoint on the untouched test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import blake3

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.imitation_dataset import ImitationDataset
from cascadia_mlx.imitation_model import ImitationModelConfig, SharedStateActionRanker
from cascadia_mlx.imitation_train import imitation_adapter
from cascadia_mlx.ranking_train import evaluate_ranking


def evaluate_imitation_test(
    run_dir: Path,
    test_dataset: Path,
    *,
    group_batch_size: int = 16,
) -> dict[str, object]:
    dataset = ImitationDataset(test_dataset)
    if dataset.split != "test":
        raise ValueError("imitation advancement requires the untouched test split")
    try:
        run = json.loads((run_dir / "run.json").read_text())
        training = run["training"]
        expected_teacher = run["datasets"]["teacher"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read imitation run manifest: {error}") from error
    if run.get("kind") != "canonical-action-imitation":
        raise ValueError("run is not a canonical action-imitation experiment")
    if dataset.manifest["teacher"] != expected_teacher:
        raise ValueError("test dataset teacher does not match the frozen training teacher")
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: SharedStateActionRanker(
            ImitationModelConfig.from_dict(values)
        ),
    )
    metrics = evaluate_ranking(
        model,
        dataset,
        group_batch_size,
        adapter=imitation_adapter(),
    )
    final_report = json.loads((run_dir / "final-report.json").read_text())
    best = json.loads((run_dir / "best.json").read_text())
    gates = {
        "validation_improved": float(best["selection_loss"])
        < float(final_report["initial_validation"]["selection_loss"]),
        "top1_accuracy": metrics["top1_accuracy"] >= 0.20,
        "top5_recall": metrics["top5_recall"] >= 0.55,
        "mean_reciprocal_rank": metrics["mean_reciprocal_rank"] >= 0.40,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "test_dataset": str(test_dataset.resolve()),
        "test_manifest_blake3": blake3.blake3(
            (test_dataset / "dataset.json").read_bytes()
        ).hexdigest(),
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
    report = evaluate_imitation_test(
        args.run_dir,
        args.test_dataset,
        group_batch_size=args.group_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
