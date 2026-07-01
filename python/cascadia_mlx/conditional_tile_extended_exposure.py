"""Extended full-cache target-only tile training for ADR 0118."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cascadia_mlx.conditional_tile_target_only import (
    BATCH_SIZE,
    OBJECTIVE_ID,
    SEED,
    target_only_tile_loss,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    HIDDEN_DIM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    StageTrainingConfig,
    evaluate_integrated,
    evaluate_mixed_stage_ceiling,
    membership_stage_selection_key,
    replay_stage,
    train_stage_with_loss,
)

EXPERIMENT_ID = "conditional-tile-extended-exposure-v1"
EPOCHS = 200


def frozen_config() -> StageTrainingConfig:
    """Return ADR 0116's contract with only the epoch budget increased."""
    return StageTrainingConfig(
        stage="tile",
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        hidden_dim=HIDDEN_DIM,
    )


def train(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Train the one frozen 200-epoch full-cache origin."""
    return train_stage_with_loss(
        train_cache_root=train_cache_root,
        validation_cache_root=validation_cache_root,
        output_root=output_root,
        config=frozen_config(),
        loss_function=target_only_tile_loss,
        selection_key=membership_stage_selection_key,
        experiment_id=EXPERIMENT_ID,
        report_metadata={
            "objective_id": OBJECTIVE_ID,
            "source_experiment_id": "conditional-tile-target-only-objective-v1",
            "source_epoch_budget": 20,
            "treatment_epoch_budget": EPOCHS,
            "rank_regression_used": False,
            "listwise_loss_used": False,
            "warm_start_used": False,
        },
    )


def _retag(report: dict[str, Any]) -> dict[str, Any]:
    report["experiment_id"] = EXPERIMENT_ID
    report["source_pipeline_experiment_id"] = "full-legal-hierarchical-factor-retrieval-pilot-v1"
    report["source_treatment_experiment_id"] = "conditional-tile-target-only-objective-v1"
    return report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-cache", type=Path, required=True)
    train_parser.add_argument("--validation-cache", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--train-cache", type=Path, required=True)
    replay_parser.add_argument("--validation-cache", type=Path, required=True)
    replay_parser.add_argument("--weights", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)

    mixed_parser = subparsers.add_parser("mixed-ceiling")
    mixed_parser.add_argument("--train-cache", type=Path, required=True)
    mixed_parser.add_argument("--validation-cache", type=Path, required=True)
    mixed_parser.add_argument("--weights", type=Path, required=True)
    mixed_parser.add_argument("--output", type=Path, required=True)

    integration_parser = subparsers.add_parser("integrated")
    integration_parser.add_argument("--train-cache", type=Path, required=True)
    integration_parser.add_argument("--validation-cache", type=Path, required=True)
    integration_parser.add_argument("--draft-weights", type=Path, required=True)
    integration_parser.add_argument("--tile-weights", type=Path, required=True)
    integration_parser.add_argument("--wildlife-weights", type=Path, required=True)
    integration_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "train":
        report = train(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            output_root=args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "replay":
        report = _retag(
            replay_stage(
                stage="tile",
                weights=args.weights,
                train_cache_root=args.train_cache,
                validation_cache_root=args.validation_cache,
            )
        )
    elif args.command == "mixed-ceiling":
        report = _retag(
            evaluate_mixed_stage_ceiling(
                stage="tile",
                weights=args.weights,
                train_cache_root=args.train_cache,
                validation_cache_root=args.validation_cache,
            )
        )
    else:
        report = _retag(
            evaluate_integrated(
                train_cache_root=args.train_cache,
                validation_cache_root=args.validation_cache,
                weights={
                    "draft": args.draft_weights,
                    "tile": args.tile_weights,
                    "wildlife": args.wildlife_weights,
                },
            )
        )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
