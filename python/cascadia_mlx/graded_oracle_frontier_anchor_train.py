"""Resumable MLX training for the frontier-anchored set proposer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
    randomly_rotate_graded_oracle_batch,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    evaluate_frontier_anchored,
    frontier_anchored_loss,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    load_promoted_graded_oracle_model,
    score_graded_oracle_batch,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking

FRONTIER_ANCHORED_TRAINING_SEEDS = frozenset(
    {
        2026061601,
        2026061602,
        2026061603,
        2026061604,
    }
)
FRONTIER_ANCHORED_GROUP_BATCH_SIZE = 64


@dataclass(frozen=True)
class FrontierAnchoredTrainingConfig:
    """The locked frontier-anchored set-proposer protocol."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = 30
    group_batch_size: int = FRONTIER_ANCHORED_GROUP_BATCH_SIZE
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 2026061601
    checkpoint_steps: int = 250
    validation_patience: int = 6
    resume: bool = False
    model: GradedOracleModelConfig = field(default_factory=GradedOracleModelConfig)

    def validate(self) -> None:
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("frontier-anchored training prohibits additional datasets")
        if self.init_model_dir is not None:
            raise ValueError("frontier-anchored training prohibits warm starts")
        if self.epochs != 30:
            raise ValueError("frontier-anchored training freezes 30 maximum epochs")
        if self.group_batch_size != FRONTIER_ANCHORED_GROUP_BATCH_SIZE:
            raise ValueError("frontier-anchored group packing configuration drifted")
        if self.maximum_actions_per_batch != GRADED_ORACLE_PACKED_ACTION_LIMIT:
            raise ValueError("frontier-anchored packed action target drifted")
        if self.maximum_group_actions != GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
            raise ValueError("frontier-anchored group ceiling drifted")
        if self.learning_rate != 1e-4 or self.weight_decay != 1e-4:
            raise ValueError("frontier-anchored optimizer hyperparameters drifted")
        if self.seed not in FRONTIER_ANCHORED_TRAINING_SEEDS:
            raise ValueError("frontier-anchored training authorizes four paired seeds")
        if self.checkpoint_steps != 250:
            raise ValueError("frontier-anchored checkpoint interval drifted")
        if self.validation_patience != 6:
            raise ValueError("frontier-anchored validation patience drifted")
        if self.model != GradedOracleModelConfig():
            raise ValueError("frontier-anchored model architecture drifted")
        self.model.validate()


def frontier_anchored_adapter() -> GroupedRankingAdapter:
    """Bind the fixed ADR 0081 architecture to set-valued supervision."""
    return GroupedRankingAdapter(
        kind="graded-oracle-frontier-anchored-ranking",
        dataset_factory=GradedOracleDataset,
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
        new_model=GradedOracleRanker,
        load_promoted=load_promoted_graded_oracle_model,
        loss=frontier_anchored_loss,
        score_batch=score_graded_oracle_batch,
        augment_batch=randomly_rotate_graded_oracle_batch,
        evaluate=evaluate_frontier_anchored,
        selection_metric="top64_r4800_winner_miss_rate",
        accuracy_metric="top64_confidence_set_coverage_95",
        tertiary_metric="mean_top64_retained_r4800_regret",
        batch_kwargs={
            "maximum_actions_per_batch": GRADED_ORACLE_PACKED_ACTION_LIMIT,
            "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        },
    )


def train_frontier_anchored(
    config: FrontierAnchoredTrainingConfig,
) -> dict[str, object]:
    """Train or resume one frozen frontier-anchored replica."""
    return train_ranking(config, adapter=frontier_anchored_adapter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=sorted(FRONTIER_ANCHORED_TRAINING_SEEDS),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_frontier_anchored(
        FrontierAnchoredTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            run_dir=args.run_dir,
            seed=args.seed,
            resume=args.resume,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
