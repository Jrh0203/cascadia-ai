"""Single-host target-only curriculum for the frontier-anchored proposer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
    randomly_rotate_graded_oracle_batch,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    evaluate_frontier_anchored,
    frontier_anchored_loss_components,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import load_frontier_warm_start
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    score_graded_oracle_batch,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking

EXPERIMENT_ID = "complete-action-frontier-target-curriculum-v1"
TARGET_CURRICULUM_SEED = 2026061605
TARGET_CURRICULUM_EPOCHS = 20
TARGET_CURRICULUM_LEARNING_RATE = 3e-5
TARGET_CURRICULUM_WEIGHT_DECAY = 1e-4
TARGET_CURRICULUM_PATIENCE = 6


@dataclass(frozen=True)
class FrontierTargetCurriculumConfig:
    """Frozen ADR 0091 one-host optimization pilot."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    init_model_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    epochs: int = TARGET_CURRICULUM_EPOCHS
    group_batch_size: int = 64
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = TARGET_CURRICULUM_LEARNING_RATE
    weight_decay: float = TARGET_CURRICULUM_WEIGHT_DECAY
    seed: int = TARGET_CURRICULUM_SEED
    checkpoint_steps: int = 250
    validation_patience: int = TARGET_CURRICULUM_PATIENCE
    resume: bool = False
    model: GradedOracleModelConfig = field(default_factory=GradedOracleModelConfig)

    def validate(self) -> None:
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("target curriculum prohibits additional datasets")
        if self.epochs != TARGET_CURRICULUM_EPOCHS:
            raise ValueError("target curriculum epoch budget drifted")
        if self.group_batch_size != 64:
            raise ValueError("target curriculum group batch size drifted")
        if self.maximum_actions_per_batch != GRADED_ORACLE_PACKED_ACTION_LIMIT:
            raise ValueError("target curriculum packed action limit drifted")
        if self.maximum_group_actions != GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
            raise ValueError("target curriculum maximum group width drifted")
        if self.learning_rate != TARGET_CURRICULUM_LEARNING_RATE:
            raise ValueError("target curriculum learning rate drifted")
        if self.weight_decay != TARGET_CURRICULUM_WEIGHT_DECAY:
            raise ValueError("target curriculum weight decay drifted")
        if self.seed != TARGET_CURRICULUM_SEED:
            raise ValueError("target curriculum seed drifted")
        if self.checkpoint_steps != 250:
            raise ValueError("target curriculum checkpoint interval drifted")
        if self.validation_patience != TARGET_CURRICULUM_PATIENCE:
            raise ValueError("target curriculum validation patience drifted")
        if self.model != GradedOracleModelConfig():
            raise ValueError("target curriculum architecture drifted")
        self.model.validate()


def frontier_target_only_loss(
    model: GradedOracleRanker,
    batch: object,
) -> mx.array:
    """Optimize only the exact deployable nonfrontier target set."""
    return frontier_anchored_loss_components(model, batch)[
        "target_set_cross_entropy"
    ]


def evaluate_frontier_target_curriculum(
    model: GradedOracleRanker,
    dataset: GradedOracleDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    """Add minimizable target miss rate to the full anchored evaluation."""
    report = evaluate_frontier_anchored(model, dataset, group_batch_size)
    report["target_positive_miss_rate"] = 1.0 - float(
        report["target_positive_recall"]
    )
    return report


def frontier_target_curriculum_adapter() -> GroupedRankingAdapter:
    """Bind target-only optimization and target-recall checkpoint selection."""
    return GroupedRankingAdapter(
        kind="graded-oracle-frontier-target-curriculum",
        dataset_factory=GradedOracleDataset,
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
        new_model=GradedOracleRanker,
        load_promoted=load_frontier_warm_start,
        loss=frontier_target_only_loss,
        score_batch=score_graded_oracle_batch,
        augment_batch=randomly_rotate_graded_oracle_batch,
        evaluate=evaluate_frontier_target_curriculum,
        selection_metric="target_positive_miss_rate",
        accuracy_metric="target_set_exact_fraction",
        tertiary_metric="mean_top64_retained_r4800_regret",
        batch_kwargs={
            "maximum_actions_per_batch": GRADED_ORACLE_PACKED_ACTION_LIMIT,
            "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        },
        init_manifest_name="checkpoint.json",
    )


def train_frontier_target_curriculum(
    config: FrontierTargetCurriculumConfig,
) -> dict[str, Any]:
    """Train or resume the one authorized ADR 0091 pilot."""
    return train_ranking(config, adapter=frontier_target_curriculum_adapter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[TARGET_CURRICULUM_SEED], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_frontier_target_curriculum(
        FrontierTargetCurriculumConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            run_dir=args.run_dir,
            init_model_dir=args.init_model_dir,
            seed=args.seed,
            resume=args.resume,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
