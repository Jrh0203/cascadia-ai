"""Resumable ADR 0088 MLX training for local-geometry complete-action ranking."""

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
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LocalGeometryModelConfig,
    LocalGeometryRanker,
    load_promoted_local_geometry_model,
)
from cascadia_mlx.graded_oracle_metrics import evaluate_graded_oracle
from cascadia_mlx.graded_oracle_model import (
    graded_oracle_loss,
    score_graded_oracle_batch,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking

LOCAL_GEOMETRY_TRAINING_SEEDS = frozenset(
    {
        2026061601,
        2026061602,
        2026061603,
    }
)
LOCAL_GEOMETRY_GROUP_BATCH_SIZE = 64


@dataclass(frozen=True)
class LocalGeometryTrainingConfig:
    """The locked ADR 0088 training protocol."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = 30
    group_batch_size: int = LOCAL_GEOMETRY_GROUP_BATCH_SIZE
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 2026061601
    checkpoint_steps: int = 250
    validation_patience: int = 6
    resume: bool = False
    model: LocalGeometryModelConfig = field(default_factory=LocalGeometryModelConfig)

    def validate(self) -> None:
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("ADR 0088 prohibits additional datasets")
        if self.init_model_dir is not None:
            raise ValueError("ADR 0088 prohibits warm starts")
        if self.epochs != 30:
            raise ValueError("ADR 0088 freezes the maximum epoch budget at 30")
        if self.group_batch_size != LOCAL_GEOMETRY_GROUP_BATCH_SIZE:
            raise ValueError("ADR 0088 group packing configuration drifted")
        if self.maximum_actions_per_batch != GRADED_ORACLE_PACKED_ACTION_LIMIT:
            raise ValueError("ADR 0088 freezes the packed action-row target at 8192")
        if self.maximum_group_actions != GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
            raise ValueError("ADR 0088 freezes the group ceiling at 16384")
        if self.learning_rate != 1e-4 or self.weight_decay != 1e-4:
            raise ValueError("ADR 0088 optimizer hyperparameters drifted")
        if self.seed not in LOCAL_GEOMETRY_TRAINING_SEEDS:
            raise ValueError("ADR 0088 authorizes exactly three paired seeds")
        if self.checkpoint_steps != 250:
            raise ValueError("ADR 0088 freezes checkpointing at 250 optimizer steps")
        if self.validation_patience != 6:
            raise ValueError("ADR 0088 freezes validation patience at six epochs")
        if self.model != LocalGeometryModelConfig():
            raise ValueError("ADR 0088 model architecture drifted")
        self.model.validate()


def local_geometry_adapter() -> GroupedRankingAdapter:
    """Bind the ADR 0088 representation to the frozen ADR 0081 protocol."""
    return GroupedRankingAdapter(
        kind="graded-oracle-local-geometry-ranking",
        dataset_factory=GradedOracleDataset,
        model_factory=lambda values: LocalGeometryRanker(
            LocalGeometryModelConfig.from_dict(values)
        ),
        new_model=LocalGeometryRanker,
        load_promoted=load_promoted_local_geometry_model,
        loss=graded_oracle_loss,
        score_batch=score_graded_oracle_batch,
        augment_batch=randomly_rotate_graded_oracle_batch,
        evaluate=evaluate_graded_oracle,
        selection_metric="mean_top64_retained_r4800_regret",
        accuracy_metric="top64_r4800_winner_recall",
        tertiary_metric="r4800_residual_mae",
        batch_kwargs={
            "maximum_actions_per_batch": GRADED_ORACLE_PACKED_ACTION_LIMIT,
            "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        },
    )


def train_local_geometry(config: LocalGeometryTrainingConfig) -> dict[str, object]:
    """Train or resume exactly one preregistered ADR 0088 replica."""
    return train_ranking(config, adapter=local_geometry_adapter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=sorted(LOCAL_GEOMETRY_TRAINING_SEEDS),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_local_geometry(
        LocalGeometryTrainingConfig(
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

