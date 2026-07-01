"""Single-host MLX training for ADR 0100 expected-rank supervision."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankDataset,
    evaluate_frontier_expected_rank,
    frontier_expected_rank_loss,
    randomly_rotate_expected_rank_batch,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    load_promoted_graded_oracle_model,
    score_graded_oracle_batch,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking

EXPECTED_RANK_SEED = 2026061626
EXPECTED_RANK_EPOCHS = 20
EXPECTED_RANK_LEARNING_RATE = 1e-4
EXPECTED_RANK_WEIGHT_DECAY = 1e-4
EXPECTED_RANK_PATIENCE = 6


@dataclass(frozen=True)
class FrontierExpectedRankTrainingConfig:
    """The frozen ADR 0100 one-host training protocol."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    train_target_cache: str
    validation_target_cache: str
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = EXPECTED_RANK_EPOCHS
    group_batch_size: int = 64
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = EXPECTED_RANK_LEARNING_RATE
    weight_decay: float = EXPECTED_RANK_WEIGHT_DECAY
    seed: int = EXPECTED_RANK_SEED
    checkpoint_steps: int = 250
    validation_patience: int = EXPECTED_RANK_PATIENCE
    resume: bool = False
    model: GradedOracleModelConfig = field(default_factory=GradedOracleModelConfig)

    def validate(self) -> None:
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("expected-rank training prohibits additional datasets")
        if self.init_model_dir is not None:
            raise ValueError("expected-rank training prohibits warm starts")
        if self.epochs != EXPECTED_RANK_EPOCHS:
            raise ValueError("expected-rank epoch budget drifted")
        if self.group_batch_size != 64:
            raise ValueError("expected-rank group batch size drifted")
        if self.maximum_actions_per_batch != GRADED_ORACLE_PACKED_ACTION_LIMIT:
            raise ValueError("expected-rank packed action limit drifted")
        if self.maximum_group_actions != GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
            raise ValueError("expected-rank maximum group width drifted")
        if self.learning_rate != EXPECTED_RANK_LEARNING_RATE:
            raise ValueError("expected-rank learning rate drifted")
        if self.weight_decay != EXPECTED_RANK_WEIGHT_DECAY:
            raise ValueError("expected-rank weight decay drifted")
        if self.seed != EXPECTED_RANK_SEED:
            raise ValueError("expected-rank seed drifted")
        if self.checkpoint_steps != 250:
            raise ValueError("expected-rank checkpoint interval drifted")
        if self.validation_patience != EXPECTED_RANK_PATIENCE:
            raise ValueError("expected-rank validation patience drifted")
        if self.model != GradedOracleModelConfig():
            raise ValueError("expected-rank architecture drifted")
        if not Path(self.train_target_cache).is_dir():
            raise ValueError("expected-rank train cache is missing")
        if not Path(self.validation_target_cache).is_dir():
            raise ValueError("expected-rank validation cache is missing")
        self.model.validate()


def frontier_expected_rank_adapter(
    config: FrontierExpectedRankTrainingConfig,
) -> GroupedRankingAdapter:
    """Bind verified target caches and the single frozen loss to the trainer."""
    cache_by_dataset = {
        config.train_dataset.resolve(): Path(config.train_target_cache),
        config.validation_dataset.resolve(): Path(config.validation_target_cache),
    }

    def dataset_factory(path: Path) -> ExpectedRankDataset:
        resolved = path.resolve()
        try:
            cache = cache_by_dataset[resolved]
        except KeyError as error:
            raise ValueError("expected-rank dataset has no frozen cache") from error
        return ExpectedRankDataset(path, cache)

    return GroupedRankingAdapter(
        kind="graded-oracle-frontier-expected-rank",
        dataset_factory=dataset_factory,
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
        new_model=GradedOracleRanker,
        load_promoted=load_promoted_graded_oracle_model,
        loss=frontier_expected_rank_loss,
        score_batch=score_graded_oracle_batch,
        augment_batch=randomly_rotate_expected_rank_batch,
        evaluate=evaluate_frontier_expected_rank,
        selection_metric="expected_rank_target_positive_miss_rate",
        accuracy_metric="expected_rank_target_set_exact_fraction",
        tertiary_metric="mean_top64_retained_r4800_regret",
        batch_kwargs={
            "maximum_actions_per_batch": GRADED_ORACLE_PACKED_ACTION_LIMIT,
            "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        },
    )


def train_frontier_expected_rank(
    config: FrontierExpectedRankTrainingConfig,
) -> dict[str, Any]:
    """Train or exactly resume the one authorized ADR 0100 model."""
    return train_ranking(config, adapter=frontier_expected_rank_adapter(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--train-target-cache", required=True)
    parser.add_argument("--validation-target-cache", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[EXPECTED_RANK_SEED], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_frontier_expected_rank(
        FrontierExpectedRankTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            train_target_cache=args.train_target_cache,
            validation_target_cache=args.validation_target_cache,
            run_dir=args.run_dir,
            seed=args.seed,
            resume=args.resume,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
