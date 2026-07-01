"""Single-host MLX training for ADR 0101 scale-16 supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    evaluate_frontier_expected_rank,
    randomly_rotate_expected_rank_batch,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    Scale16ExpectedRankDataset,
    frontier_expected_rank_scale16_loss,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_train import (
    EXPECTED_RANK_SEED,
    FrontierExpectedRankTrainingConfig,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    load_promoted_graded_oracle_model,
    score_graded_oracle_batch,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking

RUN_KIND = "graded-oracle-frontier-expected-rank-scale16"


def frontier_expected_rank_scale16_adapter(
    config: FrontierExpectedRankTrainingConfig,
) -> GroupedRankingAdapter:
    """Bind the scale-16 caches and objective to the shared trainer."""
    cache_by_dataset = {
        config.train_dataset.resolve(): Path(config.train_target_cache),
        config.validation_dataset.resolve(): Path(config.validation_target_cache),
    }

    def dataset_factory(path: Path) -> Scale16ExpectedRankDataset:
        resolved = path.resolve()
        try:
            cache = cache_by_dataset[resolved]
        except KeyError as error:
            raise ValueError(
                "scale-16 expected-rank dataset has no frozen cache"
            ) from error
        return Scale16ExpectedRankDataset(path, cache)

    return GroupedRankingAdapter(
        kind=RUN_KIND,
        dataset_factory=dataset_factory,
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
        new_model=GradedOracleRanker,
        load_promoted=load_promoted_graded_oracle_model,
        loss=frontier_expected_rank_scale16_loss,
        score_batch=score_graded_oracle_batch,
        augment_batch=randomly_rotate_expected_rank_batch,
        evaluate=evaluate_frontier_expected_rank,
        selection_metric="expected_rank_target_positive_miss_rate",
        accuracy_metric="expected_rank_target_set_exact_fraction",
        tertiary_metric="mean_top64_retained_r4800_regret",
        batch_kwargs={
            "maximum_actions_per_batch": config.maximum_actions_per_batch,
            "maximum_group_actions": config.maximum_group_actions,
        },
    )


def train_frontier_expected_rank_scale16(
    config: FrontierExpectedRankTrainingConfig,
) -> dict[str, object]:
    """Train or exactly resume the sole ADR 0101 model."""
    return train_ranking(
        config,
        adapter=frontier_expected_rank_scale16_adapter(config),
    )


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
    report = train_frontier_expected_rank_scale16(
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
