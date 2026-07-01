"""Single-host rank-matched full-boundary training for the anchored proposer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
    randomly_rotate_graded_oracle_batch,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
    evaluate_frontier_anchored,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import (
    load_frontier_warm_start,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    predict_graded_oracle_batch,
    score_graded_oracle_batch,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking

EXPERIMENT_ID = "complete-action-frontier-rank-boundary-v1"
RANK_BOUNDARY_SEED = 2026061607
RANK_BOUNDARY_EPOCHS = 20
RANK_BOUNDARY_LEARNING_RATE = 3e-5
RANK_BOUNDARY_WEIGHT_DECAY = 1e-4
RANK_BOUNDARY_PATIENCE = 6
RANK_BOUNDARY_TEMPERATURE = 1.0
RANK_BOUNDARY_MARGIN = 0.5
RANK_BOUNDARY_MAXIMUM_PAIRS = 64


@dataclass(frozen=True)
class FrontierRankBoundaryTrainingConfig:
    """Frozen ADR 0093 one-host rank-matched boundary protocol."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    init_model_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    epochs: int = RANK_BOUNDARY_EPOCHS
    group_batch_size: int = 64
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = RANK_BOUNDARY_LEARNING_RATE
    weight_decay: float = RANK_BOUNDARY_WEIGHT_DECAY
    seed: int = RANK_BOUNDARY_SEED
    checkpoint_steps: int = 250
    validation_patience: int = RANK_BOUNDARY_PATIENCE
    resume: bool = False
    model: GradedOracleModelConfig = field(default_factory=GradedOracleModelConfig)

    def validate(self) -> None:
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("rank-boundary training prohibits additional datasets")
        if self.epochs != RANK_BOUNDARY_EPOCHS:
            raise ValueError("rank-boundary training epoch budget drifted")
        if self.group_batch_size != 64:
            raise ValueError("rank-boundary training group batch size drifted")
        if self.maximum_actions_per_batch != GRADED_ORACLE_PACKED_ACTION_LIMIT:
            raise ValueError("rank-boundary training packed action limit drifted")
        if self.maximum_group_actions != GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
            raise ValueError("rank-boundary training maximum group width drifted")
        if self.learning_rate != RANK_BOUNDARY_LEARNING_RATE:
            raise ValueError("rank-boundary training learning rate drifted")
        if self.weight_decay != RANK_BOUNDARY_WEIGHT_DECAY:
            raise ValueError("rank-boundary training weight decay drifted")
        if self.seed != RANK_BOUNDARY_SEED:
            raise ValueError("rank-boundary training seed drifted")
        if self.checkpoint_steps != 250:
            raise ValueError("rank-boundary training checkpoint interval drifted")
        if self.validation_patience != RANK_BOUNDARY_PATIENCE:
            raise ValueError("rank-boundary training validation patience drifted")
        if self.model != GradedOracleModelConfig():
            raise ValueError("rank-boundary training architecture drifted")
        self.model.validate()


def rank_matched_boundary_loss_from_scores(
    scores: mx.array,
    target_mask: mx.array,
    eligible_mask: mx.array,
    *,
    temperature: float = RANK_BOUNDARY_TEMPERATURE,
    margin: float = RANK_BOUNDARY_MARGIN,
    maximum_pairs: int = RANK_BOUNDARY_MAXIMUM_PAIRS,
) -> mx.array:
    """Pair every weakest target rank with the matching hardest nontarget rank."""
    if temperature <= 0.0:
        raise ValueError("rank-boundary temperature must be positive")
    if maximum_pairs <= 0:
        raise ValueError("rank-boundary maximum_pairs must be positive")
    pair_count = min(maximum_pairs, scores.shape[-1])
    non_target_mask = eligible_mask & ~target_mask
    weakest_targets = -mx.topk(
        mx.where(target_mask, -scores, -1e9),
        k=pair_count,
        axis=-1,
    )[..., ::-1]
    hardest_non_targets = mx.topk(
        mx.where(non_target_mask, scores, -1e9),
        k=pair_count,
        axis=-1,
    )[..., ::-1]
    target_counts = mx.sum(target_mask, axis=-1, keepdims=True)
    pair_mask = (
        mx.arange(pair_count, dtype=target_counts.dtype)[None, :]
        < target_counts
    )
    violations = hardest_non_targets - weakest_targets + margin
    losses = temperature * nn.softplus(violations / temperature)
    return mx.sum(mx.where(pair_mask, losses, 0.0)) / mx.maximum(
        mx.sum(pair_mask),
        1,
    )


def frontier_rank_boundary_loss(
    model: GradedOracleRanker,
    batch: object,
) -> mx.array:
    """Optimize every occupied rank along the deployed target boundary."""
    scores = predict_graded_oracle_batch(model, batch).scores
    target = mx.array(
        build_frontier_anchored_target_mask(
            r1200_mean=np.asarray(batch.r1200_mean),
            r1200_mask=np.asarray(batch.r1200_mask),
            source_flags=np.asarray(batch.source_flags),
            candidate_mask=np.asarray(batch.candidate_mask),
            action_hashes=np.asarray(batch.action_hash),
        )
    )
    frontier = (
        batch.source_flags.astype(mx.int32) & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = batch.candidate_mask & ~frontier
    return rank_matched_boundary_loss_from_scores(scores, target, eligible)


def evaluate_frontier_rank_boundary(
    model: GradedOracleRanker,
    dataset: GradedOracleDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    """Add the minimizable target miss metric to anchored evaluation."""
    report = evaluate_frontier_anchored(model, dataset, group_batch_size)
    report["target_positive_miss_rate"] = 1.0 - float(
        report["target_positive_recall"]
    )
    return report


def frontier_rank_boundary_adapter() -> GroupedRankingAdapter:
    """Bind rank-matched boundary optimization to the unchanged ranker."""
    return GroupedRankingAdapter(
        kind="graded-oracle-frontier-rank-boundary",
        dataset_factory=GradedOracleDataset,
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
        new_model=GradedOracleRanker,
        load_promoted=load_frontier_warm_start,
        loss=frontier_rank_boundary_loss,
        score_batch=score_graded_oracle_batch,
        augment_batch=randomly_rotate_graded_oracle_batch,
        evaluate=evaluate_frontier_rank_boundary,
        selection_metric="target_positive_miss_rate",
        accuracy_metric="target_set_exact_fraction",
        tertiary_metric="mean_top64_retained_r4800_regret",
        batch_kwargs={
            "maximum_actions_per_batch": GRADED_ORACLE_PACKED_ACTION_LIMIT,
            "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        },
        init_manifest_name="checkpoint.json",
    )


def train_frontier_rank_boundary(
    config: FrontierRankBoundaryTrainingConfig,
) -> dict[str, Any]:
    """Train or resume the one authorized ADR 0093 pilot."""
    return train_ranking(config, adapter=frontier_rank_boundary_adapter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[RANK_BOUNDARY_SEED], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_frontier_rank_boundary(
        FrontierRankBoundaryTrainingConfig(
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
