from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from cascadia_mlx.graded_oracle_frontier_rank_boundary_train import (
    RANK_BOUNDARY_MARGIN,
    RANK_BOUNDARY_SEED,
    RANK_BOUNDARY_TEMPERATURE,
    FrontierRankBoundaryTrainingConfig,
    frontier_rank_boundary_adapter,
    rank_matched_boundary_loss_from_scores,
)


def test_rank_boundary_distributes_gradient_across_all_targets() -> None:
    scores = mx.array([[4.0, 3.0, 2.0, 5.0, 1.0, 0.0]])
    target = mx.array([[True, True, True, False, False, False]])
    eligible = mx.ones((1, 6), dtype=mx.bool_)
    gradient = mx.grad(
        lambda values: rank_matched_boundary_loss_from_scores(
            values,
            target,
            eligible,
            maximum_pairs=3,
        )
    )(scores)
    mx.eval(gradient)
    values = np.asarray(gradient)[0]
    assert np.all(values[:3] < 0.0)
    assert np.all(values[3:] > 0.0)


def test_rank_boundary_updates_only_matching_hard_negatives() -> None:
    scores = mx.array([[3.0, 2.0, 8.0, 7.0, 1.0, 0.0]])
    target = mx.array([[True, True, False, False, False, False]])
    eligible = mx.ones((1, 6), dtype=mx.bool_)
    gradient = mx.grad(
        lambda values: rank_matched_boundary_loss_from_scores(
            values,
            target,
            eligible,
            maximum_pairs=2,
        )
    )(scores)
    mx.eval(gradient)
    values = np.asarray(gradient)[0]
    assert np.all(values[:2] < 0.0)
    assert np.all(values[2:4] > 0.0)
    assert np.all(values[4:] == 0.0)


def test_rank_boundary_rewards_a_clean_ordered_cutoff() -> None:
    target = mx.array([[True, True, False, False]])
    eligible = mx.ones((1, 4), dtype=mx.bool_)
    bad = rank_matched_boundary_loss_from_scores(
        mx.array([[2.0, 1.0, 3.0, 0.0]]),
        target,
        eligible,
        maximum_pairs=2,
    )
    good = rank_matched_boundary_loss_from_scores(
        mx.array([[4.0, 3.0, 1.0, 0.0]]),
        target,
        eligible,
        maximum_pairs=2,
    )
    mx.eval(bad, good)
    assert float(good.item()) < float(bad.item())


def test_rank_boundary_adapter_and_config_are_frozen(tmp_path: Path) -> None:
    adapter = frontier_rank_boundary_adapter()
    assert adapter.selection_metric == "target_positive_miss_rate"
    assert adapter.init_manifest_name == "checkpoint.json"
    config = FrontierRankBoundaryTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        run_dir=tmp_path / "run",
        init_model_dir=tmp_path / "checkpoint",
    )
    config.validate()
    assert config.seed == RANK_BOUNDARY_SEED
    assert RANK_BOUNDARY_TEMPERATURE == 1.0
    assert RANK_BOUNDARY_MARGIN == 0.5
