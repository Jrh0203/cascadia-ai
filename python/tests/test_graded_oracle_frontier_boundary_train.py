from __future__ import annotations

import mlx.core as mx
import numpy as np
from cascadia_mlx.graded_oracle_frontier_boundary_train import (
    BOUNDARY_MARGIN,
    BOUNDARY_SEED,
    BOUNDARY_TEMPERATURE,
    FrontierBoundaryTrainingConfig,
    frontier_boundary_adapter,
    smooth_topk_boundary_loss_from_scores,
)


def test_boundary_loss_pushes_targets_up_and_nontargets_down() -> None:
    scores = mx.array([[4.0, 3.0, 5.0, 2.0, 0.0]])
    target = mx.array([[True, True, False, False, False]])
    eligible = mx.array([[True, True, True, True, False]])
    gradient = mx.grad(
        lambda values: smooth_topk_boundary_loss_from_scores(
            values,
            target,
            eligible,
        )
    )(scores)
    mx.eval(gradient)
    values = np.asarray(gradient)[0]
    assert np.all(values[:2] < 0.0)
    assert np.all(values[2:4] > 0.0)
    assert values[4] == 0.0


def test_boundary_loss_rewards_a_clean_cutoff() -> None:
    target = mx.array([[True, True, False, False]])
    eligible = mx.ones((1, 4), dtype=mx.bool_)
    bad = smooth_topk_boundary_loss_from_scores(
        mx.array([[2.0, 1.0, 3.0, 0.0]]),
        target,
        eligible,
    )
    good = smooth_topk_boundary_loss_from_scores(
        mx.array([[4.0, 3.0, 1.0, 0.0]]),
        target,
        eligible,
    )
    mx.eval(bad, good)
    assert float(good.item()) < float(bad.item())


def test_boundary_adapter_and_config_are_frozen(tmp_path) -> None:
    adapter = frontier_boundary_adapter()
    assert adapter.selection_metric == "target_positive_miss_rate"
    assert adapter.init_manifest_name == "checkpoint.json"
    config = FrontierBoundaryTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        run_dir=tmp_path / "run",
        init_model_dir=tmp_path / "checkpoint",
    )
    config.validate()
    assert config.seed == BOUNDARY_SEED
    assert BOUNDARY_TEMPERATURE == 0.25
    assert BOUNDARY_MARGIN == 0.5
