from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_local_geometry_adapter import (
    LOCAL_INPUT_DIM,
    LocalGeometryAdapterBatch,
    LocalGeometryResidualAdapter,
)
from cascadia_mlx.graded_oracle_frontier_local_geometry_balanced import (
    BalancedTargetBatch,
    balanced_target_loss,
    run_group,
)
from mlx.utils import tree_flatten


def _batch() -> BalancedTargetBatch:
    candidate = mx.array([[True, True, True, False]])
    adapter = LocalGeometryAdapterBatch(
        local_features=mx.zeros((1, 4, LOCAL_INPUT_DIM)),
        candidate_mask=candidate,
        base_residuals=mx.zeros((1, 4)),
        screen_value=mx.zeros((1, 4)),
        expected_rank=mx.zeros((1, 4)),
        expected_rank_mask=candidate,
        source_flags=mx.zeros((1, 4), dtype=mx.uint8),
    )
    return BalancedTargetBatch(
        adapter=adapter,
        target_mask=mx.array([[True, False, False, False]]),
        eligible_mask=candidate,
    )


def test_balanced_loss_is_finite_and_differentiable() -> None:
    model = LocalGeometryResidualAdapter(hidden_dim=12)
    loss_and_grad = nn.value_and_grad(model, balanced_target_loss)
    loss, gradients = loss_and_grad(model, _batch())
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for _name, value in tree_flatten(gradients)
    )


def test_out_of_range_group_is_rejected_first() -> None:
    with pytest.raises(ValueError, match="outside 0-3"):
        run_group(None, None, None, None, 4)  # type: ignore[arg-type]
