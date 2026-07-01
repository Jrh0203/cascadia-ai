from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_local_geometry_adapter import (
    ADAPTER_ARCHITECTURE,
    ADAPTER_HIDDEN_DIM,
    LOCAL_INPUT_DIM,
    LocalGeometryAdapterBatch,
    LocalGeometryResidualAdapter,
    _zero_initialized_equality,
    local_geometry_adapter_loss,
    run_group,
)
from mlx.utils import tree_flatten


def _adapter_batch() -> LocalGeometryAdapterBatch:
    candidate_mask = mx.array([[True, True, False]])
    return LocalGeometryAdapterBatch(
        local_features=mx.zeros((1, 3, LOCAL_INPUT_DIM)),
        candidate_mask=candidate_mask,
        base_residuals=mx.array([[2.0, -1.0, 0.0]]),
        screen_value=mx.array([[90.0, 91.0, 0.0]]),
        expected_rank=mx.array([[1.0, 2.0, 0.0]]),
        expected_rank_mask=mx.array([[True, True, False]]),
        source_flags=mx.zeros((1, 3), dtype=mx.uint8),
    )


def test_adapter_contract_is_frozen() -> None:
    assert ADAPTER_ARCHITECTURE == (
        "frozen-selected-local-geometry-adapter-v1"
    )
    assert ADAPTER_HIDDEN_DIM == 192


def test_zero_initialized_adapter_matches_frozen_base_exactly() -> None:
    model = LocalGeometryResidualAdapter(hidden_dim=12)
    batch = _adapter_batch()
    assert _zero_initialized_equality(model, batch)
    scores, residuals = model(
        batch.local_features,
        batch.candidate_mask,
        batch.base_residuals,
        batch.screen_value,
    )
    mx.eval(scores, residuals)
    np.testing.assert_array_equal(
        np.asarray(residuals),
        np.asarray(batch.base_residuals),
    )


def test_adapter_loss_and_gradient_are_finite() -> None:
    model = LocalGeometryResidualAdapter(hidden_dim=12)
    batch = _adapter_batch()
    loss_and_grad = nn.value_and_grad(
        model,
        local_geometry_adapter_loss,
    )
    loss, gradients = loss_and_grad(model, batch)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for _name, value in tree_flatten(gradients)
    )


def test_out_of_range_group_is_rejected_before_loading_inputs() -> None:
    with pytest.raises(ValueError, match="outside 0-3"):
        run_group(None, None, None, 4)  # type: ignore[arg-type]
