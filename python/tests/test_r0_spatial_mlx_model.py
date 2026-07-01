from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.r0_spatial_mlx_cache import ARM_TOKEN_CAPACITY, TARGET_DIM
from cascadia_mlx.r0_spatial_mlx_model import (
    R0SpatialIsoValueModel,
    R0SpatialMlxModelConfig,
    parameter_count,
    r0_spatial_value_loss,
)


def _inputs(capacity: int, *, batch_size: int = 2) -> tuple[mx.array, ...]:
    tokens = np.zeros((batch_size, 4, capacity, 11), dtype=np.int32)
    mask = np.zeros((batch_size, 4, capacity), dtype=np.bool_)
    mask[:, :, :5] = True
    tokens[:, :, :5, 4] = 1
    tokens[:, :, :5, 5] = 1
    tokens[:, :, :5, 6] = 5
    tokens[:, :, :5, 9] = 5
    market = np.zeros((batch_size, 4, 31), dtype=np.float32)
    market_mask = np.ones((batch_size, 4), dtype=np.bool_)
    global_features = np.zeros((batch_size, 96), dtype=np.float32)
    return (
        mx.array(tokens),
        mx.array(mask),
        mx.array(market),
        mx.array(market_mask),
        mx.array(global_features),
    )


@pytest.mark.parametrize("capacity", ARM_TOKEN_CAPACITY.values())
def test_one_parameterization_accepts_every_frozen_arm_shape(capacity: int) -> None:
    mx.random.seed(19)
    model = R0SpatialIsoValueModel()
    predictions = model.predict_components(*_inputs(capacity))
    mx.eval(predictions)
    values = np.asarray(predictions)
    assert values.shape == (2, TARGET_DIM)
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_zero_padding_is_inert_across_sequence_lengths() -> None:
    mx.random.seed(23)
    model = R0SpatialIsoValueModel()
    exact = model.predict_components(*_inputs(23))
    padded = model.predict_components(*_inputs(464))
    mx.eval(exact, padded)
    np.testing.assert_allclose(np.asarray(exact), np.asarray(padded), rtol=1e-5, atol=1e-5)


def test_parameter_count_is_shape_independent_and_frozen() -> None:
    mx.random.seed(29)
    first = R0SpatialIsoValueModel()
    mx.random.seed(29)
    second = R0SpatialIsoValueModel(R0SpatialMlxModelConfig())
    assert parameter_count(first) == parameter_count(second) == 74_635


def test_component_and_total_loss_is_finite() -> None:
    mx.random.seed(31)
    model = R0SpatialIsoValueModel()
    spatial, mask, market, market_mask, global_features = _inputs(84)
    batch = SimpleNamespace(
        spatial_tokens=spatial,
        spatial_mask=mask,
        market_features=market,
        market_mask=market_mask,
        global_features=global_features,
        targets=mx.ones((2, TARGET_DIM)),
    )
    loss = r0_spatial_value_loss(model, batch)
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))
    assert float(loss.item()) > 0


def test_architecture_drift_is_rejected() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        R0SpatialMlxModelConfig(hidden_dim=64).validate()
