from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    GRAPH_MAX_DEGREE,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TARGET_DIM,
    TOKEN_FEATURES,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    ARCHITECTURES,
    CommonStateEncoder,
    R2SparseMlxModelConfig,
    R2SparseValueModel,
    _type_summary_tokens,
    architecture_parameter_counts,
    parameter_count,
    r2_sparse_value_loss,
)


def test_common_encoder_defaults_to_legacy_92_and_rejects_live_139() -> None:
    encoder = CommonStateEncoder(64)
    batch = _batch(batch_size=1)
    assert encoder.board_token_capacity == BOARD_TOKEN_CAPACITY == 92
    encoder(
        batch.token_features,
        batch.token_mask,
        batch.market_features,
        batch.market_mask,
        batch.player_features,
        batch.player_mask,
        batch.global_features,
    )
    with pytest.raises(ValueError, match="shape drifted"):
        encoder(
            mx.zeros((1, BOARD_SLOTS, 139, TOKEN_FEATURES)),
            mx.zeros((1, BOARD_SLOTS, 139), dtype=mx.bool_),
            batch.market_features,
            batch.market_mask,
            batch.player_features,
            batch.player_mask,
            batch.global_features,
        )


def _batch(*, noisy_padding: bool = False, batch_size: int = 2) -> SimpleNamespace:
    rng = np.random.default_rng(17)
    features = np.zeros(
        (
            batch_size,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            TOKEN_FEATURES,
        ),
        dtype=np.float32,
    )
    mask = np.zeros(
        (batch_size, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.bool_,
    )
    mask[:, :, :8] = True
    features[:, :, :8] = rng.normal(
        size=(batch_size, BOARD_SLOTS, 8, TOKEN_FEATURES)
    )
    if noisy_padding:
        features[:, :, 8:] = rng.normal(
            size=(
                batch_size,
                BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY - 8,
                TOKEN_FEATURES,
            )
        )
    token_types = np.zeros(mask.shape, dtype=np.int32)
    token_types[:, :, :8] = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    neighbors = np.zeros(
        (
            batch_size,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            GRAPH_MAX_DEGREE,
        ),
        dtype=np.int32,
    )
    neighbor_mask = np.zeros_like(neighbors, dtype=np.bool_)
    relations = np.zeros_like(neighbors, dtype=np.int32)
    directions = np.zeros((*neighbors.shape, 6), dtype=np.float32)
    for source in range(8):
        neighbors[:, :, source, 0] = (source + 1) % 8
        neighbor_mask[:, :, source, 0] = True
        relations[:, :, source, 0] = 1
        directions[:, :, source, 0, source % 6] = 1
    return SimpleNamespace(
        token_features=mx.array(features),
        token_types=mx.array(token_types),
        token_mask=mx.array(mask),
        graph_neighbors=mx.array(neighbors),
        graph_neighbor_mask=mx.array(neighbor_mask),
        graph_relations=mx.array(relations),
        graph_direction_features=mx.array(directions),
        market_features=mx.zeros((batch_size, 4, MARKET_FEATURES)),
        market_mask=mx.ones((batch_size, 4), dtype=mx.bool_),
        player_features=mx.zeros((batch_size, BOARD_SLOTS, PLAYER_FEATURES)),
        player_mask=mx.ones((batch_size, BOARD_SLOTS), dtype=mx.bool_),
        global_features=mx.zeros((batch_size, GLOBAL_FEATURES)),
        targets=mx.ones((batch_size, TARGET_DIM)),
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_is_finite_and_padding_invariant(architecture: str) -> None:
    mx.random.seed(23)
    model = R2SparseValueModel(R2SparseMlxModelConfig(architecture=architecture))
    clean = model.predict_components(_batch())
    noisy = model.predict_components(_batch(noisy_padding=True))
    mx.eval(clean, noisy)
    clean_values = np.asarray(clean)
    noisy_values = np.asarray(noisy)
    assert clean_values.shape == (2, TARGET_DIM)
    assert np.isfinite(clean_values).all()
    assert (clean_values >= 0).all()
    np.testing.assert_allclose(clean_values, noisy_values, rtol=1e-5, atol=1e-5)


def test_parameter_matching_counts_every_active_trainable_surface() -> None:
    counts = architecture_parameter_counts()
    assert counts == {
        "padded-set-transformer": 141_131,
        "directional-graph-attention": 143_915,
        "perceiver-fixed-latents": 142_283,
    }
    assert (max(counts.values()) - min(counts.values())) / min(counts.values()) < 0.03
    for architecture, expected in counts.items():
        model = R2SparseValueModel(R2SparseMlxModelConfig(architecture=architecture))
        assert parameter_count(model) == expected


def test_state_encoder_is_invoked_exactly_once_per_prediction() -> None:
    class CountingEncoder(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def __call__(
            self,
            *args: mx.array,
        ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
            self.calls += 1
            return self.inner(*args)

    model = R2SparseValueModel()
    counting = CountingEncoder(model.common_encoder)
    model.common_encoder = counting
    output = model(_batch())
    mx.eval(output)
    assert counting.calls == 1


def test_component_and_total_loss_is_finite() -> None:
    model = R2SparseValueModel()
    loss = r2_sparse_value_loss(model, _batch())
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))
    assert float(loss.item()) > 0


def test_type_summary_pooling_is_invariant_to_duplicate_frontier_tokens() -> None:
    one = mx.ones((1, BOARD_TOKEN_CAPACITY, 3))
    one_type = mx.zeros((1, BOARD_TOKEN_CAPACITY), dtype=mx.int32)
    one_mask = mx.zeros((1, BOARD_TOKEN_CAPACITY), dtype=mx.bool_)
    one_type = one_type.at[:, 0].add(2)
    one_mask = one_mask.at[:, 0].add(True)

    many_type = mx.zeros((1, BOARD_TOKEN_CAPACITY), dtype=mx.int32)
    many_mask = mx.zeros((1, BOARD_TOKEN_CAPACITY), dtype=mx.bool_)
    many_type = many_type.at[:, :30].add(2)
    many_mask = many_mask.at[:, :30].add(True)
    one_summary, _ = _type_summary_tokens(one, one_type, one_mask)
    many_summary, _ = _type_summary_tokens(one, many_type, many_mask)
    mx.eval(one_summary, many_summary)
    np.testing.assert_allclose(
        np.asarray(one_summary)[:, 1],
        np.asarray(many_summary)[:, 1],
    )


def test_board_ownership_and_architecture_drift_fail_closed() -> None:
    with pytest.raises(ValueError, match="board ownership"):
        R2SparseMlxModelConfig(board_ownership_encoding="implicit").validate()
    with pytest.raises(ValueError, match="architecture"):
        R2SparseMlxModelConfig(architecture="unknown").validate()
