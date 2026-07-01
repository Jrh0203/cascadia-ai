from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3_TOKEN_FEATURES
from cascadia_mlx.r4_bounded_parent_mlx_cache import (
    ARMS,
    UNIVERSAL_PARENT_VALUE_WIDTH,
)
from cascadia_mlx.r4_bounded_parent_mlx_model import (
    R4BoundedParentModelConfig,
    R4BoundedParentRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    r4_bounded_parent_loss,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
)


def _batch(*, parent_width: int = 3) -> SimpleNamespace:
    groups = 1
    candidates = 2
    classes = np.zeros((groups, BOARD_SLOTS, parent_width), dtype=np.int32)
    values = np.zeros(
        (
            groups,
            BOARD_SLOTS,
            parent_width,
            UNIVERSAL_PARENT_VALUE_WIDTH,
        ),
        dtype=np.int16,
    )
    token_mask = np.zeros_like(classes, dtype=np.bool_)
    for board in range(BOARD_SLOTS):
        classes[0, board, 0] = board + 1
        values[0, board, 0, 0] = board + 1
        token_mask[0, board, 0] = True
    parent = SimpleNamespace(
        token_values=mx.array(values),
        token_classes=mx.array(classes),
        token_mask=mx.array(token_mask),
        market_features=mx.zeros((groups, 4, MARKET_FEATURES)),
        market_mask=mx.ones((groups, 4), dtype=mx.bool_),
        player_features=mx.zeros((groups, BOARD_SLOTS, PLAYER_FEATURES)),
        player_mask=mx.ones((groups, BOARD_SLOTS), dtype=mx.bool_),
        global_features=mx.zeros((groups, GLOBAL_FEATURES)),
    )
    candidate_features = np.zeros(
        (groups, candidates, 3, R3_TOKEN_FEATURES),
        dtype=np.float32,
    )
    candidate_features[:, :, :, 0] = 1
    candidate_token_mask = mx.ones((groups, candidates, 3), dtype=mx.bool_)
    candidate_mask = mx.ones((groups, candidates), dtype=mx.bool_)
    base = SimpleNamespace(
        candidate_mask=candidate_mask,
        action_features=mx.zeros((groups, candidates, 140)),
        prior_features=mx.zeros((groups, candidates, 8)),
        staged_market_entities=mx.zeros((groups, candidates, 4, 31)),
        staged_market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        screen_value=mx.array([[90.0, 91.0]]),
        r600_mean=mx.array([[90.5, 91.5]]),
        r600_stddev=mx.ones((groups, candidates)),
        r600_samples=mx.ones((groups, candidates)) * 600,
        r600_mask=candidate_mask,
        r1200_mean=mx.array([[91.0, 92.0]]),
        r1200_stddev=mx.ones((groups, candidates)),
        r1200_samples=mx.ones((groups, candidates)) * 1200,
        r1200_mask=candidate_mask,
        r4800_mean=mx.array([[91.5, 93.0]]),
        r4800_stddev=mx.ones((groups, candidates)),
        r4800_samples=mx.ones((groups, candidates)) * 4800,
        r4800_mask=candidate_mask,
        selected_index=mx.array([1]),
    )
    return SimpleNamespace(
        base=base,
        parent=parent,
        candidate_token_features=mx.array(candidate_features),
        candidate_token_mask=candidate_token_mask,
        supply_vector=mx.zeros((groups, EXACT_SUPPLY_DIM)),
        staged_supply_vector=mx.zeros((groups, candidates, EXACT_SUPPLY_DIM)),
        selected_archetype=mx.zeros((groups, candidates), dtype=mx.int32),
        frontier_features=mx.zeros((groups, candidates, FRONTIER_FEATURE_DIM)),
    )


def test_all_parent_arms_have_identical_layout_and_initial_tensors() -> None:
    counts = {}
    layouts = {}
    tensors = {}
    for arm in ARMS:
        mx.random.seed(2026061710)
        model = R4BoundedParentRanker(R4BoundedParentModelConfig(arm=arm))
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
        tensors[arm] = parameter_tensor_blake3(model)
    assert len(set(counts.values())) == 1
    assert len(set(layouts.values())) == 1
    assert len(set(tensors.values())) == 1


def test_parent_padding_is_prediction_invariant() -> None:
    mx.random.seed(7)
    model = R4BoundedParentRanker()
    compact = model(_batch(parent_width=1))
    padded = model(_batch(parent_width=8))
    mx.eval(compact.scores, padded.scores)
    np.testing.assert_allclose(np.asarray(compact.scores), np.asarray(padded.scores), atol=1e-6)


def test_parent_ranker_loss_is_finite() -> None:
    mx.random.seed(11)
    model = R4BoundedParentRanker()
    batch = _batch()
    loss = r4_bounded_parent_loss(model, batch)
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))
