from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.o1_ranking_intent_cache import ARMS, INTENT_FEATURE_DIM
from cascadia_mlx.o1_ranking_model import (
    O1IntentConditionedRanker,
    O1RankingModelConfig,
    o1_ranking_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3_TOKEN_FEATURES
from cascadia_mlx.r3_action_edit_mlx_model import R3ActionEditRanker
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
)
from mlx.utils import tree_flatten


def _batch(*, intent_value: float = 0.0) -> SimpleNamespace:
    groups = 1
    candidates = 2
    token_features = np.zeros(
        (
            groups,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            TOKEN_FEATURES,
        ),
        dtype=np.float32,
    )
    token_types = np.zeros(
        (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.int32,
    )
    token_mask = np.zeros_like(token_types, dtype=np.bool_)
    for board in range(BOARD_SLOTS):
        token_features[0, board, 0, board] = 1
        token_types[0, board, 0] = board % 4 + 1
        token_mask[0, board, 0] = True
    parent = SimpleNamespace(
        token_features=mx.array(token_features),
        token_types=mx.array(token_types),
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
    candidate_features[0, 1, 1, 16] = 0.5
    mask = mx.ones((groups, candidates), dtype=mx.bool_)
    base = SimpleNamespace(
        candidate_mask=mask,
        action_features=mx.zeros((groups, candidates, 140)),
        prior_features=mx.zeros((groups, candidates, 8)),
        staged_market_entities=mx.zeros((groups, candidates, 4, 31)),
        staged_market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        screen_value=mx.array([[90.0, 91.0]]),
        r600_mean=mx.array([[90.5, 91.5]]),
        r600_stddev=mx.ones((groups, candidates)),
        r600_samples=mx.ones((groups, candidates)) * 600,
        r600_mask=mask,
        r1200_mean=mx.array([[91.0, 92.0]]),
        r1200_stddev=mx.ones((groups, candidates)),
        r1200_samples=mx.ones((groups, candidates)) * 1200,
        r1200_mask=mask,
        r4800_mean=mx.array([[91.5, 93.0]]),
        r4800_stddev=mx.ones((groups, candidates)),
        r4800_samples=mx.ones((groups, candidates)) * 4800,
        r4800_mask=mask,
        selected_index=mx.array([1]),
    )
    return SimpleNamespace(
        base=base,
        parent=parent,
        candidate_token_features=mx.array(candidate_features),
        candidate_token_mask=mx.ones(
            (groups, candidates, 3),
            dtype=mx.bool_,
        ),
        supply_vector=mx.zeros((groups, EXACT_SUPPLY_DIM)),
        staged_supply_vector=mx.zeros(
            (groups, candidates, EXACT_SUPPLY_DIM)
        ),
        selected_archetype=mx.zeros(
            (groups, candidates),
            dtype=mx.int32,
        ),
        frontier_features=mx.zeros(
            (groups, candidates, FRONTIER_FEATURE_DIM)
        ),
        intent_features=mx.full(
            (groups, candidates, INTENT_FEATURE_DIM),
            intent_value,
        ),
    )


def test_all_arms_have_identical_adapter_initialization() -> None:
    counts = {}
    layouts = {}
    tensors = {}
    for arm in ARMS:
        mx.random.seed(2026061719)
        model = O1IntentConditionedRanker(O1RankingModelConfig(arm=arm))
        model.freeze_base_for_adapter_training()
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
        tensors[arm] = parameter_tensor_blake3(model)

    assert len(set(counts.values())) == 1
    assert len(set(layouts.values())) == 1
    assert len(set(tensors.values())) == 1


def test_zero_initialized_adapter_is_exact_warm_start_behavior(tmp_path) -> None:
    batch = _batch(intent_value=0.75)
    mx.random.seed(17)
    base = R3ActionEditRanker()
    weights = tmp_path / "exact-r2.safetensors"
    base.save_weights(str(weights))

    mx.random.seed(23)
    model = O1IntentConditionedRanker(
        O1RankingModelConfig(arm=ARMS[2])
    )
    model.load_weights(str(weights), strict=False)
    expected = base(batch)
    observed = model(batch)
    mx.eval(expected.scores, observed.scores)

    np.testing.assert_array_equal(
        np.asarray(observed.scores),
        np.asarray(expected.scores),
    )


def test_adapter_training_freezes_exact_r2_and_output_heads() -> None:
    model = O1IntentConditionedRanker().freeze_base_for_adapter_training()
    trainable = dict(tree_flatten(model.trainable_parameters()))

    assert trainable
    assert all(
        name.startswith(("intent_projection.", "intent_fusion.", "intent_delta."))
        for name in trainable
    )
    assert "intent_delta.weight" in trainable
    assert "residual_head.weight" not in trainable
    assert "parent_encoder.common_encoder.token_projection.layers.0.weight" not in trainable


def test_o1_ranking_loss_is_finite() -> None:
    mx.random.seed(29)
    model = O1IntentConditionedRanker()
    loss = o1_ranking_loss(model, _batch(intent_value=0.2))
    mx.eval(loss)

    assert np.isfinite(float(loss.item()))
