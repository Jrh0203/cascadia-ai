from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3_TOKEN_FEATURES
from cascadia_mlx.relational_substrate_mlx_cache import (
    ARMS,
    RELATIONAL_VALUE_WIDTH,
    S5_FEATURES,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateModelConfig,
    RelationalSubstrateRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    relational_substrate_loss,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
)


def _batch(
    *,
    r2_capacity: int = 0,
    relational_capacity: int = 2,
    derivative_value: float = 0.0,
) -> SimpleNamespace:
    groups = 1
    candidates = 2
    r2_features = np.zeros(
        (groups, BOARD_SLOTS, r2_capacity, TOKEN_FEATURES),
        dtype=np.float32,
    )
    r2_types = np.zeros((groups, BOARD_SLOTS, r2_capacity), dtype=np.int32)
    r2_mask = np.zeros_like(r2_types, dtype=np.bool_)
    if r2_capacity:
        r2_features[:, :, 0, 0] = 1.0
        r2_types[:, :, 0] = 1
        r2_mask[:, :, 0] = True

    relational_values = np.zeros(
        (
            groups,
            BOARD_SLOTS,
            relational_capacity,
            RELATIONAL_VALUE_WIDTH,
        ),
        dtype=np.int16,
    )
    relational_classes = np.zeros(
        (groups, BOARD_SLOTS, relational_capacity),
        dtype=np.int32,
    )
    relational_mask = np.zeros_like(relational_classes, dtype=np.bool_)
    if relational_capacity:
        relational_values[:, :, 0, 0] = 3
        relational_classes[:, :, 0] = 1
        relational_mask[:, :, 0] = True
    parent = SimpleNamespace(
        r2_token_features=mx.array(r2_features),
        r2_token_types=mx.array(r2_types),
        r2_token_mask=mx.array(r2_mask),
        relational_values=mx.array(relational_values),
        relational_classes=mx.array(relational_classes),
        relational_mask=mx.array(relational_mask),
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
    candidate_mask = mx.ones((groups, candidates), dtype=mx.bool_)
    base = SimpleNamespace(
        candidate_mask=candidate_mask,
        action_features=mx.zeros((groups, candidates, 140)),
        prior_features=mx.zeros((groups, candidates, 8)),
        staged_market_entities=mx.zeros((groups, candidates, 4, 31)),
        staged_market_mask=mx.ones(
            (groups, candidates, 4),
            dtype=mx.bool_,
        ),
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
        derivative_features=mx.full(
            (groups, candidates, S5_FEATURES),
            derivative_value,
        ),
    )


def test_all_relational_arms_have_identical_layout_and_initial_tensors() -> None:
    counts = {}
    layouts = {}
    tensors = {}
    for arm in ARMS:
        mx.random.seed(2026061716)
        model = RelationalSubstrateRanker(
            RelationalSubstrateModelConfig(arm=arm)
        )
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
        tensors[arm] = parameter_tensor_blake3(model)
    assert len(set(counts.values())) == 1
    assert len(set(layouts.values())) == 1
    assert len(set(tensors.values())) == 1


def test_native_r2_and_relational_only_parent_surfaces_both_execute() -> None:
    mx.random.seed(9)
    model = RelationalSubstrateRanker()
    exact = model(_batch(r2_capacity=2, relational_capacity=0))
    compact = model(_batch(r2_capacity=0, relational_capacity=2))
    mx.eval(exact.scores, compact.scores)
    assert exact.scores.shape == compact.scores.shape == (1, 2)
    assert np.isfinite(np.asarray(exact.scores)).all()
    assert np.isfinite(np.asarray(compact.scores)).all()


def test_relational_padding_is_prediction_invariant() -> None:
    mx.random.seed(17)
    model = RelationalSubstrateRanker()
    compact = model(_batch(relational_capacity=1))
    padded = model(_batch(relational_capacity=8))
    mx.eval(compact.scores, padded.scores)
    np.testing.assert_allclose(
        np.asarray(compact.scores),
        np.asarray(padded.scores),
        atol=1e-6,
    )


def test_derivative_adapter_changes_candidate_encoding() -> None:
    mx.random.seed(23)
    model = RelationalSubstrateRanker()
    absent = model.encode_candidates(_batch(derivative_value=0.0))
    present = model.encode_candidates(_batch(derivative_value=1.0))
    mx.eval(absent.hidden, present.hidden)
    assert not np.allclose(
        np.asarray(absent.hidden),
        np.asarray(present.hidden),
    )


def test_relational_ranker_loss_is_finite() -> None:
    mx.random.seed(29)
    model = RelationalSubstrateRanker()
    loss = relational_substrate_loss(model, _batch(derivative_value=0.5))
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))
