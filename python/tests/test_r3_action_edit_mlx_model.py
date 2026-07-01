from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_cache import ARMS, R3_TOKEN_FEATURES
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    r3_action_edit_loss,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
)


def _batch(*, token_width: int = 3) -> SimpleNamespace:
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
        (groups, candidates, token_width, R3_TOKEN_FEATURES),
        dtype=np.float32,
    )
    candidate_mask = np.zeros(
        (groups, candidates, token_width),
        dtype=np.bool_,
    )
    candidate_features[:, :, :3, 0] = 1
    candidate_features[0, 1, 1, 16] = 0.5
    candidate_mask[:, :, :3] = True
    mask = mx.ones((groups, candidates), dtype=mx.bool_)
    base = SimpleNamespace(
        candidate_mask=mask,
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
        candidate_token_mask=mx.array(candidate_mask),
        supply_vector=mx.zeros((groups, EXACT_SUPPLY_DIM)),
        staged_supply_vector=mx.zeros((groups, candidates, EXACT_SUPPLY_DIM)),
        selected_archetype=mx.zeros(
            (groups, candidates),
            dtype=mx.int32,
        ),
        frontier_features=mx.zeros((groups, candidates, FRONTIER_FEATURE_DIM)),
    )


def test_all_arms_have_identical_parameter_layout_and_initial_tensors() -> None:
    counts = {}
    layouts = {}
    tensors = {}
    for arm in ARMS:
        mx.random.seed(2026061708)
        model = R3ActionEditRanker(R3ActionEditModelConfig(arm=arm))
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
        tensors[arm] = parameter_tensor_blake3(model)

    assert len(set(counts.values())) == 1
    assert len(set(layouts.values())) == 1
    assert len(set(tensors.values())) == 1


def test_candidate_padding_is_prediction_invariant() -> None:
    mx.random.seed(7)
    model = R3ActionEditRanker()
    compact = model(_batch(token_width=3))
    padded = model(_batch(token_width=7))
    mx.eval(compact.scores, padded.scores)
    np.testing.assert_allclose(
        np.asarray(compact.scores),
        np.asarray(padded.scores),
        atol=1e-6,
    )


def test_chunked_scoring_reuses_parent_and_matches_full_prediction() -> None:
    mx.random.seed(11)
    model = R3ActionEditRanker()
    batch = _batch()
    full = model(batch)
    parent = model.encode_parent(batch)
    chunk = model.predict(
        batch,
        candidate_slice=slice(0, 1),
        parent_state=parent,
    )
    loss = r3_action_edit_loss(model, batch)
    mx.eval(full.scores, chunk.scores, loss)

    np.testing.assert_allclose(
        np.asarray(full.scores)[:, :1],
        np.asarray(chunk.scores),
        atol=1e-6,
    )
    assert np.isfinite(float(loss))


def test_exposed_candidate_encoding_preserves_prediction() -> None:
    mx.random.seed(13)
    model = R3ActionEditRanker()
    batch = _batch()
    direct = model(batch)
    encoding = model.encode_candidates(batch)
    exposed = model.predict_from_encoding(batch, encoding)
    mx.eval(direct.scores, exposed.scores, encoding.hidden)

    np.testing.assert_allclose(
        np.asarray(direct.scores),
        np.asarray(exposed.scores),
        atol=1e-6,
    )
    assert encoding.hidden.shape == (1, 2, 64)
