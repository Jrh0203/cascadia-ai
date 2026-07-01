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
from cascadia_mlx.r3_action_edit_mlx_cache import ARMS as R3_ARMS
from cascadia_mlx.r3_action_edit_mlx_cache import R3_TOKEN_FEATURES
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditEncoding,
    R3ActionEditModelConfig,
    R3ActionEditRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
)
from cascadia_mlx.s4_candidate_context import (
    ANCHOR_LIMIT,
    RELATION_NEIGHBOR_LIMIT,
)
from cascadia_mlx.s4_candidate_relation_census import RELATIONS
from cascadia_mlx.s4_candidate_set_mlx_model import (
    S4_ARMS,
    S4CandidateContextMlxBatch,
    S4CandidateSetModelConfig,
    S4CandidateSetRanker,
)


def _batch(*, candidates: int = 4) -> SimpleNamespace:
    groups = 1
    token_width = 3
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
    candidate_token_mask = np.ones(
        (groups, candidates, token_width),
        dtype=np.bool_,
    )
    for candidate in range(candidates):
        candidate_features[0, candidate, :, 0] = 1
        candidate_features[0, candidate, 1, 16] = candidate / 10
        candidate_features[0, candidate, 2, 24] = (candidate + 1) / 10
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
        screen_value=mx.array(
            np.arange(90, 90 + candidates, dtype=np.float32)[None, :]
        ),
    )
    return SimpleNamespace(
        base=base,
        parent=parent,
        candidate_token_features=mx.array(candidate_features),
        candidate_token_mask=mx.array(candidate_token_mask),
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
    )


def _context(
    *,
    candidates: int = 4,
    relation_shift: int = 0,
) -> S4CandidateContextMlxBatch:
    anchors = np.zeros((1, ANCHOR_LIMIT), dtype=np.int32)
    anchor_mask = np.zeros((1, ANCHOR_LIMIT), dtype=np.bool_)
    anchors[0, :candidates] = np.arange(candidates)
    anchor_mask[0, :candidates] = True
    slots = np.zeros(
        (
            1,
            candidates,
            len(RELATIONS),
            RELATION_NEIGHBOR_LIMIT,
        ),
        dtype=np.int32,
    )
    neighbor_mask = np.zeros_like(slots, dtype=np.bool_)
    counts = np.zeros(
        (1, candidates, len(RELATIONS)),
        dtype=np.float32,
    )
    for candidate in range(candidates):
        relation = (candidate + relation_shift) % len(RELATIONS)
        slots[0, candidate, relation, 0] = (candidate + 1) % candidates
        neighbor_mask[0, candidate, relation, 0] = True
        counts[0, candidate, relation] = 1
    return S4CandidateContextMlxBatch(
        rows=mx.array([0], dtype=mx.int32),
        candidate_counts=mx.array([candidates], dtype=mx.int32),
        anchor_candidate_indices=mx.array(anchors),
        anchor_mask=mx.array(anchor_mask),
        relation_neighbor_anchor_slots=mx.array(slots),
        relation_neighbor_mask=mx.array(neighbor_mask),
        relation_anchor_sibling_counts=mx.array(counts),
    )


def _activate_context_outputs(model: S4CandidateSetRanker) -> None:
    hidden = model.config.hidden_dim
    model.inducing_delta.weight = mx.eye(hidden)
    model.inducing_delta.bias = mx.zeros((hidden,))
    model.relation_delta.weight = mx.eye(hidden)
    model.relation_delta.bias = mx.zeros((hidden,))
    model.residual_head.weight = mx.ones((1, hidden)) * 0.01
    model.residual_head.bias = mx.zeros((1,))


def test_all_s4_arms_have_identical_parameter_graph_and_initialization() -> None:
    counts = {}
    layouts = {}
    tensors = {}
    for arm in S4_ARMS:
        mx.random.seed(2026061717)
        model = S4CandidateSetRanker(S4CandidateSetModelConfig(arm=arm))
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
        tensors[arm] = parameter_tensor_blake3(model)

    assert len(set(counts.values())) == 1
    assert len(set(layouts.values())) == 1
    assert len(set(tensors.values())) == 1


def test_zero_initialized_context_is_exactly_r3_behavior() -> None:
    batch = _batch()
    context = _context()
    mx.random.seed(19)
    r3 = R3ActionEditRanker(R3ActionEditModelConfig(arm=R3_ARMS[3]))
    expected = r3(batch)
    observed = []
    for arm in S4_ARMS:
        mx.random.seed(19)
        model = S4CandidateSetRanker(S4CandidateSetModelConfig(arm=arm))
        observed.append(model.predict(batch, context).scores)
    mx.eval(expected.scores, *observed)

    for scores in observed:
        np.testing.assert_array_equal(
            np.asarray(scores),
            np.asarray(expected.scores),
        )


def test_r3_checkpoint_warm_start_preserves_predictions(tmp_path) -> None:
    batch = _batch()
    context = _context()
    mx.random.seed(23)
    r3 = R3ActionEditRanker(R3ActionEditModelConfig(arm=R3_ARMS[3]))
    weights = tmp_path / "r3.safetensors"
    r3.save_weights(str(weights))

    mx.random.seed(29)
    s4 = S4CandidateSetRanker(
        S4CandidateSetModelConfig(arm=S4_ARMS[3])
    )
    s4.load_weights(str(weights), strict=False)
    expected = r3(batch)
    observed = s4.predict(batch, context)
    mx.eval(expected.scores, observed.scores)

    np.testing.assert_array_equal(
        np.asarray(observed.scores),
        np.asarray(expected.scores),
    )


def test_control_gate_is_context_invariant_after_context_heads_activate() -> None:
    batch = _batch()
    mx.random.seed(31)
    model = S4CandidateSetRanker(
        S4CandidateSetModelConfig(arm=S4_ARMS[0])
    )
    _activate_context_outputs(model)
    first = model.predict(batch, _context(relation_shift=0))
    second = model.predict(batch, _context(relation_shift=3))
    mx.eval(first.scores, second.scores)

    np.testing.assert_array_equal(
        np.asarray(first.scores),
        np.asarray(second.scores),
    )


def test_inducing_latents_break_mean_max_pooling_collision() -> None:
    mx.random.seed(37)
    model = S4CandidateSetRanker(
        S4CandidateSetModelConfig(arm=S4_ARMS[1])
    )
    first = np.zeros((1, ANCHOR_LIMIT, 64), dtype=np.float32)
    second = np.zeros_like(first)
    first[0, :4, 0] = [0.0, 0.0, 1.0, 1.0]
    second[0, :4, 0] = [0.0, 0.5, 0.5, 1.0]
    first[0, :4, 1] = [1.0, 0.0, 1.0, 0.0]
    second[0, :4, 1] = [1.0, 1.0, 0.0, 0.0]
    mask = np.zeros((1, ANCHOR_LIMIT), dtype=np.bool_)
    mask[0, :4] = True
    assert np.array_equal(first.max(axis=1), second.max(axis=1))
    np.testing.assert_array_equal(first.mean(axis=1), second.mean(axis=1))

    first_latents = model.encode_inducing_latents(
        mx.array(first),
        mx.array(mask),
    )
    second_latents = model.encode_inducing_latents(
        mx.array(second),
        mx.array(mask),
    )
    mx.eval(first_latents, second_latents)

    assert not np.allclose(
        np.asarray(first_latents),
        np.asarray(second_latents),
        atol=1e-7,
    )


def test_relation_path_distinguishes_relation_identity() -> None:
    batch = _batch()
    mx.random.seed(41)
    model = S4CandidateSetRanker(
        S4CandidateSetModelConfig(arm=S4_ARMS[2])
    )
    prepared = model.prepare_context(batch, _context())
    encoding = R3ActionEditEncoding(
        hidden=model.encode_candidates(
            batch,
            parent_state=prepared.parent_state,
        ).hidden,
        candidate_mask=batch.base.candidate_mask,
    )
    first = model._relation_context(
        encoding.hidden,
        encoding.candidate_mask,
        prepared.anchor_hidden,
        _context(relation_shift=0),
    )
    second = model._relation_context(
        encoding.hidden,
        encoding.candidate_mask,
        prepared.anchor_hidden,
        _context(relation_shift=2),
    )
    mx.eval(first, second)

    assert not np.allclose(
        np.asarray(first),
        np.asarray(second),
        atol=1e-7,
    )


def test_chunked_context_scoring_matches_full_scoring() -> None:
    batch = _batch()
    context = _context()
    mx.random.seed(43)
    model = S4CandidateSetRanker(
        S4CandidateSetModelConfig(arm=S4_ARMS[3])
    )
    _activate_context_outputs(model)
    prepared = model.prepare_context(batch, context)
    full = model.predict(
        batch,
        context,
        prepared_context=prepared,
    )
    left = model.predict(
        batch,
        context,
        candidate_slice=slice(0, 2),
        prepared_context=prepared,
    )
    right = model.predict(
        batch,
        context,
        candidate_slice=slice(2, 4),
        prepared_context=prepared,
    )
    combined = mx.concatenate([left.scores, right.scores], axis=1)
    mx.eval(full.scores, combined)

    np.testing.assert_allclose(
        np.asarray(full.scores),
        np.asarray(combined),
        atol=1e-6,
    )
