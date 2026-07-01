from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.opportunity_cross_attention_mlx_metrics import (
    _batch_inputs,
    _model_batch,
)
from cascadia_mlx.opportunity_cross_attention_mlx_model import (
    ARMS,
    OpportunityCrossAttentionModelConfig,
    OpportunityCrossAttentionRanker,
    opportunity_cross_attention_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3_TOKEN_FEATURES
from cascadia_mlx.relational_substrate_mlx_cache import (
    RELATIONAL_VALUE_WIDTH,
    S5_FEATURES,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateRanker,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_SUPPLY_DIM,
    EXACT_TOKEN_COUNT,
    FRONTIER_FEATURE_DIM,
    SUPPLY_TOKEN_DIM,
)
from mlx.utils import tree_flatten


def _batch(
    *,
    r2_capacity: int = 3,
    supply_padding: int = 0,
) -> SimpleNamespace:
    groups = 1
    candidates = 2
    r2_features = np.zeros(
        (groups, BOARD_SLOTS, r2_capacity, TOKEN_FEATURES),
        dtype=np.float32,
    )
    r2_types = np.zeros(
        (groups, BOARD_SLOTS, r2_capacity),
        dtype=np.int32,
    )
    r2_mask = np.zeros_like(r2_types, dtype=np.bool_)
    r2_features[:, :, 0, 0] = 1.0
    r2_features[:, :, 1, 1] = 1.0
    r2_types[:, :, 0] = 1
    r2_types[:, :, 1] = 2
    r2_mask[:, :, :2] = True
    parent = SimpleNamespace(
        r2_token_features=mx.array(r2_features),
        r2_token_types=mx.array(r2_types),
        r2_token_mask=mx.array(r2_mask),
        relational_values=mx.zeros(
            (
                groups,
                BOARD_SLOTS,
                0,
                RELATIONAL_VALUE_WIDTH,
            ),
            dtype=mx.int16,
        ),
        relational_classes=mx.zeros(
            (groups, BOARD_SLOTS, 0),
            dtype=mx.int32,
        ),
        relational_mask=mx.zeros(
            (groups, BOARD_SLOTS, 0),
            dtype=mx.bool_,
        ),
        market_features=mx.zeros((groups, 4, MARKET_FEATURES)),
        market_mask=mx.ones((groups, 4), dtype=mx.bool_),
        player_features=mx.zeros(
            (groups, BOARD_SLOTS, PLAYER_FEATURES)
        ),
        player_mask=mx.ones(
            (groups, BOARD_SLOTS),
            dtype=mx.bool_,
        ),
        global_features=mx.zeros((groups, GLOBAL_FEATURES)),
        transform_ids=mx.zeros((groups,), dtype=mx.int32),
    )

    candidate_features = np.zeros(
        (groups, candidates, 3, R3_TOKEN_FEATURES),
        dtype=np.float32,
    )
    candidate_features[:, 0, :, 0] = 1.0
    candidate_features[:, 1, :, 1] = 1.0
    action_features = np.zeros((groups, candidates, 140), dtype=np.float32)
    action_features[:, 0, 0] = 1.0
    action_features[:, 1, 1] = 1.0
    candidate_mask = mx.ones((groups, candidates), dtype=mx.bool_)
    base = SimpleNamespace(
        candidate_mask=candidate_mask,
        action_features=mx.array(action_features),
        prior_features=mx.zeros((groups, candidates, 8)),
        staged_market_entities=mx.zeros(
            (groups, candidates, 4, 31)
        ),
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

    supply_tokens = np.zeros(
        (groups, EXACT_TOKEN_COUNT, SUPPLY_TOKEN_DIM),
        dtype=np.float32,
    )
    supply_tokens[:, 0, 0] = 1.0
    supply_tokens[:, 1, 1] = 1.0
    supply_mask = np.ones(
        (groups, EXACT_TOKEN_COUNT),
        dtype=np.bool_,
    )
    if supply_padding:
        supply_mask[:, -supply_padding:] = False
        supply_tokens[:, -supply_padding:, :] = 99.0
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
        supply_tokens=mx.array(supply_tokens),
        supply_mask=mx.array(supply_mask),
        selected_archetype=mx.zeros(
            (groups, candidates),
            dtype=mx.int32,
        ),
        frontier_features=mx.zeros(
            (groups, candidates, FRONTIER_FEATURE_DIM)
        ),
        derivative_features=mx.zeros(
            (groups, candidates, S5_FEATURES)
        ),
    )


def test_all_query_arms_have_identical_initial_graphs() -> None:
    counts = {}
    layouts = {}
    tensors = {}
    for arm in ARMS:
        mx.random.seed(2026061717)
        model = OpportunityCrossAttentionRanker(
            OpportunityCrossAttentionModelConfig(arm=arm)
        )
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
        tensors[arm] = parameter_tensor_blake3(model)
    assert len(set(counts.values())) == 1
    assert len(set(layouts.values())) == 1
    assert len(set(tensors.values())) == 1


def test_zero_initialized_adapter_is_exact_base_behavior() -> None:
    batch = _batch()
    mx.random.seed(19)
    expected = RelationalSubstrateRanker()(batch)
    observed = []
    for arm in ARMS:
        mx.random.seed(19)
        model = OpportunityCrossAttentionRanker(
            OpportunityCrossAttentionModelConfig(arm=arm)
        )
        observed.append(model(batch).scores)
    mx.eval(expected.scores, *observed)
    for scores in observed:
        np.testing.assert_array_equal(
            np.asarray(scores),
            np.asarray(expected.scores),
        )


def test_parent_and_candidate_query_routing_differ_as_registered() -> None:
    batch = _batch()
    mx.random.seed(23)
    control = OpportunityCrossAttentionRanker(
        OpportunityCrossAttentionModelConfig(arm=ARMS[0])
    )
    parent = control.encode_parent(batch)
    encoding = control.encode_base_candidates(batch, parent_state=parent)
    control_context = control.opportunity_context(batch, encoding, parent)

    mx.random.seed(23)
    combined = OpportunityCrossAttentionRanker(
        OpportunityCrossAttentionModelConfig(arm=ARMS[3])
    )
    combined_parent = combined.encode_parent(batch)
    combined_encoding = combined.encode_base_candidates(
        batch,
        parent_state=combined_parent,
    )
    combined_context = combined.opportunity_context(
        batch,
        combined_encoding,
        combined_parent,
    )
    mx.eval(
        control_context.supply_context,
        control_context.frontier_context,
        combined_context.supply_context,
        combined_context.frontier_context,
    )
    np.testing.assert_allclose(
        np.asarray(control_context.supply_context[:, 0]),
        np.asarray(control_context.supply_context[:, 1]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(control_context.frontier_context[:, 0]),
        np.asarray(control_context.frontier_context[:, 1]),
        atol=1e-6,
    )
    assert not np.allclose(
        np.asarray(combined_context.supply_context[:, 0]),
        np.asarray(combined_context.supply_context[:, 1]),
    )
    assert not np.allclose(
        np.asarray(combined_context.frontier_context[:, 0]),
        np.asarray(combined_context.frontier_context[:, 1]),
    )


def test_masked_supply_padding_is_context_invariant() -> None:
    mx.random.seed(29)
    model = OpportunityCrossAttentionRanker(
        OpportunityCrossAttentionModelConfig(arm=ARMS[1])
    )
    compact = _batch(supply_padding=1)
    changed = _batch(supply_padding=1)
    changed_tokens = np.asarray(changed.supply_tokens)
    changed_tokens[:, -1, :] = -123.0
    changed.supply_tokens = mx.array(changed_tokens)
    compact_parent = model.encode_parent(compact)
    changed_parent = model.encode_parent(changed)
    compact_encoding = model.encode_base_candidates(
        compact,
        parent_state=compact_parent,
    )
    changed_encoding = model.encode_base_candidates(
        changed,
        parent_state=changed_parent,
    )
    compact_context = model.opportunity_context(
        compact,
        compact_encoding,
        compact_parent,
    )
    changed_context = model.opportunity_context(
        changed,
        changed_encoding,
        changed_parent,
    )
    mx.eval(compact_context.supply_context, changed_context.supply_context)
    np.testing.assert_allclose(
        np.asarray(compact_context.supply_context),
        np.asarray(changed_context.supply_context),
        atol=1e-6,
    )


def test_exact_base_checkpoint_warm_start_preserves_predictions(
    tmp_path,
) -> None:
    batch = _batch()
    mx.random.seed(31)
    base = RelationalSubstrateRanker()
    weights = tmp_path / "base.safetensors"
    base.save_weights(str(weights))

    mx.random.seed(37)
    model = OpportunityCrossAttentionRanker(
        OpportunityCrossAttentionModelConfig(arm=ARMS[3])
    )
    model.load_weights(str(weights), strict=False)
    expected = base(batch)
    observed = model(batch)
    mx.eval(expected.scores, observed.scores)
    np.testing.assert_array_equal(
        np.asarray(observed.scores),
        np.asarray(expected.scores),
    )


def test_adapter_training_freezes_every_warm_started_parameter() -> None:
    mx.random.seed(39)
    model = OpportunityCrossAttentionRanker()
    total = parameter_count(model)
    model.freeze_base_for_adapter_training()
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert 0 < sum(int(value.size) for value in trainable.values()) < total
    assert "residual_head.weight" not in trainable
    assert "parent_encoder.r2_token_projection.layers.0.weight" not in trainable
    assert "context_delta.weight" in trainable
    assert "supply_cross_attention.attention.query_proj.weight" in trainable
    assert "frontier_cross_attention.attention.query_proj.weight" in trainable


def test_opportunity_cross_attention_loss_is_finite() -> None:
    mx.random.seed(41)
    model = OpportunityCrossAttentionRanker()
    loss = opportunity_cross_attention_loss(model, _batch())
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))


def test_compiled_batch_adapter_preserves_supply_memory_and_predictions() -> None:
    batch = _batch(supply_padding=2)
    reconstructed = _model_batch(_batch_inputs(batch))
    mx.random.seed(43)
    model = OpportunityCrossAttentionRanker(
        OpportunityCrossAttentionModelConfig(arm=ARMS[3])
    )
    expected = model(batch)
    observed = model(reconstructed)
    mx.eval(expected.scores, observed.scores)
    np.testing.assert_array_equal(
        np.asarray(observed.scores),
        np.asarray(expected.scores),
    )
    np.testing.assert_array_equal(
        np.asarray(reconstructed.supply_tokens),
        np.asarray(batch.supply_tokens),
    )
    np.testing.assert_array_equal(
        np.asarray(reconstructed.supply_mask),
        np.asarray(batch.supply_mask),
    )
