from __future__ import annotations

from dataclasses import fields

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from cascadia_mlx.r2_map_model import (
    ACTION_FUSION_EXPANSION,
    ARCHITECTURE,
    EXACT_PARAMETER_COUNT,
    FORBIDDEN_MODEL_INPUT_NAMES,
    MULTITASK_DIM,
    PARAMETER_MAX,
    PARAMETER_MIN,
    PUBLIC_ACTION_TENSOR_NAMES,
    PUBLIC_MARKET_DECISION_TENSOR_NAMES,
    PUBLIC_STATE_TENSOR_NAMES,
    R2MapBatch,
    R2MapMarketDecisionBatch,
    R2MapModel,
    R2MapModelConfig,
    R2MapPrediction,
    R2MapPublicState,
    estimated_final_score,
    parameter_count,
    tensor_contract_manifest,
)
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from mlx.utils import tree_flatten


def _state(
    *,
    groups: int = 2,
    candidates: int | None = None,
    seed: int = 11,
    noisy_padding: bool = False,
) -> R2MapPublicState:
    rng = np.random.default_rng(seed)
    leading = (groups,) if candidates is None else (groups, candidates)
    token_shape = (*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY)
    token_features = np.zeros((*token_shape, TOKEN_FEATURES), dtype=np.float32)
    token_mask = np.zeros(token_shape, dtype=np.bool_)
    token_mask[..., :8] = True
    token_features[..., :8, :] = rng.normal(size=(*leading, BOARD_SLOTS, 8, TOKEN_FEATURES)).astype(
        np.float32
    )
    if noisy_padding:
        padding_rng = np.random.default_rng(seed + 10_000)
        token_features[..., 8:, :] = padding_rng.normal(
            size=(*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY - 8, TOKEN_FEATURES)
        ).astype(np.float32)
    token_types = np.zeros(token_shape, dtype=np.int32)
    token_types[..., :8] = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    return R2MapPublicState(
        token_features=mx.array(token_features),
        token_types=mx.array(token_types),
        token_mask=mx.array(token_mask),
        market_features=mx.array(
            rng.normal(size=(*leading, 4, MARKET_FEATURES)).astype(np.float32)
        ),
        market_mask=mx.ones((*leading, 4), dtype=mx.bool_),
        player_features=mx.array(
            rng.normal(size=(*leading, BOARD_SLOTS, PLAYER_FEATURES)).astype(np.float32)
        ),
        player_mask=mx.ones((*leading, BOARD_SLOTS), dtype=mx.bool_),
        global_features=mx.array(rng.normal(size=(*leading, GLOBAL_FEATURES)).astype(np.float32)),
    )


def _batch(*, noisy_padding: bool = False) -> R2MapBatch:
    candidate_mask = mx.array([[True, True, False], [True, True, True]])
    action = np.arange(2 * 3 * 140, dtype=np.float32).reshape(2, 3, 140) / 1000.0
    return R2MapBatch(
        parent=_state(seed=17, noisy_padding=noisy_padding),
        candidates=_state(candidates=3, seed=23, noisy_padding=noisy_padding),
        candidate_mask=candidate_mask,
        action_features=mx.array(action),
        exact_afterstate_scores=mx.array(
            [[20.0, 22.0, 999.0], [30.0, 31.0, 33.0]], dtype=mx.float32
        ),
    )


def _market_batch(*, noisy_padding: bool = False) -> R2MapMarketDecisionBatch:
    return R2MapMarketDecisionBatch(
        public_state=_state(seed=29, noisy_padding=noisy_padding),
        action_mask=mx.array([[True, True, False], [True, True, True]]),
        action_features=mx.array(np.arange(2 * 3 * 16, dtype=np.float32).reshape(2, 3, 16) / 100.0),
        exact_current_scores=mx.array([12.0, 34.0], dtype=mx.float32),
    )


def _evaluate(prediction: R2MapPrediction) -> None:
    mx.eval(
        prediction.action_scores,
        prediction.predicted_score_to_go,
        prediction.predicted_score_components_to_go,
        prediction.bootstrap_policy_logits,
        prediction.opponent_next_action.tile_slot_logits,
        prediction.opponent_next_action.wildlife_slot_logits,
        prediction.opponent_next_action.draft_kind_logits,
        prediction.opponent_next_action.drafted_wildlife_logits,
        prediction.opponent_next_action.replace_three_logits,
        prediction.opponent_next_action.paid_wipe_count_logits,
        prediction.opponent_next_action.paid_wipe_mask_logits,
        prediction.market_survival.disposition_logits,
        prediction.market_survival.pair_survival_logits,
        prediction.market_survival.final_slot_logits,
    )


def test_frozen_config_and_parameter_envelope() -> None:
    config = R2MapModelConfig()
    assert config.to_dict() == R2MapModelConfig.from_dict(config.to_dict()).to_dict()
    assert config.architecture == ARCHITECTURE
    assert config.hidden_dim == 192
    assert config.attention_heads == 4
    assert config.board_latents == 16
    assert config.board_latent_blocks == config.cross_board_blocks == 1
    assert config.feed_forward_multiplier == 2
    assert config.action_fusion_expansion == ACTION_FUSION_EXPANSION
    assert config.multitask_dim == MULTITASK_DIM == 24
    assert config.precision == "float32"

    mx.random.seed(20260618)
    model = R2MapModel(config)
    assert model.parent_encoder.common_encoder.board_token_capacity == (BOARD_TOKEN_CAPACITY) == 139
    count = parameter_count(model)
    assert count == EXACT_PARAMETER_COUNT == 4_531_853
    assert PARAMETER_MIN <= count <= PARAMETER_MAX
    for _, value in tree_flatten(model.trainable_parameters()):
        assert value.dtype == mx.float32


def test_live_encoder_rejects_legacy_92_slot_input() -> None:
    encoder = R2MapModel().parent_encoder.common_encoder
    state = _state(groups=1)
    with pytest.raises(ValueError, match="shape drifted"):
        encoder(
            state.token_features[:, :, :92, :],
            state.token_mask[:, :, :92],
            state.market_features,
            state.market_mask,
            state.player_features,
            state.player_mask,
            state.global_features,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hidden_dim": 256}, "width 192"),
        ({"attention_heads": 8}, "four attention heads"),
        ({"board_latents": 32}, "16 board latents"),
        ({"board_latent_blocks": 2}, "one board block"),
        ({"cross_board_blocks": 2}, "one board block"),
        ({"feed_forward_multiplier": 4}, "multiplier"),
        ({"multitask_dim": 32}, "bottleneck"),
        ({"precision": "float16"}, "float32"),
        ({"uncertainty_enabled": True}, "deferred"),
        ({"legal_affordance_enabled": True}, "deferred"),
        ({"public_transition_enabled": True}, "deferred"),
        ({"score_components_enabled": False}, "active"),
    ],
)
def test_architecture_drift_fails_closed(change: dict[str, object], message: str) -> None:
    values = R2MapModelConfig().to_dict()
    values.update(change)
    with pytest.raises(ValueError, match=message):
        R2MapModelConfig.from_dict(values)


def test_public_only_tensor_contract_is_exact_and_slotted() -> None:
    manifest = tensor_contract_manifest()
    assert manifest["parameter_count"] == EXACT_PARAMETER_COUNT
    assert manifest["parent_state_tensors"] == list(PUBLIC_STATE_TENSOR_NAMES)
    assert manifest["candidate_afterstate_tensors"] == list(PUBLIC_STATE_TENSOR_NAMES)
    assert manifest["action_tensors"] == list(PUBLIC_ACTION_TENSOR_NAMES)
    assert manifest["market_decision_tensors"] == list(PUBLIC_MARKET_DECISION_TENSOR_NAMES)
    assert set(manifest["forbidden_inputs"]) == FORBIDDEN_MODEL_INPUT_NAMES
    assert set(field.name for field in fields(R2MapPublicState)) == set(PUBLIC_STATE_TENSOR_NAMES)
    assert set(field.name for field in fields(R2MapBatch)) == {
        "parent",
        "candidates",
        *PUBLIC_ACTION_TENSOR_NAMES,
    }
    assert manifest["deferred_heads"] == [
        "uncertainty",
        "legal-affordance",
        "public-transition",
    ]
    batch = _batch()
    arguments = {field.name: getattr(batch, field.name) for field in fields(R2MapBatch)}
    with pytest.raises(TypeError):
        R2MapBatch(**arguments, policy_id=mx.array([1, 2]))
    market = _market_batch()
    market_arguments = {
        field.name: getattr(market, field.name) for field in fields(R2MapMarketDecisionBatch)
    }
    with pytest.raises(TypeError):
        R2MapMarketDecisionBatch(**market_arguments, future_refill=mx.array([1, 2]))


def test_market_decision_head_is_public_only_zero_initialized_and_tie_stable() -> None:
    mx.random.seed(20260618)
    model = R2MapModel()
    prediction = model.score_market_decisions(_market_batch())
    mx.eval(
        prediction.action_scores,
        prediction.predicted_score_to_go,
        prediction.bootstrap_policy_logits,
    )
    np.testing.assert_array_equal(
        np.asarray(prediction.action_scores)[:, :2], [[12.0, 12.0], [34.0, 34.0]]
    )
    assert np.isneginf(np.asarray(prediction.action_scores)[0, 2])
    np.testing.assert_array_equal(np.asarray(prediction.predicted_score_to_go), 0.0)
    np.testing.assert_array_equal(np.asarray(prediction.bootstrap_policy_logits)[:, :2], 0.0)
    # Canonical action zero (Keep or Stop) wins an exact untrained tie.
    np.testing.assert_array_equal(np.argmax(np.asarray(prediction.action_scores), axis=1), [0, 0])


def test_forward_shapes_are_finite_and_deferred_heads_are_absent() -> None:
    mx.random.seed(91)
    model = R2MapModel()
    prediction = model(_batch())
    _evaluate(prediction)
    assert prediction.action_scores.shape == (2, 3)
    assert prediction.predicted_score_to_go.shape == (2, 3)
    assert prediction.predicted_score_components_to_go.shape == (2, 3, 11)
    assert prediction.bootstrap_policy_logits.shape == (2, 3)
    assert prediction.opponent_next_action.tile_slot_logits.shape == (2, 3, 3, 4)
    assert prediction.opponent_next_action.wildlife_slot_logits.shape == (2, 3, 3, 4)
    assert prediction.opponent_next_action.draft_kind_logits.shape == (2, 3, 3, 2)
    assert prediction.opponent_next_action.drafted_wildlife_logits.shape == (2, 3, 3, 5)
    assert prediction.opponent_next_action.replace_three_logits.shape == (2, 3, 3, 2)
    assert prediction.opponent_next_action.paid_wipe_count_logits.shape == (2, 3, 3, 21)
    assert prediction.opponent_next_action.paid_wipe_mask_logits.shape == (2, 3, 3, 20, 16)
    assert prediction.market_survival.disposition_logits.shape == (2, 3, 4, 4)
    assert prediction.market_survival.pair_survival_logits.shape == (2, 3, 4, 2)
    assert prediction.market_survival.final_slot_logits.shape == (2, 3, 4, 4)
    assert not hasattr(model, "uncertainty_head")
    assert not hasattr(model, "legal_affordance_head")
    assert not hasattr(model, "public_transition_head")
    finite_outputs = (
        prediction.predicted_score_to_go,
        prediction.predicted_score_components_to_go,
        prediction.opponent_next_action.tile_slot_logits,
        prediction.market_survival.disposition_logits,
    )
    assert all(np.isfinite(np.asarray(output)).all() for output in finite_outputs)
    assert np.isneginf(np.asarray(prediction.action_scores)[0, 2])
    assert np.isneginf(np.asarray(prediction.bootstrap_policy_logits)[0, 2])


def test_auxiliary_logits_are_conditioned_on_each_candidate_afterstate() -> None:
    mx.random.seed(321)
    prediction = R2MapModel()(_batch())
    _evaluate(prediction)
    opponent = np.asarray(prediction.opponent_next_action.tile_slot_logits)
    survival = np.asarray(prediction.market_survival.disposition_logits)
    assert not np.array_equal(opponent[:, 0], opponent[:, 1])
    assert not np.array_equal(survival[:, 0], survival[:, 1])


def test_selected_only_training_auxiliaries_are_exactly_equivalent() -> None:
    mx.random.seed(322)
    model = R2MapModel()
    batch = _batch()
    selected = mx.array([1, 0], dtype=mx.int32)
    full = model(batch)
    compact = model(batch, selected_auxiliary_index=selected)
    indices = np.asarray(selected)

    np.testing.assert_array_equal(
        np.asarray(compact.action_scores),
        np.asarray(full.action_scores),
    )
    np.testing.assert_array_equal(
        np.asarray(compact.bootstrap_policy_logits),
        np.asarray(full.bootstrap_policy_logits),
    )
    for compact_values, full_values in (
        (
            compact.predicted_score_components_to_go,
            full.predicted_score_components_to_go,
        ),
        (
            compact.opponent_next_action.tile_slot_logits,
            full.opponent_next_action.tile_slot_logits,
        ),
        (
            compact.opponent_next_action.paid_wipe_mask_logits,
            full.opponent_next_action.paid_wipe_mask_logits,
        ),
        (
            compact.market_survival.disposition_logits,
            full.market_survival.disposition_logits,
        ),
    ):
        expected = np.asarray(full_values)[np.arange(2), indices][:, None]
        np.testing.assert_array_equal(np.asarray(compact_values), expected)


def test_live_inference_skips_training_heads_but_matches_full_action_outputs() -> None:
    mx.random.seed(77)
    model = R2MapModel()
    batch = _batch()
    full = model(batch)
    live = model.score_actions(batch)
    _evaluate(full)
    mx.eval(
        live.action_scores,
        live.predicted_score_to_go,
        live.predicted_score_components_to_go,
        live.bootstrap_policy_logits,
    )
    for name in (
        "action_scores",
        "predicted_score_to_go",
        "predicted_score_components_to_go",
        "bootstrap_policy_logits",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(live, name)),
            np.asarray(getattr(full, name)),
        )
    assert not hasattr(live, "opponent_next_action")
    assert not hasattr(live, "market_survival")


def test_fresh_model_uses_exact_afterstate_plus_predicted_remaining_equation() -> None:
    mx.random.seed(4)
    model = R2MapModel()
    batch = _batch()
    prediction = model(batch)
    _evaluate(prediction)
    expected = np.asarray(batch.exact_afterstate_scores)
    valid = np.asarray(batch.candidate_mask)
    np.testing.assert_array_equal(np.asarray(prediction.predicted_score_to_go)[valid], 0.0)
    np.testing.assert_allclose(np.asarray(prediction.action_scores)[valid], expected[valid])

    exact = mx.array([[10.0, 20.0]], dtype=mx.float32)
    remaining = mx.array([[3.0, -2.0]], dtype=mx.float32)
    final = estimated_final_score(exact, remaining)
    mx.eval(final)
    np.testing.assert_array_equal(np.asarray(final), [[13.0, 18.0]])
    with pytest.raises(ValueError, match="shapes differ"):
        estimated_final_score(exact, mx.ones((2,), dtype=mx.float32))


def test_multitask_projection_is_24_dimensional_and_tanh_bounded() -> None:
    model = R2MapModel()
    large = mx.full((2, 3, 192), 1_000.0, dtype=mx.float32)
    projected = model.multitask_state(large)
    mx.eval(projected)
    values = np.asarray(projected)
    assert values.shape == (2, 3, 24)
    assert (values >= -1.0).all() and (values <= 1.0).all()
    with pytest.raises(ValueError, match="width drifted"):
        model.multitask_state(mx.zeros((2, 191), dtype=mx.float32))


def test_padding_bytes_and_invalid_candidate_do_not_change_valid_scores() -> None:
    mx.random.seed(77)
    model = R2MapModel()
    clean_batch = _batch(noisy_padding=False)
    noisy_batch = _batch(noisy_padding=True)
    clean = model(clean_batch)
    noisy = model(noisy_batch)
    _evaluate(clean)
    _evaluate(noisy)
    valid = np.asarray(clean_batch.candidate_mask)
    np.testing.assert_allclose(
        np.asarray(clean.action_scores)[valid],
        np.asarray(noisy.action_scores)[valid],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(clean.predicted_score_components_to_go)[valid],
        np.asarray(noisy.predicted_score_components_to_go)[valid],
        rtol=1e-5,
        atol=1e-5,
    )


def test_parent_and_afterstate_encoders_run_once_per_grouped_call() -> None:
    class CountingEncoder(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def __call__(self, state: R2MapPublicState) -> mx.array:
            self.calls += 1
            return self.inner(state)

    model = R2MapModel()
    parent = CountingEncoder(model.parent_encoder)
    afterstate = CountingEncoder(model.afterstate_encoder)
    model.parent_encoder = parent
    model.afterstate_encoder = afterstate
    prediction = model(_batch())
    _evaluate(prediction)
    assert parent.calls == 1
    assert afterstate.calls == 1


def test_all_active_heads_participate_in_a_finite_gradient_step() -> None:
    mx.random.seed(123)
    model = R2MapModel()
    batch = _batch()
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)

    def synthetic_loss(candidate: R2MapModel, values: R2MapBatch) -> mx.array:
        prediction = candidate(values)
        valid = values.candidate_mask
        policy = mx.where(valid, prediction.bootstrap_policy_logits, 0.0)
        return (
            mx.mean(mx.square(prediction.predicted_score_to_go - 5.0) * valid)
            + mx.mean(mx.square(prediction.predicted_score_components_to_go - 1.0))
            + mx.mean(mx.square(policy - 0.5) * valid)
            + mx.mean(mx.square(prediction.opponent_next_action.tile_slot_logits))
            + mx.mean(mx.square(prediction.opponent_next_action.wildlife_slot_logits))
            + mx.mean(mx.square(prediction.opponent_next_action.draft_kind_logits))
            + mx.mean(mx.square(prediction.opponent_next_action.drafted_wildlife_logits))
            + mx.mean(mx.square(prediction.opponent_next_action.replace_three_logits))
            + mx.mean(mx.square(prediction.market_survival.disposition_logits))
            + mx.mean(mx.square(prediction.market_survival.pair_survival_logits))
            + mx.mean(mx.square(prediction.market_survival.final_slot_logits))
        )

    before = np.asarray(model.score_to_go_head.weight).copy()
    loss_and_gradient = nn.value_and_grad(model, synthetic_loss)
    loss, gradients = loss_and_gradient(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    assert np.isfinite(float(loss.item()))
    assert not np.array_equal(np.asarray(model.score_to_go_head.weight), before)


def test_tensor_shape_and_dtype_drift_fail_closed() -> None:
    batch = _batch()
    bad_action = R2MapBatch(
        parent=batch.parent,
        candidates=batch.candidates,
        candidate_mask=batch.candidate_mask,
        action_features=mx.zeros((2, 3, 139), dtype=mx.float32),
        exact_afterstate_scores=batch.exact_afterstate_scores,
    )
    with pytest.raises(ValueError, match="complete-action feature shape"):
        bad_action.validate()

    bad_parent = R2MapPublicState(
        token_features=batch.parent.token_features.astype(mx.float16),
        token_types=batch.parent.token_types,
        token_mask=batch.parent.token_mask,
        market_features=batch.parent.market_features,
        market_mask=batch.parent.market_mask,
        player_features=batch.parent.player_features,
        player_mask=batch.parent.player_mask,
        global_features=batch.parent.global_features,
    )
    with pytest.raises(ValueError, match="must be float32"):
        bad_parent.validate(candidates=False)
