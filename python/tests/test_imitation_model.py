from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from cascadia_mlx.imitation_model import (
    IMITATION_ARCHITECTURE_CROSS_V2,
    IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
    IMITATION_ARCHITECTURE_RESIDUAL_V2,
    IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
    ImitationModelConfig,
    SharedStateActionRanker,
    distributional_imitation_loss_from_scores,
    imitation_loss,
    score_imitation_actions,
    score_residual_imitation_loss_from_scores,
)


def test_imitation_loss_is_finite_for_one_hot_groups() -> None:
    model = SharedStateActionRanker(
        ImitationModelConfig(
            hidden_dim=16,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    batch = SimpleNamespace(
        board_entities=mx.zeros((1, 4, 23, 31)),
        board_mask=mx.ones((1, 4, 23)),
        market_entities=mx.zeros((1, 4, 31)),
        market_mask=mx.ones((1, 4)),
        global_features=mx.zeros((1, 96)),
        action_features=mx.zeros((1, 2, 52)),
        candidate_mask=mx.array([[True, True]]),
        teacher_mean=mx.array([[1.0, 0.0]]),
    )

    loss = imitation_loss(model, batch)
    mx.eval(loss)
    assert float(loss.item()) > 0.0


def test_cross_attention_ranker_scores_shared_state_actions() -> None:
    model = SharedStateActionRanker(
        ImitationModelConfig(
            architecture=IMITATION_ARCHITECTURE_CROSS_V2,
            hidden_dim=16,
            attention_heads=4,
            board_blocks=1,
            market_blocks=1,
        )
    )
    scores = model(
        mx.zeros((2, 4, 23, 31)),
        mx.ones((2, 4, 23)),
        mx.zeros((2, 4, 31)),
        mx.ones((2, 4)),
        mx.zeros((2, 96)),
        mx.zeros((2, 5, 52)),
    )

    mx.eval(scores)
    assert scores.shape == (2, 5)
    assert bool(mx.all(mx.isfinite(scores)).item())


def test_residual_ranker_preserves_monotonic_immediate_prior() -> None:
    model = SharedStateActionRanker(
        ImitationModelConfig(
            architecture=IMITATION_ARCHITECTURE_RESIDUAL_V2,
            hidden_dim=16,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            immediate_rank_prior=0.08,
        )
    )
    actions = mx.zeros((1, 2, 52))
    actions = actions.at[0, 0, 50].add(1.0 / 4096.0)
    actions = actions.at[0, 1, 50].add(2.0 / 4096.0)
    scores = score_imitation_actions(
        model,
        mx.zeros((1, 4, 23, 31)),
        mx.ones((1, 4, 23)),
        mx.zeros((1, 4, 31)),
        mx.ones((1, 4)),
        mx.zeros((1, 96)),
        actions,
        mx.array([[True, True]]),
    )

    mx.eval(scores)
    assert float(scores[0, 0].item()) > float(scores[0, 1].item())


def test_distributional_loss_prefers_rollout_order_and_selected_action() -> None:
    batch = SimpleNamespace(
        candidate_mask=mx.array([[True, True, True]]),
        teacher_mean=mx.array([[90.0, 85.0, 80.0]]),
        teacher_stddev=mx.array([[2.0, 2.0, 2.0]]),
        teacher_samples=mx.array([[40.0, 20.0, 10.0]]),
        teacher_scored=mx.array([[True, True, True]]),
        selected=mx.array([[True, False, False]]),
    )
    ordered = distributional_imitation_loss_from_scores(
        mx.array([[2.0, 0.0, -2.0]]),
        batch,
    )
    reversed_scores = distributional_imitation_loss_from_scores(
        mx.array([[-2.0, 0.0, 2.0]]),
        batch,
    )

    mx.eval(ordered, reversed_scores)
    assert bool(mx.isfinite(ordered).item())
    assert float(ordered.item()) < float(reversed_scores.item())


def test_score_residual_starts_as_exact_immediate_score() -> None:
    model = SharedStateActionRanker(
        ImitationModelConfig(
            architecture=IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
            hidden_dim=16,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    actions = mx.zeros((1, 2, 52))
    actions = actions.at[0, 0, 51].add(0.42)
    actions = actions.at[0, 1, 51].add(0.39)
    scores = score_imitation_actions(
        model,
        mx.zeros((1, 4, 23, 31)),
        mx.ones((1, 4, 23)),
        mx.zeros((1, 4, 31)),
        mx.ones((1, 4)),
        mx.zeros((1, 96)),
        actions,
        mx.array([[True, True]]),
    )

    mx.eval(scores)
    assert mx.allclose(scores, mx.array([[42.0, 39.0]]), atol=1e-5)


def test_parent_set_residual_starts_with_exact_parent_order() -> None:
    model = SharedStateActionRanker(
        ImitationModelConfig(
            architecture=IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
            hidden_dim=16,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    parent = mx.array([[94.0, 91.0, 96.0]])
    ranks = mx.array([[2.0, 3.0, 1.0]])
    mask = mx.array([[True, True, True]])
    scores = score_imitation_actions(
        model,
        mx.zeros((1, 4, 23, 31)),
        mx.ones((1, 4, 23)),
        mx.zeros((1, 4, 31)),
        mx.ones((1, 4)),
        mx.zeros((1, 96)),
        mx.zeros((1, 3, 52)),
        mask,
        parent,
        ranks,
    )

    mx.eval(scores)
    assert int(mx.argmax(scores, axis=1)[0].item()) == 2
    assert bool(mx.all(mx.isfinite(scores)).item())


def test_parent_set_residual_is_permutation_equivariant_at_initialization() -> None:
    model = SharedStateActionRanker(
        ImitationModelConfig(
            architecture=IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
            hidden_dim=16,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    actions = mx.arange(3 * 52).reshape(1, 3, 52) / 100.0
    parent = mx.array([[94.0, 91.0, 96.0]])
    ranks = mx.array([[2.0, 3.0, 1.0]])
    mask = mx.array([[True, True, True]])
    permutation = mx.array([2, 0, 1])
    common = (
        mx.zeros((1, 4, 23, 31)),
        mx.ones((1, 4, 23)),
        mx.zeros((1, 4, 31)),
        mx.ones((1, 4)),
        mx.zeros((1, 96)),
    )
    original = score_imitation_actions(
        model,
        *common,
        actions,
        mask,
        parent,
        ranks,
    )
    permuted = score_imitation_actions(
        model,
        *common,
        actions[:, permutation],
        mask,
        parent[:, permutation],
        ranks[:, permutation],
    )

    mx.eval(original, permuted)
    assert mx.allclose(permuted, original[:, permutation], atol=1e-6)


def test_score_residual_loss_prefers_point_accurate_predictions() -> None:
    batch = SimpleNamespace(
        candidate_mask=mx.array([[True, True, True]]),
        teacher_mean=mx.array([[90.0, 85.0, 80.0]]),
        teacher_stddev=mx.array([[2.0, 2.0, 2.0]]),
        teacher_samples=mx.array([[40.0, 20.0, 10.0]]),
        teacher_scored=mx.array([[True, True, True]]),
        selected=mx.array([[True, False, False]]),
    )
    accurate = score_residual_imitation_loss_from_scores(
        mx.array([[90.0, 85.0, 80.0]]),
        batch,
    )
    reversed_scores = score_residual_imitation_loss_from_scores(
        mx.array([[80.0, 85.0, 90.0]]),
        batch,
    )

    mx.eval(accurate, reversed_scores)
    assert float(accurate.item()) < float(reversed_scores.item())
