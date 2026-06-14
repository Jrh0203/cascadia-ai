from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from cascadia_mlx.action_ranking_dataset import (
    ACTION_BOARD_ENTITY_DIM,
    ACTION_DIM,
)
from cascadia_mlx.conservative_policy_model import (
    ConservativePolicyModel,
    ConservativePolicyModelConfig,
    conservative_policy_loss,
    conservative_policy_outputs,
)
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM


def _batch() -> SimpleNamespace:
    groups = 2
    candidates = 3
    board_entities = mx.zeros((groups, candidates, 4, 23, ACTION_BOARD_ENTITY_DIM))
    board_mask = mx.ones((groups, candidates, 4, 23), dtype=mx.bool_)
    market_entities = mx.zeros((groups, candidates, 4, ENTITY_DIM))
    market_mask = mx.ones((groups, candidates, 4), dtype=mx.bool_)
    global_features = mx.zeros((groups, candidates, GLOBAL_DIM))
    action_features = mx.zeros((groups, candidates, ACTION_DIM))
    return SimpleNamespace(
        anchor_board_entities=board_entities,
        anchor_board_mask=board_mask,
        anchor_market_entities=market_entities,
        anchor_market_mask=market_mask,
        anchor_global_features=global_features,
        anchor_action_features=action_features,
        candidate_board_entities=board_entities,
        candidate_board_mask=board_mask,
        candidate_market_entities=market_entities,
        candidate_market_mask=market_mask,
        candidate_global_features=global_features,
        candidate_action_features=action_features,
        candidate_mask=mx.array([[True, True, True], [True, True, False]]),
        lower_bound=mx.array([[1.0, -0.5, -1.0], [-0.25, -0.5, 0.0]]),
        selected=mx.array([[True, False, False], [False, False, False]]),
    )


def test_conservative_policy_model_outputs_two_finite_heads() -> None:
    model = ConservativePolicyModel(
        ConservativePolicyModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )

    lower_bound, policy_logits = conservative_policy_outputs(model, _batch())
    loss = conservative_policy_loss(model, _batch())
    mx.eval(lower_bound, policy_logits, loss)

    assert lower_bound.shape == (2, 3)
    assert policy_logits.shape == (2, 3)
    assert float(loss.item()) > 0.0


def test_conservative_policy_loss_rewards_selected_challenger_and_anchor() -> None:
    batch = _batch()
    selected_good = mx.array([[4.0, -2.0, -2.0], [-2.0, -2.0, -2.0]])
    selected_bad = mx.array([[-2.0, 4.0, -2.0], [4.0, -2.0, -2.0]])
    target = batch.lower_bound

    class FixedModel:
        config = ConservativePolicyModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )

        def __init__(self, logits: mx.array):
            self.logits = logits

        def __call__(self, *_args: object) -> tuple[mx.array, mx.array]:
            return target, self.logits

    good_loss = conservative_policy_loss(FixedModel(selected_good), batch)
    bad_loss = conservative_policy_loss(FixedModel(selected_bad), batch)
    mx.eval(good_loss, bad_loss)

    assert float(good_loss.item()) < float(bad_loss.item())
