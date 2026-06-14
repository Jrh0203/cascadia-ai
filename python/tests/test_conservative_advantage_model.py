from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from cascadia_mlx.action_ranking_dataset import (
    ACTION_BOARD_ENTITY_DIM,
    ACTION_DIM,
)
from cascadia_mlx.conservative_advantage_model import (
    ConservativeAdvantageModel,
    ConservativeAdvantageModelConfig,
    conservative_advantage_loss,
    conservative_advantage_scores,
)
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM


def test_conservative_advantage_model_outputs_finite_grouped_values() -> None:
    groups = 2
    candidates = 3
    board_entities = mx.zeros((groups, candidates, 4, 23, ACTION_BOARD_ENTITY_DIM))
    board_mask = mx.ones((groups, candidates, 4, 23), dtype=mx.bool_)
    market_entities = mx.zeros((groups, candidates, 4, ENTITY_DIM))
    market_mask = mx.ones((groups, candidates, 4), dtype=mx.bool_)
    global_features = mx.zeros((groups, candidates, GLOBAL_DIM))
    action_features = mx.zeros((groups, candidates, ACTION_DIM))
    batch = SimpleNamespace(
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
        lower_bound=mx.array([[1.0, -0.5, -1.0], [0.25, -0.25, 0.0]]),
        selected=mx.array([[True, False, False], [True, False, False]]),
    )
    model = ConservativeAdvantageModel(
        ConservativeAdvantageModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )

    scores = conservative_advantage_scores(model, batch)
    loss = conservative_advantage_loss(model, batch)
    mx.eval(scores, loss)

    assert scores.shape == (2, 3)
    assert float(loss.item()) > 0.0
