from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from cascadia_mlx.action_ranking_dataset import (
    ACTION_BOARD_ENTITY_DIM,
    ACTION_DIM,
)
from cascadia_mlx.action_ranking_model import (
    ActionDeltaRanker,
    ActionRankingModelConfig,
    action_ranking_loss,
)
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM


def test_action_ranker_outputs_grouped_scores_and_finite_loss() -> None:
    groups = 2
    candidates = 3
    batch = SimpleNamespace(
        board_entities=mx.zeros((groups, candidates, 4, 23, ACTION_BOARD_ENTITY_DIM)),
        board_mask=mx.ones((groups, candidates, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((groups, candidates, 4, ENTITY_DIM)),
        market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        global_features=mx.zeros((groups, candidates, GLOBAL_DIM)),
        action_features=mx.zeros((groups, candidates, ACTION_DIM)),
        candidate_mask=mx.array([[True, True, True], [True, True, False]]),
        teacher_mean=mx.array([[4.0, 3.0, 1.0], [2.0, 1.0, 0.0]]),
        teacher_stddev=mx.ones((groups, candidates)),
    )
    model = ActionDeltaRanker(
        ActionRankingModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )

    scores = model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
    )
    loss = action_ranking_loss(model, batch)
    mx.eval(scores, loss)

    assert scores.shape == (2, 3)
    assert float(loss.item()) > 0.0
