from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.ranking_model import EntitySetRanker, RankingModelConfig, ranking_loss


def _batch() -> SimpleNamespace:
    groups = 2
    candidates = 3
    return SimpleNamespace(
        board_entities=mx.zeros((groups, candidates, 4, 23, ENTITY_DIM)),
        board_mask=mx.ones((groups, candidates, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((groups, candidates, 4, ENTITY_DIM)),
        market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        global_features=mx.zeros((groups, candidates, GLOBAL_DIM)),
        candidate_mask=mx.array([[True, True, True], [True, True, False]]),
        teacher_mean=mx.array([[4.0, 3.0, 1.0], [2.0, 1.0, 0.0]]),
        teacher_stddev=mx.ones((groups, candidates)),
        immediate_rank=mx.array([[1.0, 2.0, 9.0], [1.0, 4.0, 0.0]]),
        immediate_score=mx.array([[40.0, 39.0, 35.0], [42.0, 38.0, 0.0]]),
    )


def test_ranker_outputs_one_score_per_candidate_and_finite_loss() -> None:
    model = EntitySetRanker(
        RankingModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    batch = _batch()
    scores = model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
    )
    loss = ranking_loss(model, batch)
    mx.eval(scores, loss)

    assert scores.shape == (2, 3)
    assert float(loss.item()) > 0.0
