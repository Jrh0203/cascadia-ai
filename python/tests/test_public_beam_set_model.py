from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from cascadia_mlx.action_ranking_dataset import ACTION_BOARD_ENTITY_DIM, ACTION_DIM
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.public_beam_set_model import (
    CORRECTION_SCALE,
    PublicBeamSetModelConfig,
    PublicBeamSetRanker,
    public_beam_set_loss,
    public_beam_set_scores,
)


def test_public_beam_set_model_outputs_bounded_grouped_scores() -> None:
    groups = 2
    candidates = 3
    immediate = mx.array([[90.0, 91.0, 92.0], [88.0, 89.0, 0.0]])
    batch = SimpleNamespace(
        board_entities=mx.zeros((groups, candidates, 4, 23, ACTION_BOARD_ENTITY_DIM)),
        board_mask=mx.ones((groups, candidates, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((groups, candidates, 4, ENTITY_DIM)),
        market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        global_features=mx.zeros((groups, candidates, GLOBAL_DIM)),
        action_features=mx.zeros((groups, candidates, ACTION_DIM)),
        candidate_mask=mx.array([[True, True, True], [True, True, False]]),
        target_mean=mx.array([[91.0, 92.0, 92.0], [88.0, 89.5, 0.0]]),
        immediate_score=immediate,
    )
    model = PublicBeamSetRanker(
        PublicBeamSetModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            candidate_blocks=1,
        )
    )

    scores = public_beam_set_scores(model, batch)
    loss = public_beam_set_loss(model, batch)
    mx.eval(scores, loss)

    assert scores.shape == (groups, candidates)
    assert mx.all(mx.isfinite(scores)).item()
    assert mx.max(mx.abs(scores - immediate)).item() <= CORRECTION_SCALE
    assert float(loss.item()) > 0.0
