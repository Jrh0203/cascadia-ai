from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM, TARGET_DIM
from cascadia_mlx.score_to_go_model import (
    EDGE_AWARE_HEX_SCORE_TO_GO_V2,
    ScoreToGoModelConfig,
    ScoreToGoValueModel,
    hex_graph_relations,
    score_to_go_loss,
)


def test_score_to_go_model_supports_signed_outputs_and_loss() -> None:
    model = ScoreToGoValueModel(
        ScoreToGoModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    batch = SimpleNamespace(
        board_entities=mx.zeros((2, 4, 23, ENTITY_DIM)),
        board_mask=mx.ones((2, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((2, 4, ENTITY_DIM)),
        market_mask=mx.ones((2, 4), dtype=mx.bool_),
        global_features=mx.zeros((2, GLOBAL_DIM)),
        targets=mx.array(
            [[1.0] * TARGET_DIM, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
        ),
    )

    predictions = model.predict_components(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
    )
    loss = score_to_go_loss(model, batch)
    mx.eval(predictions, loss)

    assert predictions.shape == (2, TARGET_DIM)
    assert float(loss.item()) > 0.0


def test_hex_graph_relations_preserve_direction_and_matching_terrain() -> None:
    boards = np.zeros((1, 4, 23, ENTITY_DIM), dtype=np.float32)
    mask = np.zeros((1, 4, 23), dtype=np.bool_)
    mask[0, 0, :2] = True
    boards[0, 0, 0, 2] = 1.0
    boards[0, 0, 0, 12] = 1.0
    boards[0, 0, 0, 13] = 1.0
    boards[0, 0, 1, 0] = 1.0 / 24.0
    boards[0, 0, 1, 2] = 1.0
    boards[0, 0, 1, 12] = 1.0
    boards[0, 0, 1, 13] = 1.0

    adjacency, edge_matches = hex_graph_relations(mx.array(boards), mx.array(mask))
    mx.eval(*adjacency, *edge_matches)

    assert np.asarray(adjacency[0])[0, 0, 1] == 1.0
    assert np.asarray(adjacency[3])[0, 1, 0] == 1.0
    assert np.asarray(edge_matches[0])[0, 0, 0] == 1.0
    assert np.asarray(edge_matches[3])[0, 1, 0] == 1.0
    assert sum(np.asarray(value).sum() for value in adjacency) == 2.0


def test_edge_aware_score_to_go_forward_loss_and_padding() -> None:
    mx.random.seed(19)
    model = ScoreToGoValueModel(
        ScoreToGoModelConfig(
            architecture=EDGE_AWARE_HEX_SCORE_TO_GO_V2,
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            graph_blocks=2,
            market_blocks=0,
        )
    )
    boards = np.zeros((4, 4, 23, ENTITY_DIM), dtype=np.float32)
    mask = np.zeros((4, 4, 23), dtype=np.bool_)
    mask[:, :, :2] = True
    boards[:, :, 0, 2] = 1.0
    boards[:, :, 0, 12] = 1.0
    boards[:, :, 0, 13] = 1.0
    boards[:, :, 1, 0] = 1.0 / 24.0
    boards[:, :, 1, 2] = 1.0
    boards[:, :, 1, 12] = 1.0
    boards[:, :, 1, 13] = 1.0
    changed = boards.copy()
    changed[:, :, 2:, :] = 1000.0
    batch = SimpleNamespace(
        board_entities=mx.array(boards),
        board_mask=mx.array(mask),
        market_entities=mx.zeros((4, 4, ENTITY_DIM)),
        market_mask=mx.ones((4, 4), dtype=mx.bool_),
        global_features=mx.zeros((4, GLOBAL_DIM)),
        targets=mx.ones((4, TARGET_DIM)),
        current_targets=mx.zeros((4, TARGET_DIM)),
        final_targets=mx.arange(4, dtype=mx.float32)[:, None] * mx.ones((4, TARGET_DIM)),
        game_index=mx.zeros((4,), dtype=mx.int32),
        turn=mx.arange(4, dtype=mx.int32),
    )

    predictions = model.predict_components(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
    )
    padded_predictions = model.predict_components(
        mx.array(changed),
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
    )
    loss = score_to_go_loss(model, batch)
    mx.eval(predictions, padded_predictions, loss)

    assert predictions.shape == (4, TARGET_DIM)
    assert np.isfinite(np.asarray(predictions)).all()
    assert float(loss.item()) > 0.0
    np.testing.assert_allclose(
        np.asarray(predictions),
        np.asarray(padded_predictions),
        atol=1e-6,
    )
