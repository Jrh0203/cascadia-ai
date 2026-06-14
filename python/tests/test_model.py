from __future__ import annotations

import mlx.core as mx
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.model import EntitySetValueModel, ModelConfig


def test_entity_value_model_forward_shape_and_masking() -> None:
    mx.random.seed(7)
    model = EntitySetValueModel(
        ModelConfig(hidden_dim=32, attention_heads=4, board_blocks=1, market_blocks=1)
    )
    boards = mx.zeros((2, 4, 23, ENTITY_DIM))
    board_mask = mx.zeros((2, 4, 23), dtype=mx.bool_)
    board_mask[:, :, :3] = True
    market = mx.zeros((2, 4, ENTITY_DIM))
    market_mask = mx.ones((2, 4), dtype=mx.bool_)
    global_features = mx.zeros((2, GLOBAL_DIM))

    output = model(boards, board_mask, market, market_mask, global_features)
    mx.eval(output)

    assert output.shape == (2, 11)
    assert np.isfinite(np.asarray(output)).all()


def test_padded_board_entities_do_not_change_prediction() -> None:
    mx.random.seed(11)
    model = EntitySetValueModel(
        ModelConfig(hidden_dim=32, attention_heads=4, board_blocks=1, market_blocks=0)
    )
    boards = mx.zeros((1, 4, 23, ENTITY_DIM))
    board_mask = mx.zeros((1, 4, 23), dtype=mx.bool_)
    board_mask[:, :, :3] = True
    changed = np.zeros((1, 4, 23, ENTITY_DIM), dtype=np.float32)
    changed[:, :, 3:, :] = 1000.0
    market = mx.zeros((1, 4, ENTITY_DIM))
    market_mask = mx.ones((1, 4), dtype=mx.bool_)
    globals_ = mx.zeros((1, GLOBAL_DIM))

    left = model(boards, board_mask, market, market_mask, globals_)
    right = model(mx.array(changed), board_mask, market, market_mask, globals_)
    mx.eval(left, right)

    np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-6)
