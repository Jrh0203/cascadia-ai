from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.p1_relational_pointer_model import (
    DESTINATION_EXISTING_TILE,
    DESTINATION_NEW_TILE,
    DESTINATION_NONE,
    DRAFT_OBSERVABLE_DIM,
    STAGE_ITEM_DIMS,
    STAGE_QUERY_DIMS,
    RelationalPointerModelConfig,
    RelationalPointerRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    relational_pointer_loss,
    trainable_parameter_names,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.relational_substrate_mlx_cache import RELATIONAL_VALUE_WIDTH


def _parent(groups: int = 2) -> SimpleNamespace:
    token_types = np.zeros(
        (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.int32,
    )
    token_types[:, :, :6] = np.array([1, 1, 2, 2, 3, 4])
    token_mask = token_types != 0
    rng = np.random.default_rng(9)
    return SimpleNamespace(
        r2_token_features=mx.array(
            rng.normal(
                size=(groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES)
            ).astype(np.float32)
            * token_mask[..., None]
        ),
        r2_token_types=mx.array(token_types),
        r2_token_mask=mx.array(token_mask),
        relational_values=mx.zeros(
            (groups, BOARD_SLOTS, 0, RELATIONAL_VALUE_WIDTH),
            dtype=mx.int8,
        ),
        relational_classes=mx.zeros(
            (groups, BOARD_SLOTS, 0),
            dtype=mx.int32,
        ),
        relational_mask=mx.zeros(
            (groups, BOARD_SLOTS, 0),
            dtype=mx.bool_,
        ),
        market_features=mx.zeros(
            (groups, 4, MARKET_FEATURES),
            dtype=mx.float32,
        ),
        market_mask=mx.ones((groups, 4), dtype=mx.bool_),
        player_features=mx.zeros(
            (groups, BOARD_SLOTS, PLAYER_FEATURES),
            dtype=mx.float32,
        ),
        player_mask=mx.ones((groups, BOARD_SLOTS), dtype=mx.bool_),
        global_features=mx.zeros(
            (groups, GLOBAL_FEATURES),
            dtype=mx.float32,
        ),
    )


def _batch(stage: str) -> SimpleNamespace:
    queries = 3
    items = 5
    query_dim = STAGE_QUERY_DIMS[stage]
    item_dim = STAGE_ITEM_DIMS[stage]
    item_mask = np.array(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    kinds = np.full((queries, items), DESTINATION_EXISTING_TILE, dtype=np.int32)
    kinds[0, 0] = DESTINATION_NONE
    kinds[1, 0] = DESTINATION_NEW_TILE
    return SimpleNamespace(
        shard_index=0,
        parent=_parent(),
        query_parent_indices=mx.array(np.array([0, 1, 0], dtype=np.int32)),
        query_features=mx.zeros((queries, query_dim), dtype=mx.float32),
        item_features=mx.zeros((queries, items, item_dim), dtype=mx.float32),
        item_pointer_indices=mx.array(
            np.array(
                [
                    [2, 3, 0, 0, 0],
                    [2, 3, 1, 0, 0],
                    [2, 3, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ),
        item_rotations=mx.array(
            np.tile(np.arange(items, dtype=np.int32) % 6, (queries, 1))
        ),
        item_kinds=mx.array(kinds),
        query_tile_pointer_indices=mx.array(
            np.array([2, 3, 2], dtype=np.int32)
        ),
        query_tile_rotations=mx.array(
            np.array([0, 1, 2], dtype=np.int32)
        ),
        item_mask=mx.array(item_mask),
        expected_rank=mx.array(
            np.tile(np.arange(1, items + 1, dtype=np.float32), (queries, 1))
        ),
        expected_rank_mask=mx.array(item_mask),
        target=mx.array(
            np.array(
                [
                    [True, False, False, False, False],
                    [True, True, False, False, False],
                    [False, True, False, False, False],
                ]
            )
        ),
    )


@pytest.mark.parametrize("stage", ("draft", "tile", "wildlife"))
def test_pointer_ranker_scores_only_legal_items(stage: str) -> None:
    mx.random.seed(17)
    model = RelationalPointerRanker(RelationalPointerModelConfig(stage=stage))
    batch = _batch(stage)
    scores = model(batch)
    loss = relational_pointer_loss(model, batch)
    mx.eval(scores, loss)
    values = np.asarray(scores)
    assert values.shape == (3, 5)
    assert np.isfinite(values[np.asarray(batch.item_mask)]).all()
    assert np.all(values[~np.asarray(batch.item_mask)] < -1e8)
    assert np.isfinite(float(loss.item()))


def test_parent_freeze_leaves_only_pointer_parameters_trainable() -> None:
    model = RelationalPointerRanker(
        RelationalPointerModelConfig(stage="wildlife")
    ).freeze_parent_for_pointer_training()
    names = trainable_parameter_names(model)
    assert names
    assert all(not name.startswith("parent_encoder.") for name in names)
    assert any(name.startswith("wildlife_pointer_projection.") for name in names)
    assert parameter_count(model) > 0
    assert len(parameter_layout_blake3(model, trainable_only=True)) == 64
    assert len(parameter_tensor_blake3(model, parent_only=True)) == 64
    with pytest.raises(ValueError, match="parent-only"):
        parameter_tensor_blake3(
            model,
            parent_only=True,
            trainable_only=True,
        )


def test_model_rejects_schema_drift() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        RelationalPointerModelConfig(stage="invalid").validate()
    with pytest.raises(ValueError, match="width 64"):
        RelationalPointerModelConfig(hidden_dim=128).validate()


def test_observable_dimensions_exclude_historical_descendant_statistics() -> None:
    assert DRAFT_OBSERVABLE_DIM == 275
    assert STAGE_QUERY_DIMS == {
        "draft": 1,
        "tile": 275,
        "wildlife": 283,
    }
    assert STAGE_ITEM_DIMS == {
        "draft": 275,
        "tile": 0,
        "wildlife": 0,
    }
