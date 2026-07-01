from __future__ import annotations

import numpy as np
from cascadia_mlx.p1_relational_pointer_data import (
    build_pointer_metadata,
    deterministic_transform_id,
    materialize_pointer_batch,
    validate_pointer_batch,
)
from cascadia_mlx.p1_relational_pointer_model import (
    DESTINATION_EXISTING_TILE,
    DESTINATION_NEW_TILE,
    DESTINATION_NONE,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_PAYLOAD_WIDTH,
)


def _factor_arrays() -> dict[str, np.ndarray]:
    draft = np.zeros(275, dtype=np.float32)
    draft[17 + 1] = 1.0
    tile_context = draft[None, :]
    tile_features = np.zeros((2, 249), dtype=np.float32)
    tile_features[0, :2] = np.array([1.0, 0.0]) / 24.0
    tile_features[1, :2] = np.array([0.0, 1.0]) / 24.0
    tile_features[:, 2] = 1.0
    wildlife_context = np.zeros((2, 463), dtype=np.float32)
    wildlife_context[:, :275] = draft
    wildlife_context[0, 275:277] = np.array([1.0, 0.0]) / 24.0
    wildlife_context[1, 275:277] = np.array([0.0, 1.0]) / 24.0
    wildlife_context[:, 277] = 1.0
    wildlife_features = np.zeros((4, 233), dtype=np.float32)
    wildlife_features[1, 0] = 1.0
    wildlife_features[1, 1:3] = np.array([1.0, 0.0]) / 24.0
    wildlife_features[2, 0] = 1.0
    wildlife_features[2, 1:3] = np.array([0.0, 0.0]) / 24.0
    wildlife_features[3, 0] = 1.0
    wildlife_features[3, 1:3] = np.array([0.0, 1.0]) / 24.0
    return {
        "group_id": np.array([7], dtype=np.uint64),
        "draft_query_group": np.array([0], dtype=np.int32),
        "draft_query_context": np.zeros((1, 1), dtype=np.float32),
        "draft_query_offsets": np.array([0, 1], dtype=np.int64),
        "draft_item_features": draft[None, :],
        "draft_item_rank": np.array([1.0], dtype=np.float32),
        "draft_item_rank_mask": np.array([True]),
        "draft_item_target": np.array([True]),
        "tile_query_group": np.array([0], dtype=np.int32),
        "tile_query_context": tile_context,
        "tile_query_offsets": np.array([0, 2], dtype=np.int64),
        "tile_item_features": tile_features,
        "tile_item_rank": np.array([1.0, 2.0], dtype=np.float32),
        "tile_item_rank_mask": np.array([True, True]),
        "tile_item_target": np.array([True, False]),
        "wildlife_query_group": np.array([0, 0], dtype=np.int32),
        "wildlife_query_context": wildlife_context,
        "wildlife_query_offsets": np.array([0, 2, 4], dtype=np.int64),
        "wildlife_item_features": wildlife_features,
        "wildlife_item_rank": np.array([2.0, 1.0, 1.0, 2.0], dtype=np.float32),
        "wildlife_item_rank_mask": np.ones(4, dtype=np.bool_),
        "wildlife_item_target": np.array([False, True, True, False]),
    }


def _r3_tensors() -> dict[str, np.ndarray]:
    token_types = np.zeros(
        (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.uint8,
    )
    token_payload = np.zeros(
        (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_PAYLOAD_WIDTH),
        dtype=np.int8,
    )
    token_seats = np.zeros_like(token_types)
    token_types[0, 0, :3] = np.array([1, 2, 2], dtype=np.uint8)
    token_payload[0, 0, 0, :2] = np.array([0, 0], dtype=np.int8)
    token_payload[0, 0, 1, :2] = np.array([1, 0], dtype=np.int8)
    token_payload[0, 0, 2, :2] = np.array([0, 1], dtype=np.int8)
    return {
        "parent_token_types": token_types,
        "parent_token_payload": token_payload,
        "parent_token_seats": token_seats,
        "parent_market_features": np.zeros(
            (1, 4, MARKET_FEATURES),
            dtype=np.float32,
        ),
        "parent_market_mask": np.ones((1, 4), dtype=np.uint8),
        "parent_player_features": np.zeros(
            (1, BOARD_SLOTS, PLAYER_FEATURES),
            dtype=np.float32,
        ),
        "parent_player_mask": np.ones((1, BOARD_SLOTS), dtype=np.uint8),
        "parent_global_features": np.zeros(
            (1, GLOBAL_FEATURES),
            dtype=np.float32,
        ),
    }


def test_metadata_maps_exact_frontier_and_wildlife_destinations() -> None:
    metadata = build_pointer_metadata(
        _factor_arrays(),
        r3_tensors=_r3_tensors(),
        r3_group_rows={7: 0},
    )
    assert metadata.tile_pointer_indices.tolist() == [1, 2]
    assert metadata.wildlife_query_tile_pointer_indices.tolist() == [1, 2]
    assert metadata.wildlife_kinds.tolist() == [
        DESTINATION_NONE,
        DESTINATION_NEW_TILE,
        DESTINATION_EXISTING_TILE,
        DESTINATION_NEW_TILE,
    ]
    assert metadata.wildlife_pointer_indices.tolist() == [0, 1, 0, 2]


def test_stage_specific_metadata_matches_full_resolution() -> None:
    arrays = _factor_arrays()
    tensors = _r3_tensors()
    full = build_pointer_metadata(
        arrays,
        r3_tensors=tensors,
        r3_group_rows={7: 0},
    )
    tile = build_pointer_metadata(
        arrays,
        r3_tensors=tensors,
        r3_group_rows={7: 0},
        stages=("tile",),
    )
    wildlife = build_pointer_metadata(
        arrays,
        r3_tensors=tensors,
        r3_group_rows={7: 0},
        stages=("wildlife",),
    )
    np.testing.assert_array_equal(
        tile.tile_pointer_indices,
        full.tile_pointer_indices,
    )
    np.testing.assert_array_equal(tile.tile_rotations, full.tile_rotations)
    np.testing.assert_array_equal(
        tile.tile_dual_terrain,
        full.tile_dual_terrain,
    )
    assert tile.wildlife_pointer_indices.size == 0
    np.testing.assert_array_equal(
        wildlife.wildlife_pointer_indices,
        full.wildlife_pointer_indices,
    )
    np.testing.assert_array_equal(
        wildlife.wildlife_kinds,
        full.wildlife_kinds,
    )
    np.testing.assert_array_equal(
        wildlife.wildlife_query_tile_pointer_indices,
        full.wildlife_query_tile_pointer_indices,
    )
    assert wildlife.tile_pointer_indices.size == 0


def test_materialized_batches_preserve_selected_prefix_contract() -> None:
    arrays = _factor_arrays()
    tensors = _r3_tensors()
    metadata = build_pointer_metadata(
        arrays,
        r3_tensors=tensors,
        r3_group_rows={7: 0},
    )
    for stage, selected in (
        ("draft", [0]),
        ("tile", [0]),
        ("wildlife", [0, 1]),
    ):
        batch = materialize_pointer_batch(
            arrays,
            metadata,
            r3_tensors=tensors,
            stage=stage,
            selected_queries=selected,
            seed=11,
            epoch=3,
            shard_index=0,
        )
        validate_pointer_batch(batch, stage=stage)
        assert batch.shard_index == 0
        assert batch.parent.r2_token_features.shape[0] == 1
        assert batch.parent_group_ids.tolist() == [7]
        assert batch.parent_transform_ids.shape == (1,)
        assert np.asarray(batch.item_mask).sum() == {
            "draft": 1,
            "tile": 2,
            "wildlife": 4,
        }[stage]


def test_d6_schedule_is_stable_and_uses_full_group_identity() -> None:
    first = deterministic_transform_id(
        seed=17,
        epoch=2,
        shard_index=4,
        group_id=9,
    )
    assert first == deterministic_transform_id(
        seed=17,
        epoch=2,
        shard_index=4,
        group_id=9,
    )
    values = {
        deterministic_transform_id(
            seed=17,
            epoch=2,
            shard_index=4,
            group_id=group,
        )
        for group in range(9, 40)
    }
    assert 0 <= first < 12
    assert len(values) > 1


def test_evaluation_batch_disables_d6_augmentation() -> None:
    arrays = _factor_arrays()
    tensors = _r3_tensors()
    metadata = build_pointer_metadata(
        arrays,
        r3_tensors=tensors,
        r3_group_rows={7: 0},
    )
    batch = materialize_pointer_batch(
        arrays,
        metadata,
        r3_tensors=tensors,
        stage="tile",
        selected_queries=[0],
        seed=11,
        epoch=3,
        shard_index=0,
        d6_augment=False,
    )
    assert batch.transform_ids.tolist() == [0]
    assert np.asarray(batch.item_rotations).tolist() == [[0, 0]]
