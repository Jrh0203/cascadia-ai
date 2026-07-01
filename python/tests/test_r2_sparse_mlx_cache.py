from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from cascadia_mlx.d6_contract import D6_CONTRACT
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GRAPH_MAX_DEGREE,
    TOKEN_CAPACITY,
    TOKEN_FEATURES,
    TOKEN_PAYLOAD_WIDTH,
    TOKEN_TYPE_COMPONENT,
    TOKEN_TYPE_FRONTIER,
    TOKEN_TYPE_MOTIF,
    TOKEN_TYPE_OCCUPIED,
    R2SparseMlxCache,
    R2SparseMlxCacheError,
    _layout_from_board_type_counts,
    _materialize_token_features,
    _transform_payload_in_place,
)


def test_all_d6_payload_transforms_round_trip_without_rederiving_semantics() -> None:
    token_types = np.zeros((1, TOKEN_CAPACITY), dtype=np.uint8)
    token_types[0, :4] = [
        TOKEN_TYPE_OCCUPIED,
        TOKEN_TYPE_FRONTIER,
        TOKEN_TYPE_COMPONENT,
        TOKEN_TYPE_MOTIF,
    ]
    payload = np.zeros((1, TOKEN_CAPACITY, TOKEN_PAYLOAD_WIDTH), dtype=np.int8)

    payload[0, 0, :14] = [2, -1, 0, 1, 4, 0, 1, 2, 3, 4, 0, 0x1F, 2, 0]
    payload[0, 1, :17] = [
        -2,
        1,
        0b101101,
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
        3,
        4,
        5,
        2,
        0b101,
        1,
    ]
    payload[0, 1, 17:21] = [2, 7, 3, 0b001011]
    payload[0, 2, :10] = [3, 7, 2, 1, 6, 4, -1, 0, 2, -2]
    payload[0, 3, :15] = [
        1,
        2,
        4,
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
        3,
        4,
        5,
        0b100101,
    ]

    for transform_id, inverse_id in enumerate(D6_CONTRACT.inverse_table):
        transformed = payload.copy()
        _transform_payload_in_place(
            transformed,
            token_types,
            np.asarray([transform_id], dtype=np.int64),
        )
        _transform_payload_in_place(
            transformed,
            token_types,
            np.asarray([inverse_id], dtype=np.int64),
        )
        np.testing.assert_array_equal(transformed, payload)


def test_token_features_preserve_four_board_ownership_and_zero_padding() -> None:
    token_types = np.zeros(
        (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.uint8,
    )
    token_seats = np.zeros_like(token_types)
    payload = np.zeros(
        (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_PAYLOAD_WIDTH),
        dtype=np.int8,
    )
    for seat in range(BOARD_SLOTS):
        token_types[0, seat, 0] = TOKEN_TYPE_OCCUPIED
        token_seats[0, seat, 0] = seat
        payload[0, seat, 0, 0] = seat + 1
    payload[0, 0, 20] = 63
    mask = token_types != 0

    features = _materialize_token_features(
        token_types,
        token_seats,
        payload,
        mask,
    )

    assert features.shape == (
        1,
        BOARD_SLOTS,
        BOARD_TOKEN_CAPACITY,
        TOKEN_FEATURES,
    )
    for seat in range(BOARD_SLOTS):
        np.testing.assert_array_equal(
            features[0, seat, 0, 4:8],
            np.eye(BOARD_SLOTS, dtype=np.float32)[seat],
        )
    assert np.all(features[0, 0, 20] == 0)


def test_board_local_count_layout_is_exact_and_never_truncates() -> None:
    counts = np.asarray(
        [
            [
                [23, 31, 22, 16],
                [13, 18, 9, 10],
                [21, 23, 17, 20],
                [3, 6, 4, 2],
            ]
        ],
        dtype=np.int64,
    )
    mask, token_types = _layout_from_board_type_counts(counts)
    assert mask.shape == (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY)
    assert int(mask[0, 0].sum()) == BOARD_TOKEN_CAPACITY
    assert int(mask.sum()) == int(counts.sum())
    np.testing.assert_array_equal(
        np.bincount(token_types[0, 0], minlength=5)[1:],
        counts[0, 0],
    )
    assert np.all(token_types[~mask] == 0)


def _cached_graph_source(targets: np.ndarray) -> SimpleNamespace:
    token_offsets = np.full(
        (1, TOKEN_CAPACITY + 1),
        3,
        dtype=np.uint32,
    )
    token_offsets[0, 0] = 0
    token_offsets[0, 1] = 2
    token_offsets[0, 2] = 2
    return SimpleNamespace(
        tensors={
            "graph_record_offsets": np.asarray([0, 3], dtype=np.uint64),
            "graph_token_offsets": token_offsets,
            "graph_targets": np.asarray(targets, dtype=np.uint16),
            "graph_relations": np.asarray([1, 2, 3], dtype=np.uint8),
            "graph_direction_bits": np.asarray([1, 2, 4], dtype=np.uint8),
        }
    )


def test_graph_batch_expands_cached_rust_relations_without_rederivation() -> None:
    token_mask = np.zeros(
        (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.bool_,
    )
    token_mask[0, 0, :3] = True
    cache = object.__new__(R2SparseMlxCache)
    neighbors, edge_mask, relations, directions = cache._materialize_graph(
        _cached_graph_source(np.asarray([1, 2, 0])),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        token_mask,
    )

    assert neighbors.shape == (
        1,
        BOARD_SLOTS,
        BOARD_TOKEN_CAPACITY,
        GRAPH_MAX_DEGREE,
    )
    np.testing.assert_array_equal(neighbors[0, 0, 0, :2], [1, 2])
    assert neighbors[0, 0, 2, 0] == 0
    np.testing.assert_array_equal(relations[0, 0, 0, :2], [1, 2])
    assert relations[0, 0, 2, 0] == 3
    assert edge_mask[0, 0, 0, :2].all()
    np.testing.assert_array_equal(directions[0, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(directions[0, 0, 0, 1], [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(directions[0, 0, 2, 0], [0, 0, 1, 0, 0, 0])


def test_graph_batch_rejects_cached_cross_board_edge() -> None:
    token_mask = np.zeros(
        (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.bool_,
    )
    token_mask[0, 0, :3] = True
    token_mask[0, 1, 0] = True
    cache = object.__new__(R2SparseMlxCache)
    with pytest.raises(R2SparseMlxCacheError, match="crosses boards"):
        cache._materialize_graph(
            _cached_graph_source(
                np.asarray([BOARD_TOKEN_CAPACITY, 2, 0])
            ),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            token_mask,
        )
