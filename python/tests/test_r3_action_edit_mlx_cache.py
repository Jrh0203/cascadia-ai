from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM,
    CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
    CONTROL_MATERIALIZATION_VERIFIED,
    R3_LOCAL_PATCH_TOKEN,
    R3_TOKEN_FEATURES,
    R3_TOKEN_PAYLOAD_WIDTH,
    R3ActionEditMlxCacheError,
    R3ActionEditMlxDataset,
    _control_multiset_blake3,
    _materialize_candidate_features,
    _translate_r2_payloads_in_place,
    deterministic_training_rows,
    deterministic_transform_ids,
)


def test_candidate_features_have_exact_one_hots_and_zero_padding() -> None:
    token_types = np.asarray([[[3, 7, 0]]], dtype=np.uint8)
    operations = np.asarray([[[5, 2, 0]]], dtype=np.uint8)
    payload = np.zeros((1, 1, 3, R3_TOKEN_PAYLOAD_WIDTH), dtype=np.int8)
    payload[0, 0, 0, :2] = [32, -16]
    mask = np.asarray([[[True, True, False]]])

    features = _materialize_candidate_features(
        token_types,
        operations,
        payload,
        mask,
    )

    assert features.shape == (1, 1, 3, R3_TOKEN_FEATURES)
    assert features[0, 0, 0, 2] == 1
    assert features[0, 0, 0, 10 + 5] == 1
    np.testing.assert_allclose(features[0, 0, 0, 16:18], [0.5, -0.25])
    assert np.count_nonzero(features[0, 0, 2]) == 0


def test_control_relative_translation_covers_component_members() -> None:
    token_types = np.asarray([1, 2, 3, 4], dtype=np.uint8)
    payload = np.zeros((4, 52), dtype=np.int8)
    payload[0, :2] = [4, 3]
    payload[1, :2] = [5, 4]
    payload[2, 2] = 2
    payload[2, 6:10] = [4, 3, 6, 5]
    payload[3, :2] = [3, 2]

    _translate_r2_payloads_in_place(payload, token_types, (3, 2))

    np.testing.assert_array_equal(payload[0, :2], [1, 1])
    np.testing.assert_array_equal(payload[1, :2], [2, 2])
    np.testing.assert_array_equal(payload[2, 6:10], [1, 1, 3, 3])
    np.testing.assert_array_equal(payload[3, :2], [0, 0])


def test_control_multiset_hash_is_order_invariant_but_payload_sensitive() -> None:
    token_types = np.asarray([4, 1], dtype=np.uint8)
    payload = np.zeros((2, 52), dtype=np.int8)
    payload[0, 0] = -1
    payload[1, 0] = 7

    expected = _control_multiset_blake3(token_types, payload)
    assert expected == _control_multiset_blake3(token_types[::-1], payload[::-1])
    changed = payload.copy()
    changed[0, 1] = 1
    assert expected != _control_multiset_blake3(token_types, changed)


def test_unknown_r2_token_type_fails_closed() -> None:
    with pytest.raises(R3ActionEditMlxCacheError, match="unknown R2 token"):
        _translate_r2_payloads_in_place(
            np.zeros((1, 52), dtype=np.int8),
            np.asarray([9], dtype=np.uint8),
            (0, 0),
        )


def test_patch_token_code_remains_frozen() -> None:
    assert R3_LOCAL_PATCH_TOKEN == 2


def test_training_schedule_is_deterministic_complete_and_slice_alternating() -> None:
    all_rows = np.arange(20, dtype=np.int64)
    low_supply = np.asarray([2, 4, 6], dtype=np.int64)
    independent = np.asarray([1, 3], dtype=np.int64)
    schedule = np.stack(
        [
            deterministic_training_rows(
                step=step,
                seed=2026061708,
                all_rows=all_rows,
                low_supply_rows=low_supply,
                independent_winner_rows=independent,
            )
            for step in range(20)
        ]
    )

    for stream in range(3):
        assert set(schedule[:, stream]) == set(all_rows)
    assert set(schedule[::2, 3]).issubset(set(low_supply))
    assert set(schedule[1::2, 3]).issubset(set(independent))
    np.testing.assert_array_equal(
        schedule[7],
        deterministic_training_rows(
            step=7,
            seed=2026061708,
            all_rows=all_rows,
            low_supply_rows=low_supply,
            independent_winner_rows=independent,
        ),
    )


def test_transform_schedule_is_stable_and_spans_d6() -> None:
    transforms = np.concatenate(
        [
            deterministic_transform_ids(
                step=step,
                seed=2026061708,
            )
            for step in range(100)
        ]
    )
    assert set(transforms) == set(range(12))
    np.testing.assert_array_equal(
        deterministic_transform_ids(step=9, seed=2026061708),
        deterministic_transform_ids(step=9, seed=2026061708),
    )


def test_parent_context_can_skip_exact_r2_token_materialization() -> None:
    dataset = object.__new__(R3ActionEditMlxDataset)
    dataset.source = SimpleNamespace(
        tensors={
            "parent_market_features": np.zeros(
                (2, 4, MARKET_FEATURES),
                dtype=np.float32,
            ),
            "parent_market_mask": np.ones((2, 4), dtype=np.uint8),
            "parent_player_features": np.zeros(
                (2, BOARD_SLOTS, PLAYER_FEATURES),
                dtype=np.float32,
            ),
            "parent_player_mask": np.ones(
                (2, BOARD_SLOTS),
                dtype=np.uint8,
            ),
            "parent_global_features": np.zeros(
                (2, GLOBAL_FEATURES),
                dtype=np.float32,
            ),
        }
    )

    parent = dataset._parent_batch(
        np.asarray([1], dtype=np.int64),
        np.asarray([7], dtype=np.int64),
        include_tokens=False,
    )

    assert parent.token_features.shape == (1, BOARD_SLOTS, 0, TOKEN_FEATURES)
    assert parent.token_types.shape == (1, BOARD_SLOTS, 0)
    assert parent.token_mask.shape == (1, BOARD_SLOTS, 0)
    np.testing.assert_array_equal(parent.transform_ids, [7])


def test_preverified_vectorized_control_matches_verified_materialization() -> None:
    dataset = object.__new__(R3ActionEditMlxDataset)
    dataset.open_data_verification_id = "0" * 64
    parent_payload = np.zeros((1, BOARD_SLOTS, 4, 52), dtype=np.int8)
    parent_payload[0, 0, 0, :2] = [2, -1]
    parent_payload[0, 0, 0, 3:6] = [0, 1, 3]
    parent_payload[0, 0, 1, :3] = [1, 0, 0b001011]
    parent_payload[0, 0, 2, :2] = [-1, 2]
    added_payload = np.zeros((4, 52), dtype=np.int8)
    added_payload[0, :3] = [0, 1, 0b000101]
    added_payload[1, :2] = [1, -1]
    added_payload[2, 2] = 0
    added_payload[3, :2] = [-2, 1]
    dataset.source = SimpleNamespace(
        tensors={
            "candidate_offsets": np.asarray([0, 3], dtype=np.uint64),
            "canonical_transform_ids": np.asarray([0, 1, 1], dtype=np.uint8),
            "transformed_centers": np.asarray(
                [[0, 0], [1, -1], [1, -1]],
                dtype=np.int8,
            ),
            "parent_board_type_counts": np.asarray(
                [
                    [
                        [1, 1, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ],
                dtype=np.uint16,
            ),
            "parent_token_types": np.asarray(
                [
                    [
                        [1, 2, 4, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ],
                dtype=np.uint8,
            ),
            "parent_token_payload": parent_payload,
            "control_remove_offsets": np.asarray(
                [0, 1, 2, 3],
                dtype=np.uint64,
            ),
            "control_remove_indices": np.asarray([1, 0, 2], dtype=np.uint8),
            "control_add_offsets": np.asarray(
                [0, 1, 2, 4],
                dtype=np.uint64,
            ),
            "control_add_types": np.asarray([2, 1, 3, 4], dtype=np.uint8),
            "control_add_payload": added_payload,
            "control_after_hashes": np.zeros((3, 32), dtype=np.uint8),
        }
    )
    rows = np.asarray([0], dtype=np.int64)

    verified = dataset._candidate_batch(
        rows,
        arm=CONTROL_ARM,
        verify_control_hashes=False,
        control_materialization=CONTROL_MATERIALIZATION_VERIFIED,
    )
    vectorized = dataset._candidate_batch(
        rows,
        arm=CONTROL_ARM,
        verify_control_hashes=False,
        control_materialization=CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
    )

    for expected, observed in zip(verified, vectorized, strict=True):
        np.testing.assert_array_equal(observed, expected)

    subset = dataset._candidate_batch(
        rows,
        arm=CONTROL_ARM,
        verify_control_hashes=False,
        control_materialization=CONTROL_MATERIALIZATION_VERIFIED,
        candidate_positions=(np.asarray([0, 2], dtype=np.int64),),
    )
    np.testing.assert_array_equal(subset[1].shape[:2], [1, 2])
    np.testing.assert_array_equal(subset[2], [[3, 4]])
    np.testing.assert_array_equal(subset[3], [[0, 1]])


def test_candidate_position_selection_is_strict_and_aligned() -> None:
    dataset = object.__new__(R3ActionEditMlxDataset)
    dataset.source = SimpleNamespace(
        tensors={"candidate_offsets": np.asarray([0, 5, 9], dtype=np.uint64)}
    )
    rows = np.asarray([0, 1], dtype=np.int64)

    normalized = dataset._normalize_candidate_positions(
        rows,
        (
            np.asarray([0, 2, 4], dtype=np.int64),
            np.asarray([1, 3], dtype=np.int64),
        ),
    )
    np.testing.assert_array_equal(normalized[0], [0, 2, 4])
    np.testing.assert_array_equal(normalized[1], [1, 3])

    with pytest.raises(ValueError, match="strictly increasing"):
        dataset._normalize_candidate_positions(
            rows,
            (
                np.asarray([0, 0], dtype=np.int64),
                np.asarray([1], dtype=np.int64),
            ),
        )
