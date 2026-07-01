from __future__ import annotations

import numpy as np
from cascadia_mlx.conditional_tile_local_geometry_dropout import (
    _MIX_A,
    _MIX_B,
    _MIX_C,
    DROPOUT_RATE,
    DROPOUT_SEED,
    LOCAL_LEFT,
    LOCAL_RIGHT,
    _mix_u64,
    corrupt_query_local_geometry,
    dropout_count,
    frozen_config,
    selected_item_indices,
)
from cascadia_mlx.conditional_tile_optimizer_schedule import (
    frozen_config as source_config,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    STAGE_ITEM_DIMS,
)


def _fixture(width: int = 9) -> tuple[np.ndarray, np.ndarray]:
    items = np.arange(
        width * STAGE_ITEM_DIMS["tile"],
        dtype=np.float32,
    ).reshape(width, STAGE_ITEM_DIMS["tile"])
    hashes = np.arange(width * 16, dtype=np.uint8).reshape(width, 16)
    return items, hashes


def test_treatment_preserves_source_training_contract() -> None:
    assert frozen_config() == source_config()
    assert DROPOUT_RATE == 0.5


def test_dropout_count_is_exact_half_with_two_item_floor() -> None:
    assert dropout_count(1) == 1
    assert dropout_count(2) == 2
    assert dropout_count(3) == 2
    assert dropout_count(8) == 4
    assert dropout_count(9) == 5


def test_selection_is_deterministic_and_epoch_varying() -> None:
    _items, hashes = _fixture()
    first = selected_item_indices(
        hashes,
        epoch=1,
        shard_index=0,
        query_index=0,
    )
    replay = selected_item_indices(
        hashes,
        epoch=1,
        shard_index=0,
        query_index=0,
    )
    later = selected_item_indices(
        hashes,
        epoch=2,
        shard_index=0,
        query_index=0,
    )
    np.testing.assert_array_equal(first, replay)
    assert len(first) == dropout_count(len(hashes))
    assert not np.array_equal(first, later)


def test_partition_selection_matches_frozen_full_sort() -> None:
    for width in (2, 3, 8, 9, 64, 127):
        _items, hashes = _fixture(width)
        for epoch in (1, 20, 21, 100, 200):
            prefix = np.ascontiguousarray(hashes[:, :8]).view("<u8").reshape(-1)
            salt_input = np.uint64(DROPOUT_SEED)
            with np.errstate(over="ignore"):
                salt_input ^= np.uint64(epoch) * _MIX_A
                salt_input ^= np.uint64(3) * _MIX_B
                salt_input ^= np.uint64(5) * _MIX_C
            salt = _mix_u64(salt_input).reshape(())
            keys = _mix_u64(prefix ^ salt)
            positions = np.arange(width, dtype=np.int64)
            reference = np.lexsort((positions, keys))[: dropout_count(width)]
            selected = selected_item_indices(
                hashes,
                epoch=epoch,
                shard_index=2,
                query_index=4,
            )
            np.testing.assert_array_equal(selected, reference)


def test_corruption_changes_only_selected_local_geometry() -> None:
    items, hashes = _fixture()
    changed, selected = corrupt_query_local_geometry(
        items,
        hashes,
        epoch=7,
        shard_index=2,
        query_index=3,
    )
    np.testing.assert_array_equal(changed[:, :LOCAL_LEFT], items[:, :LOCAL_LEFT])
    np.testing.assert_array_equal(
        changed[:, LOCAL_RIGHT:],
        items[:, LOCAL_RIGHT:],
    )
    unselected = np.setdiff1d(np.arange(len(items)), selected)
    np.testing.assert_array_equal(
        changed[unselected, LOCAL_LEFT:LOCAL_RIGHT],
        items[unselected, LOCAL_LEFT:LOCAL_RIGHT],
    )
    np.testing.assert_array_equal(
        changed[selected, LOCAL_LEFT:LOCAL_RIGHT],
        np.roll(
            items[selected, LOCAL_LEFT:LOCAL_RIGHT],
            shift=1,
            axis=0,
        ),
    )
