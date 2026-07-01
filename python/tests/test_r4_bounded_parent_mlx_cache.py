from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from cascadia_mlx.r4_bounded_parent_mlx_cache import (
    CONTROL_ARM,
    UNIVERSAL_PARENT_VALUE_WIDTH,
    R4BoundedParentMlxCache,
    R4BoundedParentMlxCacheError,
    _materialize_parent_sequences,
)


def test_universal_parent_materialization_is_board_local_and_zero_padded() -> None:
    classes, values, mask, counts = _materialize_parent_sequences(
        [
            [
                [(1, np.asarray([2, -3], dtype=np.int16)), (5, np.asarray([7]))],
                [(2, np.asarray([4]))],
                [],
                [(9, np.asarray([11, 12, 13]))],
            ]
        ]
    )
    assert classes.shape == (1, 4, 2)
    assert values.shape == (1, 4, 2, UNIVERSAL_PARENT_VALUE_WIDTH)
    np.testing.assert_array_equal(counts, [[2, 1, 0, 1]])
    np.testing.assert_array_equal(classes[0, 0], [1, 5])
    np.testing.assert_array_equal(values[0, 0, 0, :3], [2, -3, 0])
    assert not mask[0, 2].any()
    assert np.count_nonzero(values[~mask]) == 0


def test_universal_parent_materialization_honors_fixed_capacity() -> None:
    classes, values, mask, counts = _materialize_parent_sequences(
        [[[(1, np.asarray([3], dtype=np.int16))], [], [], []]],
        capacity=7,
    )
    assert classes.shape == (1, 4, 7)
    assert values.shape == (1, 4, 7, UNIVERSAL_PARENT_VALUE_WIDTH)
    np.testing.assert_array_equal(counts, [[1, 0, 0, 0]])
    assert mask[0, 0, 0]
    assert not mask[0, 0, 1:].any()


def test_universal_parent_materialization_rejects_noncanonical_classes() -> None:
    with pytest.raises(R4BoundedParentMlxCacheError, match="noncanonical"):
        _materialize_parent_sequences(
            [
                [
                    [(5, np.asarray([1])), (2, np.asarray([2]))],
                    [],
                    [],
                    [],
                ]
            ]
        )


def test_universal_parent_materialization_rejects_oversize_payload() -> None:
    with pytest.raises(R4BoundedParentMlxCacheError, match="universal width"):
        _materialize_parent_sequences(
            [
                [
                    [
                        (
                            1,
                            np.zeros(
                                UNIVERSAL_PARENT_VALUE_WIDTH + 1,
                                dtype=np.int16,
                            ),
                        )
                    ],
                    [],
                    [],
                    [],
                ]
            ]
        )


def test_control_parent_statistics_zero_extend_to_nine_classes() -> None:
    cache = object.__new__(R4BoundedParentMlxCache)
    cache.splits = {"train": SimpleNamespace(groups=2)}
    cache.parent_capacities = {CONTROL_ARM: 4}
    cache.r3_cache = SimpleNamespace(
        splits={
            "train": SimpleNamespace(
                tensors={
                    "parent_board_type_counts": np.ones(
                        (2, 4, 4),
                        dtype=np.uint16,
                    )
                }
            )
        }
    )
    statistics = cache.parent_token_statistics("train", CONTROL_ARM)
    assert list(statistics["class_tokens"].values()) == [
        8,
        8,
        8,
        8,
        0,
        0,
        0,
        0,
        0,
    ]
