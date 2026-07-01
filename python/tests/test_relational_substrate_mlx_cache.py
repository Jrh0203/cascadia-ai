from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.relational_substrate_mlx_cache import (
    RELATIONAL_VALUE_WIDTH,
    RelationalSubstrateMlxCacheError,
    _materialize_relational_sequences,
    _r5_minimal_values,
)


def _values() -> np.ndarray:
    return np.arange(RELATIONAL_VALUE_WIDTH, dtype=np.int16)


def test_r5_minimal_projection_matches_the_rust_contract() -> None:
    habitat = _r5_minimal_values(1, _values())
    np.testing.assert_array_equal(habitat[:42], np.arange(42))
    assert not habitat[42:].any()

    bear = _r5_minimal_values(2, _values())
    elk = _r5_minimal_values(3, _values())
    np.testing.assert_array_equal(bear, _values())
    np.testing.assert_array_equal(elk, _values())

    salmon_source = _values()
    salmon_source[0] = 3
    salmon = _r5_minimal_values(4, salmon_source)
    assert salmon[0] == 3
    assert not salmon[1:4].any()
    assert salmon[4] == 4
    assert salmon[5] == 0
    np.testing.assert_array_equal(salmon[6:12], np.arange(6, 12))
    assert not salmon[12:].any()

    hawk = _r5_minimal_values(5, _values())
    assert hawk[2] == 0
    fox = _r5_minimal_values(6, _values())
    assert not fox[3:].any()


def test_r5_projection_rejects_nonminimal_classes_and_bad_salmon_counts() -> None:
    with pytest.raises(RelationalSubstrateMlxCacheError, match="nonminimal"):
        _r5_minimal_values(7, _values())
    invalid = _values()
    invalid[0] = 40
    with pytest.raises(RelationalSubstrateMlxCacheError, match="Salmon"):
        _r5_minimal_values(4, invalid)


def test_relational_materialization_is_board_local_and_never_truncates() -> None:
    classes, values, mask, counts = _materialize_relational_sequences(
        [
            [
                [(1, np.ones(RELATIONAL_VALUE_WIDTH, dtype=np.int16))],
                [
                    (2, np.full(RELATIONAL_VALUE_WIDTH, 2, dtype=np.int16)),
                    (8, np.full(RELATIONAL_VALUE_WIDTH, 8, dtype=np.int16)),
                ],
                [],
                [],
            ]
        ],
        capacity=3,
    )
    assert classes.shape == (1, 4, 3)
    assert values.shape == (1, 4, 3, RELATIONAL_VALUE_WIDTH)
    np.testing.assert_array_equal(counts, [[1, 2, 0, 0]])
    np.testing.assert_array_equal(classes[0, 1, :2], [2, 8])
    assert mask[0, 1, :2].all()
    assert not mask[0, 1, 2]
    assert np.count_nonzero(values[~mask]) == 0

    with pytest.raises(RelationalSubstrateMlxCacheError, match="capacity"):
        _materialize_relational_sequences(
            [
                [
                    [
                        (1, np.zeros(RELATIONAL_VALUE_WIDTH, dtype=np.int16)),
                        (2, np.zeros(RELATIONAL_VALUE_WIDTH, dtype=np.int16)),
                    ],
                    [],
                    [],
                    [],
                ]
            ],
            capacity=1,
        )
