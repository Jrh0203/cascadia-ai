from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.full_legal_hierarchical_factor_oracle import (
    FactorRows,
    factor_rows,
    oracle_candidate_indices,
    run_arm,
)


def test_factor_partition_excludes_only_consequences() -> None:
    actions = np.arange(280, dtype=np.float32).reshape(2, 140)
    rows = factor_rows(actions)
    assert len(rows.draft[0]) == (34 + 83) * 4
    assert len(rows.tile[0]) == 8 * 4
    assert len(rows.wildlife[0]) == 3 * 4
    changed = actions.copy()
    changed[:, 128:140] += 1000
    assert factor_rows(changed) == rows


def test_conditional_oracle_retains_best_factor_prefix() -> None:
    factors = FactorRows(
        draft=(b"a", b"a", b"b", b"c"),
        tile=(b"x", b"y", b"x", b"x"),
        wildlife=(b"u", b"v", b"u", b"u"),
    )
    retained = oracle_candidate_indices(
        factors,
        np.asarray([2.0, 1.0, 3.0, 4.0]),
        np.ones(4, dtype=np.bool_),
        np.zeros(4, dtype=np.uint16),
        arm="conditional-compact",
    )
    np.testing.assert_array_equal(retained, np.arange(4))


def test_out_of_range_arm_is_rejected_before_evidence() -> None:
    with pytest.raises(ValueError, match="outside 0-3"):
        run_arm(None, None, None, None, None, 4)  # type: ignore[arg-type]
