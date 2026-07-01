from __future__ import annotations

import numpy as np
from cascadia_mlx.graded_oracle_frontier_arbitrary_precision import (
    solve_decimal_box_expected_rank,
)


def test_exact_float_rank_conversion_changes_fractional_objective() -> None:
    screen = np.asarray([2.0, 1.0, 0.0, 8.0])
    ranks = np.asarray([1.25, 2.75, 4.5, 0.0])
    mask = np.asarray([True, True, True, False])
    eligible = np.ones(4, dtype=np.bool_)
    integerized = solve_decimal_box_expected_rank(
        screen,
        ranks,
        mask,
        eligible,
    )
    exact = solve_decimal_box_expected_rank(
        screen,
        ranks,
        mask,
        eligible,
        exact_float_ranks=True,
    )
    assert integerized["objective"] != exact["objective"]
    assert integerized["normalization_offset"] != exact[
        "normalization_offset"
    ]
    assert exact["normalization_residual"] < 1e-60
    assert exact["kkt_violation"] < 1e-60
