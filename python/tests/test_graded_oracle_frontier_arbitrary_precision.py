from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
)
from cascadia_mlx.graded_oracle_frontier_arbitrary_precision import (
    KKT_GATE,
    NORMALIZATION_GATE,
    decimal_frontier_retained_indices,
    solve_decimal_box_expected_rank,
)


def test_decimal_solver_normalizes_and_satisfies_kkt() -> None:
    screen = np.asarray([8.0, 2.0, -1.0, 4.0])
    ranks = np.asarray([1.0, 2.0, 4.0, 8.0])
    rank_mask = np.ones(4, dtype=np.bool_)
    eligible = np.ones(4, dtype=np.bool_)
    result = solve_decimal_box_expected_rank(
        screen,
        ranks,
        rank_mask,
        eligible,
    )
    assert result["normalization_residual"] <= NORMALIZATION_GATE
    assert result["kkt_violation"] <= KKT_GATE
    assert result["all_values_finite"]
    assert result["active_lower"] + result["active_interior"] + result[
        "active_upper"
    ] == 4


def test_decimal_solver_handles_zero_mass_at_lower_bound() -> None:
    screen = np.asarray([0.0, 0.0, 30.0])
    ranks = np.asarray([1.0, 2.0, 0.0])
    rank_mask = np.asarray([True, True, False])
    eligible = np.ones(3, dtype=np.bool_)
    result = solve_decimal_box_expected_rank(
        screen,
        ranks,
        rank_mask,
        eligible,
    )
    assert result["scores"][2] == Decimal(18)
    assert result["normalization_residual"] <= NORMALIZATION_GATE
    assert result["kkt_violation"] <= KKT_GATE


def test_decimal_selector_anchors_frontier_and_breaks_ties_by_hash() -> None:
    scores = [Decimal(0), Decimal(10), Decimal(10), Decimal(9)]
    flags = np.asarray(
        [GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0, 0],
        dtype=np.uint16,
    )
    hashes = np.asarray(
        [
            [9] * 16,
            [2] * 16,
            [1] * 16,
            [0] * 16,
        ],
        dtype=np.uint8,
    )
    retained = decimal_frontier_retained_indices(
        scores=scores,
        source_flags=flags,
        action_hashes=hashes,
        width=3,
    )
    assert retained == [0, 2, 1]


def test_decimal_solver_rejects_target_mass_on_anchor() -> None:
    with pytest.raises(ValueError, match="ineligible"):
        solve_decimal_box_expected_rank(
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 2.0]),
            np.asarray([True, True]),
            np.asarray([False, True]),
        )
