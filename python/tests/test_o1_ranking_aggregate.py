from __future__ import annotations

from cascadia_mlx.o1_ranking_aggregate import (
    HIGH_REGRET_IMPROVEMENT,
    PAIRWISE_REGRESSION_TOLERANCE,
    PRIMARY_IMPROVEMENT,
)


def test_frozen_validation_thresholds_match_adr_0188() -> None:
    assert PRIMARY_IMPROVEMENT == 0.05
    assert HIGH_REGRET_IMPROVEMENT == 0.10
    assert PAIRWISE_REGRESSION_TOLERANCE == 0.005
