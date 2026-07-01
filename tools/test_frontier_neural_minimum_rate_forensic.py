from __future__ import annotations

import math

from frontier_neural_minimum_rate_forensic import enumerate_rate_paths


def test_group_2_summary_has_six_paths_with_one_failure_rate() -> None:
    paths = enumerate_rate_paths(
        accepted_updates=8,
        total_backtracks=13,
        maximum_backtracks=4,
        maximum_rate=0.006333668612083666,
        minimum_rate=0.00002474089301595182,
        mean_rate=0.0009154130415902173,
    )
    assert len(paths) == 6
    starts = {path["failed_step_starting_rate"] for path in paths}
    smallest = {
        path["smallest_failed_step_attempted_rate"]
        for path in paths
    }
    assert len(starts) == 1
    assert len(smallest) == 1
    assert math.isclose(
        next(iter(starts)),
        0.00009896357206380728,
    )
    assert next(iter(smallest)) < 1e-8
