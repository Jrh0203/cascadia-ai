from __future__ import annotations

import numpy as np
from frontier_hierarchical_tile_complementarity_audit import _Accumulator


def test_accumulator_reports_unique_complementary_target_hits() -> None:
    accumulator = _Accumulator()
    target = np.asarray([True, True, False, True], dtype=np.bool_)
    accumulator.add(target, {0, 2}, {1, 2})
    report = accumulator.report()
    assert report["learned_target_hits"] == 1
    assert report["prior_target_hits"] == 1
    assert report["union_target_hits"] == 2
    assert report["shared_target_hits"] == 0
    assert report["learned_only_target_hits"] == 1
    assert report["prior_only_target_hits"] == 1
    assert report["mean_selected_overlap_fraction"] == 0.5
