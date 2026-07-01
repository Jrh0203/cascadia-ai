from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_projected_repair import (
    _optimize_group,
    shard_group_indices,
)


def test_repair_shards_are_disjoint_and_complete() -> None:
    shards = [shard_group_indices(index) for index in range(4)]
    assert shards == [
        tuple(range(0, 6)),
        tuple(range(6, 12)),
        tuple(range(12, 18)),
        tuple(range(18, 24)),
    ]
    assert sorted(index for shard in shards for index in shard) == list(range(24))
    with pytest.raises(ValueError, match="outside"):
        shard_group_indices(4)


def test_repair_worker_converges_on_synthetic_group() -> None:
    count = 4
    payload = {
        "group_index": 0,
        "group_id": 11,
        "phase": 1,
        "winner": 0,
        "selected_scores": np.asarray([8.0, 2.0, -1.0, 4.0]),
        "screen": np.asarray([8.0, 2.0, -1.0, 4.0]),
        "ranks": np.asarray([1.0, 2.0, 4.0, 8.0]),
        "rank_mask": np.ones(count, dtype=np.bool_),
        "eligible": np.ones(count, dtype=np.bool_),
        "flags": np.zeros(count, dtype=np.uint16),
        "hashes": np.arange(count * 16, dtype=np.uint8).reshape(count, 16),
        "target": np.ones(count, dtype=np.bool_),
        "selected_residual": np.zeros(count),
    }
    report = _optimize_group(payload)
    assert report["converged"]
    assert report["kkt_violation"] <= 1e-8
    assert abs(report["objective_gap_from_analytic"]) <= 1e-7
    assert report["selection_matches_analytic"]
