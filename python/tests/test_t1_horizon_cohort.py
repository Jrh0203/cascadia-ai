from __future__ import annotations

import numpy as np
from cascadia_mlx.t1_horizon_cohort import (
    COHORT_WIDTH,
    is_strict_top64,
    strict_top64_positions,
)


def test_strict_top64_positions_is_score_ranked_and_source_ordered() -> None:
    scores = np.arange(80, dtype=np.float32)
    hashes = np.zeros((80, 32), dtype=np.uint8)
    hashes[:, -1] = np.arange(80, dtype=np.uint8)
    positions, ranks = strict_top64_positions(scores, hashes)
    assert np.array_equal(positions, np.arange(16, 80))
    assert is_strict_top64(ranks)
    assert ranks[-1] == 0
    assert ranks[0] == COHORT_WIDTH - 1


def test_strict_top64_ties_use_ascending_action_hash() -> None:
    scores = np.ones(65, dtype=np.float32)
    hashes = np.zeros((65, 32), dtype=np.uint8)
    hashes[:, -1] = np.arange(65, dtype=np.uint8)
    positions, ranks = strict_top64_positions(scores, hashes)
    assert np.array_equal(positions, np.arange(64))
    assert np.array_equal(ranks, np.arange(64))


def test_is_strict_top64_rejects_inserted_or_duplicate_rank() -> None:
    strict = np.arange(COHORT_WIDTH, dtype=np.int64)
    assert is_strict_top64(strict)
    inserted = strict.copy()
    inserted[-1] = 91
    assert not is_strict_top64(inserted)
    duplicate = strict.copy()
    duplicate[-1] = duplicate[-2]
    assert not is_strict_top64(duplicate)
