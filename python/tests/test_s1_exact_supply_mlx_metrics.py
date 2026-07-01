from __future__ import annotations

import numpy as np
from cascadia_mlx.s1_exact_supply_mlx_metrics import (
    _calibration,
    _confidence_set,
    _retained_regret,
    _stable_ranking,
)


def test_stable_ranking_breaks_score_ties_by_action_hash() -> None:
    scores = np.asarray([1.0, 2.0, 2.0])
    hashes = np.zeros((3, 32), dtype=np.uint8)
    hashes[1, 0] = 2
    hashes[2, 0] = 1
    np.testing.assert_array_equal(_stable_ranking(scores, hashes), [2, 1, 0])


def test_retained_regret_confidence_and_calibration_are_factual() -> None:
    teacher = np.asarray([10.0, 9.0, 7.0])
    mask = np.asarray([True, True, True])
    assert _retained_regret(np.asarray([1, 2]), teacher, mask) == 1.0
    confidence = _confidence_set(
        teacher,
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([100.0, 100.0, 100.0]),
        mask,
        0,
    )
    assert confidence.tolist() == [True, True, False]
    calibration = _calibration(teacher, teacher)
    assert np.isclose(calibration["calibration_slope"], 1.0)
    assert np.isclose(calibration["calibration_intercept"], 0.0)
