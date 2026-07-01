from __future__ import annotations

import numpy as np
from frontier_boundary_support import optimize_scores


def test_score_space_optimizer_recovers_clean_boundary_within_limit() -> None:
    screen = np.array([[2.0, 1.0, 3.0, 0.0]], dtype=np.float32)
    target = np.array([[True, True, False, False]])
    eligible = np.ones((1, 4), dtype=np.bool_)
    optimized = optimize_scores(
        screen,
        screen,
        target,
        eligible,
        steps=100,
        learning_rate=0.1,
        residual_limit=12.0,
    )
    assert float(np.min(optimized[target])) > float(
        np.max(optimized[eligible & ~target])
    )
    assert float(np.max(np.abs(optimized - screen))) <= 12.0


def test_score_space_optimizer_rejects_invalid_parameters() -> None:
    scores = np.zeros((1, 2), dtype=np.float32)
    target = np.array([[True, False]])
    eligible = np.ones((1, 2), dtype=np.bool_)
    for kwargs in (
        {"steps": 0},
        {"learning_rate": 0.0},
        {"residual_limit": 0.0},
    ):
        try:
            optimize_scores(scores, scores, target, eligible, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid score-space parameters were accepted")
