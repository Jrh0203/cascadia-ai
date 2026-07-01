from __future__ import annotations

import numpy as np
from cascadia_mlx.graded_oracle_frontier_rank_boundary_train import (
    rank_matched_boundary_loss_from_scores,
)
from frontier_boundary_support import optimize_scores


def test_rank_score_space_optimizer_recovers_boundary() -> None:
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
        loss_function=lambda scores, targets, candidates: (
            rank_matched_boundary_loss_from_scores(
                scores,
                targets,
                candidates,
                maximum_pairs=2,
            )
        ),
    )
    assert float(np.min(optimized[target])) > float(
        np.max(optimized[eligible & ~target])
    )
