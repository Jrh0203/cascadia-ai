from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    expected_rank_gradient,
    expected_rank_objective,
    projected_kkt_violation,
    projected_optimize_expected_rank,
    solve_box_constrained_expected_rank,
)


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    screen = np.asarray([8.0, 2.0, -1.0, 4.0], dtype=np.float64)
    expected_rank = np.asarray([1.0, 2.0, 4.0, np.inf], dtype=np.float64)
    expected_rank_mask = np.asarray([True, True, True, False])
    eligible = np.asarray([True, True, True, True])
    return screen, expected_rank, expected_rank_mask, eligible


def test_exact_box_solver_is_finite_and_satisfies_kkt() -> None:
    screen, ranks, rank_mask, eligible = _problem()
    solution = solve_box_constrained_expected_rank(
        screen,
        ranks,
        rank_mask,
        eligible,
    )
    scores = solution["scores"]
    assert np.all(np.isfinite(scores))
    assert np.max(np.abs(scores - screen)) <= 12.0 + 1e-12
    assert solution["kkt_violation"] <= 1e-10
    assert solution["active_lower"] + solution["active_interior"] + solution[
        "active_upper"
    ] == int(np.sum(eligible))


def test_projected_control_matches_exact_box_solution() -> None:
    screen, ranks, rank_mask, eligible = _problem()
    analytic = solve_box_constrained_expected_rank(
        screen,
        ranks,
        rank_mask,
        eligible,
    )
    projected = projected_optimize_expected_rank(
        screen,
        screen,
        ranks,
        rank_mask,
        eligible,
        tolerance=1e-9,
        maximum_iterations=10_000,
    )
    assert projected["converged"]
    assert projected["kkt_violation"] <= 1e-8
    assert projected["objective"] == pytest.approx(
        analytic["objective"],
        abs=1e-7,
    )


def test_objective_gradient_matches_finite_difference() -> None:
    screen, ranks, rank_mask, eligible = _problem()
    logits = -(ranks[rank_mask] - 1.0) / 16.0
    weights = np.exp(logits - np.max(logits))
    probabilities = np.zeros(len(screen), dtype=np.float64)
    probabilities[rank_mask] = weights / np.sum(weights)
    gradient = expected_rank_gradient(screen, probabilities, eligible)
    epsilon = 1e-5
    for index in range(len(screen)):
        high = screen.copy()
        low = screen.copy()
        high[index] += epsilon
        low[index] -= epsilon
        finite_difference = (
            expected_rank_objective(high, probabilities, eligible)
            - expected_rank_objective(low, probabilities, eligible)
        ) / (2.0 * epsilon)
        assert gradient[index] == pytest.approx(
            finite_difference,
            abs=1e-8,
        )


def test_kkt_violation_rejects_nonoptimal_interior_scores() -> None:
    screen, ranks, rank_mask, eligible = _problem()
    logits = -(ranks[rank_mask] - 1.0) / 16.0
    weights = np.exp(logits - np.max(logits))
    probabilities = np.zeros(len(screen), dtype=np.float64)
    probabilities[rank_mask] = weights / np.sum(weights)
    lower = screen - 12.0
    upper = screen + 12.0
    assert (
        projected_kkt_violation(
            screen,
            probabilities,
            eligible,
            lower,
            upper,
        )
        > 1e-3
    )
