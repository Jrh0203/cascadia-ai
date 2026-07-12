"""Group-sequential stopping boundaries via Lan-DeMets alpha spending.

A fixed-N paired gate spends its entire two-sided alpha at one look. A
group-sequential gate takes interim looks at preplanned information
fractions and may stop as soon as the running boundary is crossed; the
Lan-DeMets spending construction guarantees the overall type-I error
stays at alpha no matter how many looks actually run. The
O'Brien-Fleming-like spending function is deliberately stingy early —
an interim stop requires an overwhelming effect — so the final-look
boundary stays close to the fixed-N critical value (~2.02 vs 1.96 for
the campaign's 40/60/80/100 schedule).

Dependency-free (no scipy), matching `torch_benchmark_stats`:
boundaries are solved by bisection over a recursive trapezoid
integration of the sequential statistic's continuation sub-density
(Armitage-McPherson-Rowe recursion on the Brownian score scale).

Analytic anchor used by the tests: with the O'Brien-Fleming-like
spending function the FIRST look's boundary is exactly
`z_{alpha/2} / sqrt(t_1)` — the recursion must reproduce it.
"""

from __future__ import annotations

import math
from typing import Callable

# z-scale value treated as "no exit possible at this look": beyond 12
# standard deviations the crossing probability underflows any spending
# increment we could be asked to hit.
NO_EXIT_BOUNDARY = 12.0
_GRID_POINTS = 1201  # odd, so the grid includes 0
_BISECTION_ITERATIONS = 200


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF by bisection (mirrors t_quantile)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    low, high = -13.0, 13.0
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        if normal_cdf(mid) < p:
            low = mid
        else:
            high = mid
        if high - low < 1.0e-12:
            break
    return 0.5 * (low + high)


def obrien_fleming_spending(fraction: float, alpha: float) -> float:
    """Two-sided Lan-DeMets O'Brien-Fleming-like spending function."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("information fraction must be in (0, 1]")
    z = normal_quantile(1.0 - alpha / 2.0)
    return min(alpha, 2.0 * (1.0 - normal_cdf(z / math.sqrt(fraction))))


def pocock_spending(fraction: float, alpha: float) -> float:
    """Two-sided Lan-DeMets Pocock-like spending function."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("information fraction must be in (0, 1]")
    return min(alpha, alpha * math.log(1.0 + (math.e - 1.0) * fraction))


SPENDING_FUNCTIONS: dict[str, Callable[[float, float], float]] = {
    "obrien_fleming": obrien_fleming_spending,
    "pocock": pocock_spending,
}


def boundary_nominal_alpha(z_boundary: float) -> float:
    """Two-sided nominal significance level implied by a z boundary."""
    return 2.0 * (1.0 - normal_cdf(z_boundary))


def _validate_fractions(fractions: list[float]) -> None:
    if not fractions:
        raise ValueError("at least one information fraction is required")
    previous = 0.0
    for fraction in fractions:
        if not previous < fraction <= 1.0:
            raise ValueError(
                "information fractions must be strictly increasing in (0, 1]; "
                f"got {fractions}"
            )
        previous = fraction


def _normal_density(x: float, variance: float) -> float:
    return math.exp(-0.5 * x * x / variance) / math.sqrt(2.0 * math.pi * variance)


def _trapezoid_weights(count: int, step: float) -> list[float]:
    weights = [step] * count
    weights[0] = 0.5 * step
    weights[-1] = 0.5 * step
    return weights


def sequential_boundaries(
    fractions: list[float],
    alpha: float = 0.05,
    spending: str = "obrien_fleming",
) -> list[float]:
    """Symmetric two-sided z-scale boundaries for looks at `fractions`.

    The final look always spends the full remaining alpha (standard
    practice: if the realized final information differs slightly from
    the plan, the design still exhausts exactly `alpha`). Returns one
    boundary per look; `NO_EXIT_BOUNDARY` marks a look whose spending
    increment is too small to allow any stop.
    """
    _validate_fractions(fractions)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    spend = SPENDING_FUNCTIONS.get(spending)
    if spend is None:
        raise ValueError(f"unknown spending function: {spending!r}")

    look_count = len(fractions)
    cumulative_targets = [spend(fraction, alpha) for fraction in fractions]
    cumulative_targets[-1] = alpha  # exhaust alpha at the final look

    boundaries: list[float] = []
    grid: list[float] = []
    density: list[float] = []
    spent = 0.0
    for index, fraction in enumerate(fractions):
        increment_target = cumulative_targets[index] - spent
        if index == 0:
            if increment_target <= 0.0:
                boundary = NO_EXIT_BOUNDARY
            else:
                boundary = normal_quantile(1.0 - increment_target / 2.0)
            half_width = boundary * math.sqrt(fraction)
            step = 2.0 * half_width / (_GRID_POINTS - 1)
            grid = [-half_width + i * step for i in range(_GRID_POINTS)]
            density = [_normal_density(point, fraction) for point in grid]
        else:
            increment_variance = fraction - fractions[index - 1]
            weights = _trapezoid_weights(len(grid), grid[1] - grid[0])
            sqrt_variance = math.sqrt(increment_variance)

            def exit_probability(candidate: float) -> float:
                bound = candidate * math.sqrt(fraction)
                total = 0.0
                for weight, point, mass in zip(weights, grid, density, strict=True):
                    lower_tail = normal_cdf((-bound - point) / sqrt_variance)
                    upper_tail = 1.0 - normal_cdf((bound - point) / sqrt_variance)
                    total += weight * mass * (lower_tail + upper_tail)
                return total

            if increment_target <= 0.0 or exit_probability(NO_EXIT_BOUNDARY) >= increment_target:
                boundary = NO_EXIT_BOUNDARY
            else:
                low, high = 0.0, NO_EXIT_BOUNDARY
                for _ in range(_BISECTION_ITERATIONS):
                    mid = 0.5 * (low + high)
                    if exit_probability(mid) > increment_target:
                        low = mid
                    else:
                        high = mid
                    if high - low < 1.0e-10:
                        break
                boundary = 0.5 * (low + high)

            half_width = boundary * math.sqrt(fraction)
            step = 2.0 * half_width / (_GRID_POINTS - 1)
            new_grid = [-half_width + i * step for i in range(_GRID_POINTS)]
            new_density = []
            for target in new_grid:
                total = 0.0
                for weight, point, mass in zip(weights, grid, density, strict=True):
                    total += weight * mass * _normal_density(target - point, increment_variance)
                new_density.append(total)
            grid = new_grid
            density = new_density

        boundaries.append(boundary)
        spent = max(spent, cumulative_targets[index])
    return boundaries
