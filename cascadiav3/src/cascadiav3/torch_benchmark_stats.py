"""Paired-benchmark statistics helpers.

Twenty-game means with sub-point deltas are ~1-sigma reads; promotion gates
must report and act on confidence intervals, not point deltas. This module is
dependency-free (no scipy): the Student-t quantile is computed by bisection on
the CDF via the regularized incomplete beta function.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iterations = 200
    epsilon = 3.0e-12
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    x = df / (df + t * t)
    probability = 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - probability if t > 0 else probability


def t_quantile(p: float, df: float) -> float:
    """Inverse Student-t CDF by bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    low, high = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
        if high - low < 1.0e-10:
            break
    return 0.5 * (low + high)


def paired_delta_stats(
    deltas: list[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean, SE, t-CI, and bootstrap percentile CI for paired per-seed deltas."""
    n = len(deltas)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "se": None,
            "confidence": confidence,
            "t_ci_low": None,
            "t_ci_high": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "ci_excludes_zero": None,
        }
    mean_value = sum(deltas) / n
    if n == 1:
        return {
            "n": 1,
            "mean": mean_value,
            "sd": 0.0,
            "se": 0.0,
            "confidence": confidence,
            "t_ci_low": None,
            "t_ci_high": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "ci_excludes_zero": None,
        }
    variance = sum((value - mean_value) ** 2 for value in deltas) / (n - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n)
    alpha = 1.0 - confidence
    t_critical = t_quantile(1.0 - alpha / 2.0, n - 1)
    t_ci_low = mean_value - t_critical * se
    t_ci_high = mean_value + t_critical * se

    rng = random.Random(seed)
    boot_means = []
    for _ in range(bootstrap_samples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(resample) / n)
    boot_means.sort()

    def _percentile(values: list[float], q: float) -> float:
        position = (len(values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        fraction = position - lower
        return values[lower] * (1.0 - fraction) + values[upper] * fraction

    bootstrap_ci_low = _percentile(boot_means, alpha / 2.0)
    bootstrap_ci_high = _percentile(boot_means, 1.0 - alpha / 2.0)
    return {
        "n": n,
        "mean": mean_value,
        "sd": sd,
        "se": se,
        "confidence": confidence,
        "t_ci_low": t_ci_low,
        "t_ci_high": t_ci_high,
        "bootstrap_ci_low": bootstrap_ci_low,
        "bootstrap_ci_high": bootstrap_ci_high,
        "ci_excludes_zero": bool(t_ci_low > 0.0 or t_ci_high < 0.0),
    }
