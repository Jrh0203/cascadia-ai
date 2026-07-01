from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    LOSS_TOLERANCE,
    MAXIMUM_LEARNING_RATE,
    MonotoneAdamW,
    NumericalConvergence,
)


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = mx.array([3.0])


def _loss(model: _ScalarModel, target: mx.array) -> mx.array:
    return mx.mean(mx.square(model.value - target))


def test_calibrated_rate_crosses_full_tanh_range_in_budget() -> None:
    assert np.isclose(
        1200 * MAXIMUM_LEARNING_RATE,
        2 * np.arctanh(0.999),
    )


def test_monotone_adamw_decreases_loss_and_records_rates() -> None:
    model = _ScalarModel()
    optimizer = MonotoneAdamW()
    loss_and_grad = nn.value_and_grad(model, _loss)
    target = mx.array([0.0])
    previous = float(_loss(model, target).item())
    for _ in range(20):
        loss, gradients = loss_and_grad(model, target)
        value = optimizer.step(
            model,
            gradients,
            loss,
            _loss,
            target,
        )
        assert value <= previous + 1e-12
        previous = value
    summary = optimizer.summary()
    assert summary["accepted_updates"] == 20
    assert summary["loss_monotone"]
    assert summary["moments_finite"]


def test_numerical_convergence_requires_complete_finite_backtracking() -> None:
    model = _ScalarModel()
    optimizer = MonotoneAdamW()
    optimizer.accepted_rates.append(MAXIMUM_LEARNING_RATE)
    optimizer.next_rate = MAXIMUM_LEARNING_RATE / 4
    gradients = {"value": mx.ones_like(model.value)}
    target = mx.array([4.0])
    loss = _loss(model, target)
    try:
        optimizer.step(
            model,
            gradients,
            loss,
            _loss,
            target,
            allow_numerical_convergence=True,
        )
    except NumericalConvergence as convergence:
        diagnostics = convergence.diagnostics
    else:
        raise AssertionError("expected numerical convergence")
    assert diagnostics["proposals_attempted"] == 16
    assert diagnostics["all_proposals_finite"]
    assert diagnostics["current_state_finite"]
    assert diagnostics["smallest_attempted_rate"] < 1e-7
    assert (
        diagnostics["maximum_candidate_improvement"]
        <= LOSS_TOLERANCE
    )
    assert diagnostics["prior_accepted_updates"] == 1


def test_numerical_convergence_can_use_only_eligible_rate_domain() -> None:
    class _WellModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = mx.array([0.0])

    def well_loss(model: _WellModel) -> mx.array:
        in_subminimum_well = mx.abs(model.value + 9.65e-9) < 3.0e-9
        at_current = mx.abs(model.value) < 1.0e-12
        return mx.mean(
            mx.where(
                at_current,
                1.0,
                mx.where(in_subminimum_well, 0.999, 1.001),
            )
        )

    model = _WellModel()
    optimizer = MonotoneAdamW()
    optimizer.accepted_rates.append(MAXIMUM_LEARNING_RATE)
    optimizer.next_rate = 1.0e-4
    gradients = {"value": mx.ones_like(model.value)}
    loss = well_loss(model)
    try:
        optimizer.step(
            model,
            gradients,
            loss,
            lambda candidate: well_loss(candidate),
            allow_numerical_convergence=True,
            convergence_improvement_domain="eligible",
        )
    except NumericalConvergence as convergence:
        diagnostics = convergence.diagnostics
    else:
        raise AssertionError("expected eligible-domain convergence")
    assert diagnostics["convergence_improvement_domain"] == "eligible"
    assert diagnostics["maximum_eligible_candidate_improvement"] <= 0.0
    assert diagnostics["maximum_all_candidate_improvement"] > LOSS_TOLERANCE
