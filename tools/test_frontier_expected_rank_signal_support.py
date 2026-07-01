from __future__ import annotations

import numpy as np
from frontier_expected_rank_signal_support import (
    _accumulate_group,
    _distribution,
    _SignalAccumulator,
)


def _group(analysis: str) -> _SignalAccumulator:
    accumulator = _SignalAccumulator()
    _accumulate_group(
        accumulator,
        analysis=analysis,
        expected_rank=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        expected_rank_mask=np.asarray([True, True, True, False]),
        target=np.asarray([True, True, False, False]),
        eligible=np.asarray([True, True, True, True]),
        screen=np.asarray([0.0, 1.0, 4.0, 3.0], dtype=np.float32),
        action_hashes=np.arange(64, dtype=np.uint8).reshape(4, 16),
    )
    return accumulator


def test_concentration_audit_tracks_deployed_target_mass() -> None:
    accumulator = _group("concentration")
    assert accumulator.groups == 1
    assert accumulator.deployed_targets == 2
    assert 0.0 < accumulator.deployed_target_mass[0] < 1.0
    assert accumulator.support_at_mass[0.50][0] >= 1
    assert accumulator.effective_support[0] > 1.0


def test_gradient_audit_exposes_signal_outside_deployed_target() -> None:
    accumulator = _group("gradient")
    assert 0.0 < accumulator.target_gradient_fraction[0] < 1.0
    assert accumulator.outside_target_probability_mass[0] > 0.0


def test_reachability_audit_recovers_target_with_sufficient_range() -> None:
    accumulator = _group("reachability")
    assert accumulator.reachability_exact[0.0] == 0
    assert accumulator.reachability_exact[3.0] == 1
    assert accumulator.required_residual_range == [2.0]


def test_scale_sweep_concentrates_more_mass_at_smaller_scales() -> None:
    accumulator = _group("scale-sweep")
    assert accumulator.scale_target_mass[1.0][0] > accumulator.scale_target_mass[
        128.0
    ][0]
    assert accumulator.scale_entropy_bits[1.0][0] < accumulator.scale_entropy_bits[
        128.0
    ][0]


def test_distribution_is_complete() -> None:
    report = _distribution([1.0, 2.0, 3.0])
    assert report["count"] == 3
    assert report["min"] == 1.0
    assert report["median"] == 2.0
    assert report["max"] == 3.0
