from __future__ import annotations

from dataclasses import replace

import cascadia_mlx.frontier_supervision_identifiability as audit
import numpy as np
import pytest
from cascadia_mlx.frontier_supervision_identifiability import (
    SupervisionGroup,
    audit_boundary,
    audit_cross_fidelity,
    audit_expected_rank,
    audit_resampling,
    expected_normal_ranks,
    frontier_and_target,
    normal_cdf,
    stable_ranking,
)


class FakeDataset:
    split = "validation"


def make_group(group_id: int = 7, phase: int = 0) -> SupervisionGroup:
    count = 70
    source_flags = np.zeros(count, dtype=np.uint16)
    source_flags[0] = audit.GRADED_SOURCE_CHAMPION_FRONTIER
    action_hash = np.zeros((count, 32), dtype=np.uint8)
    action_hash[:, -1] = np.arange(count, dtype=np.uint8)
    r1200_mean = np.linspace(100.0, 31.0, count)
    r600_mean = r1200_mean.copy()
    samples = np.full(count, 1200.0)
    stddev = np.full(count, 0.01)
    r4800_mean = np.zeros(count)
    r4800_mean[0] = 10.0
    r4800_mean[1] = 5.0
    r4800_samples = np.zeros(count)
    r4800_samples[:2] = 4800.0
    r4800_stddev = np.zeros(count)
    r4800_stddev[:2] = 0.01
    return SupervisionGroup(
        group_id=group_id,
        phase=phase,
        selected_index=0,
        source_flags=source_flags,
        action_hash=action_hash,
        r600_mean=r600_mean,
        r600_stddev=stddev.copy(),
        r600_samples=samples.copy(),
        r1200_mean=r1200_mean,
        r1200_stddev=stddev.copy(),
        r1200_samples=samples.copy(),
        r4800_mean=r4800_mean,
        r4800_stddev=r4800_stddev,
        r4800_samples=r4800_samples,
    )


def test_stable_ranking_uses_hash_for_score_ties() -> None:
    hashes = np.zeros((3, 32), dtype=np.uint8)
    hashes[:, -1] = [2, 0, 1]
    ranking = stable_ranking(
        np.ones(3),
        hashes,
        np.arange(3, dtype=np.int32),
    )
    assert ranking.tolist() == [1, 2, 0]


def test_frontier_target_fills_exact_width() -> None:
    group = make_group()
    frontier, target, excluded = frontier_and_target(group, "r1200")
    assert frontier.tolist() == [0]
    assert len(frontier) + len(target) == 64
    assert target[0] == 1
    assert target[-1] == 63
    assert excluded.tolist() == [64, 65, 66, 67, 68, 69]


def test_normal_cdf_matches_reference_points() -> None:
    values = normal_cdf(np.array([-1.0, 0.0, 1.0]))
    assert values == pytest.approx(
        [0.158655254, 0.5, 0.841344746],
        abs=1e-7,
    )


def test_expected_rank_is_symmetric_for_equal_actions() -> None:
    ranks = expected_normal_ranks(
        np.zeros(4),
        np.ones(4),
    )
    assert ranks == pytest.approx(np.full(4, 2.5), abs=1e-7)


def test_all_four_audits_pass_on_clear_synthetic_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = [make_group(10 + phase, phase) for phase in range(3)]
    monkeypatch.setattr(audit, "iter_supervision_groups", lambda _dataset: iter(groups))

    boundary = audit_boundary(FakeDataset())
    assert boundary["gate_passed"]
    assert boundary["robust_complete_set_fraction_95"] == 1.0

    fidelity = audit_cross_fidelity(FakeDataset())
    assert fidelity["gate_passed"]
    assert fidelity["r600_r1200_exact_set_fraction"] == 1.0

    resampling = audit_resampling(FakeDataset())
    assert resampling["gate_passed"]
    assert resampling["exact_set_reproduction_fraction"] == 1.0

    ceiling = audit_expected_rank(FakeDataset())
    assert ceiling["gate_passed"]
    assert ceiling["overall"]["top64_r4800_winner_recall"] == 1.0


def test_cross_fidelity_fails_when_r600_cannot_fill_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = make_group()
    sparse_samples = group.r600_samples.copy()
    sparse_samples[10:] = 0
    group = replace(group, r600_samples=sparse_samples)
    monkeypatch.setattr(
        audit,
        "iter_supervision_groups",
        lambda _dataset: iter([group]),
    )
    report = audit_cross_fidelity(FakeDataset())
    assert report["comparable_groups"] == 0
    assert report["r600_cohort_coverage_fraction"] == 0.0
    assert not report["gate_passed"]
