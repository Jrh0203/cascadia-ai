from __future__ import annotations

import numpy as np
from cascadia_mlx.imitation_identifiability import AuditAccumulator


def test_identifiability_audit_reports_confidence_and_rank_metrics() -> None:
    audit = AuditAccumulator()
    audit.add(
        means=np.array([100.0, 99.0, 90.0]),
        stddev=np.array([2.0, 2.0, 0.0]),
        samples=np.array([4.0, 4.0, 0.0]),
        scored=np.array([True, True, False]),
        selected=np.array([True, False, False]),
        parent_rank=np.array([2.0, 1.0, 3.0]),
        immediate_rank=np.array([5.0, 1.0, 2.0]),
    )

    report = audit.report()
    assert report["groups"] == 1
    assert report["scored_candidates"] == 2
    assert report["top_two_margin"]["mean"] == 1.0
    assert np.isclose(report["combined_standard_error"]["mean"], np.sqrt(2.0))
    assert report["margin_within_combined_se_fraction"] == 1.0
    assert report["distinguishable_winner_95_fraction"] == 0.0
    assert report["confidence_set_size_68"]["mean"] == 2.0
    assert report["selected_parent_rank"]["top1_fraction"] == 0.0
    assert report["selected_parent_rank"]["top5_fraction"] == 1.0
    assert report["selected_immediate_rank"]["top5_fraction"] == 1.0


def test_identifiability_audit_detects_clear_winner() -> None:
    audit = AuditAccumulator()
    audit.add(
        means=np.array([100.0, 90.0]),
        stddev=np.array([1.0, 1.0]),
        samples=np.array([100.0, 100.0]),
        scored=np.array([True, True]),
        selected=np.array([True, False]),
        parent_rank=np.array([1.0, 2.0]),
        immediate_rank=np.array([1.0, 2.0]),
    )

    report = audit.report()
    assert report["distinguishable_winner_95_fraction"] == 1.0
    assert report["separated_confidence_intervals_95_fraction"] == 1.0
    assert report["confidence_set_size_95"]["mean"] == 1.0
    assert report["selected_parent_rank"]["top1_fraction"] == 1.0
