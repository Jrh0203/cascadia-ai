from __future__ import annotations

from pathlib import Path

import numpy as np
from cascadia_mlx.graded_oracle_target_sufficiency import (
    AuditAccumulator,
    aggregate_reports,
    analyze_decision,
    classify_interpretation,
    interpretation_gates,
)

ROOT = Path(__file__).resolve().parents[2]


def _hashes(count: int) -> np.ndarray:
    values = np.zeros((count, 32), dtype=np.uint8)
    for index in range(count):
        values[index, -1] = index
    return values


def test_r1200_cohort_oracle_prioritizes_labeled_actions() -> None:
    count = 70
    r1200_mask = np.zeros(count, dtype=bool)
    r1200_mask[:64] = True
    r4800_mask = np.zeros(count, dtype=bool)
    r4800_mask[:3] = True
    r600_mask = r1200_mask.copy()
    r1200_mean = np.zeros(count)
    r1200_mean[:64] = np.linspace(100.0, 37.0, 64)
    r4800_mean = np.zeros(count)
    r4800_mean[:3] = [100.0, 99.95, 95.0]
    screen = np.linspace(200.0, 131.0, count)
    model = screen.copy()

    observation = analyze_decision(
        model_scores=model,
        screen_scores=screen,
        action_hashes=_hashes(count),
        selected_index=0,
        r600_mask=r600_mask,
        r1200_mean=r1200_mean,
        r1200_stddev=np.ones(count),
        r1200_samples=np.where(r1200_mask, 1200.0, 0.0),
        r1200_mask=r1200_mask,
        r4800_mean=r4800_mean,
        r4800_stddev=np.ones(count),
        r4800_samples=np.where(r4800_mask, 4800.0, 0.0),
        r4800_mask=r4800_mask,
        phase=1,
        nature_token_available=True,
        independent_draft_winner=False,
        raw_seed=1,
    )

    oracle = observation.rankings["r1200_cohort_oracle"][64]
    assert oracle.exact_winner_recalled
    assert oracle.confidence_set_covered
    assert oracle.retained_regret == 0.0
    assert observation.cohort_has_top64_capacity
    assert observation.confidence_sets_intersect
    assert observation.model_top64_label_counts["screen_only"] == 0

    accumulator = AuditAccumulator()
    accumulator.add(observation)
    report = accumulator.report()
    assert report["model_top64_slots"] == 64
    assert report["model_top64_label_composition"]["r1200"]["fraction"] == 1.0


def test_target_sufficiency_gates_require_every_frozen_condition() -> None:
    oracle = {
        "confidence_set_coverage_95": 0.995,
        "distinguishable_winner_recall": 0.99,
        "exact_winner_recall": 0.96,
        "mean_retained_r4800_regret": 0.01,
    }
    overall = {
        "ranking": {"r1200_cohort_oracle": {"top64": oracle}},
        "confidence_set_intersection_fraction": 0.97,
        "cohort_top64_capacity_fraction": 1.0,
    }
    phases = {
        name: {"ranking": {"r1200_cohort_oracle": {"top64": {"confidence_set_coverage_95": 0.99}}}}
        for name in ("early", "middle", "late")
    }
    integrity = {
        "split_allowed": True,
        "groups_seen": 3,
        "expected_groups": 3,
        "all_groups_seen_once": True,
        "candidates_seen": 100,
        "expected_candidates": 100,
        "all_candidates_seen_once": True,
        "nonfinite_model_scores": 0,
        "all_model_scores_finite": True,
        "nonfinite_teacher_values": 0,
        "all_teacher_values_finite": True,
        "checkpoint_identity_passed": True,
        "model_identity_passed": True,
        "dataset_identity_passed": True,
        "source_identity_passed": True,
        "test_split_opened": False,
    }

    gates = interpretation_gates(overall, phases, integrity)
    assert gates["target_sufficient_for_set_valued_proposer"]
    assert classify_interpretation(gates) == "target_sufficient_for_set_valued_proposer"

    overall["confidence_set_intersection_fraction"] = 0.94
    gates = interpretation_gates(overall, phases, integrity)
    assert not gates["target_sufficient_for_set_valued_proposer"]
    assert classify_interpretation(gates) == "target_insufficient_for_set_valued_proposer"


def test_frozen_artifacts_reject_r1200_only_proposer() -> None:
    root = ROOT / "artifacts/experiments/complete-action-r1200-target-sufficiency-v1"
    report = aggregate_reports(
        root / "train-john1.json",
        [
            root / "validation-john1.json",
            root / "validation-john2.json",
            root / "validation-john3.json",
        ],
    )
    validation = report["validation"]
    oracle = validation["overall"]["ranking"]["r1200_cohort_oracle"]["top64"]
    assert report["passed"]
    assert report["classification"] == "target_insufficient_for_set_valued_proposer"
    assert report["cross_host_validation"]["scientific_metrics_identical"]
    assert oracle["confidence_set_coverage_95"] == 0.9708333333333333
    assert oracle["distinguishable_winner_recall"] == 0.9078947368421053
    assert not validation["interpretation_gates"]["target_sufficient_for_set_valued_proposer"]
    assert not report["sealed_test"]["opened"]
