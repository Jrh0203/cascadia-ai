from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_identifiability import (
    NORMAL_95,
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


def test_ambiguous_winner_and_equivalent_model_retention() -> None:
    observation = analyze_decision(
        model_scores=np.array([0.0, 2.0, 1.0]),
        screen_scores=np.array([3.0, 2.0, 1.0]),
        action_hashes=_hashes(3),
        selected_index=0,
        r1200_mean=np.array([9.9, 10.0, 8.0]),
        r1200_stddev=np.array([1.0, 1.0, 1.0]),
        r1200_samples=np.array([100.0, 100.0, 100.0]),
        r1200_mask=np.array([True, True, True]),
        r4800_mean=np.array([10.0, 9.9, 8.0]),
        r4800_stddev=np.array([2.0, 2.0, 2.0]),
        r4800_samples=np.array([100.0, 100.0, 100.0]),
        r4800_mask=np.array([True, True, True]),
        phase=0,
        nature_token_available=True,
        independent_draft_winner=False,
        raw_seed=1,
    )
    assert not observation.distinguishable_95
    assert observation.confidence_set_size_95 == 2
    assert not observation.r1200_r4800_argmax_agree
    assert observation.r1200_winner_in_r4800_confidence_set_95
    assert not observation.rankings["model"][1].exact_winner_recalled
    assert observation.rankings["model"][1].confidence_set_covered
    assert np.isclose(observation.rankings["model"][1].retained_regret, 0.1)

    audit = AuditAccumulator()
    audit.add(observation)
    report = audit.report()
    assert report["ranking"]["model"]["top1"][
        "misses_retaining_confidence_equivalent_fraction"
    ] == 1.0


def test_clear_winner_is_identified_and_recalled() -> None:
    observation = analyze_decision(
        model_scores=np.array([3.0, 2.0]),
        screen_scores=np.array([3.0, 2.0]),
        action_hashes=_hashes(2),
        selected_index=0,
        r1200_mean=np.array([10.0, 8.0]),
        r1200_stddev=np.array([1.0, 1.0]),
        r1200_samples=np.array([400.0, 400.0]),
        r1200_mask=np.array([True, True]),
        r4800_mean=np.array([10.0, 8.0]),
        r4800_stddev=np.array([1.0, 1.0]),
        r4800_samples=np.array([400.0, 400.0]),
        r4800_mask=np.array([True, True]),
        phase=2,
        nature_token_available=False,
        independent_draft_winner=True,
        raw_seed=2,
    )
    assert observation.top_two_margin > NORMAL_95 * observation.combined_standard_error
    assert observation.distinguishable_95
    assert observation.separated_intervals_95
    assert observation.confidence_set_size_95 == 1
    assert observation.rankings["model"][1].exact_winner_recalled


def test_interpretation_gates_classify_target_ambiguity() -> None:
    top64 = {
        "confidence_set_coverage_95": 0.99,
        "mean_retained_r4800_regret": 0.10,
        "misses_retaining_confidence_equivalent_fraction": 0.97,
        "distinguishable_winner_recall": 0.95,
    }
    overall = {
        "distinguishable_winner_95_fraction": 0.25,
        "confidence_set_size_95": {"mean": 5.0},
        "ranking": {"model": {"top64": top64}},
    }
    phases = {
        name: {
            "ranking": {
                "model": {
                    "top64": {"confidence_set_coverage_95": 0.96}
                }
            }
        }
        for name in ("early", "middle", "late")
    }
    integrity = {
        "split_allowed": True,
        "groups_seen": 3,
        "expected_groups": 3,
        "all_groups_seen_once": True,
        "candidates_seen": 10,
        "expected_candidates": 10,
        "all_candidates_seen_once": True,
        "nonfinite_model_scores": 0,
        "all_model_scores_finite": True,
        "checkpoint_identity_passed": True,
        "model_identity_passed": True,
        "dataset_identity_passed": True,
        "source_identity_passed": True,
        "test_split_opened": False,
    }
    gates = interpretation_gates(overall, phases, integrity)
    assert gates["target_ambiguity_dominant"]
    assert classify_interpretation(gates) == "target_ambiguity_dominant"


def test_frozen_artifacts_classify_representation_as_material() -> None:
    root = (
        ROOT
        / "artifacts/experiments/complete-action-r4800-identifiability-v1"
    )
    report = aggregate_reports(
        root / "train-john1.json",
        [
            root / "validation-john1.json",
            root / "validation-john2.json",
            root / "validation-john3.json",
        ],
    )
    assert report["passed"]
    assert report["classification"] == "representation_or_optimization_material"
    assert report["cross_host_validation"]["scientific_metrics_identical"]
    assert not report["sealed_test"]["opened"]
    assert report["validation"]["overall"]["ranking"]["model"]["top64"][
        "confidence_set_coverage_95"
    ] == pytest.approx(0.8625)


def test_aggregate_requires_three_distinct_cluster_hosts(tmp_path: Path) -> None:
    root = (
        ROOT
        / "artifacts/experiments/complete-action-r4800-identifiability-v1"
    )
    validation_paths = []
    for index, source_name in enumerate(
        ("validation-john1.json", "validation-john2.json", "validation-john3.json")
    ):
        report = json.loads((root / source_name).read_text())
        if index == 2:
            report["host"] = "john2"
        path = tmp_path / source_name
        path.write_text(json.dumps(report))
        validation_paths.append(path)

    with pytest.raises(ValueError, match="all three Macs"):
        aggregate_reports(root / "train-john1.json", validation_paths)
