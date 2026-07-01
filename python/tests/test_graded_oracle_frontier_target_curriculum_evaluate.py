from __future__ import annotations

from cascadia_mlx.graded_oracle_frontier_target_curriculum_evaluate import (
    target_curriculum_gates,
)


def test_target_curriculum_gates_keep_threshold_boundaries() -> None:
    report = {
        "train": {
            "target_positive_recall": 0.60,
            "target_set_exact_fraction": 0.05,
            "all_groups_scored_once": True,
            "all_candidates_scored_once": True,
            "all_scores_finite": True,
        },
        "validation": {
            "target_positive_recall": 0.50,
            "target_set_exact_fraction": 0.01,
            "top64_r4800_winner_recall": 0.75,
            "top64_confidence_set_coverage_95": 0.90,
            "mean_top64_retained_r4800_regret": 0.149,
            "all_groups_scored_once": True,
            "all_candidates_scored_once": True,
            "all_scores_finite": True,
        },
        "test_split_opened": False,
    }
    assert target_curriculum_gates(report)["pilot_passed"]
    report["validation"]["target_positive_recall"] = 0.499
    assert not target_curriculum_gates(report)["pilot_passed"]
