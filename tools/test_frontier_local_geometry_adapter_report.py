from __future__ import annotations

from frontier_local_geometry_adapter_report import (
    aggregate_evaluation_reports,
    validate_scheduler_state,
)


def test_evaluation_aggregation_weights_groups() -> None:
    aggregate = aggregate_evaluation_reports(
        [
            {
                "groups": 1,
                "candidates": 100,
                "target_slots": 8,
                "target_hits": 8,
                "target_positive_recall": 1.0,
                "target_set_exact_fraction": 1.0,
                "r4800_winner_retention": 1.0,
                "mean_objective": 2.0,
                "all_scores_finite": True,
            },
            {
                "groups": 1,
                "candidates": 200,
                "target_slots": 12,
                "target_hits": 6,
                "target_positive_recall": 0.5,
                "target_set_exact_fraction": 0.0,
                "r4800_winner_retention": 0.0,
                "mean_objective": 4.0,
                "all_scores_finite": True,
            },
        ]
    )
    assert aggregate["target_positive_recall"] == 0.7
    assert aggregate["target_set_exact_fraction"] == 0.5
    assert aggregate["mean_objective"] == 3.0


def test_scheduler_requires_distinct_cross_host_replays() -> None:
    tasks = {}
    for index in range(4):
        tasks[f"origin-{index:02d}"] = {
            "status": "done",
            "host": f"john{index + 1}",
        }
        tasks[f"replay-{index:02d}"] = {
            "status": "done",
            "host": f"john{(index + 1) % 4 + 1}",
        }
    state = {
        "experiment_id": (
            "complete-action-frontier-calibrated-local-geometry-adapter-v1"
        ),
        "tasks": tasks,
    }
    assert validate_scheduler_state(state) is state
