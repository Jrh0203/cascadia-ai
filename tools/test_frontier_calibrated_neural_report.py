from __future__ import annotations

from frontier_calibrated_neural_report import (
    aggregate_evaluation_reports,
    validate_scheduler_state,
)


def test_neural_scheduler_state_has_four_cross_host_pairs() -> None:
    tasks = {}
    for group_index in range(4):
        tasks[f"origin-{group_index:02d}"] = {
            "status": "done",
            "host": "john1",
        }
        tasks[f"replay-{group_index:02d}"] = {
            "status": "done",
            "host": "john2",
        }
    state = {
        "experiment_id": (
            "complete-action-frontier-calibrated-neural-stage-v1"
        ),
        "tasks": tasks,
    }
    assert validate_scheduler_state(state)["tasks"]


def test_evaluation_reports_aggregate_by_group_count() -> None:
    reports = [
        {
            "groups": 1,
            "candidates": 10,
            "target_slots": 4,
            "target_hits": 3,
            "target_set_exact_fraction": 0.0,
            "r4800_winner_retention": 1.0,
            "mean_objective": 2.0,
            "all_scores_finite": True,
        },
        {
            "groups": 1,
            "candidates": 20,
            "target_slots": 6,
            "target_hits": 6,
            "target_set_exact_fraction": 1.0,
            "r4800_winner_retention": 0.0,
            "mean_objective": 4.0,
            "all_scores_finite": True,
        },
    ]
    combined = aggregate_evaluation_reports(reports)
    assert combined["groups"] == 2
    assert combined["candidates"] == 30
    assert combined["target_positive_recall"] == 0.9
    assert combined["target_set_exact_fraction"] == 0.5
    assert combined["mean_objective"] == 3.0
