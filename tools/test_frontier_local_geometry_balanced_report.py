from __future__ import annotations

from frontier_local_geometry_balanced_report import (
    validate_scheduler_state,
)


def test_scheduler_requires_eight_distinct_tasks() -> None:
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
            "complete-action-frontier-local-geometry-balanced-target-control-v1"
        ),
        "tasks": tasks,
    }
    assert validate_scheduler_state(state) is state
