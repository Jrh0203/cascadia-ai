from __future__ import annotations

from frontier_hierarchical_factor_oracle_report import (
    validate_scheduler_state,
)


def test_scheduler_requires_cross_host_replays() -> None:
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
        "experiment_id": "full-legal-hierarchical-factor-oracle-v1",
        "tasks": tasks,
    }
    assert validate_scheduler_state(state) is state
