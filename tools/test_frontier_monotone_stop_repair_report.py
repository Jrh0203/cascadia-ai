from __future__ import annotations

import pytest
from frontier_monotone_stop_repair_report import (
    REPAIR_GROUPS,
    validate_scheduler_state,
)


def _state() -> dict[str, object]:
    tasks = {}
    for group_index in REPAIR_GROUPS:
        tasks[f"origin-{group_index:02d}"] = {
            "status": "done",
            "host": "john1",
        }
        tasks[f"replay-{group_index:02d}"] = {
            "status": "done",
            "host": "john2",
        }
    return {
        "experiment_id": (
            "complete-action-frontier-monotone-adamw-stop-repair-v1"
        ),
        "tasks": tasks,
    }


def test_sparse_scheduler_state_is_valid() -> None:
    assert validate_scheduler_state(_state())["tasks"]


def test_same_host_replay_is_rejected() -> None:
    state = _state()
    tasks = state["tasks"]
    assert isinstance(tasks, dict)
    tasks["replay-08"]["host"] = "john1"
    with pytest.raises(ValueError, match="failed validation"):
        validate_scheduler_state(state)
