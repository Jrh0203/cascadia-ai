from __future__ import annotations

from pathlib import Path

import pytest
from p1_relational_pointer_pilot_queue import (
    EXPERIMENT_ID,
    SCHEDULER_WORKLOAD_CLASSES,
    PointerPilotQueueError,
    campaign_spec,
    task_specs,
)

BUNDLE_ID = "a" * 64
BUNDLE_RELATIVE = Path(
    "artifacts/experiments/"
    "p1-relational-selected-prefix-pointer-pilot-v1/bundles/"
    f"{BUNDLE_ID}"
)


def test_campaign_has_four_way_launch_and_complete_terminal_graph() -> None:
    tasks = task_specs(
        bundle_relative=BUNDLE_RELATIVE,
        bundle_id=BUNDLE_ID,
    )
    assert len(tasks) == 17
    by_id = {task["id"]: task for task in tasks}
    assert len(by_id) == len(tasks)
    assert by_id["p1pilot-v1-auth-stage"]["dependencies"] == [
        "p1ptr-v1-classify"
    ]
    launch = {
        task_id: by_id[task_id]["compatible_hosts"]
        for task_id in (
            "p1pilot-v1-train-draft-john1",
            "p1pilot-v1-train-tile-john2",
            "p1pilot-v1-train-wildlife-john3",
            "p1pilot-v1-smoke-wildlife-john4",
        )
    }
    assert launch == {
        "p1pilot-v1-train-draft-john1": ["john1"],
        "p1pilot-v1-train-tile-john2": ["john2"],
        "p1pilot-v1-train-wildlife-john3": ["john3"],
        "p1pilot-v1-smoke-wildlife-john4": ["john4"],
    }
    for task_id in launch:
        assert by_id[task_id]["dependencies"] == [
            "p1pilot-v1-auth-fanout"
        ]
        assert by_id[task_id]["resources"]["uses_mlx"] is True
    terminal = by_id["p1pilot-v1-integrate"]
    assert terminal["decision_terminal"] is True
    assert terminal["compatible_hosts"] == ["john1"]
    assert "p1pilot-v1-collect-replays" in terminal["dependencies"]
    assert "--bundle-id" in by_id[
        "p1pilot-v1-train-tile-john2"
    ]["command"]
    assert BUNDLE_ID in by_id[
        "p1pilot-v1-train-tile-john2"
    ]["command"]
    assert {
        task["workload_class"] for task in tasks
    } <= SCHEDULER_WORKLOAD_CLASSES


def test_every_internal_dependency_exists_and_graph_is_acyclic() -> None:
    tasks = task_specs(
        bundle_relative=BUNDLE_RELATIVE,
        bundle_id=BUNDLE_ID,
    )
    by_id = {task["id"]: task for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise AssertionError("queue graph contains a cycle")
        visiting.add(task_id)
        for dependency in by_id[task_id]["dependencies"]:
            if dependency == "p1ptr-v1-classify":
                continue
            assert dependency in by_id
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)
    assert visited == set(by_id)


def test_campaign_envelope_and_invalid_bundle_id() -> None:
    tasks = task_specs(
        bundle_relative=BUNDLE_RELATIVE,
        bundle_id=BUNDLE_ID,
    )
    campaign = campaign_spec(tasks, bundle_id=BUNDLE_ID)
    assert campaign["experiment_id"] == EXPERIMENT_ID
    assert campaign["task_count"] == len(tasks)
    with pytest.raises(PointerPilotQueueError, match="bundle ID"):
        task_specs(
            bundle_relative=BUNDLE_RELATIVE,
            bundle_id="not-a-digest",
        )
