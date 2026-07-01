from __future__ import annotations

from pathlib import Path

from o1_opponent_intent_mlx_queue import (
    PRIMARY_HOSTS,
    REPLAY_HOSTS,
    task_specs,
)


def _tasks() -> dict[str, dict]:
    tasks = task_specs(
        bundle_relative=Path("artifacts/bundles/test"),
        bundle_id="a" * 64,
    )
    return {task["id"]: task for task in tasks}


def test_four_distinct_primary_arms_and_rotated_replays() -> None:
    tasks = _tasks()
    primary_ids = {f"o1mlx-v1-run-{role}" for role in PRIMARY_HOSTS}
    replay_ids = {f"o1mlx-v1-run-{role}" for role in REPLAY_HOSTS}

    assert len(primary_ids) == 4
    assert len(replay_ids) == 4
    assert {tasks[task_id]["compatible_hosts"][0] for task_id in primary_ids} == {
        "john1",
        "john2",
        "john3",
        "john4",
    }
    assert {tasks[task_id]["compatible_hosts"][0] for task_id in replay_ids} == {
        "john1",
        "john2",
        "john3",
        "john4",
    }
    assert all(set(tasks[task_id]["dependencies"]) == primary_ids for task_id in replay_ids)


def test_sealed_test_is_strictly_downstream_of_validation() -> None:
    tasks = _tasks()
    classify = tasks["o1mlx-v1-classify-validation"]
    terminal = tasks["o1mlx-v1-sealed-test"]

    assert terminal["dependencies"] == [classify["id"]]
    assert terminal["decision_terminal"] is True
    assert "--test-dataset" not in classify["command"]
    assert "--final-stress-dataset" not in classify["command"]
    assert "--test-dataset" in terminal["command"]
    assert "--final-stress-dataset" in terminal["command"]


def test_all_training_tasks_use_mlx_and_exact_fixed_step_runner() -> None:
    tasks = _tasks()
    runs = [task for task in tasks.values() if task["id"].startswith("o1mlx-v1-run-")]

    assert len(runs) == 8
    for task in runs:
        assert task["resources"]["uses_mlx"] is True
        assert task["resources"]["cpu_cores"] == 10
        assert "cascadia_mlx.opponent_intent_experiment" in task["command"]
        assert "run" in task["command"]
        assert "--smoke-steps" not in task["command"]
