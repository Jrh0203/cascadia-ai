from __future__ import annotations

from pathlib import Path

from v2_distributional_opportunity_queue import (
    PRIMARY_HOSTS,
    REPLAY_HOSTS,
    TASK_PREFIX,
    task_specs,
)


def _tasks() -> list[dict]:
    return task_specs(
        bundle_relative=Path("artifacts/experiments/v2dist/bundles") / ("a" * 64),
        bundle_id="a" * 64,
    )


def test_campaign_has_complete_setup_crossed_runs_and_terminal_classifier() -> None:
    tasks = _tasks()
    by_id = {task["id"]: task for task in tasks}

    assert len(tasks) == 19
    assert len(by_id) == len(tasks)
    assert set(PRIMARY_HOSTS.values()) == {"john1", "john2", "john3", "john4"}
    assert set(REPLAY_HOSTS.values()) == {"john1", "john2", "john3", "john4"}
    for role, host in PRIMARY_HOSTS.items():
        task = by_id[f"{TASK_PREFIX}-run-{role}"]
        assert task["compatible_hosts"] == [host]
        assert task["workload_class"] == "independent-experiment"
        assert task["resources"]["uses_mlx"] is True
        assert task["dependencies"] == [f"{TASK_PREFIX}-preflight-{host}"]
        preflight = by_id[f"{TASK_PREFIX}-preflight-{host}"]
        assert preflight["compatible_hosts"] == [host]
        assert "verify-authorization" in preflight["command"]
    for role, host in REPLAY_HOSTS.items():
        task = by_id[f"{TASK_PREFIX}-run-{role}"]
        assert task["compatible_hosts"] == [host]
        primary_role = f"{role.removesuffix('-replay')}-primary"
        assert task["dependencies"] == [
            f"{TASK_PREFIX}-run-{primary_role}",
            f"{TASK_PREFIX}-preflight-{host}",
        ]
        assert task["workload_class"] == "replica"
    classifier = by_id[f"{TASK_PREFIX}-classify"]
    assert classifier["decision_terminal"] is True
    assert classifier["dependencies"] == [f"{TASK_PREFIX}-collect"]


def test_commands_use_immutable_bundle_and_keep_closed_domains_closed() -> None:
    tasks = _tasks()
    bundle_id = "a" * 64
    for task in tasks:
        command = " ".join(task["command"])
        assert bundle_id in command
        assert "--test" not in command
        assert "--final" not in command
        assert "gameplay" not in command
    run_tasks = [task for task in tasks if f"{TASK_PREFIX}-run-" in task["id"]]
    assert len(run_tasks) == 8
    assert all("--role" in task["command"] for task in run_tasks)
    preflights = [task for task in tasks if f"{TASK_PREFIX}-preflight-" in task["id"]]
    assert len(preflights) == 4
    assert all("verify-authorization" in task["command"] for task in preflights)


def test_collection_binds_every_report_and_model() -> None:
    collection = next(task for task in _tasks() if task["id"] == f"{TASK_PREFIX}-collect")
    command = collection["command"]
    assert command.count("--artifact") == 16
    for role in (*PRIMARY_HOSTS, *REPLAY_HOSTS):
        joined = " ".join(command)
        assert f"{role}.json" in joined
        assert f"{role}.safetensors" in joined
