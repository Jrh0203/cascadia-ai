from __future__ import annotations

from pathlib import Path

from o1_public_belief_search_queue import (
    PRIMARY_HOSTS,
    REPLAY_HOSTS,
    TASK_PREFIX,
    task_specs,
)


def _tasks() -> list[dict]:
    return task_specs(
        bundle_relative=Path("artifacts/experiments/o1pbs/bundles") / ("a" * 64),
        bundle_id="a" * 64,
    )


def test_campaign_has_complete_four_host_primary_and_rotated_replay_graph() -> None:
    assert TASK_PREFIX == "o1pbs-v2"
    tasks = _tasks()
    by_id = {task["id"]: task for task in tasks}

    assert len(tasks) == 21
    assert len(by_id) == len(tasks)
    assert set(PRIMARY_HOSTS.values()) == {"john1", "john2", "john3", "john4"}
    assert set(REPLAY_HOSTS.values()) == {"john1", "john2", "john3", "john4"}
    for role, host in PRIMARY_HOSTS.items():
        task = by_id[f"{TASK_PREFIX}-run-{role}"]
        assert task["compatible_hosts"] == [host]
        assert task["workload_class"] == "independent-experiment"
        assert task["resources"] == {
            "cpu_cores": 10,
            "memory_gib": 8.0,
            "uses_mlx": True,
        }
        assert task["dependencies"] == [f"{TASK_PREFIX}-preflight-{host}"]
    for role, host in REPLAY_HOSTS.items():
        task = by_id[f"{TASK_PREFIX}-run-{role}"]
        primary_role = f"{role.removesuffix('-replay')}-primary"
        assert task["compatible_hosts"] == [host]
        assert task["dependencies"] == [
            f"{TASK_PREFIX}-run-{primary_role}",
            f"{TASK_PREFIX}-preflight-{host}",
        ]
        assert task["workload_class"] == "replica"
    assert by_id[f"{TASK_PREFIX}-aggregate"]["dependencies"] == [
        f"{TASK_PREFIX}-collect"
    ]
    assert by_id[f"{TASK_PREFIX}-aggregate"]["decision_terminal"] is True


def test_commands_use_bundled_binary_and_preserve_closed_domains() -> None:
    tasks = _tasks()
    bundle_id = "a" * 64
    for task in tasks:
        command = " ".join(task["command"])
        assert bundle_id in command
        assert "--maximum-groups" not in command
        assert "--test" not in command
        assert "--final" not in command
        assert "gameplay" not in command
    run_tasks = [task for task in tasks if f"{TASK_PREFIX}-run-" in task["id"]]
    assert len(run_tasks) == 8
    for task in run_tasks:
        command = task["command"]
        assert any(value.endswith("/o1-public-belief-search") for value in command)
        assert "RAYON_NUM_THREADS=10" in command
        assert "--role" in command
        assert "--host" in command
        assert any("/runs-v2/" in value for value in command)


def test_all_immutable_inputs_are_fanned_out_and_verified() -> None:
    tasks = _tasks()
    by_id = {task["id"]: task for task in tasks}
    for name in ("bundle", "dataset", "cohort", "intent", "model", "authorization"):
        assert f"{TASK_PREFIX}-{name}-fanout" in by_id
    for host in PRIMARY_HOSTS.values():
        preflight = by_id[f"{TASK_PREFIX}-preflight-{host}"]
        command = " ".join(preflight["command"])
        assert "verify-authorization" in command
        assert "high-regret-panel.json" in command
        assert "legacy-nnue-v4opp-mlx-v1" in command
        assert "authorization-package-v2" in command
        assert "preflights-v2" in command


def test_collection_binds_every_primary_and_replay_report() -> None:
    collection = next(
        task for task in _tasks() if task["id"] == f"{TASK_PREFIX}-collect"
    )
    assert collection["command"].count("--artifact") == 8
    joined = " ".join(collection["command"])
    assert "runs-v2" in joined
    assert "collected-v2" in joined
    for role in (*PRIMARY_HOSTS, *REPLAY_HOSTS):
        assert f"{role}.json" in joined
