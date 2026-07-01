from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import frontier_arbitrary_precision_cluster as queue
from frontier_arbitrary_precision_cluster import (
    GROUPS,
    HOSTS,
    build_tasks,
    choose_host,
    compatible_hosts,
)


def test_task_graph_has_origins_and_dependent_replays() -> None:
    tasks = build_tasks()
    assert len(tasks) == GROUPS * 2
    for group_index in range(GROUPS):
        origin = tasks[f"origin-{group_index:02d}"]
        replay = tasks[f"replay-{group_index:02d}"]
        assert origin["dependency"] is None
        assert replay["dependency"] == f"origin-{group_index:02d}"


def test_replay_excludes_actual_origin_host() -> None:
    tasks = build_tasks()
    tasks["origin-03"]["host"] = "john2"
    hosts = compatible_hosts(tasks["replay-03"], tasks)
    assert hosts == ("john1", "john3", "john4")


def test_host_choice_balances_active_then_assigned_work() -> None:
    tasks = build_tasks()
    active = {host: 2 for host in HOSTS}
    active["john3"] = 1
    capacities = {host: 4 for host in HOSTS}
    assigned = {host: 0 for host in HOSTS}
    assert (
        choose_host(
            tasks["origin-00"],
            tasks,
            active,
            capacities,
            assigned,
        )
        == "john3"
    )
    active = {host: 0 for host in HOSTS}
    assigned = {host: 2 for host in HOSTS}
    assigned["john4"] = 1
    assert (
        choose_host(
            tasks["origin-00"],
            tasks,
            active,
            capacities,
            assigned,
        )
        == "john4"
    )


def test_sparse_group_indices_are_preserved() -> None:
    previous_groups = queue.GROUPS
    previous_indices = queue.GROUP_INDICES
    try:
        queue.GROUPS = 3
        queue.GROUP_INDICES = (0, 8, 23)
        tasks = queue.build_tasks()
    finally:
        queue.GROUPS = previous_groups
        queue.GROUP_INDICES = previous_indices
    assert set(tasks) == {
        "origin-00",
        "replay-00",
        "origin-08",
        "replay-08",
        "origin-23",
        "replay-23",
    }


def test_collection_skips_unassigned_remote_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(queue.subprocess, "run", run)
    state = {
        "tasks": {
            "origin-00": {
                "kind": "origin",
                "host": "john2",
                "status": "done",
            },
            "replay-00": {
                "kind": "replay",
                "host": "john4",
                "status": "done",
            },
        }
    }
    queue._collect_remote_outputs(tmp_path, state)
    assert len(commands) == 2
    assert any("/origins/john2/" in command[2] for command in commands)
    assert any("/replays/john4/" in command[2] for command in commands)
