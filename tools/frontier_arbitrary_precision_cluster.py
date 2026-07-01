#!/usr/bin/env python3
"""Run ADR 0105 as a resumable dynamic four-host group queue."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "complete-action-frontier-arbitrary-precision-control-v1"
MODULE_NAME = (
    "cascadia_mlx.graded_oracle_frontier_arbitrary_precision"
)
COMMAND_NAME = "group"
HOST_ROOTS = {
    "john1": "/Users/johnherrick/cascadia",
    "john2": "/Users/john2/cascadia-bench",
    "john3": "/Users/john3/cascadia-bench",
    "john4": "/Users/john4/cascadia-bench",
}
HOSTS = tuple(HOST_ROOTS)
GROUPS = 24
GROUP_INDICES: tuple[int, ...] | None = None
INITIAL_CAPACITY = 2
MAXIMUM_CAPACITY = 10
CAPACITY_STEP = 2
ARTIFACT_ROOT = (
    "artifacts/experiments/"
    "complete-action-frontier-arbitrary-precision-control-v1"
)
DATASET = "artifacts/datasets/complete-action-graded-oracle-v1-train"
CACHE = (
    "artifacts/experiments/"
    "complete-action-frontier-expected-rank-scale16-v1/cache/john2/train"
)
SELECTED_RUN = (
    "artifacts/experiments/"
    "complete-action-frontier-expected-rank-scale16-v1/"
    "runs/john2-seed-2026061626"
)
ANALYTIC = (
    "artifacts/experiments/"
    "complete-action-frontier-free-residual-audit-v1/"
    "reports/analytic-john1.json"
)


@dataclass
class RunningTask:
    task_id: str
    host: str
    process: subprocess.Popen[str]
    started: float


def build_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    group_indices = (
        tuple(range(GROUPS))
        if GROUP_INDICES is None
        else GROUP_INDICES
    )
    if len(group_indices) != GROUPS or len(set(group_indices)) != GROUPS:
        raise ValueError("group indices must be unique and match GROUPS")
    for group_index in group_indices:
        origin_id = f"origin-{group_index:02d}"
        replay_id = f"replay-{group_index:02d}"
        tasks[origin_id] = {
            "kind": "origin",
            "group_index": group_index,
            "dependency": None,
            "status": "pending",
            "host": None,
            "attempts": 0,
        }
        tasks[replay_id] = {
            "kind": "replay",
            "group_index": group_index,
            "dependency": origin_id,
            "status": "pending",
            "host": None,
            "attempts": 0,
        }
    return tasks


def compatible_hosts(
    task: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    if task["kind"] == "origin":
        return HOSTS
    origin = tasks[str(task["dependency"])]
    origin_host = origin.get("host")
    if origin_host not in HOSTS:
        return ()
    return tuple(host for host in HOSTS if host != origin_host)


def choose_host(
    task: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    active_counts: dict[str, int],
    capacities: dict[str, int],
    assigned_counts: dict[str, int],
) -> str | None:
    available = [
        host
        for host in compatible_hosts(task, tasks)
        if active_counts[host] < capacities[host]
    ]
    if not available:
        return None
    return min(
        available,
        key=lambda host: (
            active_counts[host],
            assigned_counts[host],
            host,
        ),
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _task_output(task: dict[str, Any], host: str) -> str:
    directory = "origins" if task["kind"] == "origin" else "replays"
    return (
        f"{ARTIFACT_ROOT}/{directory}/{host}/"
        f"group-{int(task['group_index']):02d}.json"
    )


def _task_command(task: dict[str, Any], host: str) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        MODULE_NAME,
        COMMAND_NAME,
        "--dataset",
        DATASET,
        "--cache",
        CACHE,
        "--selected-run",
        SELECTED_RUN,
        "--analytic",
        ANALYTIC,
        "--group-index",
        str(task["group_index"]),
        "--output",
        _task_output(task, host),
    ]


def _launch_command(host: str, command: list[str]) -> list[str]:
    caffeinated = ["caffeinate", "-dimsu", *command]
    if host == "john1":
        return caffeinated
    remote = (
        f"cd {shlex.quote(HOST_ROOTS[host])} && "
        f"{shlex.join(caffeinated)}"
    )
    return ["ssh", host, remote]


def _qualification_from_output(output: str) -> bool:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return False
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False
    return bool(value.get("resource_qualification_passed"))


def _load_or_create_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text())
        if state.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("scheduler state has the wrong experiment ID")
        for task in state["tasks"].values():
            if task["status"] in {"running", "failed"}:
                task["status"] = "pending"
                task["host"] = None
        return state
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_unix_seconds": time.time(),
        "tasks": build_tasks(),
        "hosts": {
            host: {
                "capacity": INITIAL_CAPACITY,
                "qualification_successes": 0,
            }
            for host in HOSTS
        },
    }


def _ready_tasks(state: dict[str, Any]) -> list[str]:
    tasks = state["tasks"]
    ready = []
    for task_id, task in tasks.items():
        if task["status"] != "pending":
            continue
        dependency = task["dependency"]
        if dependency is None or tasks[dependency]["status"] == "done":
            ready.append(task_id)
    return sorted(
        ready,
        key=lambda task_id: (
            0 if tasks[task_id]["kind"] == "origin" else 1,
            int(tasks[task_id]["group_index"]),
        ),
    )


def _collect_remote_outputs(
    local_root: Path,
    state: dict[str, Any],
) -> None:
    assigned = {
        (
            str(task.get("host")),
            "origins" if task["kind"] == "origin" else "replays",
        )
        for task in state["tasks"].values()
        if task.get("status") == "done"
    }
    for host in HOSTS[1:]:
        for directory in ("origins", "replays"):
            if (host, directory) not in assigned:
                continue
            destination = (
                local_root / directory / host
            )
            destination.mkdir(parents=True, exist_ok=True)
            source = (
                f"{host}:{HOST_ROOTS[host]}/{ARTIFACT_ROOT}/"
                f"{directory}/{host}/"
            )
            completed = subprocess.run(
                ["rsync", "-a", source, f"{destination}/"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"failed to collect {host} {directory}: "
                    f"{completed.stderr.strip()}"
                )


def run_queue(state_path: Path, poll_seconds: float) -> dict[str, Any]:
    state = _load_or_create_state(state_path)
    events_path = state_path.with_name("events.jsonl")
    logs_root = state_path.with_name("logs")
    running: dict[str, RunningTask] = {}
    active_counts = {host: 0 for host in HOSTS}
    assigned_counts = {
        host: sum(
            int(
                task.get("host") == host
                and task["status"] == "done"
            )
            for task in state["tasks"].values()
        )
        for host in HOSTS
    }
    halted = False
    campaign_started = time.time()
    last_snapshot: tuple[Any, ...] | None = None
    while True:
        completed_ids = []
        for task_id, running_task in running.items():
            return_code = running_task.process.poll()
            if return_code is None:
                continue
            output, _ = running_task.process.communicate()
            logs_root.mkdir(parents=True, exist_ok=True)
            (logs_root / f"{task_id}.log").write_text(output)
            ended = time.time()
            task = state["tasks"][task_id]
            task["ended_unix_seconds"] = ended
            task["elapsed_seconds"] = ended - running_task.started
            task["return_code"] = return_code
            task["status"] = "done" if return_code == 0 else "failed"
            task["resource_qualification_passed"] = (
                return_code == 0 and _qualification_from_output(output)
            )
            active_counts[running_task.host] -= 1
            if task["resource_qualification_passed"]:
                host_state = state["hosts"][running_task.host]
                host_state["qualification_successes"] += 1
                if (
                    host_state["qualification_successes"]
                    >= host_state["capacity"]
                    and host_state["capacity"] < MAXIMUM_CAPACITY
                ):
                    previous_capacity = host_state["capacity"]
                    host_state["capacity"] = min(
                        MAXIMUM_CAPACITY,
                        host_state["capacity"] + CAPACITY_STEP,
                    )
                    host_state["qualification_successes"] = 0
                    _append_event(
                        events_path,
                        {
                            "event": "capacity-increased",
                            "host": running_task.host,
                            "previous_capacity": previous_capacity,
                            "capacity": host_state["capacity"],
                            "unix_seconds": ended,
                        },
                    )
            if return_code != 0:
                halted = True
            _append_event(
                events_path,
                {
                    "event": "finished",
                    "task_id": task_id,
                    "kind": task["kind"],
                    "group_index": task["group_index"],
                    "host": running_task.host,
                    "ended_unix_seconds": ended,
                    "elapsed_seconds": task["elapsed_seconds"],
                    "return_code": return_code,
                    "resource_qualification_passed": task[
                        "resource_qualification_passed"
                    ],
                },
            )
            completed_ids.append(task_id)
        for task_id in completed_ids:
            del running[task_id]
        if completed_ids:
            _write_json_atomic(state_path, state)

        if not halted:
            capacities = {
                host: int(state["hosts"][host]["capacity"])
                for host in HOSTS
            }
            while True:
                launched = False
                for task_id in _ready_tasks(state):
                    task = state["tasks"][task_id]
                    host = choose_host(
                        task,
                        state["tasks"],
                        active_counts,
                        capacities,
                        assigned_counts,
                    )
                    if host is None:
                        continue
                    command = _task_command(task, host)
                    process = subprocess.Popen(
                        _launch_command(host, command),
                        cwd=HOST_ROOTS["john1"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    started = time.time()
                    task["status"] = "running"
                    task["host"] = host
                    task["attempts"] = int(task["attempts"]) + 1
                    task["started_unix_seconds"] = started
                    task["output"] = _task_output(task, host)
                    running[task_id] = RunningTask(
                        task_id=task_id,
                        host=host,
                        process=process,
                        started=started,
                    )
                    active_counts[host] += 1
                    assigned_counts[host] += 1
                    _append_event(
                        events_path,
                        {
                            "event": "started",
                            "task_id": task_id,
                            "kind": task["kind"],
                            "group_index": task["group_index"],
                            "host": host,
                            "capacity": capacities[host],
                            "started_unix_seconds": started,
                            "command": command,
                        },
                    )
                    _write_json_atomic(state_path, state)
                    launched = True
                    break
                if not launched:
                    break

        ready = _ready_tasks(state)
        compatible_ready = {
            host: sum(
                int(host in compatible_hosts(state["tasks"][task_id], state["tasks"]))
                for task_id in ready
            )
            for host in HOSTS
        }
        snapshot = (
            tuple(active_counts.items()),
            tuple(
                (host, int(state["hosts"][host]["capacity"]))
                for host in HOSTS
            ),
            tuple(compatible_ready.items()),
            len(ready),
        )
        if snapshot != last_snapshot:
            _append_event(
                events_path,
                {
                    "event": "snapshot",
                    "unix_seconds": time.time(),
                    "active": dict(active_counts),
                    "capacity": {
                        host: int(state["hosts"][host]["capacity"])
                        for host in HOSTS
                    },
                    "compatible_ready": compatible_ready,
                    "ready_tasks": len(ready),
                },
            )
            last_snapshot = snapshot

        statuses = [
            task["status"] for task in state["tasks"].values()
        ]
        if not running and all(status == "done" for status in statuses):
            break
        if not running and (halted or "failed" in statuses):
            raise RuntimeError(
                "ADR 0105 queue stopped after a failed task; rerun to resume"
            )
        time.sleep(poll_seconds)

    state["completed_unix_seconds"] = time.time()
    state["campaign_wall_seconds"] = (
        state["completed_unix_seconds"] - campaign_started
    )
    _write_json_atomic(state_path, state)
    _collect_remote_outputs(state_path.parent.parent, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(ARTIFACT_ROOT) / "scheduler" / "state.json",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    state = run_queue(args.state, args.poll_seconds)
    print(
        json.dumps(
            {
                "experiment_id": state["experiment_id"],
                "campaign_wall_seconds": state["campaign_wall_seconds"],
                "completed_tasks": sum(
                    int(task["status"] == "done")
                    for task in state["tasks"].values()
                ),
                "host_capacities": {
                    host: state["hosts"][host]["capacity"]
                    for host in HOSTS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
