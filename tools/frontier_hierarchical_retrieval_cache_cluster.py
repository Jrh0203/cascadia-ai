#!/usr/bin/env python3
"""Build ADR 0115 factor-cache shards through a dynamic four-host queue."""

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

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    EXPERIMENT_ID,
)

ARTIFACT_ROOT = (
    "artifacts/experiments/"
    "full-legal-hierarchical-factor-retrieval-pilot-v1"
)
HOST_ROOTS = {
    "john1": "/Users/johnherrick/cascadia",
    "john2": "/Users/john2/cascadia-bench",
    "john3": "/Users/john3/cascadia-bench",
    "john4": "/Users/john4/cascadia-bench",
}
HOSTS = tuple(HOST_ROOTS)
SPLITS = {
    "train": {
        "shards": 7,
        "dataset": (
            "artifacts/datasets/"
            "complete-action-graded-oracle-v1-train"
        ),
        "cache": (
            "artifacts/experiments/"
            "complete-action-frontier-expected-rank-scale16-v1/"
            "cache/john2/train"
        ),
    },
    "validation": {
        "shards": 3,
        "dataset": (
            "artifacts/datasets/"
            "complete-action-graded-oracle-v1-validation"
        ),
        "cache": (
            "artifacts/experiments/"
            "complete-action-frontier-expected-rank-scale16-v1/"
            "cache/john2/validation"
        ),
    },
}


@dataclass
class RunningTask:
    task_id: str
    host: str
    process: subprocess.Popen[str]
    started: float


def _tasks() -> dict[str, dict[str, Any]]:
    return {
        f"{split}-{index:02d}": {
            "split": split,
            "shard_index": index,
            "status": "pending",
            "host": None,
            "attempts": 0,
        }
        for split, config in SPLITS.items()
        for index in range(int(config["shards"]))
    }


def _output_relative(task: dict[str, Any]) -> str:
    return (
        f"{ARTIFACT_ROOT}/cache-shards/{task['split']}/"
        f"shard-{int(task['shard_index']):02d}.npz"
    )


def _command(task: dict[str, Any]) -> list[str]:
    config = SPLITS[str(task["split"])]
    return [
        ".venv/bin/python",
        "-m",
        "cascadia_mlx.full_legal_hierarchical_factor_retrieval",
        "cache-shard",
        "--dataset",
        str(config["dataset"]),
        "--cache",
        str(config["cache"]),
        "--shard-index",
        str(task["shard_index"]),
        "--output",
        _output_relative(task),
    ]


def _launch(host: str, command: list[str]) -> list[str]:
    command = ["caffeinate", "-dimsu", *command]
    if host == "john1":
        return command
    remote = (
        f"cd {shlex.quote(HOST_ROOTS[host])} && "
        f"{shlex.join(command)}"
    )
    return ["ssh", host, remote]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _local_report(task: dict[str, Any]) -> Path:
    return (
        Path(HOST_ROOTS["john1"])
        / Path(_output_relative(task)).with_suffix(".json")
    )


def _load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text())
        if state.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("cache scheduler experiment identity drifted")
        for task in state["tasks"].values():
            if task["status"] in {"running", "failed"}:
                task["status"] = "pending"
                task["host"] = None
        return state
    tasks = _tasks()
    for task in tasks.values():
        report = _local_report(task)
        payload = report.with_suffix(".npz")
        if report.is_file() and payload.is_file():
            value = json.loads(report.read_text())
            if value.get("experiment_id") == EXPERIMENT_ID:
                task["status"] = "done"
                task["host"] = "john1"
                task["elapsed_seconds"] = float(
                    value["execution"]["elapsed_seconds"]
                )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_unix_seconds": time.time(),
        "tasks": tasks,
    }


def _sync_source() -> None:
    relative = (
        "python/cascadia_mlx/"
        "full_legal_hierarchical_factor_retrieval.py"
    )
    for host in HOSTS[1:]:
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                relative,
                f"{host}:{HOST_ROOTS[host]}/"
                "python/cascadia_mlx/",
            ],
            cwd=HOST_ROOTS["john1"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"source sync to {host} failed: "
                f"{completed.stderr.strip()}"
            )


def _collect(task: dict[str, Any], host: str) -> None:
    if host == "john1":
        return
    relative = _output_relative(task)
    for suffix in (".npz", ".json"):
        source_relative = str(Path(relative).with_suffix(suffix))
        destination = Path(HOST_ROOTS["john1"]) / source_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                f"{host}:{HOST_ROOTS[host]}/{source_relative}",
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"cache collection from {host} failed: "
                f"{completed.stderr.strip()}"
            )


def run(state_path: Path, poll_seconds: float) -> dict[str, Any]:
    _sync_source()
    state = _load_state(state_path)
    events_path = state_path.with_name("events.jsonl")
    logs_root = state_path.with_name("logs")
    running: dict[str, RunningTask] = {}
    active = {host: 0 for host in HOSTS}
    assigned = {
        host: sum(
            int(
                task.get("host") == host
                and task["status"] == "done"
            )
            for task in state["tasks"].values()
        )
        for host in HOSTS
    }
    started = time.time()
    while True:
        completed_ids = []
        for task_id, job in running.items():
            return_code = job.process.poll()
            if return_code is None:
                continue
            output, _ = job.process.communicate()
            logs_root.mkdir(parents=True, exist_ok=True)
            (logs_root / f"{task_id}.log").write_text(output)
            task = state["tasks"][task_id]
            task["status"] = "done" if return_code == 0 else "failed"
            task["return_code"] = return_code
            task["elapsed_seconds"] = time.time() - job.started
            active[job.host] -= 1
            if return_code == 0:
                _collect(task, job.host)
            _append_event(
                events_path,
                {
                    "event": "finished",
                    "task_id": task_id,
                    "host": job.host,
                    "return_code": return_code,
                    "elapsed_seconds": task["elapsed_seconds"],
                    "unix_seconds": time.time(),
                },
            )
            completed_ids.append(task_id)
        for task_id in completed_ids:
            del running[task_id]
        if completed_ids:
            _write_json(state_path, state)

        if any(
            task["status"] == "failed"
            for task in state["tasks"].values()
        ):
            if not running:
                raise RuntimeError(
                    "factor-cache queue stopped after a failed task"
                )
        else:
            while True:
                ready = [
                    task_id
                    for task_id, task in state["tasks"].items()
                    if task["status"] == "pending"
                ]
                available = [host for host in HOSTS if active[host] == 0]
                if not ready or not available:
                    break
                task_id = sorted(ready)[0]
                host = min(
                    available,
                    key=lambda value: (assigned[value], value),
                )
                task = state["tasks"][task_id]
                command = _command(task)
                process = subprocess.Popen(
                    _launch(host, command),
                    cwd=HOST_ROOTS["john1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                task["status"] = "running"
                task["host"] = host
                task["attempts"] = int(task["attempts"]) + 1
                task["started_unix_seconds"] = time.time()
                active[host] += 1
                assigned[host] += 1
                running[task_id] = RunningTask(
                    task_id=task_id,
                    host=host,
                    process=process,
                    started=time.time(),
                )
                _append_event(
                    events_path,
                    {
                        "event": "started",
                        "task_id": task_id,
                        "host": host,
                        "command": command,
                        "unix_seconds": time.time(),
                    },
                )
                _write_json(state_path, state)

        statuses = [
            task["status"] for task in state["tasks"].values()
        ]
        if not running and all(status == "done" for status in statuses):
            break
        _append_event(
            events_path,
            {
                "event": "snapshot",
                "active": active,
                "pending": statuses.count("pending"),
                "unix_seconds": time.time(),
            },
        )
        time.sleep(poll_seconds)
    state["completed_unix_seconds"] = time.time()
    state["campaign_wall_seconds"] = (
        state["completed_unix_seconds"] - started
    )
    state["scheduled_process_seconds"] = sum(
        float(task.get("elapsed_seconds", 0.0))
        for task in state["tasks"].values()
    )
    _write_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()
    state_path = (
        Path(ARTIFACT_ROOT)
        / "cache-scheduler"
        / "state.json"
    )
    state = run(state_path, args.poll_seconds)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "completed_tasks": sum(
                    int(task["status"] == "done")
                    for task in state["tasks"].values()
                ),
                "campaign_wall_seconds": state["campaign_wall_seconds"],
                "scheduled_process_seconds": state[
                    "scheduled_process_seconds"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
