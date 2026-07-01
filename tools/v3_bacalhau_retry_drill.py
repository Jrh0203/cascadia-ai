#!/usr/bin/env python3
"""Prove a running V3 container survives one real Bacalhau worker loss."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from cascadia_cluster.bacalhau_api import BacalhauAPI

ENDPOINT = "http://100.110.109.6:1234"
DRILL_SLEEP_SECONDS = 120
RETRY_COMPLETION_TIMEOUT_SECONDS = 420
NODE_NAMES = {
    "Johns-Mac-mini-local": "john1",
    "john2": "john2",
    "john3": "john3",
}


def _payload(name: str, image: str, seconds: int) -> dict[str, Any]:
    return {
        "Name": name,
        "Namespace": "cascadia",
        "Type": "batch",
        "Count": 1,
        "Labels": {"cascadia.v3.part1.retry-drill": name},
        "Meta": {"cascadia.v3.scientific_eligible": "false"},
        "Tasks": [
            {
                "Name": "main",
                "Engine": {
                    "Type": "docker",
                    "Params": {
                        "Image": image,
                        "Entrypoint": ["/bin/sh"],
                        "Parameters": ["-c", f"sleep {seconds}"],
                    },
                },
                "Publisher": {"Type": "noop"},
                "Resources": {"CPU": "10", "Memory": "1Gi", "Disk": "1Gi"},
                "Timeouts": {
                    "QueueTimeout": 300,
                    "ExecutionTimeout": 300,
                    "TotalTimeout": 600,
                },
            }
        ],
    }


def _active(api: BacalhauAPI, job_id: str) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for execution in api.executions(job_id):
        state = execution.get("ComputeState") or {}
        # Bacalhau 1.9's REST transport can retain the BidAccepted enum while
        # the state message and parent job both say Running.  These executions
        # have started their container and consume the requested resources;
        # they then transition directly to Completed.  Recognize that exact
        # observed encoding without treating a merely accepted bid as active.
        if state.get("StateType") == "Running" or (
            state.get("StateType") == "BidAccepted"
            and state.get("Message") == "Running"
        ):
            active.append(execution)
    return active


def _terminal_state(api: BacalhauAPI, job_id: str) -> str:
    return str((api.get_job(job_id)["Job"].get("State") or {}).get("StateType", "Unknown"))


def _compute_nodes(api: BacalhauAPI) -> dict[str, dict[str, Any]]:
    return {
        str((node.get("Info") or {}).get("NodeID")): node
        for node in api.nodes()
        if isinstance(node.get("Info"), dict)
    }


def _workers_idle(api: BacalhauAPI) -> bool:
    nodes = _compute_nodes(api)
    for node_id in ("john2", "john3"):
        node = nodes.get(node_id)
        if node is None or str(node.get("Connection", "")).upper() != "CONNECTED":
            return False
        compute = ((node.get("Info") or {}).get("ComputeNodeInfo") or {})
        maximum = compute.get("MaxCapacity") or {}
        available = compute.get("AvailableCapacity") or {}
        if float(maximum.get("CPU", 0)) != 10 or float(available.get("CPU", 0)) != 10:
            return False
        if int(compute.get("RunningExecutions", 0)) != 0:
            return False
        if int(compute.get("EnqueuedExecutions", 0)) != 0:
            return False
    return True


def _whole_worker_reserved(api: BacalhauAPI, node_id: str) -> bool:
    node = _compute_nodes(api).get(node_id)
    if node is None:
        return False
    compute = ((node.get("Info") or {}).get("ComputeNodeInfo") or {})
    available = compute.get("AvailableCapacity") or {}
    # RunningExecutions is eventually consistent with execution state and can
    # remain zero while a container is live.  AvailableCapacity is the field
    # the scheduler itself uses and is the admission barrier we need here.
    return float(available.get("CPU", 0)) == 0


def _wait_until(predicate, timeout: float, label: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {label}")


def _remote_zsh(service: str, command: str) -> None:
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            service,
            "/bin/zsh",
            "-lc",
            shlex.quote(command),
        ],
        check=True,
    )


def _worker(service: str, action: str) -> None:
    """Stop/start either the Aqua LaunchAgent or the headless supervisor.

    The compute installer prefers a LaunchAgent but deliberately falls back to
    ``run-forever.zsh`` when an SSH-only Mac has no Aqua bootstrap domain.  A
    failure drill must control both installation modes; merely asking
    ``launchctl`` to boot out a missing GUI domain does not stop a headless
    worker.
    """

    root = "$HOME/cascadia-cluster"
    pid_file = f"{root}/state/bacalhau-supervisor.pid"
    if action == "bootout":
        command = (
            "set -euo pipefail; "
            f"root={root}; pid_file={pid_file}; "
            "process_pattern=\"$root/bin/bacalhau serve\"; "
            "launchctl bootout gui/$(id -u)/com.cascadia.bacalhau "
            ">/dev/null 2>&1 || true; "
            "if [[ -s $pid_file ]]; then "
            "pid=$(<$pid_file); "
            "if [[ $pid =~ '^[0-9]+$' ]] && kill -0 $pid 2>/dev/null; then "
            "pkill -TERM -P $pid 2>/dev/null || true; "
            "kill -TERM $pid 2>/dev/null || true; "
            "fi; rm -f $pid_file; fi; "
            "pkill -TERM -f \"$process_pattern\" 2>/dev/null || true; "
            "for attempt in {1..100}; do "
            "pgrep -f \"$process_pattern\" >/dev/null || exit 0; "
            "sleep 0.1; done; "
            "print -u2 'Bacalhau compute process did not stop'; exit 1"
        )
    elif action == "bootstrap":
        launch_agent = "$HOME/Library/LaunchAgents/com.cascadia.bacalhau.plist"
        command = (
            "set -euo pipefail; "
            f"root={root}; pid_file={pid_file}; "
            "process_pattern=\"$root/bin/bacalhau serve\"; "
            "if pgrep -f \"$process_pattern\" >/dev/null; then exit 0; fi; "
            f"if launchctl bootstrap gui/$(id -u) {launch_agent} "
            ">/dev/null 2>&1; then :; else "
            "mkdir -p $root/logs $root/state; "
            "nohup env HOME=$HOME "
            "PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin "
            "CASCADIA_CLUSTER_ROOT=$root CASCADIA_BACALHAU_ROLE=compute "
            "DOCKER_HOST=unix://$HOME/.colima/default/docker.sock "
            "$root/bin/run-forever.zsh "
            ">$root/logs/bacalhau-supervisor.stdout.log "
            "2>$root/logs/bacalhau-supervisor.stderr.log </dev/null & fi; "
            "for attempt in {1..150}; do "
            "pgrep -f \"$process_pattern\" >/dev/null && exit 0; "
            "sleep 0.1; done; "
            "print -u2 'Bacalhau compute process did not start'; exit 1"
        )
    else:
        raise ValueError(f"unsupported worker action: {action}")
    _remote_zsh(service, command)


def _restart_worker(service: str) -> None:
    try:
        _worker(service, "bootout")
    finally:
        # Never allow a failed preflight cleanup to strand a compute node.
        _worker(service, "bootstrap")


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def drill(image: str, output: Path, *, worker: str = "john3") -> dict[str, object]:
    if worker != "john3":
        raise ValueError("the registered Part 1 failure drill is isolated to idle john3")
    api = BacalhauAPI(ENDPOINT)
    nonce = uuid.uuid4().hex[:12]
    jobs: list[str] = []
    worker_stopped = False
    try:
        # Bacalhau 1.9 can retain capacity from a cancelled execution until a
        # compute restart.  Reset both otherwise-idle workers so this drill
        # begins from an observed 10+10 CPU baseline, not stale accounting.
        for service in ("john2", "john3"):
            _restart_worker(service)
        _wait_until(lambda: _workers_idle(api), 180, "idle 10-CPU john2 and john3")
        for index in range(2):
            name = f"v3-part1-retry-{nonce}-{index}"
            response = api.submit(
                # Keep both placements observable even when an image must be
                # pulled or scheduler polling lands near a task boundary.
                _payload(name, image, DRILL_SLEEP_SECONDS),
                idempotency_token=f"v3-part1-retry-{nonce}-{index}",
            )
            jobs.append(str(response["JobID"]))
            if index == 0:
                # Let the scheduler commit the first whole-node allocation
                # before opening the second job.  Concurrent submissions can
                # race the capacity snapshot and receive bids from the same
                # 10-CPU node; sequential admission leaves placement entirely
                # scheduler-driven while making the capacity reservation real.
                first_execution = _wait_until(
                    lambda: _active(api, jobs[0]),
                    180,
                    "first running 10-CPU drill job",
                )[0]
                first_node_id = str(first_execution["NodeID"])
                _wait_until(
                    lambda node_id=first_node_id: _whole_worker_reserved(api, node_id),
                    90,
                    "first whole-worker reservation heartbeat",
                )

        placements = _wait_until(
            lambda: (
                {
                    job_id: NODE_NAMES.get(_active(api, job_id)[0]["NodeID"], "unknown")
                    for job_id in jobs
                    if _active(api, job_id)
                }
                if all(_active(api, job_id) for job_id in jobs)
                else None
            ),
            180,
            "two running 10-CPU drill jobs",
        )
        target = next((job for job, node in placements.items() if node == worker), None)
        filler = next((job for job, node in placements.items() if node != worker), None)
        if target is None or filler is None:
            raise RuntimeError(f"scheduler placements did not isolate {worker}: {placements}")
        first_execution = _active(api, target)[0]

        _worker(worker, "bootout")
        worker_stopped = True
        _wait_until(
            lambda: any(
                execution["ID"] == first_execution["ID"]
                and (execution.get("ComputeState") or {}).get("StateType")
                in {"Failed", "Cancelled"}
                for execution in api.executions(target)
            ),
            90,
            "failed first execution after worker loss",
        )
        # Let the filler finish normally.  Cancelling a live v1.9 execution can
        # leave its capacity charged until worker restart, which would make the
        # retry depend on a scheduler-accounting defect rather than failover.
        _wait_until(
            lambda: _terminal_state(api, filler) == "Completed",
            DRILL_SLEEP_SECONDS + 180,
            "normal filler completion and capacity release",
        )
        retry_execution = _wait_until(
            lambda: next(
                (
                    execution
                    for execution in _active(api, target)
                    if execution["ID"] != first_execution["ID"]
                ),
                None,
            ),
            120,
            "rescheduled retry execution",
        )
        _worker(worker, "bootstrap")
        worker_stopped = False
        _wait_until(
            lambda: _terminal_state(api, target) == "Completed",
            RETRY_COMPLETION_TIMEOUT_SECONDS,
            "successful retry completion",
        )
        executions = api.executions(target)
        receipt = {
            "schema_id": "cascadia-v3-bacalhau-worker-retry-v1",
            "passed": True,
            "scientific_eligible": False,
            "image_digest": image,
            "target_job_id": target,
            "filler_job_id": filler,
            "initial_placements": placements,
            "failed_execution_id": first_execution["ID"],
            "failed_node": placements[target],
            "retry_execution_id": retry_execution["ID"],
            "retry_node": NODE_NAMES.get(retry_execution["NodeID"], "unknown"),
            "execution_attempts": sum(
                (execution.get("ComputeState") or {}).get("StateType")
                in {"Running", "Completed", "Failed", "Cancelled"}
                for execution in executions
            ),
            "final_job_state": _terminal_state(api, target),
        }
        _write_atomic(output, receipt)
        return receipt
    finally:
        if worker_stopped:
            _worker(worker, "bootstrap")
        for job_id in jobs:
            if _terminal_state(api, job_id) not in {"Completed", "Failed", "Stopped"}:
                api.stop(job_id, reason="Part 1 retry drill cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", default="john3")
    args = parser.parse_args()
    result = drill(args.image, args.output, worker=args.worker)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
