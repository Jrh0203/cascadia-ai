#!/usr/bin/env python3
"""Manage and dispatch the manifest-backed Cascadia research queue."""

# ruff: noqa: UP045 - cluster tools must run under the macOS system Python 3.9.

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import secrets
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
LEGACY_FREEZE_MARKER = "legacy-queue-freeze-v1.json"
DEFAULT_HOSTS = ("john1", "john2", "john3", "john4")
WORKLOAD_CLASSES = {
    "independent-experiment",
    "divisible-evidence",
    "shared-prerequisite",
    "replica",
}
TASK_STATUSES = {"blocked", "ready", "running", "completed", "failed", "cancelled"}
HOST_INTENT_STATUSES = {"available", "working", "reserved", "intentionally-idle"}
DEFAULT_ROOTS = {
    "john1": "/Users/johnherrick/cascadia",
    "john2": "/Users/john2/cascadia-bench",
    "john3": "/Users/john3/cascadia-bench",
    "john4": "/Users/john4/cascadia-bench",
}


class QueueError(RuntimeError):
    """Raised when queue state or a requested transition is invalid."""


def unix_millis() -> int:
    return time.time_ns() // 1_000_000


def empty_queue(campaign_id: str, now_ms: Optional[int] = None) -> dict[str, Any]:
    timestamp = unix_millis() if now_ms is None else now_ms
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "created_unix_ms": timestamp,
        "updated_unix_ms": timestamp,
        "hosts": {
            host: {
                "root": DEFAULT_ROOTS[host],
                "intent": "available",
                "reason": None,
                "updated_unix_ms": timestamp,
            }
            for host in DEFAULT_HOSTS
        },
        "tasks": [],
        "events": [],
    }


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueueError(f"{field} must be a nonempty string")
    return value


def _require_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise QueueError(f"{field} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise QueueError(f"{field} must not contain duplicates")
    return value


def _validate_claim(claim: Any, task_id: str) -> None:
    if claim is None:
        return
    if not isinstance(claim, dict):
        raise QueueError(f"task {task_id} claim must be an object or null")
    for field in ("host", "token"):
        _require_string(claim.get(field), f"task {task_id} claim.{field}")
    for field in ("claimed_unix_ms", "heartbeat_unix_ms", "lease_expires_unix_ms"):
        if not isinstance(claim.get(field), int) or claim[field] < 0:
            raise QueueError(f"task {task_id} claim.{field} must be a nonnegative integer")
    if claim["lease_expires_unix_ms"] <= claim["claimed_unix_ms"]:
        raise QueueError(f"task {task_id} lease must expire after it is claimed")
    if claim["heartbeat_unix_ms"] < claim["claimed_unix_ms"]:
        raise QueueError(f"task {task_id} heartbeat precedes its claim")
    if claim["lease_expires_unix_ms"] <= claim["heartbeat_unix_ms"]:
        raise QueueError(f"task {task_id} lease must expire after its heartbeat")


def validate_queue(state: dict[str, Any]) -> dict[str, Any]:
    """Validate all queue invariants and return the original state."""
    if not isinstance(state, dict):
        raise QueueError("queue must be a JSON object")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise QueueError("unsupported queue schema version")
    _require_string(state.get("campaign_id"), "campaign_id")
    for field in ("created_unix_ms", "updated_unix_ms"):
        if not isinstance(state.get(field), int) or state[field] < 0:
            raise QueueError(f"{field} must be a nonnegative integer")

    hosts = state.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise QueueError("hosts must be a nonempty object")
    for host, value in hosts.items():
        _require_string(host, "host identifier")
        if not isinstance(value, dict):
            raise QueueError(f"host {host} must be an object")
        _require_string(value.get("root"), f"host {host} root")
        if value.get("intent") not in HOST_INTENT_STATUSES:
            raise QueueError(f"host {host} has an invalid intent")
        reason = value.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise QueueError(f"host {host} reason must be a string or null")

    tasks = state.get("tasks")
    if not isinstance(tasks, list):
        raise QueueError("tasks must be a list")
    task_ids: set[str] = set()
    running_hosts: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise QueueError("each task must be an object")
        task_id = _require_string(task.get("id"), "task id")
        if task_id in task_ids:
            raise QueueError(f"duplicate task id {task_id}")
        task_ids.add(task_id)
        for field in ("title", "experiment_id", "decision", "artifact_path", "stop_rule"):
            _require_string(task.get(field), f"task {task_id} {field}")
        if task.get("workload_class") not in WORKLOAD_CLASSES:
            raise QueueError(f"task {task_id} has an invalid workload_class")
        if task.get("status") not in TASK_STATUSES:
            raise QueueError(f"task {task_id} has an invalid status")
        priority = task.get("priority")
        if not isinstance(priority, int) or priority < 0:
            raise QueueError(f"task {task_id} priority must be a nonnegative integer")
        decision_value = task.get("decision_value")
        if not isinstance(decision_value, (int, float)) or decision_value <= 0:
            raise QueueError(f"task {task_id} decision_value must be positive")
        runtime = task.get("expected_runtime_seconds")
        if not isinstance(runtime, (int, float)) or runtime <= 0:
            raise QueueError(f"task {task_id} expected runtime must be positive")
        if not isinstance(task.get("critical_path"), bool):
            raise QueueError(f"task {task_id} critical_path must be boolean")
        if not isinstance(task.get("decision_terminal"), bool):
            raise QueueError(f"task {task_id} decision_terminal must be boolean")
        compatible = _require_string_list(
            task.get("compatible_hosts"), f"task {task_id} compatible_hosts"
        )
        if not compatible:
            raise QueueError(f"task {task_id} must have at least one compatible host")
        unknown_hosts = set(compatible) - set(hosts)
        if unknown_hosts:
            raise QueueError(f"task {task_id} names unknown hosts: {sorted(unknown_hosts)}")
        _require_string_list(task.get("dependencies"), f"task {task_id} dependencies")
        command = task.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise QueueError(f"task {task_id} command must not be empty")
        resources = task.get("resources")
        if not isinstance(resources, dict):
            raise QueueError(f"task {task_id} resources must be an object")
        cpu_cores = resources.get("cpu_cores")
        memory_gib = resources.get("memory_gib")
        if not isinstance(cpu_cores, int) or cpu_cores < 0:
            raise QueueError(f"task {task_id} cpu_cores must be nonnegative")
        if not isinstance(memory_gib, (int, float)) or memory_gib < 0:
            raise QueueError(f"task {task_id} memory_gib must be nonnegative")
        if not isinstance(resources.get("uses_mlx"), bool):
            raise QueueError(f"task {task_id} uses_mlx must be boolean")
        _validate_claim(task.get("claim"), task_id)
        if task["status"] == "running" and task.get("claim") is None:
            raise QueueError(f"running task {task_id} must have a claim")
        if task["status"] != "running" and task.get("claim") is not None:
            raise QueueError(f"non-running task {task_id} must not have a claim")
        claim = task.get("claim")
        if claim is not None:
            if claim["host"] not in compatible:
                raise QueueError(f"task {task_id} is claimed by an incompatible host")
            if claim["host"] in running_hosts:
                raise QueueError(f"host {claim['host']} holds multiple running claims")
            running_hosts.add(claim["host"])
        attempts = task.get("attempts")
        if not isinstance(attempts, list):
            raise QueueError(f"task {task_id} attempts must be a list")
        previous_end = -1
        for attempt in sorted(
            attempts,
            key=lambda value: (
                int(value.get("claimed_unix_ms", -1)) if isinstance(value, dict) else -1
            ),
        ):
            if not isinstance(attempt, dict):
                raise QueueError(f"task {task_id} attempt must be an object")
            if attempt.get("host") not in compatible:
                raise QueueError(f"task {task_id} attempt used an incompatible host")
            for field in ("claimed_unix_ms", "heartbeat_unix_ms", "ended_unix_ms"):
                if not isinstance(attempt.get(field), int):
                    raise QueueError(f"task {task_id} attempt.{field} must be an integer")
            if attempt["claimed_unix_ms"] < int(task["created_unix_ms"]):
                raise QueueError(f"task {task_id} attempt precedes task creation")
            if attempt["heartbeat_unix_ms"] < attempt["claimed_unix_ms"]:
                raise QueueError(f"task {task_id} attempt heartbeat precedes its claim")
            if attempt["ended_unix_ms"] < attempt["claimed_unix_ms"]:
                raise QueueError(f"task {task_id} attempt ends before it begins")
            if attempt["claimed_unix_ms"] < previous_end:
                raise QueueError(f"task {task_id} attempts overlap")
            previous_end = attempt["ended_unix_ms"]
            if attempt.get("outcome") not in {
                "completed",
                "failed",
                "cancelled",
                "lease-expired",
            }:
                raise QueueError(f"task {task_id} attempt has an invalid outcome")
        result = task.get("result")
        if result is not None:
            if not isinstance(result, dict):
                raise QueueError(f"task {task_id} result must be an object or null")
            if result.get("host") not in compatible:
                raise QueueError(f"task {task_id} result used an incompatible host")
            if not isinstance(result.get("completed_unix_ms"), int):
                raise QueueError(f"task {task_id} result.completed_unix_ms must be an integer")
            if result["completed_unix_ms"] < int(task["created_unix_ms"]):
                raise QueueError(f"task {task_id} result precedes task creation")
        if task["status"] == "completed" and result is None:
            raise QueueError(f"completed task {task_id} must have a result")
        administrative_cancellation = task.get("administrative_cancellation")
        if administrative_cancellation is not None:
            if not isinstance(administrative_cancellation, dict):
                raise QueueError(
                    f"task {task_id} administrative_cancellation must be an object or null"
                )
            if task["status"] != "cancelled":
                raise QueueError(
                    f"task {task_id} has administrative cancellation metadata "
                    "without cancelled status"
                )
            for field in ("actor", "reason"):
                _require_string(
                    administrative_cancellation.get(field),
                    f"task {task_id} administrative_cancellation.{field}",
                )
            cancelled_unix_ms = administrative_cancellation.get("cancelled_unix_ms")
            if not isinstance(cancelled_unix_ms, int) or cancelled_unix_ms < int(
                task["created_unix_ms"]
            ):
                raise QueueError(
                    f"task {task_id} administrative cancellation precedes task creation"
                )
            if administrative_cancellation.get("previous_status") not in {
                "ready",
                "blocked",
                "failed",
            }:
                raise QueueError(
                    f"task {task_id} administrative cancellation has an invalid previous_status"
                )

    for task in tasks:
        unknown = set(task["dependencies"]) - task_ids
        if unknown:
            raise QueueError(f"task {task['id']} has unknown dependencies: {sorted(unknown)}")
        if task["id"] in task["dependencies"]:
            raise QueueError(f"task {task['id']} cannot depend on itself")
    _validate_acyclic(tasks)

    events = state.get("events")
    if not isinstance(events, list):
        raise QueueError("events must be a list")
    return state


def _validate_acyclic(tasks: list[dict[str, Any]]) -> None:
    dependencies = {task["id"]: task["dependencies"] for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise QueueError("task dependency graph contains a cycle")
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in dependencies:
        visit(task_id)


def load_queue(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise QueueError(f"queue does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise QueueError(f"queue is not valid JSON: {path}") from error
    return validate_queue(state)


def _atomic_write(path: Path, state: dict[str, Any]) -> None:
    validate_queue(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def locked_queue(path: Path) -> Iterator[dict[str, Any]]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = load_queue(path)
        before = copy.deepcopy(state)
        try:
            yield state
        except Exception:
            raise
        else:
            if state != before:
                state["updated_unix_ms"] = unix_millis()
                _atomic_write(path, state)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _append_event(
    state: dict[str, Any],
    event: str,
    *,
    task_id: Optional[str] = None,
    host: Optional[str] = None,
    detail: Optional[dict[str, Any]] = None,
    now_ms: Optional[int] = None,
) -> None:
    item: dict[str, Any] = {
        "event": event,
        "unix_ms": unix_millis() if now_ms is None else now_ms,
    }
    if task_id is not None:
        item["task_id"] = task_id
    if host is not None:
        item["host"] = host
    if detail:
        item["detail"] = detail
    state["events"].append(item)


def _task_by_id(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in state["tasks"]:
        if task["id"] == task_id:
            return task
    raise QueueError(f"unknown task {task_id}")


def _completed_ids(state: dict[str, Any]) -> set[str]:
    return {task["id"] for task in state["tasks"] if task["status"] == "completed"}


def refresh_dependencies(state: dict[str, Any]) -> None:
    completed = _completed_ids(state)
    for task in state["tasks"]:
        if task["status"] not in {"ready", "blocked"}:
            continue
        task["status"] = (
            "ready"
            if all(dependency in completed for dependency in task["dependencies"])
            else "blocked"
        )


def expire_leases(state: dict[str, Any], now_ms: Optional[int] = None) -> list[str]:
    now = unix_millis() if now_ms is None else now_ms
    expired = []
    for task in state["tasks"]:
        claim = task.get("claim")
        if (
            task["status"] == "running"
            and claim is not None
            and int(claim["lease_expires_unix_ms"]) <= now
        ):
            expired.append(task["id"])
            task["attempts"].append(
                {
                    **claim,
                    "ended_unix_ms": now,
                    "outcome": "lease-expired",
                }
            )
            task["status"] = "ready"
            task["claim"] = None
            _append_event(
                state,
                "lease-expired",
                task_id=task["id"],
                host=claim["host"],
                now_ms=now,
            )
            host = claim["host"]
            if not any(
                other["status"] == "running" and (other.get("claim") or {}).get("host") == host
                for other in state["tasks"]
            ):
                state["hosts"][host]["intent"] = "available"
                state["hosts"][host]["reason"] = None
                state["hosts"][host]["updated_unix_ms"] = now
    refresh_dependencies(state)
    return expired


def _selection_key(task: dict[str, Any]) -> tuple[Any, ...]:
    decision_rate = float(task["decision_value"]) / float(task["expected_runtime_seconds"])
    return (
        int(task["priority"]),
        not bool(task["critical_path"]),
        -decision_rate,
        -float(task["expected_runtime_seconds"]),
        str(task["id"]),
    )


def claim_next(
    state: dict[str, Any],
    *,
    host: str,
    lease_seconds: float,
    now_ms: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    if host not in state["hosts"]:
        raise QueueError(f"unknown host {host}")
    if lease_seconds <= 0:
        raise QueueError("lease_seconds must be positive")
    now = unix_millis() if now_ms is None else now_ms
    expire_leases(state, now)
    refresh_dependencies(state)
    if any(
        task["status"] == "running" and (task.get("claim") or {}).get("host") == host
        for task in state["tasks"]
    ):
        return None
    ready = [
        task
        for task in state["tasks"]
        if task["status"] == "ready" and host in task["compatible_hosts"]
    ]
    if not ready:
        return None
    task = min(ready, key=_selection_key)
    token = secrets.token_hex(16)
    claim = {
        "host": host,
        "token": token,
        "claimed_unix_ms": now,
        "heartbeat_unix_ms": now,
        "lease_expires_unix_ms": now + int(lease_seconds * 1_000),
    }
    task["status"] = "running"
    task["claim"] = claim
    state["hosts"][host]["intent"] = "working"
    state["hosts"][host]["reason"] = task["id"]
    state["hosts"][host]["updated_unix_ms"] = now
    _append_event(state, "claimed", task_id=task["id"], host=host, now_ms=now)
    return copy.deepcopy(task)


def _require_claim(
    state: dict[str, Any],
    *,
    task_id: str,
    host: str,
    token: str,
) -> dict[str, Any]:
    task = _task_by_id(state, task_id)
    claim = task.get("claim")
    if task["status"] != "running" or claim is None:
        raise QueueError(f"task {task_id} is not running")
    if claim["host"] != host or claim["token"] != token:
        raise QueueError(f"task {task_id} claim token or host does not match")
    return task


def heartbeat(
    state: dict[str, Any],
    *,
    task_id: str,
    host: str,
    token: str,
    lease_seconds: float,
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    if lease_seconds <= 0:
        raise QueueError("lease_seconds must be positive")
    now = unix_millis() if now_ms is None else now_ms
    task = _require_claim(state, task_id=task_id, host=host, token=token)
    task["claim"]["heartbeat_unix_ms"] = now
    task["claim"]["lease_expires_unix_ms"] = now + int(lease_seconds * 1_000)
    return copy.deepcopy(task)


def finish_task(
    state: dict[str, Any],
    *,
    task_id: str,
    host: str,
    token: str,
    outcome: str,
    artifact: Optional[str] = None,
    error: Optional[str] = None,
    retry: bool = False,
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    if outcome not in {"completed", "failed", "cancelled"}:
        raise QueueError(f"unsupported outcome {outcome}")
    now = unix_millis() if now_ms is None else now_ms
    task = _require_claim(state, task_id=task_id, host=host, token=token)
    claim = task["claim"]
    attempt = {
        **claim,
        "ended_unix_ms": now,
        "outcome": outcome,
        "artifact": artifact,
        "error": error,
    }
    task["attempts"].append(attempt)
    task["claim"] = None
    task["status"] = "ready" if outcome == "failed" and retry else outcome
    task["result"] = (
        None
        if outcome == "failed" and retry
        else {
            "artifact": artifact,
            "error": error,
            "completed_unix_ms": now,
            "host": host,
        }
    )
    state["hosts"][host]["intent"] = "available"
    state["hosts"][host]["reason"] = None
    state["hosts"][host]["updated_unix_ms"] = now
    _append_event(
        state,
        "retry-ready" if retry and outcome == "failed" else outcome,
        task_id=task_id,
        host=host,
        detail={"artifact": artifact, "error": error},
        now_ms=now,
    )
    refresh_dependencies(state)
    return copy.deepcopy(task)


def _task_from_specification(
    state: dict[str, Any],
    specification: dict[str, Any],
    *,
    now_ms: int,
) -> dict[str, Any]:
    return {
        "id": specification.get("id"),
        "title": specification.get("title"),
        "experiment_id": specification.get("experiment_id"),
        "decision": specification.get("decision"),
        "workload_class": specification.get("workload_class"),
        "priority": specification.get("priority", 100),
        "decision_value": specification.get("decision_value", 1.0),
        "expected_runtime_seconds": specification.get("expected_runtime_seconds"),
        "critical_path": specification.get("critical_path", False),
        "decision_terminal": specification.get("decision_terminal", False),
        "compatible_hosts": specification.get("compatible_hosts", list(state["hosts"])),
        "dependencies": specification.get("dependencies", []),
        "command": specification.get("command"),
        "artifact_path": specification.get("artifact_path"),
        "stop_rule": specification.get("stop_rule"),
        "resources": specification.get(
            "resources", {"cpu_cores": 1, "memory_gib": 1.0, "uses_mlx": False}
        ),
        "status": "ready",
        "claim": None,
        "attempts": [],
        "result": None,
        "administrative_cancellation": None,
        "created_unix_ms": now_ms,
    }


def add_tasks(
    state: dict[str, Any],
    specifications: list[dict[str, Any]],
    now_ms: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Atomically validate and append a complete campaign task graph."""
    if not isinstance(specifications, list) or not specifications:
        raise QueueError("campaign task specification must be a nonempty list")
    if any(not isinstance(specification, dict) for specification in specifications):
        raise QueueError("every campaign task specification must be an object")

    requested_ids = [specification.get("id") for specification in specifications]
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in requested_ids):
        raise QueueError("every campaign task specification requires a nonempty id")
    if len(requested_ids) != len(set(requested_ids)):
        raise QueueError("campaign task specification contains duplicate task ids")
    existing_ids = {task["id"] for task in state["tasks"]}
    duplicates = sorted(existing_ids.intersection(requested_ids))
    if duplicates:
        raise QueueError(f"queue already contains campaign task ids: {duplicates}")

    now = unix_millis() if now_ms is None else now_ms
    candidate = copy.deepcopy(state)
    tasks = [
        _task_from_specification(candidate, specification, now_ms=now)
        for specification in specifications
    ]
    candidate["tasks"].extend(tasks)
    refresh_dependencies(candidate)
    validate_queue(candidate)
    for task in tasks:
        _append_event(candidate, "task-added", task_id=task["id"], now_ms=now)

    state.clear()
    state.update(candidate)
    return copy.deepcopy(tasks)


def add_task(
    state: dict[str, Any],
    specification: dict[str, Any],
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    return add_tasks(state, [specification], now_ms=now_ms)[0]


def campaign_task_specifications(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a reviewed campaign envelope without weakening queue validation."""
    if not isinstance(payload, dict):
        raise QueueError("campaign specification must be a JSON object")
    specifications = payload.get("tasks")
    if not isinstance(specifications, list):
        raise QueueError("campaign specification tasks must be a list")
    declared_count = payload.get("task_count")
    if declared_count is not None and declared_count != len(specifications):
        raise QueueError("campaign specification task_count does not match tasks")
    experiment_id = payload.get("experiment_id")
    if experiment_id is not None:
        _require_string(experiment_id, "campaign specification experiment_id")
        mismatched = sorted(
            str(specification.get("id"))
            for specification in specifications
            if isinstance(specification, dict)
            and specification.get("experiment_id") != experiment_id
        )
        if mismatched:
            raise QueueError(
                f"campaign specification contains tasks for another experiment: {mismatched}"
            )
    metadata = {
        "experiment_id": experiment_id,
        "task_count": len(specifications),
    }
    return specifications, metadata


def set_task_dependencies(
    state: dict[str, Any],
    *,
    task_id: str,
    dependencies: list[str],
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    task = _task_by_id(state, task_id)
    if task["status"] not in {"ready", "blocked"}:
        raise QueueError("dependencies may change only before a task starts")
    if len(dependencies) != len(set(dependencies)):
        raise QueueError("dependencies must not contain duplicates")
    if task_id in dependencies:
        raise QueueError("task must not depend on itself")
    unknown = set(dependencies) - {value["id"] for value in state["tasks"]}
    if unknown:
        raise QueueError(f"unknown dependencies: {sorted(unknown)}")
    task["dependencies"] = list(dependencies)
    refresh_dependencies(state)
    validate_queue(state)
    now = unix_millis() if now_ms is None else now_ms
    _append_event(
        state,
        "dependencies-updated",
        task_id=task_id,
        detail={"dependencies": dependencies},
        now_ms=now,
    )
    return copy.deepcopy(task)


def retry_failed_task(
    state: dict[str, Any],
    *,
    task_id: str,
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    task = _task_by_id(state, task_id)
    if task["status"] != "failed" or task.get("claim") is not None:
        raise QueueError("only an unclaimed failed task may be retried")
    task["status"] = "ready"
    task["result"] = None
    refresh_dependencies(state)
    validate_queue(state)
    now = unix_millis() if now_ms is None else now_ms
    _append_event(state, "manual-retry-ready", task_id=task_id, now_ms=now)
    return copy.deepcopy(task)


def cancel_pending_tasks(
    state: dict[str, Any],
    *,
    task_ids: list[str],
    actor: str,
    reason: str,
    now_ms: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Administratively cancel unclaimed work without erasing prior evidence."""
    if not task_ids:
        raise QueueError("at least one task id is required")
    if len(task_ids) != len(set(task_ids)):
        raise QueueError("task ids must not contain duplicates")
    if not actor.strip():
        raise QueueError("administrative cancellation requires an actor")
    if not reason.strip():
        raise QueueError("administrative cancellation requires a reason")

    tasks = [_task_by_id(state, task_id) for task_id in task_ids]
    for task in tasks:
        if task["status"] not in {"ready", "blocked", "failed"} or task.get("claim") is not None:
            raise QueueError(
                "only unclaimed ready, blocked, or failed tasks may be administratively cancelled"
            )

    now = unix_millis() if now_ms is None else now_ms
    cancelled: list[dict[str, Any]] = []
    for task in tasks:
        previous_status = str(task["status"])
        task["status"] = "cancelled"
        task["administrative_cancellation"] = {
            "actor": actor.strip(),
            "reason": reason.strip(),
            "cancelled_unix_ms": now,
            "previous_status": previous_status,
        }
        _append_event(
            state,
            "administratively-cancelled",
            task_id=task["id"],
            detail={
                "actor": actor.strip(),
                "reason": reason.strip(),
                "previous_status": previous_status,
            },
            now_ms=now,
        )
        cancelled.append(copy.deepcopy(task))
    refresh_dependencies(state)
    validate_queue(state)
    return cancelled


def correct_completion_time(
    state: dict[str, Any],
    *,
    task_id: str,
    completed_unix_ms: int,
    evidence: str,
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Correct an adopted external task's over-recorded completion time."""
    task = _task_by_id(state, task_id)
    if task["status"] != "completed" or not isinstance(task.get("result"), dict):
        raise QueueError("only a completed task may have its completion time corrected")
    if not isinstance(completed_unix_ms, int) or completed_unix_ms < 0:
        raise QueueError("completed_unix_ms must be a nonnegative integer")
    if not evidence.strip():
        raise QueueError("completion-time correction requires evidence")
    completed_attempts = [
        attempt
        for attempt in task["attempts"]
        if attempt.get("outcome") == "completed"
        and attempt.get("host") == task["result"].get("host")
    ]
    if len(completed_attempts) != 1:
        raise QueueError("completion-time correction requires one completed attempt")
    attempt = completed_attempts[0]
    old_completed_unix_ms = int(task["result"]["completed_unix_ms"])
    if int(attempt["ended_unix_ms"]) != old_completed_unix_ms:
        raise QueueError("completed attempt and result timestamps disagree")
    lower_bound = max(
        int(attempt["claimed_unix_ms"]),
        int(attempt["heartbeat_unix_ms"]),
    )
    if not lower_bound <= completed_unix_ms <= old_completed_unix_ms:
        raise QueueError(
            "corrected completion time must be between the last heartbeat and recorded completion"
        )
    attempt["ended_unix_ms"] = completed_unix_ms
    task["result"]["completed_unix_ms"] = completed_unix_ms
    validate_queue(state)
    _append_event(
        state,
        "completion-time-corrected",
        task_id=task_id,
        host=str(task["result"]["host"]),
        detail={
            "old_completed_unix_ms": old_completed_unix_ms,
            "completed_unix_ms": completed_unix_ms,
            "evidence": evidence.strip(),
        },
        now_ms=now_ms,
    )
    return copy.deepcopy(task)


def set_host_intent(
    state: dict[str, Any],
    *,
    host: str,
    intent: str,
    reason: Optional[str],
    now_ms: Optional[int] = None,
) -> dict[str, Any]:
    if host not in state["hosts"]:
        raise QueueError(f"unknown host {host}")
    if intent not in HOST_INTENT_STATUSES:
        raise QueueError(f"unsupported host intent {intent}")
    if intent == "intentionally-idle" and not reason:
        raise QueueError("intentionally-idle requires a reason")
    now = unix_millis() if now_ms is None else now_ms
    state["hosts"][host].update({"intent": intent, "reason": reason, "updated_unix_ms": now})
    _append_event(
        state,
        "host-intent",
        host=host,
        detail={"intent": intent, "reason": reason},
        now_ms=now,
    )
    return copy.deepcopy(state["hosts"][host])


def queue_summary(state: dict[str, Any], now_ms: Optional[int] = None) -> dict[str, Any]:
    working = copy.deepcopy(state)
    expire_leases(working, now_ms)
    refresh_dependencies(working)
    counts = {status: 0 for status in TASK_STATUSES}
    duplicate_running = 0
    decisions_completed = 0
    running_experiments: dict[str, int] = {}
    for task in working["tasks"]:
        counts[task["status"]] += 1
        if task["status"] == "running":
            experiment_id = task["experiment_id"]
            running_experiments[experiment_id] = running_experiments.get(experiment_id, 0) + 1
            if task["workload_class"] == "replica":
                duplicate_running += 1
        if task["status"] == "completed" and task["decision_terminal"]:
            decisions_completed += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": working["campaign_id"],
        "created_unix_ms": working["created_unix_ms"],
        "updated_unix_ms": working["updated_unix_ms"],
        "summary": {
            **counts,
            "total": len(working["tasks"]),
            "duplicate_running": duplicate_running,
            "decisions_completed": decisions_completed,
            "ready_decision_value": sum(
                float(task["decision_value"])
                for task in working["tasks"]
                if task["status"] == "ready"
            ),
        },
        "hosts": working["hosts"],
        "tasks": working["tasks"],
        "running_experiments": running_experiments,
    }


def _heartbeat_loop(
    queue_path: Path,
    task_id: str,
    host: str,
    token: str,
    lease_seconds: float,
    stop: threading.Event,
) -> None:
    interval = max(1.0, lease_seconds / 3.0)
    while not stop.wait(interval):
        try:
            with locked_queue(queue_path) as state:
                heartbeat(
                    state,
                    task_id=task_id,
                    host=host,
                    token=token,
                    lease_seconds=lease_seconds,
                )
        except QueueError:
            return


def _dispatch_command(
    state: dict[str, Any],
    task: dict[str, Any],
    *,
    host: str,
) -> list[str]:
    root = state["hosts"][host]["root"]
    task_command = shlex.join(task["command"])
    event_log = f"artifacts/cluster/queue-events/{state['campaign_id']}/{task['id']}-{host}.jsonl"
    wrapper = shlex.join(
        [
            sys.executable if host == "john1" else "python3",
            "tools/cluster_host_lock.py",
            "run",
            "--name",
            task["id"],
            "--wait-seconds",
            "300",
            "--event-log",
            event_log,
            "--",
        ]
    )
    shell_command = f"cd {shlex.quote(root)} && {wrapper} {task_command}"
    if host == "john1":
        return ["/bin/zsh", "-lc", shell_command]
    return ["/usr/bin/ssh", host, "/bin/zsh", "-lc", shlex.quote(shell_command)]


def dispatch_one(
    queue_path: Path,
    *,
    host: str,
    lease_seconds: float,
    dry_run: bool = False,
) -> int:
    with locked_queue(queue_path) as state:
        task = claim_next(state, host=host, lease_seconds=lease_seconds)
        if task is None:
            return 3
        command = _dispatch_command(state, task, host=host)
        token = task["claim"]["token"]
    if dry_run:
        with locked_queue(queue_path) as state:
            finish_task(
                state,
                task_id=task["id"],
                host=host,
                token=token,
                outcome="cancelled",
                error="dry-run dispatch",
            )
        print(json.dumps({"task": task, "command": command}, indent=2, sort_keys=True))
        return 0

    stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(queue_path, task["id"], host, token, lease_seconds, stop),
        daemon=True,
    )
    heartbeat_thread.start()
    return_code = 1
    launch_error: Optional[str] = None
    try:
        completed = subprocess.run(command, check=False)
        return_code = completed.returncode
    except OSError as error:
        launch_error = f"could not launch command: {error}"
    finally:
        stop.set()
        heartbeat_thread.join(timeout=max(1.0, lease_seconds / 2.0))
    with locked_queue(queue_path) as state:
        finish_task(
            state,
            task_id=task["id"],
            host=host,
            token=token,
            outcome="completed" if return_code == 0 else "failed",
            artifact=task["artifact_path"] if return_code == 0 else None,
            error=(None if return_code == 0 else launch_error or f"command exited {return_code}"),
        )
    return return_code


def dispatchable_hosts(
    state: dict[str, Any],
    *,
    hosts: list[str],
    now_ms: Optional[int] = None,
) -> list[str]:
    """Return requested hosts that can claim a ready compatible task now."""
    working = copy.deepcopy(state)
    expire_leases(working, now_ms)
    refresh_dependencies(working)
    unknown = set(hosts) - set(working["hosts"])
    if unknown:
        raise QueueError(f"unknown hosts: {sorted(unknown)}")
    busy = {
        task["claim"]["host"]
        for task in working["tasks"]
        if task["status"] == "running" and task.get("claim") is not None
    }
    ready = [task for task in working["tasks"] if task["status"] == "ready"]
    return [
        host
        for host in hosts
        if host not in busy and any(host in task["compatible_hosts"] for task in ready)
    ]


def run_coordinator(
    queue_path: Path,
    *,
    hosts: list[str],
    lease_seconds: float,
    poll_seconds: float,
    idle_timeout_seconds: float,
) -> int:
    """Continuously dispatch newly ready work across all requested hosts."""
    if lease_seconds <= 0 or poll_seconds <= 0 or idle_timeout_seconds < 0:
        raise QueueError("coordinator timing values are invalid")
    last_progress = time.monotonic()
    observed_failure = False
    futures: dict[str, Future[int]] = {}
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        while True:
            completed_dispatch = False
            for host, future in list(futures.items()):
                if not future.done():
                    continue
                try:
                    result = future.result()
                except Exception:
                    observed_failure = True
                else:
                    observed_failure |= result not in {0, 3}
                del futures[host]
                completed_dispatch = True
            if completed_dispatch:
                last_progress = time.monotonic()

            with locked_queue(queue_path) as state:
                expire_leases(state)
                refresh_dependencies(state)
                available = [
                    host for host in dispatchable_hosts(state, hosts=hosts) if host not in futures
                ]
                statuses = [task["status"] for task in state["tasks"]]
                requested_ready = [
                    task
                    for task in state["tasks"]
                    if task["status"] == "ready"
                    and any(host in task["compatible_hosts"] for host in hosts)
                ]
            if available:
                last_progress = time.monotonic()
                for host in available:
                    futures[host] = executor.submit(
                        dispatch_one,
                        queue_path,
                        host=host,
                        lease_seconds=lease_seconds,
                    )
                continue
            if not futures and not any(status == "running" for status in statuses):
                if requested_ready or any(status == "blocked" for status in statuses):
                    return 2
                return (
                    1 if observed_failure or any(status == "failed" for status in statuses) else 0
                )
            if (
                not futures
                and idle_timeout_seconds
                and time.monotonic() - last_progress >= idle_timeout_seconds
            ):
                return 4
            time.sleep(poll_seconds)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--allow-frozen-rollback",
        action="store_true",
        help="explicitly enable the frozen legacy executor during the bounded rollback window",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--campaign-id", required=True)
    initialize.add_argument("--force", action="store_true")

    add = subparsers.add_parser("add")
    add.add_argument("--spec", type=Path, required=True)

    install = subparsers.add_parser("install-spec")
    install.add_argument("--spec", type=Path, required=True)

    dependencies = subparsers.add_parser("set-dependencies")
    dependencies.add_argument("--task-id", required=True)
    dependencies.add_argument("--dependencies", nargs="*", default=[])

    retry_task = subparsers.add_parser("retry-task")
    retry_task.add_argument("--task-id", required=True)

    cancel_pending = subparsers.add_parser("cancel-pending")
    cancel_pending.add_argument("--task-id", action="append", required=True)
    cancel_pending.add_argument("--actor", required=True)
    cancel_pending.add_argument("--reason", required=True)

    correction = subparsers.add_parser("correct-completion-time")
    correction.add_argument("--task-id", required=True)
    correction.add_argument("--completed-unix-ms", type=int, required=True)
    correction.add_argument("--evidence", required=True)

    claim = subparsers.add_parser("claim")
    claim.add_argument("--host", required=True)
    claim.add_argument("--lease-seconds", type=float, default=900.0)

    beat = subparsers.add_parser("heartbeat")
    beat.add_argument("--task-id", required=True)
    beat.add_argument("--host", required=True)
    beat.add_argument("--token", required=True)
    beat.add_argument("--lease-seconds", type=float, default=900.0)

    for name in ("complete", "fail", "cancel"):
        finish = subparsers.add_parser(name)
        finish.add_argument("--task-id", required=True)
        finish.add_argument("--host", required=True)
        finish.add_argument("--token", required=True)
        finish.add_argument("--artifact")
        finish.add_argument("--error")
        if name == "fail":
            finish.add_argument("--retry", action="store_true")

    intent = subparsers.add_parser("set-host-intent")
    intent.add_argument("--host", required=True)
    intent.add_argument("--intent", choices=sorted(HOST_INTENT_STATUSES), required=True)
    intent.add_argument("--reason")

    dispatch = subparsers.add_parser("dispatch")
    dispatch.add_argument("--host", required=True)
    dispatch.add_argument("--lease-seconds", type=float, default=900.0)
    dispatch.add_argument("--dry-run", action="store_true")

    coordinator = subparsers.add_parser("run-coordinator")
    coordinator.add_argument(
        "--hosts",
        nargs="+",
        default=list(DEFAULT_HOSTS),
    )
    coordinator.add_argument("--lease-seconds", type=float, default=1800.0)
    coordinator.add_argument("--poll-seconds", type=float, default=5.0)
    coordinator.add_argument("--idle-timeout-seconds", type=float, default=0.0)

    subparsers.add_parser("status")
    subparsers.add_parser("validate")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        marker = args.queue.parent / LEGACY_FREEZE_MARKER
        if (
            marker.exists()
            and args.command_name not in {"status", "validate"}
            and not args.allow_frozen_rollback
        ):
            raise QueueError(
                "legacy queue is frozen for Bacalhau cutover; use "
                "--allow-frozen-rollback only during an authorized rollback"
            )
        if args.command_name == "init":
            if args.queue.exists() and not args.force:
                raise QueueError(f"queue already exists: {args.queue}")
            _atomic_write(args.queue, empty_queue(args.campaign_id))
            print(json.dumps(queue_summary(load_queue(args.queue)), indent=2, sort_keys=True))
            return 0
        if args.command_name == "validate":
            validate_queue(load_queue(args.queue))
            print(json.dumps({"valid": True, "queue": str(args.queue)}, sort_keys=True))
            return 0
        if args.command_name == "status":
            print(json.dumps(queue_summary(load_queue(args.queue)), indent=2, sort_keys=True))
            return 0
        if args.command_name == "dispatch":
            return dispatch_one(
                args.queue,
                host=args.host,
                lease_seconds=args.lease_seconds,
                dry_run=args.dry_run,
            )
        if args.command_name == "run-coordinator":
            return run_coordinator(
                args.queue,
                hosts=args.hosts,
                lease_seconds=args.lease_seconds,
                poll_seconds=args.poll_seconds,
                idle_timeout_seconds=args.idle_timeout_seconds,
            )

        with locked_queue(args.queue) as state:
            if args.command_name == "add":
                result = add_task(state, json.loads(args.spec.read_text()))
            elif args.command_name == "install-spec":
                specifications, metadata = campaign_task_specifications(
                    json.loads(args.spec.read_text())
                )
                installed = add_tasks(state, specifications)
                result = {
                    **metadata,
                    "installed_task_ids": [task["id"] for task in installed],
                }
            elif args.command_name == "set-dependencies":
                result = set_task_dependencies(
                    state,
                    task_id=args.task_id,
                    dependencies=args.dependencies,
                )
            elif args.command_name == "retry-task":
                result = retry_failed_task(
                    state,
                    task_id=args.task_id,
                )
            elif args.command_name == "cancel-pending":
                result = cancel_pending_tasks(
                    state,
                    task_ids=args.task_id,
                    actor=args.actor,
                    reason=args.reason,
                )
            elif args.command_name == "correct-completion-time":
                result = correct_completion_time(
                    state,
                    task_id=args.task_id,
                    completed_unix_ms=args.completed_unix_ms,
                    evidence=args.evidence,
                )
            elif args.command_name == "claim":
                result = claim_next(
                    state,
                    host=args.host,
                    lease_seconds=args.lease_seconds,
                )
            elif args.command_name == "heartbeat":
                result = heartbeat(
                    state,
                    task_id=args.task_id,
                    host=args.host,
                    token=args.token,
                    lease_seconds=args.lease_seconds,
                )
            elif args.command_name in {"complete", "fail", "cancel"}:
                outcome = {
                    "complete": "completed",
                    "fail": "failed",
                    "cancel": "cancelled",
                }[args.command_name]
                result = finish_task(
                    state,
                    task_id=args.task_id,
                    host=args.host,
                    token=args.token,
                    outcome=outcome,
                    artifact=args.artifact,
                    error=args.error,
                    retry=bool(getattr(args, "retry", False)),
                )
            else:
                result = set_host_intent(
                    state,
                    host=args.host,
                    intent=args.intent,
                    reason=args.reason,
                )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result is not None else 3
    except (QueueError, OSError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
