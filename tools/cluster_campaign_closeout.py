#!/usr/bin/env python3
"""Audit cluster scheduling and utilization for one research campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cluster_campaign_telemetry as telemetry
import cluster_research_queue as queue


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping half-open millisecond intervals."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_seconds(intervals: list[tuple[int, int]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals)) / 1_000.0


def contains(intervals: list[tuple[int, int]], timestamp: int) -> bool:
    return any(start <= timestamp < end for start, end in intervals)


def subtract_intervals(
    source: list[tuple[int, int]],
    occupied: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Subtract occupied intervals from source intervals."""
    result: list[tuple[int, int]] = []
    blockers = merge_intervals(occupied)
    for start, end in merge_intervals(source):
        cursor = start
        for blocked_start, blocked_end in blockers:
            if blocked_end <= cursor:
                continue
            if blocked_start >= end:
                break
            if blocked_start > cursor:
                result.append((cursor, min(blocked_start, end)))
            cursor = max(cursor, blocked_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def _task_completion_ms(task: dict[str, Any]) -> int | None:
    result = task.get("result")
    if task["status"] == "completed" and isinstance(result, dict):
        completed = result.get("completed_unix_ms")
        if isinstance(completed, int):
            return completed
    completed_attempts = [
        int(attempt["ended_unix_ms"])
        for attempt in task["attempts"]
        if attempt.get("outcome") == "completed"
    ]
    return max(completed_attempts, default=None)


def _dependency_ready_ms(
    task: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
) -> int | None:
    ready = int(task["created_unix_ms"])
    for dependency_id in task["dependencies"]:
        completed = _task_completion_ms(tasks[dependency_id])
        if completed is None:
            return None
        ready = max(ready, completed)
    return ready


def task_ready_intervals(
    task: dict[str, Any],
    *,
    tasks: dict[str, dict[str, Any]],
    start_unix_ms: int,
    end_unix_ms: int,
) -> list[tuple[int, int]]:
    """Return intervals where a task was claimable but unclaimed."""
    ready = _dependency_ready_ms(task, tasks)
    if ready is None:
        return []
    cursor = max(ready, start_unix_ms)
    intervals: list[tuple[int, int]] = []
    attempts = sorted(task["attempts"], key=lambda attempt: attempt["claimed_unix_ms"])
    for index, attempt in enumerate(attempts):
        claimed = min(max(int(attempt["claimed_unix_ms"]), start_unix_ms), end_unix_ms)
        if claimed > cursor:
            intervals.append((cursor, claimed))
        outcome = attempt.get("outcome")
        ended = min(max(int(attempt["ended_unix_ms"]), start_unix_ms), end_unix_ms)
        has_later_attempt = index + 1 < len(attempts)
        if outcome in {"lease-expired"} or (
            outcome == "failed" and (has_later_attempt or task["status"] == "ready")
        ):
            cursor = max(cursor, ended)
        else:
            cursor = end_unix_ms
            break

    claim = task.get("claim")
    if claim is not None and task["status"] == "running":
        claimed = min(max(int(claim["claimed_unix_ms"]), start_unix_ms), end_unix_ms)
        if claimed > cursor:
            intervals.append((cursor, claimed))
        cursor = end_unix_ms
    elif task["status"] == "ready" and cursor < end_unix_ms:
        intervals.append((cursor, end_unix_ms))
    return merge_intervals(intervals)


def task_productive_intervals(
    task: dict[str, Any],
    *,
    start_unix_ms: int,
    end_unix_ms: int,
) -> list[tuple[str, int, int]]:
    intervals = []
    for attempt in task["attempts"]:
        start = max(int(attempt["claimed_unix_ms"]), start_unix_ms)
        end = min(int(attempt["ended_unix_ms"]), end_unix_ms)
        if end > start:
            intervals.append((str(attempt["host"]), start, end))
    claim = task.get("claim")
    if task["status"] == "running" and claim is not None:
        start = max(int(claim["claimed_unix_ms"]), start_unix_ms)
        if end_unix_ms > start:
            intervals.append((str(claim["host"]), start, end_unix_ms))
    return intervals


def _healthy_idle_seconds(
    samples: list[dict[str, Any]],
    *,
    host: str,
    start_unix_ms: int,
    end_unix_ms: int,
    ready: list[tuple[int, int]],
    productive: list[tuple[int, int]],
) -> float:
    window = [
        sample
        for sample in samples
        if start_unix_ms <= int(sample["timestamp_unix_ms"]) <= end_unix_ms
    ]
    total_ms = 0
    for index, sample in enumerate(window):
        timestamp = int(sample["timestamp_unix_ms"])
        next_timestamp = (
            min(int(window[index + 1]["timestamp_unix_ms"]), end_unix_ms)
            if index + 1 < len(window)
            else min(timestamp + 30_000, end_unix_ms)
        )
        node = next(
            (node for node in sample["nodes"] if str(node["node_id"]) == host),
            None,
        )
        if (
            node is not None
            and bool(node["reachable"])
            and contains(ready, timestamp)
            and not contains(productive, timestamp)
        ):
            total_ms += max(0, next_timestamp - timestamp)
    return total_ms / 1_000.0


def audit_campaign(
    queue_state: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    start_unix_ms: int,
    end_unix_ms: int,
    cores: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a mechanical scheduling and utilization closeout."""
    if end_unix_ms < start_unix_ms:
        raise ValueError("campaign end precedes campaign start")
    queue.validate_queue(queue_state)
    tasks = {task["id"]: task for task in queue_state["tasks"]}
    productive_by_host: dict[str, list[tuple[int, int]]] = {
        host: [] for host in queue_state["hosts"]
    }
    ready_by_host: dict[str, list[tuple[int, int]]] = {host: [] for host in queue_state["hosts"]}
    duplicate_process_ms = 0
    total_process_ms = 0

    for task in queue_state["tasks"]:
        ready = task_ready_intervals(
            task,
            tasks=tasks,
            start_unix_ms=start_unix_ms,
            end_unix_ms=end_unix_ms,
        )
        for host in task["compatible_hosts"]:
            ready_by_host[host].extend(ready)
        for host, start, end in task_productive_intervals(
            task,
            start_unix_ms=start_unix_ms,
            end_unix_ms=end_unix_ms,
        ):
            productive_by_host[host].append((start, end))
            duration = end - start
            total_process_ms += duration
            if task["workload_class"] == "replica":
                duplicate_process_ms += duration

    telemetry_report = telemetry.summarize(
        samples,
        start_unix_ms=start_unix_ms,
        end_unix_ms=end_unix_ms,
        cores=cores,
    )
    host_reports = {}
    total_healthy_idle = 0.0
    for host in queue_state["hosts"]:
        productive = merge_intervals(productive_by_host[host])
        ready = merge_intervals(ready_by_host[host])
        queued_idle = subtract_intervals(ready, productive)
        healthy_idle = _healthy_idle_seconds(
            samples,
            host=host,
            start_unix_ms=start_unix_ms,
            end_unix_ms=end_unix_ms,
            ready=ready,
            productive=productive,
        )
        total_healthy_idle += healthy_idle
        host_reports[host] = {
            "productive_seconds": interval_seconds(productive),
            "compatible_work_ready_seconds": interval_seconds(ready),
            "idle_with_compatible_work_queued_seconds": interval_seconds(queued_idle),
            "healthy_idle_with_compatible_work_queued_seconds": healthy_idle,
            "completed_tasks": sum(
                1
                for task in queue_state["tasks"]
                if task["status"] == "completed"
                and isinstance(task.get("result"), dict)
                and task["result"].get("host") == host
            ),
            "failed_or_expired_attempts": sum(
                1
                for task in queue_state["tasks"]
                for attempt in task["attempts"]
                if attempt["host"] == host and attempt["outcome"] in {"failed", "lease-expired"}
            ),
            "telemetry": telemetry_report["nodes"].get(host),
        }

    duration_seconds = (end_unix_ms - start_unix_ms) / 1_000.0
    decisions_completed = sum(
        1
        for task in queue_state["tasks"]
        if task["status"] == "completed" and bool(task["decision_terminal"])
    )
    return {
        "schema_version": 1,
        "campaign_id": queue_state["campaign_id"],
        "start_unix_ms": start_unix_ms,
        "end_unix_ms": end_unix_ms,
        "campaign_wall_seconds": duration_seconds,
        "tasks": {
            status: sum(1 for task in queue_state["tasks"] if task["status"] == status)
            for status in sorted(queue.TASK_STATUSES)
        },
        "decisions_completed": decisions_completed,
        "decisions_per_campaign_hour": (
            decisions_completed / (duration_seconds / 3_600.0) if duration_seconds > 0 else 0.0
        ),
        "scheduled_process_seconds": total_process_ms / 1_000.0,
        "duplicate_process_seconds": duplicate_process_ms / 1_000.0,
        "duplicate_compute_fraction": (
            duplicate_process_ms / total_process_ms if total_process_ms else 0.0
        ),
        "healthy_idle_with_compatible_work_queued_seconds": total_healthy_idle,
        "hosts": host_reports,
        "telemetry": telemetry_report,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--start-unix-ms", type=int, required=True)
    parser.add_argument("--end-unix-ms", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_campaign(
        queue.load_queue(args.queue),
        telemetry.load_jsonl(args.telemetry),
        start_unix_ms=args.start_unix_ms,
        end_unix_ms=args.end_unix_ms,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
