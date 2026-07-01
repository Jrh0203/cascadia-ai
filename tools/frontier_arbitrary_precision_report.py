#!/usr/bin/env python3
"""Render and validate the frozen ADR 0105 arbitrary-precision result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "complete-action-frontier-arbitrary-precision-control-v1"
EXPECTED_HOSTS = {"john1", "john2", "john3", "john4"}
EXPECTED_GROUPS = set(range(24))
PHYSICAL_CLUSTER_CORES = 40


def _canonical_host(value: str) -> str:
    return "john1" if value.lower().startswith("johns-mac-mini") else value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return value


def validate_source_identities(paths: list[Path]) -> dict[str, Any]:
    reports = [load_json(path) for path in paths]
    hosts = {_canonical_host(str(report["host"])) for report in reports}
    bundles = {str(report["bundle_sha256"]) for report in reports}
    file_counts = {int(report["files"]) for report in reports}
    if hosts != EXPECTED_HOSTS:
        raise ValueError(f"source identity hosts differ: {sorted(hosts)}")
    if len(bundles) != 1 or len(file_counts) != 1:
        raise ValueError("cluster MLX source bundles are not identical")
    return {
        "hosts": sorted(hosts),
        "files": next(iter(file_counts)),
        "bundle_sha256": next(iter(bundles)),
    }


def validate_replay_comparisons(paths: list[Path]) -> dict[str, Any]:
    reports = [load_json(path) for path in paths]
    groups = {int(report["group_index"]) for report in reports}
    if len(reports) != 24 or groups != EXPECTED_GROUPS:
        raise ValueError(f"replay comparison groups differ: {sorted(groups)}")
    if not all(report.get("scientific_payload_identical") is True for report in reports):
        raise ValueError("one or more ADR 0105 replay payloads differ")
    if not all(
        _canonical_host(str(report["origin_host"]))
        != _canonical_host(str(report["replay_host"]))
        for report in reports
    ):
        raise ValueError("one or more ADR 0105 replays used the origin host")
    return {"reports": sorted(reports, key=lambda report: int(report["group_index"]))}


def validate_scheduler_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("scheduler state has the wrong experiment ID")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 48:
        raise ValueError("scheduler state does not contain 48 tasks")
    if not all(task.get("status") == "done" for task in tasks.values()):
        raise ValueError("scheduler state contains incomplete tasks")
    for group_index in EXPECTED_GROUPS:
        origin = tasks[f"origin-{group_index:02d}"]
        replay = tasks[f"replay-{group_index:02d}"]
        if origin["host"] == replay["host"]:
            raise ValueError("scheduler replay used the origin host")
    return state


def summarize_campaign_events(
    state: dict[str, Any],
    event_log: Path,
) -> dict[str, Any]:
    tasks = list(state["tasks"].values())
    origins = [task for task in tasks if task["kind"] == "origin"]
    replays = [task for task in tasks if task["kind"] == "replay"]
    origin_start = min(float(task["started_unix_seconds"]) for task in origins)
    origin_end = max(float(task["ended_unix_seconds"]) for task in origins)
    campaign_start = min(float(task["started_unix_seconds"]) for task in tasks)
    campaign_end = max(float(task["ended_unix_seconds"]) for task in tasks)
    origin_seconds = sum(float(task["elapsed_seconds"]) for task in origins)
    replay_seconds = sum(float(task["elapsed_seconds"]) for task in replays)
    total_seconds = origin_seconds + replay_seconds
    host_seconds = {host: 0.0 for host in sorted(EXPECTED_HOSTS)}
    host_tasks = {host: 0 for host in sorted(EXPECTED_HOSTS)}
    for task in tasks:
        host = _canonical_host(str(task["host"]))
        host_seconds[host] += float(task["elapsed_seconds"])
        host_tasks[host] += 1

    events = [
        json.loads(line)
        for line in event_log.read_text().splitlines()
        if line.strip()
    ]
    snapshots = sorted(
        (
            event
            for event in events
            if event.get("event") == "snapshot"
        ),
        key=lambda event: float(event["unix_seconds"]),
    )
    idle_slot_seconds = 0.0
    maximum_active = 0
    for index, snapshot in enumerate(snapshots):
        start = float(snapshot["unix_seconds"])
        end = (
            float(snapshots[index + 1]["unix_seconds"])
            if index + 1 < len(snapshots)
            else campaign_end
        )
        duration = max(0.0, end - start)
        maximum_active = max(
            maximum_active,
            sum(int(value) for value in snapshot["active"].values()),
        )
        for host in EXPECTED_HOSTS:
            free = max(
                0,
                int(snapshot["capacity"][host])
                - int(snapshot["active"][host]),
            )
            compatible = int(snapshot["compatible_ready"][host])
            idle_slot_seconds += min(free, compatible) * duration
    campaign_wall = campaign_end - campaign_start
    return {
        "origin_makespan_seconds": origin_end - origin_start,
        "campaign_wall_seconds": campaign_wall,
        "scheduled_process_seconds": total_seconds,
        "confirmation_compute_fraction": replay_seconds / total_seconds,
        "mean_active_group_processes": total_seconds / campaign_wall,
        "group_process_core_occupancy": (
            total_seconds / (PHYSICAL_CLUSTER_CORES * campaign_wall)
        ),
        "maximum_active_group_processes": maximum_active,
        "idle_slot_seconds_with_compatible_work": idle_slot_seconds,
        "host_seconds": host_seconds,
        "host_tasks": host_tasks,
        "final_capacities": {
            host: int(state["hosts"][host]["capacity"])
            for host in EXPECTED_HOSTS
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    *,
    combined_report: dict[str, Any],
    source_identity: dict[str, Any],
    replay_summary: dict[str, Any],
    scheduler_state: dict[str, Any],
    campaign_summary: dict[str, Any],
) -> str:
    if combined_report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("combined report has the wrong experiment ID")
    combined = combined_report["scientific"]
    comparison_by_group = {
        int(report["group_index"]): report
        for report in replay_summary["reports"]
    }
    lines = [
        "# Complete-Action Frontier Arbitrary-Precision Control V1 Result",
        "",
        f"Classification: `{combined['classification']}`.",
        "",
        "ADR 0105 attempted to reconstruct the frozen first 24 scale-16 box "
        "optima with 96-digit Decimal arithmetic and a breakpoint active-set "
        "derivation. It did not call the float64 analytic or projected "
        "solvers. Sealed test, gameplay, new teacher compute, cloud, and "
        "external compute remained closed.",
        "",
        "## Numerical Result",
        "",
        f"- Aggregate target recall: "
        f"{_percent(combined['aggregate']['target_positive_recall'])}.",
        f"- Exact target sets: "
        f"{_percent(combined['aggregate']['target_set_exact_fraction'])}.",
        f"- Maximum normalization residual: "
        f"`{combined['maximum_normalization_residual']}`.",
        f"- Maximum Decimal KKT violation: "
        f"`{combined['maximum_kkt_violation']}`.",
        f"- Maximum objective difference from frozen float64 analytic: "
        f"`{combined['maximum_objective_difference']}`.",
        f"- Maximum offset difference from frozen float64 analytic: "
        f"`{combined['maximum_offset_difference']}`.",
        "",
        "| Group | Origin | Replay | Recall | Exact | Objective difference | KKT |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for group in combined["groups"]:
        group_index = int(group["group_index"])
        comparison = comparison_by_group[group_index]
        lines.append(
            f"| {group_index} | "
            f"{_canonical_host(str(comparison['origin_host']))} | "
            f"{_canonical_host(str(comparison['replay_host']))} | "
            f"{_percent(float(group['target_positive_recall']))} | "
            f"{'yes' if group['target_set_exact'] else 'no'} | "
            f"`{group['objective_difference']}` | "
            f"`{group['kkt_violation']}` |"
        )
    lines.extend(
        [
            "",
            "## Frozen Gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for name, passed in sorted(combined["gates"].items()):
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Failure Cause",
            "",
            "The preregistration specified exact integer rank conversion, but "
            "the frozen expected-rank targets are fractional float64 values. "
            "The implementation therefore truncated target ranks before "
            "computing probabilities. Normalization and KKT residuals are "
            "excellent for that altered objective, but 23 of 24 groups differ "
            "from the frozen float64 objective. This is an input-conversion "
            "pipeline error, not evidence against the active-set derivation.",
            "",
            "## Dynamic Cluster Throughput",
            "",
            f"- Origin critical path: "
            f"{campaign_summary['origin_makespan_seconds']:.2f} seconds.",
            f"- End-to-end origin plus confirmation: "
            f"{campaign_summary['campaign_wall_seconds']:.2f} seconds.",
            f"- Scheduled group-process time: "
            f"{campaign_summary['scheduled_process_seconds']:.2f} seconds.",
            f"- Confirmation compute fraction: "
            f"{campaign_summary['confirmation_compute_fraction']:.2%}.",
            f"- Mean active group processes: "
            f"{campaign_summary['mean_active_group_processes']:.2f}; "
            f"peak: {campaign_summary['maximum_active_group_processes']}.",
            f"- Group-process occupancy relative to 40 physical cores: "
            f"{_percent(campaign_summary['group_process_core_occupancy'])}.",
            f"- Idle process-slot seconds while compatible work was queued: "
            f"{campaign_summary['idle_slot_seconds_with_compatible_work']:.2f}.",
            "- Duplicate discovery fraction: 0.00%; every origin group was "
            "unique and every duplicate was an explicit cross-host replay.",
            f"- Source identity: {source_identity['files']} files, "
            f"`{source_identity['bundle_sha256']}`, identical on "
            f"{', '.join(source_identity['hosts'])}.",
            "",
            "| Host | Tasks | Scheduled seconds | Final capacity |",
            "|---|---:|---:|---:|",
        ]
    )
    for host in sorted(EXPECTED_HOSTS):
        lines.append(
            f"| {host} | {campaign_summary['host_tasks'][host]} | "
            f"{campaign_summary['host_seconds'][host]:.2f} | "
            f"{campaign_summary['final_capacities'][host]} |"
        )
    lines.extend(["", "## Authorized Successor", ""])
    if combined["classification"] == "frozen_optimizer_hyperparameters_insufficient":
        lines.append(
            "The independent numerical control passes. The frozen ADR 0103 "
            "evidence therefore authorizes exactly one calibrated local "
            "optimizer mechanism before any representation change or full "
            "trainer."
        )
    else:
        lines.append(
            "Preregister one corrected replay using `Decimal.from_float` for "
            "every frozen expected-rank value. Reuse the active-set method, "
            "dynamic scheduler, and frozen evidence unchanged. No optimizer "
            "or model treatment is authorized."
        )
    lines.append("")
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, action="append", required=True)
    parser.add_argument("--replay-comparison", type=Path, action="append", required=True)
    parser.add_argument("--scheduler-state", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scheduler_state = validate_scheduler_state(load_json(args.scheduler_state))
    markdown = render_markdown(
        combined_report=load_json(args.combined),
        source_identity=validate_source_identities(args.source_identity),
        replay_summary=validate_replay_comparisons(args.replay_comparison),
        scheduler_state=scheduler_state,
        campaign_summary=summarize_campaign_events(
            scheduler_state,
            args.event_log,
        ),
    )
    write_text_atomic(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
