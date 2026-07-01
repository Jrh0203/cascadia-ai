#!/usr/bin/env python3
"""Render the frozen ADR 0107 calibrated optimizer Stage 1 result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import frontier_arbitrary_precision_report as base

EXPERIMENT_ID = "complete-action-frontier-calibrated-monotone-adamw-v1"


def validate_replays(paths: list[Path]) -> dict[str, Any]:
    reports = [base.load_json(path) for path in paths]
    groups = {int(report["group_index"]) for report in reports}
    if len(reports) != 24 or groups != set(range(24)):
        raise ValueError("ADR 0107 free replay set is incomplete")
    if not all(report["scientific_payload_identical"] is True for report in reports):
        raise ValueError("one or more ADR 0107 free replays differ")
    if not all(
        base._canonical_host(str(report["origin_host"]))
        != base._canonical_host(str(report["replay_host"]))
        for report in reports
    ):
        raise ValueError("one or more ADR 0107 replays used the origin host")
    return {"reports": sorted(reports, key=lambda report: int(report["group_index"]))}


def summarize_failed_attempts(event_log: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in event_log.read_text().splitlines()
        if line.strip()
    ]
    failed = [
        event
        for event in events
        if event.get("event") == "finished"
        and int(event.get("return_code", 0)) != 0
    ]
    return {
        "count": len(failed),
        "seconds": sum(float(event["elapsed_seconds"]) for event in failed),
        "task_ids": sorted(str(event["task_id"]) for event in failed),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    *,
    free_report: dict[str, Any],
    source_identity: dict[str, Any],
    replay_summary: dict[str, Any],
    campaign_summary: dict[str, Any],
    failed_attempts: dict[str, Any],
) -> str:
    if free_report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("free report has the wrong experiment ID")
    free = free_report["scientific"]
    failed_groups = [
        group
        for group in free["groups"]
        if group["failure"] is not None
    ]
    lines = [
        "# Complete-Action Frontier Calibrated Monotone AdamW V1 Result",
        "",
        f"Classification: `{free['classification']}`.",
        "",
        "ADR 0107 Stage 1 applied one analytically capped, same-batch "
        "backtracked AdamW mechanism to the frozen 24 free-residual groups. "
        "Neural Stage 2 did not launch because the Stage 1 pipeline gate did "
        "not pass. Sealed test, gameplay, new teacher compute, cloud, and "
        "external compute remained closed.",
        "",
        "## Free-Residual Result",
        "",
        "| Checkpoint | Recall | Exact sets | Mean objective |",
        "|---|---:|---:|---:|",
        f"| 120 updates | "
        f"{_percent(free['aggregate_at_120']['target_positive_recall'])} | "
        f"{_percent(free['aggregate_at_120']['target_set_exact_fraction'])} | "
        f"{free['aggregate_at_120']['mean_objective']:.6f} |",
        f"| terminal | "
        f"{_percent(free['aggregate']['target_positive_recall'])} | "
        f"{_percent(free['aggregate']['target_set_exact_fraction'])} | "
        f"{free['aggregate']['mean_objective']:.6f} |",
        "",
        "The terminal strength gate passed, but the pipeline gate failed. "
        "Five groups reached float32 numerical saturation before the frozen "
        "1,200-update count and could not accept another strictly "
        "loss-nonincreasing proposal within 16 backtracks.",
        "",
        "| Group | Accepted updates | Recall | Exact | Min rate | Backtracks |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for group in failed_groups:
        optimizer = group["optimizer"]
        final = group["final"]
        lines.append(
            f"| {group['group_index']} | "
            f"{optimizer['accepted_updates']} | "
            f"{_percent(final['target_positive_recall'])} | "
            f"{'yes' if final['target_set_exact'] else 'no'} | "
            f"`{optimizer['minimum_accepted_rate']:.3e}` | "
            f"{optimizer['total_backtracks']} |"
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
    for name, passed in sorted(free["gates"].items()):
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "All 24 origin/replay scientific payloads were bit-identical and "
            "all resource gates passed.",
            "",
            "## Cluster Throughput",
            "",
            f"- Successful origin-plus-confirmation wall time: "
            f"{campaign_summary['campaign_wall_seconds']:.2f} seconds.",
            f"- Successful scheduled process time: "
            f"{campaign_summary['scheduled_process_seconds']:.2f} seconds.",
            f"- Mean active MLX processes: "
            f"{campaign_summary['mean_active_group_processes']:.2f}; "
            f"peak: {campaign_summary['maximum_active_group_processes']}.",
            f"- Idle process-slot seconds while compatible work was queued: "
            f"{campaign_summary['idle_slot_seconds_with_compatible_work']:.2f}; "
            "this includes the deliberate halt while the report bug was fixed, "
            "tested, synchronized, and source identity was refrozen.",
            f"- Pre-artifact implementation failures: "
            f"{failed_attempts['count']} tasks, "
            f"{failed_attempts['seconds']:.2f} process-seconds, caused by a "
            "report-field typo and rerun after source refreeze.",
            "- Duplicate discovery fraction among retained artifacts: 0.00%; "
            "all duplicate work was explicit cross-host confirmation.",
            f"- Source identity: {source_identity['files']} files, "
            f"`{source_identity['bundle_sha256']}`, identical on "
            f"{', '.join(source_identity['hosts'])}.",
            "",
            "| Host | Tasks | Scheduled seconds |",
            "|---|---:|---:|",
        ]
    )
    for host in sorted(base.EXPECTED_HOSTS):
        lines.append(
            f"| {host} | {campaign_summary['host_tasks'][host]} | "
            f"{campaign_summary['host_seconds'][host]:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Authorized Successor",
            "",
            "Preregister a stop-rule repair for only the five saturated "
            "groups. Treat exhausted finite backtracking as numerical "
            "convergence rather than requiring meaningless extra updates, "
            "then recombine with the frozen 19 completed groups. Neural work "
            "remains unauthorized until that repaired Stage 1 pipeline passes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, action="append", required=True)
    parser.add_argument("--replay-comparison", type=Path, action="append", required=True)
    parser.add_argument("--scheduler-state", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = base.validate_scheduler_state(base.load_json(args.scheduler_state))
    markdown = render_markdown(
        free_report=base.load_json(args.free),
        source_identity=base.validate_source_identities(args.source_identity),
        replay_summary=validate_replays(args.replay_comparison),
        campaign_summary=base.summarize_campaign_events(
            state,
            args.event_log,
        ),
        failed_attempts=summarize_failed_attempts(args.event_log),
    )
    write_text_atomic(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
