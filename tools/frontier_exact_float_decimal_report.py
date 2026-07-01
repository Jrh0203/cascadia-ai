#!/usr/bin/env python3
"""Render the frozen ADR 0106 exact-float Decimal control result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import frontier_arbitrary_precision_report as base

EXPERIMENT_ID = "complete-action-frontier-exact-float-decimal-control-v1"


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    *,
    combined_report: dict[str, Any],
    source_identity: dict[str, Any],
    replay_summary: dict[str, Any],
    campaign_summary: dict[str, Any],
) -> str:
    if combined_report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("combined report has the wrong experiment ID")
    combined = combined_report["scientific"]
    comparisons = {
        int(report["group_index"]): report
        for report in replay_summary["reports"]
    }
    lines = [
        "# Complete-Action Frontier Exact-Float Decimal Control V1 Result",
        "",
        f"Classification: `{combined['classification']}`.",
        "",
        "ADR 0106 preserved every frozen fractional expected-rank bit with "
        "`Decimal.from_float`, then solved the first 24 scale-16 box optima "
        "with the independent 96-digit breakpoint active-set derivation. "
        "The scientific path did not call the float64 analytic or projected "
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
        comparison = comparisons[group_index]
        lines.append(
            f"| {group_index} | "
            f"{base._canonical_host(str(comparison['origin_host']))} | "
            f"{base._canonical_host(str(comparison['replay_host']))} | "
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
    for host in sorted(base.EXPECTED_HOSTS):
        lines.append(
            f"| {host} | {campaign_summary['host_tasks'][host]} | "
            f"{campaign_summary['host_seconds'][host]:.2f} | "
            f"{campaign_summary['final_capacities'][host]} |"
        )
    lines.extend(["", "## Authorized Successor", ""])
    if combined["classification"] == "frozen_optimizer_hyperparameters_insufficient":
        lines.append(
            "The independent exact-float numerical control passes. ADR 0103 "
            "therefore authorizes exactly one calibrated local optimizer "
            "mechanism before any representation change or full trainer."
        )
    else:
        lines.append(
            "The exact-float numerical control remains invalid. No optimizer "
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
    base.EXPERIMENT_ID = EXPERIMENT_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, action="append", required=True)
    parser.add_argument("--replay-comparison", type=Path, action="append", required=True)
    parser.add_argument("--scheduler-state", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = base.validate_scheduler_state(base.load_json(args.scheduler_state))
    markdown = render_markdown(
        combined_report=base.load_json(args.combined),
        source_identity=base.validate_source_identities(args.source_identity),
        replay_summary=base.validate_replay_comparisons(
            args.replay_comparison
        ),
        campaign_summary=base.summarize_campaign_events(
            state,
            args.event_log,
        ),
    )
    write_text_atomic(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
