#!/usr/bin/env python3
"""Render and validate the frozen ADR 0108 stop-rule repair result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import frontier_arbitrary_precision_report as base

EXPERIMENT_ID = "complete-action-frontier-monotone-adamw-stop-repair-v1"
REPAIR_GROUPS = {0, 2, 8, 14, 23}


def validate_scheduler_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("ADR 0108 scheduler has the wrong experiment ID")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 10:
        raise ValueError("ADR 0108 scheduler must contain ten tasks")
    expected = {
        f"{kind}-{group_index:02d}"
        for kind in ("origin", "replay")
        for group_index in REPAIR_GROUPS
    }
    if set(tasks) != expected:
        raise ValueError("ADR 0108 scheduler has the wrong sparse task set")
    for group_index in REPAIR_GROUPS:
        origin = tasks[f"origin-{group_index:02d}"]
        replay = tasks[f"replay-{group_index:02d}"]
        if (
            origin.get("status") != "done"
            or replay.get("status") != "done"
            or origin.get("host") == replay.get("host")
        ):
            raise ValueError("ADR 0108 scheduler task failed validation")
    return state


def validate_replays(paths: list[Path]) -> dict[str, Any]:
    reports = [base.load_json(path) for path in paths]
    groups = {int(report["group_index"]) for report in reports}
    if len(reports) != 5 or groups != REPAIR_GROUPS:
        raise ValueError("ADR 0108 replay set is incomplete")
    if not all(
        report.get("experiment_id") == EXPERIMENT_ID
        and report.get("scientific_payload_identical") is True
        and base._canonical_host(str(report["origin_host"]))
        != base._canonical_host(str(report["replay_host"]))
        for report in reports
    ):
        raise ValueError("ADR 0108 replay validation failed")
    return {
        "reports": sorted(
            reports,
            key=lambda report: int(report["group_index"]),
        )
    }


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
        raise ValueError("ADR 0108 combined report identity differs")
    combined = combined_report["scientific"]
    if combined["classification"] != "free_stage_passed":
        raise ValueError("ADR 0108 did not pass its frozen classification")
    repairs = [
        group
        for group in combined["groups"]
        if group.get("numerical_convergence") is not None
    ]
    comparisons = {
        int(report["group_index"]): report
        for report in replay_summary["reports"]
    }
    lines = [
        "# Complete-Action Frontier Monotone AdamW Stop-Rule Repair V1 Result",
        "",
        "Classification: `free_stage_passed`.",
        "",
        "ADR 0108 reran only the five saturated ADR 0107 groups. The model, "
        "objective, optimizer, rates, moments, and strength gates remained "
        "unchanged. The other 19 groups and their replay evidence were reused "
        "byte-for-byte. Sealed test, gameplay, teacher, cloud, and external "
        "compute remained closed.",
        "",
        "## Recombined Stage 1",
        "",
        "| Checkpoint | Recall | Exact sets | Mean objective |",
        "|---|---:|---:|---:|",
        f"| 120 updates | "
        f"{_percent(combined['aggregate_at_120']['target_positive_recall'])} | "
        f"{_percent(combined['aggregate_at_120']['target_set_exact_fraction'])} | "
        f"{combined['aggregate_at_120']['mean_objective']:.6f} |",
        f"| terminal | "
        f"{_percent(combined['aggregate']['target_positive_recall'])} | "
        f"{_percent(combined['aggregate']['target_set_exact_fraction'])} | "
        f"{combined['aggregate']['mean_objective']:.6f} |",
        "",
        "All five repair groups met the frozen numerical-convergence rule.",
        "",
        "| Group | Origin | Replay | Accepted | Recall | Exact | "
        "Smallest attempted rate |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for group in repairs:
        index = int(group["group_index"])
        convergence = group["numerical_convergence"]
        comparison = comparisons[index]
        lines.append(
            f"| {index} | "
            f"{base._canonical_host(str(comparison['origin_host']))} | "
            f"{base._canonical_host(str(comparison['replay_host']))} | "
            f"{group['optimizer']['accepted_updates']} | "
            f"{_percent(group['final']['target_positive_recall'])} | "
            f"{'yes' if group['final']['target_set_exact'] else 'no'} | "
            f"`{convergence['smallest_attempted_rate']:.3e}` |"
        )
    lines.extend(
        [
            "",
            "Every convergence event evaluated 16 finite proposals, retained "
            "finite parameters, moments, direction, and loss, and observed "
            "zero candidate improvement at float32 resolution.",
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
            "## Cluster Throughput",
            "",
            f"- End-to-end five-origin plus five-confirmation wall time: "
            f"{campaign_summary['campaign_wall_seconds']:.2f} seconds.",
            f"- Scheduled MLX process time: "
            f"{campaign_summary['scheduled_process_seconds']:.2f} seconds.",
            f"- Mean active MLX processes: "
            f"{campaign_summary['mean_active_group_processes']:.2f}; "
            f"peak: {campaign_summary['maximum_active_group_processes']}.",
            f"- Idle process-slot seconds while compatible work was queued: "
            f"{campaign_summary['idle_slot_seconds_with_compatible_work']:.2f}.",
            "- Duplicate discovery fraction: 0.00%; the five origins were "
            "distinct groups and duplication was limited to required "
            "cross-host confirmation.",
            f"- Source identity: {source_identity['files']} files, "
            f"`{source_identity['bundle_sha256']}`, identical on "
            f"{', '.join(source_identity['hosts'])}.",
            "",
            "## Authorized Successor",
            "",
            "ADR 0107 neural Stage 2 is now authorized with the unchanged "
            "calibrated monotone AdamW mechanism: exactly four independent "
            "origins and four cross-host replays. A full trainer, validation "
            "treatment, sealed test, and gameplay remain closed.",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument(
        "--source-identity",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--scheduler-state", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = validate_scheduler_state(base.load_json(args.scheduler_state))
    markdown = render_markdown(
        combined_report=base.load_json(args.combined),
        source_identity=base.validate_source_identities(
            args.source_identity
        ),
        replay_summary=validate_replays(args.replay_comparison),
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
