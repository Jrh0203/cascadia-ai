#!/usr/bin/env python3
"""Combine and render the ADR 0114 hierarchical factor oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import frontier_arbitrary_precision_report as base
from cascadia_mlx.full_legal_hierarchical_factor_oracle import ARMS
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    _resource_passed,
)

EXPERIMENT_ID = "full-legal-hierarchical-factor-oracle-v1"


def validate_scheduler_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("ADR 0114 scheduler has the wrong experiment ID")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 8:
        raise ValueError("ADR 0114 scheduler must contain eight tasks")
    for index in range(4):
        origin = tasks[f"origin-{index:02d}"]
        replay = tasks[f"replay-{index:02d}"]
        if (
            origin.get("status") != "done"
            or replay.get("status") != "done"
            or origin.get("host") == replay.get("host")
        ):
            raise ValueError("ADR 0114 scheduler task failed validation")
    return state


def validate_replays(paths: list[Path]) -> dict[str, dict[str, Any]]:
    reports = [base.load_json(path) for path in paths]
    by_arm = {str(report["arm"]): report for report in reports}
    if len(reports) != 4 or set(by_arm) != set(ARMS):
        raise ValueError("ADR 0114 replay set is incomplete")
    if not all(
        report.get("experiment_id") == EXPERIMENT_ID
        and report.get("scientific_payload_identical") is True
        and base._canonical_host(str(report["origin_host"]))
        != base._canonical_host(str(report["replay_host"]))
        and _resource_passed(report["origin_telemetry"])
        and _resource_passed(report["replay_telemetry"])
        for report in reports
    ):
        raise ValueError("ADR 0114 replay validation failed")
    return by_arm


def combine_result(
    arm_paths: list[Path],
    replay_paths: list[Path],
    source_identity_paths: list[Path],
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    for path in arm_paths:
        report = base.load_json(path)
        scientific = report["scientific"]
        arm = str(scientific["arm"])
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or arm not in ARMS
            or arm in arms
        ):
            raise ValueError(f"invalid ADR 0114 arm report: {path}")
        arms[arm] = scientific
        telemetry[arm] = report["telemetry"]
    if set(arms) != set(ARMS):
        raise ValueError("ADR 0114 arm set is incomplete")
    comparisons = validate_replays(replay_paths)
    source_identity = base.validate_source_identities(
        source_identity_paths
    )
    pipeline = all(
        all(bool(value) for value in arm["gates"].values())
        and arm["training_used"] is False
        and arm["gradients_used"] is False
        and arm["optimizer_updates_used"] is False
        and _resource_passed(telemetry[name])
        and arm["test_split_opened"] is False
        and arm["gameplay_opened"] is False
        and arm["new_teacher_compute_used"] is False
        and arm["external_compute_used"] is False
        and comparisons[name]["scientific_payload_identical"] is True
        for name, arm in arms.items()
    )
    wide = arms["conditional-wide"]
    strength = all(
        split["target_positive_recall"] >= 0.98
        and split["target_set_exact_fraction"] >= 0.90
        and split["r4800_winner_retention"] >= 0.99
        and split["mean_proposal_count"] <= 2048
        for split in (wide["train"], wide["validation"])
    )
    if not pipeline:
        classification = "hierarchical_factor_oracle_invalid"
    elif strength:
        classification = "hierarchical_factor_oracle_sufficient"
    else:
        classification = "hierarchical_factor_oracle_insufficient"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "classification": classification,
            "arms": [arms[name] for name in ARMS],
            "selected_arm": "conditional-wide",
            "source_identity": source_identity,
            "gates": {
                "oracle_pipeline_passed": pipeline,
                "all_four_replays_identical": all(
                    report["scientific_payload_identical"] is True
                    for report in comparisons.values()
                ),
                "conditional_wide_strength_passed": strength,
            },
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    combined: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    campaign: dict[str, Any],
) -> str:
    scientific = combined["scientific"]
    lines = [
        "# Full-Legal Hierarchical Factor Oracle V1 Result",
        "",
        f"Classification: `{scientific['classification']}`.",
        "",
        "Four distinct static factor-retrieval budgets audited every open "
        "train and validation action without training or new teacher compute.",
        "",
        "## Results",
        "",
        "| Arm | Origin | Replay | Train recall | Validation recall | "
        "Validation exact | Mean proposals |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for arm in scientific["arms"]:
        comparison = comparisons[arm["arm"]]
        lines.append(
            f"| {arm['arm']} | "
            f"{base._canonical_host(str(comparison['origin_host']))} | "
            f"{base._canonical_host(str(comparison['replay_host']))} | "
            f"{_percent(arm['train']['target_positive_recall'])} | "
            f"{_percent(arm['validation']['target_positive_recall'])} | "
            f"{_percent(arm['validation']['target_set_exact_fraction'])} | "
            f"{arm['validation']['mean_proposal_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for name, passed in sorted(scientific["gates"].items()):
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Cluster Throughput",
            "",
            f"- Campaign wall time: "
            f"{campaign['campaign_wall_seconds']:.2f} seconds.",
            f"- Scheduled process time: "
            f"{campaign['scheduled_process_seconds']:.2f} seconds.",
            f"- Mean active processes: "
            f"{campaign['mean_active_group_processes']:.2f}; "
            f"peak: {campaign['maximum_active_group_processes']}.",
            f"- Idle slot-seconds with compatible queued work: "
            f"{campaign['idle_slot_seconds_with_compatible_work']:.2f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if (
        scientific["classification"]
        == "hierarchical_factor_oracle_sufficient"
    ):
        lines.append(
            "The conditional hierarchy passes the structural Phase 2 gate "
            "and authorizes one learned factor-retrieval pilot."
        )
    elif (
        scientific["classification"]
        == "hierarchical_factor_oracle_insufficient"
    ):
        lines.append(
            "The conditional hierarchy misses the structural gate; learned "
            "factor retrieval is not authorized from this design."
        )
    else:
        lines.append(
            "The pipeline failed before a structural conclusion was eligible."
        )
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, action="append", required=True)
    parser.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--source-identity",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--scheduler-state", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    state = validate_scheduler_state(base.load_json(args.scheduler_state))
    comparisons = validate_replays(args.replay_comparison)
    combined = combine_result(
        args.arm,
        args.replay_comparison,
        args.source_identity,
    )
    markdown = render_markdown(
        combined,
        comparisons,
        base.summarize_campaign_events(state, args.event_log),
    )
    _write(
        args.json_output,
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
    )
    _write(args.markdown_output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
