#!/usr/bin/env python3
"""Combine, classify, and render the ADR 0112 static forensic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import frontier_arbitrary_precision_report as base
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    _resource_passed,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    _aggregate_metrics,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    _closed_domains,
)

EXPERIMENT_ID = (
    "complete-action-frontier-local-geometry-feasibility-forensic-v1"
)
ARM = "local-geometry-feasibility-group"


def validate_scheduler_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("ADR 0112 scheduler has the wrong experiment ID")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 8:
        raise ValueError("ADR 0112 scheduler must contain eight tasks")
    for index in range(4):
        origin = tasks[f"origin-{index:02d}"]
        replay = tasks[f"replay-{index:02d}"]
        if (
            origin.get("status") != "done"
            or replay.get("status") != "done"
            or origin.get("host") == replay.get("host")
        ):
            raise ValueError("ADR 0112 scheduler task failed validation")
    return state


def validate_replays(paths: list[Path]) -> dict[int, dict[str, Any]]:
    reports = [base.load_json(path) for path in paths]
    by_group = {
        int(report["group_index"]): report
        for report in reports
    }
    if len(reports) != 4 or set(by_group) != set(range(4)):
        raise ValueError("ADR 0112 replay set is incomplete")
    if not all(
        report.get("experiment_id") == EXPERIMENT_ID
        and report.get("arm") == ARM
        and report.get("scientific_payload_identical") is True
        and base._canonical_host(str(report["origin_host"]))
        != base._canonical_host(str(report["replay_host"]))
        and _resource_passed(report["origin_telemetry"])
        and _resource_passed(report["replay_telemetry"])
        for report in reports
    ):
        raise ValueError("ADR 0112 replay validation failed")
    return by_group


def combine_result(
    *,
    group_paths: list[Path],
    replay_paths: list[Path],
    source_identity_paths: list[Path],
) -> dict[str, Any]:
    groups: dict[int, dict[str, Any]] = {}
    telemetry: dict[int, dict[str, Any]] = {}
    for path in group_paths:
        report = base.load_json(path)
        scientific = report["scientific"]
        index = int(scientific["group_index"])
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or scientific.get("arm") != ARM
            or index not in range(4)
            or index in groups
        ):
            raise ValueError(f"invalid ADR 0112 group report: {path}")
        groups[index] = scientific
        telemetry[index] = report["telemetry"]
    if set(groups) != set(range(4)):
        raise ValueError("ADR 0112 group set is incomplete")
    comparisons = validate_replays(replay_paths)
    source_identity = base.validate_source_identities(
        source_identity_paths
    )
    ordered = [groups[index] for index in range(4)]
    group_pipeline = all(
        all(bool(value) for value in group["gates"].values())
        and group["training_used"] is False
        and group["gradients_used"] is False
        and group["optimizer_updates_used"] is False
        and _resource_passed(telemetry[index])
        and group["test_split_opened"] is False
        and group["gameplay_opened"] is False
        and group["new_teacher_compute_used"] is False
        and group["external_compute_used"] is False
        for index, group in groups.items()
    )
    replay_pipeline = all(
        comparison["scientific_payload_identical"] is True
        for comparison in comparisons.values()
    )
    pipeline = bool(group_pipeline and replay_pipeline)
    base_aggregate = _aggregate_metrics(
        [group["selected_base"] for group in ordered]
    )
    ceiling = _aggregate_metrics(
        [
            group["candidate_independent_interval_ceiling"]
            for group in ordered
        ]
    )
    mixed_classes = sum(
        int(group["equivalence_classes"]["mixed_target_classes"])
        for group in ordered
    )
    ceiling_passed = bool(
        ceiling["target_positive_recall"] >= 0.90
        and ceiling["target_set_exact_fraction"] >= 0.75
    )
    if not pipeline:
        classification = "local_geometry_feasibility_forensic_invalid"
    elif not ceiling_passed:
        classification = "bounded_adapter_output_insufficient"
    elif mixed_classes:
        classification = "exact_observable_aliasing_material"
    else:
        classification = "parameterized_fit_or_optimizer_insufficient"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "arm": "local-geometry-feasibility-combined",
            "classification": classification,
            "groups": ordered,
            "selected_base_aggregate": base_aggregate,
            "candidate_independent_interval_ceiling": ceiling,
            "mixed_target_classes": mixed_classes,
            "source_identity": source_identity,
            "gates": {
                "forensic_pipeline_passed": pipeline,
                "group_pipeline_passed": group_pipeline,
                "all_four_replays_identical": replay_pipeline,
                "independent_interval_ceiling_passed": ceiling_passed,
                "no_mixed_exact_feature_classes": mixed_classes == 0,
            },
            **_closed_domains(),
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    combined: dict[str, Any],
    comparisons: dict[int, dict[str, Any]],
    campaign: dict[str, Any],
) -> str:
    scientific = combined["scientific"]
    lines = [
        "# Complete-Action Frontier Local-Geometry Feasibility Forensic V1 Result",
        "",
        f"Classification: `{scientific['classification']}`.",
        "",
        "ADR 0112 used frozen selected-model inference and exact static "
        "analysis only. It used no training, gradients, or optimizer updates.",
        "",
        "## Group Results",
        "",
        "| Group | Origin | Replay | Base recall | Interval ceiling | "
        "Mixed classes |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for group in scientific["groups"]:
        index = int(group["group_index"])
        comparison = comparisons[index]
        interval = group["candidate_independent_interval_ceiling"]
        lines.append(
            f"| {index} | "
            f"{base._canonical_host(str(comparison['origin_host']))} | "
            f"{base._canonical_host(str(comparison['replay_host']))} | "
            f"{_percent(group['selected_base']['target_positive_recall'])} | "
            f"{_percent(interval['target_positive_recall'])} | "
            f"{group['equivalence_classes']['mixed_target_classes']} |"
        )
    ceiling = scientific["candidate_independent_interval_ceiling"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Selected-base recall: "
            f"{_percent(scientific['selected_base_aggregate']['target_positive_recall'])}.",
            f"- Independent bounded interval ceiling: "
            f"{_percent(ceiling['target_positive_recall'])} "
            "recall and "
            f"{_percent(ceiling['target_set_exact_fraction'])} "
            "exact sets.",
            f"- Mixed exact target/non-target feature classes: "
            f"{scientific['mixed_target_classes']}.",
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
        == "parameterized_fit_or_optimizer_insufficient"
    ):
        lines.append(
            "The frozen correction range and exact observable rows can "
            "represent the target sets. The remaining failure lies in the "
            "parameterized shared fit or its optimizer path."
        )
    elif scientific["classification"] == "bounded_adapter_output_insufficient":
        lines.append(
            "Even candidate-independent bounded corrections cannot meet the "
            "strength gate, so the frozen adapter output contract is too "
            "restrictive."
        )
    elif scientific["classification"] == "exact_observable_aliasing_material":
        lines.append(
            "The independent range passes, but exact observable rows require "
            "mixed target corrections; aliasing remains material."
        )
    else:
        lines.append(
            "The forensic pipeline failed, so no mechanistic conclusion is "
            "eligible."
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
    parser.add_argument("--group", type=Path, action="append", required=True)
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
        group_paths=args.group,
        replay_paths=args.replay_comparison,
        source_identity_paths=args.source_identity,
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
