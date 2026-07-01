#!/usr/bin/env python3
"""Combine, classify, and render the frozen ADR 0111 result."""

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
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    _closed_domains,
)

EXPERIMENT_ID = (
    "complete-action-frontier-calibrated-local-geometry-adapter-v1"
)
ARM = "calibrated-local-geometry-adapter-group"


def aggregate_evaluation_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("cannot aggregate an empty evaluation set")
    groups = sum(int(report["groups"]) for report in reports)
    target_slots = sum(int(report["target_slots"]) for report in reports)
    target_hits = sum(int(report["target_hits"]) for report in reports)
    return {
        "groups": groups,
        "candidates": sum(int(report["candidates"]) for report in reports),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact_fraction": sum(
            float(report["target_set_exact_fraction"])
            * int(report["groups"])
            for report in reports
        )
        / groups,
        "r4800_winner_retention": sum(
            float(report["r4800_winner_retention"])
            * int(report["groups"])
            for report in reports
        )
        / groups,
        "mean_objective": sum(
            float(report["mean_objective"]) * int(report["groups"])
            for report in reports
        )
        / groups,
        "all_scores_finite": all(
            bool(report["all_scores_finite"]) for report in reports
        ),
    }


def validate_scheduler_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("ADR 0111 scheduler has the wrong experiment ID")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 8:
        raise ValueError("ADR 0111 scheduler must contain eight tasks")
    for group_index in range(4):
        origin = tasks[f"origin-{group_index:02d}"]
        replay = tasks[f"replay-{group_index:02d}"]
        if (
            origin.get("status") != "done"
            or replay.get("status") != "done"
            or origin.get("host") == replay.get("host")
        ):
            raise ValueError("ADR 0111 scheduler task failed validation")
    return state


def validate_replays(paths: list[Path]) -> dict[int, dict[str, Any]]:
    reports = [base.load_json(path) for path in paths]
    by_group = {
        int(report["group_index"]): report
        for report in reports
    }
    if len(reports) != 4 or set(by_group) != set(range(4)):
        raise ValueError("ADR 0111 replay set is incomplete")
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
        raise ValueError("ADR 0111 replay validation failed")
    return by_group


def combine_result(
    *,
    group_paths: list[Path],
    replay_paths: list[Path],
    source_identity_paths: list[Path],
) -> dict[str, Any]:
    groups_by_index: dict[int, dict[str, Any]] = {}
    telemetry: dict[int, dict[str, Any]] = {}
    for path in group_paths:
        report = base.load_json(path)
        scientific = report["scientific"]
        index = int(scientific["group_index"])
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or scientific.get("arm") != ARM
            or index not in range(4)
            or index in groups_by_index
        ):
            raise ValueError(f"invalid ADR 0111 group report: {path}")
        groups_by_index[index] = scientific
        telemetry[index] = report["telemetry"]
    if set(groups_by_index) != set(range(4)):
        raise ValueError("ADR 0111 group set is incomplete")
    comparisons = validate_replays(replay_paths)
    source_identity = base.validate_source_identities(
        source_identity_paths
    )
    ordered = [groups_by_index[index] for index in range(4)]
    group_pipeline = all(
        all(bool(value) for value in report["gates"].values())
        and report["failure"] is None
        and report["base_model_frozen"] is True
        and report["zero_initialized_base_equality"] is True
        and _resource_passed(telemetry[index])
        and report["test_split_opened"] is False
        and report["gameplay_opened"] is False
        and report["new_teacher_compute_used"] is False
        and report["external_compute_used"] is False
        for index, report in groups_by_index.items()
    )
    replay_pipeline = all(
        comparison["scientific_payload_identical"] is True
        for comparison in comparisons.values()
    )
    pipeline = bool(group_pipeline and replay_pipeline)
    terminal = aggregate_evaluation_reports(
        [report["final"] for report in ordered]
    )
    checkpoint_events = [
        next(
            (
                event["metrics"]
                for event in report["trajectory"]
                if event["exposures_per_group"] == 120
            ),
            None,
        )
        for report in ordered
    ]
    aggregate_at_120 = (
        aggregate_evaluation_reports(
            [event for event in checkpoint_events if event is not None]
        )
        if all(event is not None for event in checkpoint_events)
        else None
    )
    terminal_strength = bool(
        terminal["target_positive_recall"] >= 0.90
        and terminal["target_set_exact_fraction"] >= 0.75
    )
    if not pipeline:
        classification = "calibrated_local_geometry_pipeline_invalid"
    elif not terminal_strength:
        classification = "calibrated_local_geometry_insufficient"
    else:
        classification = "calibrated_local_geometry_sufficient"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "arm": "calibrated-local-geometry-adapter-combined",
            "classification": classification,
            "groups": ordered,
            "aggregate_at_120": aggregate_at_120,
            "aggregate": terminal,
            "source_identity": source_identity,
            "gates": {
                "adapter_pipeline_passed": pipeline,
                "group_pipeline_passed": group_pipeline,
                "all_four_replays_identical": replay_pipeline,
                "strength_checkpoint_observed": (
                    aggregate_at_120 is not None
                ),
                "terminal_strength_gate_passed": terminal_strength,
            },
            **_closed_domains(),
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    *,
    combined: dict[str, Any],
    comparisons: dict[int, dict[str, Any]],
    campaign: dict[str, Any],
) -> str:
    scientific = combined["scientific"]
    lines = [
        "# Complete-Action Frontier Calibrated Local-Geometry Adapter V1 Result",
        "",
        f"Classification: `{scientific['classification']}`.",
        "",
        "ADR 0111 isolated exact rotation-canonical local geometry as a "
        "zero-initialized residual adapter over the frozen selected model. "
        "All four groups used distinct origins and cross-host replay.",
        "",
        "## Group Results",
        "",
        "| Group | Origin | Replay | Accepted | Completion | Recall | Exact |",
        "|---:|---|---|---:|---|---:|---:|",
    ]
    for group in scientific["groups"]:
        index = int(group["group_index"])
        comparison = comparisons[index]
        completion = (
            "numerically converged"
            if group["numerical_convergence"] is not None
            else (
                f"failed: {group['failure']}"
                if group["failure"] is not None
                else "completed"
            )
        )
        lines.append(
            f"| {index} | "
            f"{base._canonical_host(str(comparison['origin_host']))} | "
            f"{base._canonical_host(str(comparison['replay_host']))} | "
            f"{group['optimizer']['accepted_updates']} | {completion} | "
            f"{_percent(group['final']['target_positive_recall'])} | "
            f"{_percent(group['final']['target_set_exact_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Terminal target recall: "
            f"{_percent(scientific['aggregate']['target_positive_recall'])}.",
            f"- Terminal exact target sets: "
            f"{_percent(scientific['aggregate']['target_set_exact_fraction'])}.",
            f"- 120-update aggregate observed: "
            f"{'yes' if scientific['aggregate_at_120'] is not None else 'no'}.",
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
            f"- Scheduled MLX process time: "
            f"{campaign['scheduled_process_seconds']:.2f} seconds.",
            f"- Mean active MLX processes: "
            f"{campaign['mean_active_group_processes']:.2f}; "
            f"peak: {campaign['maximum_active_group_processes']}.",
            f"- Idle slot-seconds with compatible queued work: "
            f"{campaign['idle_slot_seconds_with_compatible_work']:.2f}.",
            "- Duplicate discovery fraction: 0.00%; origins tested distinct "
            "groups and all duplication was required cross-host replay.",
            f"- Source identity: {scientific['source_identity']['files']} "
            f"files, `{scientific['source_identity']['bundle_sha256']}`, "
            "identical on john1-john4.",
            "",
            "## Decision",
            "",
        ]
    )
    if (
        scientific["classification"]
        == "calibrated_local_geometry_sufficient"
    ):
        lines.append(
            "The exact frozen-base adapter passes the local representation "
            "gate and authorizes one bounded full-trainer pilot under the "
            "unchanged ADR 0111 contract."
        )
    elif (
        scientific["classification"]
        == "calibrated_local_geometry_insufficient"
    ):
        lines.append(
            "The single representation treatment authorized by ADR 0110 is "
            "exhausted without meeting the local strength gate. A second "
            "representation treatment and full trainer are not authorized."
        )
    else:
        lines.append(
            "The pipeline failed before a representation conclusion was "
            "eligible. No successor compute is authorized from this result."
        )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
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
        combined=combined,
        comparisons=comparisons,
        campaign=base.summarize_campaign_events(
            state,
            args.event_log,
        ),
    )
    _write_json(args.json_output, combined)
    _write_text(args.markdown_output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
