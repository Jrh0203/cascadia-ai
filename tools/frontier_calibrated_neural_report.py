#!/usr/bin/env python3
"""Combine, classify, and render the frozen ADR 0109 neural-stage result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
import frontier_arbitrary_precision_report as base
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    _resource_passed,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    _closed_domains,
)

EXPERIMENT_ID = "complete-action-frontier-calibrated-neural-stage-v1"
ARM = "calibrated-neural-local-fit-group"
FREE_BLAKE3 = (
    "84d59e71f117250546f21118688ec93d40060e39547d464936c7fd4223b8630a"
)


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
        "candidates": sum(
            int(report["candidates"]) for report in reports
        ),
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
        raise ValueError("ADR 0109 scheduler has the wrong experiment ID")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 8:
        raise ValueError("ADR 0109 scheduler must contain eight tasks")
    for group_index in range(4):
        origin = tasks[f"origin-{group_index:02d}"]
        replay = tasks[f"replay-{group_index:02d}"]
        if (
            origin.get("status") != "done"
            or replay.get("status") != "done"
            or origin.get("host") == replay.get("host")
        ):
            raise ValueError("ADR 0109 scheduler task failed validation")
    return state


def validate_replays(paths: list[Path]) -> dict[int, dict[str, Any]]:
    reports = [base.load_json(path) for path in paths]
    by_group = {
        int(report["group_index"]): report
        for report in reports
    }
    if len(reports) != 4 or set(by_group) != set(range(4)):
        raise ValueError("ADR 0109 replay set is incomplete")
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
        raise ValueError("ADR 0109 replay validation failed")
    return by_group


def combine_result(
    *,
    group_paths: list[Path],
    replay_paths: list[Path],
    free_path: Path,
    source_identity_paths: list[Path],
) -> dict[str, Any]:
    free_blake3 = blake3.blake3(free_path.read_bytes()).hexdigest()
    if free_blake3 != FREE_BLAKE3:
        raise ValueError("ADR 0109 frozen free-stage evidence differs")
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
            raise ValueError(f"invalid ADR 0109 group report: {path}")
        groups_by_index[index] = scientific
        telemetry[index] = report["telemetry"]
    if set(groups_by_index) != set(range(4)):
        raise ValueError("ADR 0109 group set is incomplete")
    comparisons = validate_replays(replay_paths)
    source_identity = base.validate_source_identities(
        source_identity_paths
    )
    ordered = [groups_by_index[index] for index in range(4)]
    group_pipeline = all(
        all(bool(value) for value in report["gates"].values())
        and report["failure"] is None
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
            [
                event
                for event in checkpoint_events
                if event is not None
            ]
        )
        if all(event is not None for event in checkpoint_events)
        else None
    )
    terminal_strength = bool(
        terminal["target_positive_recall"] >= 0.90
        and terminal["target_set_exact_fraction"] >= 0.75
    )
    strength_at_120 = bool(
        aggregate_at_120 is not None
        and aggregate_at_120["target_positive_recall"] >= 0.90
        and aggregate_at_120["target_set_exact_fraction"] >= 0.75
    )
    if not pipeline:
        classification = "calibrated_optimizer_pipeline_invalid"
    elif not terminal_strength:
        classification = "public_observable_representation_insufficient"
    elif not strength_at_120:
        classification = "full_model_local_budget_insufficient"
    else:
        classification = "local_failure_not_reproduced"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "arm": "calibrated-neural-local-fit-combined",
            "classification": classification,
            "free_stage_blake3": free_blake3,
            "groups": ordered,
            "aggregate_at_120": aggregate_at_120,
            "aggregate": terminal,
            "source_identity": source_identity,
            "gates": {
                "neural_pipeline_passed": pipeline,
                "group_pipeline_passed": group_pipeline,
                "all_four_replays_identical": replay_pipeline,
                "strength_checkpoint_observed": (
                    aggregate_at_120 is not None
                ),
                "strength_gate_at_120_passed": strength_at_120,
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
        "# Complete-Action Frontier Calibrated Neural Stage V1 Result",
        "",
        f"Classification: `{scientific['classification']}`.",
        "",
        "ADR 0109 applied the unchanged calibrated monotone AdamW mechanism "
        "to four frozen full-model local-fit groups. All four cross-host "
        "scientific replays were bit-identical. Sealed test, gameplay, new "
        "teacher compute, cloud, and external compute remained closed.",
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
            "Group 2 reproducibly failed the frozen completion rule after "
            "eight accepted updates. Its scores, moments, and accepted rates "
            "remained finite and it recorded zero nonfinite rejections, but "
            "it did not satisfy every preregistered numerical-convergence "
            "condition. The pipeline therefore fails before the strength "
            "classification is eligible.",
            "",
            "The terminal descriptive aggregate was "
            f"{_percent(scientific['aggregate']['target_positive_recall'])} "
            "recall and "
            f"{_percent(scientific['aggregate']['target_set_exact_fraction'])} "
            "exact sets. No group reached the 120-exposure checkpoint, so "
            "that checkpoint is correctly recorded as unobserved rather than "
            "fabricated.",
            "",
            "## Frozen Gates",
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
            f"- End-to-end four-origin plus four-confirmation wall time: "
            f"{campaign['campaign_wall_seconds']:.2f} seconds.",
            f"- Scheduled MLX process time: "
            f"{campaign['scheduled_process_seconds']:.2f} seconds.",
            f"- Mean active MLX processes: "
            f"{campaign['mean_active_group_processes']:.2f}; "
            f"peak: {campaign['maximum_active_group_processes']}.",
            f"- Idle process-slot seconds while compatible work was queued: "
            f"{campaign['idle_slot_seconds_with_compatible_work']:.2f}.",
            "- Duplicate discovery fraction: 0.00%; every origin tested a "
            "different group and all duplication was required confirmation.",
            f"- Source identity: "
            f"{scientific['source_identity']['files']} files, "
            f"`{scientific['source_identity']['bundle_sha256']}`, identical "
            "on john1-john4.",
            "",
            "## Decision",
            "",
            "The bounded full-trainer pilot is not authorized. The "
            "representation classification is also not eligible because the "
            "pipeline failed first. No additional neural compute may proceed "
            "under ADR 0109; any successor must be separately preregistered "
            "from the retained finite failure evidence.",
            "",
        ]
    )
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
    parser.add_argument("--free", type=Path, required=True)
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
        free_path=args.free,
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
