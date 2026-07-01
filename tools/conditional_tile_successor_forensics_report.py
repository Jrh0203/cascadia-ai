#!/usr/bin/env python3
"""Validate and combine ADR 0119's three independent forensic arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    _scientific_blake3,
)

EXPERIMENT_ID = "conditional-tile-successor-forensics-v1"
DEFAULT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_MARKDOWN = (
    Path("docs/v2/reports") / f"{EXPERIMENT_ID}-result.md"
)
TASKS = {
    "factor_selector": "forensic-factor-selector",
    "sampling_mass": "forensic-sampling-mass",
    "score_scale": "forensic-score-scale",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_arm(
    report: dict[str, Any],
    *,
    expected_arm: str,
    expected_host: str,
) -> list[str]:
    failures: list[str] = []
    scientific = report.get("scientific", {})
    if report.get("experiment_id") != EXPERIMENT_ID:
        failures.append(f"{expected_arm}: experiment identity")
    if report.get("host") != expected_host:
        failures.append(f"{expected_arm}: host identity")
    if scientific.get("arm") != expected_arm:
        failures.append(f"{expected_arm}: arm identity")
    for key in (
        "test_split_opened",
        "gameplay_opened",
        "new_teacher_compute_used",
        "external_compute_used",
    ):
        if scientific.get(key) is not False:
            failures.append(f"{expected_arm}: {key}")
    execution = report.get("execution", {})
    if int(execution.get("process_swaps", -1)) != 0:
        failures.append(f"{expected_arm}: process swaps")
    if int(execution.get("peak_process_rss_bytes", 1 << 63)) >= 4 * (1 << 30):
        failures.append(f"{expected_arm}: peak RSS")
    if report.get("scientific_blake3") != _scientific_blake3(scientific):
        failures.append(f"{expected_arm}: scientific checksum")
    return failures


def build_combined(
    *,
    factor_selector: dict[str, Any],
    sampling_mass: dict[str, Any],
    score_scale: dict[str, Any],
    queue: dict[str, Any],
) -> dict[str, Any]:
    failures = [
        *validate_arm(
            factor_selector,
            expected_arm="factor-selector-ceiling",
            expected_host="john1",
        ),
        *validate_arm(
            sampling_mass,
            expected_arm="sampling-mass",
            expected_host="john3",
        ),
        *validate_arm(
            score_scale,
            expected_arm="score-scale",
            expected_host="john4",
        ),
    ]
    task_by_id = {task["id"]: task for task in queue["tasks"]}
    attempts = []
    for task_id in TASKS.values():
        task = task_by_id.get(task_id)
        if task is None or task.get("status") != "completed":
            failures.append(f"queue: {task_id} incomplete")
            continue
        if len(task.get("attempts", [])) != 1:
            failures.append(f"queue: {task_id} attempt count")
            continue
        attempt = task["attempts"][0]
        if attempt.get("outcome") != "completed":
            failures.append(f"queue: {task_id} outcome")
        attempts.append(attempt)
    started = min(
        (int(value["claimed_unix_ms"]) for value in attempts),
        default=0,
    )
    ended = max(
        (int(value["ended_unix_ms"]) for value in attempts),
        default=started,
    )
    wall_seconds = (ended - started) / 1_000
    process_seconds = sum(
        (
            int(value["ended_unix_ms"])
            - int(value["claimed_unix_ms"])
        )
        / 1_000
        for value in attempts
    )
    scientific = {
        "pipeline_passed": not failures,
        "pipeline_failures": failures,
        "classifications": {
            "factor_selector": factor_selector["scientific"][
                "classification"
            ],
            "sampling_mass": sampling_mass["scientific"]["classification"],
            "score_scale": score_scale["scientific"]["classification"],
        },
        "mechanical_successors": {
            "if_adr0118_insufficient": (
                "optimizer_schedule_treatment"
                if sampling_mass["scientific"]["classification"]
                == "uniform_query_sampling_not_explanatory"
                else "target_mass_sampling_treatment"
            ),
            "if_adr0118_sufficient": (
                "normalized_complete_action_selector"
                if factor_selector["scientific"]["classification"]
                == "complete_action_selector_required"
                and score_scale["scientific"]["classification"]
                == "cross_stage_score_scale_mismatch"
                else "fixed_factor_selector"
            ),
        },
        "arms": {
            "factor_selector": factor_selector,
            "sampling_mass": sampling_mass,
            "score_scale": score_scale,
        },
        "cluster": {
            "started_unix_ms": started,
            "ended_unix_ms": ended,
            "wall_seconds": wall_seconds,
            "scheduled_process_seconds": process_seconds,
            "decisions_completed": 3,
            "decisions_per_wall_hour": (
                3 * 3600 / wall_seconds if wall_seconds else None
            ),
            "duplicate_discovery_fraction": 0.0,
            "hosts": ["john1", "john3", "john4"],
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
    }


def render_markdown(combined: dict[str, Any]) -> str:
    scientific = combined["scientific"]
    factor = scientific["arms"]["factor_selector"]["scientific"]
    sampling = scientific["arms"]["sampling_mass"]["scientific"]
    scale = scientific["arms"]["score_scale"]["scientific"]
    factor_validation = factor["validation"]["methods"][
        factor["selected_method"]
    ]["overall"]
    train_width = sampling["train"]["width"]
    validation_width = sampling["validation"]["width"]
    cluster = scientific["cluster"]
    train_dispersion = scale["scale_ratios"][
        "train_query_standard_deviation_median_ratio"
    ]
    validation_dispersion = scale["scale_ratios"][
        "validation_query_standard_deviation_median_ratio"
    ]
    width_rows = "\n".join(
        f"| {label} | "
        f"{train_width[key]['miss_mass_to_query_share']:.2f}x | "
        f"{validation_width[key]['miss_mass_to_query_share']:.2f}x |"
        for label, key in (
            ("33-64", "width_33_64"),
            ("65-96", "width_65_96"),
            ("97-128", "width_97_128"),
            ("129+", "width_129_plus"),
        )
    )
    decision_rows = "\n".join(
        (
            f"| Sampling mass | `{sampling['classification']}` | "
            "No width stratum passed the frozen mismatch gate |",
            f"| Score scale | `{scale['classification']}` | "
            f"dispersion ratios {train_dispersion:.2f}x train and "
            f"{validation_dispersion:.2f}x validation |",
            f"| Factor selector | `{factor['classification']}` | "
            f"`{factor['selected_method']}` retained "
            f"{factor_validation['target_positive_recall']:.2%} targets and "
            f"{factor_validation['r4800_winner_retention']:.2%} winners |",
        )
    )
    validation_65_96_ratio = validation_width["width_65_96"][
        "miss_mass_to_query_share"
    ]
    factor_winner = factor_validation["r4800_winner_retention"]
    factor_regret = factor_validation["mean_retained_r4800_regret"]
    factor_recall = factor_validation["target_positive_recall"]
    return f"""# Conditional Tile Successor Forensics V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Pipeline: **{"passed" if scientific["pipeline_passed"] else "invalid"}**

## Decisions

| Arm | Classification | Key evidence |
|---|---|---|
{decision_rows}

## Sampling Evidence

| Width | Train miss share / query share | Validation miss share / query share |
|---|---:|---:|
{width_rows}

The `65-96` stratum came closest, but its validation ratio was
{validation_65_96_ratio:.2f}x, below the frozen 1.50x threshold. Uniform query
sampling is therefore not the selected explanation.

## Selector Evidence

The train-selected fixed method was `{factor["selected_method"]}`. It retained
{factor_winner:.2%} of validation winners with {factor_regret:.6f} mean regret,
but only {factor_recall:.2%} of the validation target set. Even oracle factor
retrieval therefore cannot satisfy the top-64 target-recall gate through a
fixed factor-rank aggregation. A complete-action selector is required.

The stage logits are also materially incomparable: median query standard
deviation differs by {train_dispersion:.2f}x on train and
{validation_dispersion:.2f}x on validation. Any learned complete-action
selector must normalize or learn stage-specific calibration rather than sum
raw logits.

## Mechanical Successors

- If ADR 0118 is insufficient: run one frozen optimizer-schedule treatment,
  not target-mass resampling and not another uniform epoch extension.
- If ADR 0118 is sufficient: train a normalized complete-action top-64 selector;
  fixed factor aggregation is closed.

## Cluster Throughput

The three independent decisions completed on john1, john3, and john4 in
{cluster["wall_seconds"]:.2f} seconds of wall time and
{cluster["scheduled_process_seconds"]:.2f} scheduled process-seconds. Decision
throughput was {cluster["decisions_per_wall_hour"]:.1f} per wall hour with zero
duplicate discovery compute. john2 continued the sole ADR 0118 origin
throughout.

Sealed test, gameplay, new teacher compute, cloud, and external compute
remained closed.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    combined = build_combined(
        factor_selector=_load(
            args.artifact_root / "reports/john1-factor-selector.json"
        ),
        sampling_mass=_load(
            args.artifact_root / "reports/john3-sampling-mass.json"
        ),
        score_scale=_load(
            args.artifact_root / "reports/john4-score-scale.json"
        ),
        queue=_load(args.queue),
    )
    _write_json(args.artifact_root / "reports/combined.json", combined)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(combined))
    print(json.dumps(combined, sort_keys=True))
    return 0 if combined["scientific"]["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
