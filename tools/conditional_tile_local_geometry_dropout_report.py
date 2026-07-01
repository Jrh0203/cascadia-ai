#!/usr/bin/env python3
"""Validate and classify the contingent ADR 0124 dropout treatment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.conditional_tile_local_geometry_dropout import (
    CORRUPTION_ID,
    DROPOUT_RATE,
    EPOCHS,
    EXPERIMENT_ID,
    LOCAL_LEFT,
    LOCAL_RIGHT,
    SCHEDULE_ID,
)
from cascadia_mlx.conditional_tile_target_only import OBJECTIVE_ID
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    hierarchical_retrieval_gates,
)
from conditional_tile_extended_exposure_report import summarize_trajectory
from conditional_tile_optimizer_schedule_report import schedule_matches
from frontier_hierarchical_retrieval_report import compare_stage_replay

ARTIFACT_ROOT = Path("artifacts/experiments/conditional-tile-local-geometry-dropout-v1")
SOURCE_ROOT = Path("artifacts/experiments/conditional-tile-optimizer-schedule-v1")
PIPELINE_ROOT = Path("artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1")
PREFLIGHT_ROOT = Path(
    "artifacts/experiments/conditional-tile-local-geometry-dropout-preflight-repair-v1"
)
DEFAULT_MARKDOWN = Path("docs/v2/reports/conditional-tile-local-geometry-dropout-v1-result.md")


def classify_dropout(gates: dict[str, bool]) -> str:
    """Classify integrity before treatment strength."""
    if not gates["pipeline_passed"]:
        return "local_geometry_dropout_pipeline_invalid"
    if not gates["treatment_passed"]:
        return "local_geometry_dropout_tile_insufficient"
    return "local_geometry_dropout_tile_sufficient"


def dropout_trajectory_matches(
    events: list[dict[str, Any]],
    *,
    expected_selected: int,
    expected_items: int,
) -> bool:
    """Verify exact feature-corruption coverage on every epoch."""
    if len(events) != EPOCHS:
        return False
    expected_fraction = expected_selected / max(expected_items, 1)
    return all(
        int(event.get("dropout_items", -1)) == expected_selected
        and int(event.get("dropout_eligible_items", -1)) == expected_items
        and math.isclose(
            float(event.get("dropout_fraction", -1.0)),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for event in events
    )


def dropout_gates(
    *,
    source_combined: dict[str, Any],
    pipeline_combined: dict[str, Any],
    preflight_combined: dict[str, Any],
    origin: dict[str, Any],
    replay: dict[str, Any],
    mixed: dict[str, Any],
    integration: dict[str, Any],
    trajectory: dict[str, Any],
    schedule_valid: bool,
    dropout_valid: bool,
) -> dict[str, bool]:
    """Apply the frozen ADR 0124 integrity and strength gates."""
    replay_comparison = compare_stage_replay(origin, replay)
    pipeline_source = pipeline_combined["scientific"]
    integration_for_gates = copy.deepcopy(integration)
    integration_for_gates["cache_audit"] = pipeline_source["cache_audit"]
    stage_reports = dict(pipeline_source["stages"])
    stage_reports["tile"] = origin
    integrated_gates = hierarchical_retrieval_gates(
        cache_audit_identical=True,
        stage_replays_identical=True,
        stage_reports=stage_reports,
        integration=integration_for_gates,
    )
    validation_mixed = mixed["scientific"]["validation"]
    pipeline = {
        "source_adr0120_valid_insufficient": (
            source_combined["scientific"]["classification"]
            == "optimizer_schedule_tile_insufficient"
            and bool(source_combined["scientific"]["gates"]["pipeline_passed"])
        ),
        "source_pipeline_passed": bool(pipeline_source["gates"]["pipeline_passed"]),
        "preflight_passed": (
            preflight_combined["scientific"]["classification"]
            == "local_geometry_dropout_preflight_passed"
        ),
        "origin_experiment_identity": (origin.get("experiment_id") == EXPERIMENT_ID),
        "origin_objective_identity": origin.get("objective_id") == OBJECTIVE_ID,
        "origin_schedule_identity": (origin.get("schedule_id") == SCHEDULE_ID and schedule_valid),
        "origin_corruption_identity": (
            origin.get("corruption_id") == CORRUPTION_ID
            and float(origin.get("dropout_rate", -1.0)) == DROPOUT_RATE
            and origin.get("local_feature_columns") == [LOCAL_LEFT, LOCAL_RIGHT]
            and not bool(origin.get("validation_corruption_used"))
            and not bool(origin.get("inference_corruption_used"))
            and dropout_valid
        ),
        "origin_epoch_contract": int(origin["config"]["epochs"]) == EPOCHS,
        "origin_finite": bool(origin.get("finite_training"))
        and bool(origin["train"]["all_scores_finite"])
        and bool(origin["validation"]["all_scores_finite"]),
        "origin_resources": (
            int(origin["execution"]["peak_process_rss_bytes"]) < 4 * 1024**3
            and int(origin["execution"]["process_swaps"]) == 0
        ),
        "trajectory_complete": bool(trajectory["epochs_complete"])
        and bool(trajectory["all_losses_finite"]),
        "cross_host_replay": bool(replay_comparison["cross_host"]),
        "replay_identical": bool(replay_comparison["scientific_payload_identical"]),
        "sealed_domains_closed": all(
            not bool(origin[name])
            for name in (
                "test_split_opened",
                "gameplay_opened",
                "new_teacher_compute_used",
                "external_compute_used",
            )
        ),
        "integration_pipeline_passed": bool(integrated_gates["pipeline_passed"]),
    }
    treatment = {
        "train_tile_factor_recall_above_0_95": (
            float(origin["train"]["target_factor_recall"]) > 0.95
        ),
        "validation_tile_factor_recall_above_0_90": (
            float(origin["validation"]["target_factor_recall"]) > 0.90
        ),
        "mixed_validation_target_recall_above_0_98": (
            float(validation_mixed["target_positive_recall"]) > 0.98
        ),
        "mixed_validation_winner_retention_above_0_98": (
            float(validation_mixed["r4800_winner_retention"]) > 0.98
        ),
        "integrated_proposal_passed": bool(integrated_gates["proposal_passed"]),
    }
    return {
        **{f"pipeline_{name}": value for name, value in pipeline.items()},
        **treatment,
        "pipeline_passed": all(pipeline.values()),
        "treatment_passed": all(treatment.values()),
    }


def render_markdown(combined: dict[str, Any]) -> str:
    scientific = combined["scientific"]
    origin = scientific["origin"]
    source = scientific["source_comparison"]
    mixed = scientific["mixed"]["scientific"]["validation"]
    integration = scientific["integration"]["scientific"]["validation"]
    proposal = integration["oracle_inside_learned_proposal"]
    failed = [name for name, passed in scientific["gates"].items() if not passed]
    failed_text = "\n".join(f"- `{name}`" for name in failed) if failed else "- None."
    rows = "\n".join(
        (
            _comparison_row(
                "Train recall",
                source["source_train_recall"],
                origin["train"]["target_factor_recall"],
                source["train_recall_delta"],
                ">95%",
            ),
            _comparison_row(
                "Validation recall",
                source["source_validation_recall"],
                origin["validation"]["target_factor_recall"],
                source["validation_recall_delta"],
                ">90%",
            ),
            _comparison_row(
                "Train exact queries",
                source["source_train_exact"],
                origin["train"]["exact_query_fraction"],
                source["train_exact_delta"],
                "descriptive",
            ),
            _comparison_row(
                "Validation exact queries",
                source["source_validation_exact"],
                origin["validation"]["exact_query_fraction"],
                source["validation_exact_delta"],
                "descriptive",
            ),
        )
    )
    decision = (
        "The targeted structural regularizer restores the learned proposal. "
        "Freeze this tile checkpoint and proceed to the complete-action selector."
        if scientific["classification"] == "local_geometry_dropout_tile_sufficient"
        else (
            "The valid targeted regularizer is insufficient. Close this "
            "conditional pointwise tile representation and move upstream."
            if scientific["classification"] == "local_geometry_dropout_tile_insufficient"
            else "The pipeline is invalid. Repair only the failed integrity condition."
        )
    )
    return f"""# Conditional Tile Local-Geometry Dropout V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{scientific["classification"]}`**

## Tile Stage

| Metric | ADR 0120 | Dropout treatment | Delta | Gate |
|---|---:|---:|---:|---:|
{rows}

Selected epoch: `{origin["best_epoch"]}`. Origin elapsed:
`{origin["execution"]["elapsed_seconds"] / 60:.1f} minutes`. The complete
learning-rate trajectory matched `{SCHEDULE_ID}`:
`{scientific["schedule_valid"]}`. Exact dropout coverage matched
`{CORRUPTION_ID}`: `{scientific["dropout_valid"]}`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | {mixed["target_positive_recall"]:.2%} | >98% |
| Tile-only oracle-stage winner retention | {mixed["r4800_winner_retention"]:.2%} | >98% |
| Integrated proposal target recall | {proposal["target_positive_recall"]:.2%} | ADR 0115 |
| Integrated proposal winner retention | {proposal["r4800_winner_retention"]:.2%} | ADR 0115 |
| Integrated mean proposal count | {integration["mean_proposal_count"]:.1f} | <=2,048 |

## Failed Gates

{failed_text}

## Decision

{decision}
"""


def _comparison_row(
    label: str,
    source: float,
    treatment: float,
    delta: float,
    gate: str,
) -> str:
    return f"| {label} | {source:.2%} | {treatment:.2%} | {delta:+.2%} | {gate} |"


def combine(
    *,
    artifact_root: Path,
    source_root: Path,
    pipeline_root: Path,
    preflight_root: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """Combine the sole authorized origin and dependent evaluations."""
    manifest = _load(artifact_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("ADR 0124 manifest identity drifted")
    if manifest["status"] != "authorized":
        raise ValueError("ADR 0124 training branch is not authorized")
    source_combined = _load(source_root / "reports/combined.json")
    pipeline_combined = _load(pipeline_root / "reports/combined.json")
    preflight_combined = _load(preflight_root / "reports/combined.json")
    origin_paths = sorted((artifact_root / "runs").glob("*/report.json"))
    replay_paths = sorted((artifact_root / "replays").glob("*.json"))
    if len(origin_paths) != 1 or len(replay_paths) != 1:
        raise ValueError("ADR 0124 origin or replay coverage is incomplete")
    origin = _load(origin_paths[0])
    replay = _load(replay_paths[0])
    mixed = _load(artifact_root / "reports/mixed-ceiling.json")
    integration = _load(artifact_root / "reports/integration.json")
    events = [
        json.loads(line)
        for line in (origin_paths[0].parent / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    trajectory = summarize_trajectory(events)
    schedule_valid = schedule_matches(events)
    contract = preflight_combined["scientific"]["arms"]["contract"]["scientific"]
    dropout_valid = dropout_trajectory_matches(
        events,
        expected_selected=int(contract["selected_items"]),
        expected_items=int(contract["items"]),
    )
    gates = dropout_gates(
        source_combined=source_combined,
        pipeline_combined=pipeline_combined,
        preflight_combined=preflight_combined,
        origin=origin,
        replay=replay,
        mixed=mixed,
        integration=integration,
        trajectory=trajectory,
        schedule_valid=schedule_valid,
        dropout_valid=dropout_valid,
    )
    source_origin = source_combined["scientific"]["origin"]
    source_comparison = {
        "source_train_recall": source_origin["train"]["target_factor_recall"],
        "source_validation_recall": source_origin["validation"]["target_factor_recall"],
        "source_train_exact": source_origin["train"]["exact_query_fraction"],
        "source_validation_exact": source_origin["validation"]["exact_query_fraction"],
        "train_recall_delta": (
            origin["train"]["target_factor_recall"] - source_origin["train"]["target_factor_recall"]
        ),
        "validation_recall_delta": (
            origin["validation"]["target_factor_recall"]
            - source_origin["validation"]["target_factor_recall"]
        ),
        "train_exact_delta": (
            origin["train"]["exact_query_fraction"] - source_origin["train"]["exact_query_fraction"]
        ),
        "validation_exact_delta": (
            origin["validation"]["exact_query_fraction"]
            - source_origin["validation"]["exact_query_fraction"]
        ),
    }
    scientific = {
        "classification": classify_dropout(gates),
        "gates": gates,
        "origin": origin,
        "replay": replay,
        "replay_comparison": compare_stage_replay(origin, replay),
        "mixed": mixed,
        "integration": integration,
        "trajectory": trajectory,
        "trajectory_blake3": _blake3(events),
        "schedule_valid": schedule_valid,
        "dropout_valid": dropout_valid,
        "source_comparison": source_comparison,
        "source_adr0120_scientific_blake3": source_combined["scientific_blake3"],
        "source_pipeline_scientific_blake3": pipeline_combined["scientific_blake3"],
        "preflight_scientific_blake3": preflight_combined["scientific_blake3"],
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    combined = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
    }
    _write_json(artifact_root / "reports/combined.json", combined)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(combined))
    return combined


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument("--preflight-root", type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    combined = combine(
        artifact_root=args.artifact_root,
        source_root=args.source_root,
        pipeline_root=args.pipeline_root,
        preflight_root=args.preflight_root,
        markdown_path=args.markdown,
    )
    print(
        json.dumps(
            {
                "classification": combined["scientific"]["classification"],
                "scientific_blake3": combined["scientific_blake3"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
