#!/usr/bin/env python3
"""Validate, classify, and render the ADR 0118 extended-exposure run."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.conditional_tile_extended_exposure import (
    EPOCHS,
    EXPERIMENT_ID,
)
from cascadia_mlx.conditional_tile_target_only import OBJECTIVE_ID
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    hierarchical_retrieval_gates,
)
from frontier_hierarchical_retrieval_report import compare_stage_replay

ARTIFACT_ROOT = Path("artifacts/experiments/conditional-tile-extended-exposure-v1")
SOURCE_ROOT = Path("artifacts/experiments/conditional-tile-target-only-objective-v1")
PIPELINE_ROOT = Path("artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1")
DEFAULT_MARKDOWN = Path("docs/v2/reports/conditional-tile-extended-exposure-v1-result.md")


def classify_extended_exposure(gates: dict[str, bool]) -> str:
    """Classify integrity before the frozen exposure treatment."""
    if not gates["pipeline_passed"]:
        return "extended_exposure_pipeline_invalid"
    if not gates["treatment_passed"]:
        return "extended_exposure_tile_insufficient"
    return "extended_exposure_tile_sufficient"


def summarize_trajectory(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and summarize the complete train-only epoch trajectory."""
    epochs = [int(event["epoch"]) for event in events]
    complete = epochs == list(range(1, EPOCHS + 1))
    recalls = [float(event["train"]["target_factor_recall"]) for event in events]
    exact = [float(event["train"]["exact_query_fraction"]) for event in events]
    milestones = {}
    for threshold in (0.80, 0.85, 0.90, 0.95):
        milestones[f"first_epoch_at_or_above_{threshold:.2f}"] = next(
            (
                int(event["epoch"])
                for event in events
                if float(event["train"]["target_factor_recall"]) >= threshold
            ),
            None,
        )
    return {
        "events": len(events),
        "epochs_complete": complete,
        "first_epoch": epochs[0] if epochs else None,
        "last_epoch": epochs[-1] if epochs else None,
        "maximum_train_recall": max(recalls, default=0.0),
        "maximum_train_exact": max(exact, default=0.0),
        "final_train_recall": recalls[-1] if recalls else 0.0,
        "final_train_exact": exact[-1] if exact else 0.0,
        "all_losses_finite": all(
            float("-inf") < float(event["train_loss"]) < float("inf") for event in events
        ),
        **milestones,
    }


def extended_exposure_gates(
    *,
    pipeline_combined: dict[str, Any],
    origin: dict[str, Any],
    replay: dict[str, Any],
    mixed: dict[str, Any],
    integration: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, bool]:
    replay_comparison = compare_stage_replay(origin, replay)
    source = pipeline_combined["scientific"]
    integration_for_gates = copy.deepcopy(integration)
    integration_for_gates["cache_audit"] = source["cache_audit"]
    stage_reports = dict(source["stages"])
    stage_reports["tile"] = origin
    integrated_gates = hierarchical_retrieval_gates(
        cache_audit_identical=True,
        stage_replays_identical=True,
        stage_reports=stage_reports,
        integration=integration_for_gates,
    )
    validation_mixed = mixed["scientific"]["validation"]
    pipeline = {
        "source_pipeline_passed": bool(source["gates"]["pipeline_passed"]),
        "origin_experiment_identity": origin.get("experiment_id") == EXPERIMENT_ID,
        "origin_objective_identity": origin.get("objective_id") == OBJECTIVE_ID,
        "origin_epoch_contract": (
            int(origin["config"]["epochs"]) == EPOCHS
            and int(origin["source_epoch_budget"]) == 20
            and int(origin["treatment_epoch_budget"]) == EPOCHS
        ),
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
    trajectory = scientific["trajectory"]
    failed = [name for name, passed in scientific["gates"].items() if not passed]
    failed_text = "\n".join(f"- `{name}`" for name in failed) if failed else "- None."
    milestones = "\n".join(
        f"- {threshold}% train recall: `{trajectory[f'first_epoch_at_or_above_0.{threshold}']}`"
        for threshold in (80, 85, 90, 95)
    )
    stage_rows = "\n".join(
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
    return f"""# Conditional Tile Extended Exposure V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{scientific["classification"]}`**

## Tile Stage

| Metric | 20 epochs | 200 epochs | Delta | Gate |
|---|---:|---:|---:|---:|
{stage_rows}

Selected epoch: `{origin["best_epoch"]}`. Origin elapsed:
`{origin["execution"]["elapsed_seconds"] / 60:.1f} minutes`. Peak process RSS:
`{origin["execution"]["peak_process_rss_bytes"] / 1024**3:.2f} GiB`.

## Exposure Trajectory

{milestones}

Maximum train recall: `{trajectory["maximum_train_recall"]:.2%}`.
Final epoch recall: `{trajectory["final_train_recall"]:.2%}`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | {mixed["target_positive_recall"]:.2%} | >98% |
| Tile-only oracle-stage winner retention | {mixed["r4800_winner_retention"]:.2%} | >98% |
| Integrated proposal target recall | {proposal["target_positive_recall"]:.2%} | ADR 0115 |
| Integrated proposal winner retention | {proposal["r4800_winner_retention"]:.2%} | ADR 0115 |
| Integrated mean proposal count | {integration["mean_proposal_count"]:.1f} | <=2,048 |

## Integrity

- The complete 200-epoch train-only trajectory is present and finite.
- Selected weights replayed bit-identically on john3.
- john4 owned the mixed ceiling and john1 owned integration.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Failed Gates

{failed_text}

## Decision

{_decision_text(scientific["classification"])}
"""


def _comparison_row(
    label: str,
    source: float,
    treatment: float,
    delta: float,
    gate: str,
) -> str:
    return (
        f"| {label} | {float(source):.2%} | {float(treatment):.2%} | {float(delta):+.2%} | {gate} |"
    )


def _decision_text(classification: str) -> str:
    if classification == "extended_exposure_tile_sufficient":
        return (
            "Extended exposure restores the learned proposal. Freeze this tile "
            "checkpoint and proceed to the remaining selector gate."
        )
    if classification == "extended_exposure_tile_insufficient":
        return (
            "Uniform full-data exposure is insufficient. Close pure epoch "
            "extension. ADR 0119 also closed target-mass resampling, so the "
            "mechanical successor is one frozen optimizer-schedule treatment."
        )
    return (
        "The pipeline is invalid. Repair only the failed integrity condition "
        "before interpreting treatment strength."
    )


def combine(
    *,
    artifact_root: Path,
    source_root: Path,
    pipeline_root: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    manifest = _load(artifact_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID or manifest["status"] != "authorized":
        raise ValueError("ADR 0118 manifest is not authorized")
    source_combined = _load(source_root / "reports/combined.json")
    pipeline_combined = _load(pipeline_root / "reports/combined.json")
    origin_paths = sorted((artifact_root / "runs").glob("*/report.json"))
    replay_paths = sorted((artifact_root / "replays").glob("*.json"))
    if len(origin_paths) != 1 or len(replay_paths) != 1:
        raise ValueError("ADR 0118 origin or replay coverage is incomplete")
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
    gates = extended_exposure_gates(
        pipeline_combined=pipeline_combined,
        origin=origin,
        replay=replay,
        mixed=mixed,
        integration=integration,
        trajectory=trajectory,
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
        "classification": classify_extended_exposure(gates),
        "gates": gates,
        "origin": origin,
        "replay": replay,
        "replay_comparison": compare_stage_replay(origin, replay),
        "mixed": mixed,
        "integration": integration,
        "trajectory": trajectory,
        "trajectory_blake3": _blake3(events),
        "source_comparison": source_comparison,
        "source_adr0116_scientific_blake3": source_combined["scientific_blake3"],
        "source_pipeline_scientific_blake3": pipeline_combined["scientific_blake3"],
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
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    combined = combine(
        artifact_root=args.artifact_root,
        source_root=args.source_root,
        pipeline_root=args.pipeline_root,
        markdown_path=args.markdown,
    )
    print(
        json.dumps(
            {
                "classification": combined["scientific"]["classification"],
                "scientific_blake3": combined["scientific_blake3"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
