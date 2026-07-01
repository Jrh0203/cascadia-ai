#!/usr/bin/env python3
"""Validate, classify, and render the ADR 0116 target-only pilot."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.conditional_tile_target_only import (
    EXPERIMENT_ID,
    OBJECTIVE_ID,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    hierarchical_retrieval_gates,
)
from frontier_hierarchical_retrieval_report import compare_stage_replay

ARTIFACT_ROOT = Path("artifacts/experiments/conditional-tile-target-only-objective-v1")
SOURCE_ROOT = Path("artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1")
DEFAULT_MARKDOWN = Path("docs/v2/reports/conditional-tile-target-only-objective-v1-result.md")


def classify_target_only(gates: dict[str, bool]) -> str:
    """Classify integrity before the frozen scientific treatment gates."""
    if not gates["pipeline_passed"]:
        return "target_only_tile_pipeline_invalid"
    if not gates["treatment_passed"]:
        return "target_only_tile_objective_insufficient"
    return "target_only_tile_objective_sufficient"


def target_only_gates(
    *,
    source_combined: dict[str, Any],
    origin: dict[str, Any],
    replay: dict[str, Any],
    mixed: dict[str, Any],
    integration: dict[str, Any],
) -> dict[str, bool]:
    replay_comparison = compare_stage_replay(origin, replay)
    source = source_combined["scientific"]
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
        "source_classification_is_proposal_failure": (
            source["classification"] == "hierarchical_proposal_insufficient"
        ),
        "origin_experiment_identity": origin.get("experiment_id") == EXPERIMENT_ID,
        "origin_objective_identity": origin.get("objective_id") == OBJECTIVE_ID,
        "origin_finite": bool(origin.get("finite_training"))
        and bool(origin["train"]["all_scores_finite"])
        and bool(origin["validation"]["all_scores_finite"]),
        "origin_resources": (
            int(origin["execution"]["peak_process_rss_bytes"]) < 4 * 1024**3
            and int(origin["execution"]["process_swaps"]) == 0
        ),
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
    comparison = scientific["source_comparison"]
    mixed = scientific["mixed"]["scientific"]["validation"]
    integration = scientific["integration"]["scientific"]["validation"]
    proposal = integration["oracle_inside_learned_proposal"]
    failed = [name for name, passed in scientific["gates"].items() if not passed]
    failed_text = "\n".join(f"- `{name}`" for name in failed) if failed else "- None."
    factor_recall_row = (
        f"| Factor recall | "
        f"{origin['train']['target_factor_recall']:.2%} | "
        f"{origin['validation']['target_factor_recall']:.2%} | "
        f">95% / >90% |"
    )
    exact_query_row = (
        f"| Exact queries | "
        f"{origin['train']['exact_query_fraction']:.2%} | "
        f"{origin['validation']['exact_query_fraction']:.2%} | "
        f"descriptive |"
    )
    change_rows = "\n".join(
        (
            _comparison_row(
                "Train tile recall",
                comparison["source_train_tile_recall"],
                origin["train"]["target_factor_recall"],
                comparison["train_tile_recall_delta"],
            ),
            _comparison_row(
                "Validation tile recall",
                comparison["source_validation_tile_recall"],
                origin["validation"]["target_factor_recall"],
                comparison["validation_tile_recall_delta"],
            ),
            _comparison_row(
                "Integrated proposal recall",
                comparison["source_validation_proposal_recall"],
                proposal["target_positive_recall"],
                comparison["validation_proposal_recall_delta"],
            ),
            _comparison_row(
                "Integrated winner retention",
                comparison["source_validation_winner_retention"],
                proposal["r4800_winner_retention"],
                comparison["validation_winner_retention_delta"],
            ),
        )
    )
    return f"""# Conditional Tile Target-Only Objective V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{scientific["classification"]}`**

## Tile Stage

| Metric | Train | Validation | Gate |
|---|---:|---:|---:|
{factor_recall_row}
{exact_query_row}

Selected epoch: `{origin["best_epoch"]}`. Peak process RSS:
`{origin["execution"]["peak_process_rss_bytes"] / 1024**3:.2f} GiB`.
Process swaps: `{origin["execution"]["process_swaps"]}`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | {mixed["target_positive_recall"]:.2%} | >98% |
| Tile-only oracle-stage winner retention | {mixed["r4800_winner_retention"]:.2%} | >98% |
| Integrated proposal target recall | {proposal["target_positive_recall"]:.2%} | ADR 0115 |
| Integrated proposal winner retention | {proposal["r4800_winner_retention"]:.2%} | ADR 0115 |
| Integrated mean proposal count | {integration["mean_proposal_count"]:.1f} | <=2,048 |

## Change From ADR 0115

| Metric | ADR 0115 | Target-only | Delta |
|---|---:|---:|---:|
{change_rows}

## Integrity

- Selected weights replayed bit-identically on another host.
- The ADR 0115 source pipeline remained valid.
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
) -> str:
    return (
        f"| {label} | {source:.2%} | {treatment:.2%} | "
        f"{delta:+.2%} |"
    )


def _decision_text(classification: str) -> str:
    if classification == "target_only_tile_objective_sufficient":
        return (
            "The objective repair restores the learned proposal. Continue "
            "with the next frozen Phase 2 selector or Phase 3 gate."
        )
    if classification == "target_only_tile_objective_insufficient":
        return (
            "Boundary-only BCE is insufficient. Close this exact objective "
            "and audit model capacity and query-conditioned representation "
            "before another tile training run."
        )
    return (
        "The pipeline is invalid. Repair only the failed integrity condition "
        "before interpreting treatment strength."
    )


def combine(
    *,
    artifact_root: Path,
    source_root: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    manifest = _load(artifact_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID or manifest["status"] != "authorized":
        raise ValueError("ADR 0116 manifest is not authorized")
    source_combined = _load(source_root / "reports/combined.json")
    origin_paths = sorted((artifact_root / "runs").glob("*/report.json"))
    replay_paths = sorted((artifact_root / "replays").glob("*.json"))
    if len(origin_paths) != 1 or len(replay_paths) != 1:
        raise ValueError("ADR 0116 origin or replay coverage is incomplete")
    origin = _load(origin_paths[0])
    replay = _load(replay_paths[0])
    mixed = _load(artifact_root / "reports/mixed-ceiling.json")
    integration = _load(artifact_root / "reports/integration.json")
    gates = target_only_gates(
        source_combined=source_combined,
        origin=origin,
        replay=replay,
        mixed=mixed,
        integration=integration,
    )
    source_tile = source_combined["scientific"]["stages"]["tile"]
    source_proposal = source_combined["scientific"]["integration"]["scientific"][
        "validation"
    ]["oracle_inside_learned_proposal"]
    treatment_proposal = integration["scientific"]["validation"][
        "oracle_inside_learned_proposal"
    ]
    source_comparison = {
        "source_train_tile_recall": source_tile["train"]["target_factor_recall"],
        "source_validation_tile_recall": source_tile["validation"][
            "target_factor_recall"
        ],
        "source_validation_proposal_recall": source_proposal[
            "target_positive_recall"
        ],
        "source_validation_winner_retention": source_proposal[
            "r4800_winner_retention"
        ],
        "train_tile_recall_delta": (
            origin["train"]["target_factor_recall"]
            - source_tile["train"]["target_factor_recall"]
        ),
        "validation_tile_recall_delta": (
            origin["validation"]["target_factor_recall"]
            - source_tile["validation"]["target_factor_recall"]
        ),
        "validation_proposal_recall_delta": (
            treatment_proposal["target_positive_recall"]
            - source_proposal["target_positive_recall"]
        ),
        "validation_winner_retention_delta": (
            treatment_proposal["r4800_winner_retention"]
            - source_proposal["r4800_winner_retention"]
        ),
    }
    scientific = {
        "classification": classify_target_only(gates),
        "gates": gates,
        "origin": origin,
        "replay": replay,
        "replay_comparison": compare_stage_replay(origin, replay),
        "mixed": mixed,
        "integration": integration,
        "source_comparison": source_comparison,
        "source_adr0115_scientific_blake3": source_combined["scientific_blake3"],
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
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    combined = combine(
        artifact_root=args.artifact_root,
        source_root=args.source_root,
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
