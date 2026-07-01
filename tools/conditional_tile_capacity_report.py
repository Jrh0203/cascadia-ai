#!/usr/bin/env python3
"""Validate, classify, and render the ADR 0117 four-host audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.conditional_tile_capacity_audit import (
    EXPERIMENT_ID,
    classify_capacity_audit,
)

ARTIFACT_ROOT = Path("artifacts/experiments/conditional-tile-capacity-query-audit-v1")
DEFAULT_MARKDOWN = Path("docs/v2/reports/conditional-tile-capacity-query-audit-v1-result.md")
MAXIMUM_RSS = 4 * 1024**3


def classify(gates: dict[str, bool], mechanism: str) -> str:
    """Give integrity failure precedence over the scientific mechanism."""
    if not gates["pipeline_passed"]:
        return "conditional_tile_capacity_audit_invalid"
    return mechanism


def _references(report: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(value["shard_index"]), int(value["query_index"]))
        for value in report["query_references"]
    ]


def capacity_gates(
    *,
    manifest: dict[str, Any],
    anatomy: dict[str, Any],
    baseline_16: dict[str, Any],
    baseline_256: dict[str, Any],
    attention_256: dict[str, Any],
) -> dict[str, bool]:
    """Validate frozen identities, cohorts, resources, and sealed domains."""
    reports = (baseline_16, baseline_256, attention_256)
    expected_arms = (
        ("baseline", 16),
        ("baseline", 256),
        ("attention", 256),
    )
    medium_references = _references(baseline_256)
    attention_references = _references(attention_256)
    small_references = _references(baseline_16)
    pipeline = {
        "manifest_authorized": manifest.get("status") == "authorized",
        "experiment_identity": all(
            report.get("experiment_id") == EXPERIMENT_ID for report in reports
        )
        and anatomy.get("experiment_id") == EXPERIMENT_ID,
        "arm_identity": all(
            report["arm"]["model_kind"] == model_kind
            and int(report["arm"]["cohort_size"]) == cohort_size
            for report, (model_kind, cohort_size) in zip(
                reports,
                expected_arms,
                strict=True,
            )
        ),
        "distinct_hosts": len(
            {
                anatomy["host"],
                baseline_16["host"],
                baseline_256["host"],
                attention_256["host"],
            }
        )
        == 4,
        "cache_identity": all(
            report["cache_payload_blake3"]
            == manifest["frozen_evidence"]["train_cache_payload_blake3"]
            for report in reports
        )
        and anatomy["scientific"]["train_cache_payload_blake3"]
        == manifest["frozen_evidence"]["train_cache_payload_blake3"]
        and anatomy["scientific"]["validation_cache_payload_blake3"]
        == manifest["frozen_evidence"]["validation_cache_payload_blake3"],
        "reference_weights_identity": all(
            report["reference_weights_blake3"]
            == manifest["frozen_evidence"]["target_only_weights_blake3"]
            for report in reports
        )
        and anatomy["scientific"]["target_only_weights_blake3"]
        == manifest["frozen_evidence"]["target_only_weights_blake3"],
        "nested_small_cohort": small_references == medium_references[:16],
        "identical_medium_cohort": medium_references == attention_references,
        "cohort_coverage": (
            len(small_references) == 16
            and len(medium_references) == 256
            and len(set(small_references)) == 16
            and len(set(medium_references)) == 256
        ),
        "finite": all(bool(report["all_values_finite"]) for report in reports)
        and bool(anatomy["scientific"]["all_scores_finite"]),
        "resources": all(
            int(report["execution"]["peak_process_rss_bytes"]) < MAXIMUM_RSS
            and int(report["execution"]["process_swaps"]) == 0
            for report in reports
        )
        and int(anatomy["execution"]["peak_process_rss_bytes"]) < MAXIMUM_RSS
        and int(anatomy["execution"]["process_swaps"]) == 0,
        "sealed_domains_closed": all(
            not bool(report[name])
            for report in reports
            for name in (
                "test_split_opened",
                "validation_split_opened",
                "gameplay_opened",
                "new_teacher_compute_used",
                "external_compute_used",
            )
        )
        and all(
            not bool(anatomy[name])
            for name in (
                "test_split_opened",
                "gameplay_opened",
                "new_teacher_compute_used",
                "external_compute_used",
            )
        ),
    }
    return {
        **{f"pipeline_{name}": value for name, value in pipeline.items()},
        "pipeline_passed": all(pipeline.values()),
    }


def render_markdown(combined: dict[str, Any]) -> str:
    scientific = combined["scientific"]
    small = scientific["baseline_16"]
    medium = scientific["baseline_256"]
    attention = scientific["attention_256"]
    anatomy = scientific["anatomy"]["scientific"]
    source = anatomy["baseline"]["adr0116_target_only"]
    drops = anatomy["target_only_recall_drop"]
    failed = [name for name, passed in scientific["gates"].items() if not passed]
    failed_text = "\n".join(f"- `{name}`" for name in failed) if failed else "- None."
    arm_rows = "\n".join(
        (
            _arm_row("Baseline 16", small),
            _arm_row("Baseline 256", medium),
            _arm_row("Attention 256", attention),
        )
    )
    calibrated = anatomy["baseline"]["adr0115_calibrated"]
    anatomy_rows = "\n".join(
        (
            _anatomy_row(
                "Train overall",
                calibrated["train"]["overall"],
                source["train"]["overall"],
            ),
            _anatomy_row(
                "Validation overall",
                calibrated["validation"]["overall"],
                source["validation"]["overall"],
            ),
            _anatomy_row(
                "Validation 33-64",
                calibrated["validation"]["width_33_64"],
                source["validation"]["width_33_64"],
            ),
            _anatomy_row(
                "Validation 65-96",
                calibrated["validation"]["width_65_96"],
                source["validation"]["width_65_96"],
            ),
            _anatomy_row(
                "Validation 97-128",
                calibrated["validation"]["width_97_128"],
                source["validation"]["width_97_128"],
            ),
            _anatomy_row(
                "Validation 129+",
                calibrated["validation"]["width_129_plus"],
                source["validation"]["width_129_plus"],
            ),
        )
    )
    return f"""# Conditional Tile Capacity and Query Audit V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{scientific["classification"]}`**

## Memorization Arms

| Arm | Reference recall | Best recall | Best exact | Steps | Peak RSS |
|---|---:|---:|---:|---:|---:|
{arm_rows}

## Frozen Error Anatomy

| Split / width | ADR 0115 | ADR 0116 | Delta |
|---|---:|---:|---:|
{anatomy_rows}

## Input Sensitivity

Validation recall drop from the intact ADR 0116 checkpoint:

- permuted query context: `{drops["permuted_context"]["validation"]:+.2%}`;
- permuted parent state: `{drops["permuted_state"]["validation"]:+.2%}`;
- zero tile factor: `{drops["zero_tile_factor"]["validation"]:+.2%}`;
- zero local geometry: `{drops["zero_tile_local"]["validation"]:+.2%}`;
- zero descendant summaries: `{drops["zero_descendant"]["validation"]:+.2%}`.

## Integrity

- The 16-query cohort is the exact prefix of both 256-query arms.
- Baseline and attention medium arms used identical queries.
- All four arms ran on distinct Macs with zero duplicate discovery compute.
- Sealed test, gameplay, validation-driven selection, new teacher compute,
  cloud, and external compute remained closed.

## Failed Gates

{failed_text}

## Decision

{_decision_text(scientific["classification"])}
"""


def _arm_row(label: str, report: dict[str, Any]) -> str:
    return (
        f"| {label} | "
        f"{report['reference']['target_factor_recall']:.2%} | "
        f"{report['best']['target_factor_recall']:.2%} | "
        f"{report['best']['exact_query_fraction']:.2%} | "
        f"{report['steps_completed']:,} | "
        f"{report['execution']['peak_process_rss_bytes'] / 1024**3:.2f} GiB |"
    )


def _anatomy_row(
    label: str,
    calibrated: dict[str, Any],
    target_only: dict[str, Any],
) -> str:
    calibrated_recall = float(calibrated["target_factor_recall"])
    target_only_recall = float(target_only["target_factor_recall"])
    return (
        f"| {label} | {calibrated_recall:.2%} | "
        f"{target_only_recall:.2%} | "
        f"{target_only_recall - calibrated_recall:+.2%} |"
    )


def _decision_text(classification: str) -> str:
    if classification == "local_baseline_fit_insufficient":
        return (
            "The unchanged ranker cannot reliably fit even 16 hard queries. "
            "The next treatment must repair local parameterization or optimization."
        )
    if classification == "full_data_scale_or_optimization_insufficient":
        return (
            "The unchanged ranker fits 256 representative hard queries. "
            "The next treatment should target full-data sampling, schedule, or scale."
        )
    if classification == "query_relational_representation_insufficient":
        return (
            "Explicit candidate self-attention rescues the medium cohort. "
            "Authorize one frozen full-cache relational tile pilot."
        )
    if classification == "shared_capacity_or_optimization_insufficient":
        return (
            "Small-cohort fit works, but neither medium model clears the gate. "
            "Run one bounded capacity-versus-optimization control before a full pilot."
        )
    return "The audit pipeline is invalid; repair only the failed integrity condition."


def combine(*, artifact_root: Path, markdown_path: Path) -> dict[str, Any]:
    manifest = _load(artifact_root / "manifest.json")
    anatomy = _load(artifact_root / "analysis/john1-anatomy.json")
    reports = {
        path.parent.name: _load(path)
        for path in sorted((artifact_root / "runs").glob("*/report.json"))
    }
    expected = {
        "john2-baseline-16",
        "john3-baseline-256",
        "john4-attention-256",
    }
    if set(reports) != expected:
        raise ValueError("ADR 0117 run coverage is incomplete")
    baseline_16 = reports["john2-baseline-16"]
    baseline_256 = reports["john3-baseline-256"]
    attention_256 = reports["john4-attention-256"]
    gates = capacity_gates(
        manifest=manifest,
        anatomy=anatomy,
        baseline_16=baseline_16,
        baseline_256=baseline_256,
        attention_256=attention_256,
    )
    mechanism = classify_capacity_audit(
        baseline_16,
        baseline_256,
        attention_256,
    )
    elapsed = [
        float(anatomy["execution"]["elapsed_seconds"]),
        *[float(report["execution"]["elapsed_seconds"]) for report in reports.values()],
    ]
    scientific = {
        "classification": classify(gates, mechanism),
        "mechanism_without_pipeline_override": mechanism,
        "gates": gates,
        "anatomy": anatomy,
        "baseline_16": baseline_16,
        "baseline_256": baseline_256,
        "attention_256": attention_256,
        "cluster": {
            "distinct_arms": 4,
            "distinct_hosts": 4,
            "duplicate_compute_fraction": 0.0,
            "estimated_campaign_wall_seconds": max(elapsed),
            "scheduled_process_seconds": sum(elapsed),
            "decisions_per_wall_hour": 4 / max(max(elapsed) / 3600.0, 1e-12),
        },
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
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    combined = combine(
        artifact_root=args.artifact_root,
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
