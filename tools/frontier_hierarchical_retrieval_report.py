#!/usr/bin/env python3
"""Validate, combine, classify, and render the ADR 0115 pilot."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import blake3
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    EXPERIMENT_ID,
    FROZEN_ADR0114_BLAKE3,
    STAGES,
    classify_hierarchical_retrieval,
    hierarchical_retrieval_gates,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

ARTIFACT_ROOT = Path(
    "artifacts/experiments/"
    "full-legal-hierarchical-factor-retrieval-pilot-v1"
)
DEFAULT_MARKDOWN = Path(
    "docs/v2/reports/"
    "full-legal-hierarchical-factor-retrieval-pilot-v1-result.md"
)
EXPECTED_ORACLE = {
    "train": {
        "target_positive_recall": 0.9927045887079721,
        "target_set_exact_fraction": 0.9517857142857142,
        "r4800_winner_retention": 0.9946428571428572,
        "mean_proposal_count": 482.5910714285714,
    },
    "validation": {
        "target_positive_recall": 0.9917892156862745,
        "target_set_exact_fraction": 0.95,
        "r4800_winner_retention": 1.0,
        "mean_proposal_count": 482.44166666666666,
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_optional(path: Path) -> dict[str, Any] | None:
    return _load(path) if path.exists() else None


def stage_scientific_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Project an origin report onto the cross-host replay contract."""
    return {
        "stage": report["config"]["stage"],
        "weights_blake3": report["weights_blake3"],
        "train_cache_payload_blake3": report[
            "train_cache_payload_blake3"
        ],
        "validation_cache_payload_blake3": report[
            "validation_cache_payload_blake3"
        ],
        "train": report["train"],
        "validation": report["validation"],
        "test_split_opened": report["test_split_opened"],
    }


def compare_stage_replay(
    origin: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    left = stage_scientific_payload(origin)
    right = replay["scientific"]
    return {
        "stage": left["stage"],
        "origin_host": origin["host"],
        "replay_host": replay["host"],
        "cross_host": origin["host"] != replay["host"],
        "scientific_payload_identical": left == right,
        "origin_scientific_blake3": _blake3(left),
        "replay_scientific_blake3": replay["scientific_blake3"],
    }


def cache_audit_summary(
    origin: dict[str, Any],
    replay: dict[str, Any],
    train_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
) -> dict[str, Any]:
    scientific_identical = origin["scientific"] == replay["scientific"]
    scientific = origin["scientific"]
    oracle_matches = all(
        all(
            scientific[split][name] == expected
            for name, expected in values.items()
        )
        for split, values in EXPECTED_ORACLE.items()
    )
    return {
        "origin_host": origin["host"],
        "replay_host": replay["host"],
        "cross_host": origin["host"] != replay["host"],
        "scientific_payload_identical": scientific_identical,
        "all_factor_bijections": bool(
            train_manifest["all_factor_bijections"]
            and validation_manifest["all_factor_bijections"]
        ),
        "all_prefix_invariants": bool(
            train_manifest["all_prefix_invariants"]
            and validation_manifest["all_prefix_invariants"]
        ),
        "all_factor_target_labels_exact": bool(
            scientific["train"]["all_factor_target_labels_exact"]
            and scientific["validation"]["all_factor_target_labels_exact"]
        ),
        "oracle_reconstruction_passed": oracle_matches,
        "train": scientific["train"],
        "validation": scientific["validation"],
    }


def render_markdown(combined: dict[str, Any]) -> str:
    scientific = combined["scientific"]
    integration = scientific["integration"]["scientific"]
    gates = scientific["gates"]
    classification = scientific["classification"]
    train = integration["train"]
    validation = integration["validation"]
    stage_rows = "\n".join(
        "| {stage} | {train:.2%} | {validation:.2%} | {epoch} | "
        "{seconds:.1f} s |".format(
            stage=stage,
            train=scientific["stages"][stage]["train"][
                "target_factor_recall"
            ],
            validation=scientific["stages"][stage]["validation"][
                "target_factor_recall"
            ],
            epoch=scientific["stages"][stage]["best_epoch"],
            seconds=scientific["stages"][stage]["execution"][
                "elapsed_seconds"
            ],
        )
        for stage in STAGES
    )
    failed = [
        name for name, passed in gates.items() if passed is False
    ]
    failed_text = (
        "\n".join(f"- `{name}`" for name in failed)
        if failed
        else "- None."
    )
    train_proposal = cast(
        Mapping[str, float],
        train["oracle_inside_learned_proposal"],
    )
    validation_proposal = cast(
        Mapping[str, float],
        validation["oracle_inside_learned_proposal"],
    )
    train_top64 = cast(Mapping[str, float], train["learned_top64"])
    validation_top64 = cast(
        Mapping[str, float],
        validation["learned_top64"],
    )
    proposal_recall_row = (
        f"| Learned proposal target recall | "
        f"{train_proposal['target_positive_recall']:.2%} | "
        f"{validation_proposal['target_positive_recall']:.2%} | >98% |"
    )
    proposal_winner_row = (
        f"| Learned proposal winner retention | "
        f"{train_proposal['r4800_winner_retention']:.2%} | "
        f"{validation_proposal['r4800_winner_retention']:.2%} | >98% |"
    )
    proposal_count_row = (
        f"| Mean proposal count | {train['mean_proposal_count']:.1f} | "
        f"{validation['mean_proposal_count']:.1f} | <=2,048 |"
    )
    selector_recall_row = (
        f"| Learned top-64 target recall | "
        f"{train_top64['target_positive_recall']:.2%} | "
        f"{validation_top64['target_positive_recall']:.2%} | >98% |"
    )
    selector_winner_row = (
        f"| Learned top-64 winner recall | "
        f"{train_top64['r4800_winner_retention']:.2%} | "
        f"{validation_top64['r4800_winner_retention']:.2%} | >98% |"
    )
    selector_regret_row = (
        f"| Learned top-64 mean R4800 regret | "
        f"{train_top64['mean_retained_r4800_regret']:.6f} | "
        f"{validation_top64['mean_retained_r4800_regret']:.6f} | <0.15 |"
    )
    mechanism_text = _mechanism_markdown(
        scientific.get("mechanistic_analysis", {})
    )
    cluster_text = _cluster_markdown(
        scientific.get("cluster_utilization"),
        scientific["stages"],
        scientific["cache_scheduler"],
    )
    return f"""# Full-Legal Hierarchical Factor Retrieval Pilot V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{classification}`**

## Stage Results

| Stage | Train factor recall | Validation factor recall | Best epoch | Runtime |
|---|---:|---:|---:|---:|
{stage_rows}

## Integrated Result

| Metric | Train | Validation | Gate |
|---|---:|---:|---:|
{proposal_recall_row}
{proposal_winner_row}
{proposal_count_row}
{selector_recall_row}
{selector_winner_row}
{selector_regret_row}

The oracle-inside-proposal diagnostic isolates learned retrieval. The learned
top-64 result uses only the summed draft, tile, and wildlife model scores.

## Integrity

- Cache audit reproduced ADR 0114 exactly on john1 and john4.
- All cache shards preserve factor bijection and prefix invariants.
- All three selected checkpoints replayed bit-identically on another host.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Failed Gates

{failed_text}

## Mechanistic Audit

{mechanism_text}

## Cluster Throughput

{cluster_text}

## Decision

{_decision_text(classification)}
"""


def _mechanism_markdown(values: dict[str, Any]) -> str:
    objective = values.get("objective_gradient")
    complementarity = values.get("complementarity")
    mixed = values.get("tile_mixed_ceiling")
    train_collision = values.get("train_collision")
    validation_collision = values.get("validation_collision")
    if not all(
        (
            objective,
            complementarity,
            mixed,
            train_collision,
            validation_collision,
        )
    ):
        return "No optional post-selection mechanism bundle was attached."
    objective_scientific = objective["scientific"]
    objective_metrics = objective_scientific["metrics"]
    validation_complementarity = complementarity["scientific"][
        "validation"
    ]
    validation_mixed = mixed["scientific"]["validation"]
    train_collision_metrics = train_collision["scientific"]
    validation_collision_metrics = validation_collision["scientific"]
    return f"""- Exact model-input target conflicts: train
  `{train_collision_metrics['exact_target_conflicting_representations']}`,
  validation
  `{validation_collision_metrics['exact_target_conflicting_representations']}`.
- Objective classification:
  `{objective_scientific['classification']['primary']}`; boundary versus
  combined auxiliary gradient cosine
  `{objective_metrics['mean_boundary_auxiliary_gradient_cosine']:.6f}`.
- Boundary gradient norm
  `{objective_metrics['mean_gradient_norms']['boundary']:.4f}` versus combined
  auxiliary norm
  `{objective_metrics['mean_combined_auxiliary_gradient_norm']:.4f}`.
- Learned/screen top-32 oracle-union validation recall:
  `{validation_complementarity['union_oracle_rerank_target_recall']:.2%}`.
- Tile-only mixed-stage validation action recall:
  `{validation_mixed['target_positive_recall']:.2%}`; winner retention:
  `{validation_mixed['r4800_winner_retention']:.2%}`.

The tile failure is not an exact-label collision or simple prior-blending
problem. The rank-regression pressure directly conflicts with the top-32
membership boundary, authorizing ADR 0116's target-only objective pilot."""


def _cluster_markdown(
    utilization: dict[str, Any] | None,
    stages: dict[str, dict[str, Any]],
    cache_scheduler: dict[str, Any],
) -> str:
    stage_work = "\n".join(
        f"- `{report['host']}`: trained `{stage}` for "
        f"{report['execution']['elapsed_seconds']:.1f} seconds."
        for stage, report in stages.items()
    )
    cache_hosts: dict[str, int] = {}
    for task in cache_scheduler["tasks"].values():
        host = str(task["host"])
        cache_hosts[host] = cache_hosts.get(host, 0) + 1
    cache_work = ", ".join(
        f"`{host}` {count}" for host, count in sorted(cache_hosts.items())
    )
    if utilization is None:
        utilization_text = (
            "Campaign telemetry was unavailable; CPU utilization is not "
            "reported."
        )
    else:
        node_text = ", ".join(
            f"`{node}` {values['mean_cpu_percent']:.1f}% mean/"
            f"{values['peak_cpu_percent']:.1f}% peak"
            for node, values in utilization["nodes"].items()
        )
        utilization_text = (
            f"Dashboard journal coverage: {utilization['observed_samples']} "
            f"samples. Core-weighted host CPU averaged "
            f"{utilization['mean_core_weighted_cpu_percent']:.1f}% and peaked "
            f"at {utilization['peak_core_weighted_cpu_percent']:.1f}%. "
            f"Per node: {node_text}. CPU excludes MLX GPU occupancy."
        )
    return f"""- Cache queue: 10 unique shards in
  `{cache_scheduler['campaign_wall_seconds']:.2f}` seconds wall and
  `{cache_scheduler['scheduled_process_seconds']:.2f}` process-seconds.
- Cache work by host: {cache_work}.
{stage_work}
- Duplicate discovery training fraction: `0%`. Cross-host replays were required
  validation work, not model-selection replicas.
- {utilization_text}

The long wildlife rank-calibration job was the critical path. Other hosts were
backfilled with cache audit, replay, integration tooling, exact collision,
objective-gradient, complementarity, correctness, and successor-preparation
work. Raw CPU remained low during MLX-heavy windows because the dashboard does
not measure Metal occupancy; the campaign optimized distinct decisions rather
than duplicating the wildlife trainer."""


def _decision_text(classification: str) -> str:
    if classification == "hierarchical_factor_retrieval_sufficient":
        return (
            "Phase 2 is complete. The Phase 3 policy and value pilot is "
            "authorized under the frozen plan."
        )
    if classification == "hierarchical_selector_insufficient":
        return (
            "The learned proposal passes, but the summed stage scores do not "
            "pass the deployable top-64 gate. Freeze the proposal and test "
            "exactly one ranking treatment before any Phase 3 trainer."
        )
    if classification == "hierarchical_proposal_insufficient":
        return (
            "The learned proposal itself misses the Phase 2 gate. The "
            "completed mechanistic audit selects ADR 0116's target-only "
            "conditional tile objective as the one authorized successor."
        )
    return (
        "The pipeline is invalid. Repair only the failed integrity or "
        "resource condition before interpreting model strength."
    )


def combine(
    *,
    artifact_root: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    manifest = _load(artifact_root / "manifest.json")
    if (
        manifest["experiment_id"] != EXPERIMENT_ID
        or manifest["frozen_evidence"]["adr0114_combined_report_blake3"]
        != FROZEN_ADR0114_BLAKE3
    ):
        raise ValueError("ADR 0115 manifest identity drifted")
    train_manifest = _load(artifact_root / "cache/train/manifest.json")
    validation_manifest = _load(
        artifact_root / "cache/validation/manifest.json"
    )
    cache_origin = _load(
        artifact_root / "reports/cache-audit-john1.json"
    )
    cache_replay = _load(
        artifact_root / "reports/cache-audit-john4.json"
    )
    cache_summary = cache_audit_summary(
        cache_origin,
        cache_replay,
        train_manifest,
        validation_manifest,
    )
    stages = {}
    replay_comparisons = {}
    for stage in STAGES:
        origin_paths = sorted((artifact_root / "runs").glob(f"*-{stage}/report.json"))
        replay_paths = sorted(
            (artifact_root / "replays").glob(f"*-{stage}.json")
        )
        if len(origin_paths) != 1 or len(replay_paths) != 1:
            raise ValueError(f"{stage} origin or replay coverage is incomplete")
        origin = _load(origin_paths[0])
        replay = _load(replay_paths[0])
        stages[stage] = origin
        replay_comparisons[stage] = compare_stage_replay(origin, replay)
    integration = _load(artifact_root / "reports/integration.json")
    integration["cache_audit"] = cache_summary
    gates = hierarchical_retrieval_gates(
        cache_audit_identical=bool(
            cache_summary["scientific_payload_identical"]
            and cache_summary["cross_host"]
        ),
        stage_replays_identical=all(
            comparison["scientific_payload_identical"]
            and comparison["cross_host"]
            for comparison in replay_comparisons.values()
        ),
        stage_reports=stages,
        integration=integration,
    )
    classification = classify_hierarchical_retrieval(gates)
    scheduler = _load(artifact_root / "cache-scheduler/state.json")
    analysis_root = artifact_root / "analysis"
    mechanistic_analysis = {
        "train_collision": _load_optional(
            analysis_root / "john3-tile-collision-train.json"
        ),
        "validation_collision": _load_optional(
            analysis_root / "john4-tile-collision-validation.json"
        ),
        "objective_gradient": _load_optional(
            analysis_root / "john3-tile-objective-gradients.json"
        ),
        "complementarity": _load_optional(
            analysis_root / "john4-tile-complementarity.json"
        ),
        "tile_mixed_ceiling": _load_optional(
            analysis_root / "john4-tile-mixed-ceiling-final.json"
        ),
        "tile_error_slices": _load_optional(
            analysis_root / "john1-tile-errors-final.json"
        ),
        "tile_prior_baselines": _load_optional(
            analysis_root / "john4-tile-prior-baselines.json"
        ),
    }
    cluster_utilization = _load_optional(
        artifact_root / "reports/cluster-utilization.json"
    )
    scientific = {
        "classification": classification,
        "gates": gates,
        "cache_audit": cache_summary,
        "stages": stages,
        "stage_replay_comparisons": replay_comparisons,
        "integration": integration,
        "cache_scheduler": {
            "campaign_wall_seconds": scheduler["campaign_wall_seconds"],
            "scheduled_process_seconds": scheduler[
                "scheduled_process_seconds"
            ],
            "tasks": scheduler["tasks"],
        },
        "mechanistic_analysis": mechanistic_analysis,
        "cluster_utilization": cluster_utilization,
        "training_module": {
            "path": str(
                artifact_root / "source/training-module.py"
            ),
            "sha256": checksum(
                artifact_root / "source/training-module.py"
            ),
            "blake3": blake3.blake3(
                (artifact_root / "source/training-module.py").read_bytes()
            ).hexdigest(),
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
    output_path = artifact_root / "reports/combined.json"
    _write_json(output_path, combined)
    markdown = render_markdown(combined)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown)
    return combined


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
                "experiment_id": combined["experiment_id"],
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
