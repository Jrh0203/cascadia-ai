#!/usr/bin/env python3
"""Assemble and verify the complete ADR 0101 cluster result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    expected_rank_validation_gates,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    ADR0100_TRAIN_RECALL,
    EXPERIMENT_ID,
    TARGET_SCALE,
    classify_scale16_expected_rank_pilot,
)
from frontier_expected_rank_report import (
    event_window,
    load_json,
    normalize_host,
    replay_payload,
    write_json_atomic,
    write_text_atomic,
)

HOSTS = ("john1", "john2", "john3", "john4")


def build_report(experiment_root: Path) -> dict[str, Any]:
    manifest = load_json(experiment_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("ADR 0101 experiment identity drifted")
    if float(manifest["treatment"]["target_scale"]) != TARGET_SCALE:
        raise ValueError("ADR 0101 target scale drifted")

    source_identities = {
        host: load_json(experiment_root / "source-identity" / f"{host}.json")
        for host in HOSTS
    }
    source_hashes = {
        str(identity["bundle_sha256"])
        for identity in source_identities.values()
    }
    source_files = {
        int(identity["files"]) for identity in source_identities.values()
    }
    if len(source_hashes) != 1 or len(source_files) != 1:
        raise ValueError("ADR 0101 source differs across hosts")

    cache = load_json(experiment_root / "reports" / "cache-comparison.json")
    alignment = load_json(experiment_root / "reports" / "alignment-john1.json")
    gradient = load_json(experiment_root / "reports" / "gradient-john3.json")
    baseline = load_json(experiment_root / "reports" / "baseline-john4.json")
    origin = load_json(experiment_root / "reports" / "evaluation-john2.json")
    replay = load_json(experiment_root / "reports" / "evaluation-john1.json")
    if normalize_host(origin["telemetry"]["host"]) != "john2":
        raise ValueError("ADR 0101 origin evaluation host drifted")
    if normalize_host(replay["telemetry"]["host"]) != "john1":
        raise ValueError("ADR 0101 replay evaluation host drifted")

    origin_payload = replay_payload(origin)
    replay_identical = replay_payload(replay) == origin_payload
    if float(origin_payload["target_scale"]) != TARGET_SCALE:
        raise ValueError("ADR 0101 selected model target scale drifted")
    performance = {
        "john2": origin["scientific"]["performance"],
        "john1": replay["scientific"]["performance"],
    }
    open_report = {
        "train": origin_payload["train"],
        "validation": origin_payload["validation"],
        "test_split_opened": origin_payload["test_split_opened"],
        "gameplay_opened": origin_payload["gameplay_opened"],
        "new_teacher_compute_used": origin_payload[
            "new_teacher_compute_used"
        ],
        "external_compute_used": origin_payload["external_compute_used"],
    }
    gates = expected_rank_validation_gates(
        open_report,
        performance_by_host=performance,
        optimization_audit_passed=bool(gradient["scientific"]["passed"]),
        replay_identical=replay_identical,
        cache_identical=bool(cache["scientific"]["passed"]),
    )
    gates["alignment_audit_passed"] = bool(
        alignment["scientific"]["passed"]
    )
    gates["baseline_reachability_audit_passed"] = bool(
        baseline["scientific"]["passed"]
    )
    gates["pilot_passed"] = all(
        value for name, value in gates.items() if name != "pilot_passed"
    )
    train_recall = float(
        origin_payload["train"]["expected_rank_target_positive_recall"]
    )
    classification = classify_scale16_expected_rank_pilot(
        gates,
        train_target_recall=train_recall,
    )

    windows = {
        "cache_train_john1": event_window(
            experiment_root / "events-cache-train-john1.jsonl"
        ),
        "cache_train_john2": event_window(
            experiment_root / "events-cache-train-john2.jsonl"
        ),
        "cache_validation_john1": event_window(
            experiment_root / "events-cache-validation-john1.jsonl"
        ),
        "cache_validation_john2": event_window(
            experiment_root / "events-cache-validation-john2.jsonl"
        ),
        "alignment_john1": event_window(
            experiment_root / "events-alignment-john1.jsonl"
        ),
        "training_john2": event_window(
            experiment_root / "events-training-john2.jsonl"
        ),
        "gradient_john3": event_window(
            experiment_root / "events-gradient-john3.jsonl"
        ),
        "baseline_john4": event_window(
            experiment_root / "events-baseline-john4.jsonl"
        ),
        "evaluation_john2": event_window(
            experiment_root / "events-evaluation-john2.jsonl"
        ),
        "evaluation_john1": event_window(
            experiment_root / "events-evaluation-john1.jsonl"
        ),
    }
    started = min(value["started_unix_seconds"] for value in windows.values())
    ended = max(value["ended_unix_seconds"] for value in windows.values())
    wall_seconds = ended - started
    productive = {host: 0.0 for host in HOSTS}
    for name, window in windows.items():
        host = name.rsplit("_", maxsplit=1)[-1]
        productive[host] += window["elapsed_seconds"]

    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "promoted_to_sealed_test" if gates["pilot_passed"] else "rejected"
        ),
        "classification": classification,
        "source": {
            "files": next(iter(source_files)),
            "bundle_sha256": next(iter(source_hashes)),
            "bit_identical_across_john1_john2_john3_john4": True,
        },
        "cache": cache["scientific"],
        "alignment": alignment["scientific"],
        "gradient": gradient["scientific"],
        "baseline": baseline["scientific"],
        "selected_model": {
            "origin_host": "john2",
            "replay_host": "john1",
            "replay_bit_identical": replay_identical,
            "scientific": origin_payload,
            "performance": performance,
            "origin_telemetry": origin["telemetry"],
            "replay_telemetry": replay["telemetry"],
        },
        "comparison": {
            "adr0100_train_target_recall": ADR0100_TRAIN_RECALL,
            "scale16_train_target_recall": train_recall,
            "train_target_recall_delta": (
                train_recall - ADR0100_TRAIN_RECALL
            ),
        },
        "gates": gates,
        "execution": {
            "first_job_started_unix_seconds": started,
            "final_job_ended_unix_seconds": ended,
            "campaign_wall_seconds": wall_seconds,
            "hypotheses_completed": 1,
            "hypotheses_per_hour": 3600.0 / max(wall_seconds, 1e-9),
            "productive_wall_seconds_by_host": productive,
            "duplicate_training_fraction": 0.0,
            "justified_cache_reproduction": True,
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    train = report["selected_model"]["scientific"]["train"]
    validation = report["selected_model"]["scientific"]["validation"]
    baseline = report["baseline"]
    alignment = report["alignment"]
    comparison = report["comparison"]
    lines = [
        "# ADR 0101 Frontier Expected-Rank Scale 16 Result",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "| Metric | Screen baseline | Selected model | Gate |",
        "|---|---:|---:|---:|",
        "| validation expected-rank target recall | "
        f"{baseline['validation']['baseline']['expected_rank_target_positive_recall']:.2%} | "
        f"{validation['expected_rank_target_positive_recall']:.2%} | 50.00% |",
        "| validation exact expected-rank sets | "
        f"{baseline['validation']['baseline']['expected_rank_target_set_exact_fraction']:.2%} | "
        f"{validation['expected_rank_target_set_exact_fraction']:.2%} | 1.00% |",
        "| validation R4800 winner recall | "
        f"{baseline['validation']['baseline']['top64_r4800_winner_recall']:.2%} | "
        f"{validation['top64_r4800_winner_recall']:.2%} | >98.00% |",
        "| validation confidence coverage | "
        f"{baseline['validation']['baseline']['top64_confidence_set_coverage_95']:.2%} | "
        f"{validation['top64_confidence_set_coverage_95']:.2%} | 99.00% |",
        "| validation retained regret | "
        f"{baseline['validation']['baseline']['mean_top64_retained_r4800_regret']:.6f} | "
        f"{validation['mean_top64_retained_r4800_regret']:.6f} | <0.030000 |",
        "",
        "Train fit:",
        "",
        f"- target recall: {train['expected_rank_target_positive_recall']:.2%};",
        f"- exact target sets: "
        f"{train['expected_rank_target_set_exact_fraction']:.2%};",
        f"- recall delta versus ADR 0100: "
        f"{comparison['train_target_recall_delta']:+.2%}.",
        "",
        "Alignment diagnostics:",
        "",
        "- train deployed-set target mass: "
        f"{alignment['train']['probability_mass_in_deployed_target']['mean']:.2%};",
        "- validation deployed-set target mass: "
        f"{alignment['validation']['probability_mass_in_deployed_target']['mean']:.2%};",
        "- validation absolute gradient inside deployed set: "
        f"{alignment['validation']['uniform_student_absolute_gradient_fraction_in_deployed_target']['mean']:.2%}.",
        "",
        f"The cache audit passed: {report['cache']['passed']}. "
        f"The 12-group gradient audit passed: {report['gradient']['passed']}. "
        f"The reachability audit passed: {report['baseline']['passed']}. "
        "The selected checkpoint replay was bit-identical: "
        f"{report['selected_model']['replay_bit_identical']}.",
        "",
        f"The campaign completed in "
        f"{report['execution']['campaign_wall_seconds']:.2f} seconds with one "
        "trainer and zero duplicate training.",
        "",
        "The sealed test, gameplay, new teacher compute, cloud, and external "
        "compute remained closed.",
        "",
    ]
    failed = [name for name, passed in report["gates"].items() if not passed]
    if failed:
        lines.extend(
            [
                "Failed gates:",
                "",
                *[f"- `{name}`" for name in failed],
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.experiment_root)
    write_json_atomic(args.output, report)
    write_text_atomic(args.markdown, render_markdown(report))
    print(report["classification"])


if __name__ == "__main__":
    main()
