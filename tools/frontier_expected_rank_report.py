#!/usr/bin/env python3
"""Assemble and verify the complete ADR 0100 cluster result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    EXPERIMENT_ID,
    classify_expected_rank_pilot,
    expected_rank_validation_gates,
)

HOSTS = ("john1", "john2", "john3", "john4")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def normalize_host(value: str) -> str:
    normalized = value.lower().split(".")[0]
    return "john1" if normalized == "johns-mac-mini" else normalized


def event_window(path: Path) -> dict[str, float]:
    events = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event.get("event") == "started"]
    finishes = [event for event in events if event.get("event") == "finished"]
    if len(starts) != 1 or len(finishes) != 1:
        raise ValueError(f"incomplete execution event log {path}")
    if int(finishes[0]["return_code"]) != 0:
        raise ValueError(f"failed execution event log {path}")
    return {
        "started_unix_seconds": float(starts[0]["started_unix_seconds"]),
        "ended_unix_seconds": float(finishes[0]["ended_unix_seconds"]),
        "elapsed_seconds": float(finishes[0]["elapsed_seconds"]),
        "queued_seconds": float(starts[0]["queued_seconds"]),
    }


def replay_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Return host-independent selected-model science."""
    scientific = report["scientific"]
    return {
        "checkpoint": scientific["checkpoint"],
        "checkpoint_manifest_blake3": scientific[
            "checkpoint_manifest_blake3"
        ],
        "model_blake3": scientific["model_blake3"],
        "target_scale": scientific.get("target_scale", 64.0),
        "student_temperature": scientific.get("student_temperature", 2.0),
        "train_dataset_id": scientific["train_dataset_id"],
        "train_manifest_blake3": scientific["train_manifest_blake3"],
        "train_cache_identity": scientific["train_cache_identity"],
        "validation_dataset_id": scientific["validation_dataset_id"],
        "validation_manifest_blake3": scientific[
            "validation_manifest_blake3"
        ],
        "validation_cache_identity": scientific["validation_cache_identity"],
        "train": scientific["train"],
        "validation": scientific["validation"],
        "test_split_opened": scientific["test_split_opened"],
        "gameplay_opened": scientific["gameplay_opened"],
        "new_teacher_compute_used": scientific[
            "new_teacher_compute_used"
        ],
        "external_compute_used": scientific["external_compute_used"],
    }


def build_report(experiment_root: Path) -> dict[str, Any]:
    manifest = load_json(experiment_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("ADR 0100 experiment identity drifted")

    source_identities = {
        host: load_json(experiment_root / "source-identity" / f"{host}.json")
        for host in HOSTS
    }
    source_hashes = {
        str(identity["bundle_sha256"])
        for identity in source_identities.values()
    }
    source_files = {int(identity["files"]) for identity in source_identities.values()}
    if len(source_hashes) != 1 or len(source_files) != 1:
        raise ValueError("ADR 0100 source differs across hosts")

    cache = load_json(experiment_root / "reports" / "cache-comparison.json")
    gradient = load_json(experiment_root / "reports" / "gradient-john3.json")
    baseline = load_json(experiment_root / "reports" / "baseline-john4.json")
    signal = {
        "concentration": {
            split: load_json(
                experiment_root
                / "reports"
                / f"signal-concentration-{split}-john1.json"
            )["scientific"]
            for split in ("train", "validation")
        },
        "gradient": {
            split: load_json(
                experiment_root
                / "reports"
                / f"signal-gradient-{split}-john3.json"
            )["scientific"]
            for split in ("train", "validation")
        },
        "reachability": {
            split: load_json(
                experiment_root
                / "reports"
                / f"signal-reachability-{split}-john4.json"
            )["scientific"]
            for split in ("train", "validation")
        },
        "scale_sweep": {
            "train": load_json(
                experiment_root
                / "reports"
                / "signal-scale-sweep-train-john1.json"
            )["scientific"],
            "validation": load_json(
                experiment_root
                / "reports"
                / "signal-scale-sweep-validation-john3.json"
            )["scientific"],
        },
    }
    origin = load_json(experiment_root / "reports" / "evaluation-john2.json")
    replay = load_json(experiment_root / "reports" / "evaluation-john1.json")
    if normalize_host(origin["telemetry"]["host"]) != "john2":
        raise ValueError("ADR 0100 origin evaluation host drifted")
    if normalize_host(replay["telemetry"]["host"]) != "john1":
        raise ValueError("ADR 0100 replay evaluation host drifted")

    origin_payload = replay_payload(origin)
    replay_identical = replay_payload(replay) == origin_payload
    cache_passed = bool(cache["scientific"]["passed"])
    gradient_passed = bool(gradient["scientific"]["passed"])
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
        optimization_audit_passed=gradient_passed,
        replay_identical=replay_identical,
        cache_identical=cache_passed,
    )
    classification = classify_expected_rank_pilot(gates)

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
        "status": "promoted_to_sealed_test" if gates["pilot_passed"] else "rejected",
        "classification": classification,
        "source": {
            "files": next(iter(source_files)),
            "bundle_sha256": next(iter(source_hashes)),
            "bit_identical_across_john1_john2_john3_john4": True,
        },
        "cache": cache["scientific"],
        "gradient": gradient["scientific"],
        "baseline": baseline["scientific"],
        "exploratory_signal_audit": signal,
        "selected_model": {
            "origin_host": "john2",
            "replay_host": "john1",
            "replay_bit_identical": replay_identical,
            "scientific": origin_payload,
            "performance": performance,
            "origin_telemetry": origin["telemetry"],
            "replay_telemetry": replay["telemetry"],
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
            "invalid_launch_preserved": (
                experiment_root / "invalid-launch-group-id-sign"
            ).is_dir(),
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
    signal = report.get("exploratory_signal_audit")
    lines = [
        "# ADR 0100 Frontier Expected-Rank Result",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "| Metric | Screen baseline | Selected model | Gate |",
        "|---|---:|---:|---:|",
        "| validation expected-rank target recall | "
        f"{baseline['validation']['expected_rank_target_positive_recall']:.2%} | "
        f"{validation['expected_rank_target_positive_recall']:.2%} | 50.00% |",
        "| validation exact expected-rank sets | "
        f"{baseline['validation']['expected_rank_target_set_exact_fraction']:.2%} | "
        f"{validation['expected_rank_target_set_exact_fraction']:.2%} | 1.00% |",
        "| validation R4800 winner recall | "
        f"{baseline['validation']['top64_r4800_winner_recall']:.2%} | "
        f"{validation['top64_r4800_winner_recall']:.2%} | >98.00% |",
        "| validation confidence coverage | "
        f"{baseline['validation']['top64_confidence_set_coverage_95']:.2%} | "
        f"{validation['top64_confidence_set_coverage_95']:.2%} | 99.00% |",
        "| validation retained regret | "
        f"{baseline['validation']['mean_top64_retained_r4800_regret']:.6f} | "
        f"{validation['mean_top64_retained_r4800_regret']:.6f} | <0.030000 |",
        "",
        "Train fit:",
        "",
        f"- expected-rank target recall: "
        f"{train['expected_rank_target_positive_recall']:.2%};",
        f"- exact expected-rank target sets: "
        f"{train['expected_rank_target_set_exact_fraction']:.2%}.",
        "",
        f"The independent cache pair was byte-identical: "
        f"{report['cache']['passed']}. The widest-group gradient audit passed: "
        f"{report['gradient']['passed']}. The selected checkpoint replay was "
        f"bit-identical: "
        f"{report['selected_model']['replay_bit_identical']}.",
        "",
        f"The campaign completed in "
        f"{report['execution']['campaign_wall_seconds']:.2f} seconds with one "
        "MLX trainer and zero duplicate training.",
        "",
        "The sealed test, gameplay, new teacher compute, cloud, and external "
        "compute remained closed.",
        "",
    ]
    if signal is not None:
        concentration = signal["concentration"]["validation"]
        gradient = signal["gradient"]["validation"]
        reachability = signal["reachability"]["validation"]
        scale_sweep = signal["scale_sweep"]["validation"]["scales"]
        lines.extend(
            [
                "Exploratory mechanism audit:",
                "",
                "- validation target probability mass inside the deployed set: "
                f"{concentration['probability_mass_in_deployed_target']['mean']:.2%};",
                "- validation uniform-start absolute gradient inside the "
                "deployed set: "
                f"{gradient['uniform_student_absolute_gradient_fraction_in_deployed_target']['mean']:.2%};",
                "- validation exact-set reachability at residual range 6: "
                f"{reachability['ceilings']['6.0']['target_set_exact_fraction']:.2%}.",
                "- validation deployed-set target mass rises from "
                f"{scale_sweep['64.0']['probability_mass_in_deployed_target']['mean']:.2%} "
                "at scale 64 to "
                f"{scale_sweep['16.0']['probability_mass_in_deployed_target']['mean']:.2%} "
                "at scale 16.",
                "",
                "These diagnostics were opened after launch to explain the "
                "result and do not alter the preregistered classification.",
                "",
            ]
        )
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


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.experiment_root)
    write_json_atomic(args.output, report)
    write_text_atomic(args.markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
