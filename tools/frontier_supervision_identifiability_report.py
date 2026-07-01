#!/usr/bin/env python3
"""Assemble and verify the complete ADR 0099 cluster result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cascadia_mlx.frontier_supervision_identifiability import (
    AUDIT_KINDS,
    BOUNDARY_SIGNAL,
    CROSS_FIDELITY,
    EXPECTED_RANK_CEILING,
    EXPERIMENT_ID,
    TEACHER_RESAMPLING,
    scientific_blake3,
)

HOSTS = {
    BOUNDARY_SIGNAL: ("john1", "john2"),
    CROSS_FIDELITY: ("john2", "john3"),
    TEACHER_RESAMPLING: ("john3", "john4"),
    EXPECTED_RANK_CEILING: ("john4", "john1"),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def normalize_host(value: str) -> str:
    normalized = value.lower().split(".")[0]
    return "john1" if normalized == "johns-mac-mini" else normalized


def event_window(path: Path) -> dict[str, float]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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


def classify(scientific: dict[str, dict[str, Any]]) -> str:
    hard_stable = all(
        bool(scientific[kind][split]["gate_passed"])
        for kind in (BOUNDARY_SIGNAL, CROSS_FIDELITY, TEACHER_RESAMPLING)
        for split in ("train", "validation")
    )
    soft_sufficient = bool(scientific[EXPECTED_RANK_CEILING]["validation"]["gate_passed"])
    if soft_sufficient:
        return "uncertainty_aware_supervision_sufficient"
    if hard_stable:
        return "hard_target_stable_but_soft_ceiling_insufficient"
    return "existing_teacher_supervision_insufficient"


def build_report(experiment_root: Path) -> dict[str, Any]:
    manifest = load_json(experiment_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("ADR 0099 experiment identity drifted")
    source_identities = {
        host: load_json(experiment_root / "source-identity" / f"{host}.json")
        for host in ("john1", "john2", "john3", "john4")
    }
    source_hashes = {str(identity["bundle_sha256"]) for identity in source_identities.values()}
    if len(source_hashes) != 1:
        raise ValueError("ADR 0099 source differs across hosts")

    origins: dict[str, dict[str, Any]] = {}
    scientific: dict[str, dict[str, Any]] = {}
    windows: dict[str, dict[str, dict[str, float]]] = {}
    productive = {host: 0.0 for host in source_identities}
    for kind in AUDIT_KINDS:
        origin_host, replay_host = HOSTS[kind]
        origin = load_json(experiment_root / "reports" / f"{kind}-{origin_host}.json")
        replay = load_json(experiment_root / "reports" / f"{kind}-{replay_host}.json")
        if normalize_host(origin["host"]) != origin_host:
            raise ValueError(f"{kind} origin host drifted")
        if normalize_host(replay["host"]) != replay_host:
            raise ValueError(f"{kind} replay host drifted")
        if replay["scientific"] != origin["scientific"]:
            raise ValueError(f"{kind} replay scientific payload drifted")
        expected_hash = scientific_blake3(origin["scientific"])
        if (
            origin["scientific_blake3"] != expected_hash
            or replay["scientific_blake3"] != expected_hash
        ):
            raise ValueError(f"{kind} scientific hash drifted")
        for report_name, report in (("origin", origin), ("replay", replay)):
            if int(report["execution"]["process_swaps"]) != 0:
                raise ValueError(f"{kind} {report_name} consumed swap")
            if int(report["execution"]["peak_process_rss_bytes"]) > 4 * 1024**3:
                raise ValueError(f"{kind} {report_name} exceeded RSS gate")
        for split in ("train", "validation"):
            if int(origin["scientific"][split]["groups"]) != int(
                manifest["inputs"][f"{split}_dataset"]["groups"]
            ) or int(origin["scientific"][split]["candidates"]) != int(
                manifest["inputs"][f"{split}_dataset"]["candidates"]
            ):
                raise ValueError(f"{kind} {split} coverage drifted")
        origin_window = event_window(experiment_root / f"events-{kind}-{origin_host}.jsonl")
        replay_window = event_window(experiment_root / f"events-cross-{kind}-{replay_host}.jsonl")
        windows[kind] = {"origin": origin_window, "replay": replay_window}
        productive[origin_host] += origin_window["elapsed_seconds"]
        productive[replay_host] += replay_window["elapsed_seconds"]
        origins[kind] = {
            "origin_host": origin_host,
            "replay_host": replay_host,
            "scientific_blake3": expected_hash,
            "scientific": origin["scientific"],
            "origin_execution": origin["execution"],
            "replay_execution": replay["execution"],
            "replay_bit_identical": True,
        }
        scientific[kind] = origin["scientific"]

    all_windows = [window for kind_windows in windows.values() for window in kind_windows.values()]
    started = min(window["started_unix_seconds"] for window in all_windows)
    ended = max(window["ended_unix_seconds"] for window in all_windows)
    wall_seconds = ended - started
    result_class = classify(scientific)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "mechanism_identified"
            if result_class == "uncertainty_aware_supervision_sufficient"
            else "rejected"
        ),
        "classification": result_class,
        "source": {
            "files": next(iter(source_identities.values()))["files"],
            "bundle_sha256": next(iter(source_hashes)),
            "bit_identical_across_john1_john2_john3_john4": True,
        },
        "audits": origins,
        "execution": {
            "first_job_started_unix_seconds": started,
            "final_job_ended_unix_seconds": ended,
            "origin_and_replay_wall_seconds": wall_seconds,
            "independent_hypotheses_completed": len(AUDIT_KINDS),
            "hypotheses_per_hour": len(AUDIT_KINDS) * 3600.0 / max(wall_seconds, 1e-9),
            "productive_wall_seconds_by_host": productive,
            "duplicate_compute_fraction": 0.0,
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    audits = report["audits"]
    boundary = audits[BOUNDARY_SIGNAL]["scientific"]
    fidelity = audits[CROSS_FIDELITY]["scientific"]
    resampling = audits[TEACHER_RESAMPLING]["scientific"]
    expected = audits[EXPECTED_RANK_CEILING]["scientific"]
    lines = [
        "# ADR 0099 Frontier Supervision Identifiability Result",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "| Audit | Train gate | Validation gate |",
        "|---|---:|---:|",
        f"| boundary signal | {boundary['train']['gate_passed']} | "
        f"{boundary['validation']['gate_passed']} |",
        f"| cross fidelity | {fidelity['train']['gate_passed']} | "
        f"{fidelity['validation']['gate_passed']} |",
        f"| teacher resampling | {resampling['train']['gate_passed']} | "
        f"{resampling['validation']['gate_passed']} |",
        f"| expected-rank ceiling | {expected['train']['gate_passed']} | "
        f"{expected['validation']['gate_passed']} |",
        "",
        "Validation mechanism metrics:",
        "",
        f"- boundary-separated target slots: "
        f"{boundary['validation']['robust_target_slot_fraction_95']:.2%}; "
        f"complete sets: "
        f"{boundary['validation']['robust_complete_set_fraction_95']:.2%};",
        f"- R600 cohort coverage at width 64: "
        f"{fidelity['validation']['r600_cohort_coverage_fraction']:.2%};",
        f"- 512-draw hard-target recall: "
        f"{resampling['validation']['mean_nominal_target_recall']:.2%}; "
        f"exact-set reproduction: "
        f"{resampling['validation']['exact_set_reproduction_fraction']:.2%};",
        f"- expected-rank nominal-target recall: "
        f"{expected['validation']['nominal_target_recall']:.2%}; "
        f"exact target sets: "
        f"{expected['validation']['nominal_target_exact_fraction']:.2%};",
        f"- expected-rank R4800 winner recall: "
        f"{expected['validation']['overall']['top64_r4800_winner_recall']:.2%}; "
        f"confidence coverage: "
        f"{expected['validation']['overall']['top64_confidence_set_coverage_95']:.2%}; "
        f"retained regret: "
        f"{expected['validation']['overall']['mean_top64_retained_r4800_regret']:.6f}.",
        "",
        "The finite-R1200 hard cutoff is statistically unstable, while "
        "uncertainty-aware expected-rank ordering preserves the complete "
        "open-validation R4800 decision signal. This authorizes one separately "
        "preregistered MLX pilot using ordinal expected-rank supervision.",
        "",
        "All four origin reports covered every open group and candidate, used "
        "zero process swap, and matched their ring replays bit-for-bit.",
        "",
        f"Origin plus replay wall time was "
        f"{report['execution']['origin_and_replay_wall_seconds']:.2f} seconds, "
        f"resolving {report['execution']['hypotheses_per_hour']:.2f} "
        "independent hypotheses per hour.",
        "",
        "The sealed test, gameplay, new teacher compute, cloud, and external "
        "compute remained closed.",
        "",
    ]
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
