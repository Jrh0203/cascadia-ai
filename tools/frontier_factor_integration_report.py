#!/usr/bin/env python3
"""Assemble and integrity-check the complete ADR 0097 cluster result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.graded_oracle_factor_integration import (
    EXPERIMENT_ID,
    FACTOR_ATTENTION,
    MLX_CACHE_LIMIT_BYTES,
    PAIRWISE_GATED,
    PROBE_KINDS,
    SCREEN_RELATIVE,
    WIDE_CONCAT,
    factor_integration_classification,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

HOSTS = {
    WIDE_CONCAT: ("john1", "john2"),
    SCREEN_RELATIVE: ("john2", "john3"),
    FACTOR_ATTENTION: ("john3", "john4"),
    PAIRWISE_GATED: ("john4", "john1"),
}
PEAK_MEMORY_LIMIT_BYTES = 6 * 1024**3


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def scientific_blake3(value: dict[str, Any]) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def normalize_host(value: str) -> str:
    normalized = value.lower().split(".")[0]
    if normalized in {"johns-mac-mini", "john1"}:
        return "john1"
    return normalized


def event_window(path: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event.get("event") == "started"]
    finishes = [event for event in events if event.get("event") == "finished"]
    if len(starts) != 1 or len(finishes) != 1:
        raise ValueError(f"incomplete execution event log {path}")
    start = starts[0]
    finish = finishes[0]
    if int(finish["return_code"]) != 0:
        raise ValueError(f"failed execution event log {path}")
    return {
        "started_unix_seconds": float(start["started_unix_seconds"]),
        "ended_unix_seconds": float(finish["ended_unix_seconds"]),
        "elapsed_seconds": float(finish["elapsed_seconds"]),
        "queued_seconds": float(start["queued_seconds"]),
    }


def expected_replay_scientific(
    kind: str,
    origin: dict[str, Any],
    weights: Path,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "weights_blake3": checksum(weights),
        "train_cache_payload_blake3": origin[
            "train_cache_payload_blake3"
        ],
        "validation_cache_payload_blake3": origin[
            "validation_cache_payload_blake3"
        ],
        "train": origin["train"],
        "validation": origin["validation"],
        "test_split_opened": False,
    }


def validate_execution_memory(execution: dict[str, Any]) -> dict[str, Any]:
    before = execution["mlx_memory_before_clear"]
    after = execution["mlx_memory_after_clear"]
    peak = int(before["peak_active_memory_bytes"])
    cache_before = int(before["cache_memory_bytes"])
    cache_after = int(after["cache_memory_bytes"])
    passed = (
        peak <= PEAK_MEMORY_LIMIT_BYTES
        and cache_before <= MLX_CACHE_LIMIT_BYTES + 128 * 1024**2
        and cache_after == 0
    )
    return {
        "peak_active_memory_bytes": peak,
        "cache_memory_before_clear_bytes": cache_before,
        "cache_memory_after_clear_bytes": cache_after,
        "passed": passed,
    }


def build_report(experiment_root: Path) -> dict[str, Any]:
    manifest = load_json(experiment_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("ADR 0097 experiment identity drifted")

    source_identities = {
        host: load_json(
            experiment_root / "source-identity-corrected" / f"{host}.json"
        )
        for host in ("john1", "john2", "john3", "john4")
    }
    source_hashes = {
        str(identity["bundle_sha256"])
        for identity in source_identities.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("corrected MLX source differs across hosts")

    audits = {
        host: load_json(
            experiment_root / "audits" / f"max-width-backward-{host}.json"
        )
        for host in ("john1", "john4")
    }
    if not all(bool(audit["passed"]) for audit in audits.values()):
        raise ValueError("maximum-width backward audit failed")

    origins: dict[str, dict[str, Any]] = {}
    replays: dict[str, dict[str, Any]] = {}
    probe_results: dict[str, dict[str, Any]] = {}
    windows = []
    productive_by_host = {host: 0.0 for host in source_identities}
    for kind in PROBE_KINDS:
        training_host, replay_host = HOSTS[kind]
        probe_root = experiment_root / "probes" / kind
        origin = load_json(probe_root / "report.json")
        replay = load_json(
            experiment_root / "cross-host" / f"{kind}-{replay_host}.json"
        )
        weights = probe_root / "best.safetensors"
        expected_scientific = expected_replay_scientific(
            kind,
            origin,
            weights,
        )
        if replay["scientific"] != expected_scientific:
            raise ValueError(f"{kind} cross-host metrics drifted")
        expected_hash = scientific_blake3(expected_scientific)
        if replay["scientific_blake3"] != expected_hash:
            raise ValueError(f"{kind} cross-host hash drifted")
        if normalize_host(origin["host"]) != training_host:
            raise ValueError(f"{kind} training host drifted")
        if normalize_host(replay["host"]) != replay_host:
            raise ValueError(f"{kind} replay host drifted")
        if origin["test_split_opened"] or replay["scientific"][
            "test_split_opened"
        ]:
            raise ValueError(f"{kind} opened the sealed test split")
        if int(origin["execution"]["process_swaps"]) != 0:
            raise ValueError(f"{kind} process swapped")
        if int(origin["execution"]["peak_process_rss_bytes"]) > (
            PEAK_MEMORY_LIMIT_BYTES
        ):
            raise ValueError(f"{kind} exceeded process RSS gate")
        origin_memory = validate_execution_memory(origin["execution"])
        replay_memory = validate_execution_memory(replay["execution"])
        if not origin_memory["passed"] or not replay_memory["passed"]:
            raise ValueError(f"{kind} failed MLX allocator gate")
        for split in ("train", "validation"):
            metrics = origin[split]
            if not (
                metrics["all_scores_finite"]
                and metrics["all_groups_scored_once"]
                and metrics["all_candidates_scored_once"]
            ):
                raise ValueError(f"{kind} {split} coverage failed")

        origin_window = event_window(
            experiment_root / f"events-{kind}-{training_host}.jsonl"
        )
        replay_window = event_window(
            experiment_root
            / f"events-cross-{kind}-{replay_host}.jsonl"
        )
        windows.extend([origin_window, replay_window])
        productive_by_host[training_host] += origin_window["elapsed_seconds"]
        productive_by_host[replay_host] += replay_window["elapsed_seconds"]
        origins[kind] = origin
        replays[kind] = replay
        probe_results[kind] = {
            "training_host": training_host,
            "replay_host": replay_host,
            "seed": origin["probe"]["seed"],
            "best_epoch": origin["best_epoch"],
            "weights_blake3": checksum(weights),
            "train": origin["train"],
            "validation": origin["validation"],
            "execution": origin["execution"],
            "memory": origin_memory,
            "cross_replay_memory": replay_memory,
            "cross_replay_scientific_blake3": expected_hash,
            "cross_replay_metrics_bit_identical": True,
        }

    classification = factor_integration_classification(origins)
    started = min(window["started_unix_seconds"] for window in windows)
    ended = max(window["ended_unix_seconds"] for window in windows)
    wall_seconds = ended - started
    status = (
        "mechanism_identified"
        if classification["classification"].endswith("_sufficient")
        else "rejected"
    )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        **classification,
        "source": {
            "files": next(iter(source_identities.values()))["files"],
            "bundle_sha256": next(iter(source_hashes)),
            "bit_identical_across_john1_john2_john3_john4": True,
        },
        "audits": {
            host: {
                "passed": audit["passed"],
                "candidate_count": audit["candidate_count"],
                "maximum_peak_active_memory_bytes": max(
                    value["mlx_memory_before_clear"][
                        "peak_active_memory_bytes"
                    ]
                    for value in audit["probes"].values()
                ),
                "system_swap_delta_bytes": audit["system_swap_delta_bytes"],
            }
            for host, audit in audits.items()
        },
        "probes": probe_results,
        "execution": {
            "first_job_started_unix_seconds": started,
            "final_job_ended_unix_seconds": ended,
            "probe_and_replay_wall_seconds": wall_seconds,
            "independent_hypotheses_completed": len(PROBE_KINDS),
            "hypotheses_per_hour": (
                len(PROBE_KINDS) * 3600.0 / max(wall_seconds, 1e-9)
            ),
            "productive_wall_seconds_by_host": productive_by_host,
        },
        "invalid_launch_partial_metrics_used": False,
        "factor_caches_regenerated_after_correction": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "external_compute_used": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ADR 0097 Candidate-Factor Integration Result",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "| Probe | Train recall | Train exact | Validation recall | "
        "Validation exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in PROBE_KINDS:
        probe = report["probes"][kind]
        lines.append(
            f"| {kind} | "
            f"{probe['train']['target_positive_recall']:.6f} | "
            f"{probe['train']['target_set_exact_fraction']:.6f} | "
            f"{probe['validation']['target_positive_recall']:.6f} | "
            f"{probe['validation']['target_set_exact_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "All four origin reports passed coverage, finite-score, memory, "
            "swap, source-identity, and ring-replay integrity checks.",
            "",
            f"Probe plus replay wall time was "
            f"{report['execution']['probe_and_replay_wall_seconds']:.2f} "
            "seconds, resolving "
            f"{report['execution']['hypotheses_per_hour']:.2f} independent "
            "hypotheses per hour.",
            "",
            "The invalid allocator-default launch contributed no selection "
            "or classification data. The sealed test split, gameplay, cloud, "
            "and external compute remained closed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.experiment_root)
    write_text_atomic(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    write_text_atomic(args.markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
