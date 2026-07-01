#!/usr/bin/env python3
"""Assemble and integrity-check the complete ADR 0098 cluster result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.graded_oracle_factor_integration import (
    MLX_CACHE_LIMIT_BYTES,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.graded_oracle_raw_factor_construction import (
    COMPLETE_RAW_FLAT,
    EXACT_LOCAL_RELATION,
    EXPERIMENT_ID,
    EXPLICIT_MARKET_TRANSITION,
    FRESH_ENTITY_CROSS,
    PROBE_KINDS,
    raw_factor_construction_classification,
)

HOSTS = {
    FRESH_ENTITY_CROSS: ("john1", "john2"),
    COMPLETE_RAW_FLAT: ("john2", "john3"),
    EXACT_LOCAL_RELATION: ("john3", "john4"),
    EXPLICIT_MARKET_TRANSITION: ("john4", "john1"),
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
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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
        "parameter_count": origin["parameter_count"],
        "train_dataset_manifest_blake3": origin["train_dataset_manifest_blake3"],
        "validation_dataset_manifest_blake3": origin["validation_dataset_manifest_blake3"],
        "train": origin["train"],
        "validation": origin["validation"],
        "test_split_opened": False,
        "gameplay_opened": False,
        "external_compute_used": False,
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


def validate_audit(audit: dict[str, Any], kind: str, host: str) -> None:
    if not bool(audit["passed"]):
        raise ValueError(f"{kind} maximum-width audit failed on {host}")
    if audit["kind"] != kind or normalize_host(audit["host"]) != host:
        raise ValueError(f"{kind} maximum-width audit identity drifted")
    if int(audit["candidate_count"]) != 10854:
        raise ValueError(f"{kind} maximum-width candidate count drifted")
    if audit["test_split_opened"] or audit["gameplay_opened"] or audit["external_compute_used"]:
        raise ValueError(f"{kind} maximum-width audit opened a closed domain")


def build_report(experiment_root: Path) -> dict[str, Any]:
    manifest = load_json(experiment_root / "manifest.json")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("ADR 0098 experiment identity drifted")

    source_identities = {
        host: load_json(experiment_root / "source-identity" / f"{host}.json")
        for host in ("john1", "john2", "john3", "john4")
    }
    source_hashes = {str(identity["bundle_sha256"]) for identity in source_identities.values()}
    if len(source_hashes) != 1:
        raise ValueError("ADR 0098 MLX source differs across hosts")

    audits: dict[str, dict[str, dict[str, Any]]] = {}
    for host in ("john1", "john4"):
        audits[host] = {}
        for kind in PROBE_KINDS:
            audit = load_json(experiment_root / "audits" / f"max-width-{kind}-{host}.json")
            validate_audit(audit, kind, host)
            audits[host][kind] = audit

    origins: dict[str, dict[str, Any]] = {}
    probe_results: dict[str, dict[str, Any]] = {}
    train_windows: dict[str, dict[str, Any]] = {}
    replay_windows: dict[str, dict[str, Any]] = {}
    productive_by_host = {host: 0.0 for host in ("john1", "john2", "john3", "john4")}
    for kind in PROBE_KINDS:
        training_host, replay_host = HOSTS[kind]
        probe_root = experiment_root / "probes" / kind
        origin = load_json(probe_root / "report.json")
        replay = load_json(experiment_root / "cross-host" / f"{kind}-{replay_host}.json")
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
        for domain in (
            "test_split_opened",
            "gameplay_opened",
            "external_compute_used",
        ):
            if origin[domain] or replay["scientific"][domain]:
                raise ValueError(f"{kind} opened closed domain {domain}")
        for execution_name, execution in (
            ("training", origin["execution"]),
            ("replay", replay["execution"]),
        ):
            if int(execution["process_swaps"]) != 0:
                raise ValueError(f"{kind} {execution_name} process swapped")
            if int(execution["peak_process_rss_bytes"]) > (PEAK_MEMORY_LIMIT_BYTES):
                raise ValueError(f"{kind} {execution_name} exceeded RSS gate")
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

        train_window = event_window(experiment_root / f"events-{kind}-{training_host}.jsonl")
        replay_window = event_window(experiment_root / f"events-cross-{kind}-{replay_host}.jsonl")
        train_windows[kind] = train_window
        replay_windows[kind] = replay_window
        productive_by_host[training_host] += train_window["elapsed_seconds"]
        productive_by_host[replay_host] += replay_window["elapsed_seconds"]
        origins[kind] = origin
        probe_results[kind] = {
            "training_host": training_host,
            "replay_host": replay_host,
            "seed": origin["probe"]["seed"],
            "best_epoch": origin["best_epoch"],
            "parameter_count": origin["parameter_count"],
            "weights_blake3": checksum(weights),
            "train": origin["train"],
            "validation": origin["validation"],
            "execution": origin["execution"],
            "memory": origin_memory,
            "cross_replay_execution": replay["execution"],
            "cross_replay_memory": replay_memory,
            "cross_replay_scientific_blake3": expected_hash,
            "cross_replay_metrics_bit_identical": True,
        }

    classification = raw_factor_construction_classification(origins)
    windows = [*train_windows.values(), *replay_windows.values()]
    started = min(window["started_unix_seconds"] for window in windows)
    ended = max(window["ended_unix_seconds"] for window in windows)
    wall_seconds = ended - started
    dependency_blocked_by_host = {host: 0.0 for host in productive_by_host}
    idle_with_work_by_host = {host: 0.0 for host in productive_by_host}
    for kind, replay_window in replay_windows.items():
        _training_host, replay_host = HOSTS[kind]
        destination_train_kind = next(
            candidate for candidate, (host, _replay) in HOSTS.items() if host == replay_host
        )
        destination_free = train_windows[destination_train_kind]["ended_unix_seconds"]
        source_ready = train_windows[kind]["ended_unix_seconds"]
        ready = max(destination_free, source_ready)
        dependency_blocked_by_host[replay_host] += max(
            0.0,
            source_ready - destination_free,
        )
        idle_with_work_by_host[replay_host] += max(
            0.0,
            replay_window["started_unix_seconds"] - ready,
        )

    status = (
        "mechanism_identified"
        if classification["classification"] == "raw_factor_construction_sufficient"
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
                kind: {
                    "passed": audit["passed"],
                    "candidate_count": audit["candidate_count"],
                    "peak_active_memory_bytes": audit["update_mlx_memory"][
                        "peak_active_memory_bytes"
                    ],
                    "system_swap_delta_bytes": audit["system_swap_delta_bytes"],
                }
                for kind, audit in host_audits.items()
            }
            for host, host_audits in audits.items()
        },
        "probes": probe_results,
        "execution": {
            "first_job_started_unix_seconds": started,
            "final_job_ended_unix_seconds": ended,
            "probe_and_replay_wall_seconds": wall_seconds,
            "independent_hypotheses_completed": len(PROBE_KINDS),
            "hypotheses_per_hour": (len(PROBE_KINDS) * 3600.0 / max(wall_seconds, 1e-9)),
            "productive_wall_seconds_by_host": productive_by_host,
            "dependency_blocked_idle_seconds_by_host": (dependency_blocked_by_host),
            "idle_with_compatible_work_queued_seconds_by_host": (idle_with_work_by_host),
            "duplicate_compute_fraction": 0.0,
        },
        "derived_feature_cache_created": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "external_compute_used": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_kind"] or "none"
    lines = [
        "# ADR 0098 Raw Factor Construction Result",
        "",
        f"Classification: `{report['classification']}`",
        "",
        f"Selected construction: `{selected}`",
        "",
        "| Probe | Train recall | Train exact | Validation recall | Validation exact |",
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
            "All four origin reports passed exact coverage, finite-score, "
            "memory, swap, source-identity, maximum-width, and ring-replay "
            "integrity checks.",
            "",
            "Every arm missed the 80% train-recall and 25% exact-train-set "
            "gates. Another neural constructor, head, pool, width increase, "
            "or optimizer variation is therefore closed; the next experiment "
            "must audit target learnability and supervision structure.",
            "",
            f"Probe plus replay wall time was "
            f"{report['execution']['probe_and_replay_wall_seconds']:.2f} "
            "seconds, resolving "
            f"{report['execution']['hypotheses_per_hour']:.2f} independent "
            "hypotheses per hour.",
            "",
            "No derived feature cache was created. The sealed test split, "
            "gameplay, cloud, and external compute remained closed.",
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
