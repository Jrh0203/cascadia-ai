#!/usr/bin/env python3
"""Replay an ADR 0150 checkpoint on a different Apple Silicon host."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.r3_action_edit_mlx_benchmark import (
    run_isolated_serving_benchmark,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ADR_ID,
    ARMS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    open_data_verification_id,
)

REPLAY_SCHEMA_VERSION = 1
REPLAY_KIND = "cross-host-checkpoint-portability"


def _canonical_blake3(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake3.blake3(payload).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_host(host: str) -> str:
    lowered = host.lower()
    for known in ("john1", "john2", "john3", "john4"):
        if known in lowered:
            return known
    return host


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("origin performance denominator must be positive")
    return numerator / denominator


def build_replay_report(
    *,
    origin: dict[str, Any],
    authorization: dict[str, Any],
    performance: dict[str, Any],
    checkpoint: Path,
    replay_host: str,
) -> dict[str, Any]:
    """Bind a fresh-host service result to its immutable origin checkpoint."""
    if (
        origin.get("experiment_id") != EXPERIMENT_ID
        or origin.get("protocol_id") != PROTOCOL_ID
        or origin.get("adr") != ADR_ID
        or origin.get("arm") not in ARMS
        or origin.get("mode") != "production"
    ):
        raise ValueError("origin report is not a production ADR 0150 arm")
    if (
        authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("protocol_id") != PROTOCOL_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
    ):
        raise ValueError("authorization is not an approved ADR 0150 control")

    identity = authorization.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("authorization identity is missing")
    open_data = identity.get("open_data_verification")
    if not isinstance(open_data, dict):
        raise ValueError("authorization open-data proof is missing")
    proof_id = open_data_verification_id(open_data)
    if proof_id != identity.get("open_data_verification_id"):
        raise ValueError("authorization open-data proof digest differs")

    origin_checkpoint = origin.get("checkpoint")
    if not isinstance(origin_checkpoint, dict):
        raise ValueError("origin checkpoint identity is missing")
    observed_manifest = _checksum(checkpoint / "checkpoint.json")
    observed_model = _checksum(checkpoint / "model.safetensors")
    if (
        observed_manifest != origin_checkpoint.get("manifest_blake3")
        or observed_model != origin_checkpoint.get("model_blake3")
    ):
        raise ValueError("replay checkpoint bytes differ from the origin")

    measurement = performance.get("measurement")
    complete = performance.get("complete_decisions")
    fixed = performance.get("fixed_chunk")
    memory = performance.get("memory")
    if not all(isinstance(value, dict) for value in (measurement, complete, fixed, memory)):
        raise ValueError("replay performance payload is incomplete")
    worker_runtime = measurement.get("worker_runtime")
    if not isinstance(worker_runtime, dict):
        raise ValueError("replay worker runtime is missing")
    normalized_replay_host = _normalize_host(replay_host)
    if _normalize_host(str(worker_runtime.get("host", ""))) != normalized_replay_host:
        raise ValueError("replay benchmark ran on an unexpected host")
    if (
        measurement.get("isolated_process") is not True
        or measurement.get("checkpoint_model_blake3") != observed_model
        or measurement.get("open_data_verification_id") != proof_id
        or measurement.get("verification_source") != "cluster-preflight"
    ):
        raise ValueError("replay benchmark identity is inconsistent")

    origin_performance = origin.get("performance")
    if not isinstance(origin_performance, dict):
        raise ValueError("origin performance payload is missing")
    origin_complete = origin_performance["complete_decisions"]
    origin_fixed = origin_performance["fixed_chunk"]
    origin_memory = origin_performance["memory"]
    comparison = {
        "complete_action_throughput_ratio": _ratio(
            float(complete["action_scores_per_second"]),
            float(origin_complete["action_scores_per_second"]),
        ),
        "complete_p99_latency_ratio": _ratio(
            float(complete["latency_milliseconds"]["p99"]),
            float(origin_complete["latency_milliseconds"]["p99"]),
        ),
        "fixed_chunk_throughput_ratio": _ratio(
            float(fixed["action_scores_per_second"]),
            float(origin_fixed["action_scores_per_second"]),
        ),
        "peak_active_memory_ratio": _ratio(
            float(memory["peak_active_bytes"]),
            float(origin_memory["peak_active_bytes"]),
        ),
        "peak_process_rss_ratio": _ratio(
            float(memory["peak_process_rss_bytes"]),
            float(origin_memory["peak_process_rss_bytes"]),
        ),
    }
    assertions = {
        "checkpoint_manifest_identical": True,
        "checkpoint_model_identical": True,
        "different_host": normalized_replay_host != origin.get("host"),
        "isolated_process": True,
        "model_load_and_open_data_reverification_passed": True,
        "positive_complete_throughput": float(complete["action_scores_per_second"]) > 0.0,
        "positive_fixed_chunk_throughput": float(fixed["action_scores_per_second"]) > 0.0,
        "process_swaps_zero": int(memory["process_swaps"]) == 0,
    }
    scientific_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "replay_kind": REPLAY_KIND,
        "arm": origin["arm"],
        "origin_host": origin["host"],
        "replay_host": normalized_replay_host,
        "origin_report_id": origin["report_id"],
        "authorization_id": authorization["authorization_id"],
        "checkpoint": {
            "manifest_blake3": observed_manifest,
            "model_blake3": observed_model,
            "global_step": int(origin["optimization"]["global_step"]),
        },
        "open_data_verification_id": proof_id,
        "assertions": assertions,
        "classifier_eligible": False,
        "scientific_use": "operational-portability-diagnostic-only",
    }
    report = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        **scientific_identity,
        "scientific_identity": scientific_identity,
        "origin_performance": origin_performance,
        "replay_performance": performance,
        "host_performance_comparison": comparison,
        "operational_pass": all(assertions.values()),
    }
    report["report_id"] = _canonical_blake3(report)
    return report


def _latest_checkpoint(run_dir: Path) -> Path:
    latest = _read_json(run_dir / "latest.json", "latest checkpoint pointer")
    name = latest.get("checkpoint")
    if not isinstance(name, str) or not name:
        raise ValueError("latest checkpoint pointer is malformed")
    checkpoint = run_dir / "checkpoints" / name
    if not checkpoint.is_dir():
        raise ValueError(f"latest checkpoint does not exist: {checkpoint}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-report", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--steady-iterations", type=int, default=30)
    args = parser.parse_args()

    origin = _read_json(args.origin_report, "origin report")
    authorization = _read_json(args.authorization, "authorization")
    checkpoint = _latest_checkpoint(args.run_dir)
    identity = authorization["identity"]
    performance = run_isolated_serving_benchmark(
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        cache=args.cache,
        s1_cache=args.s1_cache,
        run_dir=args.run_dir,
        checkpoint=checkpoint,
        arm=origin["arm"],
        global_step=int(origin["optimization"]["global_step"]),
        open_data_verification=identity["open_data_verification"],
        verification_source="cluster-preflight",
        warmup_iterations=args.warmup_iterations,
        steady_iterations=args.steady_iterations,
    )
    report = build_replay_report(
        origin=origin,
        authorization=authorization,
        performance=performance,
        checkpoint=checkpoint,
        replay_host=socket.gethostname().split(".")[0],
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["operational_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
