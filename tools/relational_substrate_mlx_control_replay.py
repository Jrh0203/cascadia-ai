#!/usr/bin/env python3
"""Benchmark the exact ADR 0161 C0 checkpoint on one treatment host."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import blake3
import numpy as np
from cascadia_mlx.relational_substrate_mlx_benchmark import (
    run_isolated_serving_benchmark,
)
from cascadia_mlx.relational_substrate_mlx_cache import (
    ADR_ID,
    ARMS,
    CONTROL_ARM,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    open_data_verification_id,
)
from cascadia_mlx.relational_substrate_mlx_train import (
    ARM_HOSTS,
    TRAINING_STEPS,
)

REPLAY_SCHEMA_VERSION = 1
REPLAY_KIND = "same-host-exact-c0-serving-control-with-r6"
VALIDATION_GROUPS = 240
VALIDATION_ACTIONS = 860_203


def build_replay_report(
    *,
    control_report: dict[str, Any],
    authorization: dict[str, Any],
    performance: dict[str, Any],
    checkpoint: Path,
    r6_binary: Path,
    treatment_arm: str,
    replay_host: str,
) -> dict[str, Any]:
    """Bind one same-host C0 serving measurement to production evidence."""
    normalized_host = _normalize_host(replay_host)
    expected_host = ARM_HOSTS.get(treatment_arm)
    if treatment_arm not in ARMS[1:] or expected_host != normalized_host:
        raise ValueError(
            "ADR 0161 C0 replay treatment/host assignment is invalid"
        )
    if (
        control_report.get("schema_version") != 1
        or control_report.get("experiment_id") != EXPERIMENT_ID
        or control_report.get("protocol_id") != PROTOCOL_ID
        or control_report.get("adr") != ADR_ID
        or control_report.get("mode") != "production"
        or control_report.get("arm") != CONTROL_ARM
        or control_report.get("host") != ARM_HOSTS[CONTROL_ARM]
        or control_report.get("optimization", {}).get("global_step")
        != TRAINING_STEPS
    ):
        raise ValueError(
            "C0 origin report is not the completed ADR 0161 control"
        )
    control_identity = control_report.get("scientific_identity")
    if not isinstance(control_identity, dict) or _canonical_blake3(
        control_identity
    ) != control_report.get("report_id"):
        raise ValueError("C0 origin report identity is malformed")

    authorization_identity = authorization.get("identity")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("protocol_id") != PROTOCOL_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization_identity, dict)
        or _canonical_blake3(authorization_identity)
        != authorization.get("authorization_id")
    ):
        raise ValueError(
            "ADR 0161 authorization is stale or malformed"
        )
    open_data = authorization_identity.get("open_data_verification")
    if not isinstance(open_data, dict):
        raise ValueError(
            "ADR 0161 authorization lacks its open-data proof"
        )
    proof_id = open_data_verification_id(open_data)
    if proof_id != authorization_identity.get(
        "open_data_verification_id"
    ):
        raise ValueError(
            "ADR 0161 authorization open-data identity differs"
        )

    origin_checkpoint = control_report.get("checkpoint")
    if not isinstance(origin_checkpoint, dict):
        raise ValueError("C0 origin checkpoint identity is missing")
    observed_manifest = _checksum(checkpoint / "checkpoint.json")
    observed_model = _checksum(checkpoint / "model.safetensors")
    observed_r6 = _checksum(r6_binary)
    if (
        observed_manifest != origin_checkpoint.get("manifest_blake3")
        or observed_model != origin_checkpoint.get("model_blake3")
    ):
        raise ValueError(
            "host-paired C0 checkpoint bytes differ from john1"
        )
    if observed_r6 != authorization_identity.get("r6_binary_blake3"):
        raise ValueError(
            "host-paired R6 replay binary differs from authorization"
        )

    measurement = performance.get("measurement")
    combined = performance.get("combined_with_r6")
    fixed = performance.get("fixed_chunk")
    memory = performance.get("memory")
    r6 = performance.get("r6_apply_undo")
    if not all(
        isinstance(value, dict)
        for value in (measurement, combined, fixed, memory, r6)
    ):
        raise ValueError(
            "host-paired C0 performance payload is incomplete"
        )
    worker_runtime = measurement.get("worker_runtime")
    if (
        not isinstance(worker_runtime, dict)
        or _normalize_host(str(worker_runtime.get("host", "")))
        != normalized_host
        or measurement.get("isolated_process") is not True
        or measurement.get("checkpoint_model_blake3") != observed_model
        or measurement.get("open_data_verification_id") != proof_id
        or measurement.get("verification_source")
        != "cluster-preflight"
        or combined.get("groups") != VALIDATION_GROUPS
        or combined.get("actions") != VALIDATION_ACTIONS
        or combined.get("r6_exact_parity_pass") is not True
        or r6.get("exact_parity_pass") is not True
        or r6.get("apply_failures") != 0
        or r6.get("undo_failures") != 0
    ):
        raise ValueError(
            "host-paired C0 benchmark identity or coverage differs"
        )

    assertions = {
        "control_checkpoint_manifest_identical": True,
        "control_checkpoint_model_identical": True,
        "r6_binary_identical": True,
        "replay_host_is_treatment_host": normalized_host == expected_host,
        "replay_host_differs_from_control_host": (
            normalized_host != ARM_HOSTS[CONTROL_ARM]
        ),
        "isolated_process": True,
        "open_data_reverified": True,
        "all_validation_decisions_measured": True,
        "all_validation_actions_measured": True,
        "r6_apply_undo_exact": True,
    }
    scientific_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "replay_kind": REPLAY_KIND,
        "treatment_arm": treatment_arm,
        "host": normalized_host,
        "control_arm": CONTROL_ARM,
        "control_report_id": control_report["report_id"],
        "authorization_id": authorization["authorization_id"],
        "r3_cache_id": control_report["r3_cache_id"],
        "relational_cache_id": control_report[
            "relational_cache_id"
        ],
        "s1_cache_id": control_report["s1_cache_id"],
        "checkpoint": {
            "manifest_blake3": observed_manifest,
            "model_blake3": observed_model,
            "global_step": TRAINING_STEPS,
        },
        "r6_binary_blake3": observed_r6,
        "open_data_verification_id": proof_id,
        "benchmark_request_id": measurement["request_id"],
        "benchmark_result_id": measurement["result_id"],
        "assertions": assertions,
    }
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "treatment_arm": treatment_arm,
        "host": normalized_host,
        "control_arm": CONTROL_ARM,
        "control_report_id": control_report["report_id"],
        "scientific_identity": scientific_identity,
        "performance": performance,
        "replay_id": _canonical_blake3(scientific_identity),
    }


def _latest_checkpoint(run_dir: Path) -> Path:
    latest = _read_json(
        run_dir / "latest.json",
        "latest checkpoint pointer",
    )
    name = latest.get("checkpoint")
    if not isinstance(name, str) or not name:
        raise ValueError("latest checkpoint pointer is malformed")
    checkpoint = run_dir / "checkpoints" / name
    if not checkpoint.is_dir():
        raise ValueError(
            f"latest checkpoint does not exist: {checkpoint}"
        )
    return checkpoint


def _complete_validation_rows(
    control_report: dict[str, Any],
) -> np.ndarray:
    metrics = control_report.get("metrics")
    if (
        not isinstance(metrics, dict)
        or metrics.get("groups") != VALIDATION_GROUPS
        or metrics.get("expected_groups") != VALIDATION_GROUPS
        or metrics.get("candidates") != VALIDATION_ACTIONS
        or metrics.get("expected_candidates") != VALIDATION_ACTIONS
        or metrics.get("all_groups_scored_once") is not True
        or metrics.get("all_candidates_scored_once") is not True
    ):
        raise ValueError(
            "C0 origin report does not certify validation coverage"
        )
    return np.arange(VALIDATION_GROUPS, dtype=np.int64)


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


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
    return host.removesuffix(".local")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument(
        "--treatment-arm",
        choices=ARMS[1:],
        required=True,
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument(
        "--relational-cache",
        type=Path,
        required=True,
    )
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--r6-binary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--steady-iterations", type=int, default=30)
    args = parser.parse_args()

    control_report = _read_json(
        args.control_report,
        "C0 origin report",
    )
    authorization = _read_json(
        args.authorization,
        "ADR 0161 authorization",
    )
    checkpoint = _latest_checkpoint(args.run_dir)
    decision_rows = _complete_validation_rows(control_report)
    benchmark_artifact_dir = (
        args.output.parent
        / "paired-control-benchmarks"
        / args.treatment_arm
    )
    performance = run_isolated_serving_benchmark(
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        r3_cache=args.r3_cache,
        relational_cache=args.relational_cache,
        s1_cache=args.s1_cache,
        r6_binary=args.r6_binary,
        run_dir=args.run_dir,
        checkpoint=checkpoint,
        arm=CONTROL_ARM,
        global_step=TRAINING_STEPS,
        open_data_verification=authorization["identity"][
            "open_data_verification"
        ],
        verification_source="cluster-preflight",
        warmup_iterations=args.warmup_iterations,
        steady_iterations=args.steady_iterations,
        decision_rows=decision_rows,
        artifact_dir=benchmark_artifact_dir,
    )
    report = build_replay_report(
        control_report=control_report,
        authorization=authorization,
        performance=performance,
        checkpoint=checkpoint,
        r6_binary=args.r6_binary,
        treatment_arm=args.treatment_arm,
        replay_host=socket.gethostname().split(".")[0],
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
