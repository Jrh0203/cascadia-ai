"""Isolated serving benchmark for the frozen ADR 0150 comparison."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import load_latest_checkpoint_with_factory
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ADR_ID,
    ARMS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.r3_action_edit_mlx_metrics import (
    CANDIDATE_CHUNK,
    benchmark_r3_action_edit,
)
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

BENCHMARK_SCHEMA_VERSION = 1
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
VERIFICATION_SOURCES = frozenset(("cluster-preflight", "in-process-full"))


class R3ServingBenchmarkError(RuntimeError):
    """An isolated benchmark request or result is inconsistent."""


def run_isolated_serving_benchmark(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    cache: Path,
    s1_cache: Path,
    run_dir: Path,
    checkpoint: Path,
    arm: str,
    global_step: int,
    open_data_verification: dict[str, Any],
    verification_source: str,
    warmup_iterations: int,
    steady_iterations: int,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run one fresh-process benchmark and return its bound performance payload."""
    request = create_serving_benchmark_request(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        cache=cache,
        s1_cache=s1_cache,
        run_dir=run_dir,
        checkpoint=checkpoint,
        arm=arm,
        global_step=global_step,
        open_data_verification=open_data_verification,
        verification_source=verification_source,
        warmup_iterations=warmup_iterations,
        steady_iterations=steady_iterations,
        decision_rows=decision_rows,
    )
    request_path = run_dir / "serving-benchmark-request.json"
    result_path = run_dir / "serving-benchmark-result.json"
    _write_json_atomic(request_path, request)
    result_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "cascadia_mlx.r3_action_edit_mlx_benchmark",
            "--request",
            str(request_path.resolve()),
            "--output",
            str(result_path.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise R3ServingBenchmarkError(f"isolated R3 serving benchmark failed: {detail}")
    result = _read_json(result_path, "isolated R3 serving benchmark result")
    _validate_result(result, request)
    performance = dict(result["scientific_identity"]["performance"])
    performance["measurement"] = {
        "isolated_process": True,
        "request_id": request["request_id"],
        "result_id": result["result_id"],
        "checkpoint_model_blake3": request["scientific_identity"]["checkpoint"]["model_blake3"],
        "open_data_verification_id": request["scientific_identity"]["open_data_verification_id"],
        "verification_source": verification_source,
        "worker_runtime": result["scientific_identity"]["runtime"],
    }
    return performance


def create_serving_benchmark_request(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    cache: Path,
    s1_cache: Path,
    run_dir: Path,
    checkpoint: Path,
    arm: str,
    global_step: int,
    open_data_verification: dict[str, Any],
    verification_source: str,
    warmup_iterations: int,
    steady_iterations: int,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("R3 serving benchmark arm is unknown")
    if verification_source not in VERIFICATION_SOURCES:
        raise ValueError("R3 serving benchmark verification source is invalid")
    if global_step <= 0 or warmup_iterations <= 0 or steady_iterations <= 0:
        raise ValueError("R3 serving benchmark dimensions must be positive")
    proof_id = open_data_verification_id(open_data_verification)
    rows = None if decision_rows is None else [int(value) for value in decision_rows]
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": arm,
        "train_dataset": str(train_dataset.resolve()),
        "validation_dataset": str(validation_dataset.resolve()),
        "cache": str(cache.resolve()),
        "s1_cache": str(s1_cache.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
            "model_blake3": _checksum(checkpoint / "model.safetensors"),
            "global_step": global_step,
        },
        "open_data_verification_id": proof_id,
        "open_data_verification": open_data_verification,
        "verification_source": verification_source,
        "require_complete_open_corpus": verification_source == "cluster-preflight",
        "candidate_chunk": CANDIDATE_CHUNK,
        "warmup_iterations": warmup_iterations,
        "steady_iterations": steady_iterations,
        "decision_rows": rows,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "request_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }


def execute_serving_benchmark_request(request: dict[str, Any]) -> dict[str, Any]:
    identity = _validate_request(request)
    cache = R3ActionEditMlxCache(
        identity["cache"],
        verify_checksums=False,
        verify_semantics=False,
        require_complete=identity["require_complete_open_corpus"],
    )
    s1_cache = S1ExactSupplyCache(
        identity["s1_cache"],
        verify_checksums=False,
        verify_semantics=False,
        require_complete=identity["require_complete_open_corpus"],
    )
    observed_open_data = open_data_verification_identity(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=identity["train_dataset"],
        validation_dataset=identity["validation_dataset"],
    )
    if (
        observed_open_data != identity["open_data_verification"]
        or open_data_verification_id(observed_open_data) != identity["open_data_verification_id"]
    ):
        raise R3ServingBenchmarkError(
            "isolated benchmark open-data identity differs from its proof"
        )
    validation = cache.bind_dataset(
        identity["validation_dataset"],
        s1_cache=s1_cache,
        verify_dataset_checksums=False,
        preverified_open_data_proof_id=identity["open_data_verification_id"],
    )
    mx.set_default_device(mx.gpu)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    model, _optimizer, state, checkpoint = load_latest_checkpoint_with_factory(
        identity["run_dir"],
        learning_rate=1e-4,
        weight_decay=1e-4,
        model_factory=lambda values: R3ActionEditRanker(R3ActionEditModelConfig.from_dict(values)),
    )
    expected_checkpoint = identity["checkpoint"]
    if (
        checkpoint.resolve() != Path(expected_checkpoint["path"])
        or _checksum(checkpoint / "checkpoint.json") != expected_checkpoint["manifest_blake3"]
        or _checksum(checkpoint / "model.safetensors") != expected_checkpoint["model_blake3"]
        or state.global_step != expected_checkpoint["global_step"]
        or model.config.arm != identity["arm"]
    ):
        raise R3ServingBenchmarkError("isolated benchmark checkpoint differs from its request")
    model.eval()
    decision_rows = identity["decision_rows"]
    performance = benchmark_r3_action_edit(
        model,
        validation,
        arm=identity["arm"],
        candidate_chunk=identity["candidate_chunk"],
        warmup_iterations=identity["warmup_iterations"],
        steady_iterations=identity["steady_iterations"],
        decision_rows=(
            None if decision_rows is None else np.asarray(decision_rows, dtype=np.int64)
        ),
    )
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "default_device": str(mx.default_device()),
        "host": socket.gethostname().split(".")[0],
        "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
        "previous_mlx_cache_limit_bytes": previous_cache_limit,
    }
    result_identity = {
        "request_id": request["request_id"],
        "arm": identity["arm"],
        "checkpoint": expected_checkpoint,
        "open_data_verification_id": identity["open_data_verification_id"],
        "runtime": runtime,
        "performance": performance,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "result_id": _canonical_blake3(result_identity),
        "scientific_identity": result_identity,
    }


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    identity = request.get("scientific_identity")
    if (
        request.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or not isinstance(identity, dict)
        or _canonical_blake3(identity) != request.get("request_id")
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("protocol_id") != PROTOCOL_ID
        or identity.get("adr") != ADR_ID
        or identity.get("arm") not in ARMS
        or identity.get("verification_source") not in VERIFICATION_SOURCES
        or identity.get("require_complete_open_corpus")
        != (identity.get("verification_source") == "cluster-preflight")
        or identity.get("candidate_chunk") != CANDIDATE_CHUNK
        or not isinstance(identity.get("open_data_verification"), dict)
        or open_data_verification_id(identity["open_data_verification"])
        != identity.get("open_data_verification_id")
    ):
        raise R3ServingBenchmarkError("isolated R3 serving benchmark request is malformed")
    return identity


def _validate_result(result: dict[str, Any], request: dict[str, Any]) -> None:
    identity = result.get("scientific_identity")
    if (
        result.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("adr") != ADR_ID
        or not isinstance(identity, dict)
        or identity.get("request_id") != request.get("request_id")
        or _canonical_blake3(identity) != result.get("result_id")
        or not isinstance(identity.get("performance"), dict)
    ):
        raise R3ServingBenchmarkError("isolated R3 serving benchmark result is malformed")


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
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R3ServingBenchmarkError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise R3ServingBenchmarkError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated ADR 0150 benchmark")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    request = _read_json(args.request, "isolated R3 serving benchmark request")
    result = execute_serving_benchmark_request(request)
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
