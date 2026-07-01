"""Strict, compact status publication for the R2-MAP cluster dashboard.

The publisher consumes only objects or explicitly named compact JSON files. It
never discovers datasets, checkpoints, receipts, or benchmark artifacts by
walking campaign directories.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from string import hexdigits
from typing import Any

from cascadia_mlx.r2_map_contracts import (
    ALLOWED_HOSTS,
    CAMPAIGN_ID,
    LEGAL_TRANSITIONS,
    Phase,
    _contains_forbidden_host,
    _fsync_directory,
    validate_state,
)

DASHBOARD_STATUS_SCHEMA_ID = "cascadia.r2-map.dashboard-status.v1"
DASHBOARD_STATUS_SCHEMA_VERSION = 1
MAX_COMPACT_JSON_BYTES = 1 << 20
MAX_SERVING_PROJECTION_BYTES = 64 << 10
MAX_LOSS_SAMPLES = 512
MAX_HOST_DETAIL_BYTES = 512
U64_MAX = (1 << 64) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

HOST_INTENTS = {
    "control",
    "generate",
    "validate",
    "train",
    "benchmark",
    "candidate-gate",
    "idle",
}
CLASSIFICATIONS = {"pending", "promote", "reject", "inconclusive", "invalid"}
SERVING_PROJECTION_SCHEMA_ID = "cascadia.r2-map.dashboard-serving-projection.v1"

MODEL_KEYS = {"id", "blake3"}
HOST_KEYS = {
    "intent",
    "detail",
    "generation_games_completed",
    "generation_games_target",
    "generation_seed_prefix",
    "benchmark_pairs_completed",
    "benchmark_pairs_total",
    "eta_seconds",
    "throughput_games_per_second",
    "rss_bytes",
    "swap_delta_bytes",
}
TRAINING_KEYS = {
    "active",
    "latest_verified_checkpoint",
    "current_step",
    "total_steps",
    "examples_per_second",
    "loss_samples",
}
BENCHMARK_KEYS = {
    "active",
    "stage",
    "pairs_completed",
    "pairs_total",
    "eta_seconds",
    "throughput_games_per_second",
    "peak_rss_bytes",
    "swap_delta_bytes",
    "focal",
    "paired_delta",
    "classification",
}
STATUS_KEYS = {
    "schema_version",
    "schema_id",
    "campaign_id",
    "updated_unix_ms",
    "stale_after_seconds",
    "phase",
    "legal_next_transitions",
    "round_index",
    "models",
    "hosts",
    "training",
    "benchmark",
}


class DashboardStatusError(RuntimeError):
    """The compact dashboard status or one of its explicit inputs is invalid."""


@dataclass(frozen=True)
class DashboardStatusInputs:
    """All explicit inputs needed to build one self-contained dashboard mirror."""

    campaign_state: Mapping[str, Any]
    host_receipts: Mapping[str, Mapping[str, Any]] | None = None
    host_safety: Mapping[str, Any] | None = None
    training_progress: Mapping[str, Any] | None = None
    benchmark_aggregate: Mapping[str, Any] | None = None
    model_manifest: Mapping[str, Any] | None = None
    pool_manifest: Mapping[str, Any] | None = None
    stale_after_seconds: int = 30


def read_compact_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read one explicitly named compact object without inspecting siblings."""
    try:
        metadata = path.stat()
        if not path.is_file():
            raise DashboardStatusError(f"{label} is not a regular file: {path}")
        if metadata.st_size > MAX_COMPACT_JSON_BYTES:
            raise DashboardStatusError(
                f"{label} is {metadata.st_size} bytes; maximum is {MAX_COMPACT_JSON_BYTES}"
            )
        value = json.loads(path.read_bytes())
    except DashboardStatusError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardStatusError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise DashboardStatusError(f"{label} must be a JSON object")
    return value


def build_dashboard_status(
    inputs: DashboardStatusInputs,
    *,
    updated_unix_ms: int,
) -> dict[str, Any]:
    """Build a status snapshot entirely from the supplied compact values."""
    state = validate_state(inputs.campaign_state)
    phase = Phase(state["phase"])
    models = _build_models(state, inputs.model_manifest, inputs.pool_manifest)
    hosts = _build_hosts(state, inputs.host_receipts, inputs.host_safety)
    training = (
        _default_training(phase)
        if inputs.training_progress is None
        else dict(inputs.training_progress)
    )
    benchmark = (
        _default_benchmark(phase)
        if inputs.benchmark_aggregate is None
        else dict(inputs.benchmark_aggregate)
    )
    status = {
        "schema_version": DASHBOARD_STATUS_SCHEMA_VERSION,
        "schema_id": DASHBOARD_STATUS_SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "updated_unix_ms": updated_unix_ms,
        "stale_after_seconds": inputs.stale_after_seconds,
        "phase": phase.value,
        "legal_next_transitions": sorted(
            transition.value for transition in LEGAL_TRANSITIONS[phase]
        ),
        "round_index": state["round_index"],
        "models": models,
        "hosts": hosts,
        "training": training,
        "benchmark": benchmark,
    }
    return validate_dashboard_status(status)


def write_dashboard_status(path: Path, status: Mapping[str, Any]) -> int:
    """Validate, fsync, and atomically replace the one dashboard mirror."""
    encoded = _encode_dashboard_status(status)
    if len(encoded) > MAX_COMPACT_JSON_BYTES:
        raise DashboardStatusError(
            f"dashboard status is {len(encoded)} bytes; maximum is {MAX_COMPACT_JSON_BYTES}"
        )
    _atomic_write_bytes(path, encoded, mode=0o600)
    return len(encoded)


def write_serving_projection(
    path: Path,
    *,
    canonical_path: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a bounded projection that binds the canonical John2 status object."""
    import blake3

    canonical_payload = _encode_dashboard_status(status)
    canonical_blake3 = blake3.blake3(canonical_payload).hexdigest()
    projection = {
        "schema_version": 1,
        "schema_id": SERVING_PROJECTION_SCHEMA_ID,
        "canonical_path": str(canonical_path),
        "canonical_blake3": canonical_blake3,
        "canonical_updated_unix_ms": status["updated_unix_ms"],
        "canonical_payload": canonical_payload.decode(),
    }
    encoded = json.dumps(projection, sort_keys=True, indent=2).encode() + b"\n"
    if len(encoded) > MAX_SERVING_PROJECTION_BYTES:
        raise DashboardStatusError(
            f"serving projection is {len(encoded)} bytes; maximum is {MAX_SERVING_PROJECTION_BYTES}"
        )
    _atomic_write_bytes(path, encoded, mode=0o444)
    return {
        "path": str(path),
        "bytes": len(encoded),
        "canonical_path": str(canonical_path),
        "canonical_blake3": canonical_blake3,
        "canonical_updated_unix_ms": status["updated_unix_ms"],
    }


def _encode_dashboard_status(status: Mapping[str, Any]) -> bytes:
    normalized = validate_dashboard_status(status)
    return json.dumps(normalized, sort_keys=True, indent=2).encode() + b"\n"


def _atomic_write_bytes(path: Path, encoded: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fchmod(handle.fileno(), mode)
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            _fsync_directory(path.parent)
        finally:
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_dashboard_status(status: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the Rust reader's strict version-1 validation contract."""
    value = _exact_object(status, STATUS_KEYS, "dashboard status")
    if (
        value["schema_version"] != DASHBOARD_STATUS_SCHEMA_VERSION
        or value["schema_id"] != DASHBOARD_STATUS_SCHEMA_ID
    ):
        raise DashboardStatusError("unsupported dashboard status schema")
    if value["campaign_id"] != CAMPAIGN_ID:
        raise DashboardStatusError("dashboard status names the wrong campaign")
    if _contains_forbidden_host(value):
        raise DashboardStatusError("dashboard status may not name john4")
    _u64(value["updated_unix_ms"], "updated_unix_ms")
    stale_after = _u64(value["stale_after_seconds"], "stale_after_seconds")
    if not 5 <= stale_after <= 3_600:
        raise DashboardStatusError("stale_after_seconds must be between 5 and 3600")
    _nonempty(value["phase"], "phase")
    transitions = _string_array(value["legal_next_transitions"], "legal_next_transitions")
    if len(transitions) != len(set(transitions)):
        raise DashboardStatusError("legal_next_transitions must be unique")
    if value["round_index"] is not None:
        round_index = _u64(value["round_index"], "round_index")
        if round_index > (1 << 32) - 1:
            raise DashboardStatusError("round_index exceeds u32")
    _validate_models(value["models"])
    hosts = _exact_object(value["hosts"], set(ALLOWED_HOSTS), "hosts")
    for host in ALLOWED_HOSTS:
        _validate_host(host, hosts[host])
    _validate_training(value["training"])
    _validate_benchmark(value["benchmark"])
    return json.loads(json.dumps(value, allow_nan=False))


def _build_models(
    state: Mapping[str, Any],
    model_manifest: Mapping[str, Any] | None,
    pool_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if model_manifest is None:
        models = {
            "incumbent": _state_model(state["incumbent_checkpoint_id"]),
            "candidate": _state_model(state["candidate_checkpoint_id"]),
        }
    else:
        models = _exact_object(model_manifest, {"incumbent", "candidate"}, "model manifest")
    pool = (
        {"opponent_pool": []}
        if pool_manifest is None
        else _exact_object(pool_manifest, {"opponent_pool"}, "pool manifest")
    )
    for field, state_field in (
        ("incumbent", "incumbent_checkpoint_id"),
        ("candidate", "candidate_checkpoint_id"),
    ):
        reference = models[field]
        expected_id = state[state_field]
        if (reference is None) != (expected_id is None):
            raise DashboardStatusError(f"{field} model presence disagrees with campaign state")
        if reference is not None:
            reference = _exact_object(reference, MODEL_KEYS, f"{field} model")
            if reference["id"] != expected_id:
                raise DashboardStatusError(f"{field} model id disagrees with campaign state")
            models[field] = reference
    return {
        "incumbent": models["incumbent"],
        "candidate": models["candidate"],
        "opponent_pool": pool["opponent_pool"],
    }


def _state_model(identifier: str | None) -> dict[str, Any] | None:
    return None if identifier is None else {"id": identifier, "blake3": None}


def _build_hosts(
    state: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]] | None,
    host_safety: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supplied = {} if receipts is None else dict(receipts)
    unexpected = set(supplied) - set(ALLOWED_HOSTS)
    if unexpected:
        raise DashboardStatusError(f"host receipts name unexpected hosts: {sorted(unexpected)}")
    result: dict[str, Any] = {}
    for host in ALLOWED_HOSTS:
        raw_receipt = supplied.get(host, _default_host(state["host_intents"][host]))
        if not isinstance(raw_receipt, Mapping):
            raise DashboardStatusError(f"{host} receipt must be an object")
        receipt = dict(raw_receipt)
        if receipt.get("intent") != state["host_intents"][host]:
            raise DashboardStatusError(f"{host} receipt intent disagrees with campaign state")
        result[host] = receipt
    if host_safety is not None:
        from cascadia_mlx.r2_map_apfs_lifecycle import host_dashboard_receipt

        result["john1"] = host_dashboard_receipt(state["host_intents"]["john1"], host_safety)
    return result


def _default_host(intent: str) -> dict[str, Any]:
    return {
        "intent": intent,
        "detail": None,
        "generation_games_completed": 0,
        "generation_games_target": None,
        "generation_seed_prefix": None,
        "benchmark_pairs_completed": 0,
        "benchmark_pairs_total": None,
        "eta_seconds": None,
        "throughput_games_per_second": None,
        "rss_bytes": None,
        "swap_delta_bytes": None,
    }


def _default_training(phase: Phase) -> dict[str, Any]:
    return {
        "active": phase in {Phase.BOOTSTRAP_TRAINING, Phase.TRAINING_AND_BENCHMARKING},
        "latest_verified_checkpoint": None,
        "current_step": None,
        "total_steps": None,
        "examples_per_second": None,
        "loss_samples": [],
    }


def _default_benchmark(phase: Phase) -> dict[str, Any]:
    return {
        "active": phase in {Phase.BOOTSTRAP_CANDIDATE_GATE, Phase.TRAINING_AND_BENCHMARKING},
        "stage": None,
        "pairs_completed": 0,
        "pairs_total": None,
        "eta_seconds": None,
        "throughput_games_per_second": None,
        "peak_rss_bytes": None,
        "swap_delta_bytes": None,
        "focal": None,
        "paired_delta": None,
        "classification": "pending",
    }


def _validate_models(raw: Any) -> None:
    models = _exact_object(raw, {"incumbent", "candidate", "opponent_pool"}, "models")
    pool = models["opponent_pool"]
    if not isinstance(pool, list):
        raise DashboardStatusError("opponent_pool must be an array")
    references = [models["incumbent"], models["candidate"], *pool]
    ids: set[str] = set()
    for reference in references:
        if reference is None:
            continue
        model = _exact_object(reference, MODEL_KEYS, "model reference")
        identifier = _nonempty(model["id"], "model id")
        if identifier in ids:
            raise DashboardStatusError(f"model identity {identifier} is duplicated")
        ids.add(identifier)
        digest = model["blake3"]
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in hexdigits for character in digest)
        ):
            raise DashboardStatusError(f"model {identifier} has an invalid BLAKE3 digest")


def _validate_host(host: str, raw: Any) -> None:
    value = _exact_object(raw, HOST_KEYS, f"{host} status")
    if value["intent"] not in HOST_INTENTS:
        raise DashboardStatusError(f"{host} has an invalid intent")
    detail = _optional_string(value["detail"], f"{host} detail")
    if detail is not None and len(detail.encode("utf-8")) > MAX_HOST_DETAIL_BYTES:
        raise DashboardStatusError(
            f"{host} detail exceeds the {MAX_HOST_DETAIL_BYTES}-byte UTF-8 limit"
        )
    generation = _u64(value["generation_games_completed"], f"{host} generation progress")
    generation_target = _optional_u64(value["generation_games_target"], f"{host} generation target")
    if generation_target is not None and generation > generation_target:
        raise DashboardStatusError(f"{host} generation progress exceeds target")
    _optional_string(value["generation_seed_prefix"], f"{host} generation seed prefix")
    benchmark = _u64(value["benchmark_pairs_completed"], f"{host} benchmark progress")
    benchmark_total = _optional_u64(value["benchmark_pairs_total"], f"{host} benchmark total")
    if benchmark_total is not None and benchmark > benchmark_total:
        raise DashboardStatusError(f"{host} benchmark progress exceeds total")
    _optional_nonnegative_number(value["eta_seconds"], f"{host} ETA")
    _optional_nonnegative_number(value["throughput_games_per_second"], f"{host} throughput")
    _optional_u64(value["rss_bytes"], f"{host} RSS")
    _optional_i64(value["swap_delta_bytes"], f"{host} swap delta")


def _validate_training(raw: Any) -> None:
    value = _exact_object(raw, TRAINING_KEYS, "training")
    _boolean(value["active"], "training active")
    if value["latest_verified_checkpoint"] is not None:
        _validate_model_reference(value["latest_verified_checkpoint"], "verified checkpoint")
    current = _optional_u64(value["current_step"], "training current step")
    total = _optional_u64(value["total_steps"], "training total steps")
    if current is not None and total is not None and current > total:
        raise DashboardStatusError("training current step exceeds total steps")
    _optional_nonnegative_number(value["examples_per_second"], "training throughput")
    samples = value["loss_samples"]
    if not isinstance(samples, list) or len(samples) > MAX_LOSS_SAMPLES:
        raise DashboardStatusError("loss_samples must be an array with at most 512 entries")
    previous_step: int | None = None
    for raw_sample in samples:
        sample = _exact_object(
            raw_sample, {"step", "train_total", "validation_total"}, "loss sample"
        )
        step = _u64(sample["step"], "loss sample step")
        if previous_step is not None and step <= previous_step:
            raise DashboardStatusError("loss sample steps must be strictly increasing")
        _nonnegative_number(sample["train_total"], "training loss")
        _optional_nonnegative_number(sample["validation_total"], "validation loss")
        previous_step = step


def _validate_benchmark(raw: Any) -> None:
    value = _exact_object(raw, BENCHMARK_KEYS, "benchmark")
    _boolean(value["active"], "benchmark active")
    _optional_string(value["stage"], "benchmark stage")
    completed = _u64(value["pairs_completed"], "benchmark pairs completed")
    total = _optional_u64(value["pairs_total"], "benchmark pairs total")
    if total is not None and completed > total:
        raise DashboardStatusError("benchmark progress exceeds total")
    _optional_nonnegative_number(value["eta_seconds"], "benchmark ETA")
    _optional_nonnegative_number(value["throughput_games_per_second"], "benchmark throughput")
    _optional_u64(value["peak_rss_bytes"], "benchmark peak RSS")
    _optional_i64(value["swap_delta_bytes"], "benchmark swap delta")
    if value["focal"] is not None:
        _validate_focal(value["focal"])
    if value["paired_delta"] is not None:
        delta = _exact_object(value["paired_delta"], {"mean", "confidence_95"}, "paired delta")
        _finite_number(delta["mean"], "paired delta mean")
        interval = delta["confidence_95"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise DashboardStatusError("paired delta confidence_95 must contain two values")
        lower = _finite_number(interval[0], "paired delta lower bound")
        upper = _finite_number(interval[1], "paired delta upper bound")
        if lower > upper:
            raise DashboardStatusError("paired delta confidence interval is reversed")
    if value["classification"] not in CLASSIFICATIONS:
        raise DashboardStatusError("benchmark classification is invalid")


def _validate_focal(raw: Any) -> None:
    focal = _exact_object(raw, {"base_total", "animals", "habitat", "pinecones"}, "focal")
    _validate_distribution(focal["base_total"], "base total")
    groups = (
        ("animals", {"aggregate", "bear", "elk", "salmon", "hawk", "fox"}),
        ("habitat", {"aggregate", "mountain", "forest", "prairie", "wetland", "river"}),
        (
            "pinecones",
            {
                "earned",
                "independent_draft_spend",
                "paid_wipe_spend",
                "total_spend",
                "remaining",
                "free_replacements",
            },
        ),
    )
    for group_name, keys in groups:
        group = _exact_object(focal[group_name], keys, group_name)
        for label, distribution in group.items():
            _validate_distribution(distribution, f"{group_name} {label}")


def _validate_distribution(raw: Any, label: str) -> None:
    value = _exact_object(raw, {"mean", "p10", "p50", "p90"}, label)
    _finite_number(value["mean"], f"{label} mean")
    p10 = _finite_number(value["p10"], f"{label} p10")
    p50 = _finite_number(value["p50"], f"{label} p50")
    p90 = _finite_number(value["p90"], f"{label} p90")
    if p10 > p50 or p50 > p90:
        raise DashboardStatusError(f"{label} quantiles are not ordered")


def _validate_model_reference(raw: Any, label: str) -> None:
    model = _exact_object(raw, MODEL_KEYS, label)
    _nonempty(model["id"], f"{label} id")
    digest = model["blake3"]
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in hexdigits for character in digest)
    ):
        raise DashboardStatusError(f"{label} has an invalid BLAKE3 digest")


def _exact_object(raw: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DashboardStatusError(f"{label} must be an object")
    actual = set(raw)
    if actual != keys:
        raise DashboardStatusError(
            f"{label} keys differ: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return dict(raw)


def _string_array(raw: Any, label: str) -> list[str]:
    if not isinstance(raw, list):
        raise DashboardStatusError(f"{label} must be an array")
    return [_nonempty(item, label) for item in raw]


def _nonempty(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise DashboardStatusError(f"{label} must be a nonempty string")
    return raw


def _optional_string(raw: Any, label: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise DashboardStatusError(f"{label} must be a string or null")
    return raw


def _boolean(raw: Any, label: str) -> bool:
    if not isinstance(raw, bool):
        raise DashboardStatusError(f"{label} must be a boolean")
    return raw


def _u64(raw: Any, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or not 0 <= raw <= U64_MAX:
        raise DashboardStatusError(f"{label} must be a u64 integer")
    return raw


def _optional_u64(raw: Any, label: str) -> int | None:
    return None if raw is None else _u64(raw, label)


def _optional_i64(raw: Any, label: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or not I64_MIN <= raw <= I64_MAX:
        raise DashboardStatusError(f"{label} must be an i64 integer or null")
    return raw


def _finite_number(raw: Any, label: str) -> float | int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise DashboardStatusError(f"{label} must be finite")
    return raw


def _nonnegative_number(raw: Any, label: str) -> float | int:
    value = _finite_number(raw, label)
    if value < 0:
        raise DashboardStatusError(f"{label} must be nonnegative")
    return value


def _optional_nonnegative_number(raw: Any, label: str) -> float | int | None:
    return None if raw is None else _nonnegative_number(raw, label)
