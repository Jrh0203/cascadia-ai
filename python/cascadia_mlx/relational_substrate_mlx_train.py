"""Frozen training and bounded smoke runner for ADR 0161."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3ActionEditMlxCache
from cascadia_mlx.relational_substrate_mlx_benchmark import (
    run_isolated_serving_benchmark,
)
from cascadia_mlx.relational_substrate_mlx_cache import (
    ADR_ID,
    ARMS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    RelationalSubstrateBatch,
    RelationalSubstrateMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.relational_substrate_mlx_metrics import (
    CANDIDATE_CHUNK,
    evaluate_relational_substrate,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateModelConfig,
    RelationalSubstrateRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    relational_substrate_loss,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

TRAINING_SEED = 2026061716
TRAINING_STEPS = 3000
GROUPS_PER_STEP = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 250
METRIC_STEPS = 100
VALIDATION_PROBE_GROUPS = 24
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
MAX_SMOKE_STEPS = 10

ARM_HOSTS = {
    ARMS[0]: "john1",
    ARMS[1]: "john2",
    ARMS[2]: "john3",
    ARMS[3]: "john4",
}
HOST_ALIASES = {"Johns-Mac-mini": "john1"}


@dataclass(frozen=True)
class RelationalSubstrateTrainingProtocol:
    """All scientific constants held equal across the four tournament arms."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    optimizer: str = "adamw"
    training_steps: int = TRAINING_STEPS
    groups_per_step: int = GROUPS_PER_STEP
    train_candidate_cap: int = 512
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    validation_probe_groups: int = VALIDATION_PROBE_GROUPS
    candidate_chunk: int = CANDIDATE_CHUNK
    warm_start: bool = False
    early_stopping: bool = False
    schedule: str = (
        "three-independent-all-group-permutations-plus-alternating-"
        "low-supply-and-independent-winner-permutations"
    )
    d6_schedule: str = (
        "blake3-of-seed-step-slot-over-rust-d6-ids-0-through-11"
    )
    candidate_surface: str = (
        "exact-r2-control-versus-exact-r3-radius-one-global-treatments"
    )
    parent_surface: str = (
        "native-exact-r2-control-versus-r5-minimal-or-s3-rich-relational"
    )
    derivative_surface: str = "154-s5-fields-enabled-only-for-d3"
    loss: str = (
        "r1200_huber+4*r4800_huber+0.5*r1200_listwise+"
        "r4800_winner+0.1*standard_error_calibration+"
        "0.01*screen_only_regularization"
    )

    def validate(self) -> None:
        if self != RelationalSubstrateTrainingProtocol():
            raise ValueError(
                "relational training protocol drifted from ADR 0161"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class RelationalSubstrateTrainingConfig:
    """One production or bounded-smoke arm invocation."""

    train_dataset: Path
    validation_dataset: Path
    r3_cache: Path
    relational_cache: Path
    s1_cache: Path
    r6_binary: Path
    run_dir: Path
    output: Path
    arm: str
    resume: bool = False
    smoke_steps: int | None = None
    authorization: Path | None = None
    preflight: Path | None = None
    protocol: RelationalSubstrateTrainingProtocol = field(
        default_factory=RelationalSubstrateTrainingProtocol
    )

    @property
    def production(self) -> bool:
        return self.smoke_steps is None

    @property
    def target_steps(self) -> int:
        return TRAINING_STEPS if self.production else int(self.smoke_steps)

    def validate(self) -> None:
        self.protocol.validate()
        if self.arm not in ARMS:
            raise ValueError("relational training arm is unknown")
        if not self.r6_binary.is_file():
            raise ValueError("relational R6 replay binary is absent")
        if self.production:
            if self.authorization is None or self.preflight is None:
                raise ValueError(
                    "production relational training requires authorization "
                    "and host preflight"
                )
        elif (
            self.smoke_steps is None
            or self.smoke_steps <= 0
            or self.smoke_steps > MAX_SMOKE_STEPS
            or self.resume
            or self.authorization is not None
            or self.preflight is not None
        ):
            raise ValueError(
                "bounded relational smoke must be fresh, uncontrolled, "
                "and at most 10 steps"
            )


def run_relational_substrate_training(
    config: RelationalSubstrateTrainingConfig,
) -> dict[str, Any]:
    """Train one arm under the frozen schedule and emit complete evidence."""
    config.validate()
    require_complete = config.production
    source = source_provenance(Path(__file__).resolve().parents[2])
    mx.set_default_device(mx.gpu)
    runtime = _runtime_identity()
    if config.production:
        _require_production_runtime(runtime)
        actual_host = _normalize_host(socket.gethostname().split(".")[0])
        if actual_host != ARM_HOSTS[config.arm]:
            raise ValueError(
                f"relational arm {config.arm} must run on "
                f"{ARM_HOSTS[config.arm]}, not {actual_host}"
            )

    r3 = R3ActionEditMlxCache(
        config.r3_cache,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
        require_complete=require_complete,
    )
    relational_cache = RelationalSubstrateMlxCache(
        config.relational_cache,
        r3_cache=r3,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
        require_complete=require_complete,
    )
    s1_cache = S1ExactSupplyCache(
        config.s1_cache,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
        require_complete=require_complete,
    )
    open_data = open_data_verification_identity(
        cache=relational_cache,
        s1_cache=s1_cache,
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
    )
    controls = (
        validate_launch_controls(
            config.authorization,
            config.preflight,
            arm=config.arm,
            r3_cache_id=r3.cache_id,
            relational_cache_id=relational_cache.cache_id,
            s1_cache_id=s1_cache.cache_id,
            r6_binary_blake3=_checksum(config.r6_binary),
            source=source,
            runtime=runtime,
            open_data_verification=open_data,
        )
        if config.production
        else None
    )
    proof_id = (
        str(controls["open_data_verification_id"])
        if controls is not None
        else None
    )
    train = relational_cache.bind_dataset(
        config.train_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=not config.production,
        preverified_open_data_proof_id=proof_id,
    )
    validation = relational_cache.bind_dataset(
        config.validation_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=not config.production,
        preverified_open_data_proof_id=proof_id,
    )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    cross_arm = _cross_arm_initialization()
    model_config = RelationalSubstrateModelConfig(arm=config.arm)
    run_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "arm": config.arm,
        "target_steps": config.target_steps,
        "protocol": config.protocol.to_dict(),
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational_cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "r6_binary": {
            "path": str(config.r6_binary.resolve()),
            "blake3": _checksum(config.r6_binary),
        },
        "train_dataset_id": train.base.manifest["dataset_id"],
        "validation_dataset_id": validation.base.manifest["dataset_id"],
        "source": source,
        "runtime": runtime,
        "cross_arm_initialization": cross_arm,
        "controls": controls,
    }
    if config.resume:
        existing = _read_json(
            config.run_dir / "run.json",
            "relational run manifest",
        )
        if existing != run_manifest:
            raise ValueError(
                "relational resume manifest differs from the frozen run"
            )
        model, optimizer, state, _ = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            model_factory=lambda values: RelationalSubstrateRanker(
                RelationalSubstrateModelConfig.from_dict(values)
            ),
        )
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError(
                "relational run already has checkpoints; pass --resume"
            )
        mx.random.seed(TRAINING_SEED)
        model = RelationalSubstrateRanker(model_config)
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)
    if (
        model.config != model_config
        or parameter_count(model) != cross_arm["parameter_count"]
        or parameter_layout_blake3(model)
        != cross_arm["parameter_layout_blake3"]
        or (
            not config.resume
            and parameter_tensor_blake3(model)
            != cross_arm["initial_parameter_tensor_blake3"]
        )
    ):
        raise ValueError(
            "relational model graph differs from matched initialization"
        )

    loss_and_grad = nn.value_and_grad(model, relational_substrate_loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    batch_trace_path = config.run_dir / "batch-trace.jsonl"
    loss_trace: list[dict[str, Any]] = []
    measured_seconds = 0.0
    measured_candidates = 0
    model.train()
    invocation_started = time.perf_counter()
    while state.global_step < config.target_steps:
        step = state.global_step
        started = time.perf_counter()
        batch = train.deterministic_training_batch(
            step=step,
            seed=TRAINING_SEED,
            arm=config.arm,
        )
        batch_identity = scientific_batch_blake3(batch)
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise ValueError(
                f"relational training produced nonfinite loss at step {step}"
            )
        candidates = int(
            np.asarray(batch.base.candidate_mask, dtype=np.bool_).sum()
        )
        measured_seconds += elapsed
        measured_candidates += candidates
        state.global_step += 1
        state.batch_in_epoch = state.global_step
        state.elapsed_seconds += elapsed
        event = {
            "schema_version": 1,
            "step": state.global_step,
            "batch_blake3": batch_identity,
            "loss": loss_value,
            "candidates": candidates,
            "elapsed_seconds": elapsed,
        }
        loss_trace.append(event)
        _append_json(batch_trace_path, event)
        if state.global_step % METRIC_STEPS == 0:
            probe = evaluate_relational_substrate(
                model,
                validation,
                arm=config.arm,
                rows=_validation_probe_rows(validation.group_count),
                prediction_panel_size=16,
            )
            metric_event = {
                "schema_version": 1,
                "step": state.global_step,
                "mean_recent_loss": float(
                    np.mean(
                        [
                            value["loss"]
                            for value in loss_trace[-METRIC_STEPS:]
                        ]
                    )
                ),
                "candidates_per_second": (
                    measured_candidates / max(measured_seconds, 1e-12)
                ),
                "validation_probe": probe,
                "peak_active_memory_bytes": int(mx.get_peak_memory()),
            }
            _append_json(metrics_path, metric_event)
            print(json.dumps(metric_event, sort_keys=True), flush=True)
            model.train()
        if state.global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(config.run_dir, model, optimizer, state)
            prune_checkpoints(config.run_dir)

    checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
    prune_checkpoints(config.run_dir)
    training_peak_memory = int(mx.get_peak_memory())
    training_wall_seconds = time.perf_counter() - invocation_started
    model.eval()
    if config.production:
        validation_metrics = evaluate_relational_substrate(
            model,
            validation,
            arm=config.arm,
        )
        benchmark_rows = np.arange(
            validation.group_count,
            dtype=np.int64,
        )
        warmup_iterations = 5
        steady_iterations = 30
        verification_source = "cluster-preflight"
    else:
        benchmark_rows = np.arange(
            min(validation.group_count, 5),
            dtype=np.int64,
        )
        validation_metrics = evaluate_relational_substrate(
            model,
            validation,
            arm=config.arm,
            rows=benchmark_rows,
            prediction_panel_size=16,
        )
        warmup_iterations = 1
        steady_iterations = 3
        verification_source = "in-process-full"
    performance = run_isolated_serving_benchmark(
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
        r3_cache=config.r3_cache,
        relational_cache=config.relational_cache,
        s1_cache=config.s1_cache,
        r6_binary=config.r6_binary,
        run_dir=config.run_dir,
        checkpoint=checkpoint,
        arm=config.arm,
        global_step=state.global_step,
        open_data_verification=open_data,
        verification_source=verification_source,
        warmup_iterations=warmup_iterations,
        steady_iterations=steady_iterations,
        decision_rows=benchmark_rows,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "arm": config.arm,
        "host": _normalize_host(socket.gethostname().split(".")[0]),
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational_cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "r6_binary": {
            "path": str(config.r6_binary.resolve()),
            "blake3": _checksum(config.r6_binary),
        },
        "protocol": config.protocol.to_dict(),
        "model": {
            "config": model.config.to_dict(),
            **cross_arm,
            "final_parameter_tensor_blake3": parameter_tensor_blake3(model),
        },
        "optimization": {
            "global_step": state.global_step,
            "candidates": measured_candidates,
            "training_seconds": measured_seconds,
            "training_wall_seconds": training_wall_seconds,
            "candidates_per_second": (
                measured_candidates / max(measured_seconds, 1e-12)
            ),
            "training_peak_active_memory_bytes": training_peak_memory,
            "loss_trace": loss_trace,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "manifest_blake3": _checksum(
                checkpoint / "checkpoint.json"
            ),
            "model_blake3": _checksum(checkpoint / "model.safetensors"),
        },
        "metrics": validation_metrics,
        "performance": performance,
        "runtime": {
            **runtime,
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "source": source,
        "controls": controls,
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_read": False,
            "future_refill_read": False,
        },
        "claims": {
            "offline_comparison_complete": config.production,
            "bounded_smoke_complete": not config.production,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = {
        key: report[key]
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "arm",
            "host",
            "r3_cache_id",
            "relational_cache_id",
            "s1_cache_id",
            "r6_binary",
            "protocol",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
            "performance",
            "runtime",
            "source",
            "controls",
            "information_boundary",
            "claims",
        )
    }
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def scientific_batch_blake3(batch: RelationalSubstrateBatch) -> str:
    """Hash only arm-invariant group, target, action, and D6 identities."""
    digest = blake3.blake3()
    digest.update(b"relational-substrate-mlx-scientific-batch-v1")
    group_ids = np.asarray(batch.base.group_id, dtype=np.uint64)
    transforms = np.asarray(batch.parent.transform_ids, dtype=np.uint8)
    mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)
    source_indices = np.asarray(
        batch.source_candidate_indices,
        dtype=np.uint16,
    )
    action_hashes = np.asarray(batch.base.action_hash, dtype=np.uint8)
    for row, group_id in enumerate(group_ids):
        count = int(mask[row].sum())
        digest.update(int(group_id).to_bytes(8, "little"))
        digest.update(bytes([int(transforms[row])]))
        digest.update(count.to_bytes(8, "little"))
        digest.update(source_indices[row, :count].astype("<u2").tobytes())
        digest.update(action_hashes[row, :count].tobytes(order="C"))
    return digest.hexdigest()


def validate_launch_controls(
    authorization_path: Path | None,
    preflight_path: Path | None,
    *,
    arm: str,
    r3_cache_id: str,
    relational_cache_id: str,
    s1_cache_id: str,
    r6_binary_blake3: str,
    source: dict[str, Any],
    runtime: dict[str, Any],
    open_data_verification: dict[str, Any],
) -> dict[str, Any]:
    if authorization_path is None or preflight_path is None:
        raise ValueError("relational production launch controls are absent")
    authorization = _read_json(
        authorization_path,
        "relational authorization",
    )
    preflight = _read_json(preflight_path, "relational preflight")
    authorization_identity = authorization.get("identity")
    preflight_identity = preflight.get("identity")
    verification_id = open_data_verification_id(
        open_data_verification
    )
    preflight_checks = preflight.get("checks")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization_identity, dict)
        or _canonical_blake3(authorization_identity)
        != authorization.get("authorization_id")
        or authorization_identity.get("protocol_id") != PROTOCOL_ID
        or authorization_identity.get("r3_cache_id") != r3_cache_id
        or authorization_identity.get("relational_cache_id")
        != relational_cache_id
        or authorization_identity.get("s1_cache_id") != s1_cache_id
        or authorization_identity.get("r6_binary_blake3")
        != r6_binary_blake3
        or authorization_identity.get("authorized_arms") != list(ARMS)
        or authorization_identity.get("arm_hosts") != ARM_HOSTS
        or authorization_identity.get("source_blake3")
        != source.get("v2_source_blake3")
        or authorization_identity.get("protocol")
        != RelationalSubstrateTrainingProtocol().to_dict()
        or authorization_identity.get("open_data_verification")
        != open_data_verification
        or authorization_identity.get("open_data_verification_id")
        != verification_id
    ):
        raise ValueError(
            "relational production authorization is stale or malformed"
        )
    if (
        preflight.get("schema_version") != 1
        or preflight.get("experiment_id") != EXPERIMENT_ID
        or preflight.get("arm") != arm
        or not isinstance(preflight_identity, dict)
        or _canonical_blake3(preflight_identity)
        != preflight.get("preflight_id")
        or preflight_identity.get("authorization_id")
        != authorization.get("authorization_id")
        or preflight_identity.get("r3_cache_id") != r3_cache_id
        or preflight_identity.get("relational_cache_id")
        != relational_cache_id
        or preflight_identity.get("s1_cache_id") != s1_cache_id
        or preflight_identity.get("r6_binary_blake3")
        != r6_binary_blake3
        or preflight_identity.get("arm") != arm
        or preflight_identity.get("host") != ARM_HOSTS[arm]
        or preflight_identity.get("runtime") != runtime
        or preflight_identity.get("open_data_verification_id")
        != verification_id
        or preflight_identity.get("mlx_gpu_verified") is not True
        or preflight_identity.get("open_data_only_verified") is not True
        or preflight_identity.get("initialization_parity_verified")
        is not True
        or preflight_identity.get("smoke_replay_verified") is not True
        or preflight_identity.get("candidate_identity_verified")
        is not True
        or preflight_identity.get("parent_surface_verified") is not True
        or preflight_identity.get("derivative_surface_verified")
        is not True
        or not isinstance(preflight_checks, dict)
        or any(
            value is not True
            for key, value in preflight_checks.items()
            if key != "production_training_started"
        )
        or preflight_checks.get("production_training_started") is not False
    ):
        raise ValueError(
            "relational arm preflight is stale or incomplete"
        )
    return {
        "authorization_id": authorization["authorization_id"],
        "preflight_id": preflight["preflight_id"],
        "open_data_verification_id": verification_id,
        "full_preflight_verification_reused": True,
    }


def _cross_arm_initialization() -> dict[str, Any]:
    counts: dict[str, int] = {}
    layouts: dict[str, str] = {}
    tensors: dict[str, str] = {}
    for arm in ARMS:
        mx.random.seed(TRAINING_SEED)
        candidate = RelationalSubstrateRanker(
            RelationalSubstrateModelConfig(arm=arm)
        )
        counts[arm] = parameter_count(candidate)
        layouts[arm] = parameter_layout_blake3(candidate)
        tensors[arm] = parameter_tensor_blake3(candidate)
    if (
        len(set(counts.values())) != 1
        or len(set(layouts.values())) != 1
        or len(set(tensors.values())) != 1
    ):
        raise ValueError(
            "relational arm initialization or parameter graph differs"
        )
    return {
        "parameter_count": next(iter(counts.values())),
        "parameter_layout_blake3": next(iter(layouts.values())),
        "initial_parameter_tensor_blake3": next(iter(tensors.values())),
        "cross_arm_parameter_counts": counts,
        "cross_arm_parameter_layout_blake3": layouts,
        "cross_arm_initial_parameter_tensor_blake3": tensors,
    }


def cross_arm_initialization() -> dict[str, Any]:
    return _cross_arm_initialization()


def _validation_probe_rows(group_count: int) -> np.ndarray:
    count = min(VALIDATION_PROBE_GROUPS, group_count)
    return np.unique(
        np.linspace(0, group_count - 1, count, dtype=np.int64)
    )


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "default_device": str(mx.default_device()),
        "host": _normalize_host(socket.gethostname().split(".")[0]),
    }


def runtime_identity() -> dict[str, Any]:
    return _runtime_identity()


def _require_production_runtime(runtime: dict[str, Any]) -> None:
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    if (
        runtime.get("machine") != "arm64"
        or "gpu" not in str(mx.default_device()).lower()
    ):
        raise ValueError(
            "relational production training requires Apple Silicon MLX GPU"
        )


def _normalize_host(host: str) -> str:
    return HOST_ALIASES.get(host, host)


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
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or smoke one ADR 0161 tournament arm"
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument("--relational-cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--r6-binary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--preflight", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_relational_substrate_training(
        RelationalSubstrateTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            r3_cache=args.r3_cache,
            relational_cache=args.relational_cache,
            s1_cache=args.s1_cache,
            r6_binary=args.r6_binary,
            run_dir=args.run_dir,
            output=args.output,
            arm=args.arm,
            resume=args.resume,
            smoke_steps=args.smoke_steps,
            authorization=args.authorization,
            preflight=args.preflight,
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
