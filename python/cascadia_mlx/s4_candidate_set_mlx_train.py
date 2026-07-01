"""Matched training runner for the S4 relational candidate-context tournament."""

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
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ARMS as R3_ARMS,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.r3_action_edit_mlx_train import scientific_batch_blake3
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache
from cascadia_mlx.s4_candidate_context_cache import S4CandidateContextCache
from cascadia_mlx.s4_candidate_set_mlx_benchmark import (
    run_isolated_serving_benchmark,
)
from cascadia_mlx.s4_candidate_set_mlx_data import S4CandidateSetDataset
from cascadia_mlx.s4_candidate_set_mlx_metrics import (
    CANDIDATE_CHUNK,
    evaluate_s4_candidate_set,
)
from cascadia_mlx.s4_candidate_set_mlx_model import (
    S4_ARMS,
    S4CandidateSetModelConfig,
    S4CandidateSetRanker,
    s4_candidate_set_loss,
)

EXPERIMENT_ID = "s4-candidate-context-mlx-comparison-v1"
PROTOCOL_ID = "s4-candidate-context-matched-comparison-v1"
ADR_ID = "0153"
TRAINING_SEED = 2026061721
TRAINING_STEPS = 3000
GROUPS_PER_STEP = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 250
METRIC_STEPS = 250
VALIDATION_PROBE_GROUPS = 12
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
MAX_SMOKE_STEPS = 10
ARM_HOSTS = {
    S4_ARMS[0]: "john1",
    S4_ARMS[1]: "john2",
    S4_ARMS[2]: "john3",
    S4_ARMS[3]: "john4",
}
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
}


@dataclass(frozen=True)
class S4CandidateSetTrainingProtocol:
    """Every choice held equal in the failed-substrate rescue comparison."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    optimizer: str = "adamw"
    training_steps: int = TRAINING_STEPS
    groups_per_step: int = GROUPS_PER_STEP
    train_candidate_cap: int = 512
    anchor_limit: int = 256
    inducing_latents: int = 16
    relation_neighbor_limit: int = 8
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    validation_probe_groups: int = VALIDATION_PROBE_GROUPS
    candidate_chunk: int = CANDIDATE_CHUNK
    warm_start: bool = True
    warm_start_substrate: str = R3_ARMS[3]
    warm_start_substrate_status: str = "failed-r3-compact-treatment"
    context_delta_zero_initialized: bool = True
    base_jointly_finetuned: bool = True
    early_stopping: bool = False
    schedule: str = (
        "r3-three-independent-all-group-permutations-plus-alternating-"
        "low-supply-and-independent-winner-permutations"
    )
    d6_schedule: str = (
        "r3-blake3-of-seed-step-slot-over-rust-d6-ids-0-through-11"
    )
    loss: str = (
        "r1200_huber+4*r4800_huber+0.5*r1200_listwise+"
        "r4800_winner+0.1*standard_error_calibration+"
        "0.01*screen_only_regularization"
    )

    def validate(self) -> None:
        if self != S4CandidateSetTrainingProtocol():
            raise ValueError("S4 training protocol drifted from ADR 0153")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class S4CandidateSetTrainingConfig:
    """One production or bounded-smoke S4 arm invocation."""

    train_dataset: Path
    validation_dataset: Path
    cache: Path
    s1_cache: Path
    context_cache: Path
    warm_start_checkpoint: Path
    run_dir: Path
    output: Path
    arm: str
    resume: bool = False
    smoke_steps: int | None = None
    authorization: Path | None = None
    preflight: Path | None = None
    protocol: S4CandidateSetTrainingProtocol = field(
        default_factory=S4CandidateSetTrainingProtocol
    )

    @property
    def production(self) -> bool:
        return self.smoke_steps is None

    @property
    def target_steps(self) -> int:
        return TRAINING_STEPS if self.production else int(self.smoke_steps)

    def validate(self) -> None:
        self.protocol.validate()
        if self.arm not in S4_ARMS:
            raise ValueError("S4 training arm is unknown")
        if self.production:
            if self.authorization is None or self.preflight is None:
                raise ValueError(
                    "production S4 training requires authorization and preflight"
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
                "bounded S4 smoke must be fresh, uncontrolled, and at most 10 steps"
            )


def run_s4_candidate_set_training(
    config: S4CandidateSetTrainingConfig,
) -> dict[str, Any]:
    """Train one matched arm and emit complete optimization and metric evidence."""
    config.validate()
    source = source_provenance(Path(__file__).resolve().parents[2])
    mx.set_default_device(mx.gpu)
    runtime = _runtime_identity()
    if config.production:
        _require_production_runtime(runtime)
        actual_host = _normalize_host(socket.gethostname().split(".")[0])
        if actual_host != ARM_HOSTS[config.arm]:
            raise ValueError(
                f"S4 arm {config.arm} must run on "
                f"{ARM_HOSTS[config.arm]}, not {actual_host}"
            )

    require_complete = config.production
    cache = R3ActionEditMlxCache(
        config.cache,
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
    context_cache = S4CandidateContextCache(
        config.context_cache,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
    )
    open_data = open_data_verification_identity(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
    )
    warm_start = _warm_start_identity(
        config.warm_start_checkpoint,
        require_production=config.production,
    )
    controls = (
        validate_launch_controls(
            config.authorization,
            config.preflight,
            arm=config.arm,
            cache_id=cache.cache_id,
            s1_cache_id=s1_cache.cache_id,
            context_cache_id=context_cache.cache_id,
            warm_start=warm_start,
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
    train_r3 = cache.bind_dataset(
        config.train_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=not config.production,
        preverified_open_data_proof_id=proof_id,
    )
    validation_r3 = cache.bind_dataset(
        config.validation_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=not config.production,
        preverified_open_data_proof_id=proof_id,
    )
    train = S4CandidateSetDataset(
        train_r3,
        context_cache=context_cache,
    )
    validation = S4CandidateSetDataset(
        validation_r3,
        context_cache=context_cache,
    )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    cross_arm = cross_arm_initialization(config.warm_start_checkpoint)
    model_config = S4CandidateSetModelConfig(arm=config.arm)
    parity = _initial_prediction_parity(
        validation,
        warm_start_checkpoint=config.warm_start_checkpoint,
        arm=config.arm,
    )
    run_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "arm": config.arm,
        "target_steps": config.target_steps,
        "protocol": config.protocol.to_dict(),
        "cache_id": cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "context_cache_id": context_cache.cache_id,
        "train_dataset_id": train.manifest["dataset_id"],
        "validation_dataset_id": validation.manifest["dataset_id"],
        "warm_start": warm_start,
        "initial_prediction_parity": parity,
        "source": source,
        "runtime": runtime,
        "cross_arm_initialization": cross_arm,
        "controls": controls,
    }

    if config.resume:
        existing = _read_json(
            config.run_dir / "run.json",
            "S4 run manifest",
        )
        if existing != run_manifest:
            raise ValueError("S4 resume manifest differs from the frozen run")
        model, optimizer, state, _ = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            model_factory=lambda values: S4CandidateSetRanker(
                S4CandidateSetModelConfig.from_dict(values)
            ),
        )
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError("S4 run already has checkpoints; pass --resume")
        mx.random.seed(TRAINING_SEED)
        model = S4CandidateSetRanker(model_config)
        model.load_weights(
            str(config.warm_start_checkpoint / "model.safetensors"),
            strict=False,
        )
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
        raise ValueError("S4 model graph differs from matched initialization")

    loss_and_grad = nn.value_and_grad(model, s4_candidate_set_loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    trace_path = config.run_dir / "batch-trace.jsonl"
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
        batch_identity = scientific_batch_blake3(batch.r3)
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise ValueError(
                f"S4 training produced nonfinite loss at step {step}"
            )
        candidate_count = int(
            np.asarray(batch.base.candidate_mask, dtype=np.bool_).sum()
        )
        measured_seconds += elapsed
        measured_candidates += candidate_count
        state.global_step += 1
        state.batch_in_epoch = state.global_step
        state.elapsed_seconds += elapsed
        event = {
            "schema_version": 1,
            "step": state.global_step,
            "batch_blake3": batch_identity,
            "loss": loss_value,
            "candidates": candidate_count,
            "elapsed_seconds": elapsed,
        }
        loss_trace.append(event)
        _append_json(trace_path, event)

        if state.global_step % METRIC_STEPS == 0:
            probe = evaluate_s4_candidate_set(
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
        validation_metrics = evaluate_s4_candidate_set(
            model,
            validation,
            arm=config.arm,
        )
        performance = run_isolated_serving_benchmark(
            train_dataset=config.train_dataset,
            validation_dataset=config.validation_dataset,
            cache=config.cache,
            s1_cache=config.s1_cache,
            context_cache=config.context_cache,
            run_dir=config.run_dir,
            checkpoint=checkpoint,
            arm=config.arm,
            global_step=state.global_step,
            open_data_verification=open_data,
            verification_source="cluster-preflight",
            warmup_iterations=5,
            steady_iterations=30,
        )
    else:
        smoke_rows = np.arange(
            min(validation.group_count, 5),
            dtype=np.int64,
        )
        validation_metrics = evaluate_s4_candidate_set(
            model,
            validation,
            arm=config.arm,
            rows=smoke_rows,
            prediction_panel_size=16,
        )
        performance = run_isolated_serving_benchmark(
            train_dataset=config.train_dataset,
            validation_dataset=config.validation_dataset,
            cache=config.cache,
            s1_cache=config.s1_cache,
            context_cache=config.context_cache,
            run_dir=config.run_dir,
            checkpoint=checkpoint,
            arm=config.arm,
            global_step=state.global_step,
            open_data_verification=open_data,
            verification_source="in-process-full",
            warmup_iterations=3,
            steady_iterations=10,
            decision_rows=smoke_rows,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "arm": config.arm,
        "host": _normalize_host(socket.gethostname().split(".")[0]),
        "cache_id": cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "context_cache_id": context_cache.cache_id,
        "protocol": config.protocol.to_dict(),
        "warm_start": warm_start,
        "initial_prediction_parity": parity,
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
            "cache_id",
            "s1_cache_id",
            "context_cache_id",
            "protocol",
            "warm_start",
            "initial_prediction_parity",
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
    report["report_id"] = _canonical_blake3(
        report["scientific_identity"]
    )
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def cross_arm_initialization(
    warm_start_checkpoint: Path,
) -> dict[str, Any]:
    """Return exact graph and tensor parity after the shared R3 warm start."""
    weights = warm_start_checkpoint / "model.safetensors"
    counts: dict[str, int] = {}
    layouts: dict[str, str] = {}
    tensors: dict[str, str] = {}
    for arm in S4_ARMS:
        mx.random.seed(TRAINING_SEED)
        candidate = S4CandidateSetRanker(
            S4CandidateSetModelConfig(arm=arm)
        )
        candidate.load_weights(str(weights), strict=False)
        counts[arm] = parameter_count(candidate)
        layouts[arm] = parameter_layout_blake3(candidate)
        tensors[arm] = parameter_tensor_blake3(candidate)
    if (
        len(set(counts.values())) != 1
        or len(set(layouts.values())) != 1
        or len(set(tensors.values())) != 1
    ):
        raise ValueError("S4 arm warm start or parameter graph differs")
    return {
        "parameter_count": next(iter(counts.values())),
        "parameter_layout_blake3": next(iter(layouts.values())),
        "initial_parameter_tensor_blake3": next(iter(tensors.values())),
        "cross_arm_parameter_counts": counts,
        "cross_arm_parameter_layout_blake3": layouts,
        "cross_arm_initial_parameter_tensor_blake3": tensors,
    }


def validate_launch_controls(
    authorization_path: Path | None,
    preflight_path: Path | None,
    *,
    arm: str,
    cache_id: str,
    s1_cache_id: str,
    context_cache_id: str,
    warm_start: dict[str, Any],
    source: dict[str, Any],
    runtime: dict[str, Any],
    open_data_verification: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless coordinator authorization and host preflight align."""
    if authorization_path is None or preflight_path is None:
        raise ValueError("S4 production launch controls are absent")
    authorization = _read_json(
        authorization_path,
        "S4 authorization",
    )
    preflight = _read_json(preflight_path, "S4 preflight")
    authorization_identity = authorization.get("identity")
    preflight_identity = preflight.get("identity")
    proof_id = open_data_verification_id(open_data_verification)
    checks = preflight.get("checks")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization_identity, dict)
        or _canonical_blake3(authorization_identity)
        != authorization.get("authorization_id")
        or authorization_identity.get("protocol_id") != PROTOCOL_ID
        or authorization_identity.get("cache_id") != cache_id
        or authorization_identity.get("s1_cache_id") != s1_cache_id
        or authorization_identity.get("context_cache_id")
        != context_cache_id
        or authorization_identity.get("warm_start") != warm_start
        or authorization_identity.get("authorized_arms") != list(S4_ARMS)
        or authorization_identity.get("arm_hosts") != ARM_HOSTS
        or authorization_identity.get("source_blake3")
        != source.get("v2_source_blake3")
        or authorization_identity.get("protocol")
        != S4CandidateSetTrainingProtocol().to_dict()
        or authorization_identity.get("open_data_verification")
        != open_data_verification
        or authorization_identity.get("open_data_verification_id")
        != proof_id
    ):
        raise ValueError("S4 production authorization is stale or malformed")
    if (
        preflight.get("schema_version") != 1
        or preflight.get("experiment_id") != EXPERIMENT_ID
        or preflight.get("arm") != arm
        or not isinstance(preflight_identity, dict)
        or _canonical_blake3(preflight_identity)
        != preflight.get("preflight_id")
        or preflight_identity.get("authorization_id")
        != authorization.get("authorization_id")
        or preflight_identity.get("arm") != arm
        or preflight_identity.get("host") != ARM_HOSTS[arm]
        or preflight_identity.get("runtime") != runtime
        or preflight_identity.get("warm_start") != warm_start
        or preflight_identity.get("open_data_verification_id") != proof_id
        or preflight_identity.get("mlx_gpu_verified") is not True
        or preflight_identity.get("context_cache_verified") is not True
        or preflight_identity.get("initialization_parity_verified")
        is not True
        or preflight_identity.get("prediction_parity_verified") is not True
        or preflight_identity.get("smoke_replay_verified") is not True
        or not isinstance(checks, dict)
        or any(
            value is not True
            for key, value in checks.items()
            if key != "production_training_started"
        )
        or checks.get("production_training_started") is not False
    ):
        raise ValueError("S4 arm preflight is stale or incomplete")
    return {
        "authorization_id": authorization["authorization_id"],
        "preflight_id": preflight["preflight_id"],
        "open_data_verification_id": proof_id,
        "full_preflight_verification_reused": True,
    }


def _initial_prediction_parity(
    validation: S4CandidateSetDataset,
    *,
    warm_start_checkpoint: Path,
    arm: str,
) -> dict[str, Any]:
    batch = validation.batch([0], arm=arm, transform_ids=[0])
    r3 = R3ActionEditRanker(
        R3ActionEditModelConfig(arm=R3_ARMS[3])
    )
    r3.load_weights(
        str(warm_start_checkpoint / "model.safetensors")
    )
    mx.random.seed(TRAINING_SEED)
    s4 = S4CandidateSetRanker(
        S4CandidateSetModelConfig(arm=arm)
    )
    s4.load_weights(
        str(warm_start_checkpoint / "model.safetensors"),
        strict=False,
    )
    expected = r3(batch.r3)
    observed = s4(batch)
    mx.eval(
        expected.scores,
        expected.standard_errors,
        observed.scores,
        observed.standard_errors,
    )
    expected_scores = np.asarray(expected.scores, dtype="<f4")
    observed_scores = np.asarray(observed.scores, dtype="<f4")
    expected_uncertainty = np.asarray(
        expected.standard_errors,
        dtype="<f4",
    )
    observed_uncertainty = np.asarray(
        observed.standard_errors,
        dtype="<f4",
    )
    score_equal = np.array_equal(expected_scores, observed_scores)
    uncertainty_equal = np.array_equal(
        expected_uncertainty,
        observed_uncertainty,
    )
    if not score_equal or not uncertainty_equal:
        raise ValueError("S4 zero-context warm start differs from R3 predictions")
    digest = blake3.blake3()
    digest.update(expected_scores.tobytes(order="C"))
    digest.update(expected_uncertainty.tobytes(order="C"))
    return {
        "row": 0,
        "candidates": int(
            np.asarray(batch.base.candidate_mask, dtype=np.bool_).sum()
        ),
        "scores_byte_identical": score_equal,
        "standard_errors_byte_identical": uncertainty_equal,
        "prediction_blake3": digest.hexdigest(),
    }


def _warm_start_identity(
    checkpoint: Path,
    *,
    require_production: bool,
) -> dict[str, Any]:
    manifest = _read_json(
        checkpoint / "checkpoint.json",
        "R3 warm-start checkpoint manifest",
    )
    state = _read_json(
        checkpoint / "state.json",
        "R3 warm-start trainer state",
    )
    model_config = R3ActionEditModelConfig.from_dict(
        manifest["model_config"]
    )
    files = manifest.get("files")
    model_path = checkpoint / "model.safetensors"
    if (
        manifest.get("schema_version") != 1
        or model_config.arm != R3_ARMS[3]
        or not isinstance(files, dict)
        or "model.safetensors" not in files
        or model_path.stat().st_size
        != files["model.safetensors"].get("bytes")
        or _checksum(model_path)
        != files["model.safetensors"].get("blake3")
        or not isinstance(state.get("global_step"), int)
        or state["global_step"] <= 0
        or (require_production and state["global_step"] != 3000)
    ):
        raise ValueError("S4 R3 warm-start checkpoint is invalid")
    return {
        "checkpoint_id": manifest["checkpoint_id"],
        "global_step": state["global_step"],
        "model_config": model_config.to_dict(),
        "manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(model_path),
    }


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


def _require_production_runtime(runtime: dict[str, Any]) -> None:
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    if (
        runtime.get("machine") != "arm64"
        or "gpu" not in str(mx.default_device()).lower()
    ):
        raise ValueError("S4 production training requires Apple Silicon MLX GPU")


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
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=S4_ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--preflight", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_s4_candidate_set_training(
        S4CandidateSetTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            cache=args.cache,
            s1_cache=args.s1_cache,
            context_cache=args.context_cache,
            warm_start_checkpoint=args.warm_start_checkpoint,
            run_dir=args.run_dir,
            output=args.output,
            arm=args.arm,
            resume=args.resume,
            smoke_steps=args.smoke_steps,
            authorization=args.authorization,
            preflight=args.preflight,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
