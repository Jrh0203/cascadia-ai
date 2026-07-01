"""Controlled MLX iso-architecture tournament runner for one R0 spatial arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
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
from cascadia_mlx.r0_spatial_mlx_cache import (
    ARM_TOKEN_CAPACITY,
    EXPERIMENT_ID,
    TARGET_DIM,
    R0SpatialMlxBatch,
    R0SpatialMlxCache,
)
from cascadia_mlx.r0_spatial_mlx_model import (
    R0SpatialIsoValueModel,
    R0SpatialMlxModelConfig,
    parameter_count,
    r0_spatial_value_loss,
)
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest

ADR_ID = "0142"
PROTOCOL_ID = "r0-spatial-mlx-iso-architecture-v1"
TRAINING_SEED = 2026061701
TRAINING_STEPS = 500
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 100
METRIC_STEPS = 25
EVALUATION_BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 64
WARMUP_ITERATIONS = 5
STEADY_ITERATIONS = 30
TRAINING_BENCHMARK_WARMUP_ITERATIONS = 2
TRAINING_BENCHMARK_STEADY_ITERATIONS = 10
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
AUTHORIZED_ARMS = tuple(ARM_TOKEN_CAPACITY)


@dataclass(frozen=True)
class R0SpatialMlxTournamentProtocol:
    """The preregistered variables held constant across all five arms."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    optimizer: str = "adamw"
    training_steps: int = TRAINING_STEPS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE
    inference_batch_size: int = INFERENCE_BATCH_SIZE
    warmup_iterations: int = WARMUP_ITERATIONS
    steady_iterations: int = STEADY_ITERATIONS
    training_benchmark_warmup_iterations: int = TRAINING_BENCHMARK_WARMUP_ITERATIONS
    training_benchmark_steady_iterations: int = TRAINING_BENCHMARK_STEADY_ITERATIONS
    d6_sampling: str = "uniform-per-example-over-rust-exported-ids-0-through-11"
    model: R0SpatialMlxModelConfig = field(default_factory=R0SpatialMlxModelConfig)

    def validate(self) -> None:
        if self != R0SpatialMlxTournamentProtocol():
            raise ValueError("R0 MLX tournament protocol drifted from ADR 0142")
        self.model.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["model"] = self.model.to_dict()
        return value


@dataclass(frozen=True)
class R0SpatialMlxTournamentConfig:
    cache: Path
    corpus_lock: Path
    run_dir: Path
    output: Path
    authorization: Path
    resume: bool = False
    protocol: R0SpatialMlxTournamentProtocol = field(default_factory=R0SpatialMlxTournamentProtocol)

    def validate(self) -> None:
        self.protocol.validate()
        if self.output.resolve() == self.authorization.resolve():
            raise ValueError("report output cannot overwrite the authorization")


def run_tournament(config: R0SpatialMlxTournamentConfig) -> dict[str, Any]:
    """Train, evaluate, and benchmark one authorized representation arm."""
    config.validate()
    cache = R0SpatialMlxCache(config.cache, corpus_lock=config.corpus_lock)
    source = source_provenance(Path(__file__).resolve().parents[2])
    authorization = validate_authorization(
        config.authorization,
        cache=cache,
        source=source,
    )
    runtime = _runtime_identity()
    run_manifest = _run_manifest(config, cache, source, runtime, authorization)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    if config.resume:
        validate_resume_manifest(
            config.run_dir,
            training=run_manifest["training"],
            datasets=run_manifest["datasets"],
            runtime=run_manifest["runtime"],
            source=run_manifest["source"],
        )
        model, optimizer, state, _ = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            model_factory=lambda values: R0SpatialIsoValueModel(
                R0SpatialMlxModelConfig.from_dict(values)
            ),
        )
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError("run already contains checkpoints; pass --resume")
        mx.random.seed(TRAINING_SEED)
        model = R0SpatialIsoValueModel()
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    expected_parameters = parameter_count(R0SpatialIsoValueModel())
    actual_parameters = parameter_count(model)
    if actual_parameters != expected_parameters:
        raise ValueError("R0 model parameter count drifted")
    loss_and_grad = nn.value_and_grad(model, r0_spatial_value_loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    measured_examples = 0
    measured_seconds = 0.0
    recent_losses: list[float] = []

    model.train()
    invocation_started = time.perf_counter()
    while state.global_step < TRAINING_STEPS:
        step = state.global_step
        step_started = time.perf_counter()
        batch = cache.deterministic_training_batch(
            step=step,
            batch_size=BATCH_SIZE,
            seed=TRAINING_SEED,
        )
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - step_started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise ValueError(f"training produced a nonfinite loss at step {step}")
        measured_examples += BATCH_SIZE
        measured_seconds += elapsed
        recent_losses.append(loss_value)
        state.global_step += 1
        state.epoch = 0
        state.batch_in_epoch = state.global_step
        state.elapsed_seconds += elapsed

        if state.global_step % METRIC_STEPS == 0:
            event = {
                "schema_version": 1,
                "arm": cache.arm,
                "global_step": state.global_step,
                "mean_loss": float(np.mean(recent_losses)),
                "examples_per_second": measured_examples / max(measured_seconds, 1e-12),
                "peak_active_memory_bytes": int(mx.get_peak_memory()),
            }
            _append_json(metrics_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            recent_losses.clear()
        if state.global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(config.run_dir, model, optimizer, state)
            prune_checkpoints(config.run_dir)

    checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
    prune_checkpoints(config.run_dir)
    training_peak_memory = int(mx.get_peak_memory())
    invocation_wall_seconds = time.perf_counter() - invocation_started
    training_examples = state.global_step * BATCH_SIZE
    model.eval()
    train_metrics = evaluate_model(model, cache, "train")
    validation_metrics = evaluate_model(model, cache, "validation")
    arm_performance = benchmark_model(model, cache)
    exact_shape_performance = benchmark_model(model, cache, exact_shape_control=True)
    arm_training_step = benchmark_training_step(model, cache)
    exact_shape_training_step = benchmark_training_step(
        model,
        cache,
        exact_shape_control=True,
    )
    performance = {
        **arm_performance,
        "same_host_exact_shape_control": exact_shape_performance,
        "same_host_training_step": arm_training_step,
        "same_host_exact_shape_training_step": exact_shape_training_step,
        "same_host_shape_ratios": {
            "inference_examples_per_second": (
                arm_performance["steady_examples_per_second"]
                / exact_shape_performance["steady_examples_per_second"]
            ),
            "training_examples_per_second": (
                arm_training_step["examples_per_second"]
                / exact_shape_training_step["examples_per_second"]
            ),
            "inference_peak_memory_fraction": (
                arm_performance["inference_peak_active_memory_bytes"]
                / max(
                    exact_shape_performance["inference_peak_active_memory_bytes"],
                    1,
                )
            ),
            "training_peak_memory_fraction": (
                arm_training_step["peak_active_memory_bytes"]
                / max(exact_shape_training_step["peak_active_memory_bytes"], 1)
            ),
        },
    }
    peak_process_rss = _peak_process_rss_bytes()

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol_id": PROTOCOL_ID,
        "arm": cache.arm,
        "cache": {
            "root": str(cache.root.resolve()),
            "cache_id": cache.cache_id,
            "manifest_blake3": _checksum(cache.manifest_path),
            "corpus_lock_id": cache.corpus_lock_id,
            "source_semantic_blake3": cache.source_semantic_blake3,
            "d6_semantic_blake3": cache.d6_semantic_blake3,
            "target_blake3": cache.target_blake3,
            "spatial_token_capacity": cache.token_capacity,
            "local_capacity": cache.local_capacity,
        },
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "approved_by": authorization["identity"]["approved_by"],
            "approved_unix_ms": authorization["identity"]["approved_unix_ms"],
        },
        "protocol": config.protocol.to_dict(),
        "model": {
            "config": model.config.to_dict(),
            "parameter_count": actual_parameters,
        },
        "optimization": {
            "global_step": state.global_step,
            "training_examples": training_examples,
            "training_seconds": state.elapsed_seconds,
            "training_examples_per_second": training_examples / max(state.elapsed_seconds, 1e-12),
            "invocation_training_examples": measured_examples,
            "invocation_training_seconds": measured_seconds,
            "invocation_wall_seconds": invocation_wall_seconds,
            "last_checkpoint": str(checkpoint.resolve()),
            "last_checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
        },
        "performance": {
            **performance,
            "training_peak_active_memory_bytes": training_peak_memory,
            "peak_process_rss_bytes": peak_process_rss,
        },
        "runtime": {
            **runtime,
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "source": source,
        "integrity": {
            "cache_verified": True,
            "padding_verified": True,
            "semantic_round_trip_verified": True,
            "overflow_exact_entities_retained": True,
            "all_metrics_finite": _all_finite(
                {
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "performance": performance,
                }
            ),
            "test_or_final_data_opened": False,
        },
        "claims": {
            "learned_representation_screen_complete": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = _report_scientific_identity(report)
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def evaluate_model(
    model: R0SpatialIsoValueModel,
    cache: R0SpatialMlxCache,
    split: str,
) -> dict[str, Any]:
    """Evaluate every row exactly once using the identity transform."""
    model.eval()
    count = 0
    loss_sum = 0.0
    absolute_component = np.zeros(TARGET_DIM, dtype=np.float64)
    squared_component = np.zeros(TARGET_DIM, dtype=np.float64)
    signed_component = np.zeros(TARGET_DIM, dtype=np.float64)
    absolute_total = 0.0
    squared_total = 0.0
    statistics = np.zeros(5, dtype=np.float64)
    for batch in cache.sequential_batches(
        split,
        EVALUATION_BATCH_SIZE,
        transform_id=0,
    ):
        predictions = model.predict_components(
            batch.spatial_tokens,
            batch.spatial_mask,
            batch.market_features,
            batch.market_mask,
            batch.global_features,
        )
        loss = r0_spatial_value_loss(model, batch)
        mx.eval(predictions, loss)
        predicted = np.asarray(predictions, dtype=np.float64)
        target = np.asarray(batch.targets, dtype=np.float64)
        errors = predicted - target
        batch_count = len(predicted)
        count += batch_count
        loss_sum += float(loss.item()) * batch_count
        absolute_component += np.abs(errors).sum(axis=0)
        squared_component += np.square(errors).sum(axis=0)
        signed_component += errors.sum(axis=0)
        predicted_total = predicted.sum(axis=1)
        target_total = target.sum(axis=1)
        total_errors = predicted_total - target_total
        absolute_total += float(np.abs(total_errors).sum())
        squared_total += float(np.square(total_errors).sum())
        statistics += np.array(
            [
                predicted_total.sum(),
                target_total.sum(),
                np.square(predicted_total).sum(),
                np.square(target_total).sum(),
                (predicted_total * target_total).sum(),
            ]
        )
    if count != cache.sample_count(split):
        raise ValueError(f"{split} evaluation did not consume every cache row exactly once")
    return {
        "samples": count,
        "loss": loss_sum / count,
        "component_mae": (absolute_component / count).tolist(),
        "component_rmse": np.sqrt(squared_component / count).tolist(),
        "component_bias": (signed_component / count).tolist(),
        "mean_component_mae": float(np.mean(absolute_component / count)),
        "total_mae": absolute_total / count,
        "total_rmse": math.sqrt(squared_total / count),
        **_calibration_metrics(count, statistics.tolist()),
    }


def benchmark_model(
    model: R0SpatialIsoValueModel,
    cache: R0SpatialMlxCache,
    *,
    exact_shape_control: bool = False,
) -> dict[str, Any]:
    """Measure cold compile, warmup, steady inference, and allocator-native memory."""
    count = min(INFERENCE_BATCH_SIZE, cache.sample_count("validation"))
    indices = np.arange(count, dtype=np.int64)
    batch = cache.batch(
        "validation",
        indices,
        transform_ids=np.zeros(count, dtype=np.int64),
    )
    if exact_shape_control:
        batch = _pack_exact_shape(batch)

    def predict(
        spatial_tokens: mx.array,
        spatial_mask: mx.array,
        market_features: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        return model.predict_components(
            spatial_tokens,
            spatial_mask,
            market_features,
            market_mask,
            global_features,
        )

    compiled = mx.compile(predict, inputs=model.state)
    mx.clear_cache()
    mx.reset_peak_memory()
    compile_started = time.perf_counter()
    output = compiled(
        batch.spatial_tokens,
        batch.spatial_mask,
        batch.market_features,
        batch.market_mask,
        batch.global_features,
    )
    mx.eval(output)
    compile_seconds = time.perf_counter() - compile_started

    warmup_started = time.perf_counter()
    for _ in range(WARMUP_ITERATIONS):
        output = compiled(
            batch.spatial_tokens,
            batch.spatial_mask,
            batch.market_features,
            batch.market_mask,
            batch.global_features,
        )
        mx.eval(output)
    warmup_seconds = time.perf_counter() - warmup_started

    latencies = np.empty(STEADY_ITERATIONS, dtype=np.float64)
    for iteration in range(STEADY_ITERATIONS):
        started = time.perf_counter()
        output = compiled(
            batch.spatial_tokens,
            batch.spatial_mask,
            batch.market_features,
            batch.market_mask,
            batch.global_features,
        )
        mx.eval(output)
        latencies[iteration] = time.perf_counter() - started
    steady_seconds = float(latencies.sum())
    total_examples = count * STEADY_ITERATIONS
    return {
        "definition": (
            "compile_seconds is the first compiled invocation including mandatory first "
            "execution; actions_per_second treats one value-scored afterstate as one action"
        ),
        "spatial_token_capacity": int(batch.spatial_tokens.shape[2]),
        "same_host_exact_shape_control": exact_shape_control,
        "compile_seconds": compile_seconds,
        "compile_batch_examples": count,
        "warmup_iterations": WARMUP_ITERATIONS,
        "warmup_seconds": warmup_seconds,
        "warmup_examples_per_second": count * WARMUP_ITERATIONS / max(warmup_seconds, 1e-12),
        "steady_iterations": STEADY_ITERATIONS,
        "steady_seconds": steady_seconds,
        "steady_examples_per_second": total_examples / max(steady_seconds, 1e-12),
        "inference_actions_per_second": total_examples / max(steady_seconds, 1e-12),
        "latency_milliseconds": {
            "p50": float(np.quantile(latencies, 0.50) * 1000.0),
            "p90": float(np.quantile(latencies, 0.90) * 1000.0),
            "p99": float(np.quantile(latencies, 0.99) * 1000.0),
        },
        "inference_peak_active_memory_bytes": int(mx.get_peak_memory()),
        "inference_active_memory_bytes": int(mx.get_active_memory()),
        "inference_cache_memory_bytes": int(mx.get_cache_memory()),
    }


def benchmark_training_step(
    model: R0SpatialIsoValueModel,
    cache: R0SpatialMlxCache,
    *,
    exact_shape_control: bool = False,
) -> dict[str, Any]:
    """Measure forward/backward throughput without mutating model or optimizer state."""
    count = min(BATCH_SIZE, cache.sample_count("validation"))
    batch = cache.batch(
        "validation",
        np.arange(count, dtype=np.int64),
        transform_ids=np.zeros(count, dtype=np.int64),
    )
    if exact_shape_control:
        batch = _pack_exact_shape(batch)
    loss_and_grad = nn.value_and_grad(model, r0_spatial_value_loss)
    mx.clear_cache()
    mx.reset_peak_memory()
    model.train()
    for _ in range(TRAINING_BENCHMARK_WARMUP_ITERATIONS):
        loss, gradients = loss_and_grad(model, batch)
        mx.eval(loss, gradients)
    started = time.perf_counter()
    for _ in range(TRAINING_BENCHMARK_STEADY_ITERATIONS):
        loss, gradients = loss_and_grad(model, batch)
        mx.eval(loss, gradients)
    elapsed = time.perf_counter() - started
    model.eval()
    examples = count * TRAINING_BENCHMARK_STEADY_ITERATIONS
    return {
        "definition": "forward plus backward only; no optimizer mutation",
        "spatial_token_capacity": int(batch.spatial_tokens.shape[2]),
        "same_host_exact_shape_control": exact_shape_control,
        "warmup_iterations": TRAINING_BENCHMARK_WARMUP_ITERATIONS,
        "steady_iterations": TRAINING_BENCHMARK_STEADY_ITERATIONS,
        "seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-12),
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def _pack_exact_shape(batch: R0SpatialMlxBatch) -> R0SpatialMlxBatch:
    """Pack active Rust-authored rows into the exact control's 23-token shape."""
    source = np.asarray(batch.spatial_tokens, dtype=np.int32)
    source_mask = np.asarray(batch.spatial_mask, dtype=np.bool_)
    packed = np.zeros((len(source), 4, 23, source.shape[-1]), dtype=np.int32)
    packed_mask = np.zeros((len(source), 4, 23), dtype=np.bool_)
    for batch_index in range(len(source)):
        for board_index in range(4):
            active = source[batch_index, board_index][source_mask[batch_index, board_index]]
            if len(active) > 23:
                raise ValueError("R0 board exceeds the exact entity control capacity")
            packed[batch_index, board_index, : len(active)] = active
            packed_mask[batch_index, board_index, : len(active)] = True
    return R0SpatialMlxBatch(
        spatial_tokens=mx.array(packed),
        spatial_mask=mx.array(packed_mask),
        market_features=batch.market_features,
        market_mask=batch.market_mask,
        global_features=batch.global_features,
        targets=batch.targets,
        game_index=batch.game_index,
        turn=batch.turn,
    )


def validate_authorization(
    path: Path,
    *,
    cache: R0SpatialMlxCache,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Require an explicit parent-created gate before any production optimizer step."""
    try:
        authorization = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read R0 MLX authorization: {error}") from error
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization.get("identity"), dict)
    ):
        raise ValueError("R0 MLX production training is not authorized")
    identity = authorization["identity"]
    if _canonical_blake3(identity) != authorization.get("authorization_id"):
        raise ValueError("R0 MLX authorization identity hash drifted")
    if (
        identity.get("protocol_id") != PROTOCOL_ID
        or identity.get("protocol_blake3")
        != _canonical_blake3(R0SpatialMlxTournamentProtocol().to_dict())
        or identity.get("corpus_lock_id") != cache.corpus_lock_id
        or identity.get("mlx_source_blake3") != source.get("v2_source_blake3")
        or identity.get("exporter_executable_blake3") != cache.exporter_executable_blake3
        or identity.get("authorized_arms") != list(AUTHORIZED_ARMS)
        or cache.arm not in identity.get("authorized_arms", [])
        or not isinstance(identity.get("approved_by"), str)
        or not identity["approved_by"].strip()
        or not isinstance(identity.get("approved_unix_ms"), int)
    ):
        raise ValueError("R0 MLX authorization does not match this source, corpus, or arm")
    return authorization


def verify_only(cache_path: Path, corpus_lock: Path) -> dict[str, Any]:
    cache = R0SpatialMlxCache(cache_path, corpus_lock=corpus_lock)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": cache.arm,
        "cache_id": cache.cache_id,
        "corpus_lock_id": cache.corpus_lock_id,
        "train_records": cache.sample_count("train"),
        "validation_records": cache.sample_count("validation"),
        "spatial_token_capacity": cache.token_capacity,
        "semantic_integrity_verified": True,
        "padding_integrity_verified": True,
        "training_started": False,
    }


def _run_manifest(
    config: R0SpatialMlxTournamentConfig,
    cache: R0SpatialMlxCache,
    source: dict[str, Any],
    runtime: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training": {
            "protocol": config.protocol.to_dict(),
            "arm": cache.arm,
            "cache_id": cache.cache_id,
            "run_dir": str(config.run_dir.resolve()),
            "resume": config.resume,
        },
        "datasets": {
            "cache_id": cache.cache_id,
            "cache_manifest_blake3": _checksum(cache.manifest_path),
            "corpus_lock_id": cache.corpus_lock_id,
            "source_semantic_blake3": cache.source_semantic_blake3,
            "d6_semantic_blake3": cache.d6_semantic_blake3,
            "target_blake3": cache.target_blake3,
            "train_samples": cache.sample_count("train"),
            "validation_samples": cache.sample_count("validation"),
        },
        "runtime": runtime,
        "source": source,
        "authorization": {
            "authorization_id": authorization["authorization_id"],
        },
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "mlx_version": version("mlx"),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
    }


def _report_scientific_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "adr": report["adr"],
        "arm": report["arm"],
        "cache": {
            key: report["cache"][key]
            for key in (
                "cache_id",
                "corpus_lock_id",
                "source_semantic_blake3",
                "d6_semantic_blake3",
                "target_blake3",
                "spatial_token_capacity",
                "local_capacity",
            )
        },
        "experiment_id": report["experiment_id"],
        "integrity": report["integrity"],
        "metrics": report["metrics"],
        "model": report["model"],
        "optimization": {
            key: report["optimization"][key]
            for key in (
                "global_step",
                "training_examples",
                "training_seconds",
                "training_examples_per_second",
                "last_checkpoint_manifest_blake3",
            )
        },
        "performance": report["performance"],
        "protocol": report["protocol"],
        "protocol_id": report["protocol_id"],
        "runtime": report["runtime"],
        "source": report["source"],
    }


def _calibration_metrics(count: int, statistics: list[float]) -> dict[str, float]:
    predicted_sum, target_sum, predicted_square_sum, target_square_sum, cross_sum = statistics
    predicted_mean = predicted_sum / count
    target_mean = target_sum / count
    predicted_variance = max(predicted_square_sum / count - predicted_mean**2, 0.0)
    target_variance = max(target_square_sum / count - target_mean**2, 0.0)
    covariance = cross_sum / count - predicted_mean * target_mean
    denominator = math.sqrt(predicted_variance * target_variance)
    return {
        "predicted_total_mean": predicted_mean,
        "target_total_mean": target_mean,
        "total_bias": predicted_mean - target_mean,
        "total_correlation": covariance / denominator if denominator > 0 else 0.0,
        "calibration_slope": covariance / predicted_variance if predicted_variance > 0 else 0.0,
        "calibration_intercept": target_mean
        - (covariance / predicted_variance if predicted_variance > 0 else 0.0) * predicted_mean,
    }


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, bool, str)) or value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _peak_process_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus-lock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        report = verify_only(args.cache, args.corpus_lock)
    else:
        missing = [
            name
            for name, value in (
                ("--run-dir", args.run_dir),
                ("--output", args.output),
                ("--authorization", args.authorization),
            )
            if value is None
        ]
        if missing:
            parser.error(f"production training requires {', '.join(missing)}")
        report = run_tournament(
            R0SpatialMlxTournamentConfig(
                cache=args.cache,
                corpus_lock=args.corpus_lock,
                run_dir=args.run_dir,
                output=args.output,
                authorization=args.authorization,
                resume=args.resume,
            )
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
