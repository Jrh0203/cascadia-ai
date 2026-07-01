"""Authorized matched-architecture MLX tournament runner for ADR 0146."""

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
from types import SimpleNamespace
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
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_OWNERSHIP_ENCODING,
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    EXPERIMENT_ID,
    FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS,
    FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS,
    GRAPH_MAX_DEGREE,
    TARGET_DIM,
    TOKEN_CAPACITY,
    TOKEN_TYPE_NAMES,
    R2SparseMlxBatch,
    R2SparseMlxCache,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    ARCHITECTURES,
    R2SparseMlxModelConfig,
    R2SparseValueModel,
    architecture_parameter_counts,
    parameter_count,
    r2_sparse_value_loss,
)
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest

ADR_ID = "0146"
PROTOCOL_ID = "r2-sparse-mlx-matched-architecture-v1"
TRAINING_SEED = 2026061702
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
VALIDATION_PROBE_ROWS = 256
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
MAX_PARAMETER_SPREAD_FRACTION = 0.03
R0_BINDING_CONTRACT = "r2-sparse-mlx-r0-control-binding-v1"
R0_COMPLETE_CLASSIFICATION = "r0_spatial_mlx_tournament_complete"
R0_EXACT_CONTROL = "exact-entity-control"

RUN_ARCHITECTURES = {
    "set-primary": "padded-set-transformer",
    "graph-primary": "directional-graph-attention",
    "perceiver-primary": "perceiver-fixed-latents",
    "set-replay": "padded-set-transformer",
}
AUTHORIZED_RUNS = tuple(RUN_ARCHITECTURES)
PRIMARY_RUNS = tuple(run for run in AUTHORIZED_RUNS if run != "set-replay")


@dataclass(frozen=True)
class R2SparseMlxTournamentProtocol:
    """Variables held identical across all architecture and replay runs."""

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
    training_benchmark_warmup_iterations: int = (
        TRAINING_BENCHMARK_WARMUP_ITERATIONS
    )
    training_benchmark_steady_iterations: int = (
        TRAINING_BENCHMARK_STEADY_ITERATIONS
    )
    validation_probe_rows: int = VALIDATION_PROBE_ROWS
    d6_sampling: str = "uniform-per-example-over-rust-verified-ids-0-through-11"
    state_encoding: str = (
        "one-common-encoder-invocation-board-local-4x92-per-position-decision"
    )
    board_interaction: str = (
        "board-local-token-trunks-cross-board-only-via-explicit-global-market-player-context"
    )
    pooling: str = "four-type-balanced-summary-tokens-per-board"
    parameter_spread_fraction: float = MAX_PARAMETER_SPREAD_FRACTION
    parameter_count_scope: str = (
        "all-trainable-parameters-including-input-adapters-embeddings-common-encoder-and-trunk"
    )
    board_ownership_encoding: str = BOARD_OWNERSHIP_ENCODING
    board_slots: int = BOARD_SLOTS
    padded_token_capacity_per_board: int = BOARD_TOKEN_CAPACITY
    padded_token_capacity_per_position: int = TOKEN_CAPACITY
    foundation_per_board_p99_active_tokens: int = (
        FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
    )
    foundation_per_board_max_active_tokens: int = (
        FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
    )
    architectures: tuple[str, ...] = ARCHITECTURES
    run_architectures: dict[str, str] = field(
        default_factory=lambda: dict(RUN_ARCHITECTURES)
    )
    parameter_counts: dict[str, int] = field(
        default_factory=architecture_parameter_counts
    )

    def validate(self) -> None:
        expected = R2SparseMlxTournamentProtocol()
        if self != expected:
            raise ValueError("R2 MLX tournament protocol drifted from ADR 0146")
        counts = list(self.parameter_counts.values())
        spread = (max(counts) - min(counts)) / min(counts)
        if spread > MAX_PARAMETER_SPREAD_FRACTION:
            raise ValueError("R2 architecture parameters exceed the matched-capacity gate")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class R2SparseMlxTournamentConfig:
    cache: Path
    corpus_lock: Path
    run_dir: Path
    output: Path
    authorization: Path
    r0_control: Path
    run_role: str
    resume: bool = False
    protocol: R2SparseMlxTournamentProtocol = field(
        default_factory=R2SparseMlxTournamentProtocol
    )

    def validate(self) -> None:
        self.protocol.validate()
        if self.run_role not in AUTHORIZED_RUNS:
            raise ValueError(f"unknown R2 MLX run role: {self.run_role}")
        if self.output.resolve() == self.authorization.resolve():
            raise ValueError("report output cannot overwrite the authorization")


def run_tournament(config: R2SparseMlxTournamentConfig) -> dict[str, Any]:
    """Train, evaluate, and benchmark one authorized architecture run."""
    config.validate()
    cache = R2SparseMlxCache(config.cache, corpus_lock=config.corpus_lock)
    source = source_provenance(Path(__file__).resolve().parents[2])
    authorization = validate_authorization(
        config.authorization,
        cache=cache,
        source=source,
        run_role=config.run_role,
        r0_control=config.r0_control,
    )
    runtime = _runtime_identity()
    _require_production_runtime(runtime)
    run_manifest = _run_manifest(config, cache, source, runtime, authorization)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    architecture = RUN_ARCHITECTURES[config.run_role]
    model_config = R2SparseMlxModelConfig(architecture=architecture)
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
            model_factory=lambda values: R2SparseValueModel(
                R2SparseMlxModelConfig.from_dict(values)
            ),
        )
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError("run already contains checkpoints; pass --resume")
        mx.random.seed(TRAINING_SEED)
        model = R2SparseValueModel(model_config)
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    expected_parameters = architecture_parameter_counts()[architecture]
    actual_parameters = parameter_count(model)
    if model.config != model_config or actual_parameters != expected_parameters:
        raise ValueError("R2 model architecture or parameter count drifted")

    loss_and_grad = nn.value_and_grad(model, r2_sparse_value_loss)
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
                "run_role": config.run_role,
                "architecture": architecture,
                "global_step": state.global_step,
                "mean_loss": float(np.mean(recent_losses)),
                "examples_per_second": measured_examples
                / max(measured_seconds, 1e-12),
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
    validation_type_ablations = evaluate_type_ablations(
        model,
        cache,
        validation_metrics,
    )
    validation_probe = validation_probe_predictions(model, cache)
    performance = benchmark_model(model, cache)
    training_step_performance = benchmark_training_step(model, cache)
    peak_process_rss = _peak_process_rss_bytes()

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol_id": PROTOCOL_ID,
        "run_role": config.run_role,
        "architecture": architecture,
        "cache": {
            "root": str(cache.root.resolve()),
            "cache_id": cache.cache_id,
            "manifest_blake3": _checksum(cache.manifest_path),
            "corpus_lock_id": cache.corpus_lock_id,
            "identity_semantic_blake3": cache.identity_semantic_blake3,
            "d6_semantic_blake3": cache.d6_semantic_blake3,
            "target_blake3": cache.target_blake3,
            "token_capacity": TOKEN_CAPACITY,
            "board_slots": BOARD_SLOTS,
            "board_token_capacity": BOARD_TOKEN_CAPACITY,
            "graph_max_degree": GRAPH_MAX_DEGREE,
            "board_ownership_encoding": BOARD_OWNERSHIP_ENCODING,
            "active_token_statistics": cache.active_token_statistics,
        },
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "r0_control_binding_id": authorization["identity"][
                "r0_control_binding_id"
            ],
            "approved_by": authorization["identity"]["approved_by"],
            "approved_unix_ms": authorization["identity"]["approved_unix_ms"],
        },
        "protocol": config.protocol.to_dict(),
        "model": {
            "config": model.config.to_dict(),
            "parameter_count": actual_parameters,
            "parameter_count_scope": config.protocol.parameter_count_scope,
            "state_encoder_invocations_per_prediction": 1,
        },
        "optimization": {
            "global_step": state.global_step,
            "training_examples": training_examples,
            "training_seconds": state.elapsed_seconds,
            "training_examples_per_second": training_examples
            / max(state.elapsed_seconds, 1e-12),
            "invocation_training_examples": measured_examples,
            "invocation_training_seconds": measured_seconds,
            "invocation_wall_seconds": invocation_wall_seconds,
            "last_checkpoint": str(checkpoint.resolve()),
            "last_checkpoint_manifest_blake3": _checksum(
                checkpoint / "checkpoint.json"
            ),
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_type_ablations": validation_type_ablations,
            "validation_probe": validation_probe,
        },
        "performance": {
            **performance,
            "training_step": training_step_performance,
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
            "exact_no_truncation_verified": True,
            "padding_zero_verified": True,
            "board_local_layout_verified": True,
            "graph_degree_bound_verified": True,
            "d6_regeneration_verified_by_exporter": True,
            "derived_relations_loaded_from_content_addressed_cache": True,
            "state_trunk_encoded_once": True,
            "type_balanced_pooling_verified": True,
            "typewise_ablation_reported": True,
            "all_metrics_finite": _all_finite(
                {
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "performance": performance,
                    "training_step": training_step_performance,
                }
            ),
            "test_or_final_data_opened": False,
        },
        "claims": {
            "matched_architecture_screen_complete": True,
            "gameplay_strength_measured": False,
            "production_model_selected": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = report_scientific_identity(report)
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def evaluate_model(
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
    split: str,
    *,
    ablated_token_type: int | None = None,
) -> dict[str, Any]:
    """Evaluate every row once with the identity D6 transform."""
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
        if ablated_token_type is not None:
            batch = _ablate_token_type(batch, ablated_token_type)
        predictions = model.predict_components(batch)
        loss = r2_sparse_value_loss(model, batch)
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
        raise ValueError(f"{split} evaluation did not consume every row exactly once")
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


def evaluate_type_ablations(
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
    full_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Measure each exact token class by masking it without rebuilding semantics."""
    output: dict[str, Any] = {}
    for token_type, name in TOKEN_TYPE_NAMES.items():
        metrics = evaluate_model(
            model,
            cache,
            "validation",
            ablated_token_type=token_type,
        )
        output[name] = {
            "masked_token_type": token_type,
            "samples": metrics["samples"],
            "mean_component_mae": metrics["mean_component_mae"],
            "total_mae": metrics["total_mae"],
            "total_rmse": metrics["total_rmse"],
            "total_bias": metrics["total_bias"],
            "total_correlation": metrics["total_correlation"],
            "delta_mean_component_mae": (
                metrics["mean_component_mae"]
                - full_metrics["mean_component_mae"]
            ),
            "delta_total_mae": metrics["total_mae"] - full_metrics["total_mae"],
            "delta_total_rmse": metrics["total_rmse"] - full_metrics["total_rmse"],
        }
    return output


def validation_probe_predictions(
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
) -> dict[str, Any]:
    """Return a fixed prediction panel used to classify the independent replay."""
    count = min(VALIDATION_PROBE_ROWS, cache.sample_count("validation"))
    indices = np.arange(count, dtype=np.int64)
    batch = cache.batch(
        "validation",
        indices,
        transform_ids=np.zeros(count, dtype=np.int64),
    )
    predictions = model.predict_components(batch)
    mx.eval(predictions)
    values = np.asarray(predictions, dtype="<f4")
    return {
        "rows": count,
        "indices": indices.tolist(),
        "predictions": values.tolist(),
        "prediction_blake3": blake3.blake3(values.tobytes(order="C")).hexdigest(),
    }


def benchmark_model(
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
) -> dict[str, Any]:
    """Measure compiled inference latency, throughput, and MLX memory."""
    count = min(INFERENCE_BATCH_SIZE, cache.sample_count("validation"))
    batch = cache.batch(
        "validation",
        np.arange(count, dtype=np.int64),
        transform_ids=np.zeros(count, dtype=np.int64),
    )
    inputs = _batch_inputs(batch)

    def predict(*values: mx.array) -> mx.array:
        return model.predict_components(_model_batch(values))

    compiled = mx.compile(predict, inputs=model.state)
    mx.clear_cache()
    mx.reset_peak_memory()
    compile_started = time.perf_counter()
    output = compiled(*inputs)
    mx.eval(output)
    compile_seconds = time.perf_counter() - compile_started

    warmup_started = time.perf_counter()
    for _ in range(WARMUP_ITERATIONS):
        output = compiled(*inputs)
        mx.eval(output)
    warmup_seconds = time.perf_counter() - warmup_started

    latencies = np.empty(STEADY_ITERATIONS, dtype=np.float64)
    for iteration in range(STEADY_ITERATIONS):
        started = time.perf_counter()
        output = compiled(*inputs)
        mx.eval(output)
        latencies[iteration] = time.perf_counter() - started
    steady_seconds = float(latencies.sum())
    total_examples = count * STEADY_ITERATIONS
    return {
        "definition": (
            "compile_seconds includes the first compiled execution; one value-scored "
            "public afterstate is one inference action"
        ),
        "batch_examples": count,
        "token_capacity": TOKEN_CAPACITY,
        "compile_seconds": compile_seconds,
        "warmup_iterations": WARMUP_ITERATIONS,
        "warmup_seconds": warmup_seconds,
        "warmup_examples_per_second": count
        * WARMUP_ITERATIONS
        / max(warmup_seconds, 1e-12),
        "steady_iterations": STEADY_ITERATIONS,
        "steady_seconds": steady_seconds,
        "steady_examples_per_second": total_examples / max(steady_seconds, 1e-12),
        "inference_actions_per_second": total_examples
        / max(steady_seconds, 1e-12),
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
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
) -> dict[str, Any]:
    """Measure forward/backward throughput without mutating optimizer state."""
    count = min(BATCH_SIZE, cache.sample_count("validation"))
    batch = cache.batch(
        "validation",
        np.arange(count, dtype=np.int64),
        transform_ids=np.zeros(count, dtype=np.int64),
    )
    loss_and_grad = nn.value_and_grad(model, r2_sparse_value_loss)
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
        "definition": "forward plus backward only; optimizer state is not mutated",
        "batch_examples": count,
        "warmup_iterations": TRAINING_BENCHMARK_WARMUP_ITERATIONS,
        "steady_iterations": TRAINING_BENCHMARK_STEADY_ITERATIONS,
        "seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-12),
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def validate_authorization(
    path: Path,
    *,
    cache: R2SparseMlxCache,
    source: dict[str, Any],
    run_role: str,
    r0_control: Path,
) -> dict[str, Any]:
    """Require a parent-created authorization bound to R0, source, and corpus."""
    try:
        authorization = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read R2 MLX authorization: {error}") from error
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization.get("identity"), dict)
    ):
        raise ValueError("R2 MLX production training is not authorized")
    identity = authorization["identity"]
    if _canonical_blake3(identity) != authorization.get("authorization_id"):
        raise ValueError("R2 MLX authorization identity hash drifted")
    if (
        identity.get("protocol_id") != PROTOCOL_ID
        or identity.get("protocol_blake3")
        != _canonical_blake3(R2SparseMlxTournamentProtocol().to_dict())
        or identity.get("corpus_lock_id") != cache.corpus_lock_id
        or identity.get("mlx_source_blake3") != source.get("v2_source_blake3")
        or identity.get("exporter_executable_blake3")
        != cache.exporter_executable_blake3
        or identity.get("authorized_runs") != list(AUTHORIZED_RUNS)
        or identity.get("run_architectures") != RUN_ARCHITECTURES
        or run_role not in identity.get("authorized_runs", [])
        or not _is_digest(identity.get("r0_control_binding_id"))
        or not isinstance(identity.get("approved_by"), str)
        or not identity["approved_by"].strip()
        or not isinstance(identity.get("approved_unix_ms"), int)
        or isinstance(identity.get("approved_unix_ms"), bool)
    ):
        raise ValueError(
            "R2 MLX authorization does not match this R0 binding, source, corpus, or run"
        )
    _validate_r0_control_binding(
        r0_control,
        expected_binding_id=identity["r0_control_binding_id"],
    )
    return authorization


def _validate_r0_control_binding(
    path: Path,
    *,
    expected_binding_id: str,
) -> dict[str, Any]:
    try:
        binding = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read R0 control binding: {error}") from error
    identity = binding.get("identity") if isinstance(binding, dict) else None
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != 1
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("adr") != ADR_ID
        or binding.get("contract_id") != R0_BINDING_CONTRACT
        or not isinstance(identity, dict)
        or _canonical_blake3(identity) != binding.get("binding_id")
        or binding.get("binding_id") != expected_binding_id
        or identity.get("r0_classification") != R0_COMPLETE_CLASSIFICATION
        or identity.get("classification_order_byte_identical") is not True
        or not isinstance(identity.get("validation"), dict)
        or not _all_finite(identity["validation"])
    ):
        raise ValueError("R0 selected-control binding is absent, stale, or incomplete")
    selected = identity.get("r0_selected_stage2_candidate")
    expected_control = selected if selected is not None else R0_EXACT_CONTROL
    if identity.get("selected_control_arm") != expected_control:
        raise ValueError("R0 selected-control fallback is not explicit")
    return binding


def verify_only(cache_path: Path, corpus_lock: Path) -> dict[str, Any]:
    cache = R2SparseMlxCache(cache_path, corpus_lock=corpus_lock)
    counts = architecture_parameter_counts()
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "cache_id": cache.cache_id,
        "corpus_lock_id": cache.corpus_lock_id,
        "train_records": cache.sample_count("train"),
        "validation_records": cache.sample_count("validation"),
        "token_capacity": TOKEN_CAPACITY,
        "board_slots": BOARD_SLOTS,
        "board_token_capacity": BOARD_TOKEN_CAPACITY,
        "active_token_statistics": cache.active_token_statistics,
        "parameter_counts": counts,
        "parameter_spread_fraction": (max(counts.values()) - min(counts.values()))
        / min(counts.values()),
        "semantic_integrity_verified": True,
        "padding_integrity_verified": True,
        "training_started": False,
    }


def _ablate_token_type(
    batch: R2SparseMlxBatch,
    token_type: int,
) -> R2SparseMlxBatch:
    if token_type not in TOKEN_TYPE_NAMES:
        raise ValueError("R2 token ablation type must be in [1, 4]")
    types = np.asarray(batch.token_types, dtype=np.int32)
    mask = np.asarray(batch.token_mask, dtype=np.bool_)
    selected = mask & (types == token_type)
    retained_mask = mask & ~selected
    features = np.asarray(batch.token_features).copy()
    features[selected] = 0.0
    retained_types = types.copy()
    retained_types[selected] = 0

    neighbors = np.asarray(batch.graph_neighbors, dtype=np.int32)
    neighbor_mask = np.asarray(batch.graph_neighbor_mask, dtype=np.bool_)
    batch_index = np.arange(types.shape[0])[:, None, None, None]
    board_index = np.arange(BOARD_SLOTS)[None, :, None, None]
    target_types = types[batch_index, board_index, neighbors]
    retained_edges = (
        neighbor_mask
        & (~selected[..., None])
        & (target_types != token_type)
    )
    return R2SparseMlxBatch(
        token_features=mx.array(features),
        token_types=mx.array(retained_types),
        token_mask=mx.array(retained_mask),
        graph_neighbors=batch.graph_neighbors,
        graph_neighbor_mask=mx.array(retained_edges),
        graph_relations=batch.graph_relations,
        graph_direction_features=batch.graph_direction_features,
        market_features=batch.market_features,
        market_mask=batch.market_mask,
        player_features=batch.player_features,
        player_mask=batch.player_mask,
        global_features=batch.global_features,
        targets=batch.targets,
        game_index=batch.game_index,
        turn=batch.turn,
        transform_ids=batch.transform_ids,
    )


def _batch_inputs(batch: R2SparseMlxBatch) -> tuple[mx.array, ...]:
    return (
        batch.token_features,
        batch.token_types,
        batch.token_mask,
        batch.graph_neighbors,
        batch.graph_neighbor_mask,
        batch.graph_relations,
        batch.graph_direction_features,
        batch.market_features,
        batch.market_mask,
        batch.player_features,
        batch.player_mask,
        batch.global_features,
    )


def _model_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        token_features=values[0],
        token_types=values[1],
        token_mask=values[2],
        graph_neighbors=values[3],
        graph_neighbor_mask=values[4],
        graph_relations=values[5],
        graph_direction_features=values[6],
        market_features=values[7],
        market_mask=values[8],
        player_features=values[9],
        player_mask=values[10],
        global_features=values[11],
    )


def _run_manifest(
    config: R2SparseMlxTournamentConfig,
    cache: R2SparseMlxCache,
    source: dict[str, Any],
    runtime: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training": {
            "protocol": config.protocol.to_dict(),
            "run_role": config.run_role,
            "architecture": RUN_ARCHITECTURES[config.run_role],
            "cache_id": cache.cache_id,
            "run_dir": str(config.run_dir.resolve()),
            "resume": config.resume,
        },
        "datasets": {
            "cache_id": cache.cache_id,
            "cache_manifest_blake3": _checksum(cache.manifest_path),
            "corpus_lock_id": cache.corpus_lock_id,
            "identity_semantic_blake3": cache.identity_semantic_blake3,
            "d6_semantic_blake3": cache.d6_semantic_blake3,
            "target_blake3": cache.target_blake3,
            "train_samples": cache.sample_count("train"),
            "validation_samples": cache.sample_count("validation"),
        },
        "runtime": runtime,
        "source": source,
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "r0_control_binding_id": authorization["identity"][
                "r0_control_binding_id"
            ],
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


def _require_production_runtime(runtime: dict[str, Any]) -> None:
    if platform.system() != "Darwin" or runtime.get("machine") != "arm64":
        raise ValueError("R2 MLX production requires Apple Silicon macOS")
    if "gpu" not in str(runtime.get("device", "")).lower():
        raise ValueError("R2 MLX production requires the MLX GPU device")


def report_scientific_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Return the exact report fields bound by the report content address."""
    return {
        "schema_version": report["schema_version"],
        "adr": report["adr"],
        "architecture": report["architecture"],
        "authorization": report["authorization"],
        "cache": {
            key: report["cache"][key]
            for key in (
                "cache_id",
                "corpus_lock_id",
                "identity_semantic_blake3",
                "d6_semantic_blake3",
                "target_blake3",
                "manifest_blake3",
                "token_capacity",
                "board_slots",
                "board_token_capacity",
                "graph_max_degree",
                "board_ownership_encoding",
                "active_token_statistics",
            )
        },
        "experiment_id": report["experiment_id"],
        "claims": report["claims"],
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
        "run_role": report["run_role"],
        "runtime": report["runtime"],
        "source": report["source"],
    }


def _calibration_metrics(count: int, statistics: list[float]) -> dict[str, float]:
    predicted_sum, target_sum, predicted_square_sum, target_square_sum, cross_sum = (
        statistics
    )
    predicted_mean = predicted_sum / count
    target_mean = target_sum / count
    predicted_variance = max(
        predicted_square_sum / count - predicted_mean**2,
        0.0,
    )
    target_variance = max(target_square_sum / count - target_mean**2, 0.0)
    covariance = cross_sum / count - predicted_mean * target_mean
    denominator = math.sqrt(predicted_variance * target_variance)
    slope = covariance / predicted_variance if predicted_variance > 0 else 0.0
    return {
        "predicted_total_mean": predicted_mean,
        "target_total_mean": target_mean,
        "total_bias": predicted_mean - target_mean,
        "total_correlation": covariance / denominator if denominator > 0 else 0.0,
        "calibration_slope": slope,
        "calibration_intercept": target_mean - slope * predicted_mean,
    }


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, bool, str)) or value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
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
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus-lock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--r0-control", type=Path)
    parser.add_argument("--run-role", choices=AUTHORIZED_RUNS)
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
                ("--r0-control", args.r0_control),
                ("--run-role", args.run_role),
            )
            if value is None
        ]
        if missing:
            parser.error(f"production training requires {', '.join(missing)}")
        report = run_tournament(
            R2SparseMlxTournamentConfig(
                cache=args.cache,
                corpus_lock=args.corpus_lock,
                run_dir=args.run_dir,
                output=args.output,
                authorization=args.authorization,
                r0_control=args.r0_control,
                run_role=args.run_role,
                resume=args.resume,
            )
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
