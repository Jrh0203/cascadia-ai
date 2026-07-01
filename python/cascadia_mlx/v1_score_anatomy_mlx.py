"""Matched scalar-versus-component supervision on the exact sparse R2 trunk."""

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
from mlx.utils import tree_flatten

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_TOKEN_CAPACITY,
    TARGET_DIM,
    R2SparseMlxCache,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    TARGET_SCALES,
    R2SparseMlxModelConfig,
    R2SparseValueModel,
    parameter_count,
    r2_sparse_value_loss,
)
from cascadia_mlx.run_manifest import validate_resume_manifest

EXPERIMENT_ID = "v1-score-anatomy-matched-r2-v1"
ADR_ID = "0176"
PROTOCOL_ID = "v1-score-anatomy-matched-r2-v1"
ARMS = ("scalar-total-control", "component-anatomy")
ROLES = (
    "scalar-primary",
    "anatomy-primary",
    "scalar-replay",
    "anatomy-replay",
)
ROLE_ARMS = {
    "scalar-primary": "scalar-total-control",
    "anatomy-primary": "component-anatomy",
    "scalar-replay": "scalar-total-control",
    "anatomy-replay": "component-anatomy",
}
COMPONENT_NAMES = (
    "mountain",
    "forest",
    "prairie",
    "wetland",
    "river",
    "bear",
    "elk",
    "salmon",
    "hawk",
    "fox",
    "nature_tokens",
)

TRAINING_SEED = 2026061801
TRAINING_STEPS = 3_000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 500
METRIC_STEPS = 100
EVALUATION_BATCH_SIZE = 64
PAIRWISE_TEMPERATURE = 2.0
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
DENSE_HEX_BUDGET_REFERENCE = 121
MIN_CORRELATION_GAIN = 0.03
MIN_PAIRWISE_LOG_LOSS_GAIN = 0.005
MAX_TOTAL_MAE_REGRESSION = 0.05


class V1ScoreAnatomyError(ValueError):
    """Raised when the frozen V1 comparison contract is violated."""


@dataclass(frozen=True)
class V1ScoreAnatomyProtocol:
    """Every variable held fixed across scalar and component supervision."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    training_steps: int = TRAINING_STEPS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE
    pairwise_temperature: float = PAIRWISE_TEMPERATURE
    architecture: str = "perceiver-fixed-latents"
    target_scales: tuple[float, ...] = (
        23.0,
        23.0,
        23.0,
        23.0,
        23.0,
        30.0,
        28.0,
        28.0,
        28.0,
        40.0,
        20.0,
    )
    component_names: tuple[str, ...] = COMPONENT_NAMES
    scalar_objective: str = "mean-square-final-total-error-divided-by-100-squared"
    anatomy_objective: str = (
        "normalized-component-mse-plus-half-normalized-total-mse"
    )
    state_contract: str = "exact-r2-sparse-occupied-frontier-component-motif"
    dense_hex_budget_reference: int = DENSE_HEX_BUDGET_REFERENCE
    max_sparse_objects_per_board: int = BOARD_TOKEN_CAPACITY

    def validate(self) -> None:
        expected = V1ScoreAnatomyProtocol()
        if self != expected:
            raise V1ScoreAnatomyError("V1 score-anatomy protocol drifted")
        if self.max_sparse_objects_per_board > self.dense_hex_budget_reference:
            raise V1ScoreAnatomyError("exact sparse state exceeds the 121-object reference")
        if len(self.component_names) != TARGET_DIM:
            raise V1ScoreAnatomyError("component catalog no longer matches target width")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class V1ScoreAnatomyRunConfig:
    cache: Path
    corpus_lock: Path
    authorization: Path
    run_dir: Path
    output: Path
    bundle_id: str
    role: str
    resume: bool = False
    protocol: V1ScoreAnatomyProtocol = field(
        default_factory=V1ScoreAnatomyProtocol
    )

    def validate(self) -> None:
        self.protocol.validate()
        if self.role not in ROLES:
            raise V1ScoreAnatomyError(f"unknown V1 run role: {self.role}")
        _require_digest(self.bundle_id, "bundle ID")


def scalar_total_loss(model: R2SparseValueModel, batch: object) -> mx.array:
    """Train only the final total while preserving the 11-output parameter graph."""
    normalized_components = model(batch)
    predicted_total = mx.sum(normalized_components * TARGET_SCALES, axis=-1)
    target_total = mx.sum(batch.targets, axis=-1)
    return mx.mean(mx.square((predicted_total - target_total) / 100.0))


def loss_for_arm(
    arm: str,
):
    if arm == "scalar-total-control":
        return scalar_total_loss
    if arm == "component-anatomy":
        return r2_sparse_value_loss
    raise V1ScoreAnatomyError(f"unknown V1 supervision arm: {arm}")


def build_authorization(
    *,
    cache: Path,
    corpus_lock: Path,
    bundle_id: str,
) -> dict[str, Any]:
    """Bind the exact cache, common initialization, and immutable source bundle."""
    _require_digest(bundle_id, "bundle ID")
    protocol = V1ScoreAnatomyProtocol()
    protocol.validate()
    sparse_cache = R2SparseMlxCache(cache, corpus_lock=corpus_lock)
    mx.random.seed(TRAINING_SEED)
    model = R2SparseValueModel(
        R2SparseMlxModelConfig(architecture=protocol.architecture)
    )
    initial_tensor = parameter_tensor_blake3(model)
    identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_blake3": canonical_blake3(protocol.to_dict()),
        "bundle_id": bundle_id,
        "cache_id": sparse_cache.cache_id,
        "corpus_lock_id": sparse_cache.corpus_lock_id,
        "target_blake3": sparse_cache.target_blake3,
        "roles": list(ROLES),
        "role_arms": ROLE_ARMS,
        "model_config": model.config.to_dict(),
        "parameter_count": parameter_count(model),
        "parameter_layout_blake3": parameter_layout_blake3(model),
        "initial_parameter_tensor_blake3": initial_tensor,
        "state_contract": {
            "representation": protocol.state_contract,
            "padded_sparse_objects_per_board": BOARD_TOKEN_CAPACITY,
            "dense_hex_budget_reference": DENSE_HEX_BUDGET_REFERENCE,
            "exact_no_truncation_required": True,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "approved": True,
        "authorization_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_authorization(
    path: Path,
    *,
    cache: R2SparseMlxCache,
    bundle_id: str,
    role: str,
    model: R2SparseValueModel,
) -> dict[str, Any]:
    authorization = _read_json(path, "authorization")
    identity = authorization.get("identity")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != authorization.get("authorization_id")
    ):
        raise V1ScoreAnatomyError("V1 authorization is malformed or stale")
    protocol = V1ScoreAnatomyProtocol()
    expected = {
        "protocol_id": PROTOCOL_ID,
        "protocol_blake3": canonical_blake3(protocol.to_dict()),
        "bundle_id": bundle_id,
        "cache_id": cache.cache_id,
        "corpus_lock_id": cache.corpus_lock_id,
        "target_blake3": cache.target_blake3,
        "roles": list(ROLES),
        "role_arms": ROLE_ARMS,
        "model_config": model.config.to_dict(),
        "parameter_count": parameter_count(model),
        "parameter_layout_blake3": parameter_layout_blake3(model),
        "initial_parameter_tensor_blake3": parameter_tensor_blake3(model),
        "state_contract": {
            "representation": protocol.state_contract,
            "padded_sparse_objects_per_board": BOARD_TOKEN_CAPACITY,
            "dense_hex_budget_reference": DENSE_HEX_BUDGET_REFERENCE,
            "exact_no_truncation_required": True,
        },
    }
    if identity != expected or role not in identity["roles"]:
        raise V1ScoreAnatomyError(
            "V1 authorization does not match cache, source, initialization, or role"
        )
    return authorization


def run_experiment(config: V1ScoreAnatomyRunConfig) -> dict[str, Any]:
    """Train, evaluate, benchmark, and report one frozen comparison role."""
    config.validate()
    cache = R2SparseMlxCache(config.cache, corpus_lock=config.corpus_lock)
    arm = ROLE_ARMS[config.role]
    model_config = R2SparseMlxModelConfig(
        architecture=config.protocol.architecture
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()

    mx.random.seed(TRAINING_SEED)
    fresh_model = R2SparseValueModel(model_config)
    authorization = validate_authorization(
        config.authorization,
        cache=cache,
        bundle_id=config.bundle_id,
        role=config.role,
        model=fresh_model,
    )
    initial_tensor_blake3 = parameter_tensor_blake3(fresh_model)
    run_manifest = _run_manifest(config, cache, authorization, arm)

    if config.resume:
        validate_resume_manifest(
            config.run_dir,
            training=run_manifest["training"],
            datasets=run_manifest["datasets"],
            runtime=run_manifest["runtime"],
            source=run_manifest["source"],
        )
        model, optimizer, state, _checkpoint = (
            load_latest_checkpoint_with_factory(
                config.run_dir,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                model_factory=lambda values: R2SparseValueModel(
                    R2SparseMlxModelConfig.from_dict(values)
                ),
            )
        )
        if model.config != model_config:
            raise V1ScoreAnatomyError("resume model configuration drifted")
    else:
        if (config.run_dir / "latest.json").exists():
            raise V1ScoreAnatomyError(
                "run already contains checkpoints; pass --resume"
            )
        model = fresh_model
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    loss_and_grad = nn.value_and_grad(model, loss_for_arm(arm))
    metrics_path = config.run_dir / "metrics.jsonl"
    recent_losses: list[float] = []
    measured_examples = 0
    measured_seconds = 0.0
    invocation_started = time.perf_counter()

    model.train()
    while state.global_step < TRAINING_STEPS:
        step = state.global_step
        started = time.perf_counter()
        batch = cache.deterministic_training_batch(
            step=step,
            batch_size=BATCH_SIZE,
            seed=TRAINING_SEED,
        )
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise V1ScoreAnatomyError(
                f"training produced a nonfinite loss at step {step}"
            )
        state.global_step += 1
        state.epoch = 0
        state.batch_in_epoch = state.global_step
        state.elapsed_seconds += elapsed
        measured_examples += BATCH_SIZE
        measured_seconds += elapsed
        recent_losses.append(loss_value)

        if state.global_step % METRIC_STEPS == 0:
            event = {
                "schema_version": 1,
                "role": config.role,
                "arm": arm,
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
    model.eval()
    validation = evaluate_model(model, cache)
    performance = benchmark_model(model, cache)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol_id": PROTOCOL_ID,
        "role": config.role,
        "arm": arm,
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "bundle_id": config.bundle_id,
        },
        "protocol": config.protocol.to_dict(),
        "cache": {
            "cache_id": cache.cache_id,
            "corpus_lock_id": cache.corpus_lock_id,
            "target_blake3": cache.target_blake3,
            "padded_sparse_objects_per_board": BOARD_TOKEN_CAPACITY,
            "dense_hex_budget_reference": DENSE_HEX_BUDGET_REFERENCE,
            "exact_no_truncation_verified": True,
        },
        "model": {
            "config": model.config.to_dict(),
            "parameter_count": parameter_count(model),
            "parameter_layout_blake3": parameter_layout_blake3(model),
            "initial_parameter_tensor_blake3": initial_tensor_blake3,
            "final_parameter_tensor_blake3": parameter_tensor_blake3(model),
        },
        "optimization": {
            "global_step": state.global_step,
            "training_examples": state.global_step * BATCH_SIZE,
            "training_seconds": state.elapsed_seconds,
            "training_examples_per_second": (
                state.global_step * BATCH_SIZE
            )
            / max(state.elapsed_seconds, 1e-12),
            "invocation_wall_seconds": time.perf_counter()
            - invocation_started,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest_blake3": _checksum(
                checkpoint / "checkpoint.json"
            ),
        },
        "metrics": {
            "validation": validation,
        },
        "performance": {
            **performance,
            "training_peak_active_memory_bytes": training_peak_memory,
            "peak_process_rss_bytes": _peak_process_rss_bytes(),
        },
        "runtime": {
            **_runtime_identity(),
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "integrity": {
            "same_parameter_graph_for_both_arms": True,
            "same_initialization_for_both_arms": True,
            "identity_d6_validation": True,
            "exact_sparse_state": True,
            "all_metrics_finite": _all_finite(validation),
            "test_or_final_data_opened": False,
        },
        "claims": {
            "offline_head_factorial_complete": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = report_scientific_identity(report)
    report["report_id"] = canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def evaluate_model(
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
) -> dict[str, Any]:
    """Evaluate every open-validation row and preserve paired rank evidence."""
    component_predictions: list[np.ndarray] = []
    component_targets: list[np.ndarray] = []
    game_indices: list[np.ndarray] = []
    turns: list[np.ndarray] = []
    model.eval()
    for batch in cache.sequential_batches(
        "validation",
        EVALUATION_BATCH_SIZE,
    ):
        predictions = model.predict_components(batch)
        mx.eval(predictions)
        component_predictions.append(
            np.asarray(predictions, dtype=np.float32)
        )
        component_targets.append(
            np.asarray(batch.targets, dtype=np.float32)
        )
        game_indices.append(np.asarray(batch.game_index, dtype=np.int64))
        turns.append(np.asarray(batch.turn, dtype=np.int32))
    predictions = np.concatenate(component_predictions)
    targets = np.concatenate(component_targets)
    games = np.concatenate(game_indices)
    turn_values = np.concatenate(turns)
    if len(predictions) != cache.sample_count("validation"):
        raise V1ScoreAnatomyError("validation row count drifted")

    totals = predictions.sum(axis=1, dtype=np.float64)
    target_totals = targets.sum(axis=1, dtype=np.float64)
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    component_metrics = []
    for index, name in enumerate(COMPONENT_NAMES):
        component_metrics.append(
            {
                "name": name,
                "mae": float(np.mean(np.abs(errors[:, index]))),
                "rmse": float(np.sqrt(np.mean(np.square(errors[:, index])))),
                "bias": float(np.mean(errors[:, index])),
                "correlation": _correlation(
                    predictions[:, index],
                    targets[:, index],
                ),
            }
        )
    total_metrics = _regression_metrics(totals, target_totals)
    pairwise = _within_round_pairwise_metrics(
        totals,
        target_totals,
        games,
        turn_values,
    )
    phase_metrics = {}
    for name, lower, upper in (
        ("opening", 0, 20),
        ("early_middle", 20, 40),
        ("late_middle", 40, 60),
        ("endgame", 60, 80),
    ):
        selected = (turn_values >= lower) & (turn_values < upper)
        phase_metrics[name] = {
            "samples": int(np.sum(selected)),
            **_regression_metrics(totals[selected], target_totals[selected]),
        }
    probe_count = min(256, len(predictions))
    probe_payload = np.concatenate(
        [
            predictions[:probe_count].astype("<f4", copy=False).reshape(-1),
            targets[:probe_count].astype("<f4", copy=False).reshape(-1),
        ]
    )
    return {
        "samples": len(predictions),
        "component_catalog": list(COMPONENT_NAMES),
        "component_metrics": component_metrics,
        "mean_component_mae": float(
            np.mean([metric["mae"] for metric in component_metrics])
        ),
        "mean_wildlife_component_mae": float(
            np.mean([metric["mae"] for metric in component_metrics[5:10]])
        ),
        "total": total_metrics,
        "within_round_pairwise": pairwise,
        "phase": phase_metrics,
        "prediction_probe_rows": probe_count,
        "prediction_probe_blake3": blake3.blake3(
            probe_payload.tobytes()
        ).hexdigest(),
    }


def benchmark_model(
    model: R2SparseValueModel,
    cache: R2SparseMlxCache,
) -> dict[str, Any]:
    indices = np.arange(64, dtype=np.int64)
    batch = cache.batch(
        "validation",
        indices,
        transform_ids=np.zeros(len(indices), dtype=np.int64),
    )
    for _ in range(5):
        values = model.predict_components(batch)
        mx.eval(values)
    mx.reset_peak_memory()
    started = time.perf_counter()
    iterations = 30
    for _ in range(iterations):
        values = model.predict_components(batch)
        mx.eval(values)
    elapsed = time.perf_counter() - started
    return {
        "batch_examples": len(indices),
        "steady_iterations": iterations,
        "steady_seconds": elapsed,
        "examples_per_second": len(indices) * iterations / max(elapsed, 1e-12),
        "latency_milliseconds": elapsed / iterations * 1_000.0,
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def classify_reports(
    *,
    scalar_primary: Path,
    anatomy_primary: Path,
    scalar_replay: Path,
    anatomy_replay: Path,
) -> dict[str, Any]:
    """Apply the preregistered determinism and V1 promotion gates."""
    reports = {
        "scalar-primary": _read_json(scalar_primary, "scalar primary report"),
        "anatomy-primary": _read_json(anatomy_primary, "anatomy primary report"),
        "scalar-replay": _read_json(scalar_replay, "scalar replay report"),
        "anatomy-replay": _read_json(anatomy_replay, "anatomy replay report"),
    }
    for role, report in reports.items():
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or report.get("protocol_id") != PROTOCOL_ID
            or report.get("role") != role
            or report.get("arm") != ROLE_ARMS[role]
            or report.get("integrity", {}).get("all_metrics_finite") is not True
        ):
            raise V1ScoreAnatomyError(f"invalid or incomplete report for {role}")

    common_fields = (
        ("authorization", "authorization_id"),
        ("authorization", "bundle_id"),
        ("cache", "cache_id"),
        ("cache", "corpus_lock_id"),
        ("cache", "target_blake3"),
        ("model", "parameter_count"),
        ("model", "parameter_layout_blake3"),
        ("model", "initial_parameter_tensor_blake3"),
    )
    parity = {}
    for section, field_name in common_fields:
        values = {
            role: report[section][field_name]
            for role, report in reports.items()
        }
        parity[f"{section}.{field_name}"] = len(set(values.values())) == 1
    replay_parity = {}
    for arm, primary_role, replay_role in (
        ("scalar-total-control", "scalar-primary", "scalar-replay"),
        ("component-anatomy", "anatomy-primary", "anatomy-replay"),
    ):
        primary = reports[primary_role]
        replay = reports[replay_role]
        replay_parity[arm] = {
            "final_parameter_tensor_exact": (
                primary["model"]["final_parameter_tensor_blake3"]
                == replay["model"]["final_parameter_tensor_blake3"]
            ),
            "prediction_probe_exact": (
                primary["metrics"]["validation"]["prediction_probe_blake3"]
                == replay["metrics"]["validation"]["prediction_probe_blake3"]
            ),
            "scientific_identity_exact": (
                _role_neutral_scientific_identity(primary)
                == _role_neutral_scientific_identity(replay)
            ),
        }
    integrity_pass = all(parity.values()) and all(
        all(values.values()) for values in replay_parity.values()
    )

    scalar = reports["scalar-primary"]["metrics"]["validation"]
    anatomy = reports["anatomy-primary"]["metrics"]["validation"]
    gates = {
        "total_mae_noninferior": (
            anatomy["total"]["mae"]
            <= scalar["total"]["mae"] + MAX_TOTAL_MAE_REGRESSION
        ),
        "total_correlation_gain": (
            anatomy["total"]["correlation"]
            >= scalar["total"]["correlation"] + MIN_CORRELATION_GAIN
        ),
        "pairwise_log_loss_gain": (
            anatomy["within_round_pairwise"]["log_loss"]
            <= scalar["within_round_pairwise"]["log_loss"]
            - MIN_PAIRWISE_LOG_LOSS_GAIN
        ),
        "component_metrics_finite": _all_finite(
            anatomy["component_metrics"]
        ),
        "exact_sparse_budget": (
            reports["anatomy-primary"]["cache"][
                "padded_sparse_objects_per_board"
            ]
            <= DENSE_HEX_BUDGET_REFERENCE
        ),
    }
    promoted = integrity_pass and all(gates.values())
    classification = (
        "v1_score_anatomy_promoted_to_action_value"
        if promoted
        else "v1_score_anatomy_not_promoted"
    )
    scientific = {
        "classification": classification,
        "promoted": promoted,
        "integrity_pass": integrity_pass,
        "parity": parity,
        "replay_parity": replay_parity,
        "gates": gates,
        "deltas_anatomy_minus_scalar": {
            "total_mae": anatomy["total"]["mae"]
            - scalar["total"]["mae"],
            "total_rmse": anatomy["total"]["rmse"]
            - scalar["total"]["rmse"],
            "total_correlation": anatomy["total"]["correlation"]
            - scalar["total"]["correlation"],
            "pairwise_accuracy": (
                anatomy["within_round_pairwise"]["accuracy"]
                - scalar["within_round_pairwise"]["accuracy"]
            ),
            "pairwise_log_loss": (
                anatomy["within_round_pairwise"]["log_loss"]
                - scalar["within_round_pairwise"]["log_loss"]
            ),
        },
        "scalar": scalar,
        "anatomy": anatomy,
        "claims": {
            "gameplay_strength_measured": False,
            "action_value_successor_authorized": promoted,
            "progress_to_100_claimed": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "classification_id": canonical_blake3(scientific),
        "scientific": scientific,
    }


def report_scientific_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": report["experiment_id"],
        "adr": report["adr"],
        "protocol_id": report["protocol_id"],
        "role": report["role"],
        "arm": report["arm"],
        "authorization": report["authorization"],
        "protocol": report["protocol"],
        "cache": report["cache"],
        "model": report["model"],
        "optimization": {
            key: report["optimization"][key]
            for key in (
                "global_step",
                "training_examples",
                "checkpoint_manifest_blake3",
            )
        },
        "metrics": report["metrics"],
        "integrity": report["integrity"],
        "claims": report["claims"],
    }


def _role_neutral_scientific_identity(
    report: dict[str, Any],
) -> dict[str, Any]:
    identity = json.loads(json.dumps(report["scientific_identity"]))
    identity["role"] = ROLE_ARMS[report["role"]]
    optimization = identity.get("optimization")
    if isinstance(optimization, dict):
        # The checkpoint manifest covers state.json, whose elapsed wall time is
        # intentionally host-specific. Tensor and prediction hashes carry the
        # reproducibility claim; wall-clock metadata must not defeat parity.
        optimization.pop("checkpoint_manifest_blake3", None)
    return identity


def parameter_layout_blake3(model: R2SparseValueModel) -> str:
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(model.trainable_parameters())
    ]
    return canonical_blake3(layout)


def parameter_tensor_blake3(model: R2SparseValueModel) -> str:
    digest = blake3.blake3()
    for name, value in tree_flatten(model.trainable_parameters()):
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode()
        )
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()


def canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _run_manifest(
    config: V1ScoreAnatomyRunConfig,
    cache: R2SparseMlxCache,
    authorization: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training": {
            "experiment_id": EXPERIMENT_ID,
            "adr": ADR_ID,
            "protocol": config.protocol.to_dict(),
            "role": config.role,
            "arm": arm,
            "run_dir": str(config.run_dir.resolve()),
            "resume": config.resume,
        },
        "datasets": {
            "cache_id": cache.cache_id,
            "corpus_lock_id": cache.corpus_lock_id,
            "target_blake3": cache.target_blake3,
            "train_samples": cache.sample_count("train"),
            "validation_samples": cache.sample_count("validation"),
        },
        "runtime": _runtime_identity(),
        "source": {
            "bundle_id": config.bundle_id,
            "authorization_id": authorization["authorization_id"],
        },
    }


def _regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(predictions) == 0:
        raise V1ScoreAnatomyError("cannot score an empty metric slice")
    errors = predictions - targets
    predicted_variance = float(np.var(predictions))
    covariance = float(
        np.mean(
            (predictions - np.mean(predictions))
            * (targets - np.mean(targets))
        )
    )
    slope = covariance / predicted_variance if predicted_variance > 0 else 0.0
    intercept = float(np.mean(targets) - slope * np.mean(predictions))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "correlation": _correlation(predictions, targets),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "predicted_mean": float(np.mean(predictions)),
        "target_mean": float(np.mean(targets)),
        "predicted_stddev": float(np.std(predictions)),
        "target_stddev": float(np.std(targets)),
    }


def _within_round_pairwise_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    games: np.ndarray,
    turns: np.ndarray,
) -> dict[str, float | int]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, (game, turn) in enumerate(zip(games, turns, strict=True)):
        grouped.setdefault((int(game), int(turn) // 4), []).append(index)
    pairs = 0
    ordered_pairs = 0
    correct = 0
    log_loss = 0.0
    for indices in grouped.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                target_delta = float(targets[left] - targets[right])
                predicted_delta = float(
                    predictions[left] - predictions[right]
                )
                target_probability = 1.0 / (
                    1.0
                    + math.exp(-target_delta / PAIRWISE_TEMPERATURE)
                )
                logit = predicted_delta / PAIRWISE_TEMPERATURE
                log_loss += float(
                    np.logaddexp(0.0, logit)
                    - target_probability * logit
                )
                pairs += 1
                if target_delta != 0.0:
                    ordered_pairs += 1
                    correct += int(
                        (predicted_delta > 0.0) == (target_delta > 0.0)
                    )
    if pairs == 0:
        raise V1ScoreAnatomyError("validation produced no within-round pairs")
    return {
        "groups": len(grouped),
        "pairs": pairs,
        "ordered_pairs": ordered_pairs,
        "accuracy": correct / max(ordered_pairs, 1),
        "log_loss": log_loss / pairs,
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _runtime_identity() -> dict[str, str]:
    return {
        "mlx_version": version("mlx"),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
    }


def _peak_process_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return False


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise V1ScoreAnatomyError(
            f"{name} must be a lowercase 64-character digest"
        )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise V1ScoreAnatomyError(
            f"cannot read {description} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise V1ScoreAnatomyError(f"{description} must be a JSON object")
    return value


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
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--cache", type=Path, required=True)
    authorize.add_argument("--corpus-lock", type=Path, required=True)
    authorize.add_argument("--bundle-id", required=True)
    authorize.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--corpus-lock", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bundle-id", required=True)
    run.add_argument("--role", choices=ROLES, required=True)
    run.add_argument("--resume", action="store_true")

    classify = subparsers.add_parser("classify")
    classify.add_argument("--scalar-primary", type=Path, required=True)
    classify.add_argument("--anatomy-primary", type=Path, required=True)
    classify.add_argument("--scalar-replay", type=Path, required=True)
    classify.add_argument("--anatomy-replay", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "authorize":
        value = build_authorization(
            cache=args.cache,
            corpus_lock=args.corpus_lock,
            bundle_id=args.bundle_id,
        )
        _write_json_atomic(args.output, value)
    elif args.command == "run":
        value = run_experiment(
            V1ScoreAnatomyRunConfig(
                cache=args.cache,
                corpus_lock=args.corpus_lock,
                authorization=args.authorization,
                run_dir=args.run_dir,
                output=args.output,
                bundle_id=args.bundle_id,
                role=args.role,
                resume=args.resume,
            )
        )
    else:
        value = classify_reports(
            scalar_primary=args.scalar_primary,
            anatomy_primary=args.anatomy_primary,
            scalar_replay=args.scalar_replay,
            anatomy_replay=args.anatomy_replay,
        )
        _write_json_atomic(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
