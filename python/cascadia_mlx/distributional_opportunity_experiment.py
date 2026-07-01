"""Matched distributional supervision study on qualified R12 opportunities."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import shutil
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
from cascadia_mlx.counterfactual_advantage_dataset import (
    COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES,
    CounterfactualAdvantageDataset,
    decode_counterfactual_advantage_records,
)
from cascadia_mlx.distributional_opportunity_model import (
    ARMS,
    ATOM_COUNT,
    DistributionalOpportunityModelConfig,
    DistributionalOpportunityRanker,
    distributional_opportunity_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)

EXPERIMENT_ID = "v2-distributional-opportunity-supervision-v1"
ADR_ID = "0179"
PROTOCOL_ID = "matched-r12-distributional-opportunity-v1"
ROLES = (
    "c0-primary",
    "g1-primary",
    "q2-primary",
    "e3-primary",
    "c0-replay",
    "g1-replay",
    "q2-replay",
    "e3-replay",
)
ROLE_ARMS = {
    "c0-primary": ARMS[0],
    "g1-primary": ARMS[1],
    "q2-primary": ARMS[2],
    "e3-primary": ARMS[3],
    "c0-replay": ARMS[0],
    "g1-replay": ARMS[1],
    "q2-replay": ARMS[2],
    "e3-replay": ARMS[3],
}
ARM_ROLES = {
    arm: (f"{arm.split('-', 1)[0]}-primary", f"{arm.split('-', 1)[0]}-replay") for arm in ARMS
}

TRAINING_SEED = 2026061802
TRAINING_STEPS = 3_000
GROUP_BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 500
METRIC_STEPS = 100
EVALUATION_BATCH_SIZE = 32
MLX_CACHE_LIMIT_BYTES = 1_073_741_824

MAX_CENTERED_MAE_REGRESSION = 0.03
MAX_MEAN_REGRET_REGRESSION = 0.05
MAX_TOP_VALUE_RECALL_REGRESSION = 0.01
MIN_CRPS_IMPROVEMENT = 0.02
MIN_PAIRWISE_BRIER_IMPROVEMENT = 0.005
MIN_UNCERTAINTY_ERROR_CORRELATION_GAIN = 0.05
MIN_WINNER_SET_COVERAGE = 0.90
MAX_MEAN_WINNER_SET_SIZE = 3.50


class DistributionalOpportunityError(ValueError):
    """Raised when the frozen ADR 0179 contract is violated."""


@dataclass(frozen=True)
class DistributionalOpportunityProtocol:
    """Variables held fixed across all four uncertainty formulations."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    training_steps: int = TRAINING_STEPS
    group_batch_size: int = GROUP_BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE
    atom_count: int = ATOM_COUNT
    arms: tuple[str, ...] = ARMS
    target: str = "shared-seed-candidate-centered-terminal-return"
    action_objective: str = "predicted-expected-score-only"
    mean_objective: str = "huber-plus-hard-top-plus-soft-listwise"
    distribution_weight: float = 0.25
    validation_during_training: bool = False
    checkpoint_selection: str = "fixed-final-step"
    test_or_final_opened: bool = False

    def validate(self) -> None:
        if self != DistributionalOpportunityProtocol():
            raise DistributionalOpportunityError("distributional-opportunity protocol drifted")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["arms"] = list(self.arms)
        return value


@dataclass(frozen=True)
class DistributionalOpportunityRunConfig:
    train_dataset: Path
    validation_dataset: Path
    authorization: Path
    run_dir: Path
    output: Path
    bundle_id: str
    role: str
    resume: bool = False
    protocol: DistributionalOpportunityProtocol = field(
        default_factory=DistributionalOpportunityProtocol
    )

    def validate(self) -> None:
        self.protocol.validate()
        if self.role not in ROLES:
            raise DistributionalOpportunityError(
                f"unknown distributional-opportunity role: {self.role}"
            )
        _require_digest(self.bundle_id, "bundle ID")


def frozen_homoscedastic_offsets(
    dataset: CounterfactualAdvantageDataset,
) -> np.ndarray:
    """Fit one train-only residual distribution shared by every C0 candidate."""
    residuals: list[np.ndarray] = []
    for batch in dataset.batches(EVALUATION_BATCH_SIZE):
        samples = np.asarray(batch.target_centered_samples, dtype=np.float32)
        residuals.append(samples - np.mean(samples, axis=-1, keepdims=True))
    if not residuals:
        raise DistributionalOpportunityError("training dataset is empty")
    values = np.concatenate([value.reshape(-1) for value in residuals])
    levels = (np.arange(ATOM_COUNT, dtype=np.float64) + 0.5) / ATOM_COUNT
    offsets = np.quantile(values.astype(np.float64), levels, method="linear")
    offsets -= np.mean(offsets)
    result = offsets.astype(np.float32)
    if not np.all(np.diff(result) >= 0) or not np.all(np.isfinite(result)):
        raise DistributionalOpportunityError("homoscedastic residual atoms are invalid")
    return result


def target_reliability_audit(
    dataset: CounterfactualAdvantageDataset,
) -> dict[str, Any]:
    """Measure how much of each R12 target is reproducible across two R6 halves."""
    samples = _all_centered_samples(dataset)
    first = samples[..., :6]
    second = samples[..., 6:12]
    first_mean = np.mean(first, axis=-1)
    second_mean = np.mean(second, axis=-1)
    full_mean = np.mean(samples, axis=-1)
    first_quantiles = np.quantile(first, (0.1, 0.5, 0.9), axis=-1)
    second_quantiles = np.quantile(second, (0.1, 0.5, 0.9), axis=-1)
    full_best = np.max(full_mean, axis=1)
    first_choice = np.argmax(first_mean, axis=1)
    second_choice = np.argmax(second_mean, axis=1)
    group_indices = np.arange(len(samples))

    first_pairwise: list[float] = []
    second_pairwise: list[float] = []
    for left in range(samples.shape[1]):
        for right in range(left + 1, samples.shape[1]):
            first_pairwise.extend(
                np.mean(
                    (first[:, left] > first[:, right]).astype(np.float64)
                    + 0.5 * (first[:, left] == first[:, right]),
                    axis=-1,
                ).tolist()
            )
            second_pairwise.extend(
                np.mean(
                    (second[:, left] > second[:, right]).astype(np.float64)
                    + 0.5 * (second[:, left] == second[:, right]),
                    axis=-1,
                ).tolist()
            )

    return {
        "split": dataset.split,
        "dataset_id": dataset.manifest["dataset_id"],
        "groups": int(samples.shape[0]),
        "candidates": int(samples.shape[0] * samples.shape[1]),
        "samples_per_candidate": int(samples.shape[2]),
        "centered_target_min": float(np.min(samples)),
        "centered_target_max": float(np.max(samples)),
        "centered_target_stddev": float(np.std(samples)),
        "split_half": {
            "mean_correlation": _correlation(
                first_mean.reshape(-1),
                second_mean.reshape(-1),
            ),
            "stddev_correlation": _correlation(
                np.std(first, axis=-1, ddof=1).reshape(-1),
                np.std(second, axis=-1, ddof=1).reshape(-1),
            ),
            "q10_correlation": _correlation(
                first_quantiles[0].reshape(-1),
                second_quantiles[0].reshape(-1),
            ),
            "median_correlation": _correlation(
                first_quantiles[1].reshape(-1),
                second_quantiles[1].reshape(-1),
            ),
            "q90_correlation": _correlation(
                first_quantiles[2].reshape(-1),
                second_quantiles[2].reshape(-1),
            ),
            "width80_correlation": _correlation(
                (first_quantiles[2] - first_quantiles[0]).reshape(-1),
                (second_quantiles[2] - second_quantiles[0]).reshape(-1),
            ),
            "winner_agreement": float(np.mean(first_choice == second_choice)),
            "first_half_choice_full_r12_regret": float(
                np.mean(full_best - full_mean[group_indices, first_choice])
            ),
            "second_half_choice_full_r12_regret": float(
                np.mean(full_best - full_mean[group_indices, second_choice])
            ),
            "pairwise_win_probability_correlation": _correlation(
                np.asarray(first_pairwise),
                np.asarray(second_pairwise),
            ),
        },
    }


def build_authorization(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    bundle_id: str,
) -> dict[str, Any]:
    """Bind exact datasets, common graph, initialization, and train-only prior."""
    _require_digest(bundle_id, "bundle ID")
    protocol = DistributionalOpportunityProtocol()
    protocol.validate()
    train = CounterfactualAdvantageDataset(train_dataset)
    validation = CounterfactualAdvantageDataset(validation_dataset)
    if train.split != "train" or validation.split != "validation":
        raise DistributionalOpportunityError("authorization requires train and validation splits")
    offsets = frozen_homoscedastic_offsets(train)
    fingerprints = []
    for arm in ARMS:
        mx.random.seed(TRAINING_SEED)
        model = DistributionalOpportunityRanker(DistributionalOpportunityModelConfig(arm=arm))
        fingerprints.append(
            {
                "arm": arm,
                "parameter_count": parameter_count(model),
                "parameter_layout_blake3": parameter_layout_blake3(model),
                "initial_parameter_tensor_blake3": parameter_tensor_blake3(model),
            }
        )
    shared = {
        (
            value["parameter_count"],
            value["parameter_layout_blake3"],
            value["initial_parameter_tensor_blake3"],
        )
        for value in fingerprints
    }
    if len(shared) != 1:
        raise DistributionalOpportunityError(
            "matched arms do not share one graph and initialization"
        )
    parameter_count_value, layout, initial = next(iter(shared))
    target_audit = {
        "train": target_reliability_audit(train),
        "validation": target_reliability_audit(validation),
    }
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol": protocol.to_dict(),
        "protocol_blake3": canonical_blake3(protocol.to_dict()),
        "bundle_id": bundle_id,
        "roles": list(ROLES),
        "role_arms": ROLE_ARMS,
        "datasets": {
            "train": _dataset_identity(train),
            "validation": _dataset_identity(validation),
        },
        "model": {
            "architecture": (DistributionalOpportunityModelConfig().architecture),
            "parameter_count": parameter_count_value,
            "parameter_layout_blake3": layout,
            "initial_parameter_tensor_blake3": initial,
            "arm_fingerprints": fingerprints,
        },
        "homoscedastic_offsets": {
            "values": offsets.tolist(),
            "blake3": _array_blake3(offsets),
            "source_split": "train",
        },
        "target_reliability": target_audit,
        "target_reliability_blake3": canonical_blake3(target_audit),
        "claim_boundary": {
            "validation_only": True,
            "test_or_final_opened": False,
            "gameplay_strength_measured": False,
            "progress_to_100_claimed": False,
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
    train_dataset: Path,
    validation_dataset: Path,
    bundle_id: str,
    role: str,
) -> dict[str, Any]:
    authorization = _read_json(path, "authorization")
    expected = build_authorization(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        bundle_id=bundle_id,
    )
    if authorization != expected or role not in expected["identity"]["roles"]:
        raise DistributionalOpportunityError(
            "authorization does not match data, source, graph, or role"
        )
    return authorization


def verify_authorization(
    *,
    path: Path,
    train_dataset: Path,
    validation_dataset: Path,
    bundle_id: str,
    role: str,
) -> dict[str, Any]:
    """Emit a host-local receipt without creating a run or optimizer."""
    authorization = validate_authorization(
        path,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        bundle_id=bundle_id,
        role=role,
    )
    identity = authorization["identity"]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "role": role,
        "arm": ROLE_ARMS[role],
        "authorization_id": authorization["authorization_id"],
        "bundle_id": bundle_id,
        "protocol_blake3": identity["protocol_blake3"],
        "train_manifest_blake3": identity["datasets"]["train"]["manifest_blake3"],
        "validation_manifest_blake3": identity["datasets"]["validation"]["manifest_blake3"],
        "initial_parameter_tensor_blake3": identity["model"]["initial_parameter_tensor_blake3"],
        "homoscedastic_offsets_blake3": identity["homoscedastic_offsets"]["blake3"],
        "runtime": _runtime_identity(),
        "run_directory_created": False,
        "optimizer_created": False,
        "passed": True,
    }


def run_experiment(
    config: DistributionalOpportunityRunConfig,
) -> dict[str, Any]:
    """Train exactly one role, evaluate all validation rows, and seal evidence."""
    config.validate()
    train = CounterfactualAdvantageDataset(config.train_dataset)
    validation = CounterfactualAdvantageDataset(config.validation_dataset)
    authorization = validate_authorization(
        config.authorization,
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
        bundle_id=config.bundle_id,
        role=config.role,
    )
    arm = ROLE_ARMS[config.role]
    offsets_np = np.asarray(
        authorization["identity"]["homoscedastic_offsets"]["values"],
        dtype=np.float32,
    )
    offsets = mx.array(offsets_np)
    records = _materialize_records(train)
    if len(records) % GROUP_BATCH_SIZE:
        raise DistributionalOpportunityError("training groups must divide the frozen batch size")

    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(TRAINING_SEED)
    model_config = DistributionalOpportunityModelConfig(arm=arm)
    fresh_model = DistributionalOpportunityRanker(model_config)
    initial_tensor = parameter_tensor_blake3(fresh_model)
    run_manifest = _run_manifest(
        config,
        train,
        validation,
        authorization,
        model_config,
    )

    if config.resume:
        _validate_run_manifest(config.run_dir, run_manifest)
        model, optimizer, state, _checkpoint = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            model_factory=lambda values: DistributionalOpportunityRanker(
                DistributionalOpportunityModelConfig.from_dict(values)
            ),
        )
        if model.config != model_config:
            raise DistributionalOpportunityError("resume model configuration drifted")
    else:
        if (config.run_dir / "latest.json").exists():
            raise DistributionalOpportunityError("run already contains checkpoints; pass --resume")
        model = fresh_model
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    loss_and_grad = nn.value_and_grad(
        model,
        lambda candidate, batch: distributional_opportunity_loss(
            candidate,
            batch,
            homoscedastic_offsets=(offsets if arm == "c0-homoscedastic-mean" else None),
        ),
    )
    batches_per_epoch = len(records) // GROUP_BATCH_SIZE
    current_epoch = -1
    permutation: np.ndarray | None = None
    recent_losses: list[float] = []
    measured_groups = 0
    measured_seconds = 0.0
    metrics_path = config.run_dir / "metrics.jsonl"
    invocation_started = time.perf_counter()
    model.train()

    while state.global_step < TRAINING_STEPS:
        epoch = state.global_step // batches_per_epoch
        batch_in_epoch = state.global_step % batches_per_epoch
        if epoch != current_epoch:
            permutation = np.random.default_rng(TRAINING_SEED + epoch).permutation(len(records))
            current_epoch = epoch
        assert permutation is not None
        start = batch_in_epoch * GROUP_BATCH_SIZE
        batch_indices = permutation[start : start + GROUP_BATCH_SIZE]
        started = time.perf_counter()
        batch = decode_counterfactual_advantage_records(records[batch_indices])
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise DistributionalOpportunityError(f"nonfinite loss at step {state.global_step}")
        state.global_step += 1
        state.epoch = state.global_step // batches_per_epoch
        state.batch_in_epoch = state.global_step % batches_per_epoch
        state.elapsed_seconds += elapsed
        recent_losses.append(loss_value)
        measured_groups += GROUP_BATCH_SIZE
        measured_seconds += elapsed

        if state.global_step % METRIC_STEPS == 0:
            event = {
                "schema_version": 1,
                "role": config.role,
                "arm": arm,
                "global_step": state.global_step,
                "mean_loss": float(np.mean(recent_losses)),
                "groups_per_second": measured_groups / max(measured_seconds, 1e-12),
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
    final_model = config.run_dir / "final-model.safetensors"
    shutil.copy2(checkpoint / "model.safetensors", final_model)
    training_peak = int(mx.get_peak_memory())
    model.eval()
    metrics = evaluate_model(
        model,
        validation,
        homoscedastic_offsets=offsets_np,
    )
    performance = benchmark_model(
        model,
        validation,
        homoscedastic_offsets=offsets_np,
    )
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
        "datasets": authorization["identity"]["datasets"],
        "target_reliability": authorization["identity"]["target_reliability"],
        "homoscedastic_offsets": authorization["identity"]["homoscedastic_offsets"],
        "model": {
            "config": model.config.to_dict(),
            "parameter_count": parameter_count(model),
            "parameter_layout_blake3": parameter_layout_blake3(model),
            "initial_parameter_tensor_blake3": initial_tensor,
            "final_parameter_tensor_blake3": parameter_tensor_blake3(model),
            "final_model_file_blake3": _checksum(final_model),
        },
        "optimization": {
            "global_step": state.global_step,
            "training_groups": state.global_step * GROUP_BATCH_SIZE,
            "training_seconds": state.elapsed_seconds,
            "training_groups_per_second": (state.global_step * GROUP_BATCH_SIZE)
            / max(state.elapsed_seconds, 1e-12),
            "invocation_wall_seconds": time.perf_counter() - invocation_started,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        },
        "metrics": {"validation": metrics},
        "performance": {
            **performance,
            "training_peak_active_memory_bytes": training_peak,
            "peak_process_rss_bytes": _peak_process_rss_bytes(),
        },
        "runtime": {
            **_runtime_identity(),
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "integrity": {
            "same_parameter_graph_for_all_arms": True,
            "same_initialization_for_all_arms": True,
            "validation_during_training": False,
            "fixed_final_checkpoint": True,
            "expected_score_is_only_action_objective": True,
            "all_metrics_finite": _all_finite(metrics),
            "test_or_final_data_opened": False,
        },
        "claims": {
            "offline_distributional_factorial_complete": True,
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
    model: DistributionalOpportunityRanker,
    dataset: CounterfactualAdvantageDataset,
    *,
    homoscedastic_offsets: np.ndarray,
) -> dict[str, Any]:
    """Evaluate means, marginal distributions, calibration, and decisions."""
    model.eval()
    means_all: list[np.ndarray] = []
    atoms_all: list[np.ndarray] = []
    samples_all: list[np.ndarray] = []
    immediate_all: list[np.ndarray] = []
    for batch in dataset.batches(EVALUATION_BATCH_SIZE):
        means, atoms, _uncertainty = model.distribution(
            batch,
            homoscedastic_offsets=(
                mx.array(homoscedastic_offsets)
                if model.config.arm == "c0-homoscedastic-mean"
                else None
            ),
        )
        mx.eval(means, atoms)
        means_all.append(np.asarray(means, dtype=np.float32))
        atoms_all.append(np.asarray(atoms, dtype=np.float32))
        samples_all.append(np.asarray(batch.target_centered_samples, dtype=np.float32))
        immediate = np.asarray(batch.immediate_score, dtype=np.float32)
        immediate_all.append(immediate - np.mean(immediate, axis=1, keepdims=True))
    if not means_all:
        raise DistributionalOpportunityError("validation dataset is empty")
    means = np.concatenate(means_all)
    atoms = np.concatenate(atoms_all)
    samples = np.concatenate(samples_all)
    immediate = np.concatenate(immediate_all)
    targets = np.mean(samples, axis=-1)
    errors = means - targets
    decisions = _decision_metrics(means, targets)
    immediate_decisions = _decision_metrics(immediate, targets)
    pairwise = _pairwise_metrics(means, atoms, samples)
    crps = _empirical_crps(atoms, samples)
    q10 = np.quantile(atoms, 0.10, axis=-1)
    q90 = np.quantile(atoms, 0.90, axis=-1)
    width80 = q90 - q10
    target_stddev = np.std(samples, axis=-1, ddof=1)
    predicted_stddev = np.std(atoms, axis=-1, ddof=1)
    confidence = _winner_confidence_set_metrics(
        q10,
        q90,
        targets,
    )
    probe_groups = min(64, len(means))
    probe = np.concatenate(
        [
            means[:probe_groups].astype("<f4", copy=False).reshape(-1),
            atoms[:probe_groups].astype("<f4", copy=False).reshape(-1),
            targets[:probe_groups].astype("<f4", copy=False).reshape(-1),
        ]
    )
    return {
        "groups": len(means),
        "candidates": int(means.size),
        "samples": int(samples.size),
        "centered_mean_absolute_error": float(np.mean(np.abs(errors))),
        "centered_root_mean_squared_error": float(np.sqrt(np.mean(np.square(errors)))),
        "centered_advantage_correlation": _correlation(
            means.reshape(-1),
            targets.reshape(-1),
        ),
        "pairwise_accuracy": pairwise["mean_order_accuracy"],
        **decisions,
        "immediate_baseline": immediate_decisions,
        "empirical_crps": float(np.mean(crps)),
        "interval80": {
            "coverage": float(np.mean((samples >= q10[..., None]) & (samples <= q90[..., None]))),
            "mean_width": float(np.mean(width80)),
        },
        "uncertainty": {
            "absolute_mean_error_correlation": _correlation(
                width80.reshape(-1),
                np.abs(errors).reshape(-1),
            ),
            "target_stddev_correlation": _correlation(
                predicted_stddev.reshape(-1),
                target_stddev.reshape(-1),
            ),
            "stddev_mean_absolute_error": float(np.mean(np.abs(predicted_stddev - target_stddev))),
        },
        "winner_confidence_set": confidence,
        "pairwise_probability": pairwise,
        "prediction_probe_groups": probe_groups,
        "prediction_probe_blake3": blake3.blake3(probe.tobytes()).hexdigest(),
    }


def benchmark_model(
    model: DistributionalOpportunityRanker,
    dataset: CounterfactualAdvantageDataset,
    *,
    homoscedastic_offsets: np.ndarray,
) -> dict[str, Any]:
    batch = next(dataset.batches(16))
    offsets = (
        mx.array(homoscedastic_offsets) if model.config.arm == "c0-homoscedastic-mean" else None
    )
    for _ in range(5):
        values = model.distribution(
            batch,
            homoscedastic_offsets=offsets,
        )
        mx.eval(values)
    mx.reset_peak_memory()
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        values = model.distribution(
            batch,
            homoscedastic_offsets=offsets,
        )
        mx.eval(values)
    elapsed = time.perf_counter() - started
    batch_groups = int(batch.candidate_mask.shape[0])
    return {
        "batch_groups": batch_groups,
        "steady_iterations": iterations,
        "steady_seconds": elapsed,
        "groups_per_second": (batch_groups * iterations / max(elapsed, 1e-12)),
        "latency_milliseconds": elapsed / iterations * 1_000.0,
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def classify_reports(
    reports: dict[str, Path],
    *,
    models: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Require exact replays, then apply frozen uncertainty-success gates."""
    if set(reports) != set(ROLES):
        raise DistributionalOpportunityError("classification requires all eight frozen roles")
    loaded = {role: _read_json(path, f"{role} report") for role, path in reports.items()}
    for role, report in loaded.items():
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or report.get("protocol_id") != PROTOCOL_ID
            or report.get("role") != role
            or report.get("arm") != ROLE_ARMS[role]
            or report.get("report_id") != canonical_blake3(report.get("scientific_identity"))
            or report.get("integrity", {}).get("all_metrics_finite") is not True
        ):
            raise DistributionalOpportunityError(f"invalid or incomplete report for {role}")
    model_evidence = {}
    if models is not None:
        if set(models) != set(ROLES):
            raise DistributionalOpportunityError("classification model evidence is incomplete")
        for role, path in models.items():
            observed = _checksum(path)
            expected = loaded[role]["model"]["final_model_file_blake3"]
            model_evidence[role] = {
                "path": str(path),
                "observed_blake3": observed,
                "expected_blake3": expected,
                "matches_report": observed == expected,
            }

    common_fields = (
        ("authorization", "authorization_id"),
        ("authorization", "bundle_id"),
        ("datasets", "train"),
        ("datasets", "validation"),
        ("homoscedastic_offsets", "blake3"),
        ("model", "parameter_count"),
        ("model", "parameter_layout_blake3"),
        ("model", "initial_parameter_tensor_blake3"),
    )
    parity = {}
    for section, field_name in common_fields:
        values = [
            canonical_blake3(report[section][field_name])
            if isinstance(report[section][field_name], (dict, list))
            else report[section][field_name]
            for report in loaded.values()
        ]
        parity[f"{section}.{field_name}"] = len(set(values)) == 1

    replay_parity = {}
    for arm, (primary_role, replay_role) in ARM_ROLES.items():
        primary = loaded[primary_role]
        replay = loaded[replay_role]
        replay_parity[arm] = {
            "final_parameter_tensor_exact": (
                primary["model"]["final_parameter_tensor_blake3"]
                == replay["model"]["final_parameter_tensor_blake3"]
            ),
            "final_model_file_exact": (
                primary["model"]["final_model_file_blake3"]
                == replay["model"]["final_model_file_blake3"]
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
    integrity_pass = (
        all(parity.values())
        and all(all(values.values()) for values in replay_parity.values())
        and (
            not model_evidence or all(value["matches_report"] for value in model_evidence.values())
        )
    )

    control = loaded["c0-primary"]["metrics"]["validation"]
    treatments = {}
    eligible = []
    for arm in ARMS[1:]:
        role = ARM_ROLES[arm][0]
        metrics = loaded[role]["metrics"]["validation"]
        gates = {
            "centered_mae_noninferior": (
                metrics["centered_mean_absolute_error"]
                <= control["centered_mean_absolute_error"] + MAX_CENTERED_MAE_REGRESSION
            ),
            "mean_regret_noninferior": (
                metrics["mean_top_action_regret"]
                <= control["mean_top_action_regret"] + MAX_MEAN_REGRET_REGRESSION
            ),
            "top_value_recall_noninferior": (
                metrics["top_value_recall"]
                >= control["top_value_recall"] - MAX_TOP_VALUE_RECALL_REGRESSION
            ),
            "crps_improves": (
                metrics["empirical_crps"] <= control["empirical_crps"] - MIN_CRPS_IMPROVEMENT
            ),
            "pairwise_brier_improves": (
                metrics["pairwise_probability"]["brier_score"]
                <= control["pairwise_probability"]["brier_score"] - MIN_PAIRWISE_BRIER_IMPROVEMENT
            ),
            "uncertainty_tracks_error": (
                metrics["uncertainty"]["absolute_mean_error_correlation"]
                >= control["uncertainty"]["absolute_mean_error_correlation"]
                + MIN_UNCERTAINTY_ERROR_CORRELATION_GAIN
            ),
            "winner_set_coverage": (
                metrics["winner_confidence_set"]["coverage"] >= MIN_WINNER_SET_COVERAGE
            ),
            "winner_set_is_informative": (
                metrics["winner_confidence_set"]["mean_size"] <= MAX_MEAN_WINNER_SET_SIZE
            ),
        }
        arm_eligible = integrity_pass and all(gates.values())
        if arm_eligible:
            eligible.append(arm)
        treatments[arm] = {
            "eligible": arm_eligible,
            "gates": gates,
            "metrics": metrics,
            "deltas_vs_control": {
                "centered_mean_absolute_error": (
                    metrics["centered_mean_absolute_error"]
                    - control["centered_mean_absolute_error"]
                ),
                "mean_top_action_regret": (
                    metrics["mean_top_action_regret"] - control["mean_top_action_regret"]
                ),
                "top_value_recall": (metrics["top_value_recall"] - control["top_value_recall"]),
                "empirical_crps": (metrics["empirical_crps"] - control["empirical_crps"]),
                "pairwise_brier_score": (
                    metrics["pairwise_probability"]["brier_score"]
                    - control["pairwise_probability"]["brier_score"]
                ),
                "uncertainty_error_correlation": (
                    metrics["uncertainty"]["absolute_mean_error_correlation"]
                    - control["uncertainty"]["absolute_mean_error_correlation"]
                ),
            },
        }
    selected = (
        min(
            eligible,
            key=lambda arm: (
                treatments[arm]["metrics"]["empirical_crps"],
                treatments[arm]["metrics"]["pairwise_probability"]["brier_score"],
                treatments[arm]["metrics"]["centered_mean_absolute_error"],
                treatments[arm]["metrics"]["mean_top_action_regret"],
                arm,
            ),
        )
        if eligible
        else None
    )
    classification = (
        "distributional_opportunity_arm_selected"
        if selected is not None
        else "distributional_opportunity_factorial_null"
    )
    scientific = {
        "classification": classification,
        "selected_arm": selected,
        "integrity_pass": integrity_pass,
        "parity": parity,
        "replay_parity": replay_parity,
        "model_evidence": model_evidence,
        "control": control,
        "treatments": treatments,
        "target_reliability": loaded["c0-primary"]["target_reliability"],
        "claim_boundary": {
            "offline_validation_complete": True,
            "test_or_final_opened": False,
            "gameplay_strength_measured": False,
            "progress_to_100_claimed": False,
            "successor_training_authorized": selected is not None,
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
    """Exclude host timing and transport paths from scientific identity."""
    return {
        "experiment_id": report["experiment_id"],
        "adr": report["adr"],
        "protocol_id": report["protocol_id"],
        "role": report["role"],
        "arm": report["arm"],
        "authorization": report["authorization"],
        "protocol": report["protocol"],
        "datasets": report["datasets"],
        "target_reliability": report["target_reliability"],
        "homoscedastic_offsets": report["homoscedastic_offsets"],
        "model": report["model"],
        "optimization": {
            "global_step": report["optimization"]["global_step"],
            "training_groups": report["optimization"]["training_groups"],
        },
        "metrics": report["metrics"],
        "integrity": report["integrity"],
        "claims": report["claims"],
    }


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _dataset_identity(
    dataset: CounterfactualAdvantageDataset,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.manifest["dataset_id"],
        "split": dataset.split,
        "manifest_blake3": _checksum(dataset.root / "dataset.json"),
        "groups": dataset.group_count,
        "candidates": dataset.candidate_count,
        "continuations": int(dataset.manifest["total_continuations"]),
        "shard_set_blake3": canonical_blake3(
            [
                {
                    "file": entry["file"],
                    "blake3": entry["blake3"],
                    "byte_count": entry["byte_count"],
                }
                for entry in dataset.manifest["shards"]
            ]
        ),
    }


def _run_manifest(
    config: DistributionalOpportunityRunConfig,
    train: CounterfactualAdvantageDataset,
    validation: CounterfactualAdvantageDataset,
    authorization: dict[str, Any],
    model_config: DistributionalOpportunityModelConfig,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": EXPERIMENT_ID,
        "role": config.role,
        "arm": ROLE_ARMS[config.role],
        "authorization_id": authorization["authorization_id"],
        "bundle_id": config.bundle_id,
        "protocol": config.protocol.to_dict(),
        "datasets": {
            "train": _dataset_identity(train),
            "validation": _dataset_identity(validation),
        },
        "model": model_config.to_dict(),
        "runtime_contract": {
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "device": str(mx.default_device()),
        },
    }


def _validate_run_manifest(
    run_dir: Path,
    expected: dict[str, Any],
) -> None:
    observed = _read_json(run_dir / "run.json", "run manifest")
    if observed != expected:
        raise DistributionalOpportunityError("resume manifest does not match the original run")


def _materialize_records(
    dataset: CounterfactualAdvantageDataset,
) -> np.ndarray:
    return np.concatenate([np.asarray(shard.records()).copy() for shard in dataset.shards])


def _all_centered_samples(
    dataset: CounterfactualAdvantageDataset,
) -> np.ndarray:
    values = [
        np.asarray(batch.target_centered_samples, dtype=np.float32)
        for batch in dataset.batches(EVALUATION_BATCH_SIZE)
    ]
    if not values:
        raise DistributionalOpportunityError("dataset is empty")
    result = np.concatenate(values)
    if result.shape[2] != COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES:
        raise DistributionalOpportunityError("dataset is not exact R12")
    return result


def _decision_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    choices = np.argmax(scores, axis=1)
    target_best = np.max(targets, axis=1)
    chosen = targets[np.arange(len(targets)), choices]
    strict = np.argmax(targets, axis=1)
    return {
        "top_action_agreement": float(np.mean(choices == strict)),
        "top_value_recall": float(np.mean(chosen == target_best)),
        "mean_top_action_regret": float(np.mean(target_best - chosen)),
    }


def _empirical_crps(
    atoms: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    sample_distance = np.mean(
        np.abs(atoms[..., :, None] - samples[..., None, :]),
        axis=(-2, -1),
    )
    atom_distance = np.mean(
        np.abs(atoms[..., :, None] - atoms[..., None, :]),
        axis=(-2, -1),
    )
    return sample_distance - 0.5 * atom_distance


def _pairwise_metrics(
    means: np.ndarray,
    atoms: np.ndarray,
    samples: np.ndarray,
) -> dict[str, Any]:
    predicted_probabilities: list[float] = []
    outcomes: list[float] = []
    mean_order_correct = 0
    mean_order_count = 0
    for group in range(len(means)):
        for left in range(means.shape[1]):
            for right in range(left + 1, means.shape[1]):
                comparisons = (atoms[group, left, :, None] > atoms[group, right, None, :]).astype(
                    np.float64
                )
                ties = atoms[group, left, :, None] == atoms[group, right, None, :]
                probability = float(np.mean(comparisons + 0.5 * ties))
                observed = (samples[group, left] > samples[group, right]).astype(np.float64)
                observed += 0.5 * (samples[group, left] == samples[group, right])
                predicted_probabilities.extend([probability] * len(observed))
                outcomes.extend(observed.tolist())
                target_difference = float(
                    np.mean(samples[group, left]) - np.mean(samples[group, right])
                )
                if target_difference:
                    mean_order_correct += int(
                        np.sign(means[group, left] - means[group, right])
                        == np.sign(target_difference)
                    )
                    mean_order_count += 1
    probabilities = np.clip(
        np.asarray(predicted_probabilities, dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    truth = np.asarray(outcomes, dtype=np.float64)
    return {
        "comparisons": len(truth),
        "mean_order_accuracy": mean_order_correct / max(mean_order_count, 1),
        "brier_score": float(np.mean(np.square(probabilities - truth))),
        "log_loss": float(
            np.mean(-truth * np.log(probabilities) - (1.0 - truth) * np.log(1.0 - probabilities))
        ),
        "probability_outcome_correlation": _correlation(
            probabilities,
            truth,
        ),
    }


def _winner_confidence_set_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    threshold = np.max(lower, axis=1, keepdims=True)
    selected = upper >= threshold
    winners = targets == np.max(targets, axis=1, keepdims=True)
    return {
        "coverage": float(np.mean(np.any(selected & winners, axis=1))),
        "mean_size": float(np.mean(np.sum(selected, axis=1))),
        "singleton_fraction": float(np.mean(np.sum(selected, axis=1) == 1)),
    }


def _role_neutral_scientific_identity(
    report: dict[str, Any],
) -> dict[str, Any]:
    identity = json.loads(json.dumps(report["scientific_identity"]))
    identity["role"] = report["arm"]
    return identity


def _runtime_identity() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "device": str(mx.default_device()),
    }


def _peak_process_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _correlation(
    left: np.ndarray | list[float],
    right: np.ndarray | list[float],
) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if len(left_array) < 2 or float(np.std(left_array)) == 0.0 or float(np.std(right_array)) == 0.0:
        return 0.0
    value = np.corrcoef(left_array, right_array)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


def _array_blake3(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4")
    return blake3.blake3(array.tobytes()).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DistributionalOpportunityError(f"{label} is not a lowercase BLAKE3 digest")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DistributionalOpportunityError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise DistributionalOpportunityError(f"{label} must be an object")
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return False


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _parse_role_paths(values: list[str], label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role not in ROLES or not path:
            raise DistributionalOpportunityError(f"invalid {label} mapping: {value}")
        if role in result:
            raise DistributionalOpportunityError(f"duplicate {label} mapping for {role}")
        result[role] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--train-dataset", type=Path, required=True)
    authorize.add_argument("--validation-dataset", type=Path, required=True)
    authorize.add_argument("--bundle-id", required=True)
    authorize.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-authorization")
    verify.add_argument("--train-dataset", type=Path, required=True)
    verify.add_argument("--validation-dataset", type=Path, required=True)
    verify.add_argument("--authorization", type=Path, required=True)
    verify.add_argument("--bundle-id", required=True)
    verify.add_argument("--role", choices=ROLES, required=True)
    verify.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--train-dataset", type=Path, required=True)
    run.add_argument("--validation-dataset", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bundle-id", required=True)
    run.add_argument("--role", choices=ROLES, required=True)
    run.add_argument("--resume", action="store_true")

    classify = subparsers.add_parser("classify")
    classify.add_argument("--report", action="append", required=True)
    classify.add_argument("--model", action="append", default=[])
    classify.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "audit":
        report = target_reliability_audit(CounterfactualAdvantageDataset(args.dataset))
        _write_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "authorize":
        report = build_authorization(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            bundle_id=args.bundle_id,
        )
        _write_json_atomic(args.output, report)
        print(
            json.dumps(
                {
                    "authorization_id": report["authorization_id"],
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify-authorization":
        report = verify_authorization(
            path=args.authorization,
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            bundle_id=args.bundle_id,
            role=args.role,
        )
        _write_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "run":
        report = run_experiment(
            DistributionalOpportunityRunConfig(
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                authorization=args.authorization,
                run_dir=args.run_dir,
                output=args.output,
                bundle_id=args.bundle_id,
                role=args.role,
                resume=args.resume,
            )
        )
        print(
            json.dumps(
                {
                    "role": report["role"],
                    "arm": report["arm"],
                    "report_id": report["report_id"],
                    "validation": report["metrics"]["validation"],
                },
                sort_keys=True,
            )
        )
        return 0

    reports = _parse_role_paths(args.report, "report")
    models = _parse_role_paths(args.model, "model") if args.model else None
    result = classify_reports(reports, models=models)
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
