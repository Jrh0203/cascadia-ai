"""Resumable MLX rollout-return fine-tuning for the qualified sparse NNUE."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_checkpoint_pointer_with_factory,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.legacy_nnue import (
    LegacyRustExactSparseNnue,
    LegacySparseNnue,
    checksum_file,
    load_legacy_nnue_manifest,
    package_derived_legacy_nnue,
)
from cascadia_mlx.rollout_value_dataset import RolloutValueDataset
from cascadia_mlx.rollout_value_model import (
    ROLLOUT_ROOT_SELECTED_WEIGHT,
    ROLLOUT_ROOT_TEACHER_TEMPERATURE,
    ROLLOUT_ROOT_TEACHER_WEIGHT,
    ROLLOUT_VALUE_HUBER_DELTA,
    VALUE_TENSOR_NAMES,
    RolloutValueNnue,
    RolloutValueNnueConfig,
    rollout_value_loss,
)
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest

ADR65_PARENT_MANIFEST_BLAKE3 = "dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d"
ADR65_TRAIN_FIRST_GAME = 94_000
ADR65_TRAIN_GAMES = 4
ADR65_VALIDATION_FIRST_GAME = 94_000
ADR65_VALIDATION_GAMES = 2
ADR65_MIN_TRAIN_TRAJECTORIES = 100_000
ADR65_MIN_VALIDATION_TRAJECTORIES = 40_000


@dataclass(frozen=True)
class RolloutValueTrainingConfig:
    parent_model_dir: Path
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    derived_model_dir: Path
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 3e-6
    weight_decay: float = 0.0
    seed: int = 20_260_620
    checkpoint_steps: int = 500
    validation_patience: int = 4
    resume: bool = False

    def validate(self) -> None:
        if not 1 <= self.epochs <= 12:
            raise ValueError("ADR65 epochs must be between one and twelve")
        if self.batch_size != 512:
            raise ValueError("ADR65 batch size is frozen at 512")
        if self.learning_rate != 3e-6:
            raise ValueError("ADR65 learning rate is frozen at 3e-6")
        if self.weight_decay != 0.0:
            raise ValueError("ADR65 weight decay is frozen at zero")
        if self.seed != 20_260_620:
            raise ValueError("ADR65 seed is frozen at 20260620")
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.validation_patience != 4:
            raise ValueError("ADR65 validation patience is frozen at four")


def train_rollout_value(config: RolloutValueTrainingConfig) -> dict[str, Any]:
    config.validate()
    train_dataset = RolloutValueDataset(config.train_dataset)
    validation_dataset = RolloutValueDataset(config.validation_dataset)
    parent_manifest = load_legacy_nnue_manifest(config.parent_model_dir)
    _validate_adr65_inputs(config, train_dataset, validation_dataset)
    parent_identity_before = _parent_identity(config.parent_model_dir)
    parent = LegacySparseNnue.load(config.parent_model_dir)
    parent_policy = {
        "w3_policy": parent.tensors["w3_policy"],
        "b3_policy": parent.tensors["b3_policy"],
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _build_run_manifest(
        config,
        train_dataset,
        validation_dataset,
        parent_manifest,
        parent_identity_before,
    )

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
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            model_factory=lambda values: RolloutValueNnue(RolloutValueNnueConfig.from_dict(values)),
        )
        if config.epochs < state.epoch:
            raise ValueError("resume epoch budget is behind the checkpoint")
        final_path = config.run_dir / "final-report.json"
        if final_path.exists():
            return json.loads(final_path.read_text())
    else:
        if (config.run_dir / "latest.json").exists() or (config.run_dir / "run.json").exists():
            raise ValueError("ADR65 run already exists; pass --resume")
        mx.random.seed(config.seed)
        model = RolloutValueNnue.from_parent(config.parent_model_dir)
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    loss_and_grad = nn.value_and_grad(model, rollout_value_loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    started = time.perf_counter()
    stopped_early = state.value_epochs_without_improvement >= config.validation_patience
    for epoch in range(state.epoch, config.epochs) if not stopped_early else ():
        model.train()
        epoch_loss = 0.0
        trained_batches = 0
        resume_batch = state.batch_in_epoch if epoch == state.epoch else 0
        for batch_index, batch in enumerate(
            train_dataset.batches(
                config.batch_size,
                kind="trajectory",
                shuffle=True,
                seed=config.seed + epoch,
            )
        ):
            if batch_index < resume_batch:
                continue
            loss, gradients = loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise ValueError("ADR65 training produced a non-finite loss")
            epoch_loss += loss_value
            trained_batches += 1
            state.global_step += 1
            state.epoch = epoch
            state.batch_in_epoch = batch_index + 1
            state.elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if state.global_step % config.checkpoint_steps == 0:
                save_checkpoint(config.run_dir, model, optimizer, state)
                prune_checkpoints(config.run_dir)

        validation = evaluate_rollout_trajectories(
            model,
            parent_policy,
            validation_dataset,
            config.batch_size,
        )
        state.elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        improved = (
            state.best_validation_loss is None
            or validation["huber_loss"] < state.best_validation_loss
        )
        if improved:
            state.best_validation_loss = validation["huber_loss"]
            state.best_validation_rmse = validation["rmse"]
            state.value_epochs_without_improvement = 0
        else:
            state.value_epochs_without_improvement += 1
        state.epoch = epoch + 1
        state.batch_in_epoch = 0
        event = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_huber_loss": epoch_loss / max(trained_batches, 1),
            "validation": validation,
            "best_validation_loss": state.best_validation_loss,
            "epochs_without_improvement": state.value_epochs_without_improvement,
            "elapsed_seconds": state.elapsed_seconds,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
        if improved:
            set_checkpoint_pointer(
                config.run_dir,
                "best",
                checkpoint,
                {"validation": validation},
            )
        prune_checkpoints(config.run_dir)
        if state.value_epochs_without_improvement >= config.validation_patience:
            stopped_early = True
            break

    selected_model, _, selected_state, selected_checkpoint = load_checkpoint_pointer_with_factory(
        config.run_dir,
        pointer="best",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        model_factory=lambda values: RolloutValueNnue(RolloutValueNnueConfig.from_dict(values)),
    )
    parent_model = RolloutValueNnue.from_parent(config.parent_model_dir)
    parent_trajectory = evaluate_rollout_trajectories(
        parent_model,
        parent_policy,
        validation_dataset,
        config.batch_size,
    )
    selected_trajectory = evaluate_rollout_trajectories(
        selected_model,
        parent_policy,
        validation_dataset,
        config.batch_size,
    )
    parent_root = evaluate_rollout_roots(
        parent_model,
        parent_policy,
        validation_dataset,
        config.batch_size,
    )
    selected_root = evaluate_rollout_roots(
        selected_model,
        parent_policy,
        validation_dataset,
        config.batch_size,
    )
    parent_identity_after = _parent_identity(config.parent_model_dir)
    gates = _validation_gates(
        train_dataset,
        validation_dataset,
        parent_trajectory,
        selected_trajectory,
        parent_root,
        selected_root,
        parent_identity_before == parent_identity_after,
    )
    derivation = {
        "kind": "mlx-rollout-return-finetune-v1",
        "parent_manifest_blake3": parent_identity_before["manifest_blake3"],
        "parent_model_blake3": parent_identity_before["model_blake3"],
        "train_dataset_manifest_blake3": train_dataset.manifest_blake3,
        "validation_dataset_manifest_blake3": validation_dataset.manifest_blake3,
        "run_manifest_blake3": checksum_file(config.run_dir / "run.json"),
        "selected_checkpoint": selected_checkpoint.name,
        "checkpoint_manifest_blake3": checksum_file(selected_checkpoint / "checkpoint.json"),
    }
    derived_manifest = package_derived_legacy_nnue(
        config.parent_model_dir,
        config.derived_model_dir,
        _value_tensors(selected_model),
        derivation,
    )
    _verify_packaged_values(config.derived_model_dir, selected_model, parent_policy)
    state.elapsed_seconds += time.perf_counter() - started
    final_report = {
        "schema_version": 1,
        "experiment_id": "exact-mlx-rollout-return-finetune-v1-20260612",
        "status": "passed-validation" if gates["passed"] else "rejected-validation",
        "selected_checkpoint": selected_checkpoint.name,
        "selected_epoch": selected_state.epoch,
        "global_step": selected_state.global_step,
        "stopped_early": stopped_early,
        "parent_trajectory": parent_trajectory,
        "selected_trajectory": selected_trajectory,
        "parent_root": parent_root,
        "selected_root": selected_root,
        "gates": gates,
        "derived_artifact": {
            "path": str(config.derived_model_dir.resolve()),
            "manifest_blake3": checksum_file(config.derived_model_dir / "model.json"),
            "model_blake3": derived_manifest["files"]["model.safetensors"]["blake3"],
        },
        "parent_identity_before": parent_identity_before,
        "parent_identity_after": parent_identity_after,
        "elapsed_seconds": state.elapsed_seconds,
        "device": str(mx.default_device()),
    }
    _write_json_atomic(config.run_dir / "final-report.json", final_report)
    return final_report


def evaluate_rollout_trajectories(
    model: RolloutValueNnue,
    policy_tensors: dict[str, mx.array],
    dataset: RolloutValueDataset,
    batch_size: int,
) -> dict[str, Any]:
    exact = _exact_model(model, policy_tensors)
    count = 0
    absolute_error = 0.0
    squared_error = 0.0
    signed_error = 0.0
    huber = 0.0
    prediction_sum = 0.0
    target_sum = 0.0
    prediction_square_sum = 0.0
    target_square_sum = 0.0
    cross_sum = 0.0
    quartile_count = np.zeros(4, dtype=np.int64)
    quartile_squared = np.zeros(4, dtype=np.float64)
    turn_count = np.zeros(20, dtype=np.int64)
    turn_prediction_sum = np.zeros(20, dtype=np.float64)
    turn_target_sum = np.zeros(20, dtype=np.float64)
    turn_prediction_square_sum = np.zeros(20, dtype=np.float64)
    turn_target_square_sum = np.zeros(20, dtype=np.float64)
    turn_cross_sum = np.zeros(20, dtype=np.float64)
    for batch in dataset.batches(batch_size, kind="trajectory"):
        offsets, indices = batch.exact_csr()
        predictions = exact(offsets, indices)
        mx.eval(predictions)
        predicted = np.asarray(predictions, dtype=np.float32).astype(np.float64)
        targets = np.asarray(batch.target_remaining, dtype=np.float32).astype(np.float64)
        errors = predicted - targets
        if not np.all(np.isfinite(predicted)):
            raise ValueError("rollout-value evaluation produced non-finite predictions")
        count += len(errors)
        absolute_error += float(np.abs(errors).sum())
        squared_error += float(np.square(errors).sum())
        signed_error += float(errors.sum())
        absolute = np.abs(errors)
        huber += float(
            np.where(
                absolute <= ROLLOUT_VALUE_HUBER_DELTA,
                0.5 * np.square(errors),
                ROLLOUT_VALUE_HUBER_DELTA * (absolute - 0.5 * ROLLOUT_VALUE_HUBER_DELTA),
            ).sum()
        )
        prediction_sum += float(predicted.sum())
        target_sum += float(targets.sum())
        prediction_square_sum += float(np.square(predicted).sum())
        target_square_sum += float(np.square(targets).sum())
        cross_sum += float((predicted * targets).sum())
        quartiles = (batch.personal_turn.astype(np.int64) - 1) // 5
        for quartile in range(4):
            selected = quartiles == quartile
            quartile_count[quartile] += int(selected.sum())
            quartile_squared[quartile] += float(np.square(errors[selected]).sum())
        for turn in np.unique(batch.personal_turn):
            turn_index = int(turn) - 1
            selected = batch.personal_turn == turn
            selected_predictions = predicted[selected]
            selected_targets = targets[selected]
            turn_count[turn_index] += len(selected_predictions)
            turn_prediction_sum[turn_index] += float(selected_predictions.sum())
            turn_target_sum[turn_index] += float(selected_targets.sum())
            turn_prediction_square_sum[turn_index] += float(np.square(selected_predictions).sum())
            turn_target_square_sum[turn_index] += float(np.square(selected_targets).sum())
            turn_cross_sum[turn_index] += float((selected_predictions * selected_targets).sum())
    if count == 0:
        raise ValueError("rollout-value trajectory dataset is empty")
    prediction_mean = prediction_sum / count
    target_mean = target_sum / count
    prediction_variance = max(prediction_square_sum / count - prediction_mean**2, 0.0)
    target_variance = max(target_square_sum / count - target_mean**2, 0.0)
    covariance = cross_sum / count - prediction_mean * target_mean
    denominator = math.sqrt(prediction_variance * target_variance)
    residual_prediction_variance = 0.0
    residual_target_variance = 0.0
    residual_covariance = 0.0
    for turn in range(20):
        if turn_count[turn] == 0:
            continue
        residual_prediction_variance += (
            turn_prediction_square_sum[turn] - turn_prediction_sum[turn] ** 2 / turn_count[turn]
        )
        residual_target_variance += (
            turn_target_square_sum[turn] - turn_target_sum[turn] ** 2 / turn_count[turn]
        )
        residual_covariance += (
            turn_cross_sum[turn]
            - turn_prediction_sum[turn] * turn_target_sum[turn] / turn_count[turn]
        )
    residual_denominator = math.sqrt(
        max(residual_prediction_variance, 0.0) * max(residual_target_variance, 0.0)
    )
    return {
        "samples": count,
        "huber_loss": huber / count,
        "mae": absolute_error / count,
        "rmse": math.sqrt(squared_error / count),
        "bias": signed_error / count,
        "pearson": covariance / denominator if denominator > 0 else 0.0,
        "turn_residual_pearson": (
            float(residual_covariance / residual_denominator) if residual_denominator > 0 else 0.0
        ),
        "predicted_mean": prediction_mean,
        "target_mean": target_mean,
        "turn_quartile_rmse": [
            math.sqrt(quartile_squared[index] / quartile_count[index])
            if quartile_count[index]
            else 0.0
            for index in range(4)
        ],
        "turn_quartile_samples": quartile_count.tolist(),
    }


def evaluate_rollout_roots(
    model: RolloutValueNnue,
    policy_tensors: dict[str, mx.array],
    dataset: RolloutValueDataset,
    batch_size: int,
) -> dict[str, Any]:
    exact = _exact_model(model, policy_tensors)
    groups: dict[tuple[int, int], list[tuple[float, float, bool]]] = {}
    for batch in dataset.batches(batch_size, kind="root"):
        offsets, indices = batch.exact_csr()
        predictions = exact(offsets, indices)
        mx.eval(predictions)
        remaining = np.asarray(predictions, dtype=np.float32).astype(np.float64)
        immediate = np.asarray(batch.immediate_score, dtype=np.float32).astype(np.float64)
        teacher_remaining = np.asarray(batch.target_remaining, dtype=np.float32).astype(np.float64)
        if not np.all(np.isfinite(remaining)):
            raise ValueError("rollout root evaluation produced non-finite predictions")
        for row in range(batch.size):
            key = (int(batch.game_index[row]), int(batch.decision_index[row]))
            groups.setdefault(key, []).append(
                (
                    float(immediate[row] + remaining[row]),
                    float(immediate[row] + teacher_remaining[row]),
                    bool(batch.selected[row]),
                )
            )
    if not groups:
        raise ValueError("rollout-value root dataset is empty")
    pair_count = 0
    correct_pairs = 0
    top1_correct = 0
    regrets = []
    conditional_regrets = []
    centered_huber_sum = 0.0
    selected_listwise_sum = 0.0
    teacher_listwise_sum = 0.0
    candidate_count = 0
    for key, rows in groups.items():
        predicted = np.asarray([row[0] for row in rows], dtype=np.float64)
        teacher = np.asarray([row[1] for row in rows], dtype=np.float64)
        selected = np.flatnonzero([row[2] for row in rows])
        if selected.size != 1:
            raise ValueError(f"rollout root group {key} has {selected.size} selected rows")
        predicted_top = int(np.argmax(predicted))
        selected_index = int(selected[0])
        top1_correct += int(predicted_top == selected_index)
        regret = max(float(teacher[selected_index] - teacher[predicted_top]), 0.0)
        regrets.append(regret)
        if regret > 1e-9:
            conditional_regrets.append(regret)
        teacher_delta = teacher[:, None] - teacher[None, :]
        predicted_delta = predicted[:, None] - predicted[None, :]
        comparable = np.triu(np.abs(teacher_delta) > 1e-9, k=1)
        pair_count += int(comparable.sum())
        correct_pairs += int(((teacher_delta * predicted_delta > 0) & comparable).sum())
        centered_error = (predicted - predicted.mean()) - (teacher - teacher.mean())
        centered_absolute = np.abs(centered_error)
        centered_huber_sum += float(
            np.where(
                centered_absolute <= 1.0,
                0.5 * np.square(centered_error),
                centered_absolute - 0.5,
            ).sum()
        )
        candidate_count += len(rows)
        log_probabilities = _numpy_log_softmax(predicted / ROLLOUT_ROOT_TEACHER_TEMPERATURE)
        teacher_probabilities = np.exp(
            _numpy_log_softmax(teacher / ROLLOUT_ROOT_TEACHER_TEMPERATURE)
        )
        selected_listwise_sum -= float(log_probabilities[selected_index])
        teacher_listwise_sum -= float((teacher_probabilities * log_probabilities).sum())
    centered_huber = centered_huber_sum / candidate_count
    selected_listwise = selected_listwise_sum / len(groups)
    teacher_listwise = teacher_listwise_sum / len(groups)
    return {
        "groups": len(groups),
        "candidates": dataset.root_count,
        "pairwise_comparisons": pair_count,
        "pairwise_accuracy": correct_pairs / pair_count if pair_count else 0.0,
        "selected_action_top1": top1_correct / len(groups),
        "mean_regret": float(np.mean(regrets)),
        "conditional_mean_regret": (
            float(np.mean(conditional_regrets)) if conditional_regrets else 0.0
        ),
        "mistake_groups": len(conditional_regrets),
        "centered_huber_loss": centered_huber,
        "selected_listwise_loss": selected_listwise,
        "teacher_listwise_loss": teacher_listwise,
        "selection_loss": (
            centered_huber
            + ROLLOUT_ROOT_SELECTED_WEIGHT * selected_listwise
            + ROLLOUT_ROOT_TEACHER_WEIGHT * teacher_listwise
        ),
    }


def _validation_gates(
    train_dataset: RolloutValueDataset,
    validation_dataset: RolloutValueDataset,
    parent_trajectory: dict[str, Any],
    selected_trajectory: dict[str, Any],
    parent_root: dict[str, Any],
    selected_root: dict[str, Any],
    parent_unchanged: bool,
) -> dict[str, Any]:
    parent_rmse = parent_trajectory["rmse"]
    rmse_improvement = (
        (parent_rmse - selected_trajectory["rmse"]) / parent_rmse if parent_rmse else 0.0
    )
    pearson_improvement = (
        selected_trajectory["turn_residual_pearson"] - parent_trajectory["turn_residual_pearson"]
    )
    quartile_regressions = [
        selected - parent
        for parent, selected in zip(
            parent_trajectory["turn_quartile_rmse"],
            selected_trajectory["turn_quartile_rmse"],
            strict=True,
        )
    ]
    pairwise_improvement = selected_root["pairwise_accuracy"] - parent_root["pairwise_accuracy"]
    values = {
        "train_trajectory_minimum": train_dataset.trajectory_count >= ADR65_MIN_TRAIN_TRAJECTORIES,
        "validation_trajectory_minimum": validation_dataset.trajectory_count
        >= ADR65_MIN_VALIDATION_TRAJECTORIES,
        "validation_rmse_improvement": rmse_improvement >= 0.02,
        "validation_turn_residual_pearson_improvement": pearson_improvement >= 0.02,
        "turn_quartiles": all(regression <= 0.10 for regression in quartile_regressions),
        "root_pairwise_accuracy": pairwise_improvement >= 0.005,
        "root_selected_top1": selected_root["selected_action_top1"]
        >= parent_root["selected_action_top1"],
        "root_conditional_mean_regret": selected_root["conditional_mean_regret"]
        <= parent_root["conditional_mean_regret"],
        "parent_unchanged": parent_unchanged,
        "finite": _all_finite(
            parent_trajectory,
            selected_trajectory,
            parent_root,
            selected_root,
        ),
    }
    return {
        "passed": all(values.values()),
        "checks": values,
        "rmse_relative_improvement": rmse_improvement,
        "pearson_improvement": pearson_improvement,
        "turn_quartile_rmse_regressions": quartile_regressions,
        "root_pairwise_accuracy_improvement": pairwise_improvement,
    }


def _validate_adr65_inputs(
    config: RolloutValueTrainingConfig,
    train: RolloutValueDataset,
    validation: RolloutValueDataset,
) -> None:
    parent_manifest_hash = checksum_file(config.parent_model_dir / "model.json")
    if parent_manifest_hash != ADR65_PARENT_MANIFEST_BLAKE3:
        raise ValueError("ADR65 parent model manifest is not the preregistered artifact")
    if train.split != "train" or validation.split != "validation":
        raise ValueError("ADR65 datasets use incorrect splits")
    if train.manifest["teacher"] != validation.manifest["teacher"]:
        raise ValueError("ADR65 train and validation teachers differ")
    teacher = train.manifest["teacher"]
    if (
        teacher["rollouts"] != 600
        or teacher["trace_modulus"] != 8
        or teacher["parent_model_manifest_blake3"] != ADR65_PARENT_MANIFEST_BLAKE3
    ):
        raise ValueError("ADR65 teacher protocol differs from preregistration")
    if (
        train.manifest["first_game_index"] != ADR65_TRAIN_FIRST_GAME
        or train.manifest["completed_games"] != ADR65_TRAIN_GAMES
        or validation.manifest["first_game_index"] != ADR65_VALIDATION_FIRST_GAME
        or validation.manifest["completed_games"] != ADR65_VALIDATION_GAMES
    ):
        raise ValueError("ADR65 dataset game ranges differ from preregistration")
    if (
        train.trajectory_count < ADR65_MIN_TRAIN_TRAJECTORIES
        or validation.trajectory_count < ADR65_MIN_VALIDATION_TRAJECTORIES
    ):
        raise ValueError("ADR65 datasets do not meet preregistered trajectory minimums")


def _build_run_manifest(
    config: RolloutValueTrainingConfig,
    train: RolloutValueDataset,
    validation: RolloutValueDataset,
    parent_manifest: dict[str, Any],
    parent_identity: dict[str, str],
) -> dict[str, Any]:
    training = asdict(config)
    for field in (
        "parent_model_dir",
        "train_dataset",
        "validation_dataset",
        "run_dir",
        "derived_model_dir",
    ):
        training[field] = str(getattr(config, field).resolve())
    training["huber_delta"] = ROLLOUT_VALUE_HUBER_DELTA
    return {
        "schema_version": 1,
        "kind": "mlx-rollout-return-finetune-v1",
        "training": training,
        "datasets": {
            "teacher": train.manifest["teacher"],
            "parent_manifest_blake3": parent_identity["manifest_blake3"],
            "parent_model_blake3": parent_identity["model_blake3"],
            "train_manifest_blake3": train.manifest_blake3,
            "validation_manifest_blake3": validation.manifest_blake3,
            "train_trajectory_records": train.trajectory_count,
            "validation_trajectory_records": validation.trajectory_count,
            "validation_root_records": validation.root_count,
        },
        "parent": {
            "manifest": parent_manifest,
            **parent_identity,
        },
        "runtime": {
            "mlx_version": version("mlx"),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "device": str(mx.default_device()),
        },
        "source": source_provenance(Path(__file__).resolve().parents[2]),
    }


def _exact_model(
    model: RolloutValueNnue,
    policy_tensors: dict[str, mx.array],
) -> LegacyRustExactSparseNnue:
    return LegacyRustExactSparseNnue({**_value_tensors(model), **policy_tensors})


def _value_tensors(model: RolloutValueNnue) -> dict[str, mx.array]:
    return {name: getattr(model, name) for name in VALUE_TENSOR_NAMES}


def _parent_identity(parent_model_dir: Path) -> dict[str, str]:
    return {
        "manifest_blake3": checksum_file(parent_model_dir / "model.json"),
        "model_blake3": checksum_file(parent_model_dir / "model.safetensors"),
    }


def _verify_packaged_values(
    derived_model_dir: Path,
    selected_model: RolloutValueNnue,
    parent_policy: dict[str, mx.array],
) -> None:
    packaged = LegacySparseNnue.load(derived_model_dir)
    for name, expected in {**_value_tensors(selected_model), **parent_policy}.items():
        mx.eval(expected, packaged.tensors[name])
        if not np.array_equal(np.asarray(expected), np.asarray(packaged.tensors[name])):
            raise ValueError(f"packaged rollout-value tensor {name} changed")


def _all_finite(*values: object) -> bool:
    def visit(value: object) -> bool:
        if isinstance(value, dict):
            return all(visit(item) for item in value.values())
        if isinstance(value, list):
            return all(visit(item) for item in value)
        if isinstance(value, (int, bool)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        return False

    return all(visit(value) for value in values)


def _numpy_log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def _append_json(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--derived-model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20_260_620)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_rollout_value(RolloutValueTrainingConfig(**vars(args)))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
