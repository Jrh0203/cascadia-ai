"""Resumable MLX training for signed score-to-go value prediction."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
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
    load_checkpoint_pointer_with_factory,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest
from cascadia_mlx.score_to_go_dataset import (
    ScoreToGoDataset,
    randomly_rotate_score_to_go_batch,
)
from cascadia_mlx.score_to_go_model import (
    EDGE_AWARE_HEX_SCORE_TO_GO_V2,
    ENTITY_SET_SCORE_TO_GO_V1,
    PAIRWISE_TEMPERATURE,
    ScoreToGoModelConfig,
    ScoreToGoValueModel,
    score_to_go_loss,
)


@dataclass(frozen=True)
class ScoreToGoTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20260611
    checkpoint_steps: int = 500
    validation_patience: int = 0
    hex_rotation_augmentation: bool = False
    baseline_run_dir: Path | None = None
    resume: bool = False
    model: ScoreToGoModelConfig = field(default_factory=ScoreToGoModelConfig)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.validation_patience < 0:
            raise ValueError("validation_patience cannot be negative")
        if (
            self.model.architecture == EDGE_AWARE_HEX_SCORE_TO_GO_V2
            and not self.hex_rotation_augmentation
        ):
            raise ValueError("edge-aware score-to-go requires rotation augmentation")
        self.model.validate()


def train_score_to_go(config: ScoreToGoTrainingConfig) -> dict[str, Any]:
    config.validate()
    train_dataset = ScoreToGoDataset(config.train_dataset)
    validation_dataset = ScoreToGoDataset(config.validation_dataset)
    if train_dataset.split != "train":
        raise ValueError("score-to-go training dataset must use the train split")
    if validation_dataset.split != "validation":
        raise ValueError("score-to-go validation dataset must use validation")
    if train_dataset.manifest["teacher"] != validation_dataset.manifest["teacher"]:
        raise ValueError("score-to-go datasets must use the same frozen teacher")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _build_run_manifest(config, train_dataset, validation_dataset)
    baseline_validation = None
    if config.baseline_run_dir is not None:
        baseline, _, _, baseline_checkpoint = load_checkpoint_pointer_with_factory(
            config.baseline_run_dir,
            pointer="best",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            model_factory=lambda values: ScoreToGoValueModel(
                ScoreToGoModelConfig.from_dict(values)
            ),
        )
        if baseline.config.architecture != ENTITY_SET_SCORE_TO_GO_V1:
            raise ValueError("score-to-go baseline must be the frozen entity-set model")
        baseline_validation = evaluate_score_to_go(
            baseline,
            validation_dataset,
            config.batch_size,
        )
        run_manifest["baseline"] = {
            "run_dir": str(config.baseline_run_dir.resolve()),
            "checkpoint": str(baseline_checkpoint.resolve()),
            "validation": baseline_validation,
        }

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
            model_factory=lambda values: ScoreToGoValueModel(
                ScoreToGoModelConfig.from_dict(values)
            ),
        )
        if model.config != config.model:
            raise ValueError("resume score-to-go model configuration does not match")
        if config.epochs < state.epoch:
            raise ValueError("resume epoch budget is behind the checkpoint")
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError("score-to-go run already has checkpoints; pass --resume")
        mx.random.seed(config.seed)
        model = ScoreToGoValueModel(config.model)
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    loss_and_grad = nn.value_and_grad(model, score_to_go_loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    latest_validation: dict[str, Any] = {}
    started = time.perf_counter()

    for epoch in range(state.epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        trained_batches = 0
        resume_batch = state.batch_in_epoch if epoch == state.epoch else 0
        for batch_index, batch in enumerate(
            train_dataset.batches(
                config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            )
        ):
            if batch_index < resume_batch:
                continue
            if config.hex_rotation_augmentation:
                batch = randomly_rotate_score_to_go_batch(
                    batch,
                    config.seed + epoch * 1_000_000 + batch_index,
                )
            loss, gradients = loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
            trained_batches += 1
            state.global_step += 1
            state.epoch = epoch
            state.batch_in_epoch = batch_index + 1
            if state.global_step % config.checkpoint_steps == 0:
                save_checkpoint(config.run_dir, model, optimizer, state)
                prune_checkpoints(config.run_dir)

        latest_validation = evaluate_score_to_go(
            model,
            validation_dataset,
            config.batch_size,
        )
        selection_metric = latest_validation["selection_metric"]
        improved = (
            state.best_validation_loss is None or selection_metric < state.best_validation_loss
        )
        if improved:
            state.best_validation_loss = selection_metric
            state.best_validation_mae = latest_validation["final"]["total_mae"]
            state.value_epochs_without_improvement = 0
        else:
            state.value_epochs_without_improvement += 1
        state.epoch = epoch + 1
        state.batch_in_epoch = 0
        event = {
            "epoch": epoch + 1,
            "global_step": state.global_step,
            "train_loss": epoch_loss / max(trained_batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "validation": latest_validation,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
        if improved:
            set_checkpoint_pointer(
                config.run_dir,
                "best",
                checkpoint,
                {"validation": latest_validation},
            )
        prune_checkpoints(config.run_dir)
        if (
            config.validation_patience
            and state.value_epochs_without_improvement >= config.validation_patience
        ):
            break

    selected, _, selected_state, selected_checkpoint = load_checkpoint_pointer_with_factory(
        config.run_dir,
        pointer="best",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        model_factory=lambda values: ScoreToGoValueModel(ScoreToGoModelConfig.from_dict(values)),
    )
    selected_validation = evaluate_score_to_go(
        selected,
        validation_dataset,
        config.batch_size,
    )
    final_report = {
        "schema_version": 1,
        "epochs": state.epoch,
        "global_step": state.global_step,
        "best_validation_mae": state.best_validation_mae,
        "best_validation_loss": state.best_validation_loss,
        "selected_epoch": selected_state.epoch,
        "selected_checkpoint": str(selected_checkpoint.resolve()),
        "baseline_validation": baseline_validation,
        "validation": selected_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(mx.default_device()),
    }
    _write_json_atomic(config.run_dir / "final-report.json", final_report)
    return final_report


def evaluate_score_to_go(
    model: ScoreToGoValueModel,
    dataset: ScoreToGoDataset,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    count = 0
    residual = _MetricAccumulator()
    final = _MetricAccumulator()
    pairwise = _PairwiseAccumulator()
    for batch in dataset.batches(batch_size):
        predicted_residual = model.predict_components(
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
        )
        predicted_final = batch.current_targets + predicted_residual
        residual.add(predicted_residual, batch.targets)
        final.add(predicted_final, batch.final_targets)
        pairwise.add(
            predicted_final,
            batch.final_targets,
            batch.game_index,
            batch.turn,
        )
        count += predicted_residual.shape[0]
    if count == 0:
        raise ValueError("score-to-go evaluation dataset is empty")
    final_metrics = final.finish(count)
    pairwise_metrics = pairwise.finish()
    return {
        "samples": count,
        "residual": residual.finish(count),
        "final": final_metrics,
        "within_round_pairwise": pairwise_metrics,
        "selection_metric": (pairwise_metrics["log_loss"] + 0.1 * final_metrics["total_mae"]),
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self.absolute_component = mx.zeros((11,), dtype=mx.float32)
        self.squared_component = mx.zeros((11,), dtype=mx.float32)
        self.signed_component = mx.zeros((11,), dtype=mx.float32)
        self.absolute_total = mx.array(0.0)
        self.squared_total = mx.array(0.0)
        self.total_statistics = mx.zeros((5,), dtype=mx.float32)

    def add(self, predictions: mx.array, targets: mx.array) -> None:
        errors = predictions - targets
        predicted_totals = mx.sum(predictions, axis=-1)
        target_totals = mx.sum(targets, axis=-1)
        total_errors = predicted_totals - target_totals
        self.absolute_component += mx.sum(mx.abs(errors), axis=0)
        self.squared_component += mx.sum(mx.square(errors), axis=0)
        self.signed_component += mx.sum(errors, axis=0)
        self.absolute_total += mx.sum(mx.abs(total_errors))
        self.squared_total += mx.sum(mx.square(total_errors))
        self.total_statistics += mx.stack(
            [
                mx.sum(predicted_totals),
                mx.sum(target_totals),
                mx.sum(mx.square(predicted_totals)),
                mx.sum(mx.square(target_totals)),
                mx.sum(predicted_totals * target_totals),
            ]
        )

    def finish(self, count: int) -> dict[str, Any]:
        mx.eval(
            self.absolute_component,
            self.squared_component,
            self.signed_component,
            self.absolute_total,
            self.squared_total,
            self.total_statistics,
        )
        calibration = _calibration_metrics(
            count,
            [float(value) for value in self.total_statistics.tolist()],
        )
        return {
            "component_mae": (self.absolute_component / count).tolist(),
            "component_rmse": mx.sqrt(self.squared_component / count).tolist(),
            "component_bias": (self.signed_component / count).tolist(),
            "total_mae": float((self.absolute_total / count).item()),
            "total_rmse": float(mx.sqrt(self.squared_total / count).item()),
            **calibration,
        }


class _PairwiseAccumulator:
    def __init__(self) -> None:
        self.pairs = 0
        self.ordered_pairs = 0
        self.correct = 0
        self.log_loss = 0.0

    def add(
        self,
        predicted_components: mx.array,
        target_components: mx.array,
        game_index: mx.array,
        turn: mx.array,
    ) -> None:
        predicted = np.asarray(mx.sum(predicted_components, axis=-1), dtype=np.float64)
        target = np.asarray(mx.sum(target_components, axis=-1), dtype=np.float64)
        games = np.asarray(game_index, dtype=np.int64)
        rounds = np.asarray(turn, dtype=np.int64) // 4
        for left in range(len(predicted)):
            for right in range(left + 1, len(predicted)):
                if games[left] != games[right] or rounds[left] != rounds[right]:
                    continue
                target_difference = target[left] - target[right]
                predicted_difference = predicted[left] - predicted[right]
                target_probability = 1.0 / (
                    1.0 + math.exp(-target_difference / PAIRWISE_TEMPERATURE)
                )
                student_logit = predicted_difference / PAIRWISE_TEMPERATURE
                self.log_loss += (
                    np.logaddexp(0.0, student_logit) - target_probability * student_logit
                )
                self.pairs += 1
                if target_difference != 0.0:
                    self.ordered_pairs += 1
                    self.correct += int((predicted_difference > 0.0) == (target_difference > 0.0))

    def finish(self) -> dict[str, float | int]:
        if self.pairs == 0:
            raise ValueError("score-to-go evaluation produced no within-round pairs")
        return {
            "pairs": self.pairs,
            "ordered_pairs": self.ordered_pairs,
            "accuracy": self.correct / max(self.ordered_pairs, 1),
            "log_loss": self.log_loss / self.pairs,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=0)
    parser.add_argument("--hex-rotation-augmentation", action="store_true")
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--architecture",
        choices=[ENTITY_SET_SCORE_TO_GO_V1, EDGE_AWARE_HEX_SCORE_TO_GO_V2],
        default=ENTITY_SET_SCORE_TO_GO_V1,
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--graph-blocks", type=int, default=0)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_score_to_go(
        ScoreToGoTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            checkpoint_steps=args.checkpoint_steps,
            validation_patience=args.validation_patience,
            hex_rotation_augmentation=args.hex_rotation_augmentation,
            baseline_run_dir=args.baseline_run_dir,
            resume=args.resume,
            model=ScoreToGoModelConfig(
                architecture=args.architecture,
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                graph_blocks=args.graph_blocks,
                market_blocks=args.market_blocks,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _build_run_manifest(
    config: ScoreToGoTrainingConfig,
    train_dataset: ScoreToGoDataset,
    validation_dataset: ScoreToGoDataset,
) -> dict[str, Any]:
    training = asdict(config)
    training["train_dataset"] = str(config.train_dataset.resolve())
    training["validation_dataset"] = str(config.validation_dataset.resolve())
    training["run_dir"] = str(config.run_dir.resolve())
    training["baseline_run_dir"] = (
        str(config.baseline_run_dir.resolve()) if config.baseline_run_dir is not None else None
    )
    training["model"] = config.model.to_dict()
    return {
        "schema_version": 1,
        "kind": "signed-score-to-go",
        "training": training,
        "datasets": {
            "teacher": train_dataset.manifest["teacher"],
            "train_manifest_blake3": _checksum(config.train_dataset / "dataset.json"),
            "validation_manifest_blake3": _checksum(config.validation_dataset / "dataset.json"),
            "train_samples": train_dataset.sample_count,
            "validation_samples": validation_dataset.sample_count,
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


def _append_json(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _calibration_metrics(count: int, statistics: list[float]) -> dict[str, float]:
    predicted_sum, target_sum, predicted_square_sum, target_square_sum, cross_sum = statistics
    predicted_mean = predicted_sum / count
    target_mean = target_sum / count
    predicted_variance = max(predicted_square_sum / count - predicted_mean**2, 0.0)
    target_variance = max(target_square_sum / count - target_mean**2, 0.0)
    covariance = cross_sum / count - predicted_mean * target_mean
    correlation_denominator = math.sqrt(predicted_variance * target_variance)
    correlation = covariance / correlation_denominator if correlation_denominator > 0 else 0.0
    calibration_slope = covariance / predicted_variance if predicted_variance > 0 else 0.0
    calibration_intercept = target_mean - calibration_slope * predicted_mean
    return {
        "predicted_total_mean": predicted_mean,
        "target_total_mean": target_mean,
        "total_bias": predicted_mean - target_mean,
        "total_correlation": correlation,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


if __name__ == "__main__":
    main()
