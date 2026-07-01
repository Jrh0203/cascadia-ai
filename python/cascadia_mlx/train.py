"""Resumable local MLX training for the v2 decomposed value model."""

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

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.dataset import Dataset
from cascadia_mlx.model import EntitySetValueModel, ModelConfig, value_loss
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible training inputs and hyperparameters."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20260610
    checkpoint_steps: int = 500
    resume: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)

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
        self.model.validate()


def train(config: TrainingConfig) -> dict[str, Any]:
    """Train or resume and return the final validation report."""
    config.validate()
    train_dataset = Dataset(config.train_dataset)
    validation_dataset = Dataset(config.validation_dataset)
    if train_dataset.split != "train":
        raise ValueError("training dataset must use the train split")
    if validation_dataset.split not in {"validation", "test"}:
        raise ValueError("validation dataset must use validation or test split")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _build_run_manifest(config, train_dataset, validation_dataset)

    if config.resume:
        validate_resume_manifest(
            config.run_dir,
            training=run_manifest["training"],
            datasets=run_manifest["datasets"],
            runtime=run_manifest["runtime"],
            source=run_manifest["source"],
        )
        model, optimizer, state, _ = load_latest_checkpoint(
            config.run_dir,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if model.config != config.model:
            raise ValueError("resume model configuration does not match")
        if config.epochs < state.epoch:
            raise ValueError("resume epoch budget is behind the checkpoint")
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError("run already has checkpoints; pass --resume")
        mx.random.seed(config.seed)
        model = EntitySetValueModel(config.model)
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    loss_and_grad = nn.value_and_grad(model, value_loss)
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

        latest_validation = evaluate(model, validation_dataset, config.batch_size)
        improved = (
            state.best_validation_mae is None
            or latest_validation["total_mae"] < state.best_validation_mae
        )
        if improved:
            state.best_validation_mae = latest_validation["total_mae"]
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

    final_report = {
        "schema_version": 1,
        "epochs": state.epoch,
        "global_step": state.global_step,
        "best_validation_mae": state.best_validation_mae,
        "validation": latest_validation or evaluate(model, validation_dataset, config.batch_size),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(mx.default_device()),
    }
    _write_json_atomic(config.run_dir / "final-report.json", final_report)
    return final_report


def evaluate(
    model: EntitySetValueModel,
    dataset: Dataset,
    batch_size: int,
) -> dict[str, Any]:
    """Compute held-out component and total MAE/RMSE."""
    model.eval()
    count = 0
    absolute_component = mx.zeros((11,), dtype=mx.float32)
    squared_component = mx.zeros((11,), dtype=mx.float32)
    signed_component = mx.zeros((11,), dtype=mx.float32)
    absolute_total = mx.array(0.0)
    squared_total = mx.array(0.0)
    total_statistics = mx.zeros((5,), dtype=mx.float32)
    for batch in dataset.batches(batch_size):
        predictions = model.predict_components(
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
        )
        errors = predictions - batch.targets
        batch_count = errors.shape[0]
        absolute_component += mx.sum(mx.abs(errors), axis=0)
        squared_component += mx.sum(mx.square(errors), axis=0)
        signed_component += mx.sum(errors, axis=0)
        predicted_totals = mx.sum(predictions, axis=-1)
        target_totals = mx.sum(batch.targets, axis=-1)
        total_errors = predicted_totals - target_totals
        absolute_total += mx.sum(mx.abs(total_errors))
        squared_total += mx.sum(mx.square(total_errors))
        total_statistics += mx.stack(
            [
                mx.sum(predicted_totals),
                mx.sum(target_totals),
                mx.sum(mx.square(predicted_totals)),
                mx.sum(mx.square(target_totals)),
                mx.sum(predicted_totals * target_totals),
            ]
        )
        count += batch_count
    if count == 0:
        raise ValueError("evaluation dataset is empty")
    mx.eval(
        absolute_component,
        squared_component,
        signed_component,
        absolute_total,
        squared_total,
        total_statistics,
    )
    calibration = _calibration_metrics(count, [float(value) for value in total_statistics.tolist()])
    return {
        "samples": count,
        "component_mae": (absolute_component / count).tolist(),
        "component_rmse": mx.sqrt(squared_component / count).tolist(),
        "component_bias": (signed_component / count).tolist(),
        "total_mae": float((absolute_total / count).item()),
        "total_rmse": float(mx.sqrt(squared_total / count).item()),
        **calibration,
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train(
        TrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            checkpoint_steps=args.checkpoint_steps,
            resume=args.resume,
            model=ModelConfig(
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                market_blocks=args.market_blocks,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _build_run_manifest(
    config: TrainingConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training": {
            **asdict(config),
            "train_dataset": str(config.train_dataset.resolve()),
            "validation_dataset": str(config.validation_dataset.resolve()),
            "run_dir": str(config.run_dir.resolve()),
            "model": config.model.to_dict(),
        },
        "datasets": {
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
