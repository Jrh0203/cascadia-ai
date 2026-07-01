"""Resumable MLX training for public beam-state continuation values."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.public_beam_value_dataset import PublicBeamValueDataset
from cascadia_mlx.public_beam_value_model import (
    PublicBeamValueModel,
    PublicBeamValueModelConfig,
    public_beam_value_loss,
    public_beam_value_scores,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class PublicBeamValueTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = 20
    group_batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260612
    checkpoint_steps: int = 100
    validation_patience: int = 5
    resume: bool = False
    model: PublicBeamValueModelConfig = field(default_factory=PublicBeamValueModelConfig)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.validation_patience <= 0:
            raise ValueError("validation_patience must be positive")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("public beam-value v1 uses exactly one train and validation dataset")
        if self.init_model_dir is not None:
            raise ValueError("public beam-value v1 does not support warm starts")
        self.model.validate()


def public_beam_value_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="public-beam-value",
        dataset_factory=PublicBeamValueDataset,
        model_factory=lambda values: PublicBeamValueModel(
            PublicBeamValueModelConfig.from_dict(values)
        ),
        new_model=PublicBeamValueModel,
        load_promoted=_unsupported_warm_start,
        loss=public_beam_value_loss,
        score_batch=public_beam_value_scores,
        evaluate=evaluate_public_beam_value,
        selection_metric="validation_objective",
        accuracy_metric="top_action_agreement",
    )


def train_public_beam_value(config: PublicBeamValueTrainingConfig) -> dict[str, Any]:
    return train_ranking(config, adapter=public_beam_value_adapter())


def evaluate_public_beam_value(
    model: PublicBeamValueModel,
    dataset: PublicBeamValueDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    groups = 0
    candidates = 0
    squared_error = 0.0
    absolute_error = 0.0
    centered_squared_error = 0.0
    centered_absolute_error = 0.0
    agreements = 0
    regret = 0.0
    predictions_all: list[float] = []
    targets_all: list[float] = []
    centered_predictions_all: list[float] = []
    centered_targets_all: list[float] = []

    for batch in dataset.batches(group_batch_size):
        predictions = public_beam_value_scores(model, batch)
        mx.eval(predictions)
        prediction_values = np.asarray(predictions)
        target_values = np.asarray(batch.target_mean)
        masks = np.asarray(batch.candidate_mask)
        for prediction, target, mask in zip(
            prediction_values,
            target_values,
            masks,
            strict=True,
        ):
            prediction = prediction[mask]
            target = target[mask]
            error = prediction - target
            centered_prediction = prediction - np.mean(prediction)
            centered_target = target - np.mean(target)
            centered_error = centered_prediction - centered_target
            groups += 1
            candidates += len(target)
            squared_error += float(np.sum(error**2))
            absolute_error += float(np.sum(np.abs(error)))
            centered_squared_error += float(np.sum(centered_error**2))
            centered_absolute_error += float(np.sum(np.abs(centered_error)))
            predicted_top = int(np.argmax(prediction))
            target_top = int(np.argmax(target))
            agreements += int(predicted_top == target_top)
            regret += float(target[target_top] - target[predicted_top])
            predictions_all.extend(prediction.tolist())
            targets_all.extend(target.tolist())
            centered_predictions_all.extend(centered_prediction.tolist())
            centered_targets_all.extend(centered_target.tolist())

    if groups == 0 or candidates == 0:
        raise ValueError("public beam-value evaluation dataset is empty")
    mean_squared_error = squared_error / candidates
    centered_mean_squared_error = centered_squared_error / candidates
    return {
        "groups": groups,
        "candidates": candidates,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": mean_squared_error**0.5,
        "mean_absolute_error": absolute_error / candidates,
        "value_correlation": _correlation(predictions_all, targets_all),
        "centered_mean_squared_error": centered_mean_squared_error,
        "centered_root_mean_squared_error": centered_mean_squared_error**0.5,
        "centered_mean_absolute_error": centered_absolute_error / candidates,
        "centered_advantage_correlation": _correlation(
            centered_predictions_all,
            centered_targets_all,
        ),
        "top_action_agreement": agreements / groups,
        "mean_top_action_regret": regret / groups,
        "validation_objective": centered_mean_squared_error + 0.1 * mean_squared_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_public_beam_value(
        PublicBeamValueTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            run_dir=args.run_dir,
            epochs=args.epochs,
            group_batch_size=args.group_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            checkpoint_steps=args.checkpoint_steps,
            validation_patience=args.validation_patience,
            resume=args.resume,
            model=PublicBeamValueModelConfig(
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                market_blocks=args.market_blocks,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _unsupported_warm_start(_model_dir: Path) -> PublicBeamValueModel:
    raise ValueError("public beam-value v1 does not support warm starts")


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


if __name__ == "__main__":
    main()
