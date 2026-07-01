"""Resumable MLX training for joint public beam candidate-set ranking."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.public_beam_set_model import (
    PublicBeamSetModelConfig,
    PublicBeamSetRanker,
    public_beam_set_loss,
    public_beam_set_scores,
)
from cascadia_mlx.public_beam_value_dataset import PublicBeamValueDataset
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class PublicBeamSetTrainingConfig:
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
    seed: int = 20260613
    checkpoint_steps: int = 100
    validation_patience: int = 5
    resume: bool = False
    model: PublicBeamSetModelConfig = field(default_factory=PublicBeamSetModelConfig)

    def validate(self) -> None:
        if self.epochs <= 0 or self.group_batch_size <= 0:
            raise ValueError("epochs and group_batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.checkpoint_steps <= 0 or self.validation_patience <= 0:
            raise ValueError("checkpoint_steps and validation_patience must be positive")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("public beam set ranker v1 uses one train and validation dataset")
        if self.init_model_dir is not None:
            raise ValueError("public beam set ranker v1 does not support warm starts")
        self.model.validate()


def public_beam_set_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="public-beam-set-ranking",
        dataset_factory=PublicBeamValueDataset,
        model_factory=lambda values: PublicBeamSetRanker(
            PublicBeamSetModelConfig.from_dict(values)
        ),
        new_model=PublicBeamSetRanker,
        load_promoted=_unsupported_warm_start,
        loss=public_beam_set_loss,
        score_batch=public_beam_set_scores,
        evaluate=evaluate_public_beam_set,
        selection_metric="decision_objective",
        accuracy_metric="top_value_recall",
    )


def train_public_beam_set(config: PublicBeamSetTrainingConfig) -> dict[str, Any]:
    return train_ranking(config, adapter=public_beam_set_adapter())


def evaluate_public_beam_set(
    model: PublicBeamSetRanker,
    dataset: PublicBeamValueDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    groups = 0
    candidates = 0
    centered_squared_error = 0.0
    centered_absolute_error = 0.0
    exact_agreements = 0
    top_value_recalls = 0
    regret = 0.0
    immediate_top_value_recalls = 0
    immediate_regret = 0.0
    centered_predictions_all: list[float] = []
    centered_targets_all: list[float] = []

    for batch in dataset.batches(group_batch_size):
        predictions = public_beam_set_scores(model, batch)
        mx.eval(predictions)
        for prediction, target, immediate, mask in zip(
            np.asarray(predictions),
            np.asarray(batch.target_mean),
            np.asarray(batch.immediate_score),
            np.asarray(batch.candidate_mask),
            strict=True,
        ):
            prediction = prediction[mask]
            target = target[mask]
            immediate = immediate[mask]
            centered_prediction = prediction - np.mean(prediction)
            centered_target = target - np.mean(target)
            centered_error = centered_prediction - centered_target
            target_top = int(np.argmax(target))
            predicted_top = int(np.argmax(prediction))
            immediate_top = int(np.argmax(immediate))
            best_value = float(target[target_top])
            groups += 1
            candidates += len(target)
            centered_squared_error += float(np.sum(centered_error**2))
            centered_absolute_error += float(np.sum(np.abs(centered_error)))
            exact_agreements += int(predicted_top == target_top)
            top_value_recalls += int(float(target[predicted_top]) == best_value)
            regret += best_value - float(target[predicted_top])
            immediate_top_value_recalls += int(float(target[immediate_top]) == best_value)
            immediate_regret += best_value - float(target[immediate_top])
            centered_predictions_all.extend(centered_prediction.tolist())
            centered_targets_all.extend(centered_target.tolist())

    if groups == 0 or candidates == 0:
        raise ValueError("public beam set evaluation dataset is empty")
    centered_mse = centered_squared_error / candidates
    top_value_recall = top_value_recalls / groups
    mean_regret = regret / groups
    return {
        "groups": groups,
        "candidates": candidates,
        "centered_mean_squared_error": centered_mse,
        "centered_root_mean_squared_error": centered_mse**0.5,
        "centered_mean_absolute_error": centered_absolute_error / candidates,
        "centered_advantage_correlation": _correlation(
            centered_predictions_all,
            centered_targets_all,
        ),
        "top_action_agreement": exact_agreements / groups,
        "top_value_recall": top_value_recall,
        "mean_top_action_regret": mean_regret,
        "immediate_top_value_recall": immediate_top_value_recalls / groups,
        "immediate_mean_top_action_regret": immediate_regret / groups,
        "decision_objective": mean_regret + 0.25 * (1.0 - top_value_recall) + 0.10 * centered_mse,
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
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_public_beam_set(
        PublicBeamSetTrainingConfig(
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
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _unsupported_warm_start(_model_dir: Path) -> PublicBeamSetRanker:
    raise ValueError("public beam set ranker v1 does not support warm starts")


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


if __name__ == "__main__":
    main()
