"""Resumable MLX training for the ADR 0078 R12 set ranker."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.counterfactual_advantage_dataset import (
    CounterfactualAdvantageDataset,
)
from cascadia_mlx.counterfactual_advantage_model import (
    CounterfactualAdvantageModelConfig,
    CounterfactualAdvantageRanker,
    counterfactual_advantage_loss,
    counterfactual_advantage_scores,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class CounterfactualAdvantageTrainingConfig:
    """Frozen optimization contract for ADR 0078."""

    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = 20
    group_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260614
    checkpoint_steps: int = 100
    validation_patience: int = 5
    resume: bool = False
    model: CounterfactualAdvantageModelConfig = field(
        default_factory=CounterfactualAdvantageModelConfig
    )

    def validate(self) -> None:
        if self.epochs <= 0 or self.group_batch_size <= 0:
            raise ValueError("epochs and group_batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.checkpoint_steps <= 0 or self.validation_patience <= 0:
            raise ValueError("checkpoint_steps and validation_patience must be positive")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("ADR 0078 uses exactly one train and validation dataset")
        if self.init_model_dir is not None:
            raise ValueError("ADR 0078 does not support warm starts")
        self.model.validate()


def counterfactual_advantage_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="r12-counterfactual-advantage-set-ranking",
        dataset_factory=CounterfactualAdvantageDataset,
        model_factory=lambda values: CounterfactualAdvantageRanker(
            CounterfactualAdvantageModelConfig.from_dict(values)
        ),
        new_model=CounterfactualAdvantageRanker,
        load_promoted=_unsupported_warm_start,
        loss=counterfactual_advantage_loss,
        score_batch=counterfactual_advantage_scores,
        evaluate=evaluate_counterfactual_advantage,
        selection_metric="decision_objective",
        accuracy_metric="top_value_recall",
    )


def train_counterfactual_advantage(
    config: CounterfactualAdvantageTrainingConfig,
) -> dict[str, Any]:
    return train_ranking(config, adapter=counterfactual_advantage_adapter())


def evaluate_counterfactual_advantage(
    model: CounterfactualAdvantageRanker,
    dataset: CounterfactualAdvantageDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    """Measure point fidelity and decisions against immediate and H6 baselines."""
    model.eval()
    groups = 0
    candidates = 0
    centered_squared_error = 0.0
    centered_absolute_error = 0.0
    centered_predictions_all: list[float] = []
    centered_targets_all: list[float] = []
    pairwise_correct = 0
    pairwise_count = 0
    model_decisions = _DecisionMetrics()
    immediate_decisions = _DecisionMetrics()
    shallow_decisions = _DecisionMetrics()
    selected_decisions = _DecisionMetrics()
    standard_errors: list[float] = []

    for batch in dataset.batches(group_batch_size):
        predictions = counterfactual_advantage_scores(model, batch)
        mx.eval(predictions)
        for prediction, target, immediate, shallow, selected, mask, standard_error in zip(
            np.asarray(predictions),
            np.asarray(batch.target_mean),
            np.asarray(batch.immediate_score),
            np.asarray(batch.shallow_mean),
            np.asarray(batch.selected_index),
            np.asarray(batch.candidate_mask),
            np.asarray(batch.target_standard_error),
            strict=True,
        ):
            prediction = prediction[mask]
            target = target[mask]
            immediate = immediate[mask]
            shallow = shallow[mask]
            standard_error = standard_error[mask]
            centered_prediction = prediction - np.mean(prediction)
            centered_target = target - np.mean(target)
            centered_error = centered_prediction - centered_target
            groups += 1
            candidates += len(target)
            centered_squared_error += float(np.sum(centered_error**2))
            centered_absolute_error += float(np.sum(np.abs(centered_error)))
            centered_predictions_all.extend(centered_prediction.tolist())
            centered_targets_all.extend(centered_target.tolist())
            standard_errors.extend(standard_error.tolist())
            model_decisions.add(prediction, target)
            immediate_decisions.add(immediate, target)
            shallow_decisions.add(shallow, target)
            selected_decisions.add_choice(int(selected), target)
            for left in range(len(target)):
                for right in range(left + 1, len(target)):
                    target_difference = float(target[left] - target[right])
                    if target_difference == 0.0:
                        continue
                    pairwise_correct += int(
                        np.sign(float(prediction[left] - prediction[right]))
                        == np.sign(target_difference)
                    )
                    pairwise_count += 1

    if groups == 0 or candidates == 0:
        raise ValueError("counterfactual-advantage evaluation dataset is empty")
    centered_mse = centered_squared_error / candidates
    model_metrics = model_decisions.to_dict(groups)
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
        "pairwise_accuracy": pairwise_correct / max(pairwise_count, 1),
        **model_metrics,
        "immediate_baseline": immediate_decisions.to_dict(groups),
        "shallow_argmax_baseline": shallow_decisions.to_dict(groups),
        "h6_selected_baseline": selected_decisions.to_dict(groups),
        "target_standard_error_mean": float(np.mean(standard_errors)),
        "target_standard_error_p90": float(np.percentile(standard_errors, 90)),
        "decision_objective": (
            model_metrics["mean_top_action_regret"]
            + 0.25 * (1.0 - model_metrics["top_value_recall"])
            + 0.10 * centered_mse
        ),
    }


@dataclass
class _DecisionMetrics:
    strict_agreements: int = 0
    value_recalls: int = 0
    regret: float = 0.0

    def add(self, scores: np.ndarray, target: np.ndarray) -> None:
        self.add_choice(int(np.argmax(scores)), target)

    def add_choice(self, choice: int, target: np.ndarray) -> None:
        target_top = int(np.argmax(target))
        best_value = float(target[target_top])
        chosen_value = float(target[choice])
        self.strict_agreements += int(choice == target_top)
        self.value_recalls += int(chosen_value == best_value)
        self.regret += best_value - chosen_value

    def to_dict(self, groups: int) -> dict[str, float]:
        return {
            "top_action_agreement": self.strict_agreements / groups,
            "top_value_recall": self.value_recalls / groups,
            "mean_top_action_regret": self.regret / groups,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_counterfactual_advantage(
        CounterfactualAdvantageTrainingConfig(
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


def _unsupported_warm_start(_model_dir: Path) -> CounterfactualAdvantageRanker:
    raise ValueError("ADR 0078 does not support warm starts")


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


if __name__ == "__main__":
    main()
