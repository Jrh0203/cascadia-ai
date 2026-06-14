"""Resumable MLX training for balanced conservative groupwise policy learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.conservative_advantage_dataset import ConservativeAdvantageDataset
from cascadia_mlx.conservative_policy_model import (
    ConservativePolicyModel,
    ConservativePolicyModelConfig,
    conservative_policy_loss,
    conservative_policy_outputs,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class ConservativePolicyTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = 20
    group_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260611
    checkpoint_steps: int = 200
    validation_patience: int = 5
    resume: bool = False
    model: ConservativePolicyModelConfig = field(default_factory=ConservativePolicyModelConfig)

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
        self.model.validate()


def conservative_policy_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="conservative-policy",
        dataset_factory=ConservativeAdvantageDataset,
        model_factory=lambda values: ConservativePolicyModel(
            ConservativePolicyModelConfig.from_dict(values)
        ),
        new_model=ConservativePolicyModel,
        load_promoted=_unsupported_warm_start,
        loss=conservative_policy_loss,
        score_batch=lambda model, batch: conservative_policy_outputs(model, batch)[1],
        evaluate=evaluate_conservative_policy,
        selection_metric="balanced_policy_cross_entropy",
        accuracy_metric="exact_policy_agreement",
    )


def train_conservative_policy(config: ConservativePolicyTrainingConfig) -> dict[str, Any]:
    return train_ranking(config, adapter=conservative_policy_adapter())


def evaluate_conservative_policy(
    model: ConservativePolicyModel,
    dataset: ConservativeAdvantageDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    group_count = 0
    candidate_count = 0
    squared_error = 0.0
    absolute_error = 0.0
    zero_squared_error = 0.0
    exact_agreement = 0
    total_regret = 0.0
    total_policy_cross_entropy = 0.0
    anchor_policy_cross_entropy = 0.0
    challenger_policy_cross_entropy = 0.0
    anchor_groups = 0
    anchor_false_positives = 0
    challenger_groups = 0
    challenger_exact_recall = 0
    targets_all: list[float] = []
    predictions_all: list[float] = []

    for batch in dataset.batches(group_batch_size):
        lower_bounds, policy_logits = conservative_policy_outputs(model, batch)
        mx.eval(lower_bounds, policy_logits)
        predicted_values = np.asarray(lower_bounds)
        policy_values = np.asarray(policy_logits)
        targets = np.asarray(batch.lower_bound)
        masks = np.asarray(batch.candidate_mask)
        selected = np.asarray(batch.selected)
        for prediction, policy, target, mask, selected_mask in zip(
            predicted_values,
            policy_values,
            targets,
            masks,
            selected,
            strict=True,
        ):
            prediction = prediction[mask]
            policy = policy[mask]
            target = target[mask]
            selected_mask = selected_mask[mask]
            count = len(target)
            candidate_count += count
            group_count += 1
            error = prediction - target
            squared_error += float(np.sum(error**2))
            absolute_error += float(np.sum(np.abs(error)))
            zero_squared_error += float(np.sum(target**2))
            targets_all.extend(target.tolist())
            predictions_all.extend(prediction.tolist())

            teacher_indices = np.flatnonzero(selected_mask)
            teacher_choice = int(teacher_indices[0]) if len(teacher_indices) else -1
            predicted_index = int(np.argmax(policy))
            predicted_choice = predicted_index if float(policy[predicted_index]) > 0.0 else -1
            exact_agreement += int(predicted_choice == teacher_choice)

            teacher_utility = float(target[teacher_choice]) if teacher_choice >= 0 else 0.0
            predicted_utility = float(target[predicted_choice]) if predicted_choice >= 0 else 0.0
            total_regret += max(0.0, teacher_utility - predicted_utility)

            action_logits = np.concatenate(([0.0], policy.astype(np.float64)))
            maximum = float(np.max(action_logits))
            log_normalizer = maximum + float(np.log(np.sum(np.exp(action_logits - maximum))))
            target_index = teacher_choice + 1
            total_policy_cross_entropy += log_normalizer - float(action_logits[target_index])
            policy_cross_entropy = log_normalizer - float(action_logits[target_index])

            if teacher_choice < 0:
                anchor_groups += 1
                anchor_false_positives += int(predicted_choice >= 0)
                anchor_policy_cross_entropy += policy_cross_entropy
            else:
                challenger_groups += 1
                challenger_exact_recall += int(predicted_choice == teacher_choice)
                challenger_policy_cross_entropy += policy_cross_entropy

    if group_count == 0 or candidate_count == 0:
        raise ValueError("conservative-policy evaluation dataset is empty")
    anchor_false_positive_rate = anchor_false_positives / max(anchor_groups, 1)
    selected_challenger_recall = challenger_exact_recall / max(challenger_groups, 1)
    return {
        "groups": group_count,
        "candidates": candidate_count,
        "mean_squared_error": squared_error / candidate_count,
        "root_mean_squared_error": (squared_error / candidate_count) ** 0.5,
        "mean_absolute_error": absolute_error / candidate_count,
        "zero_predictor_mean_squared_error": zero_squared_error / candidate_count,
        "lower_bound_correlation": _correlation(predictions_all, targets_all),
        "policy_cross_entropy": total_policy_cross_entropy / group_count,
        "balanced_policy_cross_entropy": 0.5 * anchor_policy_cross_entropy / max(anchor_groups, 1)
        + 0.5 * challenger_policy_cross_entropy / max(challenger_groups, 1),
        "mean_policy_regret": total_regret / group_count,
        "exact_policy_agreement": exact_agreement / group_count,
        "anchor_groups": anchor_groups,
        "anchor_false_positive_rate": anchor_false_positive_rate,
        "challenger_groups": challenger_groups,
        "selected_challenger_recall": selected_challenger_recall,
        "balanced_policy_error": 0.5 * anchor_false_positive_rate
        + 0.5 * (1.0 - selected_challenger_recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--checkpoint-steps", type=int, default=200)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_conservative_policy(
        ConservativePolicyTrainingConfig(
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
            model=ConservativePolicyModelConfig(
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                market_blocks=args.market_blocks,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _unsupported_warm_start(_model_dir: Path) -> ConservativePolicyModel:
    raise ValueError("conservative-policy v2 does not support warm start")


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


if __name__ == "__main__":
    main()
