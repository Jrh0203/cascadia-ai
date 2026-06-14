"""Resumable MLX training from full-frontier MCE action distributions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from cascadia_mlx.imitation_model import (
    IMITATION_ARCHITECTURE_V1,
    IMITATION_DISTRIBUTION_SCORE_FLOOR,
    ImitationModelConfig,
    SharedStateActionRanker,
    distributional_imitation_loss,
    score_imitation_actions,
)
from cascadia_mlx.imitation_targets_dataset import ImitationEvidenceDataset
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class ImitationDistributionTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    epochs: int = 20
    group_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260616
    checkpoint_steps: int = 500
    validation_patience: int = 5
    resume: bool = False
    init_model_dir: Path | None = None
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    model: ImitationModelConfig = field(default_factory=ImitationModelConfig)

    def validate(self) -> None:
        if self.epochs <= 0 or self.group_batch_size <= 0 or self.checkpoint_steps <= 0:
            raise ValueError("distributional imitation training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("distributional imitation optimizer configuration is invalid")
        if self.validation_patience <= 0:
            raise ValueError("validation patience must be positive")
        if self.init_model_dir is not None:
            raise ValueError("the first distributional imitation run does not warm-start")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("the first distributional run accepts one train and validation split")
        self.model.validate()
        if self.model.architecture != IMITATION_ARCHITECTURE_V1:
            raise ValueError("the first distributional run freezes the v1 shared-state ranker")


def distributional_imitation_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="canonical-action-mce-distribution",
        dataset_factory=ImitationEvidenceDataset,
        model_factory=lambda values: SharedStateActionRanker(
            ImitationModelConfig.from_dict(values)
        ),
        new_model=SharedStateActionRanker,
        load_promoted=_reject_warm_start,
        loss=distributional_imitation_loss,
        score_batch=lambda model, batch: score_imitation_actions(
            model,
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.action_features,
            batch.candidate_mask,
            getattr(batch, "parent_total", None),
            getattr(batch, "parent_rank", None),
        ),
        evaluate=evaluate_distributional_imitation,
        selection_metric="distributional_loss",
        accuracy_metric="top1_accuracy",
    )


def _reject_warm_start(_path: Path) -> SharedStateActionRanker:
    raise ValueError("the first distributional imitation run does not warm-start")


def train_distributional_imitation(
    config: ImitationDistributionTrainingConfig,
) -> dict[str, object]:
    train_dataset = ImitationEvidenceDataset(config.train_dataset)
    validation_dataset = ImitationEvidenceDataset(config.validation_dataset)
    if (
        train_dataset.source.manifest["candidates"]
        != validation_dataset.source.manifest["candidates"]
    ):
        raise ValueError("distributional train and validation candidate contracts differ")
    return train_ranking(config, adapter=distributional_imitation_adapter())


def evaluate_distributional_imitation(
    model: SharedStateActionRanker,
    dataset: ImitationEvidenceDataset,
    group_batch_size: int,
) -> dict[str, object]:
    return evaluate_imitation_evidence(
        model,
        dataset,
        group_batch_size,
        loss_function=distributional_imitation_loss,
        loss_metric="distributional_loss",
    )


def evaluate_imitation_evidence(
    model: object,
    dataset: ImitationEvidenceDataset,
    group_batch_size: int,
    *,
    loss_function,
    loss_metric: str,
    score_function=None,
) -> dict[str, object]:
    model.eval()
    groups = 0
    candidates = 0
    total_loss = 0.0
    top1 = 0
    top5 = 0
    reciprocal_rank = 0.0
    predicted_teacher_coverage = 0
    conditional_regret: list[float] = []
    scored_top1 = 0
    pairwise_correct = 0
    pairwise_count = 0
    pairwise_log_loss = 0.0
    pairwise_brier = 0.0
    teacher_differences: list[float] = []
    student_differences: list[float] = []
    rank_correlations: list[float] = []

    for batch in dataset.batches(group_batch_size):
        scores = (
            score_function(model, batch)
            if score_function is not None
            else score_imitation_actions(
                model,
                batch.board_entities,
                batch.board_mask,
                batch.market_entities,
                batch.market_mask,
                batch.global_features,
                batch.action_features,
                batch.candidate_mask,
                getattr(batch, "parent_total", None),
                getattr(batch, "parent_rank", None),
            )
        )
        loss = loss_function(model, batch)
        mx.eval(scores, loss)
        score_values = np.asarray(scores)
        candidate_masks = np.asarray(batch.candidate_mask)
        selected_values = np.asarray(batch.selected)
        scored_values = np.asarray(batch.teacher_scored)
        teacher_means = np.asarray(batch.teacher_mean)
        teacher_stddev = np.asarray(batch.teacher_stddev)
        teacher_samples = np.asarray(batch.teacher_samples)
        batch_groups = len(score_values)
        total_loss += float(loss.item()) * batch_groups
        groups += batch_groups

        for student, mask, selected, scored, means, stddev, samples in zip(
            score_values,
            candidate_masks,
            selected_values,
            scored_values,
            teacher_means,
            teacher_stddev,
            teacher_samples,
            strict=True,
        ):
            student = student[mask]
            selected = selected[mask]
            scored = scored[mask]
            means = means[mask]
            stddev = stddev[mask]
            samples = samples[mask]
            candidates += len(student)
            selected_index = int(np.flatnonzero(selected)[0])
            ranking = np.argsort(-student, kind="stable")
            selected_rank = int(np.flatnonzero(ranking == selected_index)[0]) + 1
            top1 += int(selected_rank == 1)
            top5 += int(selected_rank <= 5)
            reciprocal_rank += 1.0 / selected_rank

            predicted = int(ranking[0])
            if scored[predicted]:
                predicted_teacher_coverage += 1
                conditional_regret.append(float(means[selected_index] - means[predicted]))

            scored_indices = np.flatnonzero(scored)
            scored_student = student[scored_indices]
            scored_means = means[scored_indices]
            scored_best = float(np.max(scored_means))
            scored_prediction = int(np.argmax(scored_student))
            scored_top1 += int(float(scored_means[scored_prediction]) == scored_best)
            if len(scored_indices) > 1:
                rank_correlations.append(_rank_correlation(scored_student, scored_means))
            standard_error = stddev / np.sqrt(np.maximum(samples, 1.0))
            for left_offset, left in enumerate(scored_indices):
                for right in scored_indices[left_offset + 1 :]:
                    teacher_difference = float(means[left] - means[right])
                    student_difference = float(student[left] - student[right])
                    if teacher_difference != 0:
                        pairwise_correct += int(
                            np.sign(teacher_difference) == np.sign(student_difference)
                        )
                        teacher_differences.append(teacher_difference)
                        student_differences.append(student_difference)
                    scale = float(
                        np.sqrt(
                            standard_error[left] ** 2
                            + standard_error[right] ** 2
                            + IMITATION_DISTRIBUTION_SCORE_FLOOR**2
                        )
                    )
                    teacher_probability = _sigmoid(teacher_difference / scale)
                    student_probability = _sigmoid(student_difference)
                    clipped = np.clip(student_probability, 1e-7, 1.0 - 1e-7)
                    pairwise_log_loss += -(
                        teacher_probability * np.log(clipped)
                        + (1.0 - teacher_probability) * np.log(1.0 - clipped)
                    )
                    pairwise_brier += (student_probability - teacher_probability) ** 2
                    pairwise_count += 1

    if groups == 0:
        raise ValueError("distributional imitation evaluation dataset is empty")
    return {
        "groups": groups,
        "candidates": candidates,
        loss_metric: total_loss / groups,
        "top1_accuracy": top1 / groups,
        "top5_recall": top5 / groups,
        "mean_reciprocal_rank": reciprocal_rank / groups,
        "predicted_teacher_coverage": predicted_teacher_coverage / groups,
        "conditional_mean_regret": (
            float(np.mean(conditional_regret)) if conditional_regret else None
        ),
        "scored_top1_value_recall": scored_top1 / groups,
        "scored_pairwise_accuracy": pairwise_correct / max(len(teacher_differences), 1),
        "scored_pairwise_log_loss": pairwise_log_loss / max(pairwise_count, 1),
        "scored_pairwise_brier": pairwise_brier / max(pairwise_count, 1),
        "mean_scored_rank_correlation": (
            float(np.mean(rank_correlations)) if rank_correlations else 0.0
        ),
        "scored_value_difference_correlation": _correlation(
            student_differences,
            teacher_differences,
        ),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + np.exp(-value))
    exponent = np.exp(value)
    return float(exponent / (1.0 + exponent))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = np.argsort(np.argsort(left)).astype(np.float64)
    right_ranks = np.argsort(np.argsort(right)).astype(np.float64)
    return _correlation(left_ranks.tolist(), right_ranks.tolist())


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_distributional_imitation(
        ImitationDistributionTrainingConfig(
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
            model=ImitationModelConfig(
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                market_blocks=args.market_blocks,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
