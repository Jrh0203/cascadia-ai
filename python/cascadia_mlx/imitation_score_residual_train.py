"""Train an immediate-score anchored MCE continuation residual on MLX."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from cascadia_mlx.imitation_distribution_train import evaluate_imitation_evidence
from cascadia_mlx.imitation_model import (
    IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
    ImitationModelConfig,
    SharedStateActionRanker,
    score_imitation_actions,
    score_residual_imitation_loss,
)
from cascadia_mlx.imitation_targets_dataset import ImitationEvidenceDataset
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


def _score_residual_config() -> ImitationModelConfig:
    return ImitationModelConfig(
        architecture=IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
    )


@dataclass(frozen=True)
class ImitationScoreResidualTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    epochs: int = 20
    group_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260617
    checkpoint_steps: int = 500
    validation_patience: int = 5
    resume: bool = False
    init_model_dir: Path | None = None
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    model: ImitationModelConfig = field(default_factory=_score_residual_config)

    def validate(self) -> None:
        if self.epochs <= 0 or self.group_batch_size <= 0 or self.checkpoint_steps <= 0:
            raise ValueError("score-residual training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("score-residual optimizer configuration is invalid")
        if self.validation_patience <= 0:
            raise ValueError("validation patience must be positive")
        if self.init_model_dir is not None:
            raise ValueError("the first score-residual run does not warm-start")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("the first score-residual run accepts one train and validation split")
        self.model.validate()
        if self.model.architecture != IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3:
            raise ValueError("score-residual training requires the frozen v3 architecture")


def score_residual_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="canonical-action-mce-score-residual",
        dataset_factory=ImitationEvidenceDataset,
        model_factory=lambda values: SharedStateActionRanker(
            ImitationModelConfig.from_dict(values)
        ),
        new_model=SharedStateActionRanker,
        load_promoted=_reject_warm_start,
        loss=score_residual_imitation_loss,
        score_batch=lambda model, batch: score_imitation_actions(
            model,
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.action_features,
            batch.candidate_mask,
        ),
        evaluate=lambda model, dataset, group_batch_size: evaluate_imitation_evidence(
            model,
            dataset,
            group_batch_size,
            loss_function=score_residual_imitation_loss,
            loss_metric="anchored_loss",
        ),
        selection_metric="anchored_loss",
        accuracy_metric="top1_accuracy",
    )


def _reject_warm_start(_path: Path) -> SharedStateActionRanker:
    raise ValueError("the first score-residual run does not warm-start")


def train_score_residual(
    config: ImitationScoreResidualTrainingConfig,
) -> dict[str, object]:
    train_dataset = ImitationEvidenceDataset(config.train_dataset)
    validation_dataset = ImitationEvidenceDataset(config.validation_dataset)
    if (
        train_dataset.source.manifest["candidates"]
        != validation_dataset.source.manifest["candidates"]
    ):
        raise ValueError("score-residual train and validation candidate contracts differ")
    return train_ranking(config, adapter=score_residual_adapter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_score_residual(
        ImitationScoreResidualTrainingConfig(
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
                architecture=IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
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
