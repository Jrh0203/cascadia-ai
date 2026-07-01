"""Resumable MLX training for canonical selected-action imitation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from cascadia_mlx.imitation_dataset import (
    ImitationDataset,
    randomly_rotate_imitation_batch,
)
from cascadia_mlx.imitation_model import (
    IMITATION_ARCHITECTURE_CROSS_V2,
    IMITATION_ARCHITECTURE_RESIDUAL_V2,
    IMITATION_ARCHITECTURE_V1,
    ImitationModelConfig,
    SharedStateActionRanker,
    imitation_loss,
    score_imitation_actions,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class ImitationTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    epochs: int = 20
    group_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260612
    checkpoint_steps: int = 500
    validation_patience: int = 5
    resume: bool = False
    init_model_dir: Path | None = None
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    hex_rotation_augmentation: bool = False
    model: ImitationModelConfig = field(default_factory=ImitationModelConfig)

    def validate(self) -> None:
        if self.epochs <= 0 or self.group_batch_size <= 0 or self.checkpoint_steps <= 0:
            raise ValueError("imitation training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("imitation optimizer configuration is invalid")
        if self.validation_patience <= 0:
            raise ValueError("validation patience must be positive")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError(
                "the frozen first imitation run accepts one train and validation split"
            )
        self.model.validate()


def imitation_adapter(
    hex_rotation_augmentation: bool = False,
) -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="canonical-action-imitation",
        dataset_factory=ImitationDataset,
        model_factory=lambda values: SharedStateActionRanker(
            ImitationModelConfig.from_dict(values)
        ),
        new_model=SharedStateActionRanker,
        load_promoted=_reject_warm_start,
        loss=imitation_loss,
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
        augment_batch=(randomly_rotate_imitation_batch if hex_rotation_augmentation else None),
        selection_metric="listwise_loss",
        accuracy_metric="top1_accuracy",
    )


def _reject_warm_start(_path: Path) -> SharedStateActionRanker:
    raise ValueError("the frozen first imitation run does not warm-start")


def train_imitation(config: ImitationTrainingConfig) -> dict[str, object]:
    train_dataset = ImitationDataset(config.train_dataset)
    validation_dataset = ImitationDataset(config.validation_dataset)
    if train_dataset.manifest["candidates"] != validation_dataset.manifest["candidates"]:
        raise ValueError("imitation train and validation candidate contracts differ")
    return train_ranking(
        config,
        adapter=imitation_adapter(config.hex_rotation_augmentation),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--architecture",
        choices=(
            IMITATION_ARCHITECTURE_V1,
            IMITATION_ARCHITECTURE_CROSS_V2,
            IMITATION_ARCHITECTURE_RESIDUAL_V2,
        ),
        default=IMITATION_ARCHITECTURE_V1,
    )
    parser.add_argument("--immediate-rank-prior", type=float, default=0.0)
    parser.add_argument("--hex-rotation-augmentation", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_imitation(
        ImitationTrainingConfig(
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
            hex_rotation_augmentation=args.hex_rotation_augmentation,
            model=ImitationModelConfig(
                architecture=args.architecture,
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                market_blocks=args.market_blocks,
                immediate_rank_prior=args.immediate_rank_prior,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
