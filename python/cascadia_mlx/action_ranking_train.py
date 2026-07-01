"""Resumable MLX training for explicit action-delta ranking."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from cascadia_mlx.action_ranking_dataset import ActionRankingDataset
from cascadia_mlx.action_ranking_model import (
    ActionDeltaRanker,
    ActionRankingModelConfig,
    action_ranking_loss,
)
from cascadia_mlx.action_ranking_promote import load_promoted_action_ranking_model
from cascadia_mlx.ranking_train import (
    GroupedRankingAdapter,
    train_ranking,
)


@dataclass(frozen=True)
class ActionRankingTrainingConfig:
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
    checkpoint_steps: int = 500
    validation_patience: int = 5
    resume: bool = False
    model: ActionRankingModelConfig = field(default_factory=ActionRankingModelConfig)

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
        all_train = (self.train_dataset, *self.additional_train_datasets)
        all_validation = (
            self.validation_dataset,
            *self.regression_validation_datasets,
        )
        if len({path.resolve() for path in all_train}) != len(all_train):
            raise ValueError("action-ranking training datasets must be unique")
        if len({path.resolve() for path in all_validation}) != len(all_validation):
            raise ValueError("action-ranking validation datasets must be unique")
        self.model.validate()


def action_ranking_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="action-delta-ranking",
        dataset_factory=ActionRankingDataset,
        model_factory=lambda values: ActionDeltaRanker(ActionRankingModelConfig.from_dict(values)),
        new_model=ActionDeltaRanker,
        load_promoted=load_promoted_action_ranking_model,
        loss=action_ranking_loss,
        score_batch=lambda model, batch: model(
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.action_features,
        ),
    )


def train_action_ranking(config: ActionRankingTrainingConfig) -> dict[str, object]:
    return train_ranking(config, adapter=action_ranking_adapter())


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
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_action_ranking(
        ActionRankingTrainingConfig(
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
            model=ActionRankingModelConfig(
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
