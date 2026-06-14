from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.counterfactual_advantage_dataset import (
    CounterfactualAdvantageDataset,
)
from cascadia_mlx.counterfactual_advantage_model import (
    CORRECTION_SCALE,
    CounterfactualAdvantageModelConfig,
    CounterfactualAdvantageRanker,
    counterfactual_advantage_loss,
    counterfactual_advantage_scores,
)
from cascadia_mlx.counterfactual_advantage_train import (
    CounterfactualAdvantageTrainingConfig,
    train_counterfactual_advantage,
)
from test_counterfactual_advantage_dataset import (
    write_counterfactual_advantage_dataset,
)


def _small_model() -> CounterfactualAdvantageRanker:
    return CounterfactualAdvantageRanker(
        CounterfactualAdvantageModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            candidate_blocks=1,
        )
    )


def test_counterfactual_advantage_model_starts_at_exact_immediate_score(
    tmp_path: Path,
) -> None:
    root = tmp_path / "counterfactual"
    write_counterfactual_advantage_dataset(root, split="train", game_index=9_996)
    batch = next(CounterfactualAdvantageDataset(root).batches(4))
    model = _small_model()

    scores = counterfactual_advantage_scores(model, batch)
    loss = counterfactual_advantage_loss(model, batch)
    mx.eval(scores, loss)

    np.testing.assert_array_equal(
        np.asarray(scores),
        np.asarray(batch.immediate_score),
    )
    assert np.isfinite(float(loss.item()))


def test_counterfactual_advantage_optimizer_changes_bounded_scores(tmp_path: Path) -> None:
    root = tmp_path / "counterfactual"
    write_counterfactual_advantage_dataset(root, split="train", game_index=9_996)
    batch = next(CounterfactualAdvantageDataset(root).batches(4))
    model = _small_model()
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, counterfactual_advantage_loss)

    loss, gradients = loss_and_grad(model, batch)
    optimizer.update(model, gradients)
    scores = counterfactual_advantage_scores(model, batch)
    mx.eval(model.parameters(), optimizer.state, loss, scores)

    differences = np.asarray(scores) - np.asarray(batch.immediate_score)
    assert np.any(np.abs(differences) > 0)
    assert np.max(np.abs(differences)) <= CORRECTION_SCALE


def test_counterfactual_advantage_training_checkpoints_and_resumes(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    run_dir = tmp_path / "run"
    write_counterfactual_advantage_dataset(train_root, split="train", game_index=9_996)
    write_counterfactual_advantage_dataset(
        validation_root,
        split="validation",
        game_index=9_997,
    )
    model_config = CounterfactualAdvantageModelConfig(
        hidden_dim=32,
        attention_heads=4,
        board_blocks=0,
        market_blocks=0,
        candidate_blocks=1,
    )
    first = train_counterfactual_advantage(
        CounterfactualAdvantageTrainingConfig(
            train_dataset=train_root,
            validation_dataset=validation_root,
            run_dir=run_dir,
            epochs=1,
            group_batch_size=4,
            checkpoint_steps=1,
            validation_patience=2,
            model=model_config,
        )
    )
    resumed = train_counterfactual_advantage(
        CounterfactualAdvantageTrainingConfig(
            train_dataset=train_root,
            validation_dataset=validation_root,
            run_dir=run_dir,
            epochs=2,
            group_batch_size=4,
            checkpoint_steps=1,
            validation_patience=2,
            resume=True,
            model=model_config,
        )
    )

    assert first["epochs"] == 1
    assert resumed["epochs"] == 2
    assert resumed["global_step"] == 2
    assert (run_dir / "best.json").is_file()
    assert (run_dir / "final-report.json").is_file()
