from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.checkpoint import (
    CheckpointError,
    TrainerState,
    load_latest_checkpoint,
    save_checkpoint,
)
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.model import EntitySetValueModel, ModelConfig


def _inputs() -> tuple[mx.array, ...]:
    return (
        mx.zeros((1, 4, 23, ENTITY_DIM)),
        mx.ones((1, 4, 23), dtype=mx.bool_),
        mx.zeros((1, 4, ENTITY_DIM)),
        mx.ones((1, 4), dtype=mx.bool_),
        mx.zeros((1, GLOBAL_DIM)),
    )


def test_checkpoint_round_trip_preserves_model_optimizer_and_cursor(tmp_path: Path) -> None:
    mx.random.seed(5)
    config = ModelConfig(hidden_dim=32, attention_heads=4, board_blocks=0, market_blocks=0)
    model = EntitySetValueModel(config)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    inputs = _inputs()

    loss_and_grad = nn.value_and_grad(model, lambda current, values: mx.mean(current(*values) ** 2))
    loss, gradients = loss_and_grad(model, inputs)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    expected = model(*inputs)
    mx.eval(expected)
    state = TrainerState(
        epoch=2,
        batch_in_epoch=7,
        global_step=19,
        elapsed_seconds=12.5,
        best_validation_mae=4.5,
        best_ranking_loss=1.25,
        best_top1_accuracy=0.75,
        ranking_epochs_without_improvement=3,
    )

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        state,
        metadata={"kind": "unit-test", "epoch": 2},
    )
    loaded, loaded_optimizer, loaded_state, _ = load_latest_checkpoint(
        tmp_path,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    actual = loaded(*inputs)
    mx.eval(actual)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0, rtol=0)
    assert loaded_state == state
    assert int(loaded_optimizer.state["step"].item()) == 1
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    assert manifest["metadata"] == {"kind": "unit-test", "epoch": 2}


def test_checkpoint_detects_tampered_weights(tmp_path: Path) -> None:
    model = EntitySetValueModel(
        ModelConfig(hidden_dim=32, attention_heads=4, board_blocks=0, market_blocks=0)
    )
    optimizer = optim.AdamW(learning_rate=1e-3)
    checkpoint = save_checkpoint(tmp_path, model, optimizer, TrainerState())
    weights = checkpoint / "model.safetensors"
    content = bytearray(weights.read_bytes())
    content[-1] ^= 1
    weights.write_bytes(content)

    try:
        load_latest_checkpoint(tmp_path, learning_rate=1e-3, weight_decay=0.01)
    except CheckpointError as error:
        assert "integrity" in str(error)
    else:
        raise AssertionError("tampered checkpoint was accepted")
