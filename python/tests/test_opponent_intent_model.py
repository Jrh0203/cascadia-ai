from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from cascadia_mlx.opponent_intent_dataset import OpponentIntentDataset
from cascadia_mlx.opponent_intent_model import (
    ARMS,
    OpponentIntentModelConfig,
    OpponentIntentSurvivalModel,
    opponent_intent_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from test_opponent_intent_dataset import write_opponent_intent_dataset


def test_all_arms_share_graph_and_initialization(tmp_path: Path) -> None:
    root = write_opponent_intent_dataset(
        tmp_path / "train",
        split="train",
        game_index=300,
    )
    batch = next(OpponentIntentDataset(root).batches(4))
    fingerprints = []
    for arm in ARMS:
        mx.random.seed(2_026_061_704)
        model = OpponentIntentSurvivalModel(OpponentIntentModelConfig(arm=arm))
        prediction = model(batch)
        loss = opponent_intent_loss(model, batch)
        mx.eval(prediction.disposition_logits, loss)
        fingerprints.append(
            (
                parameter_count(model),
                parameter_layout_blake3(model),
                parameter_tensor_blake3(model),
            )
        )
        assert prediction.disposition_logits.shape == (4, 4, 4)
        assert prediction.tile_slot_logits.shape == (4, 3, 4)
        assert np.isfinite(float(loss.item()))
    assert len(set(fingerprints)) == 1


def test_history_gate_is_isolated_from_public_state_control(
    tmp_path: Path,
) -> None:
    root = write_opponent_intent_dataset(
        tmp_path / "train",
        split="train",
        game_index=400,
    )
    batch = next(OpponentIntentDataset(root).batches(4))
    changed = replace(
        batch,
        history_features=batch.history_features + mx.ones_like(batch.history_features) * 0.5,
    )
    outputs = {}
    for arm in ARMS[:2]:
        mx.random.seed(2_026_061_704)
        model = OpponentIntentSurvivalModel(OpponentIntentModelConfig(arm=arm))
        original = model(batch).disposition_logits
        mutated = model(changed).disposition_logits
        mx.eval(original, mutated)
        outputs[arm] = (
            np.asarray(original),
            np.asarray(mutated),
        )
    assert np.array_equal(*outputs[ARMS[0]])
    assert not np.array_equal(*outputs[ARMS[1]])
