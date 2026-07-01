from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from cascadia_mlx.counterfactual_advantage_dataset import (
    CounterfactualAdvantageDataset,
)
from cascadia_mlx.distributional_opportunity_model import (
    ARMS,
    ATOM_COUNT,
    DistributionalOpportunityModelConfig,
    DistributionalOpportunityRanker,
    distributional_opportunity_loss,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from test_counterfactual_advantage_dataset import (
    write_counterfactual_advantage_dataset,
)


def _batch(tmp_path: Path):
    root = tmp_path / "counterfactual"
    write_counterfactual_advantage_dataset(root, split="train", game_index=9_996)
    return next(CounterfactualAdvantageDataset(root).batches(4))


def _model(arm: str) -> DistributionalOpportunityRanker:
    return DistributionalOpportunityRanker(
        DistributionalOpportunityModelConfig(
            arm=arm,
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            candidate_blocks=1,
        )
    )


def test_distributional_arms_share_layout_and_initial_tensor() -> None:
    layouts = set()
    tensors = set()
    for arm in ARMS:
        mx.random.seed(20260618)
        model = _model(arm)
        layouts.add(parameter_layout_blake3(model))
        tensors.add(parameter_tensor_blake3(model))

    assert len(layouts) == 1
    assert len(tensors) == 1


@pytest.mark.parametrize("arm", ARMS)
def test_distributional_arm_has_finite_gradient(
    tmp_path: Path,
    arm: str,
) -> None:
    batch = _batch(tmp_path)
    mx.random.seed(20260618)
    model = _model(arm)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    offsets = mx.array(np.linspace(-2.0, 2.0, ATOM_COUNT, dtype=np.float32))
    loss_and_grad = nn.value_and_grad(
        model,
        lambda candidate, values: distributional_opportunity_loss(
            candidate,
            values,
            homoscedastic_offsets=(offsets if arm == "c0-homoscedastic-mean" else None),
        ),
    )

    loss, gradients = loss_and_grad(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)

    assert np.isfinite(float(loss.item()))


@pytest.mark.parametrize("arm", ARMS)
def test_distributional_outputs_are_centered_and_ordered(
    tmp_path: Path,
    arm: str,
) -> None:
    batch = _batch(tmp_path)
    model = _model(arm)
    offsets = mx.array(np.linspace(-2.0, 2.0, ATOM_COUNT, dtype=np.float32))

    means, atoms, uncertainty = model.distribution(
        batch,
        homoscedastic_offsets=offsets if arm == ARMS[0] else None,
    )
    mx.eval(means, atoms, uncertainty)
    means_np = np.asarray(means)
    atoms_np = np.asarray(atoms)
    uncertainty_np = np.asarray(uncertainty)

    np.testing.assert_allclose(means_np.mean(axis=1), 0.0, atol=1e-6)
    assert atoms_np.shape == (4, 4, ATOM_COUNT)
    assert np.all(np.diff(atoms_np, axis=-1) >= -1e-6)
    assert np.all(uncertainty_np >= 0.0)


def test_homoscedastic_control_requires_frozen_offsets(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    model = _model(ARMS[0])

    with pytest.raises(ValueError, match="requires frozen offsets"):
        model.distribution(batch)
    with pytest.raises(ValueError, match="requires frozen offsets"):
        distributional_opportunity_loss(model, batch)
