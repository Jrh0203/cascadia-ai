from __future__ import annotations

import pytest
from cascadia_v3_mlx.checkpoint_eval import CheckpointEvaluationError, _optimizer_settings


def test_optimizer_settings_accept_scheduled_manifest() -> None:
    assert _optimizer_settings(
        {"base_learning_rate": 5e-4, "weight_decay": 1e-6}
    ) == (5e-4, 1e-6)


def test_optimizer_settings_accept_legacy_training_manifest() -> None:
    assert _optimizer_settings(
        {"optimizer": {"learning_rate": 1e-3, "weight_decay": 2e-6}}
    ) == (1e-3, 2e-6)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"base_learning_rate": 0, "weight_decay": 0},
        {"base_learning_rate": 1e-3, "weight_decay": -1},
    ],
)
def test_optimizer_settings_reject_invalid_values(value: dict[str, object]) -> None:
    with pytest.raises(CheckpointEvaluationError):
        _optimizer_settings(value)
