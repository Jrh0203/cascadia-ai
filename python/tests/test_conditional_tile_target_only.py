from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from cascadia_mlx.conditional_tile_target_only import (
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENT_ID,
    OBJECTIVE_ID,
    SEED,
    frozen_config,
    target_only_tile_loss,
)


class _FixedScores:
    def __init__(self, scores: mx.array):
        self.scores = scores

    def __call__(
        self,
        _state: mx.array,
        _context: mx.array,
        _items: mx.array,
        _item_mask: mx.array,
    ) -> mx.array:
        return self.scores


def _inputs(target: list[bool]) -> tuple[mx.array, ...]:
    width = len(target)
    return (
        mx.zeros((1, 1)),
        mx.zeros((1, 1)),
        mx.zeros((1, width, 1)),
        mx.ones((1, width), dtype=mx.bool_),
        mx.zeros((1, width)),
        mx.ones((1, width), dtype=mx.bool_),
        mx.array([target], dtype=mx.bool_),
    )


def test_frozen_contract_is_complete() -> None:
    config = frozen_config()
    assert EXPERIMENT_ID == "conditional-tile-target-only-objective-v1"
    assert OBJECTIVE_ID == "balanced-top32-membership-bce-v1"
    assert config.stage == "tile"
    assert config.seed == SEED
    assert config.epochs == EPOCHS
    assert config.batch_size == BATCH_SIZE


def test_target_only_loss_is_balanced_across_classes() -> None:
    scores = mx.array([[1.0, -1.0, 1.0, -1.0]])
    loss = target_only_tile_loss(
        _FixedScores(scores),
        *_inputs([True, True, False, False]),
    )
    expected = nn.softplus(mx.array(-1.0)) + nn.softplus(mx.array(1.0))
    mx.eval(loss, expected)
    assert abs(float(loss.item()) - float(expected.item())) < 1e-6


def test_within_budget_query_has_zero_training_pressure() -> None:
    loss = target_only_tile_loss(
        _FixedScores(mx.array([[2.0, -3.0]])),
        *_inputs([True, True]),
    )
    mx.eval(loss)
    assert float(loss.item()) == 0.0
