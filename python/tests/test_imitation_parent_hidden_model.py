from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.imitation_model import _masked_standardize
from cascadia_mlx.imitation_parent_hidden_model import (
    ParentHiddenSetResidual,
    parent_hidden_distributional_loss,
    score_parent_hidden_actions,
)


def _batch() -> SimpleNamespace:
    parent_total = mx.array([[94.0, 91.0, 96.0]])
    return SimpleNamespace(
        parent_hidden=mx.arange(3 * 64, dtype=mx.float32).reshape(1, 3, 64) / 100.0,
        parent_immediate=mx.array([[40.0, 39.0, 41.0]]),
        parent_remaining=mx.array([[54.0, 52.0, 55.0]]),
        parent_total=parent_total,
        parent_rank=mx.array([[2.0, 3.0, 1.0]]),
        candidate_mask=mx.array([[True, True, True]]),
        teacher_mean=mx.array([[93.0, 90.0, 97.0]]),
        teacher_stddev=mx.array([[2.0, 3.0, 2.0]]),
        teacher_samples=mx.array([[50.0, 25.0, 50.0]]),
        teacher_scored=mx.array([[True, True, True]]),
        selected=mx.array([[False, False, True]]),
    )


def test_parent_hidden_residual_starts_as_exact_standardized_parent() -> None:
    batch = _batch()
    model = ParentHiddenSetResidual()
    scores = score_parent_hidden_actions(model, batch)
    expected = _masked_standardize(batch.parent_total, batch.candidate_mask)

    mx.eval(scores, expected)
    assert np.array_equal(np.asarray(scores), np.asarray(expected))
    assert int(mx.argmax(scores, axis=1)[0].item()) == 2


def test_parent_hidden_residual_is_permutation_equivariant() -> None:
    batch = _batch()
    model = ParentHiddenSetResidual()
    permutation = mx.array([2, 0, 1])
    permuted = SimpleNamespace(
        **{name: value[:, permutation] for name, value in vars(batch).items()}
    )
    original_scores = score_parent_hidden_actions(model, batch)
    permuted_scores = score_parent_hidden_actions(model, permuted)

    mx.eval(original_scores, permuted_scores)
    assert np.array_equal(
        np.asarray(permuted_scores),
        np.asarray(original_scores[:, permutation]),
    )


def test_parent_hidden_distributional_loss_is_finite() -> None:
    loss = parent_hidden_distributional_loss(ParentHiddenSetResidual(), _batch())
    mx.eval(loss)
    assert bool(mx.isfinite(loss).item())
