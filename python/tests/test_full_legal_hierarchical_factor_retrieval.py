from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    PARENT_STATE_DIM,
    STAGE_CONTEXT_DIMS,
    STAGE_ITEM_DIMS,
    STAGES,
    StageTrainingConfig,
    _factor_values,
    _selected_stage_items,
    build_stage_model,
    classify_hierarchical_retrieval,
    hierarchical_factor_loss,
)


def test_factor_partition_ignores_immediate_consequences() -> None:
    actions = np.arange(280, dtype=np.float32).reshape(2, 140)
    before = _factor_values(actions)
    changed = actions.copy()
    changed[:, 128:140] += 10_000
    after = _factor_values(changed)
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)


def test_frozen_stage_config_rejects_drift() -> None:
    config = StageTrainingConfig.frozen("tile")
    config.validate()
    with pytest.raises(ValueError, match="contract drifted"):
        StageTrainingConfig(
            stage="tile",
            seed=config.seed,
            epochs=config.epochs + 1,
            batch_size=config.batch_size,
        ).validate()


@pytest.mark.parametrize("stage", STAGES)
def test_stage_model_and_loss_are_finite(stage: str) -> None:
    model = build_stage_model(stage)
    batch = 2
    width = 5
    item_mask = mx.ones((batch, width), dtype=mx.bool_)
    expected_rank = mx.array(
        [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
        dtype=mx.float32,
    )
    target = mx.array(
        [[True, True, False, False, False]] * batch,
    )
    loss, gradients = nn.value_and_grad(
        model,
        hierarchical_factor_loss,
    )(
        model,
        mx.zeros((batch, PARENT_STATE_DIM)),
        mx.zeros((batch, STAGE_CONTEXT_DIMS[stage])),
        mx.zeros((batch, width, STAGE_ITEM_DIMS[stage])),
        item_mask,
        expected_rank,
        item_mask,
        target,
    )
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert tuple(model(
        mx.zeros((batch, PARENT_STATE_DIM)),
        mx.zeros((batch, STAGE_CONTEXT_DIMS[stage])),
        mx.zeros((batch, width, STAGE_ITEM_DIMS[stage])),
        item_mask,
    ).shape) == (batch, width)


def test_stage_selection_uses_score_then_exact_factor_order() -> None:
    selected = _selected_stage_items(
        scores=np.asarray([1.0, 1.0, 0.5], dtype=np.float32),
        offsets=np.asarray([0, 3], dtype=np.int64),
        width=2,
    )
    np.testing.assert_array_equal(
        selected,
        np.asarray([True, True, False]),
    )


@pytest.mark.parametrize(
    ("pipeline", "proposal", "selector", "expected"),
    [
        (
            False,
            True,
            True,
            "hierarchical_retrieval_pipeline_invalid",
        ),
        (
            True,
            False,
            True,
            "hierarchical_proposal_insufficient",
        ),
        (
            True,
            True,
            False,
            "hierarchical_selector_insufficient",
        ),
        (
            True,
            True,
            True,
            "hierarchical_factor_retrieval_sufficient",
        ),
    ],
)
def test_classification_precedence(
    pipeline: bool,
    proposal: bool,
    selector: bool,
    expected: str,
) -> None:
    assert classify_hierarchical_retrieval(
        {
            "pipeline_passed": pipeline,
            "proposal_passed": proposal,
            "selector_passed": selector,
        }
    ) == expected
