from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ARCHETYPE_COUNT,
    ARMS,
    EXACT_SUPPLY_DIM,
    EXACT_TOKEN_COUNT,
    FRONTIER_FEATURE_DIM,
    S1ExactSupplyBatch,
)
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
    S1ExactSupplyRanker,
    parameter_count,
    parameter_layout_blake3,
    s1_exact_supply_loss,
)


def _batch() -> S1ExactSupplyBatch:
    groups = 2
    candidates = 3
    candidate_mask = mx.array([[True, True, False], [True, True, True]])
    board_mask = np.zeros((groups, 4, 23), dtype=np.bool_)
    board_mask[:, :, 0] = True
    screen = mx.zeros((groups, candidates))
    teacher = mx.array([[2.0, 1.0, 0.0], [1.0, 3.0, 2.0]])
    base = SimpleNamespace(
        board_entities=mx.zeros((groups, 4, 23, ENTITY_DIM)),
        board_mask=mx.array(board_mask),
        market_entities=mx.zeros((groups, 4, ENTITY_DIM)),
        market_mask=mx.ones((groups, 4), dtype=mx.bool_),
        global_features=mx.zeros((groups, GLOBAL_DIM)),
        staged_market_entities=mx.zeros((groups, candidates, 4, ENTITY_DIM)),
        staged_market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        action_features=mx.zeros((groups, candidates, GRADED_ORACLE_ACTION_DIM)),
        prior_features=mx.zeros((groups, candidates, GRADED_ORACLE_PRIOR_DIM)),
        candidate_mask=candidate_mask,
        screen_value=screen,
        r600_mean=teacher,
        r600_stddev=mx.ones((groups, candidates)),
        r600_samples=mx.ones((groups, candidates)) * 600,
        r600_mask=candidate_mask,
        r1200_mean=teacher,
        r1200_stddev=mx.ones((groups, candidates)),
        r1200_samples=mx.ones((groups, candidates)) * 1200,
        r1200_mask=candidate_mask,
        r4800_mean=teacher,
        r4800_stddev=mx.ones((groups, candidates)),
        r4800_samples=mx.ones((groups, candidates)) * 4800,
        r4800_mask=candidate_mask,
        selected_index=mx.array([0, 1]),
    )
    return S1ExactSupplyBatch(
        base=base,
        supply_vector=mx.zeros((groups, EXACT_SUPPLY_DIM)),
        staged_supply_vector=mx.zeros((groups, candidates, EXACT_SUPPLY_DIM)),
        supply_tokens=mx.zeros((groups, EXACT_TOKEN_COUNT, 32)),
        supply_mask=mx.ones((groups, EXACT_TOKEN_COUNT), dtype=mx.bool_),
        refill_target=mx.ones((groups, ARCHETYPE_COUNT)) / ARCHETYPE_COUNT,
        selected_archetype=mx.zeros((groups, candidates), dtype=mx.int32),
        frontier_features=mx.zeros((groups, candidates, FRONTIER_FEATURE_DIM)),
    )


def test_all_arms_have_the_exact_same_frozen_parameter_capacity() -> None:
    counts = {}
    layouts = {}
    for arm in ARMS:
        model = S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        counts[arm] = parameter_count(model)
        layouts[arm] = parameter_layout_blake3(model)
    assert set(counts.values()) == {FROZEN_PARAMETER_COUNT}
    assert len(set(layouts.values())) == 1


def test_every_arm_executes_finitely_with_the_shared_shapes_and_objective() -> None:
    for arm in ARMS:
        mx.random.seed(2026061707)
        model = S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        prediction = model(_batch())
        loss = s1_exact_supply_loss(model, _batch())
        mx.eval(
            prediction.scores,
            prediction.standard_errors,
            prediction.refill_probabilities,
            loss,
        )
        assert prediction.scores.shape == (2, 3)
        assert prediction.refill_probabilities.shape == (2, ARCHETYPE_COUNT)
        assert np.isfinite(np.asarray(prediction.scores)).all()
        assert np.isfinite(np.asarray(prediction.refill_probabilities)).all()
        assert np.isfinite(float(loss.item()))
