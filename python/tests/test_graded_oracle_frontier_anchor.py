from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
    frontier_anchored_loss,
    frontier_anchored_retained_indices,
    frontier_anchored_target_ceiling_gates,
    frontier_anchored_validation_gates,
)
from cascadia_mlx.graded_oracle_frontier_anchor_train import (
    FRONTIER_ANCHORED_TRAINING_SEEDS,
    FrontierAnchoredTrainingConfig,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)


def _hashes(groups: int, candidates: int) -> np.ndarray:
    values = np.zeros((groups, candidates, 32), dtype=np.uint8)
    for index in range(candidates):
        values[:, index, -2:] = np.frombuffer(
            index.to_bytes(2, "big"),
            dtype=np.uint8,
        )
    return values


def test_target_keeps_frontier_outside_learned_quota_and_uses_stable_r1200() -> None:
    candidate_mask = np.ones((1, 6), dtype=np.bool_)
    source_flags = np.array(
        [[GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0, GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0]],
        dtype=np.int32,
    )
    r1200_mask = np.ones((1, 6), dtype=np.bool_)
    r1200_mean = np.array([[100.0, 95.0, 95.0, 90.0, 80.0, 70.0]])
    target = build_frontier_anchored_target_mask(
        r1200_mean=r1200_mean,
        r1200_mask=r1200_mask,
        source_flags=source_flags,
        candidate_mask=candidate_mask,
        action_hashes=_hashes(1, 6),
        width=4,
    )
    assert target.tolist() == [[False, True, True, False, False, False]]


def test_selector_retains_every_frontier_action_then_fills_to_exact_width() -> None:
    scores = np.array([0.0, 100.0, 90.0, -10.0, 80.0, 70.0])
    source_flags = np.array(
        [GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0, GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0]
    )
    retained = frontier_anchored_retained_indices(
        scores=scores,
        source_flags=source_flags,
        action_hashes=_hashes(1, 6)[0],
        width=4,
    )
    assert set(retained.tolist()) == {0, 1, 2, 3}
    assert len(retained) == 4


def test_set_objective_is_finite_and_reaches_the_score_head() -> None:
    candidates = 70
    mask = mx.ones((1, candidates), dtype=mx.bool_)
    source_flags = np.zeros((1, candidates), dtype=np.int32)
    source_flags[0, 0] = GRADED_SOURCE_CHAMPION_FRONTIER
    r1200_mean = np.linspace(100.0, 31.0, candidates, dtype=np.float32)[None, :]
    batch = SimpleNamespace(
        board_entities=mx.zeros((1, 4, 23, ENTITY_DIM)),
        board_mask=mx.zeros((1, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((1, 4, ENTITY_DIM)),
        market_mask=mx.ones((1, 4), dtype=mx.bool_),
        global_features=mx.zeros((1, GLOBAL_DIM)),
        public_supply=mx.zeros((1, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)),
        action_features=mx.zeros((1, candidates, GRADED_ORACLE_ACTION_DIM)),
        prior_features=mx.zeros((1, candidates, GRADED_ORACLE_PRIOR_DIM)),
        staged_market_entities=mx.zeros((1, candidates, 4, ENTITY_DIM)),
        staged_market_mask=mx.ones((1, candidates, 4), dtype=mx.bool_),
        staged_public_supply=mx.zeros(
            (1, candidates, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)
        ),
        candidate_mask=mask,
        source_flags=mx.array(source_flags),
        screen_value=mx.array(
            np.linspace(80.0, 79.0, candidates, dtype=np.float32)[None, :]
        ),
        r600_mean=mx.array(r1200_mean),
        r600_mask=mask,
        r1200_mean=mx.array(r1200_mean),
        r1200_mask=mask,
        r4800_mean=mx.array(r1200_mean),
        r4800_mask=mask,
        action_hash=_hashes(1, candidates),
    )
    config = GradedOracleModelConfig(
        hidden_dim=24,
        attention_heads=4,
        board_blocks=0,
        market_blocks=0,
        feed_forward_multiplier=2,
    )
    model = GradedOracleRanker(config)
    loss_and_grad = nn.value_and_grad(model, frontier_anchored_loss)
    loss, gradients = loss_and_grad(model, batch)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert np.any(np.asarray(gradients["residual_head"]["weight"]) != 0.0)


def test_validation_gates_preserve_strict_recall_boundary() -> None:
    metrics = {
        "top64_r4800_winner_recall": 0.99,
        "top64_confidence_set_coverage_95": 0.99,
        "top64_distinguishable_winner_recall": 0.98,
        "mean_top64_retained_r4800_regret": 0.10,
        "all_groups_scored_once": True,
        "all_candidates_scored_once": True,
        "all_scores_finite": True,
        "proposal_width": 64,
        "phase": {
            name: {
                "top64_r4800_winner_recall": 0.98,
                "top64_confidence_set_coverage_95": 0.98,
                "mean_top64_retained_r4800_regret": 0.10,
            }
            for name in ("early", "middle", "late")
        },
        "subsets": {
            name: {
                "groups": 20,
                "top64_r4800_winner_recall": 0.95,
                "mean_top64_retained_r4800_regret": 0.20,
            }
            for name in ("nature_token_available", "independent_draft_winner")
        },
    }
    assert all(frontier_anchored_validation_gates(metrics).values())
    metrics["top64_r4800_winner_recall"] = 0.98
    gates = frontier_anchored_validation_gates(metrics)
    assert not gates["top64_r4800_winner_recall_strictly_greater_than_0_98"]


def test_target_ceiling_requires_every_phase_to_clear_frozen_gates() -> None:
    slice_values = {
        "groups": 80,
        "top64_r4800_winner_recall": 0.99,
        "top64_confidence_set_coverage_95": 1.0,
        "top64_distinguishable_winner_recall": 1.0,
        "distinguishable_groups": 20,
        "mean_top64_retained_r4800_regret": 0.001,
    }
    report = {
        "overall": dict(slice_values),
        "phase": {
            name: dict(slice_values) for name in ("early", "middle", "late")
        },
        "subsets": {
            name: dict(slice_values)
            for name in ("nature_token_available", "independent_draft_winner")
        },
        "all_groups_seen_once": True,
        "all_candidates_seen_once": True,
        "proposal_width": 64,
        "test_split_opened": False,
    }
    assert frontier_anchored_target_ceiling_gates(report)[
        "target_ceiling_passed"
    ]
    report["phase"]["middle"]["top64_r4800_winner_recall"] = 0.97
    assert not frontier_anchored_target_ceiling_gates(report)[
        "target_ceiling_passed"
    ]
    report["phase"]["middle"]["top64_r4800_winner_recall"] = 0.99
    report["subsets"]["independent_draft_winner"][
        "top64_r4800_winner_recall"
    ] = 0.94
    assert not frontier_anchored_target_ceiling_gates(report)[
        "target_ceiling_passed"
    ]


def test_training_config_keeps_architecture_and_optimizer_fixed(tmp_path) -> None:
    config = FrontierAnchoredTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        run_dir=tmp_path / "run",
    )
    config.validate()
    assert config.model == GradedOracleModelConfig()
    assert {
        2026061601,
        2026061602,
        2026061603,
        2026061604,
    } == FRONTIER_ANCHORED_TRAINING_SEEDS
