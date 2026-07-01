from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.frontier_supervision_identifiability import SupervisionGroup
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankTargetCache,
    _expected_rank_cache_group,
    build_expected_rank_target_mask,
    classify_expected_rank_pilot,
    expected_rank_loss_from_scores,
    expected_rank_validation_gates,
    frontier_expected_rank_loss,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_train import (
    EXPECTED_RANK_SEED,
    FrontierExpectedRankTrainingConfig,
    frontier_expected_rank_adapter,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)


def _hashes(candidates: int) -> np.ndarray:
    values = np.zeros((1, candidates, 32), dtype=np.uint8)
    for index in range(candidates):
        values[0, index, -2:] = np.frombuffer(
            index.to_bytes(2, "big"),
            dtype=np.uint8,
        )
    return values


def _batch(candidates: int = 70) -> SimpleNamespace:
    mask = mx.ones((1, candidates), dtype=mx.bool_)
    source_flags = np.zeros((1, candidates), dtype=np.int32)
    source_flags[0, 0] = GRADED_SOURCE_CHAMPION_FRONTIER
    expected_rank = np.zeros((1, candidates), dtype=np.float32)
    expected_rank[0, 1:] = np.arange(1, candidates, dtype=np.float32)
    target_mask = np.ones((1, candidates), dtype=np.bool_)
    target_mask[0, 0] = False
    return SimpleNamespace(
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
        expected_rank=mx.array(expected_rank),
        expected_rank_mask=mx.array(target_mask),
        action_hash=_hashes(candidates),
    )


def _group() -> SupervisionGroup:
    candidates = 6
    flags = np.zeros(candidates, dtype=np.uint16)
    flags[0] = GRADED_SOURCE_CHAMPION_FRONTIER
    hashes = _hashes(candidates)[0]
    means = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
    samples = np.full(candidates, 1200.0)
    return SupervisionGroup(
        group_id=9,
        phase=0,
        selected_index=0,
        source_flags=flags,
        action_hash=hashes,
        r600_mean=means.copy(),
        r600_stddev=np.ones(candidates),
        r600_samples=samples.copy(),
        r1200_mean=means.copy(),
        r1200_stddev=np.ones(candidates),
        r1200_samples=samples.copy(),
        r4800_mean=means.copy(),
        r4800_stddev=np.ones(candidates),
        r4800_samples=np.full(candidates, 4800.0),
    )


def _slice() -> dict[str, float | int]:
    return {
        "groups": 40,
        "top64_r4800_winner_recall": 0.99,
        "top64_confidence_set_coverage_95": 0.99,
        "top64_distinguishable_winner_recall": 0.98,
        "distinguishable_groups": 20,
        "mean_top64_retained_r4800_regret": 0.01,
    }


def _metrics(*, train: bool) -> dict[str, object]:
    return {
        "expected_rank_target_positive_recall": 0.80 if train else 0.50,
        "expected_rank_target_set_exact_fraction": 0.25 if train else 0.01,
        "top64_r4800_winner_recall": 0.99,
        "top64_confidence_set_coverage_95": 0.99,
        "top64_distinguishable_winner_recall": 0.98,
        "mean_top64_retained_r4800_regret": 0.01,
        "all_groups_scored_once": True,
        "all_candidates_scored_once": True,
        "all_scores_finite": True,
        "phase": {name: _slice() for name in ("early", "middle", "late")},
        "subsets": {
            name: _slice()
            for name in ("nature_token_available", "independent_draft_winner")
        },
    }


def test_cache_group_excludes_frontier_and_preserves_continuous_ranks() -> None:
    result = _expected_rank_cache_group(_group())
    ranks = result["expected_ranks"]
    assert np.isnan(ranks[0])
    assert np.all(np.isfinite(ranks[1:]))
    assert np.all(np.diff(ranks[1:]) > 0.0)
    assert result["candidate_count"] == 6


def test_cache_lookup_normalizes_signed_mlx_group_ids() -> None:
    cache = object.__new__(ExpectedRankTargetCache)
    unsigned = (1 << 64) - 7
    cache._group_index = {unsigned: 0}
    cache.candidate_counts = np.array([2], dtype=np.uint32)
    cache.offsets = np.array([0, 2], dtype=np.uint64)
    cache.expected_ranks = np.array([1.0, 2.0], dtype=np.float32)
    batch = SimpleNamespace(
        group_id=mx.array(np.array([unsigned], dtype=np.uint64).view(np.int64)),
        candidate_mask=mx.array([[True, True]]),
    )
    ranks, mask = cache.ranks_for_batch(batch)
    assert np.asarray(ranks).tolist() == [[1.0, 2.0]]
    assert np.asarray(mask).tolist() == [[True, True]]


def test_expected_rank_target_uses_lowest_rank_and_hash_ties() -> None:
    ranks = np.array([[0.0, 1.0, 2.0, 2.0, 4.0, 5.0]], dtype=np.float32)
    mask = np.array([[False, True, True, True, True, True]])
    flags = np.array(
        [[GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0, 0, 0, 0]],
        dtype=np.int32,
    )
    target = build_expected_rank_target_mask(
        expected_rank=ranks,
        expected_rank_mask=mask,
        source_flags=flags,
        candidate_mask=np.ones_like(mask),
        action_hashes=_hashes(6),
        width=4,
    )
    assert target.tolist() == [[False, True, True, True, False, False]]


def test_expected_rank_loss_is_finite_and_rewards_correct_order() -> None:
    ranks = mx.array([[1.0, 2.0, 3.0, 0.0]])
    target = mx.array([[True, True, True, False]])
    eligible = mx.array([[True, True, True, True]])
    ordered = expected_rank_loss_from_scores(
        mx.array([[3.0, 2.0, 1.0, -3.0]]),
        ranks,
        target,
        eligible,
    )
    reversed_loss = expected_rank_loss_from_scores(
        mx.array([[1.0, 2.0, 3.0, -3.0]]),
        ranks,
        target,
        eligible,
    )
    mx.eval(ordered, reversed_loss)
    assert np.isfinite(float(ordered.item()))
    assert float(ordered.item()) < float(reversed_loss.item())


def test_expected_rank_objective_reaches_residual_head() -> None:
    model = GradedOracleRanker(
        GradedOracleModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            feed_forward_multiplier=2,
        )
    )
    loss, gradients = nn.value_and_grad(model, frontier_expected_rank_loss)(
        model,
        _batch(),
    )
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert np.any(np.asarray(gradients["residual_head"]["weight"]) != 0.0)


def test_validation_gates_and_classification_preserve_strict_boundaries() -> None:
    report = {
        "train": _metrics(train=True),
        "validation": _metrics(train=False),
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    gates = expected_rank_validation_gates(report)
    assert gates["pilot_passed"]
    assert classify_expected_rank_pilot(gates) == "expected_rank_model_sufficient"

    report["validation"]["top64_r4800_winner_recall"] = 0.98
    gates = expected_rank_validation_gates(report)
    assert not gates["validation_r4800_winner_recall_strictly_above_0_98"]
    assert classify_expected_rank_pilot(gates) == "expected_rank_train_fit_only"

    report["train"]["expected_rank_target_positive_recall"] = 0.79
    gates = expected_rank_validation_gates(report)
    assert classify_expected_rank_pilot(gates) == "expected_rank_optimization_underfit"

    gates["cache_identity_passed"] = False
    assert classify_expected_rank_pilot(gates) == "expected_rank_pipeline_invalid"


def test_training_config_and_adapter_are_frozen(tmp_path: Path) -> None:
    train_cache = tmp_path / "train-cache"
    validation_cache = tmp_path / "validation-cache"
    train_cache.mkdir()
    validation_cache.mkdir()
    config = FrontierExpectedRankTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        run_dir=tmp_path / "run",
        train_target_cache=str(train_cache),
        validation_target_cache=str(validation_cache),
    )
    config.validate()
    adapter = frontier_expected_rank_adapter(config)
    assert config.seed == EXPECTED_RANK_SEED
    assert config.init_model_dir is None
    assert adapter.selection_metric == "expected_rank_target_positive_miss_rate"
    assert adapter.accuracy_metric == "expected_rank_target_set_exact_fraction"

    with pytest.raises(ValueError, match="warm starts"):
        FrontierExpectedRankTrainingConfig(
            train_dataset=tmp_path / "train",
            validation_dataset=tmp_path / "validation",
            run_dir=tmp_path / "run",
            train_target_cache=str(train_cache),
            validation_target_cache=str(validation_cache),
            init_model_dir=tmp_path / "model",
        ).validate()
