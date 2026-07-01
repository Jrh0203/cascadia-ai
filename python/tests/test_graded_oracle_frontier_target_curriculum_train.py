from __future__ import annotations

from pathlib import Path
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
from cascadia_mlx.graded_oracle_frontier_anchor import GRADED_SOURCE_CHAMPION_FRONTIER
from cascadia_mlx.graded_oracle_frontier_target_curriculum_train import (
    TARGET_CURRICULUM_SEED,
    FrontierTargetCurriculumConfig,
    frontier_target_curriculum_adapter,
    frontier_target_only_loss,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)


def _batch(candidates: int = 70) -> SimpleNamespace:
    mask = mx.ones((1, candidates), dtype=mx.bool_)
    source_flags = np.zeros((1, candidates), dtype=np.int32)
    source_flags[0, 0] = GRADED_SOURCE_CHAMPION_FRONTIER
    means = np.linspace(100.0, 31.0, candidates, dtype=np.float32)[None, :]
    hashes = np.zeros((1, candidates, 32), dtype=np.uint8)
    for index in range(candidates):
        hashes[0, index, -2:] = np.frombuffer(
            index.to_bytes(2, "big"),
            dtype=np.uint8,
        )
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
        r600_mean=mx.array(means),
        r600_mask=mask,
        r1200_mean=mx.array(means),
        r1200_mask=mask,
        r4800_mean=mx.array(means),
        r4800_mask=mask,
        action_hash=hashes,
    )


def test_target_only_curriculum_reaches_score_head() -> None:
    model = GradedOracleRanker(
        GradedOracleModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            feed_forward_multiplier=2,
        )
    )
    loss, gradients = nn.value_and_grad(model, frontier_target_only_loss)(
        model,
        _batch(),
    )
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert np.any(np.asarray(gradients["residual_head"]["weight"]) != 0.0)


def test_adapter_selects_target_recall_and_uses_checkpoint_manifest() -> None:
    adapter = frontier_target_curriculum_adapter()
    assert adapter.selection_metric == "target_positive_miss_rate"
    assert adapter.accuracy_metric == "target_set_exact_fraction"
    assert adapter.init_manifest_name == "checkpoint.json"


def test_config_freezes_single_host_pilot(tmp_path: Path) -> None:
    config = FrontierTargetCurriculumConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        run_dir=tmp_path / "run",
        init_model_dir=tmp_path / "checkpoint",
    )
    config.validate()
    assert config.seed == TARGET_CURRICULUM_SEED
    assert config.learning_rate == 3e-5
    assert config.epochs == 20
