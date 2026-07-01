from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.r3_action_edit_mlx_cache import ARMS
from cascadia_mlx.r3_action_edit_mlx_train import (
    MAX_SMOKE_STEPS,
    R3ActionEditTrainingConfig,
    R3ActionEditTrainingProtocol,
    _cross_arm_initialization,
    _validation_probe_rows,
    scientific_batch_blake3,
)


def _config(**values: object) -> R3ActionEditTrainingConfig:
    defaults: dict[str, object] = {
        "train_dataset": Path("train"),
        "validation_dataset": Path("validation"),
        "cache": Path("cache"),
        "s1_cache": Path("s1"),
        "run_dir": Path("run"),
        "output": Path("report.json"),
        "arm": ARMS[0],
        "smoke_steps": 3,
    }
    defaults.update(values)
    return R3ActionEditTrainingConfig(**defaults)


def test_training_protocol_and_smoke_boundary_fail_closed() -> None:
    _config().validate()
    with pytest.raises(ValueError, match="at most 10"):
        _config(smoke_steps=MAX_SMOKE_STEPS + 1).validate()
    with pytest.raises(ValueError, match="authorization"):
        _config(smoke_steps=None).validate()
    drifted = replace(R3ActionEditTrainingProtocol(), training_steps=2999)
    with pytest.raises(ValueError, match="drifted"):
        drifted.validate()


def test_scientific_batch_identity_excludes_arm_specific_tokens() -> None:
    base = SimpleNamespace(
        group_id=mx.array([17], dtype=mx.int64),
        candidate_mask=mx.array([[True, True]]),
        action_hash=np.arange(64, dtype=np.uint8).reshape(1, 2, 32),
    )
    common = {
        "base": base,
        "parent": SimpleNamespace(transform_ids=mx.array([7])),
        "source_candidate_indices": mx.array([[3, 9]], dtype=mx.int32),
    }
    left = SimpleNamespace(
        **common,
        candidate_token_features=mx.zeros((1, 2, 3, 80)),
    )
    right = SimpleNamespace(
        **common,
        candidate_token_features=mx.ones((1, 2, 9, 80)),
    )
    assert scientific_batch_blake3(left) == scientific_batch_blake3(right)


def test_cross_arm_initialization_is_exactly_identical() -> None:
    identity = _cross_arm_initialization()
    assert len(set(identity["cross_arm_parameter_counts"].values())) == 1
    assert len(set(identity["cross_arm_parameter_layout_blake3"].values())) == 1
    assert len(set(identity["cross_arm_initial_parameter_tensor_blake3"].values())) == 1


def test_validation_probe_is_fixed_unique_and_bounded() -> None:
    rows = _validation_probe_rows(240)
    assert len(rows) == 24
    assert rows[0] == 0
    assert rows[-1] == 239
    assert len(np.unique(rows)) == len(rows)
