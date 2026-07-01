from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.relational_substrate_mlx_cache import ARMS
from cascadia_mlx.relational_substrate_mlx_train import (
    TRAINING_SEED,
    RelationalSubstrateTrainingConfig,
    RelationalSubstrateTrainingProtocol,
    cross_arm_initialization,
    scientific_batch_blake3,
)


def test_relational_protocol_and_initialization_are_frozen() -> None:
    protocol = RelationalSubstrateTrainingProtocol()
    protocol.validate()
    assert protocol.seed == TRAINING_SEED
    parity = cross_arm_initialization()
    assert len(set(parity["cross_arm_parameter_counts"].values())) == 1
    assert (
        len(
            set(
                parity[
                    "cross_arm_initial_parameter_tensor_blake3"
                ].values()
            )
        )
        == 1
    )


def test_bounded_smoke_requires_a_real_r6_binary(tmp_path: Path) -> None:
    config = RelationalSubstrateTrainingConfig(
        train_dataset=tmp_path,
        validation_dataset=tmp_path,
        r3_cache=tmp_path,
        relational_cache=tmp_path,
        s1_cache=tmp_path,
        r6_binary=tmp_path / "missing",
        run_dir=tmp_path / "run",
        output=tmp_path / "report.json",
        arm=ARMS[0],
        smoke_steps=1,
    )
    with pytest.raises(ValueError, match="R6 replay binary"):
        config.validate()


def test_scientific_batch_hash_ignores_representation_surface() -> None:
    base = SimpleNamespace(
        group_id=mx.array([17], dtype=mx.uint64),
        candidate_mask=mx.array([[True, True, False]]),
        action_hash=mx.array(
            np.arange(96, dtype=np.uint8).reshape(1, 3, 32)
        ),
    )
    common = {
        "base": base,
        "parent": SimpleNamespace(transform_ids=mx.array([7])),
        "source_candidate_indices": mx.array([[4, 9, 0]]),
    }
    left = SimpleNamespace(**common, arm=ARMS[0])
    right = SimpleNamespace(**common, arm=ARMS[3])
    assert scientific_batch_blake3(left) == scientific_batch_blake3(
        right
    )
