from __future__ import annotations

import numpy as np
from cascadia_mlx.r3_action_edit_mlx_cache import (
    deterministic_training_rows,
    deterministic_transform_ids,
)
from cascadia_mlx.r4_bounded_parent_mlx_cache import ARMS
from cascadia_mlx.r4_bounded_parent_mlx_train import (
    TRAINING_SEED,
    R4BoundedParentTrainingProtocol,
    cross_arm_initialization,
)


def test_frozen_protocol_and_cross_arm_initialization_are_exact() -> None:
    R4BoundedParentTrainingProtocol().validate()
    identity = cross_arm_initialization()
    assert set(identity["cross_arm_parameter_counts"]) == set(ARMS)
    assert len(set(identity["cross_arm_parameter_counts"].values())) == 1
    assert len(set(identity["cross_arm_parameter_layout_blake3"].values())) == 1
    assert len(set(identity["cross_arm_initial_parameter_tensor_blake3"].values())) == 1


def test_new_seed_preserves_complete_common_schedule() -> None:
    all_rows = np.arange(20, dtype=np.int64)
    low_supply = np.asarray([2, 4, 6], dtype=np.int64)
    independent = np.asarray([1, 3], dtype=np.int64)
    schedule = np.stack(
        [
            deterministic_training_rows(
                step=step,
                seed=TRAINING_SEED,
                all_rows=all_rows,
                low_supply_rows=low_supply,
                independent_winner_rows=independent,
            )
            for step in range(20)
        ]
    )
    for stream in range(3):
        assert set(schedule[:, stream]) == set(all_rows)
    transforms = np.concatenate(
        [deterministic_transform_ids(step=step, seed=TRAINING_SEED) for step in range(100)]
    )
    assert set(transforms) == set(range(12))
