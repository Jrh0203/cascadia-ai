from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.o1_ranking_dataset import deterministic_training_rows


def test_training_schedule_is_deterministic_label_blind_and_nonrepeating() -> None:
    group_ids = np.arange(16, dtype=np.uint64) * 101 + 17

    first = deterministic_training_rows(
        step=3,
        seed=2026061719,
        group_ids=group_ids,
        groups_per_step=8,
    )
    repeated = deterministic_training_rows(
        step=3,
        seed=2026061719,
        group_ids=group_ids,
        groups_per_step=8,
    )
    changed_seed = deterministic_training_rows(
        step=3,
        seed=2026061720,
        group_ids=group_ids,
        groups_per_step=8,
    )

    assert np.array_equal(first, repeated)
    assert len(np.unique(first)) == 8
    assert not np.array_equal(first, changed_seed)


def test_training_schedule_crosses_epoch_without_repeating_group() -> None:
    group_ids = np.arange(6, dtype=np.uint64) + 100

    rows = deterministic_training_rows(
        step=1,
        seed=17,
        group_ids=group_ids,
        groups_per_step=4,
    )

    assert len(rows) == 4
    assert len(np.unique(rows)) == 4


def test_training_schedule_rejects_oversized_batch() -> None:
    with pytest.raises(ValueError, match="invalid"):
        deterministic_training_rows(
            step=0,
            seed=17,
            group_ids=np.arange(3, dtype=np.uint64),
            groups_per_step=4,
        )
