from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.o1_ranking_cohort import (
    COHORT_WIDTH,
    cohort_row_blake3,
    select_cohort_positions,
    stable_score_ranking,
)


def _hashes(count: int) -> np.ndarray:
    values = np.zeros((count, 32), dtype=np.uint8)
    values[:, 0] = np.arange(count, dtype=np.uint8)
    return values


def test_stable_score_ranking_uses_hashes_only_for_score_ties() -> None:
    scores = np.asarray([2.0, 3.0, 3.0, 1.0], dtype=np.float32)
    hashes = _hashes(4)
    hashes[1, 0] = 9
    hashes[2, 0] = 4

    ranking = stable_score_ranking(scores, hashes)

    np.testing.assert_array_equal(ranking, [2, 1, 0, 3])


def test_train_cohort_inserts_selected_only_when_top64_omits_it() -> None:
    scores = np.arange(70, dtype=np.float32)
    hashes = _hashes(70)

    positions, ranks = select_cohort_positions(
        scores,
        hashes,
        split="train",
        selected_position=0,
    )

    assert len(positions) == COHORT_WIDTH
    assert 0 in positions
    assert 6 not in positions
    assert set(int(value) for value in ranks if value < 63) == set(range(63))
    assert int(ranks[np.flatnonzero(positions == 0)[0]]) == 69


def test_train_and_validation_use_strict_top64_when_selected_is_already_present() -> None:
    scores = np.arange(70, dtype=np.float32)
    hashes = _hashes(70)

    train_positions, train_ranks = select_cohort_positions(
        scores,
        hashes,
        split="train",
        selected_position=69,
    )
    validation_positions, validation_ranks = select_cohort_positions(
        scores,
        hashes,
        split="validation",
        selected_position=0,
    )

    np.testing.assert_array_equal(train_positions, np.arange(6, 70))
    np.testing.assert_array_equal(validation_positions, np.arange(6, 70))
    assert set(train_ranks) == set(range(COHORT_WIDTH))
    assert set(validation_ranks) == set(range(COHORT_WIDTH))


def test_cohort_row_hash_binds_scores_indices_ranks_and_actions() -> None:
    positions = np.arange(COHORT_WIDTH, dtype=np.uint16)
    sources = positions + 10
    ranks = positions[::-1].copy()
    scores = np.linspace(80.0, 100.0, COHORT_WIDTH, dtype=np.float32)
    hashes = np.zeros((COHORT_WIDTH, 32), dtype=np.uint8)
    hashes[:, 0] = positions

    expected = cohort_row_blake3(
        group_id=123,
        candidate_positions=positions,
        source_candidate_indices=sources,
        base_ranks=ranks,
        base_scores=scores,
        action_hashes=hashes,
    )
    assert expected == cohort_row_blake3(
        group_id=123,
        candidate_positions=positions,
        source_candidate_indices=sources,
        base_ranks=ranks,
        base_scores=scores,
        action_hashes=hashes,
    )

    changed = scores.copy()
    changed[0] += 0.25
    assert expected != cohort_row_blake3(
        group_id=123,
        candidate_positions=positions,
        source_candidate_indices=sources,
        base_ranks=ranks,
        base_scores=changed,
        action_hashes=hashes,
    )


def test_cohort_selection_rejects_short_or_unknown_splits() -> None:
    with pytest.raises(ValueError, match="at least 64"):
        select_cohort_positions(
            np.ones(63, dtype=np.float32),
            _hashes(63),
            split="validation",
            selected_position=0,
        )
    with pytest.raises(ValueError, match="train or validation"):
        select_cohort_positions(
            np.ones(64, dtype=np.float32),
            _hashes(64),
            split="test",
            selected_position=0,
        )
