from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_dataset import _CANDIDATE_DTYPE
from cascadia_mlx.s4_candidate_context import (
    ANCHOR_LIMIT,
    MISSING_INDEX,
    RELATION_NEIGHBOR_LIMIT,
    CandidateContextIndex,
    build_candidate_context_index,
    verify_candidate_context_index,
)
from cascadia_mlx.s4_candidate_relation_census import RELATIONS


def _candidates(count: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros(count, dtype=_CANDIDATE_DTYPE)
    candidates["screen_rank"] = np.arange(count, 0, -1, dtype=np.uint16)
    candidates["action_hash"][:, 0] = np.arange(count, dtype=np.uint8)
    candidates["action"]["draft_kind"] = np.arange(count) % 2
    candidates["action"]["tile_slot"] = np.arange(count) % 2
    candidates["action"]["wildlife_slot"] = np.arange(count) % 2
    candidates["action"]["tile_id"] = np.arange(count) % 2
    candidates["action"]["tile_q"] = 1
    candidates["action"]["tile_r"] = 2
    candidates["action"]["rotation"] = np.arange(count) % 3
    candidates["action"]["wildlife_present"] = 1
    candidates["action"]["wildlife_q"] = np.arange(count) % 4
    candidates["action"]["wildlife_r"] = 0
    afterstates = np.zeros((count, 32), dtype=np.uint8)
    afterstates[:, 0] = np.arange(count) % 3
    return candidates, afterstates


def test_candidate_context_index_is_stable_bounded_and_exact() -> None:
    candidates, afterstates = _candidates(12)
    index = build_candidate_context_index(candidates, afterstates)

    assert index.anchor_indices[:12].tolist() == list(reversed(range(12)))
    assert np.all(index.anchor_indices[12:] == MISSING_INDEX)
    assert index.relation_neighbor_indices.shape == (
        12,
        len(RELATIONS),
        RELATION_NEIGHBOR_LIMIT,
    )
    frontier = RELATIONS.index("same_frontier")
    assert index.relation_neighbor_counts[0, frontier] == 8
    assert index.relation_anchor_sibling_counts[0, frontier] == 11
    assert 0 not in index.relation_neighbor_indices[0, frontier, :8]
    verify_candidate_context_index(index, candidates, afterstates)


def test_candidate_context_caps_anchors_at_256() -> None:
    candidates, afterstates = _candidates(300)
    candidates["action_hash"][:, 0] = np.arange(300, dtype=np.uint16) % 256
    index = build_candidate_context_index(candidates, afterstates)

    assert index.anchor_indices.shape == (ANCHOR_LIMIT,)
    assert np.all(index.anchor_indices != MISSING_INDEX)
    assert set(index.anchor_indices.astype(int)) == set(range(44, 300))
    verify_candidate_context_index(index, candidates, afterstates)


def test_candidate_context_handles_absent_wildlife_relation() -> None:
    candidates, afterstates = _candidates(6)
    candidates["action"]["wildlife_present"][0] = 0
    index = build_candidate_context_index(candidates, afterstates)

    wildlife = RELATIONS.index("same_wildlife_destination")
    assert index.relation_neighbor_counts[0, wildlife] == 0
    assert index.relation_anchor_sibling_counts[0, wildlife] == 0
    verify_candidate_context_index(index, candidates, afterstates)


def test_candidate_context_validation_rejects_self_edges() -> None:
    candidates, afterstates = _candidates(4)
    index = build_candidate_context_index(candidates, afterstates)
    neighbors = index.relation_neighbor_indices.copy()
    counts = index.relation_neighbor_counts.copy()
    relation = RELATIONS.index("same_frontier")
    neighbors[0, relation, 0] = 0
    counts[0, relation] = max(1, counts[0, relation])
    corrupted = CandidateContextIndex(
        anchor_indices=index.anchor_indices,
        relation_neighbor_indices=neighbors,
        relation_neighbor_counts=counts,
        relation_anchor_sibling_counts=index.relation_anchor_sibling_counts,
    )

    with pytest.raises(ValueError, match="self"):
        corrupted.validate(len(candidates))
