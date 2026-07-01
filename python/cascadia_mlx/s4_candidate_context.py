"""Deterministic anchor and exact-relation indices for S4 candidate context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cascadia_mlx.s4_candidate_relation_census import (
    RELATIONS,
    candidate_relation_keys,
    stable_screen_order,
)

ANCHOR_LIMIT = 256
RELATION_NEIGHBOR_LIMIT = 8
MISSING_INDEX = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class CandidateContextIndex:
    """One complete decision's bounded context routing surface."""

    anchor_indices: np.ndarray
    relation_neighbor_indices: np.ndarray
    relation_neighbor_counts: np.ndarray
    relation_anchor_sibling_counts: np.ndarray

    def validate(self, candidate_count: int) -> None:
        anchor_count = min(candidate_count, ANCHOR_LIMIT)
        if (
            self.anchor_indices.shape != (ANCHOR_LIMIT,)
            or self.anchor_indices.dtype != np.uint16
            or self.relation_neighbor_indices.shape
            != (
                candidate_count,
                len(RELATIONS),
                RELATION_NEIGHBOR_LIMIT,
            )
            or self.relation_neighbor_indices.dtype != np.uint16
            or self.relation_neighbor_counts.shape
            != (candidate_count, len(RELATIONS))
            or self.relation_neighbor_counts.dtype != np.uint8
            or self.relation_anchor_sibling_counts.shape
            != (candidate_count, len(RELATIONS))
            or self.relation_anchor_sibling_counts.dtype != np.uint16
        ):
            raise ValueError("S4 candidate-context index tensor contract drifted")
        anchors = self.anchor_indices[:anchor_count].astype(np.int64)
        if (
            len(set(int(index) for index in anchors)) != anchor_count
            or np.any(anchors < 0)
            or np.any(anchors >= candidate_count)
            or np.any(self.anchor_indices[anchor_count:] != MISSING_INDEX)
        ):
            raise ValueError("S4 candidate-context anchor indices are invalid")
        counts = self.relation_neighbor_counts.astype(np.int64)
        if np.any(counts > RELATION_NEIGHBOR_LIMIT):
            raise ValueError("S4 relation-neighbor count exceeds its fixed limit")
        slots = np.arange(RELATION_NEIGHBOR_LIMIT)[None, None, :]
        present = slots < counts[..., None]
        neighbors = self.relation_neighbor_indices.astype(np.int64)
        if (
            np.any(neighbors[present] < 0)
            or np.any(neighbors[present] >= candidate_count)
            or np.any(neighbors[~present] != MISSING_INDEX)
            or np.any(
                self.relation_anchor_sibling_counts
                < self.relation_neighbor_counts
            )
        ):
            raise ValueError("S4 relation-neighbor indices are malformed")
        candidate_rows = np.arange(candidate_count)[:, None, None]
        if np.any((neighbors == candidate_rows) & present):
            raise ValueError("S4 relation-neighbor lists may not include self")


def build_candidate_context_index(
    candidates: np.ndarray,
    afterstate_hashes: np.ndarray,
) -> CandidateContextIndex:
    """Build stable top-256 anchors and top-8 exact siblings per relation."""
    candidates = np.asarray(candidates)
    candidate_count = len(candidates)
    if candidate_count <= 0 or candidate_count >= MISSING_INDEX:
        raise ValueError("S4 candidate count is outside the uint16 index contract")
    order = stable_screen_order(candidates)
    anchor_count = min(candidate_count, ANCHOR_LIMIT)
    anchors = order[:anchor_count].astype(np.uint16)
    anchor_mask = np.zeros(candidate_count, dtype=np.bool_)
    anchor_mask[anchors.astype(np.int64)] = True
    padded_anchors = np.full(ANCHOR_LIMIT, MISSING_INDEX, dtype=np.uint16)
    padded_anchors[:anchor_count] = anchors

    neighbor_indices = np.full(
        (
            candidate_count,
            len(RELATIONS),
            RELATION_NEIGHBOR_LIMIT,
        ),
        MISSING_INDEX,
        dtype=np.uint16,
    )
    neighbor_counts = np.zeros(
        (candidate_count, len(RELATIONS)),
        dtype=np.uint8,
    )
    anchor_sibling_counts = np.zeros(
        (candidate_count, len(RELATIONS)),
        dtype=np.uint16,
    )
    keys_by_relation = candidate_relation_keys(
        candidates["action"],
        afterstate_hashes,
    )
    for relation_index, relation_name in enumerate(RELATIONS):
        keys, valid = keys_by_relation[relation_name]
        anchor_groups: dict[bytes, list[int]] = {}
        for anchor in anchors.astype(np.int64):
            if valid[anchor]:
                anchor_groups.setdefault(bytes(keys[anchor]), []).append(int(anchor))
        for candidate in np.flatnonzero(valid):
            siblings = anchor_groups.get(bytes(keys[candidate]), [])
            sibling_count = len(siblings) - int(anchor_mask[candidate])
            anchor_sibling_counts[candidate, relation_index] = min(
                sibling_count,
                int(MISSING_INDEX) - 1,
            )
            retained = [
                sibling
                for sibling in siblings
                if sibling != candidate
            ][:RELATION_NEIGHBOR_LIMIT]
            neighbor_counts[candidate, relation_index] = len(retained)
            neighbor_indices[
                candidate,
                relation_index,
                : len(retained),
            ] = retained

    result = CandidateContextIndex(
        anchor_indices=padded_anchors,
        relation_neighbor_indices=neighbor_indices,
        relation_neighbor_counts=neighbor_counts,
        relation_anchor_sibling_counts=anchor_sibling_counts,
    )
    result.validate(candidate_count)
    return result


def verify_candidate_context_index(
    index: CandidateContextIndex,
    candidates: np.ndarray,
    afterstate_hashes: np.ndarray,
) -> None:
    """Recompute exact relation membership for every retained sibling edge."""
    candidates = np.asarray(candidates)
    index.validate(len(candidates))
    relation_keys = candidate_relation_keys(
        candidates["action"],
        afterstate_hashes,
    )
    order = stable_screen_order(candidates)[: min(len(candidates), ANCHOR_LIMIT)]
    if not np.array_equal(
        index.anchor_indices[: len(order)].astype(np.int64),
        order,
    ):
        raise ValueError("S4 candidate-context anchors are not stable screen order")
    anchor_mask = np.zeros(len(candidates), dtype=np.bool_)
    anchor_mask[order] = True
    for relation_index, relation_name in enumerate(RELATIONS):
        keys, valid = relation_keys[relation_name]
        anchor_groups: dict[bytes, list[int]] = {}
        for anchor in order:
            if valid[anchor]:
                anchor_groups.setdefault(bytes(keys[anchor]), []).append(int(anchor))
        for candidate in range(len(candidates)):
            count = int(index.relation_neighbor_counts[candidate, relation_index])
            neighbors = index.relation_neighbor_indices[
                candidate,
                relation_index,
                :count,
            ].astype(np.int64)
            siblings = (
                anchor_groups.get(bytes(keys[candidate]), [])
                if valid[candidate]
                else []
            )
            expected_count = len(siblings) - int(
                valid[candidate] and anchor_mask[candidate]
            )
            expected = []
            for sibling in siblings:
                if sibling != candidate:
                    expected.append(sibling)
                    if len(expected) == RELATION_NEIGHBOR_LIMIT:
                        break
            if int(
                index.relation_anchor_sibling_counts[
                    candidate,
                    relation_index,
                ]
            ) != expected_count or not np.array_equal(
                neighbors,
                np.asarray(expected, dtype=np.int64),
            ):
                raise ValueError(
                    f"S4 {relation_name} neighbor order or count drifted"
                )
