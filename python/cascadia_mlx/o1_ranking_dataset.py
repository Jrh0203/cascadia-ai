"""Exact top-64 data join and deterministic training schedule for ADR 0188."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.o1_ranking_cohort import COHORT_WIDTH, O1RankingCohortCache
from cascadia_mlx.o1_ranking_intent_cache import ARMS, O1RankingIntentCache
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM,
    R3ActionEditBatch,
    R3ActionEditMlxDataset,
)

TRAINING_SCHEDULE_DOMAIN = b"cascadia-v2-o1-ranking-training-schedule-v1"


@dataclass(frozen=True)
class O1RankingBatch:
    """One matched exact-R2 top-64 batch plus routed O1 probabilities."""

    r3: R3ActionEditBatch
    intent_features: mx.array
    cohort_rows: mx.array
    arm: str

    def __getattr__(self, name: str) -> object:
        return getattr(self.r3, name)


class O1RankingDataset:
    """Fail-closed join of graded labels, cohort membership, and O1 features."""

    def __init__(
        self,
        r3: R3ActionEditMlxDataset,
        *,
        cohort: O1RankingCohortCache,
        intent: O1RankingIntentCache,
        arm: str,
    ):
        if arm not in ARMS:
            raise ValueError(f"unknown O1 ranking arm: {arm}")
        self.r3 = r3
        self.cohort = cohort
        self.intent = intent
        self.arm = arm
        self.split = r3.split
        self.source = cohort.split(self.split)
        intent_source = intent.split(self.split)
        if self.source.groups != intent_source.groups:
            raise ValueError("O1 cohort and intent group counts differ")
        self.r3_rows = np.asarray(
            [
                r3.source.group_rows[int(group_id)]
                for group_id in np.asarray(
                    self.source.tensors["group_ids"],
                    dtype=np.uint64,
                )
            ],
            dtype=np.int64,
        )
        if len(np.unique(self.r3_rows)) != self.source.groups:
            raise ValueError("O1 cohort does not map one-to-one onto R3 groups")

    @property
    def group_count(self) -> int:
        return self.source.groups

    @property
    def candidate_count(self) -> int:
        return self.source.groups * COHORT_WIDTH

    def batch(
        self,
        rows: Sequence[int] | np.ndarray,
    ) -> O1RankingBatch:
        selected = _normalize_rows(rows, self.group_count)
        r3_rows = self.r3_rows[selected]
        positions = self.source.positions(selected)
        r3_batch = self.r3.batch(
            r3_rows,
            arm=CONTROL_ARM,
            transform_ids=np.zeros(len(selected), dtype=np.int64),
            candidate_positions=positions,
            require_selected_action=self.split == "train",
            require_champion_action=False,
            verify_control_hashes=False,
        )
        observed_sources = np.asarray(
            r3_batch.source_candidate_indices,
            dtype=np.uint16,
        )
        observed_hashes = np.asarray(r3_batch.base.action_hash, dtype=np.uint8)
        expected_sources = np.asarray(
            self.source.tensors["source_candidate_indices"][selected],
            dtype=np.uint16,
        )
        expected_hashes = np.asarray(
            self.source.tensors["action_hashes"][selected],
            dtype=np.uint8,
        )
        if (
            observed_sources.shape != (len(selected), COHORT_WIDTH)
            or observed_hashes.shape != (len(selected), COHORT_WIDTH, 32)
            or not np.array_equal(observed_sources, expected_sources)
            or not np.array_equal(observed_hashes, expected_hashes)
        ):
            raise ValueError("O1 batch action identity differs from frozen cohort")
        intent_features = self.intent.arm_features(
            self.split,
            self.arm,
            selected,
        )
        return O1RankingBatch(
            r3=r3_batch,
            intent_features=mx.array(intent_features),
            cohort_rows=mx.array(selected.astype(np.int32)),
            arm=self.arm,
        )

    def deterministic_training_batch(
        self,
        *,
        step: int,
        seed: int,
        groups_per_step: int,
    ) -> O1RankingBatch:
        if self.split != "train":
            raise ValueError("O1 deterministic training requires the train split")
        rows = deterministic_training_rows(
            step=step,
            seed=seed,
            group_ids=np.asarray(
                self.source.tensors["group_ids"],
                dtype=np.uint64,
            ),
            groups_per_step=groups_per_step,
        )
        return self.batch(rows)


def deterministic_training_rows(
    *,
    step: int,
    seed: int,
    group_ids: np.ndarray,
    groups_per_step: int,
) -> np.ndarray:
    """Draw fixed, balanced rows from one BLAKE3-keyed cyclic permutation."""
    groups = np.asarray(group_ids, dtype=np.uint64)
    if (
        step < 0
        or seed < 0
        or groups_per_step <= 0
        or groups.ndim != 1
        or not len(groups)
        or len(np.unique(groups)) != len(groups)
        or groups_per_step > len(groups)
    ):
        raise ValueError("O1 training schedule inputs are invalid")
    permutation = _training_permutation(groups, seed=seed)
    positions = (
        step * groups_per_step + np.arange(groups_per_step, dtype=np.int64)
    ) % len(groups)
    result = permutation[positions]
    if len(np.unique(result)) != len(result):
        raise AssertionError("O1 training step repeated a group")
    return result


def _training_permutation(
    group_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    keyed: list[tuple[bytes, int]] = []
    for row, group_id in enumerate(group_ids):
        digest = blake3.blake3()
        digest.update(TRAINING_SCHEDULE_DOMAIN)
        digest.update(int(seed).to_bytes(8, "little"))
        digest.update(int(group_id).to_bytes(8, "little"))
        keyed.append((digest.digest(), row))
    return np.asarray(
        [row for _key, row in sorted(keyed)],
        dtype=np.int64,
    )


def _normalize_rows(
    rows: Sequence[int] | np.ndarray,
    group_count: int,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64)
    if (
        values.ndim != 1
        or not len(values)
        or np.any(values < 0)
        or np.any(values >= group_count)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("O1 batch rows must be unique, nonempty, and in range")
    return values
