"""Exact R3-plus-context dataset binding for the S4 candidate-set comparison."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cascadia_mlx.r3_action_edit_mlx_cache import (
    ARMS as R3_ARMS,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    R3ActionEditMlxDataset,
    deterministic_training_rows,
    deterministic_transform_ids,
)
from cascadia_mlx.s4_candidate_context_cache import (
    S4CandidateContextCache,
    S4CandidateContextCacheError,
)
from cascadia_mlx.s4_candidate_set_mlx_model import (
    S4_ARMS,
    S4CandidateSetBatch,
    mlx_candidate_context,
)


class S4CandidateSetDataset:
    """The failed R3 radius-one substrate bound to exact S4 context."""

    def __init__(
        self,
        r3: R3ActionEditMlxDataset,
        *,
        context_cache: S4CandidateContextCache,
    ):
        if r3.split not in context_cache.splits:
            raise S4CandidateContextCacheError(
                f"S4 context cache has no {r3.split} split"
            )
        context_split = context_cache.splits[r3.split]
        if (
            len(context_split.rows) != r3.group_count
            or len(context_split.action_hashes) != r3.candidate_count
        ):
            raise S4CandidateContextCacheError(
                "S4 context coverage differs from its R3 dataset"
            )
        self.r3 = r3
        self.context_cache = context_cache
        self.root = r3.base.root
        self.manifest = r3.base.manifest
        self.split = r3.split
        self.group_count = r3.group_count
        self.candidate_count = r3.candidate_count
        self.low_supply_rows = r3.low_supply_rows
        self.independent_winner_rows = r3.independent_winner_rows

    def batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        arm: str,
        transform_ids: Sequence[int] | np.ndarray | None = None,
        verify_control_hashes: bool = True,
    ) -> S4CandidateSetBatch:
        if arm not in S4_ARMS:
            raise ValueError(f"unknown S4 comparison arm: {arm}")
        selected_rows = np.asarray(rows, dtype=np.int64)
        r3_batch = self.r3.batch(
            selected_rows,
            arm=R3_ARMS[3],
            transform_ids=transform_ids,
            verify_control_hashes=verify_control_hashes,
        )
        context = self.context_cache.materialize(
            self.split,
            selected_rows,
            action_hashes=np.asarray(r3_batch.base.action_hash, dtype=np.uint8),
            candidate_mask=np.asarray(
                r3_batch.base.candidate_mask,
                dtype=np.bool_,
            ),
        )
        return S4CandidateSetBatch(
            r3=r3_batch,
            context=mlx_candidate_context(context),
        )

    def deterministic_training_batch(
        self,
        *,
        step: int,
        seed: int,
        arm: str,
        verify_control_hashes: bool = True,
    ) -> S4CandidateSetBatch:
        if self.split != "train":
            raise ValueError("deterministic S4 training requires the train split")
        rows = deterministic_training_rows(
            step=step,
            seed=seed,
            all_rows=np.arange(self.group_count, dtype=np.int64),
            low_supply_rows=self.low_supply_rows,
            independent_winner_rows=self.independent_winner_rows,
        )
        transforms = deterministic_transform_ids(
            step=step,
            seed=seed,
            slots=len(rows),
        )
        return self.batch(
            rows,
            arm=arm,
            transform_ids=transforms,
            verify_control_hashes=verify_control_hashes,
        )
