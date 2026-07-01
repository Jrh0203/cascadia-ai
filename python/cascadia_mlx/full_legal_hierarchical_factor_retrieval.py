"""ADR 0115 learned hierarchical complete-action factor retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
    GradedOracleBatch,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    build_expected_rank_target_mask,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    Scale16ExpectedRankDataset,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LOCAL_GEOMETRY_RELATION_DIM,
    candidate_local_geometry,
)

EXPERIMENT_ID = "full-legal-hierarchical-factor-retrieval-pilot-v1"
CACHE_SCHEMA = "hierarchical-factor-retrieval-cache-v1"
CACHE_SCHEMA_VERSION = 1
FROZEN_ADR0114_BLAKE3 = (
    "0de5dedb15d1608068348bb2b2dd2d47f8b3ec27a27e3b7c4418379dea89e700"
)
STAGES = ("draft", "tile", "wildlife")
STAGE_WIDTHS = {"draft": 16, "tile": 32, "wildlife": 8}
STAGE_EPOCHS = {"draft": 20, "tile": 20, "wildlife": 10}
STAGE_BATCH_SIZES = {"draft": 32, "tile": 32, "wildlife": 256}
STAGE_SEEDS = {
    "draft": 2026061645,
    "tile": 2026061646,
    "wildlife": 2026061647,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
TARGET_SCALE = 16.0
STUDENT_TEMPERATURE = 2.0

DRAFT_FACTOR_DIM = 34 + (128 - 45)
TILE_FACTOR_DIM = 8
WILDLIFE_FACTOR_DIM = 3
STAGED_PUBLIC_DIM = 4 * 31 + 4 + GRADED_ORACLE_PUBLIC_SUPPLY_SIZE
DESCENDANT_VALUE_DIM = GRADED_ORACLE_PRIOR_DIM + 12
DESCENDANT_STATS_DIM = DESCENDANT_VALUE_DIM * 3 + 1
TILE_LOCAL_DIM = 6 * LOCAL_GEOMETRY_RELATION_DIM
WILDLIFE_LOCAL_DIM = 7 * LOCAL_GEOMETRY_RELATION_DIM
PARENT_STATE_DIM = (
    4 * 23 * 31
    + 4 * 23
    + 4 * 31
    + 4
    + 96
    + GRADED_ORACLE_PUBLIC_SUPPLY_SIZE
)
STAGE_CONTEXT_DIMS = {
    "draft": 1,
    "tile": DRAFT_FACTOR_DIM + STAGED_PUBLIC_DIM,
    "wildlife": (
        DRAFT_FACTOR_DIM
        + STAGED_PUBLIC_DIM
        + TILE_FACTOR_DIM
        + TILE_LOCAL_DIM
    ),
}
STAGE_ITEM_DIMS = {
    "draft": DRAFT_FACTOR_DIM + STAGED_PUBLIC_DIM + DESCENDANT_STATS_DIM,
    "tile": TILE_FACTOR_DIM + TILE_LOCAL_DIM + DESCENDANT_STATS_DIM,
    "wildlife": (
        WILDLIFE_FACTOR_DIM
        + WILDLIFE_LOCAL_DIM
        + DESCENDANT_VALUE_DIM
    ),
}


@dataclass(frozen=True)
class StageTrainingConfig:
    """Frozen training contract for one conditional stage."""

    stage: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    hidden_dim: int = HIDDEN_DIM

    @classmethod
    def frozen(cls, stage: str) -> StageTrainingConfig:
        if stage not in STAGES:
            raise ValueError("unsupported hierarchical retrieval stage")
        return cls(
            stage=stage,
            seed=STAGE_SEEDS[stage],
            epochs=STAGE_EPOCHS[stage],
            batch_size=STAGE_BATCH_SIZES[stage],
        )

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ValueError("unsupported hierarchical retrieval stage")
        if self != StageTrainingConfig.frozen(self.stage):
            raise ValueError("hierarchical retrieval training contract drifted")


class HierarchicalFactorRanker(nn.Module):
    """Stage-specific calibrated set ranker over public observables."""

    def __init__(
        self,
        *,
        context_dim: int,
        item_dim: int,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        if context_dim <= 0 or item_dim <= 0 or hidden_dim <= 0:
            raise ValueError("hierarchical ranker dimensions must be positive")
        self.context_dim = context_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(PARENT_STATE_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(
        self,
        state: mx.array,
        context: mx.array,
        items: mx.array,
        item_mask: mx.array,
    ) -> mx.array:
        state_value = self.state_encoder(state)
        context_value = self.context_encoder(context)
        item_value = self.item_encoder(items)
        state_items = mx.broadcast_to(
            state_value[:, None, :],
            item_value.shape,
        )
        context_items = mx.broadcast_to(
            context_value[:, None, :],
            item_value.shape,
        )
        hidden = self.interaction(
            mx.concatenate(
                [
                    item_value,
                    state_items,
                    context_items,
                    item_value * state_items,
                    item_value * context_items,
                    mx.abs(item_value - state_items),
                    mx.abs(item_value - context_items),
                ],
                axis=-1,
            )
        )
        hidden = hidden * item_mask[..., None]
        denominator = mx.maximum(
            mx.sum(item_mask, axis=1, keepdims=True),
            1,
        )
        mean = mx.sum(hidden, axis=1) / denominator
        maximum = mx.max(
            mx.where(item_mask[..., None], hidden, -1e9),
            axis=1,
        )
        mean_items = mx.broadcast_to(mean[:, None, :], hidden.shape)
        maximum_items = mx.broadcast_to(
            maximum[:, None, :],
            hidden.shape,
        )
        scores = self.output(
            mx.concatenate(
                [hidden, mean_items, maximum_items, hidden - mean_items],
                axis=-1,
            )
        ).reshape(item_mask.shape)
        return mx.where(item_mask, scores, -1e9)


def hierarchical_factor_loss(
    model: HierarchicalFactorRanker,
    state: mx.array,
    context: mx.array,
    items: mx.array,
    item_mask: mx.array,
    expected_rank: mx.array,
    expected_rank_mask: mx.array,
    target: mx.array,
) -> mx.array:
    """Calibrate absolute ranks while preserving local listwise boundaries."""
    scores = model(state, context, items, item_mask)
    regression_target = -mx.log1p(expected_rank)
    delta = mx.abs(scores - regression_target)
    smooth_l1 = mx.where(delta < 1.0, 0.5 * delta * delta, delta - 0.5)
    regression = _masked_query_mean(smooth_l1, expected_rank_mask)

    target_logits = mx.where(
        expected_rank_mask,
        -(expected_rank - 1.0) / TARGET_SCALE,
        -1e9,
    )
    target_probability = mx.softmax(target_logits, axis=-1)
    student_logits = mx.where(
        expected_rank_mask,
        scores / STUDENT_TEMPERATURE,
        -1e9,
    )
    log_probability = student_logits - mx.logsumexp(
        student_logits,
        axis=-1,
        keepdims=True,
    )
    listwise = -mx.sum(
        mx.where(
            expected_rank_mask,
            target_probability * log_probability,
            0.0,
        ),
        axis=-1,
    )
    listwise = mx.mean(listwise)

    negative = item_mask & ~target
    positive_count = mx.sum(target, axis=-1)
    negative_count = mx.sum(negative, axis=-1)
    positive_loss = mx.sum(
        mx.where(target, nn.softplus(-scores), 0.0),
        axis=-1,
    ) / mx.maximum(positive_count, 1)
    negative_loss = mx.sum(
        mx.where(negative, nn.softplus(scores), 0.0),
        axis=-1,
    ) / mx.maximum(negative_count, 1)
    boundary_valid = (positive_count > 0) & (negative_count > 0)
    boundary = mx.sum(
        mx.where(boundary_valid, positive_loss + negative_loss, 0.0)
    ) / mx.maximum(mx.sum(boundary_valid), 1)
    return regression + listwise + boundary


def _masked_query_mean(values: mx.array, mask: mx.array) -> mx.array:
    per_query = mx.sum(mx.where(mask, values, 0.0), axis=-1) / mx.maximum(
        mx.sum(mask, axis=-1),
        1,
    )
    valid = mx.any(mask, axis=-1)
    return mx.sum(mx.where(valid, per_query, 0.0)) / mx.maximum(
        mx.sum(valid),
        1,
    )


def _parent_state(batch: GradedOracleBatch, row: int) -> np.ndarray:
    values = np.concatenate(
        [
            np.asarray(batch.board_entities)[row].reshape(-1),
            np.asarray(batch.board_mask)[row].astype(np.float32).reshape(-1),
            np.asarray(batch.market_entities)[row].reshape(-1),
            np.asarray(batch.market_mask)[row].astype(np.float32).reshape(-1),
            np.asarray(batch.global_features)[row],
            np.asarray(batch.public_supply)[row],
        ]
    ).astype(np.float32, copy=False)
    if values.shape != (PARENT_STATE_DIM,):
        raise AssertionError("hierarchical parent-state dimension drifted")
    return values


def _factor_values(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([actions[:, :34], actions[:, 45:128]], axis=-1),
        actions[:, 34:42],
        actions[:, 42:45],
    )


def _row_keys(values: np.ndarray) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(row, dtype=np.float32).tobytes()
        for row in values
    )


def _staged_public(batch: GradedOracleBatch, row: int) -> np.ndarray:
    count = int(np.sum(np.asarray(batch.candidate_mask)[row]))
    values = np.concatenate(
        [
            np.asarray(batch.staged_market_entities)[row, :count].reshape(
                count,
                -1,
            ),
            np.asarray(batch.staged_market_mask)[row, :count].astype(
                np.float32
            ),
            np.asarray(batch.staged_public_supply)[row, :count],
        ],
        axis=-1,
    )
    if values.shape[-1] != STAGED_PUBLIC_DIM:
        raise AssertionError("staged-public feature dimension drifted")
    return values.astype(np.float32, copy=False)


def _descendant_stats(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != DESCENDANT_VALUE_DIM:
        raise ValueError("descendant statistic input dimension drifted")
    return np.concatenate(
        [
            np.min(values, axis=0),
            np.mean(values, axis=0),
            np.max(values, axis=0),
            np.asarray([math.log1p(len(values)) / 10.0], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def _minimum_rank(
    indices: list[int],
    ranks: np.ndarray,
    rank_mask: np.ndarray,
) -> tuple[float, bool]:
    finite = [float(ranks[index]) for index in indices if rank_mask[index]]
    return (min(finite), True) if finite else (0.0, False)


def _target_keys(
    items: list[tuple[bytes, float, bool]],
    width: int,
) -> set[bytes]:
    finite = [
        (key, rank)
        for key, rank, rank_mask in items
        if rank_mask
    ]
    return {
        key
        for key, _rank in sorted(
            finite,
            key=lambda value: (value[1], value[0]),
        )[: min(width, len(finite))]
    }


def _hash_key(key: bytes) -> np.ndarray:
    return np.frombuffer(blake3.blake3(key).digest(length=16), dtype=np.uint8)


class _CacheAccumulator:
    """Mutable shard builder with deterministic contiguous query storage."""

    def __init__(self) -> None:
        self.group_state: list[np.ndarray] = []
        self.group_id: list[int] = []
        self.phase: list[int] = []
        self.nature_tokens: list[int] = []
        self.selected_index: list[int] = []
        self.group_action_offsets = [0]
        self.action_source_flags: list[np.ndarray] = []
        self.action_hash: list[np.ndarray] = []
        self.action_expected_rank: list[np.ndarray] = []
        self.action_expected_rank_mask: list[np.ndarray] = []
        self.action_r4800_mean: list[np.ndarray] = []
        self.action_r4800_stddev: list[np.ndarray] = []
        self.action_r4800_samples: list[np.ndarray] = []
        self.action_r4800_mask: list[np.ndarray] = []
        self.action_draft_kind: list[np.ndarray] = []
        self.action_maps = {
            stage: [] for stage in STAGES
        }
        self.query_group = {stage: [] for stage in STAGES}
        self.query_context = {stage: [] for stage in STAGES}
        self.query_offsets = {stage: [0] for stage in STAGES}
        self.item_features = {stage: [] for stage in STAGES}
        self.item_rank = {stage: [] for stage in STAGES}
        self.item_rank_mask = {stage: [] for stage in STAGES}
        self.item_target = {stage: [] for stage in STAGES}
        self.item_hash = {stage: [] for stage in STAGES}
        self.prefix_invariants = True
        self.factor_bijections = True

    def add_group(
        self,
        batch: GradedOracleBatch,
        expected_rank: np.ndarray,
        expected_rank_mask: np.ndarray,
    ) -> None:
        row = 0
        count = int(np.sum(np.asarray(batch.candidate_mask)[row]))
        group_index = len(self.group_state)
        actions = np.asarray(batch.action_features)[row, :count].astype(
            np.float32,
            copy=False,
        )
        priors = np.asarray(batch.prior_features)[row, :count].astype(
            np.float32,
            copy=False,
        )
        flags = np.asarray(batch.source_flags)[row, :count].astype(
            np.int32,
            copy=False,
        )
        frontier = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
        eligible_indices = np.flatnonzero(~frontier).tolist()
        ranks = expected_rank[:count].astype(np.float32, copy=False)
        rank_mask = expected_rank_mask[:count].astype(np.bool_, copy=False)
        draft_values, tile_values, wildlife_values = _factor_values(actions)
        draft_keys = _row_keys(draft_values)
        tile_keys = _row_keys(tile_values)
        wildlife_keys = _row_keys(wildlife_values)
        self.factor_bijections &= (
            len(
                set(
                    zip(
                        draft_keys,
                        tile_keys,
                        wildlife_keys,
                        strict=True,
                    )
                )
            )
            == count
        )
        local_geometry = candidate_local_geometry(
            batch.board_entities,
            batch.board_mask,
            batch.action_features,
            batch.candidate_mask,
        )
        mx.eval(local_geometry)
        local = np.asarray(local_geometry)[row, :count].astype(
            np.float32,
            copy=False,
        )
        staged = _staged_public(batch, row)
        descendant_values = np.concatenate(
            [priors, actions[:, 128:140]],
            axis=-1,
        )

        by_draft: dict[bytes, list[int]] = {}
        by_tile: dict[tuple[bytes, bytes], list[int]] = {}
        by_wildlife: dict[tuple[bytes, bytes, bytes], list[int]] = {}
        for index in eligible_indices:
            draft = draft_keys[index]
            tile = tile_keys[index]
            wildlife = wildlife_keys[index]
            by_draft.setdefault(draft, []).append(index)
            by_tile.setdefault((draft, tile), []).append(index)
            by_wildlife.setdefault((draft, tile, wildlife), []).append(index)

        action_maps = {
            stage: np.full(count, -1, dtype=np.int32)
            for stage in STAGES
        }

        draft_items = []
        for key in sorted(by_draft):
            indices = by_draft[key]
            representative = indices[0]
            invariant = np.all(staged[indices] == staged[representative])
            self.prefix_invariants &= bool(invariant)
            rank, finite = _minimum_rank(indices, ranks, rank_mask)
            feature = np.concatenate(
                [
                    draft_values[representative],
                    staged[representative],
                    _descendant_stats(descendant_values[indices]),
                ]
            ).astype(np.float32, copy=False)
            draft_items.append((key, indices, feature, rank, finite))
        targets = _target_keys(
            [(key, rank, finite) for key, _, _, rank, finite in draft_items],
            STAGE_WIDTHS["draft"],
        )
        self.query_group["draft"].append(group_index)
        self.query_context["draft"].append(np.zeros(1, dtype=np.float32))
        for key, indices, feature, rank, finite in draft_items:
            item_index = len(self.item_features["draft"])
            self.item_features["draft"].append(feature)
            self.item_rank["draft"].append(rank)
            self.item_rank_mask["draft"].append(finite)
            self.item_target["draft"].append(key in targets)
            self.item_hash["draft"].append(_hash_key(key))
            action_maps["draft"][indices] = item_index
        self.query_offsets["draft"].append(
            len(self.item_features["draft"])
        )

        for draft in sorted(by_draft):
            draft_indices = by_draft[draft]
            draft_representative = draft_indices[0]
            query_context = np.concatenate(
                [
                    draft_values[draft_representative],
                    staged[draft_representative],
                ]
            ).astype(np.float32, copy=False)
            tile_items = []
            for draft_key, tile in sorted(
                key for key in by_tile if key[0] == draft
            ):
                indices = by_tile[(draft_key, tile)]
                representative = indices[0]
                tile_local = local[representative, :TILE_LOCAL_DIM]
                invariant = np.all(
                    local[indices, :TILE_LOCAL_DIM] == tile_local
                )
                self.prefix_invariants &= bool(invariant)
                rank, finite = _minimum_rank(indices, ranks, rank_mask)
                feature = np.concatenate(
                    [
                        tile_values[representative],
                        tile_local,
                        _descendant_stats(descendant_values[indices]),
                    ]
                ).astype(np.float32, copy=False)
                tile_items.append(
                    (tile, indices, feature, rank, finite)
                )
            tile_targets = _target_keys(
                [
                    (key, rank, finite)
                    for key, _, _, rank, finite in tile_items
                ],
                STAGE_WIDTHS["tile"],
            )
            self.query_group["tile"].append(group_index)
            self.query_context["tile"].append(query_context)
            for key, indices, feature, rank, finite in tile_items:
                item_index = len(self.item_features["tile"])
                self.item_features["tile"].append(feature)
                self.item_rank["tile"].append(rank)
                self.item_rank_mask["tile"].append(finite)
                self.item_target["tile"].append(key in tile_targets)
                self.item_hash["tile"].append(_hash_key(draft + key))
                action_maps["tile"][indices] = item_index
            self.query_offsets["tile"].append(
                len(self.item_features["tile"])
            )

        for draft, tile in sorted(by_tile):
            indices = by_tile[(draft, tile)]
            representative = indices[0]
            query_context = np.concatenate(
                [
                    draft_values[representative],
                    staged[representative],
                    tile_values[representative],
                    local[representative, :TILE_LOCAL_DIM],
                ]
            ).astype(np.float32, copy=False)
            wildlife_items = []
            for draft_key, tile_key, wildlife in sorted(
                key
                for key in by_wildlife
                if key[0] == draft and key[1] == tile
            ):
                wildlife_indices = by_wildlife[
                    (draft_key, tile_key, wildlife)
                ]
                if len(wildlife_indices) != 1:
                    raise ValueError(
                        "complete action factor bijection drifted"
                    )
                index = wildlife_indices[0]
                rank, finite = _minimum_rank(
                    wildlife_indices,
                    ranks,
                    rank_mask,
                )
                feature = np.concatenate(
                    [
                        wildlife_values[index],
                        local[index, TILE_LOCAL_DIM:],
                        descendant_values[index],
                    ]
                ).astype(np.float32, copy=False)
                wildlife_items.append(
                    (wildlife, wildlife_indices, feature, rank, finite)
                )
            wildlife_targets = _target_keys(
                [
                    (key, rank, finite)
                    for key, _, _, rank, finite in wildlife_items
                ],
                STAGE_WIDTHS["wildlife"],
            )
            self.query_group["wildlife"].append(group_index)
            self.query_context["wildlife"].append(query_context)
            for key, wildlife_indices, feature, rank, finite in wildlife_items:
                item_index = len(self.item_features["wildlife"])
                self.item_features["wildlife"].append(feature)
                self.item_rank["wildlife"].append(rank)
                self.item_rank_mask["wildlife"].append(finite)
                self.item_target["wildlife"].append(
                    key in wildlife_targets
                )
                self.item_hash["wildlife"].append(
                    _hash_key(draft + tile + key)
                )
                action_maps["wildlife"][wildlife_indices] = item_index
            self.query_offsets["wildlife"].append(
                len(self.item_features["wildlife"])
            )

        for stage in STAGES:
            if np.any(action_maps[stage][~frontier] < 0):
                raise ValueError(
                    f"{stage} mapping omitted an eligible action"
                )
            self.action_maps[stage].append(action_maps[stage])
        self.group_state.append(_parent_state(batch, row))
        self.group_id.append(
            int(np.asarray(batch.group_id)[row]) & ((1 << 64) - 1)
        )
        self.phase.append(int(np.asarray(batch.phase)[row]))
        self.nature_tokens.append(
            int(np.asarray(batch.active_nature_tokens)[row])
        )
        self.selected_index.append(
            int(np.asarray(batch.selected_index)[row])
        )
        self.group_action_offsets.append(
            self.group_action_offsets[-1] + count
        )
        self.action_source_flags.append(flags)
        self.action_hash.append(np.asarray(batch.action_hash)[row, :count])
        self.action_expected_rank.append(ranks)
        self.action_expected_rank_mask.append(rank_mask)
        self.action_r4800_mean.append(
            np.asarray(batch.r4800_mean)[row, :count]
        )
        self.action_r4800_stddev.append(
            np.asarray(batch.r4800_stddev)[row, :count]
        )
        self.action_r4800_samples.append(
            np.asarray(batch.r4800_samples)[row, :count]
        )
        self.action_r4800_mask.append(
            np.asarray(batch.r4800_mask)[row, :count]
        )
        self.action_draft_kind.append(
            np.asarray(batch.draft_kind)[row, :count]
        )

    def arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "group_state": np.stack(self.group_state).astype(np.float32),
            "group_id": np.asarray(self.group_id, dtype=np.uint64),
            "phase": np.asarray(self.phase, dtype=np.int8),
            "nature_tokens": np.asarray(
                self.nature_tokens,
                dtype=np.int8,
            ),
            "selected_index": np.asarray(
                self.selected_index,
                dtype=np.int32,
            ),
            "group_action_offsets": np.asarray(
                self.group_action_offsets,
                dtype=np.int64,
            ),
            "action_source_flags": np.concatenate(
                self.action_source_flags
            ).astype(np.int32),
            "action_hash": np.concatenate(self.action_hash).astype(np.uint8),
            "action_expected_rank": np.concatenate(
                self.action_expected_rank
            ).astype(np.float32),
            "action_expected_rank_mask": np.concatenate(
                self.action_expected_rank_mask
            ).astype(np.bool_),
            "action_r4800_mean": np.concatenate(
                self.action_r4800_mean
            ).astype(np.float32),
            "action_r4800_stddev": np.concatenate(
                self.action_r4800_stddev
            ).astype(np.float32),
            "action_r4800_samples": np.concatenate(
                self.action_r4800_samples
            ).astype(np.float32),
            "action_r4800_mask": np.concatenate(
                self.action_r4800_mask
            ).astype(np.bool_),
            "action_draft_kind": np.concatenate(
                self.action_draft_kind
            ).astype(np.int8),
        }
        for stage in STAGES:
            arrays[f"{stage}_action_item"] = np.concatenate(
                self.action_maps[stage]
            ).astype(np.int32)
            arrays[f"{stage}_query_group"] = np.asarray(
                self.query_group[stage],
                dtype=np.int32,
            )
            arrays[f"{stage}_query_context"] = np.stack(
                self.query_context[stage]
            ).astype(np.float32)
            arrays[f"{stage}_query_offsets"] = np.asarray(
                self.query_offsets[stage],
                dtype=np.int64,
            )
            arrays[f"{stage}_item_features"] = np.stack(
                self.item_features[stage]
            ).astype(np.float32)
            arrays[f"{stage}_item_rank"] = np.asarray(
                self.item_rank[stage],
                dtype=np.float32,
            )
            arrays[f"{stage}_item_rank_mask"] = np.asarray(
                self.item_rank_mask[stage],
                dtype=np.bool_,
            )
            arrays[f"{stage}_item_target"] = np.asarray(
                self.item_target[stage],
                dtype=np.bool_,
            )
            arrays[f"{stage}_item_hash"] = np.stack(
                self.item_hash[stage]
            ).astype(np.uint8)
        return arrays


def build_cache_shard(
    *,
    dataset_root: Path,
    expected_rank_cache_root: Path,
    shard_index: int,
    output_path: Path,
) -> dict[str, Any]:
    """Export one independently reproducible source-shard factor cache."""
    started = time.perf_counter()
    dataset = Scale16ExpectedRankDataset(
        dataset_root,
        expected_rank_cache_root,
    )
    if not 0 <= shard_index < len(dataset.base.shards):
        raise ValueError("factor-cache shard index is out of range")
    shard = dataset.base.shards[shard_index]
    accumulator = _CacheAccumulator()
    raw = shard.bytes()
    for ref in shard.groups:
        batch = decode_graded_oracle_groups(raw, (ref,))
        ranks, rank_mask = dataset.cache.ranks_for_batch(batch)
        accumulator.add_group(
            batch,
            np.asarray(ranks)[0],
            np.asarray(rank_mask)[0],
        )
    arrays = accumulator.arrays()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output_path)
    usage = _resource_usage()
    report = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "split": dataset.split,
        "shard_index": shard_index,
        "source_file": shard.path.name,
        "source_blake3": checksum(shard.path),
        "dataset_manifest_blake3": checksum(
            dataset.root / "dataset.json"
        ),
        "expected_rank_manifest_blake3": checksum(
            expected_rank_cache_root / "manifest.json"
        ),
        "cache_file": output_path.name,
        "cache_blake3": checksum(output_path),
        "cache_bytes": output_path.stat().st_size,
        "groups": len(accumulator.group_id),
        "candidates": accumulator.group_action_offsets[-1],
        "queries": {
            stage: len(accumulator.query_group[stage])
            for stage in STAGES
        },
        "items": {
            stage: len(accumulator.item_features[stage])
            for stage in STAGES
        },
        "maximum_query_width": {
            stage: int(
                np.max(
                    np.diff(
                        np.asarray(
                            accumulator.query_offsets[stage],
                            dtype=np.int64,
                        )
                    )
                )
            )
            for stage in STAGES
        },
        "factor_bijections": accumulator.factor_bijections,
        "prefix_invariants": accumulator.prefix_invariants,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **usage,
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    report["scientific_blake3"] = _scientific_blake3(report)
    _write_json_atomic(output_path.with_suffix(".json"), report)
    return report


class HierarchicalFactorCache:
    """Verified collection of immutable factor-cache shards."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or self.manifest.get("cache_schema") != CACHE_SCHEMA
            or self.manifest.get("experiment_id") != EXPERIMENT_ID
        ):
            raise ValueError("unsupported hierarchical factor cache")
        self.split = str(self.manifest["split"])
        self.group_count = int(self.manifest["groups"])
        self.candidate_count = int(self.manifest["candidates"])
        self.shards = tuple(self.manifest["shards"])
        for entry in self.shards:
            path = self.root / entry["cache_file"]
            if checksum(path) != entry["cache_blake3"]:
                raise ValueError("hierarchical factor cache checksum mismatch")

    def iter_shards(self) -> Iterator[dict[str, np.ndarray]]:
        for entry in self.shards:
            with np.load(self.root / entry["cache_file"]) as loaded:
                yield {name: loaded[name] for name in loaded.files}


def combine_cache_shards(
    *,
    reports: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    """Combine independently exported shards without rewriting payloads."""
    if not reports:
        raise ValueError("factor cache combine requires shard reports")
    values = [json.loads(path.read_text()) for path in reports]
    if any(
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("cache_schema") != CACHE_SCHEMA
        for value in values
    ):
        raise ValueError("factor cache shard report identity drifted")
    splits = {str(value["split"]) for value in values}
    if len(splits) != 1:
        raise ValueError("factor cache combine cannot mix splits")
    indices = [int(value["shard_index"]) for value in values]
    if indices != sorted(set(indices)):
        raise ValueError("factor cache shard indices are not unique and sorted")
    output_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for report_path, value in zip(reports, values, strict=True):
        source = report_path.with_suffix(".npz")
        destination = output_root / source.name
        if source.resolve() != destination.resolve():
            if destination.exists():
                if checksum(destination) != checksum(source):
                    raise ValueError("factor cache destination collision")
            else:
                os.link(source, destination)
        entries.append(
            {
                "shard_index": int(value["shard_index"]),
                "cache_file": destination.name,
                "cache_blake3": str(value["cache_blake3"]),
                "cache_bytes": int(value["cache_bytes"]),
                "groups": int(value["groups"]),
                "candidates": int(value["candidates"]),
                "queries": value["queries"],
                "items": value["items"],
                "maximum_query_width": value["maximum_query_width"],
                "source_file": value["source_file"],
                "source_blake3": value["source_blake3"],
                "scientific_blake3": value["scientific_blake3"],
            }
        )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "split": next(iter(splits)),
        "dataset_manifest_blake3": values[0]["dataset_manifest_blake3"],
        "expected_rank_manifest_blake3": values[0][
            "expected_rank_manifest_blake3"
        ],
        "groups": sum(int(value["groups"]) for value in values),
        "candidates": sum(int(value["candidates"]) for value in values),
        "queries": {
            stage: sum(
                int(value["queries"][stage]) for value in values
            )
            for stage in STAGES
        },
        "items": {
            stage: sum(int(value["items"][stage]) for value in values)
            for stage in STAGES
        },
        "maximum_query_width": {
            stage: max(
                int(value["maximum_query_width"][stage])
                for value in values
            )
            for stage in STAGES
        },
        "all_factor_bijections": all(
            bool(value["factor_bijections"]) for value in values
        ),
        "all_prefix_invariants": all(
            bool(value["prefix_invariants"]) for value in values
        ),
        "shards": entries,
        "test_split_opened": False,
    }
    manifest["payload_blake3"] = _scientific_blake3(manifest)
    _write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def _query_batches(
    arrays: dict[str, np.ndarray],
    *,
    stage: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, ...]]:
    offsets = arrays[f"{stage}_query_offsets"]
    query_count = len(offsets) - 1
    order = np.arange(query_count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, query_count, batch_size):
        selected = order[start : start + batch_size]
        widths = offsets[selected + 1] - offsets[selected]
        maximum = int(np.max(widths))
        item_dim = STAGE_ITEM_DIMS[stage]
        items = np.zeros((len(selected), maximum, item_dim), dtype=np.float32)
        item_mask = np.zeros((len(selected), maximum), dtype=np.bool_)
        ranks = np.zeros((len(selected), maximum), dtype=np.float32)
        rank_mask = np.zeros((len(selected), maximum), dtype=np.bool_)
        target = np.zeros((len(selected), maximum), dtype=np.bool_)
        for row, query_index in enumerate(selected):
            left = int(offsets[query_index])
            right = int(offsets[query_index + 1])
            width = right - left
            items[row, :width] = arrays[f"{stage}_item_features"][
                left:right
            ]
            item_mask[row, :width] = True
            ranks[row, :width] = arrays[f"{stage}_item_rank"][left:right]
            rank_mask[row, :width] = arrays[
                f"{stage}_item_rank_mask"
            ][left:right]
            target[row, :width] = arrays[f"{stage}_item_target"][
                left:right
            ]
        group_indices = arrays[f"{stage}_query_group"][selected]
        yield (
            arrays["group_state"][group_indices],
            arrays[f"{stage}_query_context"][selected],
            items,
            item_mask,
            ranks,
            rank_mask,
            target,
        )


def build_stage_model(stage: str) -> HierarchicalFactorRanker:
    if stage not in STAGES:
        raise ValueError("unsupported hierarchical retrieval stage")
    return HierarchicalFactorRanker(
        context_dim=STAGE_CONTEXT_DIMS[stage],
        item_dim=STAGE_ITEM_DIMS[stage],
    )


StageLoss = Callable[..., mx.array]
StageSelectionKey = Callable[[dict[str, Any]], tuple[float, ...]]
EpochLearningRate = Callable[[int, int], float]


def calibrated_stage_selection_key(
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    """Select calibrated checkpoints under the frozen ADR 0115 contract."""
    return (
        float(metrics["target_factor_recall"]),
        float(metrics["exact_query_fraction"]),
        -float(metrics["rank_mean_absolute_error"]),
    )


def membership_stage_selection_key(
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    """Select target-membership checkpoints without a rank-error tiebreak."""
    return (
        float(metrics["target_factor_recall"]),
        float(metrics["exact_query_fraction"]),
    )


def train_stage(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
    config: StageTrainingConfig,
) -> dict[str, Any]:
    """Train exactly one stage and select its checkpoint on train metrics."""
    config.validate()
    return train_stage_with_loss(
        train_cache_root=train_cache_root,
        validation_cache_root=validation_cache_root,
        output_root=output_root,
        config=config,
        loss_function=hierarchical_factor_loss,
        selection_key=calibrated_stage_selection_key,
        experiment_id=EXPERIMENT_ID,
        report_metadata=None,
    )


def train_stage_with_loss(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
    config: StageTrainingConfig,
    loss_function: StageLoss,
    selection_key: StageSelectionKey,
    experiment_id: str,
    report_metadata: dict[str, Any] | None,
    epoch_learning_rate: EpochLearningRate | None = None,
) -> dict[str, Any]:
    """Train one stage under an explicitly supplied frozen objective."""
    if output_root.exists():
        raise ValueError("hierarchical stage output already exists")
    if not experiment_id:
        raise ValueError("hierarchical stage experiment id is empty")
    if (
        config.stage not in STAGES
        or config.seed < 0
        or config.epochs <= 0
        or config.batch_size <= 0
        or config.learning_rate <= 0
        or config.weight_decay < 0
        or config.hidden_dim != HIDDEN_DIM
    ):
        raise ValueError("hierarchical stage training config is invalid")
    allocator = configure_mlx_memory()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("hierarchical stage cache split mismatch")
    mx.random.seed(config.seed)
    model = build_stage_model(config.stage)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, loss_function)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    finite_training = True
    for epoch in range(1, config.epochs + 1):
        if epoch_learning_rate is not None:
            learning_rate = float(epoch_learning_rate(epoch, config.epochs))
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("epoch learning-rate schedule returned an invalid value")
            optimizer.learning_rate = learning_rate
        model.train()
        epoch_loss = 0.0
        batches = 0
        for shard_index, arrays in enumerate(train_cache.iter_shards()):
            for values in _query_batches(
                arrays,
                stage=config.stage,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + epoch * 1000 + shard_index,
            ):
                loss, gradients = loss_and_grad(
                    model,
                    *(mx.array(value) for value in values),
                )
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state, loss)
                loss_value = float(loss.item())
                finite_training &= math.isfinite(loss_value) and _tree_finite(
                    model.parameters()
                ) and _tree_finite(optimizer.state)
                if not finite_training:
                    raise RuntimeError(
                        "hierarchical stage training became nonfinite"
                    )
                epoch_loss += loss_value
                batches += 1
        train_metrics = evaluate_stage(model, train_cache, config.stage)
        event = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "train": train_metrics,
        }
        if epoch_learning_rate is not None:
            event["learning_rate"] = float(optimizer.learning_rate.item())
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        key = selection_key(train_metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            mx.save_safetensors(
                str(output_root / "best.safetensors"),
                dict(tree_flatten(model.parameters())),
            )
            _write_json_atomic(output_root / "best.json", event)
        mx.clear_cache()
    model.load_weights(str(output_root / "best.safetensors"))
    mx.eval(model.parameters())
    train_metrics = evaluate_stage(model, train_cache, config.stage)
    validation_metrics = evaluate_stage(
        model,
        validation_cache,
        config.stage,
    )
    usage = _resource_usage()
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "host": _host(),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "parameter_count": sum(
            int(value.size)
            for _name, value in tree_flatten(model.parameters())
        ),
        "weights_blake3": checksum(output_root / "best.safetensors"),
        "train_cache_payload_blake3": train_cache.manifest[
            "payload_blake3"
        ],
        "validation_cache_payload_blake3": validation_cache.manifest[
            "payload_blake3"
        ],
        "train": train_metrics,
        "validation": validation_metrics,
        "finite_training": finite_training,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **usage,
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    if report_metadata:
        collisions = set(report) & set(report_metadata)
        if collisions:
            raise ValueError(
                "hierarchical stage report metadata collides with "
                f"{sorted(collisions)}"
            )
        report.update(report_metadata)
    report["scientific_blake3"] = _scientific_blake3(report)
    _write_json_atomic(output_root / "report.json", report)
    return report


def evaluate_stage(
    model: HierarchicalFactorRanker,
    cache: HierarchicalFactorCache,
    stage: str,
) -> dict[str, Any]:
    model.eval()
    target_total = 0
    target_hits = 0
    exact = 0
    queries = 0
    items = 0
    absolute_error = 0.0
    ranked_items = 0
    finite = True
    for arrays in cache.iter_shards():
        scores = score_stage_shard(model, arrays, stage)
        finite &= bool(np.all(np.isfinite(scores)))
        offsets = arrays[f"{stage}_query_offsets"]
        targets = arrays[f"{stage}_item_target"]
        rank_mask = arrays[f"{stage}_item_rank_mask"]
        ranks = arrays[f"{stage}_item_rank"]
        for left, right in pairwise(offsets):
            left = int(left)
            right = int(right)
            width = min(STAGE_WIDTHS[stage], right - left)
            selected = sorted(
                range(left, right),
                key=lambda index: (
                    -float(scores[index]),
                    index,
                ),
            )[:width]
            quota = int(np.sum(targets[left:right]))
            hits = int(np.sum(targets[selected]))
            target_total += quota
            target_hits += hits
            exact += int(hits == quota)
            queries += 1
        finite_indices = np.flatnonzero(rank_mask)
        absolute_error += float(
            np.sum(
                np.abs(
                    scores[finite_indices]
                    + np.log1p(ranks[finite_indices])
                )
            )
        )
        ranked_items += len(finite_indices)
        items += len(scores)
    return {
        "queries": queries,
        "items": items,
        "target_factors": target_total,
        "target_hits": target_hits,
        "target_factor_recall": target_hits / max(target_total, 1),
        "exact_query_fraction": exact / max(queries, 1),
        "rank_mean_absolute_error": absolute_error / max(ranked_items, 1),
        "all_scores_finite": finite,
        "all_queries_scored_once": queries
        == int(cache.manifest["queries"][stage]),
        "all_items_scored_once": items
        == int(cache.manifest["items"][stage]),
    }


def score_stage_shard(
    model: HierarchicalFactorRanker,
    arrays: dict[str, np.ndarray],
    stage: str,
) -> np.ndarray:
    scores = np.empty(
        len(arrays[f"{stage}_item_features"]),
        dtype=np.float32,
    )
    model.eval()
    offsets = arrays[f"{stage}_query_offsets"]
    cursor = 0
    for start in range(0, len(offsets) - 1, STAGE_BATCH_SIZES[stage]):
        selected = np.arange(
            start,
            min(start + STAGE_BATCH_SIZES[stage], len(offsets) - 1),
        )
        widths = offsets[selected + 1] - offsets[selected]
        maximum = int(np.max(widths))
        items = np.zeros(
            (len(selected), maximum, STAGE_ITEM_DIMS[stage]),
            dtype=np.float32,
        )
        mask = np.zeros((len(selected), maximum), dtype=np.bool_)
        for row, query_index in enumerate(selected):
            left = int(offsets[query_index])
            right = int(offsets[query_index + 1])
            width = right - left
            items[row, :width] = arrays[f"{stage}_item_features"][left:right]
            mask[row, :width] = True
        groups = arrays[f"{stage}_query_group"][selected]
        output = model(
            mx.array(arrays["group_state"][groups]),
            mx.array(arrays[f"{stage}_query_context"][selected]),
            mx.array(items),
            mx.array(mask),
        )
        mx.eval(output)
        values = np.asarray(output)
        for row, width in enumerate(widths):
            width = int(width)
            scores[cursor : cursor + width] = values[row, :width]
            cursor += width
    if cursor != len(scores):
        raise AssertionError("hierarchical stage score coverage drifted")
    return scores


@dataclass
class _SelectionAccumulator:
    groups: int = 0
    target_slots: int = 0
    target_hits: int = 0
    exact_sets: int = 0
    winner_hits: int = 0
    regret: float = 0.0

    def add(
        self,
        *,
        retained: np.ndarray,
        target: np.ndarray,
        source_flags: np.ndarray,
        winner: int,
        r4800_mean: np.ndarray,
        r4800_mask: np.ndarray,
    ) -> None:
        nonfrontier = retained[
            (
                source_flags[retained]
                & GRADED_SOURCE_CHAMPION_FRONTIER
            )
            == 0
        ]
        quota = int(np.sum(target))
        hits = int(np.sum(target[nonfrontier]))
        self.groups += 1
        self.target_slots += quota
        self.target_hits += hits
        self.exact_sets += int(hits == quota)
        self.winner_hits += int(winner in retained)
        retained_labeled = retained[r4800_mask[retained]]
        if np.any(r4800_mask) and len(retained_labeled):
            self.regret += max(
                0.0,
                float(np.max(r4800_mean[r4800_mask]))
                - float(np.max(r4800_mean[retained_labeled])),
            )

    def report(self) -> dict[str, float | int | None]:
        if not self.groups:
            return {
                "groups": 0,
                "target_positive_recall": None,
                "target_set_exact_fraction": None,
                "r4800_winner_retention": None,
                "mean_retained_r4800_regret": None,
            }
        return {
            "groups": self.groups,
            "target_slots": self.target_slots,
            "target_hits": self.target_hits,
            "target_positive_recall": self.target_hits
            / max(self.target_slots, 1),
            "target_set_exact_fraction": self.exact_sets / self.groups,
            "r4800_winner_retention": self.winner_hits / self.groups,
            "mean_retained_r4800_regret": self.regret / self.groups,
        }


def _selected_stage_items(
    *,
    scores: np.ndarray,
    offsets: np.ndarray,
    width: int,
) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=np.bool_)
    for left, right in pairwise(offsets):
        left = int(left)
        right = int(right)
        ranking = sorted(
            range(left, right),
            key=lambda index: (
                -float(scores[index]),
                index,
            ),
        )
        selected[ranking[: min(width, right - left)]] = True
    return selected


def _group_target(
    *,
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
    source_flags: np.ndarray,
    action_hash: np.ndarray,
) -> np.ndarray:
    count = len(expected_rank)
    return build_expected_rank_target_mask(
        expected_rank=expected_rank.reshape(1, count),
        expected_rank_mask=expected_rank_mask.reshape(1, count),
        source_flags=source_flags.reshape(1, count),
        candidate_mask=np.ones((1, count), dtype=np.bool_),
        action_hashes=action_hash.reshape(1, count, -1),
    )[0]


def _evaluate_integrated_split(
    *,
    cache: HierarchicalFactorCache,
    models: dict[str, HierarchicalFactorRanker],
) -> dict[str, Any]:
    learned = _SelectionAccumulator()
    proposal = _SelectionAccumulator()
    learned_phases = {
        0: _SelectionAccumulator(),
        1: _SelectionAccumulator(),
        2: _SelectionAccumulator(),
    }
    proposal_phases = {
        0: _SelectionAccumulator(),
        1: _SelectionAccumulator(),
        2: _SelectionAccumulator(),
    }
    learned_subsets = {
        "nature_token_available": _SelectionAccumulator(),
        "independent_draft_winner": _SelectionAccumulator(),
    }
    proposal_subsets = {
        "nature_token_available": _SelectionAccumulator(),
        "independent_draft_winner": _SelectionAccumulator(),
    }
    proposal_counts: list[int] = []
    candidates = 0
    finite = True
    for arrays in cache.iter_shards():
        stage_scores = {
            stage: score_stage_shard(models[stage], arrays, stage)
            for stage in STAGES
        }
        stage_selected = {
            stage: _selected_stage_items(
                scores=stage_scores[stage],
                offsets=arrays[f"{stage}_query_offsets"],
                width=STAGE_WIDTHS[stage],
            )
            for stage in STAGES
        }
        action_offsets = arrays["group_action_offsets"]
        for group_index, (left, right) in enumerate(
            pairwise(action_offsets)
        ):
            left = int(left)
            right = int(right)
            count = right - left
            flags = arrays["action_source_flags"][left:right]
            frontier = (
                flags & GRADED_SOURCE_CHAMPION_FRONTIER
            ) != 0
            maps = {
                stage: arrays[f"{stage}_action_item"][left:right]
                for stage in STAGES
            }
            eligible = ~frontier
            passing = eligible.copy()
            for stage in STAGES:
                valid = maps[stage] >= 0
                passing &= valid
                passing[valid] &= stage_selected[stage][maps[stage][valid]]
            proposal_indices = np.flatnonzero(frontier | passing).astype(
                np.int32
            )
            proposal_counts.append(len(proposal_indices))
            combined_scores = np.zeros(count, dtype=np.float32)
            for stage in STAGES:
                valid = maps[stage] >= 0
                combined_scores[valid] += stage_scores[stage][
                    maps[stage][valid]
                ]
            local_retained = frontier_anchored_retained_indices(
                scores=combined_scores[proposal_indices],
                source_flags=flags[proposal_indices],
                action_hashes=arrays["action_hash"][
                    left:right
                ][proposal_indices],
            )
            learned_retained = proposal_indices[local_retained]
            ranks = arrays["action_expected_rank"][left:right]
            rank_mask = arrays["action_expected_rank_mask"][left:right]
            oracle_scores = np.where(
                rank_mask[proposal_indices],
                -ranks[proposal_indices],
                -1e9,
            )
            oracle_local = frontier_anchored_retained_indices(
                scores=oracle_scores,
                source_flags=flags[proposal_indices],
                action_hashes=arrays["action_hash"][
                    left:right
                ][proposal_indices],
            )
            oracle_retained = proposal_indices[oracle_local]
            target = _group_target(
                expected_rank=ranks,
                expected_rank_mask=rank_mask,
                source_flags=flags,
                action_hash=arrays["action_hash"][left:right],
            )
            winner = int(arrays["selected_index"][group_index])
            r4800_mean = arrays["action_r4800_mean"][left:right]
            r4800_mask = arrays["action_r4800_mask"][left:right]
            kwargs = {
                "target": target,
                "source_flags": flags,
                "winner": winner,
                "r4800_mean": r4800_mean,
                "r4800_mask": r4800_mask,
            }
            learned.add(retained=learned_retained, **kwargs)
            proposal.add(retained=oracle_retained, **kwargs)
            phase = int(arrays["phase"][group_index])
            learned_phases[phase].add(
                retained=learned_retained,
                **kwargs,
            )
            proposal_phases[phase].add(
                retained=oracle_retained,
                **kwargs,
            )
            if int(arrays["nature_tokens"][group_index]) > 0:
                learned_subsets["nature_token_available"].add(
                    retained=learned_retained,
                    **kwargs,
                )
                proposal_subsets["nature_token_available"].add(
                    retained=oracle_retained,
                    **kwargs,
                )
            if (
                int(
                    arrays["action_draft_kind"][
                        left + winner
                    ]
                )
                == 1
            ):
                learned_subsets["independent_draft_winner"].add(
                    retained=learned_retained,
                    **kwargs,
                )
                proposal_subsets["independent_draft_winner"].add(
                    retained=oracle_retained,
                    **kwargs,
                )
            finite &= bool(np.all(np.isfinite(combined_scores)))
            candidates += count
    proposal_values = np.asarray(proposal_counts, dtype=np.float64)
    phase_names = {0: "early", 1: "middle", 2: "late"}
    return {
        "groups": learned.groups,
        "candidates": candidates,
        "all_groups_scored_once": learned.groups == cache.group_count,
        "all_candidates_scored_once": candidates == cache.candidate_count,
        "all_scores_finite": finite,
        "mean_proposal_count": float(np.mean(proposal_values)),
        "p90_proposal_count": float(
            np.quantile(proposal_values, 0.90, method="higher")
        ),
        "p99_proposal_count": float(
            np.quantile(proposal_values, 0.99, method="higher")
        ),
        "maximum_proposal_count": int(np.max(proposal_values)),
        "learned_top64": learned.report(),
        "oracle_inside_learned_proposal": proposal.report(),
        "learned_phase": {
            phase_names[index]: accumulator.report()
            for index, accumulator in learned_phases.items()
        },
        "proposal_phase": {
            phase_names[index]: accumulator.report()
            for index, accumulator in proposal_phases.items()
        },
        "learned_subsets": {
            name: accumulator.report()
            for name, accumulator in learned_subsets.items()
        },
        "proposal_subsets": {
            name: accumulator.report()
            for name, accumulator in proposal_subsets.items()
        },
    }


def evaluate_integrated(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    weights: dict[str, Path],
) -> dict[str, Any]:
    """Evaluate learned proposal and deployable selector on both open splits."""
    allocator = configure_mlx_memory()
    if set(weights) != set(STAGES):
        raise ValueError("integration requires exactly three stage weights")
    models = {
        stage: load_stage_model(stage, weights[stage])
        for stage in STAGES
    }
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    scientific = {
        "weights_blake3": {
            stage: checksum(weights[stage]) for stage in STAGES
        },
        "train_cache_payload_blake3": train.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation.manifest[
            "payload_blake3"
        ],
        "train": _evaluate_integrated_split(cache=train, models=models),
        "validation": _evaluate_integrated_split(
            cache=validation,
            models=models,
        ),
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
            **_resource_usage(),
        },
    }


def audit_cache_split(cache: HierarchicalFactorCache) -> dict[str, Any]:
    """Reconstruct the ADR 0114 oracle solely from one frozen cache."""
    accumulator = _SelectionAccumulator()
    phase_accumulators = {
        0: _SelectionAccumulator(),
        1: _SelectionAccumulator(),
        2: _SelectionAccumulator(),
    }
    proposal_counts: list[int] = []
    groups = 0
    candidates = 0
    mappings_complete = True
    target_labels_exact = True
    for arrays in cache.iter_shards():
        stage_selected: dict[str, np.ndarray] = {}
        for stage in STAGES:
            rank = arrays[f"{stage}_item_rank"]
            rank_mask = arrays[f"{stage}_item_rank_mask"]
            offsets = arrays[f"{stage}_query_offsets"]
            recomputed = np.zeros(len(rank), dtype=np.bool_)
            for left, right in pairwise(offsets):
                left = int(left)
                right = int(right)
                finite = [
                    index
                    for index in range(left, right)
                    if rank_mask[index]
                ]
                selected = sorted(
                    finite,
                    key=lambda index: (
                        float(rank[index]),
                        index,
                    ),
                )[: STAGE_WIDTHS[stage]]
                recomputed[selected] = True
            target_labels_exact &= bool(
                np.array_equal(
                    recomputed,
                    arrays[f"{stage}_item_target"],
                )
            )
            stage_selected[stage] = arrays[
                f"{stage}_item_target"
            ].astype(np.bool_, copy=False)
        action_offsets = arrays["group_action_offsets"]
        for group_index, (left, right) in enumerate(
            pairwise(action_offsets)
        ):
            left = int(left)
            right = int(right)
            count = right - left
            flags = arrays["action_source_flags"][left:right]
            frontier = (
                flags & GRADED_SOURCE_CHAMPION_FRONTIER
            ) != 0
            eligible = ~frontier
            passing = eligible.copy()
            for stage in STAGES:
                mapping = arrays[f"{stage}_action_item"][left:right]
                valid = mapping >= 0
                mappings_complete &= bool(np.all(valid[eligible]))
                passing &= valid
                passing[valid] &= stage_selected[stage][mapping[valid]]
            proposal_indices = np.flatnonzero(frontier | passing).astype(
                np.int32
            )
            proposal_counts.append(len(proposal_indices))
            ranks = arrays["action_expected_rank"][left:right]
            rank_mask = arrays["action_expected_rank_mask"][left:right]
            local = frontier_anchored_retained_indices(
                scores=np.where(
                    rank_mask[proposal_indices],
                    -ranks[proposal_indices],
                    -1e9,
                ),
                source_flags=flags[proposal_indices],
                action_hashes=arrays["action_hash"][
                    left:right
                ][proposal_indices],
            )
            retained = proposal_indices[local]
            target = _group_target(
                expected_rank=ranks,
                expected_rank_mask=rank_mask,
                source_flags=flags,
                action_hash=arrays["action_hash"][left:right],
            )
            kwargs = {
                "retained": retained,
                "target": target,
                "source_flags": flags,
                "winner": int(arrays["selected_index"][group_index]),
                "r4800_mean": arrays["action_r4800_mean"][left:right],
                "r4800_mask": arrays["action_r4800_mask"][left:right],
            }
            accumulator.add(**kwargs)
            phase_accumulators[int(arrays["phase"][group_index])].add(
                **kwargs
            )
            groups += 1
            candidates += count
    proposal_values = np.asarray(proposal_counts, dtype=np.float64)
    phase_names = {0: "early", 1: "middle", 2: "late"}
    return {
        "groups": groups,
        "candidates": candidates,
        "all_groups_covered": groups == cache.group_count,
        "all_candidates_covered": candidates == cache.candidate_count,
        "all_eligible_mappings_complete": mappings_complete,
        "all_factor_target_labels_exact": target_labels_exact,
        "target_positive_recall": accumulator.report()[
            "target_positive_recall"
        ],
        "target_set_exact_fraction": accumulator.report()[
            "target_set_exact_fraction"
        ],
        "r4800_winner_retention": accumulator.report()[
            "r4800_winner_retention"
        ],
        "mean_retained_r4800_regret": accumulator.report()[
            "mean_retained_r4800_regret"
        ],
        "mean_proposal_count": float(np.mean(proposal_values)),
        "p99_proposal_count": float(
            np.quantile(proposal_values, 0.99, method="higher")
        ),
        "maximum_proposal_count": int(np.max(proposal_values)),
        "phase": {
            phase_names[index]: value.report()
            for index, value in phase_accumulators.items()
        },
    }


def audit_caches(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Audit combined caches without training or gradients."""
    started = time.perf_counter()
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    scientific = {
        "adr0114_combined_report_blake3": FROZEN_ADR0114_BLAKE3,
        "train_cache_payload_blake3": train.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation.manifest[
            "payload_blake3"
        ],
        "train": audit_cache_split(train),
        "validation": audit_cache_split(validation),
        "training_used": False,
        "gradients_used": False,
        "optimizer_updates_used": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **_resource_usage(),
        },
    }


def hierarchical_retrieval_gates(
    *,
    cache_audit_identical: bool,
    stage_replays_identical: bool,
    stage_reports: dict[str, dict[str, Any]],
    integration: dict[str, Any],
) -> dict[str, bool]:
    """Apply the frozen ADR 0115 pipeline, proposal, and selector gates."""
    audit = integration.get("cache_audit", {})
    scientific = integration["scientific"]
    train = scientific["train"]
    validation = scientific["validation"]
    pipeline = {
        "cache_audits_identical": cache_audit_identical,
        "cache_factor_bijections": bool(
            audit.get("all_factor_bijections", False)
        ),
        "cache_prefix_invariants": bool(
            audit.get("all_prefix_invariants", False)
        ),
        "cache_oracle_reconstruction": bool(
            audit.get("oracle_reconstruction_passed", False)
        ),
        "stage_replays_identical": stage_replays_identical,
        "all_stage_reports_present": set(stage_reports) == set(STAGES),
        "all_stage_training_finite": all(
            bool(report.get("finite_training"))
            and bool(report["train"]["all_scores_finite"])
            and bool(report["validation"]["all_scores_finite"])
            for report in stage_reports.values()
        ),
        "all_stage_resources_pass": all(
            int(report["execution"]["peak_process_rss_bytes"])
            < 4 * 1024**3
            and int(report["execution"]["process_swaps"]) == 0
            for report in stage_reports.values()
        ),
        "integration_coverage": all(
            bool(values["all_groups_scored_once"])
            and bool(values["all_candidates_scored_once"])
            and bool(values["all_scores_finite"])
            for values in (train, validation)
        ),
        "sealed_domains_closed": all(
            not bool(scientific[name])
            for name in (
                "test_split_opened",
                "gameplay_opened",
                "new_teacher_compute_used",
                "external_compute_used",
            )
        ),
    }
    proposal = {
        "proposal_train_recall_above_0_98": (
            float(
                train["oracle_inside_learned_proposal"][
                    "target_positive_recall"
                ]
            )
            > 0.98
        ),
        "proposal_validation_recall_above_0_98": (
            float(
                validation["oracle_inside_learned_proposal"][
                    "target_positive_recall"
                ]
            )
            > 0.98
        ),
        "proposal_train_winner_retention_above_0_98": (
            float(
                train["oracle_inside_learned_proposal"][
                    "r4800_winner_retention"
                ]
            )
            > 0.98
        ),
        "proposal_validation_winner_retention_above_0_98": (
            float(
                validation["oracle_inside_learned_proposal"][
                    "r4800_winner_retention"
                ]
            )
            > 0.98
        ),
        "proposal_mean_count_at_most_2048": (
            float(train["mean_proposal_count"]) <= 2048
            and float(validation["mean_proposal_count"]) <= 2048
        ),
    }
    for name, values in validation["proposal_phase"].items():
        proposal[f"proposal_{name}_recall_at_least_0_97"] = (
            float(values["target_positive_recall"]) >= 0.97
        )
    for name, values in validation["proposal_subsets"].items():
        if int(values["groups"]) >= 20:
            proposal[f"proposal_{name}_recall_at_least_0_95"] = (
                float(values["target_positive_recall"]) >= 0.95
            )

    selector = {
        "selector_train_recall_above_0_98": (
            float(train["learned_top64"]["target_positive_recall"]) > 0.98
        ),
        "selector_validation_recall_above_0_98": (
            float(validation["learned_top64"]["target_positive_recall"])
            > 0.98
        ),
        "selector_train_winner_recall_above_0_98": (
            float(train["learned_top64"]["r4800_winner_retention"]) > 0.98
        ),
        "selector_validation_winner_recall_above_0_98": (
            float(
                validation["learned_top64"]["r4800_winner_retention"]
            )
            > 0.98
        ),
        "selector_train_regret_below_0_15": (
            float(
                train["learned_top64"]["mean_retained_r4800_regret"]
            )
            < 0.15
        ),
        "selector_validation_regret_below_0_15": (
            float(
                validation["learned_top64"][
                    "mean_retained_r4800_regret"
                ]
            )
            < 0.15
        ),
    }
    for name, values in validation["learned_phase"].items():
        selector[f"selector_{name}_winner_recall_at_least_0_97"] = (
            float(values["r4800_winner_retention"]) >= 0.97
        )
        selector[f"selector_{name}_regret_below_0_20"] = (
            float(values["mean_retained_r4800_regret"]) < 0.20
        )
    for name, values in validation["learned_subsets"].items():
        if int(values["groups"]) >= 20:
            selector[f"selector_{name}_winner_recall_at_least_0_95"] = (
                float(values["r4800_winner_retention"]) >= 0.95
            )
            selector[f"selector_{name}_regret_below_0_25"] = (
                float(values["mean_retained_r4800_regret"]) < 0.25
            )
    return {
        **{f"pipeline_{name}": value for name, value in pipeline.items()},
        **proposal,
        **selector,
        "pipeline_passed": all(pipeline.values()),
        "proposal_passed": all(proposal.values()),
        "selector_passed": all(selector.values()),
    }


def classify_hierarchical_retrieval(gates: dict[str, bool]) -> str:
    """Classify ADR 0115 in frozen precedence order."""
    if not gates["pipeline_passed"]:
        return "hierarchical_retrieval_pipeline_invalid"
    if not gates["proposal_passed"]:
        return "hierarchical_proposal_insufficient"
    if not gates["selector_passed"]:
        return "hierarchical_selector_insufficient"
    return "hierarchical_factor_retrieval_sufficient"


def load_stage_model(stage: str, weights: Path) -> HierarchicalFactorRanker:
    model = build_stage_model(stage)
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def replay_stage(
    *,
    stage: str,
    weights: Path,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    allocator = configure_mlx_memory()
    model = load_stage_model(stage, weights)
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    scientific = {
        "stage": stage,
        "weights_blake3": checksum(weights),
        "train_cache_payload_blake3": train.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation.manifest[
            "payload_blake3"
        ],
        "train": evaluate_stage(model, train, stage),
        "validation": evaluate_stage(model, validation, stage),
        "test_split_opened": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "mlx_allocator": allocator,
            **_resource_usage(),
        },
    }


def _stage_error_split(
    *,
    model: HierarchicalFactorRanker,
    cache: HierarchicalFactorCache,
    stage: str,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    for arrays in cache.iter_shards():
        scores = score_stage_shard(model, arrays, stage)
        offsets = arrays[f"{stage}_query_offsets"]
        targets = arrays[f"{stage}_item_target"]
        groups = arrays[f"{stage}_query_group"]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            left = int(left)
            right = int(right)
            width = right - left
            selected = sorted(
                range(left, right),
                key=lambda index: (-float(scores[index]), index),
            )[: min(STAGE_WIDTHS[stage], width)]
            quota = int(np.sum(targets[left:right]))
            hits = int(np.sum(targets[selected]))
            phase = int(arrays["phase"][groups[query_index]])
            width_name = (
                "within_budget"
                if width <= STAGE_WIDTHS[stage]
                else "up_to_2x_budget"
                if width <= STAGE_WIDTHS[stage] * 2
                else "above_2x_budget"
            )
            names = (
                "overall",
                f"phase_{phase}",
                width_name,
                f"phase_{phase}_{width_name}",
            )
            for name in names:
                bucket = buckets.setdefault(
                    name,
                    {
                        "queries": 0,
                        "target_factors": 0,
                        "target_hits": 0,
                        "exact_queries": 0,
                    },
                )
                bucket["queries"] += 1
                bucket["target_factors"] += quota
                bucket["target_hits"] += hits
                bucket["exact_queries"] += int(hits == quota)
    return {
        name: {
            **values,
            "target_factor_recall": values["target_hits"]
            / max(values["target_factors"], 1),
            "exact_query_fraction": values["exact_queries"]
            / max(values["queries"], 1),
        }
        for name, values in sorted(buckets.items())
    }


def analyze_stage_errors(
    *,
    stage: str,
    weights: Path,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Decompose selected stage misses without modifying the treatment."""
    allocator = configure_mlx_memory()
    model = load_stage_model(stage, weights)
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    scientific = {
        "stage": stage,
        "weights_blake3": checksum(weights),
        "train": _stage_error_split(
            model=model,
            cache=train,
            stage=stage,
        ),
        "validation": _stage_error_split(
            model=model,
            cache=validation,
            stage=stage,
        ),
        "test_split_opened": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "mlx_allocator": allocator,
            **_resource_usage(),
        },
    }


def _mixed_stage_ceiling_split(
    *,
    model: HierarchicalFactorRanker,
    cache: HierarchicalFactorCache,
    learned_stage: str,
) -> dict[str, Any]:
    accumulator = _SelectionAccumulator()
    proposal_counts: list[int] = []
    groups = 0
    candidates = 0
    for arrays in cache.iter_shards():
        learned_scores = score_stage_shard(
            model,
            arrays,
            learned_stage,
        )
        selected = {
            stage: (
                _selected_stage_items(
                    scores=learned_scores,
                    offsets=arrays[f"{stage}_query_offsets"],
                    width=STAGE_WIDTHS[stage],
                )
                if stage == learned_stage
                else arrays[f"{stage}_item_target"].astype(
                    np.bool_,
                    copy=False,
                )
            )
            for stage in STAGES
        }
        for group_index, (left, right) in enumerate(
            pairwise(arrays["group_action_offsets"])
        ):
            left = int(left)
            right = int(right)
            flags = arrays["action_source_flags"][left:right]
            frontier = (
                flags & GRADED_SOURCE_CHAMPION_FRONTIER
            ) != 0
            passing = ~frontier
            for stage in STAGES:
                mapping = arrays[f"{stage}_action_item"][left:right]
                valid = mapping >= 0
                passing &= valid
                passing[valid] &= selected[stage][mapping[valid]]
            proposal = np.flatnonzero(frontier | passing).astype(np.int32)
            proposal_counts.append(len(proposal))
            ranks = arrays["action_expected_rank"][left:right]
            rank_mask = arrays["action_expected_rank_mask"][left:right]
            local = frontier_anchored_retained_indices(
                scores=np.where(
                    rank_mask[proposal],
                    -ranks[proposal],
                    -1e9,
                ),
                source_flags=flags[proposal],
                action_hashes=arrays["action_hash"][left:right][proposal],
            )
            retained = proposal[local]
            accumulator.add(
                retained=retained,
                target=_group_target(
                    expected_rank=ranks,
                    expected_rank_mask=rank_mask,
                    source_flags=flags,
                    action_hash=arrays["action_hash"][left:right],
                ),
                source_flags=flags,
                winner=int(arrays["selected_index"][group_index]),
                r4800_mean=arrays["action_r4800_mean"][left:right],
                r4800_mask=arrays["action_r4800_mask"][left:right],
            )
            groups += 1
            candidates += right - left
    proposal_values = np.asarray(proposal_counts, dtype=np.float64)
    return {
        **accumulator.report(),
        "candidates": candidates,
        "all_groups_covered": groups == cache.group_count,
        "all_candidates_covered": candidates == cache.candidate_count,
        "mean_proposal_count": float(np.mean(proposal_values)),
        "p99_proposal_count": float(
            np.quantile(proposal_values, 0.99, method="higher")
        ),
        "maximum_proposal_count": int(np.max(proposal_values)),
    }


def evaluate_mixed_stage_ceiling(
    *,
    stage: str,
    weights: Path,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Measure one learned stage with every other stage held oracle-perfect."""
    allocator = configure_mlx_memory()
    model = load_stage_model(stage, weights)
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    scientific = {
        "learned_stage": stage,
        "weights_blake3": checksum(weights),
        "other_stages_oracle": True,
        "final_selector_oracle": True,
        "train": _mixed_stage_ceiling_split(
            model=model,
            cache=train,
            learned_stage=stage,
        ),
        "validation": _mixed_stage_ceiling_split(
            model=model,
            cache=validation,
            learned_stage=stage,
        ),
        "test_split_opened": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "mlx_allocator": allocator,
            **_resource_usage(),
        },
    }


def _tree_finite(tree: object) -> bool:
    return all(
        bool(mx.all(mx.isfinite(value)).item())
        for _name, value in tree_flatten(tree)
    )


def _resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak *= 1024
    return {
        "peak_process_rss_bytes": peak,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
    }


def _host() -> str:
    host = socket.gethostname().split(".")[0].lower()
    aliases = {
        "johns-mac-mini": "john1",
        "johns-mac-mini-2": "john2",
        "johns-mac-mini-3": "john3",
        "johns-mac-mini-4": "john4",
    }
    return aliases.get(host, host)


def _scientific_blake3(value: dict[str, Any]) -> str:
    excluded = {"host", "execution", "cache_file"}
    scientific = {
        key: item for key, item in value.items() if key not in excluded
    }
    return blake3.blake3(
        json.dumps(
            scientific,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_parser = subparsers.add_parser("cache-shard")
    cache_parser.add_argument("--dataset", type=Path, required=True)
    cache_parser.add_argument("--cache", type=Path, required=True)
    cache_parser.add_argument("--shard-index", type=int, required=True)
    cache_parser.add_argument("--output", type=Path, required=True)

    combine_parser = subparsers.add_parser("combine-cache")
    combine_parser.add_argument("--reports", type=Path, nargs="+", required=True)
    combine_parser.add_argument("--output", type=Path, required=True)

    audit_parser = subparsers.add_parser("audit-cache")
    audit_parser.add_argument("--train-cache", type=Path, required=True)
    audit_parser.add_argument(
        "--validation-cache",
        type=Path,
        required=True,
    )
    audit_parser.add_argument("--output", type=Path, required=True)

    train_parser = subparsers.add_parser("train-stage")
    train_parser.add_argument("--stage", choices=STAGES, required=True)
    train_parser.add_argument("--train-cache", type=Path, required=True)
    train_parser.add_argument("--validation-cache", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay-stage")
    replay_parser.add_argument("--stage", choices=STAGES, required=True)
    replay_parser.add_argument("--weights", type=Path, required=True)
    replay_parser.add_argument("--train-cache", type=Path, required=True)
    replay_parser.add_argument(
        "--validation-cache",
        type=Path,
        required=True,
    )
    replay_parser.add_argument("--output", type=Path, required=True)

    analysis_parser = subparsers.add_parser("analyze-stage")
    analysis_parser.add_argument("--stage", choices=STAGES, required=True)
    analysis_parser.add_argument("--weights", type=Path, required=True)
    analysis_parser.add_argument("--train-cache", type=Path, required=True)
    analysis_parser.add_argument(
        "--validation-cache",
        type=Path,
        required=True,
    )
    analysis_parser.add_argument("--output", type=Path, required=True)

    ceiling_parser = subparsers.add_parser("mixed-stage-ceiling")
    ceiling_parser.add_argument("--stage", choices=STAGES, required=True)
    ceiling_parser.add_argument("--weights", type=Path, required=True)
    ceiling_parser.add_argument("--train-cache", type=Path, required=True)
    ceiling_parser.add_argument(
        "--validation-cache",
        type=Path,
        required=True,
    )
    ceiling_parser.add_argument("--output", type=Path, required=True)

    integration_parser = subparsers.add_parser("evaluate-integrated")
    integration_parser.add_argument(
        "--train-cache",
        type=Path,
        required=True,
    )
    integration_parser.add_argument(
        "--validation-cache",
        type=Path,
        required=True,
    )
    for stage in STAGES:
        integration_parser.add_argument(
            f"--{stage}-weights",
            type=Path,
            required=True,
        )
    integration_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "cache-shard":
        report = build_cache_shard(
            dataset_root=args.dataset,
            expected_rank_cache_root=args.cache,
            shard_index=args.shard_index,
            output_path=args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "combine-cache":
        report = combine_cache_shards(
            reports=args.reports,
            output_root=args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "audit-cache":
        report = audit_caches(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
        )
        _write_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "train-stage":
        report = train_stage(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            output_root=args.output,
            config=StageTrainingConfig.frozen(args.stage),
        )
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "evaluate-integrated":
        report = evaluate_integrated(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            weights={
                stage: getattr(args, f"{stage}_weights")
                for stage in STAGES
            },
        )
        _write_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "analyze-stage":
        report = analyze_stage_errors(
            stage=args.stage,
            weights=args.weights,
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
        )
        _write_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "mixed-stage-ceiling":
        report = evaluate_mixed_stage_ceiling(
            stage=args.stage,
            weights=args.weights,
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
        )
        _write_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return
    report = replay_stage(
        stage=args.stage,
        weights=args.weights,
        train_cache_root=args.train_cache,
        validation_cache_root=args.validation_cache,
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
