"""Direct raw-observable factor-construction probes for ADR 0098."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LOCAL_GEOMETRY_CONTEXT_DIM,
    candidate_local_geometry,
)
from cascadia_mlx.graded_oracle_prepool_context import (
    stable_screen_topk_indices,
)
from cascadia_mlx.model import SetAttentionBlock, _masked_pool

EXPERIMENT_ID = "complete-action-frontier-raw-factor-construction-v1"

COMPLETE_RAW_FLAT = "complete-raw-flat"
EXACT_LOCAL_RELATION = "exact-local-relation"
EXPLICIT_MARKET_TRANSITION = "explicit-market-transition"
FRESH_ENTITY_CROSS = "fresh-entity-cross"
PROBE_KINDS = (
    COMPLETE_RAW_FLAT,
    EXACT_LOCAL_RELATION,
    EXPLICIT_MARKET_TRANSITION,
    FRESH_ENTITY_CROSS,
)
PROBE_SEEDS = {
    COMPLETE_RAW_FLAT: 2026061621,
    EXACT_LOCAL_RELATION: 2026061622,
    EXPLICIT_MARKET_TRANSITION: 2026061623,
    FRESH_ENTITY_CROSS: 2026061624,
}
PROBE_EPOCHS = 20
PROBE_LEARNING_RATE = 3e-4
PROBE_WEIGHT_DECAY = 1e-4
CONSTRUCTION_DIM = 384
ENTITY_HIDDEN_DIM = 192
SCREEN_CONTEXT_WIDTH = 64

PARENT_RAW_DIM = (
    4 * 23 * ENTITY_DIM
    + 4 * 23
    + 4 * ENTITY_DIM
    + 4
    + GLOBAL_DIM
    + GRADED_ORACLE_PUBLIC_SUPPLY_SIZE
)
CANDIDATE_RAW_DIM = (
    GRADED_ORACLE_ACTION_DIM
    + GRADED_ORACLE_PRIOR_DIM
    + 4 * ENTITY_DIM
    + 4
    + GRADED_ORACLE_PUBLIC_SUPPLY_SIZE
)
COMPLETE_RAW_DIM = PARENT_RAW_DIM + CANDIDATE_RAW_DIM
MARKET_TRANSITION_DIM = (
    4 * ENTITY_DIM * 4
    + 4 * 2
    + GRADED_ORACLE_PUBLIC_SUPPLY_SIZE * 4
    + GRADED_ORACLE_ACTION_DIM
    + GRADED_ORACLE_PRIOR_DIM
)


@dataclass(frozen=True)
class RawFactorProbeConfig:
    kind: str
    seed: int
    epochs: int = PROBE_EPOCHS
    learning_rate: float = PROBE_LEARNING_RATE
    weight_decay: float = PROBE_WEIGHT_DECAY

    def validate(self) -> None:
        if self.kind not in PROBE_KINDS:
            raise ValueError("unsupported raw factor-construction probe")
        if (
            self.seed != PROBE_SEEDS[self.kind]
            or self.epochs != PROBE_EPOCHS
            or self.learning_rate != PROBE_LEARNING_RATE
            or self.weight_decay != PROBE_WEIGHT_DECAY
        ):
            raise ValueError("raw factor-construction configuration drifted")


class RawCandidateSetScorer(nn.Module):
    """Shared observable set context for every ADR 0098 constructor."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CONSTRUCTION_DIM * 7, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
            nn.Linear(CONSTRUCTION_DIM, 1),
        )

    def __call__(
        self,
        candidates: mx.array,
        counts: tuple[int, ...],
        screen_rank: np.ndarray,
        action_hash: np.ndarray,
    ) -> mx.array:
        if len(counts) != candidates.shape[0]:
            raise ValueError("candidate count metadata drifted")
        width = candidates.shape[1]
        outputs = []
        for group_index, count in enumerate(counts):
            group = candidates[group_index, :count]
            mean = mx.mean(group, axis=0, keepdims=True)
            maximum = mx.max(group, axis=0, keepdims=True)
            landmarks = stable_screen_topk_indices(
                screen_rank[group_index, :count],
                action_hash[group_index, :count],
                width=SCREEN_CONTEXT_WIDTH,
            )
            landmark_values = group[mx.array(landmarks)]
            landmark_mean = mx.mean(
                landmark_values,
                axis=0,
                keepdims=True,
            )
            landmark_maximum = mx.max(
                landmark_values,
                axis=0,
                keepdims=True,
            )
            context = mx.concatenate(
                [
                    group,
                    mx.broadcast_to(mean, group.shape),
                    mx.broadcast_to(maximum, group.shape),
                    mx.broadcast_to(landmark_mean, group.shape),
                    mx.broadcast_to(landmark_maximum, group.shape),
                    group - landmark_mean,
                    group - landmark_maximum,
                ],
                axis=-1,
            )
            valid_scores = self.network(context).reshape(-1)
            outputs.append(
                mx.concatenate(
                    [
                        valid_scores,
                        mx.zeros((width - count,), dtype=valid_scores.dtype),
                    ]
                )
            )
        return mx.stack(outputs)


class RawFactorProbe(nn.Module):
    """Base class that fixes the shared set scorer."""

    def __init__(self):
        super().__init__()
        self.scorer = RawCandidateSetScorer()

    def encode_candidates(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        raise NotImplementedError

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        candidate_mask: mx.array,
        counts: tuple[int, ...],
        screen_rank: np.ndarray,
        action_hash: np.ndarray,
    ) -> mx.array:
        candidates = self.encode_candidates(
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            public_supply,
            action_features,
            prior_features,
            staged_market_entities,
            staged_market_mask,
            staged_public_supply,
            candidate_mask,
        )
        candidates = candidates * candidate_mask[..., None]
        return self.scorer(
            candidates,
            counts,
            screen_rank,
            action_hash,
        )


class CompleteRawFlatProbe(RawFactorProbe):
    """Unrestricted dense construction from every lossless public input."""

    def __init__(self):
        super().__init__()
        self.constructor = nn.Sequential(
            nn.Linear(COMPLETE_RAW_DIM, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )

    def encode_candidates(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        parent = raw_parent_features(
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            public_supply,
        )
        candidate = raw_candidate_features(
            action_features,
            prior_features,
            staged_market_entities,
            staged_market_mask,
            staged_public_supply,
        )
        repeated_parent = mx.broadcast_to(
            parent[:, None, :],
            (*candidate.shape[:2], PARENT_RAW_DIM),
        )
        return (
            self.constructor(mx.concatenate([repeated_parent, candidate], axis=-1))
            * candidate_mask[..., None]
        )


class ExactLocalRelationProbe(RawFactorProbe):
    """Fresh parent/action construction with exact local board geometry."""

    def __init__(self):
        super().__init__()
        self.parent_projection = nn.Sequential(
            nn.Linear(PARENT_RAW_DIM, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(
                CANDIDATE_RAW_DIM + LOCAL_GEOMETRY_CONTEXT_DIM,
                768,
            ),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )
        self.integration = relation_integration_network()

    def encode_candidates(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        parent = self.parent_projection(
            raw_parent_features(
                board_entities,
                board_mask,
                market_entities,
                market_mask,
                global_features,
                public_supply,
            )
        )
        raw_candidate = raw_candidate_features(
            action_features,
            prior_features,
            staged_market_entities,
            staged_market_mask,
            staged_public_supply,
        )
        local = candidate_local_geometry(
            board_entities,
            board_mask,
            action_features,
            candidate_mask,
        )
        candidate = self.candidate_projection(mx.concatenate([raw_candidate, local], axis=-1))
        return (
            integrate_parent_candidate(
                self.integration,
                parent,
                candidate,
            )
            * candidate_mask[..., None]
        )


class ExplicitMarketTransitionProbe(RawFactorProbe):
    """Construct explicit current-to-staged market and supply relations."""

    def __init__(self):
        super().__init__()
        self.parent_projection = nn.Sequential(
            nn.Linear(PARENT_RAW_DIM, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )
        self.transition_projection = nn.Sequential(
            nn.Linear(MARKET_TRANSITION_DIM, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )
        self.integration = relation_integration_network()

    def encode_candidates(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        parent = self.parent_projection(
            raw_parent_features(
                board_entities,
                board_mask,
                market_entities,
                market_mask,
                global_features,
                public_supply,
            )
        )
        groups, candidates = action_features.shape[:2]
        before_market = mx.broadcast_to(
            market_entities[:, None, :, :],
            (groups, candidates, 4, ENTITY_DIM),
        )
        before_market_mask = mx.broadcast_to(
            market_mask[:, None, :],
            (groups, candidates, 4),
        )
        before_supply = mx.broadcast_to(
            public_supply[:, None, :],
            (
                groups,
                candidates,
                GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
            ),
        )
        transition = mx.concatenate(
            [
                before_market.reshape(groups, candidates, -1),
                staged_market_entities.reshape(groups, candidates, -1),
                (staged_market_entities - before_market).reshape(
                    groups,
                    candidates,
                    -1,
                ),
                (staged_market_entities * before_market).reshape(
                    groups,
                    candidates,
                    -1,
                ),
                before_market_mask.astype(mx.float32),
                staged_market_mask.astype(mx.float32),
                before_supply,
                staged_public_supply,
                staged_public_supply - before_supply,
                staged_public_supply * before_supply,
                action_features,
                prior_features,
            ],
            axis=-1,
        )
        if transition.shape[-1] != MARKET_TRANSITION_DIM:
            raise AssertionError("market transition dimension drifted")
        candidate = self.transition_projection(transition)
        return (
            integrate_parent_candidate(
                self.integration,
                parent,
                candidate,
            )
            * candidate_mask[..., None]
        )


class FreshEntityCrossProbe(RawFactorProbe):
    """Fresh target-supervised candidate-to-entity cross attention."""

    def __init__(self):
        super().__init__()
        hidden = ENTITY_HIDDEN_DIM
        self.board_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.market_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.seat_embedding = nn.Embedding(4, hidden)
        self.board_blocks = [SetAttentionBlock(hidden, 6, 3) for _ in range(2)]
        self.market_blocks = [SetAttentionBlock(hidden, 6, 3)]
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.supply_projection = nn.Sequential(
            nn.Linear(GRADED_ORACLE_PUBLIC_SUPPLY_SIZE, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.parent_projection = nn.Sequential(
            nn.Linear(hidden * 12, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(
                GRADED_ORACLE_ACTION_DIM
                + GRADED_ORACLE_PRIOR_DIM
                + hidden * 2
                + GRADED_ORACLE_PUBLIC_SUPPLY_SIZE
                + CONSTRUCTION_DIM,
                512,
            ),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.board_cross = nn.MultiHeadAttention(hidden, 6, bias=True)
        self.staged_cross = nn.MultiHeadAttention(hidden, 6, bias=True)
        self.board_cross_norm = nn.LayerNorm(hidden)
        self.staged_cross_norm = nn.LayerNorm(hidden)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden * 3 + CONSTRUCTION_DIM, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, CONSTRUCTION_DIM),
            nn.GELU(),
            nn.LayerNorm(CONSTRUCTION_DIM),
        )

    def encode_candidates(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        groups, candidates = action_features.shape[:2]
        hidden = ENTITY_HIDDEN_DIM

        board = self.board_projection(board_entities)
        board = board + self.seat_embedding(mx.arange(4))[None, :, None, :]
        flat_board = board.reshape(groups * 4, 23, hidden)
        flat_board_mask = board_mask.reshape(groups * 4, 23)
        flat_board = flat_board * flat_board_mask[..., None]
        for block in self.board_blocks:
            flat_board = block(flat_board, flat_board_mask)
        board_summary = _masked_pool(
            flat_board,
            flat_board_mask,
        ).reshape(groups, hidden * 8)
        board_tokens = flat_board.reshape(groups, 4 * 23, hidden)
        board_token_mask = board_mask.reshape(groups, 4 * 23)

        market = self.market_projection(market_entities)
        for block in self.market_blocks:
            market = block(market, market_mask)
        market_summary = _masked_pool(market, market_mask)
        parent = self.parent_projection(
            mx.concatenate(
                [
                    board_summary,
                    market_summary,
                    self.global_projection(global_features),
                    self.supply_projection(public_supply),
                ],
                axis=-1,
            )
        )

        staged = self.market_projection(staged_market_entities)
        flat_staged = staged.reshape(groups * candidates, 4, hidden)
        flat_staged_mask = staged_market_mask.reshape(
            groups * candidates,
            4,
        )
        for block in self.market_blocks:
            flat_staged = block(flat_staged, flat_staged_mask)
        staged_summary = _masked_pool(
            flat_staged,
            flat_staged_mask,
        ).reshape(groups, candidates, hidden * 2)
        repeated_parent = mx.broadcast_to(
            parent[:, None, :],
            (groups, candidates, CONSTRUCTION_DIM),
        )
        query = self.query_projection(
            mx.concatenate(
                [
                    action_features,
                    prior_features,
                    staged_summary,
                    staged_public_supply,
                    repeated_parent,
                ],
                axis=-1,
            )
        )

        board_attention_mask = mx.where(
            board_token_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        normalized_board = self.board_cross_norm(board_tokens)
        board_cross = self.board_cross(
            query,
            normalized_board,
            normalized_board,
            mask=board_attention_mask,
        )

        flat_query = query.reshape(groups * candidates, 1, hidden)
        normalized_staged = self.staged_cross_norm(flat_staged)
        staged_attention_mask = mx.where(
            flat_staged_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        staged_cross = self.staged_cross(
            flat_query,
            normalized_staged,
            normalized_staged,
            mask=staged_attention_mask,
        ).reshape(groups, candidates, hidden)
        return (
            self.output_projection(
                mx.concatenate(
                    [
                        query,
                        board_cross,
                        staged_cross,
                        repeated_parent,
                    ],
                    axis=-1,
                )
            )
            * candidate_mask[..., None]
        )


def relation_integration_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(CONSTRUCTION_DIM * 4, 768),
        nn.GELU(),
        nn.LayerNorm(768),
        nn.Linear(768, CONSTRUCTION_DIM),
        nn.GELU(),
        nn.LayerNorm(CONSTRUCTION_DIM),
    )


def integrate_parent_candidate(
    network: nn.Module,
    parent: mx.array,
    candidate: mx.array,
) -> mx.array:
    repeated_parent = mx.broadcast_to(
        parent[:, None, :],
        candidate.shape,
    )
    return network(
        mx.concatenate(
            [
                candidate,
                repeated_parent,
                candidate * repeated_parent,
                mx.abs(candidate - repeated_parent),
            ],
            axis=-1,
        )
    )


def raw_parent_features(
    board_entities: mx.array,
    board_mask: mx.array,
    market_entities: mx.array,
    market_mask: mx.array,
    global_features: mx.array,
    public_supply: mx.array,
) -> mx.array:
    groups = board_entities.shape[0]
    values = mx.concatenate(
        [
            board_entities.reshape(groups, -1),
            board_mask.astype(mx.float32).reshape(groups, -1),
            market_entities.reshape(groups, -1),
            market_mask.astype(mx.float32).reshape(groups, -1),
            global_features,
            public_supply,
        ],
        axis=-1,
    )
    if values.shape[-1] != PARENT_RAW_DIM:
        raise AssertionError("raw parent dimension drifted")
    return values


def raw_candidate_features(
    action_features: mx.array,
    prior_features: mx.array,
    staged_market_entities: mx.array,
    staged_market_mask: mx.array,
    staged_public_supply: mx.array,
) -> mx.array:
    groups, candidates = action_features.shape[:2]
    values = mx.concatenate(
        [
            action_features,
            prior_features,
            staged_market_entities.reshape(groups, candidates, -1),
            staged_market_mask.astype(mx.float32),
            staged_public_supply,
        ],
        axis=-1,
    )
    if values.shape[-1] != CANDIDATE_RAW_DIM:
        raise AssertionError("raw candidate dimension drifted")
    return values


def build_raw_factor_probe(kind: str) -> RawFactorProbe:
    if kind == COMPLETE_RAW_FLAT:
        return CompleteRawFlatProbe()
    if kind == EXACT_LOCAL_RELATION:
        return ExactLocalRelationProbe()
    if kind == EXPLICIT_MARKET_TRANSITION:
        return ExplicitMarketTransitionProbe()
    if kind == FRESH_ENTITY_CROSS:
        return FreshEntityCrossProbe()
    raise ValueError("unsupported raw factor-construction probe")


def parameter_count(model: nn.Module) -> int:
    return sum(int(np.prod(value.shape)) for _, value in tree_flatten(model.parameters()))


def batch_counts(batch: object) -> tuple[int, ...]:
    mask = np.asarray(batch.candidate_mask)
    return tuple(int(value) for value in np.sum(mask, axis=1, dtype=np.int64))


def score_raw_factor_batch(
    model: RawFactorProbe,
    batch: object,
    counts: tuple[int, ...] | None = None,
) -> mx.array:
    counts = counts or batch_counts(batch)
    return model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.candidate_mask,
        counts,
        np.asarray(batch.screen_rank),
        batch.action_hash,
    )


def balanced_score_binary_loss(
    scores: mx.array,
    target: mx.array,
    eligible: mx.array,
    counts: tuple[int, ...],
) -> mx.array:
    losses = []
    for group_index, count in enumerate(counts):
        group_target = target[group_index, :count]
        group_eligible = eligible[group_index, :count]
        group_scores = scores[group_index, :count]
        negative = group_eligible & ~group_target
        positive_loss = mx.sum(
            mx.where(group_target, nn.softplus(-group_scores), 0.0)
        ) / mx.maximum(mx.sum(group_target), 1)
        negative_loss = mx.sum(mx.where(negative, nn.softplus(group_scores), 0.0)) / mx.maximum(
            mx.sum(negative), 1
        )
        losses.append(positive_loss + negative_loss)
    return mx.mean(mx.stack(losses))


def raw_factor_probe_loss(
    model: RawFactorProbe,
    board_entities: mx.array,
    board_mask: mx.array,
    market_entities: mx.array,
    market_mask: mx.array,
    global_features: mx.array,
    public_supply: mx.array,
    action_features: mx.array,
    prior_features: mx.array,
    staged_market_entities: mx.array,
    staged_market_mask: mx.array,
    staged_public_supply: mx.array,
    candidate_mask: mx.array,
    target: mx.array,
    eligible: mx.array,
    counts: tuple[int, ...],
    screen_rank: np.ndarray,
    action_hash: np.ndarray,
) -> mx.array:
    scores = model(
        board_entities,
        board_mask,
        market_entities,
        market_mask,
        global_features,
        public_supply,
        action_features,
        prior_features,
        staged_market_entities,
        staged_market_mask,
        staged_public_supply,
        candidate_mask,
        counts,
        screen_rank,
        action_hash,
    )
    return balanced_score_binary_loss(scores, target, eligible, counts)


def train_raw_factor_probe(
    *,
    train_dataset_root: Path,
    validation_dataset_root: Path,
    output_root: Path,
    config: RawFactorProbeConfig,
) -> dict[str, Any]:
    config.validate()
    allocator = configure_mlx_memory()
    if output_root.exists():
        raise ValueError("raw factor-construction output already exists")
    train = GradedOracleDataset(train_dataset_root, verify_checksums=True)
    validation = GradedOracleDataset(
        validation_dataset_root,
        verify_checksums=True,
    )
    if train.split != "train" or validation.split != "validation":
        raise ValueError("raw factor-construction dataset split mismatch")

    mx.random.seed(config.seed)
    model = build_raw_factor_probe(config.kind)
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, raw_factor_probe_loss)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in _dataset_batches(
            train,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            counts = batch_counts(batch)
            target = _target_mask(batch)
            eligible = _eligible_mask(batch)
            loss, gradients = loss_and_grad(
                model,
                batch.board_entities,
                batch.board_mask,
                batch.market_entities,
                batch.market_mask,
                batch.global_features,
                batch.public_supply,
                batch.action_features,
                batch.prior_features,
                batch.staged_market_entities,
                batch.staged_market_mask,
                batch.staged_public_supply,
                batch.candidate_mask,
                mx.array(target),
                mx.array(eligible),
                counts,
                np.asarray(batch.screen_rank),
                batch.action_hash,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            total_loss += float(loss.item())
            batches += 1

        train_metrics = evaluate_raw_factor_probe(model, train)
        validation_metrics = evaluate_raw_factor_probe(model, validation)
        memory_before_clear = mlx_memory_snapshot()
        mx.clear_cache()
        memory_after_clear = mlx_memory_snapshot()
        event = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "train": train_metrics,
            "validation": validation_metrics,
            "mlx_memory_before_clear": memory_before_clear,
            "mlx_memory_after_clear": memory_after_clear,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        key = (
            float(train_metrics["target_positive_recall"]),
            float(train_metrics["target_set_exact_fraction"]),
            float(validation_metrics["target_positive_recall"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            mx.save_safetensors(
                str(output_root / "best.safetensors"),
                dict(tree_flatten(model.parameters())),
            )
            _write_json_atomic(output_root / "best.json", event)

    model.load_weights(str(output_root / "best.safetensors"))
    mx.eval(model.parameters())
    train_metrics = evaluate_raw_factor_probe(model, train)
    validation_metrics = evaluate_raw_factor_probe(model, validation)
    final_memory_before_clear = mlx_memory_snapshot()
    mx.clear_cache()
    final_memory_after_clear = mlx_memory_snapshot()
    usage = _resource_usage()
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "probe": asdict(config),
        "host": _canonical_host(),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(model),
        "train_dataset_manifest_blake3": checksum(train.root / "dataset.json"),
        "validation_dataset_manifest_blake3": checksum(validation.root / "dataset.json"),
        "train": train_metrics,
        "validation": validation_metrics,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "candidate_tensor_passes": (
                config.epochs
                * (train.candidate_count + train.candidate_count + validation.candidate_count)
                + train.candidate_count
                + validation.candidate_count
            ),
            **usage,
            "mlx_allocator": allocator,
            "mlx_memory_before_clear": final_memory_before_clear,
            "mlx_memory_after_clear": final_memory_after_clear,
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "external_compute_used": False,
    }
    report["execution"]["candidate_tensor_passes_per_second"] = report["execution"][
        "candidate_tensor_passes"
    ] / max(report["execution"]["elapsed_seconds"], 1e-9)
    _write_json_atomic(output_root / "report.json", report)
    return report


def evaluate_raw_factor_probe(
    model: RawFactorProbe,
    dataset: GradedOracleDataset,
) -> dict[str, Any]:
    model.eval()
    groups = 0
    candidates = 0
    target_positives = 0
    target_hits = 0
    exact_sets = 0
    winner_hits = 0
    regret = 0.0
    finite = True
    total_loss = 0.0
    for batch in _dataset_batches(dataset):
        counts = batch_counts(batch)
        target = _target_mask(batch)
        eligible = _eligible_mask(batch)
        scores = score_raw_factor_batch(model, batch, counts)
        loss = balanced_score_binary_loss(
            scores,
            mx.array(target),
            mx.array(eligible),
            counts,
        )
        mx.eval(scores, loss)
        values = np.asarray(scores)
        finite &= bool(np.all(np.isfinite(values)))
        total_loss += float(loss.item()) * len(counts)
        source_flags = np.asarray(batch.source_flags)
        selected_indices = np.asarray(batch.selected_index)
        r4800_mean = np.asarray(batch.r4800_mean)
        r4800_mask = np.asarray(batch.r4800_mask)
        for group_index, count in enumerate(counts):
            group_scores = values[group_index, :count]
            group_flags = source_flags[group_index, :count]
            group_hashes = batch.action_hash[group_index, :count]
            group_target = target[group_index, :count]
            retained = frontier_anchored_retained_indices(
                scores=group_scores,
                source_flags=group_flags,
                action_hashes=group_hashes,
            )
            retained_nonfrontier = retained[
                (group_flags[retained] & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
            ]
            quota = int(np.sum(group_target))
            recalled = int(np.sum(group_target[retained_nonfrontier]))
            target_positives += quota
            target_hits += recalled
            exact_sets += int(recalled == quota)
            winner_hits += int(int(selected_indices[group_index]) in retained)
            labeled = r4800_mask[group_index, :count]
            retained_labeled = retained[labeled[retained]]
            if np.any(labeled) and len(retained_labeled):
                regret += max(
                    0.0,
                    float(np.max(r4800_mean[group_index, :count][labeled]))
                    - float(np.max(r4800_mean[group_index, :count][retained_labeled])),
                )
            groups += 1
            candidates += count
    return {
        "groups": groups,
        "candidates": candidates,
        "balanced_binary_loss": total_loss / groups,
        "target_positives": target_positives,
        "target_positive_recall": target_hits / target_positives,
        "target_set_exact_fraction": exact_sets / groups,
        "top64_r4800_winner_recall": winner_hits / groups,
        "mean_top64_retained_r4800_regret": regret / groups,
        "all_scores_finite": finite,
        "all_groups_scored_once": groups == dataset.group_count,
        "all_candidates_scored_once": candidates == dataset.candidate_count,
    }


def raw_factor_construction_classification(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(reports) != set(PROBE_KINDS):
        raise ValueError("raw factor classification requires all probes")
    train_gates = {kind: _train_gate(report) for kind, report in reports.items()}
    validation_gates = {kind: _validation_gate(report) for kind, report in reports.items()}
    passing = [kind for kind in PROBE_KINDS if train_gates[kind] and validation_gates[kind]]
    selected = None
    if passing:
        fixed_order = {
            EXPLICIT_MARKET_TRANSITION: 3,
            EXACT_LOCAL_RELATION: 2,
            COMPLETE_RAW_FLAT: 1,
            FRESH_ENTITY_CROSS: 0,
        }
        selected = max(
            passing,
            key=lambda kind: (
                float(reports[kind]["validation"]["target_positive_recall"]),
                float(reports[kind]["validation"]["target_set_exact_fraction"]),
                float(reports[kind]["train"]["target_positive_recall"]),
                -int(
                    reports[kind]["execution"]["mlx_memory_before_clear"][
                        "peak_active_memory_bytes"
                    ]
                ),
                fixed_order[kind],
            ),
        )
        classification = "raw_factor_construction_sufficient"
    elif any(train_gates.values()):
        classification = "raw_factor_construction_train_separable_not_generalized"
    else:
        classification = "raw_factor_construction_insufficient"
    return {
        "train_gates": train_gates,
        "validation_gates": validation_gates,
        "selected_kind": selected,
        "classification": classification,
    }


def load_raw_factor_probe(
    *,
    kind: str,
    weights: Path,
) -> RawFactorProbe:
    model = build_raw_factor_probe(kind)
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def evaluate_saved_raw_factor_probe(
    *,
    kind: str,
    weights: Path,
    train_dataset_root: Path,
    validation_dataset_root: Path,
) -> dict[str, Any]:
    allocator = configure_mlx_memory()
    train = GradedOracleDataset(train_dataset_root, verify_checksums=True)
    validation = GradedOracleDataset(
        validation_dataset_root,
        verify_checksums=True,
    )
    model = load_raw_factor_probe(kind=kind, weights=weights)
    started = time.perf_counter()
    train_metrics = evaluate_raw_factor_probe(model, train)
    validation_metrics = evaluate_raw_factor_probe(model, validation)
    evaluation_seconds = time.perf_counter() - started
    memory_before_clear = mlx_memory_snapshot()
    mx.clear_cache()
    memory_after_clear = mlx_memory_snapshot()
    usage = _resource_usage()
    scientific = {
        "kind": kind,
        "weights_blake3": checksum(weights),
        "parameter_count": parameter_count(model),
        "train_dataset_manifest_blake3": checksum(train.root / "dataset.json"),
        "validation_dataset_manifest_blake3": checksum(validation.root / "dataset.json"),
        "train": train_metrics,
        "validation": validation_metrics,
        "test_split_opened": False,
        "gameplay_opened": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _canonical_host(),
        "scientific": scientific,
        "execution": {
            "evaluation_seconds": evaluation_seconds,
            "candidates_per_second": (
                (train.candidate_count + validation.candidate_count) / max(evaluation_seconds, 1e-9)
            ),
            **usage,
            "mlx_allocator": allocator,
            "mlx_memory_before_clear": memory_before_clear,
            "mlx_memory_after_clear": memory_after_clear,
        },
        "scientific_blake3": blake3.blake3(
            json.dumps(
                scientific,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def _dataset_batches(
    dataset: GradedOracleDataset,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Any:
    return dataset.batches(
        64,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        shuffle=shuffle,
        seed=seed,
    )


def _target_mask(batch: object) -> np.ndarray:
    return build_frontier_anchored_target_mask(
        r1200_mean=np.asarray(batch.r1200_mean),
        r1200_mask=np.asarray(batch.r1200_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=np.asarray(batch.candidate_mask),
        action_hashes=batch.action_hash,
    )


def _eligible_mask(batch: object) -> np.ndarray:
    source_flags = np.asarray(batch.source_flags)
    return np.asarray(batch.candidate_mask) & (
        (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    )


def _train_gate(report: dict[str, Any]) -> bool:
    return (
        float(report["train"]["target_positive_recall"]) >= 0.80
        and float(report["train"]["target_set_exact_fraction"]) >= 0.25
    )


def _validation_gate(report: dict[str, Any]) -> bool:
    return (
        float(report["validation"]["target_positive_recall"]) >= 0.50
        and float(report["validation"]["target_set_exact_fraction"]) >= 0.01
    )


def _resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
    }


def _canonical_host() -> str:
    host = socket.gethostname().split(".")[0].lower()
    return "john1" if host == "johns-mac-mini" else host


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--kind", choices=PROBE_KINDS, required=True)
    probe.add_argument("--train-dataset", type=Path, required=True)
    probe.add_argument("--validation-dataset", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--kind", choices=PROBE_KINDS, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--train-dataset", type=Path, required=True)
    evaluate.add_argument("--validation-dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "probe":
        report = train_raw_factor_probe(
            train_dataset_root=args.train_dataset,
            validation_dataset_root=args.validation_dataset,
            output_root=args.output,
            config=RawFactorProbeConfig(
                kind=args.kind,
                seed=PROBE_SEEDS[args.kind],
            ),
        )
    else:
        report = evaluate_saved_raw_factor_probe(
            kind=args.kind,
            weights=args.weights,
            train_dataset_root=args.train_dataset,
            validation_dataset_root=args.validation_dataset,
        )
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
