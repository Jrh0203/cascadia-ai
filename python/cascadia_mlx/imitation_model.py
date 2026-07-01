"""Shared-state MLX scorer for complete canonical action sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.imitation_dataset import (
    PROPOSAL_ACTION_DIM,
    PROPOSAL_ACTION_IMMEDIATE_RANK_INDEX,
    PROPOSAL_ACTION_IMMEDIATE_SCORE_INDEX,
)
from cascadia_mlx.model import SetAttentionBlock, _masked_pool

IMITATION_ARCHITECTURE_V1 = "shared-state-action-imitation-v1"
IMITATION_ARCHITECTURE_CROSS_V2 = "shared-state-action-cross-ranker-v2"
IMITATION_ARCHITECTURE_RESIDUAL_V2 = "shared-state-action-residual-ranker-v2"
IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3 = "shared-state-action-score-residual-v3"
IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4 = "exact-parent-candidate-set-residual-v4"
IMITATION_DISTRIBUTION_SCORE_FLOOR = 1.0
IMITATION_DISTRIBUTION_SELECTED_WEIGHT = 0.25
IMITATION_SCORE_RESIDUAL_SCALE = 100.0
IMITATION_SCORE_RESIDUAL_ERROR_SCALE = 10.0
IMITATION_SCORE_RESIDUAL_SELECTION_TEMPERATURE = 5.0
IMITATION_SCORE_RESIDUAL_SELECTED_WEIGHT = 0.25


@dataclass(frozen=True)
class ImitationModelConfig:
    schema_version: int = 1
    architecture: str = IMITATION_ARCHITECTURE_V1
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    feed_forward_multiplier: int = 3
    immediate_rank_prior: float = 0.0

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture not in {
            IMITATION_ARCHITECTURE_V1,
            IMITATION_ARCHITECTURE_CROSS_V2,
            IMITATION_ARCHITECTURE_RESIDUAL_V2,
            IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
            IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
        }:
            raise ValueError("unsupported imitation model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.board_blocks < 0 or self.market_blocks < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed-forward multiplier must be positive")
        if self.architecture == IMITATION_ARCHITECTURE_RESIDUAL_V2:
            if self.immediate_rank_prior <= 0:
                raise ValueError("residual imitation requires a positive rank prior")
        elif self.immediate_rank_prior != 0:
            raise ValueError("rank prior is only valid for residual imitation")

    def to_dict(self) -> dict[str, float | int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ImitationModelConfig:
        config = cls(**values)
        config.validate()
        return config


class SharedStateActionRanker(nn.Module):
    def __init__(self, config: ImitationModelConfig | None = None):
        super().__init__()
        config = config or ImitationModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
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
        self.board_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.board_blocks)
        ]
        self.market_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.market_blocks)
        ]
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(PROPOSAL_ACTION_DIM, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        if config.architecture == IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4:
            self.parent_projection = nn.Sequential(
                nn.Linear(3, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
            )
        if config.architecture == IMITATION_ARCHITECTURE_CROSS_V2:
            self.action_query_norm = nn.LayerNorm(hidden)
            self.board_cross_norm = nn.LayerNorm(hidden)
            self.market_cross_norm = nn.LayerNorm(hidden)
            self.board_cross_attention = nn.MultiHeadAttention(
                hidden,
                config.attention_heads,
                bias=True,
            )
            self.market_cross_attention = nn.MultiHeadAttention(
                hidden,
                config.attention_heads,
                bias=True,
            )
            trunk_width = hidden * 14
        elif config.architecture == IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4:
            trunk_width = hidden * 16
        else:
            trunk_width = hidden * 12
        self.trunk = nn.Sequential(
            nn.Linear(trunk_width, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1),
        )
        if config.architecture in {
            IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3,
            IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
        }:
            output = self.trunk.layers[-1]
            output.weight = mx.zeros_like(output.weight)
            output.bias = mx.zeros_like(output.bias)

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        action_features: mx.array,
        candidate_mask: mx.array | None = None,
        parent_total: mx.array | None = None,
        parent_rank: mx.array | None = None,
    ) -> mx.array:
        groups, candidates = action_features.shape[:2]
        hidden = self.config.hidden_dim

        boards = self.board_projection(board_entities)
        boards = boards + self.seat_embedding(mx.arange(4))[None, :, None, :]
        boards = boards.reshape(groups * 4, 23, hidden)
        flat_board_mask = board_mask.reshape(groups * 4, 23)
        boards = boards * flat_board_mask[..., None]
        for block in self.board_blocks:
            boards = block(boards, flat_board_mask)
        board_summary = _masked_pool(boards, flat_board_mask).reshape(groups, -1)
        board_tokens = boards.reshape(groups, 4 * 23, hidden)
        board_token_mask = board_mask.reshape(groups, 4 * 23)

        market = self.market_projection(market_entities)
        for block in self.market_blocks:
            market = block(market, market_mask)
        market_summary = _masked_pool(market, market_mask)

        state = mx.concatenate(
            [
                board_summary,
                market_summary,
                self.global_projection(global_features),
            ],
            axis=-1,
        )
        state = mx.broadcast_to(state[:, None, :], (groups, candidates, state.shape[-1]))
        actions = self.action_projection(action_features)
        combined = [state, actions]
        if self.config.architecture == IMITATION_ARCHITECTURE_CROSS_V2:
            queries = self.action_query_norm(actions)
            board_attention_mask = mx.where(
                board_token_mask[:, None, None, :],
                0.0,
                -1e9,
            )
            market_attention_mask = mx.where(
                market_mask[:, None, None, :],
                0.0,
                -1e9,
            )
            board_tokens = self.board_cross_norm(board_tokens)
            market_tokens = self.market_cross_norm(market)
            combined.extend(
                [
                    self.board_cross_attention(
                        queries,
                        board_tokens,
                        board_tokens,
                        mask=board_attention_mask,
                    ),
                    self.market_cross_attention(
                        queries,
                        market_tokens,
                        market_tokens,
                        mask=market_attention_mask,
                    ),
                ]
            )
        elif self.config.architecture == IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4:
            if candidate_mask is None or parent_total is None or parent_rank is None:
                raise ValueError("parent-set residual requires candidate masks and parent priors")
            standardized_parent = _masked_standardize(parent_total, candidate_mask)
            reciprocal_rank = 1.0 / mx.maximum(parent_rank, 1.0)
            parent_features = mx.stack(
                [
                    parent_total / 100.0,
                    reciprocal_rank,
                    standardized_parent,
                ],
                axis=-1,
            )
            parent = self.parent_projection(parent_features)
            candidate = (actions + parent) * candidate_mask[..., None]
            pooled = _masked_pool(candidate, candidate_mask)
            mean = pooled[..., :hidden]
            maximum = pooled[..., hidden:]
            mean = mx.broadcast_to(mean[:, None, :], candidate.shape)
            maximum = mx.broadcast_to(maximum[:, None, :], candidate.shape)
            combined.extend([parent, mean, maximum, candidate - mean])
        return self.trunk(mx.concatenate(combined, axis=-1)).reshape(groups, candidates)


def imitation_loss(model: SharedStateActionRanker, batch: object) -> mx.array:
    scores = score_imitation_actions(
        model,
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
        batch.candidate_mask,
        getattr(batch, "parent_total", None),
        getattr(batch, "parent_rank", None),
    )
    masked_scores = mx.where(batch.candidate_mask, scores, -1e9)
    log_probabilities = masked_scores - mx.logsumexp(masked_scores, axis=-1, keepdims=True)
    return -mx.mean(mx.sum(batch.teacher_mean * log_probabilities, axis=-1))


def distributional_imitation_loss(
    model: SharedStateActionRanker,
    batch: object,
) -> mx.array:
    scores = score_imitation_actions(
        model,
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
        batch.candidate_mask,
        getattr(batch, "parent_total", None),
        getattr(batch, "parent_rank", None),
    )
    return distributional_imitation_loss_from_scores(scores, batch)


def distributional_imitation_loss_from_scores(
    scores: mx.array,
    batch: object,
) -> mx.array:
    """Uncertainty-aware pairwise distillation plus full-legal winner recall."""
    scored = batch.teacher_scored
    samples = mx.maximum(batch.teacher_samples, 1.0)
    standard_error = batch.teacher_stddev / mx.sqrt(samples)
    mean_difference = batch.teacher_mean[:, :, None] - batch.teacher_mean[:, None, :]
    pair_scale = mx.sqrt(
        mx.square(standard_error[:, :, None])
        + mx.square(standard_error[:, None, :])
        + IMITATION_DISTRIBUTION_SCORE_FLOOR**2
    )
    teacher_probability = mx.sigmoid(mean_difference / pair_scale)
    student_logit = scores[:, :, None] - scores[:, None, :]
    pair_loss = mx.logaddexp(0.0, student_logit) - teacher_probability * student_logit
    indices = mx.arange(scores.shape[1])
    pair_mask = (
        scored[:, :, None] & scored[:, None, :] & (indices[:, None] < indices[None, :])[None, :, :]
    )
    confidence = 0.25 + 0.75 * mx.abs(2.0 * teacher_probability - 1.0)
    pair_weights = pair_mask.astype(scores.dtype) * confidence
    pairwise = mx.sum(pair_loss * pair_weights) / mx.maximum(mx.sum(pair_weights), 1.0)

    masked_scores = mx.where(batch.candidate_mask, scores, -1e9)
    log_probabilities = masked_scores - mx.logsumexp(masked_scores, axis=-1, keepdims=True)
    selected = batch.selected.astype(scores.dtype)
    selected_listwise = -mx.mean(mx.sum(selected * log_probabilities, axis=-1))
    return pairwise + IMITATION_DISTRIBUTION_SELECTED_WEIGHT * selected_listwise


def score_residual_imitation_loss(
    model: SharedStateActionRanker,
    batch: object,
) -> mx.array:
    scores = score_imitation_actions(
        model,
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
        batch.candidate_mask,
        getattr(batch, "parent_total", None),
        getattr(batch, "parent_rank", None),
    )
    return score_residual_imitation_loss_from_scores(scores, batch)


def score_residual_imitation_loss_from_scores(
    scores: mx.array,
    batch: object,
) -> mx.array:
    """Fit point-scale final score while preserving exact immediate score."""
    samples = mx.maximum(batch.teacher_samples, 1.0)
    standard_error = batch.teacher_stddev / mx.sqrt(samples)
    confidence = 1.0 / (1.0 + mx.square(standard_error))
    scored = batch.teacher_scored.astype(scores.dtype)
    error = (scores - batch.teacher_mean) / IMITATION_SCORE_RESIDUAL_ERROR_SCALE
    absolute_error = mx.abs(error)
    huber = mx.where(
        absolute_error <= 1.0,
        0.5 * mx.square(error),
        absolute_error - 0.5,
    )
    regression_weights = scored * confidence
    regression = mx.sum(huber * regression_weights) / mx.maximum(
        mx.sum(regression_weights),
        1.0,
    )

    logits = scores / IMITATION_SCORE_RESIDUAL_SELECTION_TEMPERATURE
    masked_logits = mx.where(batch.candidate_mask, logits, -1e9)
    log_probabilities = masked_logits - mx.logsumexp(
        masked_logits,
        axis=-1,
        keepdims=True,
    )
    selected = batch.selected.astype(scores.dtype)
    selected_listwise = -mx.mean(mx.sum(selected * log_probabilities, axis=-1))
    return regression + IMITATION_SCORE_RESIDUAL_SELECTED_WEIGHT * selected_listwise


def score_imitation_actions(
    model: SharedStateActionRanker,
    board_entities: mx.array,
    board_mask: mx.array,
    market_entities: mx.array,
    market_mask: mx.array,
    global_features: mx.array,
    action_features: mx.array,
    candidate_mask: mx.array | None = None,
    parent_total: mx.array | None = None,
    parent_rank: mx.array | None = None,
) -> mx.array:
    raw_scores = model(
        board_entities,
        board_mask,
        market_entities,
        market_mask,
        global_features,
        action_features,
        candidate_mask,
        parent_total,
        parent_rank,
    )
    architecture = getattr(getattr(model, "config", None), "architecture", None)
    if architecture == IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4:
        if candidate_mask is None or parent_total is None:
            raise ValueError("parent-set residual requires parent totals and candidate masks")
        return _masked_standardize(parent_total, candidate_mask) + raw_scores
    if architecture == IMITATION_ARCHITECTURE_SCORE_RESIDUAL_V3:
        immediate_score = (
            action_features[..., PROPOSAL_ACTION_IMMEDIATE_SCORE_INDEX]
            * IMITATION_SCORE_RESIDUAL_SCALE
        )
        return immediate_score + raw_scores * IMITATION_SCORE_RESIDUAL_SCALE
    strength = getattr(getattr(model, "config", None), "immediate_rank_prior", 0.0)
    if strength == 0:
        return raw_scores
    if candidate_mask is None:
        candidate_mask = mx.ones(raw_scores.shape, dtype=mx.bool_)
    rank_feature = action_features[..., PROPOSAL_ACTION_IMMEDIATE_RANK_INDEX]
    reciprocal_rank = 1.0 / mx.maximum(rank_feature, 1.0 / 4096.0)
    return _masked_standardize(raw_scores, candidate_mask) + strength * _masked_standardize(
        reciprocal_rank,
        candidate_mask,
    )


def _masked_standardize(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    count = mx.maximum(mx.sum(weights, axis=-1, keepdims=True), 1.0)
    mean = mx.sum(values * weights, axis=-1, keepdims=True) / count
    centered = (values - mean) * weights
    variance = mx.sum(mx.square(centered), axis=-1, keepdims=True) / count
    return centered / mx.sqrt(variance + 1e-6)
