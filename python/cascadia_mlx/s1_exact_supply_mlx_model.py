"""Iso-parameter MLX ranker for the ADR 0147 exact-supply comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import blake3
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    GRADED_ORACLE_UNCERTAINTY_FLOOR,
)
from cascadia_mlx.model import SetAttentionBlock, _masked_pool
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ARCHETYPE_COUNT,
    ARMS,
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
    SUPPLY_TOKEN_DIM,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "s1-exact-supply-iso-complete-action-v1"
REFILL_LOSS_WEIGHT = 0.25
FROZEN_PARAMETER_COUNT = 3_073_101


@dataclass(frozen=True)
class S1ExactSupplyModelConfig:
    """The only model parameterization admitted to the S1 comparison."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    arm: str = ARMS[0]
    hidden_dim: int = 128
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    supply_blocks: int = 1
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION or self.architecture != ARCHITECTURE:
            raise ValueError("unsupported S1 exact-supply model schema")
        if self.arm not in ARMS:
            raise ValueError("S1 model names an unknown comparison arm")
        if self.hidden_dim != 128 or self.attention_heads != 4:
            raise ValueError("ADR 0147 freezes hidden width 128 and four attention heads")
        if self.board_blocks != 2 or self.market_blocks != 1 or self.supply_blocks != 1:
            raise ValueError("ADR 0147 attention-block counts drifted")
        if self.feed_forward_multiplier != 3:
            raise ValueError("ADR 0147 freezes the feed-forward multiplier at three")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> S1ExactSupplyModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class S1ExactSupplyPrediction:
    """Complete-action scores, uncertainty, and learned next-tile law."""

    scores: mx.array
    residuals: mx.array
    standard_errors: mx.array
    refill_probabilities: mx.array


class S1ExactSupplyRanker(nn.Module):
    """One shared parameter budget with arm-specific factual input routing."""

    def __init__(self, config: S1ExactSupplyModelConfig | None = None):
        super().__init__()
        config = config or S1ExactSupplyModelConfig()
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
        self.supply_vector_projection = nn.Sequential(
            nn.Linear(EXACT_SUPPLY_DIM, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.supply_token_projection = nn.Sequential(
            nn.Linear(SUPPLY_TOKEN_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.supply_token_embedding = nn.Embedding(EXACT_SUPPLY_DIM, hidden)
        self.supply_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.supply_blocks)
        ]
        self.parent_projection = nn.Sequential(
            nn.Linear(hidden * 14, hidden * 3),
            nn.GELU(),
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(GRADED_ORACLE_ACTION_DIM, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(GRADED_ORACLE_PRIOR_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.staged_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.selected_archetype_embedding = nn.Embedding(ARCHETYPE_COUNT, hidden)
        self.frontier_projection = nn.Sequential(
            nn.Linear(FRONTIER_FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.frontier_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.action_query_norm = nn.LayerNorm(hidden)
        self.board_cross_norm = nn.LayerNorm(hidden)
        self.staged_market_cross_norm = nn.LayerNorm(hidden)
        self.supply_cross_norm = nn.LayerNorm(hidden)
        self.board_cross_attention = nn.MultiHeadAttention(
            hidden,
            config.attention_heads,
            bias=True,
        )
        self.staged_market_cross_attention = nn.MultiHeadAttention(
            hidden,
            config.attention_heads,
            bias=True,
        )
        self.supply_cross_attention = nn.MultiHeadAttention(
            hidden,
            config.attention_heads,
            bias=True,
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden * 9, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
        )
        self.output_trunk = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 3),
            nn.GELU(),
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
        )
        self.refill_trunk = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, ARCHETYPE_COUNT),
        )
        self.residual_head = nn.Linear(hidden, 1)
        self.standard_error_head = nn.Linear(hidden, 1)
        self.residual_head.weight = mx.zeros_like(self.residual_head.weight)
        self.residual_head.bias = mx.zeros_like(self.residual_head.bias)
        self.standard_error_head.weight = mx.zeros_like(self.standard_error_head.weight)
        self.standard_error_head.bias = (
            mx.ones_like(self.standard_error_head.bias) * 0.541324854612918
        )

    def __call__(self, batch: object) -> S1ExactSupplyPrediction:
        base = batch.base
        groups, candidates = base.screen_value.shape
        hidden = self.config.hidden_dim

        boards = self.board_projection(base.board_entities)
        boards = boards + self.seat_embedding(mx.arange(4))[None, :, None, :]
        boards = boards.reshape(groups * 4, 23, hidden)
        flat_board_mask = base.board_mask.reshape(groups * 4, 23)
        boards = boards * flat_board_mask[..., None]
        for block in self.board_blocks:
            boards = block(boards, flat_board_mask)
        board_summary = _masked_pool(boards, flat_board_mask).reshape(groups, hidden * 8)
        board_tokens = boards.reshape(groups, 4 * 23, hidden)
        board_token_mask = base.board_mask.reshape(groups, 4 * 23)

        market = self.market_projection(base.market_entities)
        for block in self.market_blocks:
            market = block(market, base.market_mask)
        market_summary = _masked_pool(market, base.market_mask)

        supply_tokens = self.supply_token_projection(batch.supply_tokens)
        token_ids = mx.arange(batch.supply_tokens.shape[1])
        supply_tokens = supply_tokens + self.supply_token_embedding(token_ids)[None, :, :]
        supply_tokens = supply_tokens * batch.supply_mask[..., None]
        for block in self.supply_blocks:
            supply_tokens = block(supply_tokens, batch.supply_mask)
        supply_pool = _masked_pool(supply_tokens, batch.supply_mask)
        supply_vector = self.supply_vector_projection(batch.supply_vector)
        parent = self.parent_projection(
            mx.concatenate(
                [
                    board_summary,
                    market_summary,
                    self.global_projection(base.global_features),
                    supply_vector,
                    supply_pool,
                ],
                axis=-1,
            )
        )

        staged_market = self.market_projection(base.staged_market_entities)
        flat_staged_market = staged_market.reshape(groups * candidates, 4, hidden)
        flat_staged_mask = base.staged_market_mask.reshape(groups * candidates, 4)
        for block in self.market_blocks:
            flat_staged_market = block(flat_staged_market, flat_staged_mask)
        staged_market = flat_staged_market.reshape(groups, candidates, 4, hidden)
        staged_summary = _masked_pool(
            flat_staged_market,
            flat_staged_mask,
        ).reshape(groups, candidates, hidden * 2)
        staged = self.staged_projection(
            mx.concatenate(
                [
                    staged_summary,
                    self.supply_vector_projection(batch.staged_supply_vector),
                ],
                axis=-1,
            )
        )

        action = self.action_projection(base.action_features)
        prior = self.prior_projection(base.prior_features)
        frontier = self.frontier_fusion(
            mx.concatenate(
                [
                    self.selected_archetype_embedding(batch.selected_archetype),
                    self.frontier_projection(batch.frontier_features),
                ],
                axis=-1,
            )
        )
        parent_candidates = mx.broadcast_to(
            parent[:, None, :],
            (groups, candidates, hidden),
        )
        query = self.action_query_norm(
            action + prior + parent_candidates + staged + frontier
        )

        board_attention_mask = mx.where(
            board_token_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        normalized_boards = self.board_cross_norm(board_tokens)
        board_cross = self.board_cross_attention(
            query,
            normalized_boards,
            normalized_boards,
            mask=board_attention_mask,
        )

        flat_query = query.reshape(groups * candidates, 1, hidden)
        staged_attention_mask = mx.where(
            flat_staged_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        normalized_staged_market = self.staged_market_cross_norm(flat_staged_market)
        staged_cross = self.staged_market_cross_attention(
            flat_query,
            normalized_staged_market,
            normalized_staged_market,
            mask=staged_attention_mask,
        ).reshape(groups, candidates, hidden)

        supply_query = query if self.config.arm == ARMS[2] else parent_candidates
        supply_attention_mask = mx.where(
            batch.supply_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        normalized_supply = self.supply_cross_norm(supply_tokens)
        supply_cross = self.supply_cross_attention(
            supply_query,
            normalized_supply,
            normalized_supply,
            mask=supply_attention_mask,
        )
        candidate = self.candidate_projection(
            mx.concatenate(
                [
                    action,
                    prior,
                    parent_candidates,
                    staged,
                    frontier,
                    board_cross,
                    staged_cross,
                    supply_cross,
                    action * parent_candidates,
                ],
                axis=-1,
            )
        )
        candidate = candidate * base.candidate_mask[..., None]
        candidate_pool = _masked_pool(candidate, base.candidate_mask)
        candidate_mean = mx.broadcast_to(
            candidate_pool[:, None, :hidden],
            candidate.shape,
        )
        candidate_maximum = mx.broadcast_to(
            candidate_pool[:, None, hidden:],
            candidate.shape,
        )
        output = self.output_trunk(
            mx.concatenate(
                [
                    candidate,
                    candidate_mean,
                    candidate_maximum,
                    candidate - candidate_mean,
                ],
                axis=-1,
            )
        )
        raw_residual = self.residual_head(output).reshape(groups, candidates)
        residuals = (
            GRADED_ORACLE_RESIDUAL_RANGE
            * mx.tanh(raw_residual)
            * base.candidate_mask
        )
        standard_errors = (
            nn.softplus(self.standard_error_head(output).reshape(groups, candidates))
            + 1e-4
        ) * base.candidate_mask
        refill_logits = self.refill_trunk(
            mx.concatenate([supply_vector, supply_pool], axis=-1)
        )
        return S1ExactSupplyPrediction(
            scores=base.screen_value + residuals,
            residuals=residuals,
            standard_errors=standard_errors,
            refill_probabilities=mx.softmax(refill_logits, axis=-1),
        )


def s1_exact_supply_loss_components(
    model: S1ExactSupplyRanker,
    batch: object,
) -> dict[str, mx.array]:
    """Apply the frozen graded objective plus one shared refill-decoding target."""
    prediction = model(batch)
    base = batch.base
    scores = prediction.scores
    residuals = prediction.residuals
    r1200_regression = _uncertainty_weighted_huber(
        residuals,
        base.r1200_mean - base.screen_value,
        base.r1200_stddev,
        base.r1200_samples,
        base.r1200_mask,
    )
    r4800_regression = _uncertainty_weighted_huber(
        residuals,
        base.r4800_mean - base.screen_value,
        base.r4800_stddev,
        base.r4800_samples,
        base.r4800_mask,
    )
    r1200_listwise = _masked_soft_target_cross_entropy(
        scores,
        base.r1200_mean,
        base.r1200_mask,
        temperature=2.0,
    )
    r4800_winner = _hard_cross_entropy(
        scores,
        base.candidate_mask,
        base.selected_index,
        temperature=1.0,
    )
    highest_mean = mx.where(
        base.r4800_mask,
        base.r4800_mean,
        mx.where(base.r1200_mask, base.r1200_mean, base.r600_mean),
    )
    highest_stddev = mx.where(
        base.r4800_mask,
        base.r4800_stddev,
        mx.where(base.r1200_mask, base.r1200_stddev, base.r600_stddev),
    )
    highest_samples = mx.where(
        base.r4800_mask,
        base.r4800_samples,
        mx.where(base.r1200_mask, base.r1200_samples, base.r600_samples),
    )
    scored_mask = base.r4800_mask | base.r1200_mask | base.r600_mask
    teacher_standard_error = _teacher_standard_error(
        highest_stddev,
        highest_samples,
    )
    predicted_variance = mx.maximum(prediction.standard_errors**2, 1e-8)
    gaussian = mx.log(mx.maximum(prediction.standard_errors, 1e-4)) + (
        teacher_standard_error**2 + (scores - highest_mean) ** 2
    ) / (2.0 * predicted_variance)
    standard_error_calibration = _masked_mean(gaussian, scored_mask)
    screen_only_regularization = _masked_mean(
        residuals**2,
        base.candidate_mask & ~scored_mask,
    )
    refill_cross_entropy = -mx.mean(
        mx.sum(
            batch.refill_target
            * mx.log(mx.maximum(prediction.refill_probabilities, 1e-9)),
            axis=-1,
        )
    )
    return {
        "r1200_huber": r1200_regression,
        "r4800_huber": r4800_regression,
        "r1200_listwise": r1200_listwise,
        "r4800_winner": r4800_winner,
        "standard_error_calibration": standard_error_calibration,
        "screen_only_regularization": screen_only_regularization,
        "refill_cross_entropy": refill_cross_entropy,
    }


def s1_exact_supply_loss(model: S1ExactSupplyRanker, batch: object) -> mx.array:
    """Return the exact ADR 0147 scalar objective."""
    components = s1_exact_supply_loss_components(model, batch)
    return (
        components["r1200_huber"]
        + 4.0 * components["r4800_huber"]
        + 0.5 * components["r1200_listwise"]
        + components["r4800_winner"]
        + 0.1 * components["standard_error_calibration"]
        + 0.01 * components["screen_only_regularization"]
        + REFILL_LOSS_WEIGHT * components["refill_cross_entropy"]
    )


def score_s1_exact_supply_batch(
    model: S1ExactSupplyRanker,
    batch: object,
) -> mx.array:
    return model(batch).scores


def parameter_count(model: S1ExactSupplyRanker) -> int:
    """Return the trainable scalar count used for cross-arm equality."""
    return sum(int(value.size) for _, value in tree_flatten(model.trainable_parameters()))


def parameter_layout_blake3(model: S1ExactSupplyRanker) -> str:
    """Hash every trainable parameter name, shape, and dtype."""
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(model.trainable_parameters())
    ]
    payload = json.dumps(
        layout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _teacher_standard_error(stddev: mx.array, samples: mx.array) -> mx.array:
    empirical_variance = stddev**2 / mx.maximum(samples, 1.0)
    return mx.sqrt(empirical_variance + GRADED_ORACLE_UNCERTAINTY_FLOOR**2)


def _uncertainty_weighted_huber(
    prediction: mx.array,
    target: mx.array,
    stddev: mx.array,
    samples: mx.array,
    mask: mx.array,
) -> mx.array:
    uncertainty = _teacher_standard_error(stddev, samples)
    error = (prediction - target) / uncertainty
    absolute = mx.abs(error)
    huber = mx.where(absolute <= 1.0, 0.5 * error**2, absolute - 0.5)
    return _masked_mean(huber, mask)


def _masked_soft_target_cross_entropy(
    scores: mx.array,
    targets: mx.array,
    mask: mx.array,
    *,
    temperature: float,
) -> mx.array:
    masked_scores = mx.where(mask, scores / temperature, -1e9)
    masked_targets = mx.where(mask, targets / temperature, -1e9)
    target_probabilities = mx.softmax(masked_targets, axis=-1)
    log_probabilities = masked_scores - mx.logsumexp(
        masked_scores,
        axis=-1,
        keepdims=True,
    )
    per_group = -mx.sum(
        mx.where(mask, target_probabilities * log_probabilities, 0.0),
        axis=-1,
    )
    return _masked_mean(per_group, mx.any(mask, axis=-1))


def _hard_cross_entropy(
    scores: mx.array,
    mask: mx.array,
    selected_index: mx.array,
    *,
    temperature: float,
) -> mx.array:
    masked = mx.where(mask, scores / temperature, -1e9)
    selected_mask = mx.arange(scores.shape[-1])[None, :] == selected_index[:, None]
    selected_score = mx.sum(mx.where(selected_mask, masked, 0.0), axis=-1)
    return mx.mean(mx.logsumexp(masked, axis=-1) - selected_score)


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    return mx.sum(values * weights) / mx.maximum(mx.sum(weights), 1.0)
