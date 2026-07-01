"""MLX complete-candidate-set ranker for qualified R12 advantages."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.action_ranking_model import (
    encode_action_afterstates,
    initialize_action_afterstate_encoder,
)
from cascadia_mlx.counterfactual_advantage_dataset import (
    COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.model import SetAttentionBlock

CORRECTION_SCALE = 4.0
TEACHER_TEMPERATURE = 0.5
HARD_TOP_WEIGHT = 0.50
SOFT_LISTWISE_WEIGHT = 0.25


@dataclass(frozen=True)
class CounterfactualAdvantageModelConfig:
    """Serializable frozen ADR 0078 architecture."""

    schema_version: int = 1
    architecture: str = "mlx-r12-counterfactual-advantage-set-ranker-v1"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    candidate_blocks: int = 2
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.architecture != "mlx-r12-counterfactual-advantage-set-ranker-v1"
        ):
            raise ValueError("unsupported counterfactual-advantage model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if min(self.board_blocks, self.market_blocks, self.candidate_blocks) < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> CounterfactualAdvantageModelConfig:
        config = cls(**values)
        config.validate()
        return config


class CounterfactualAdvantageRanker(nn.Module):
    """Rank four actions from observable afterstates and exact public supply."""

    def __init__(self, config: CounterfactualAdvantageModelConfig | None = None):
        super().__init__()
        config = config or CounterfactualAdvantageModelConfig()
        config.validate()
        self.config = config
        initialize_action_afterstate_encoder(self, config)
        hidden = config.hidden_dim
        self.public_supply_projection = nn.Sequential(
            nn.Linear(COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden * 13, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.candidate_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.candidate_blocks)
        ]
        self.correction_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        output = self.correction_head.layers[-1]
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
        public_supply: mx.array,
        immediate_score: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        groups, candidates = global_features.shape[:2]
        encoded = encode_action_afterstates(
            self,
            self.config,
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            action_features,
        )
        supply = self.public_supply_projection(public_supply)
        supply = mx.repeat(supply[:, None, :], candidates, axis=1).reshape(
            groups * candidates,
            self.config.hidden_dim,
        )
        values = self.candidate_projection(mx.concatenate([encoded, supply], axis=-1))
        values = values.reshape(groups, candidates, self.config.hidden_dim)
        values = values * candidate_mask[..., None]
        for block in self.candidate_blocks:
            values = block(values, candidate_mask)
        correction = mx.tanh(self.correction_head(values).squeeze(-1)) * CORRECTION_SCALE
        return immediate_score + correction


def counterfactual_advantage_scores(
    model: CounterfactualAdvantageRanker,
    batch: object,
) -> mx.array:
    return model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
        batch.public_supply,
        batch.immediate_score,
        batch.candidate_mask,
    )


def counterfactual_advantage_loss(
    model: CounterfactualAdvantageRanker,
    batch: object,
) -> mx.array:
    """Fit centered value and decision ordering with R12 uncertainty weights."""
    mask = batch.candidate_mask
    predictions = counterfactual_advantage_scores(model, batch)
    targets = batch.target_mean
    candidate_weights = mx.where(
        mask,
        1.0 / (1.0 + batch.target_standard_error),
        0.0,
    )
    candidate_count = mx.maximum(mx.sum(mask, axis=-1, keepdims=True), 1)

    prediction_mean = mx.sum(mx.where(mask, predictions, 0.0), axis=-1, keepdims=True)
    prediction_mean /= candidate_count
    target_mean = mx.sum(mx.where(mask, targets, 0.0), axis=-1, keepdims=True)
    target_mean /= candidate_count
    centered_error = (predictions - prediction_mean) - (targets - target_mean)
    centered = mx.sum(candidate_weights * _huber(centered_error)) / mx.sum(candidate_weights)

    masked_predictions = mx.where(mask, predictions, -1e9)
    log_probabilities = masked_predictions - mx.logsumexp(
        masked_predictions,
        axis=-1,
        keepdims=True,
    )
    target_max = mx.max(mx.where(mask, targets, -1e9), axis=-1, keepdims=True)
    hard_top_mask = mask & (mx.abs(targets - target_max) < 1e-6)
    hard_top_count = mx.maximum(mx.sum(hard_top_mask, axis=-1, keepdims=True), 1)
    hard_top_probabilities = hard_top_mask / hard_top_count
    hard_top = -mx.sum(hard_top_probabilities * log_probabilities, axis=-1)

    teacher_logits = mx.where(mask, targets / TEACHER_TEMPERATURE, -1e9)
    teacher_probabilities = mx.softmax(teacher_logits, axis=-1)
    soft_listwise = -mx.sum(teacher_probabilities * log_probabilities, axis=-1)

    group_weights = mx.sum(candidate_weights, axis=-1) / candidate_count.squeeze(-1)
    group_weights /= mx.sum(group_weights)
    ranking = mx.sum(
        group_weights * (HARD_TOP_WEIGHT * hard_top + SOFT_LISTWISE_WEIGHT * soft_listwise)
    )
    return centered + ranking


def _huber(errors: mx.array) -> mx.array:
    absolute = mx.abs(errors)
    return mx.where(absolute <= 1.0, 0.5 * mx.square(errors), absolute - 0.5)
