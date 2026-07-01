"""MLX scalar continuation model for public beam-state values."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.action_ranking_model import (
    encode_action_afterstates,
    initialize_action_afterstate_encoder,
)

SCORE_TO_GO_SCALE = 20.0


@dataclass(frozen=True)
class PublicBeamValueModelConfig:
    """Serializable public continuation-value architecture."""

    schema_version: int = 1
    architecture: str = "mlx-public-beam-value-v1"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "mlx-public-beam-value-v1":
            raise ValueError("unsupported public beam-value model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.board_blocks < 0 or self.market_blocks < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> PublicBeamValueModelConfig:
        config = cls(**values)
        config.validate()
        return config


class PublicBeamValueModel(nn.Module):
    """Predict terminal score from one public action afterstate."""

    def __init__(self, config: PublicBeamValueModelConfig | None = None):
        super().__init__()
        config = config or PublicBeamValueModelConfig()
        config.validate()
        self.config = config
        initialize_action_afterstate_encoder(self, config)
        hidden = config.hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 12, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1),
        )

    def score_to_go(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        action_features: mx.array,
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
        return self.trunk(encoded).reshape(groups, candidates) * SCORE_TO_GO_SCALE

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        action_features: mx.array,
        current_base_score: mx.array,
    ) -> mx.array:
        return current_base_score + self.score_to_go(
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            action_features,
        )


def public_beam_value_scores(model: PublicBeamValueModel, batch: object) -> mx.array:
    return model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
        batch.current_base_score,
    )


def public_beam_value_loss(model: PublicBeamValueModel, batch: object) -> mx.array:
    """Fit terminal score and within-decision advantages with listwise support."""
    mask = batch.candidate_mask
    predictions = public_beam_value_scores(model, batch)
    targets = batch.target_mean
    uncertainty = mx.sqrt(
        (mx.square(batch.batch_a_stddev) + mx.square(batch.batch_b_stddev)) / 32.0
    )
    weights = mx.where(mask, 1.0 / (1.0 + uncertainty), 0.0)

    absolute = _huber(predictions - targets)
    candidate_count = mx.maximum(mx.sum(mask, axis=-1, keepdims=True), 1)
    prediction_mean = mx.sum(mx.where(mask, predictions, 0.0), axis=-1, keepdims=True)
    prediction_mean /= candidate_count
    target_mean = mx.sum(mx.where(mask, targets, 0.0), axis=-1, keepdims=True)
    target_mean /= candidate_count
    centered = _huber((predictions - prediction_mean) - (targets - target_mean))

    masked_predictions = mx.where(mask, predictions, -1e9)
    masked_targets = mx.where(mask, targets, -1e9)
    teacher_probabilities = mx.softmax(masked_targets, axis=-1)
    log_probabilities = masked_predictions - mx.logsumexp(
        masked_predictions,
        axis=-1,
        keepdims=True,
    )
    listwise = -mx.sum(teacher_probabilities * log_probabilities, axis=-1)

    regression_weight = mx.sum(weights)
    regression = mx.sum(weights * (absolute + centered)) / regression_weight
    group_weight = mx.sum(weights, axis=-1) / candidate_count.squeeze(-1)
    ranking = mx.sum(group_weight * listwise) / mx.sum(group_weight)
    return regression + 0.25 * ranking


def _huber(errors: mx.array) -> mx.array:
    absolute = mx.abs(errors)
    return mx.where(absolute <= 1.0, 0.5 * mx.square(errors), absolute - 0.5)
