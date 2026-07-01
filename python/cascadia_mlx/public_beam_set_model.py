"""Joint candidate-set ranker for public beam continuation decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.action_ranking_model import (
    encode_action_afterstates,
    initialize_action_afterstate_encoder,
)
from cascadia_mlx.model import SetAttentionBlock

CORRECTION_SCALE = 4.0
TEACHER_TEMPERATURE = 0.5


@dataclass(frozen=True)
class PublicBeamSetModelConfig:
    """Serializable joint candidate-set architecture."""

    schema_version: int = 1
    architecture: str = "mlx-public-beam-set-ranker-v1"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    candidate_blocks: int = 2
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "mlx-public-beam-set-ranker-v1":
            raise ValueError("unsupported public beam set-ranker configuration")
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
    def from_dict(cls, values: dict[str, object]) -> PublicBeamSetModelConfig:
        config = cls(**values)
        config.validate()
        return config


class PublicBeamSetRanker(nn.Module):
    """Rank a complete legal action set with an immediate-score residual anchor."""

    def __init__(self, config: PublicBeamSetModelConfig | None = None):
        super().__init__()
        config = config or PublicBeamSetModelConfig()
        config.validate()
        self.config = config
        initialize_action_afterstate_encoder(self, config)
        hidden = config.hidden_dim
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden * 12, hidden * 2),
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

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        action_features: mx.array,
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
        values = self.candidate_projection(encoded).reshape(
            groups,
            candidates,
            self.config.hidden_dim,
        )
        values = values * candidate_mask[..., None]
        for block in self.candidate_blocks:
            values = block(values, candidate_mask)
        correction = mx.tanh(self.correction_head(values).squeeze(-1)) * CORRECTION_SCALE
        return immediate_score + correction


def public_beam_set_scores(model: PublicBeamSetRanker, batch: object) -> mx.array:
    return model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.action_features,
        batch.immediate_score,
        batch.candidate_mask,
    )


def public_beam_set_loss(model: PublicBeamSetRanker, batch: object) -> mx.array:
    """Optimize centered value, exact top ties, and soft listwise ordering."""
    mask = batch.candidate_mask
    predictions = public_beam_set_scores(model, batch)
    targets = batch.target_mean
    candidate_count = mx.maximum(mx.sum(mask, axis=-1, keepdims=True), 1)

    prediction_mean = mx.sum(mx.where(mask, predictions, 0.0), axis=-1, keepdims=True)
    prediction_mean /= candidate_count
    target_mean = mx.sum(mx.where(mask, targets, 0.0), axis=-1, keepdims=True)
    target_mean /= candidate_count
    centered_error = (predictions - prediction_mean) - (targets - target_mean)
    centered = mx.sum(mx.where(mask, _huber(centered_error), 0.0)) / mx.sum(mask)

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
    hard_top = -mx.mean(mx.sum(hard_top_probabilities * log_probabilities, axis=-1))

    teacher_logits = mx.where(mask, targets / TEACHER_TEMPERATURE, -1e9)
    teacher_probabilities = mx.softmax(teacher_logits, axis=-1)
    soft_listwise = -mx.mean(mx.sum(teacher_probabilities * log_probabilities, axis=-1))
    return centered + 0.50 * hard_top + 0.25 * soft_listwise


def _huber(errors: mx.array) -> mx.array:
    absolute = mx.abs(errors)
    return mx.where(absolute <= 1.0, 0.5 * mx.square(errors), absolute - 0.5)
