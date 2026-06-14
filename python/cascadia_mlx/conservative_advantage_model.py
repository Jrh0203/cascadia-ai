"""Shared-weight MLX model for paired c90 lower-bound prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.action_ranking_model import (
    encode_action_afterstates,
    initialize_action_afterstate_encoder,
)


@dataclass(frozen=True)
class ConservativeAdvantageModelConfig:
    schema_version: int = 1
    architecture: str = "conservative-advantage-v1"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "conservative-advantage-v1":
            raise ValueError("unsupported conservative-advantage model configuration")
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
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> ConservativeAdvantageModelConfig:
        config = cls(**values)
        config.validate()
        return config


class ConservativeAdvantageModel(nn.Module):
    """Regress candidate-minus-anchor c90 lower bounds in score points."""

    def __init__(self, config: ConservativeAdvantageModelConfig | None = None):
        super().__init__()
        config = config or ConservativeAdvantageModelConfig()
        config.validate()
        self.config = config
        initialize_pair_encoder(self, config)
        hidden = config.hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1),
        )

    def _encode(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        action_features: mx.array,
    ) -> mx.array:
        return encode_pair_side(
            self,
            self.config,
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            action_features,
        )

    def __call__(
        self,
        anchor_board_entities: mx.array,
        anchor_board_mask: mx.array,
        anchor_market_entities: mx.array,
        anchor_market_mask: mx.array,
        anchor_global_features: mx.array,
        anchor_action_features: mx.array,
        candidate_board_entities: mx.array,
        candidate_board_mask: mx.array,
        candidate_market_entities: mx.array,
        candidate_market_mask: mx.array,
        candidate_global_features: mx.array,
        candidate_action_features: mx.array,
    ) -> mx.array:
        anchor = self._encode(
            anchor_board_entities,
            anchor_board_mask,
            anchor_market_entities,
            anchor_market_mask,
            anchor_global_features,
            anchor_action_features,
        )
        candidate = self._encode(
            candidate_board_entities,
            candidate_board_mask,
            candidate_market_entities,
            candidate_market_mask,
            candidate_global_features,
            candidate_action_features,
        )
        pair = mx.concatenate(
            [anchor, candidate, candidate - anchor, candidate * anchor],
            axis=-1,
        )
        return self.trunk(pair).squeeze(-1)


def conservative_advantage_scores(
    model: ConservativeAdvantageModel,
    batch: object,
) -> mx.array:
    return model(
        batch.anchor_board_entities,
        batch.anchor_board_mask,
        batch.anchor_market_entities,
        batch.anchor_market_mask,
        batch.anchor_global_features,
        batch.anchor_action_features,
        batch.candidate_board_entities,
        batch.candidate_board_mask,
        batch.candidate_market_entities,
        batch.candidate_market_mask,
        batch.candidate_global_features,
        batch.candidate_action_features,
    )


def conservative_advantage_loss(
    model: ConservativeAdvantageModel,
    batch: object,
) -> mx.array:
    predictions = conservative_advantage_scores(model, batch)
    mask = batch.candidate_mask
    targets = batch.lower_bound
    boundary_weight = 1.0 + (mx.abs(targets) <= 1.0) + batch.selected
    squared_error = (predictions - targets) ** 2
    return mx.sum(mx.where(mask, squared_error * boundary_weight, 0.0)) / mx.sum(
        mx.where(mask, boundary_weight, 0.0)
    )


def initialize_pair_encoder(model: nn.Module, config: object) -> None:
    """Attach the shared action-afterstate encoder without nesting parameter names."""
    initialize_action_afterstate_encoder(model, config)
    hidden = config.hidden_dim
    model.summary_projection = nn.Sequential(
        nn.Linear(hidden * 12, hidden * 2),
        nn.GELU(),
        nn.LayerNorm(hidden * 2),
    )


def encode_pair_side(
    model: nn.Module,
    config: object,
    board_entities: mx.array,
    board_mask: mx.array,
    market_entities: mx.array,
    market_mask: mx.array,
    global_features: mx.array,
    action_features: mx.array,
) -> mx.array:
    """Encode one side of each anchor/challenger pair."""
    groups, candidates = global_features.shape[:2]
    encoded = encode_action_afterstates(
        model,
        config,
        board_entities,
        board_mask,
        market_entities,
        market_mask,
        global_features,
        action_features,
    )
    return model.summary_projection(encoded).reshape(groups, candidates, -1)
