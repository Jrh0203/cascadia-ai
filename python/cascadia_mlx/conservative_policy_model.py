"""Balanced groupwise MLX policy for conservative terminal improvement."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.conservative_advantage_model import (
    encode_pair_side,
    initialize_pair_encoder,
)


@dataclass(frozen=True)
class ConservativePolicyModelConfig:
    schema_version: int = 1
    architecture: str = "conservative-policy-v2"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    feed_forward_multiplier: int = 3
    challenger_group_weight: float = 3.56
    regression_weight: float = 0.25

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "conservative-policy-v2":
            raise ValueError("unsupported conservative-policy model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.board_blocks < 0 or self.market_blocks < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive")
        if self.challenger_group_weight <= 0:
            raise ValueError("challenger_group_weight must be positive")
        if self.regression_weight < 0:
            raise ValueError("regression_weight cannot be negative")

    def to_dict(self) -> dict[str, float | int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ConservativePolicyModelConfig:
        config = cls(**values)
        config.validate()
        return config


class ConservativePolicyModel(nn.Module):
    """Predict both c90 lower bounds and anchor-relative policy logits."""

    def __init__(self, config: ConservativePolicyModelConfig | None = None):
        super().__init__()
        config = config or ConservativePolicyModelConfig()
        config.validate()
        self.config = config
        initialize_pair_encoder(self, config)
        hidden = config.hidden_dim
        self.regression_trunk = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1),
        )
        self.policy_trunk = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1),
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
    ) -> tuple[mx.array, mx.array]:
        anchor = encode_pair_side(
            self,
            self.config,
            anchor_board_entities,
            anchor_board_mask,
            anchor_market_entities,
            anchor_market_mask,
            anchor_global_features,
            anchor_action_features,
        )
        candidate = encode_pair_side(
            self,
            self.config,
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
        return (
            self.regression_trunk(pair).squeeze(-1),
            self.policy_trunk(pair).squeeze(-1),
        )


def conservative_policy_outputs(
    model: ConservativePolicyModel,
    batch: object,
) -> tuple[mx.array, mx.array]:
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


def conservative_policy_loss(
    model: ConservativePolicyModel,
    batch: object,
) -> mx.array:
    lower_bound, policy_logits = conservative_policy_outputs(model, batch)
    mask = batch.candidate_mask
    selected = batch.selected

    anchor_logits = mx.zeros((policy_logits.shape[0], 1))
    masked_challengers = mx.where(mask, policy_logits, -1e9)
    action_logits = mx.concatenate([anchor_logits, masked_challengers], axis=1)
    selected_group = mx.any(selected, axis=1)
    target = mx.concatenate([(~selected_group)[:, None], selected], axis=1)
    log_normalizer = mx.logsumexp(action_logits, axis=1)
    chosen_logit = mx.sum(mx.where(target, action_logits, 0.0), axis=1)
    group_weight = mx.where(
        selected_group,
        model.config.challenger_group_weight,
        1.0,
    )
    policy_loss = mx.sum((log_normalizer - chosen_logit) * group_weight) / mx.sum(group_weight)

    boundary_weight = 1.0 + (mx.abs(batch.lower_bound) <= 1.0) + selected
    squared_error = (lower_bound - batch.lower_bound) ** 2
    regression_loss = mx.sum(mx.where(mask, squared_error * boundary_weight, 0.0)) / mx.sum(
        mx.where(mask, boundary_weight, 0.0)
    )
    return policy_loss + model.config.regression_weight * regression_loss
