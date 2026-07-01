"""Frozen iso-architecture MLX model for the R0 spatial tournament."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.r0_spatial_mlx_cache import (
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    TARGET_DIM,
    TOKEN_FIELDS,
)

TARGET_SCALES = mx.array(
    [23.0, 23.0, 23.0, 23.0, 23.0, 30.0, 28.0, 28.0, 28.0, 40.0, 20.0],
    dtype=mx.float32,
)


@dataclass(frozen=True)
class R0SpatialMlxModelConfig:
    """The only architecture admitted to the R0 iso-architecture screen."""

    schema_version: int = 1
    architecture: str = "r0-spatial-iso-set-value-v1"
    hidden_dim: int = 32
    attention_heads: int = 4
    board_blocks: int = 1
    feed_forward_multiplier: int = 2

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "r0-spatial-iso-set-value-v1":
            raise ValueError("unsupported R0 spatial model schema")
        if self.hidden_dim != 32:
            raise ValueError("R0 freezes hidden_dim at 32")
        if self.attention_heads != 4:
            raise ValueError("R0 freezes attention_heads at 4")
        if self.board_blocks != 1:
            raise ValueError("R0 freezes one board attention block")
        if self.feed_forward_multiplier != 2:
            raise ValueError("R0 freezes the feed-forward multiplier at two")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> R0SpatialMlxModelConfig:
        config = cls(**values)
        config.validate()
        return config


class _MaskedAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, multiplier: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiHeadAttention(hidden_dim, heads, bias=True)
        self.norm_feed_forward = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * multiplier),
            nn.GELU(),
            nn.Linear(hidden_dim * multiplier, hidden_dim),
        )

    def __call__(self, values: mx.array, mask: mx.array) -> mx.array:
        normalized = self.norm_attention(values)
        attention_mask = mx.where(mask[:, None, None, :], 0.0, -1e9)
        values = values + self.attention(
            normalized,
            normalized,
            normalized,
            mask=attention_mask,
        )
        values = values + self.feed_forward(self.norm_feed_forward(values))
        return values * mask[..., None]


class R0SpatialIsoValueModel(nn.Module):
    """One parameterization whose sequence length is the tournament variable."""

    def __init__(self, config: R0SpatialMlxModelConfig | None = None):
        super().__init__()
        config = config or R0SpatialMlxModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim

        self.coordinate_projection = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
        )
        self.path_embedding = nn.Embedding(4, hidden)
        self.terrain_a_embedding = nn.Embedding(5, hidden)
        self.terrain_b_embedding = nn.Embedding(6, hidden)
        self.rotation_embedding = nn.Embedding(6, hidden)
        self.allowed_wildlife_embedding = nn.Embedding(32, hidden)
        self.placed_wildlife_embedding = nn.Embedding(6, hidden)
        self.keystone_embedding = nn.Embedding(2, hidden)
        self.token_normalization = nn.LayerNorm(hidden)
        self.seat_embedding = nn.Embedding(4, hidden)
        self.board_blocks = [
            _MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.board_blocks)
        ]
        self.market_projection = nn.Sequential(
            nn.Linear(MARKET_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 11, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, TARGET_DIM),
        )

    def __call__(
        self,
        spatial_tokens: mx.array,
        spatial_mask: mx.array,
        market_features: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        if spatial_tokens.shape[-1] != TOKEN_FIELDS:
            raise ValueError("R0 spatial token width drifted")
        batch_size, boards, _tokens, _fields = spatial_tokens.shape
        if boards != 4:
            raise ValueError("R0 spatial model requires four relative board slots")
        hidden = self.config.hidden_dim
        token_values = spatial_tokens.astype(mx.int32)
        coordinates = token_values[..., :4].astype(mx.float32) / 24.0
        encoded = (
            self.coordinate_projection(coordinates)
            + self.path_embedding(token_values[..., 4])
            + self.terrain_a_embedding(token_values[..., 5])
            + self.terrain_b_embedding(token_values[..., 6])
            + self.rotation_embedding(token_values[..., 7])
            + self.allowed_wildlife_embedding(token_values[..., 8])
            + self.placed_wildlife_embedding(token_values[..., 9])
            + self.keystone_embedding(token_values[..., 10])
        )
        encoded = self.token_normalization(encoded)
        encoded = encoded + self.seat_embedding(mx.arange(4))[None, :, None, :]
        encoded = encoded * spatial_mask[..., None]
        encoded = encoded.reshape(batch_size * 4, spatial_tokens.shape[2], hidden)
        flat_mask = spatial_mask.reshape(batch_size * 4, spatial_tokens.shape[2])
        for block in self.board_blocks:
            encoded = block(encoded, flat_mask)
        board_summary = _masked_pool(encoded, flat_mask).reshape(batch_size, hidden * 8)

        market = self.market_projection(market_features)
        market_summary = _masked_pool(market, market_mask)
        global_summary = self.global_projection(global_features)
        combined = mx.concatenate([board_summary, market_summary, global_summary], axis=-1)
        return nn.softplus(self.trunk(combined) - 2.0)

    def predict_components(
        self,
        spatial_tokens: mx.array,
        spatial_mask: mx.array,
        market_features: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        return (
            self(
                spatial_tokens,
                spatial_mask,
                market_features,
                market_mask,
                global_features,
            )
            * TARGET_SCALES
        )


def r0_spatial_value_loss(model: R0SpatialIsoValueModel, batch: object) -> mx.array:
    predictions = model(
        batch.spatial_tokens,
        batch.spatial_mask,
        batch.market_features,
        batch.market_mask,
        batch.global_features,
    )
    normalized_targets = batch.targets / TARGET_SCALES
    component_loss = mx.mean(mx.square(predictions - normalized_targets))
    predicted_total = mx.sum(predictions * TARGET_SCALES, axis=-1)
    target_total = mx.sum(batch.targets, axis=-1)
    total_loss = mx.mean(mx.square((predicted_total - target_total) / 100.0))
    return component_loss + 0.5 * total_loss


def parameter_count(model: R0SpatialIsoValueModel) -> int:
    """Return the exact trainable scalar count for cross-arm equality checks."""
    return sum(int(value.size) for _, value in tree_flatten(model.trainable_parameters()))


def _masked_pool(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask[..., None]
    count = mx.maximum(mx.sum(weights, axis=1), 1.0)
    mean = mx.sum(values * weights, axis=1) / count
    maximum = mx.max(mx.where(weights, values, -1e9), axis=1)
    maximum = mx.where(mx.any(mask, axis=1, keepdims=True), maximum, 0.0)
    return mx.concatenate([mean, maximum], axis=-1)
