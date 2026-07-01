"""MLX entity-set models for Cascadia value prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM, TARGET_DIM

TARGET_SCALES = mx.array(
    [23.0, 23.0, 23.0, 23.0, 23.0, 30.0, 28.0, 28.0, 28.0, 40.0, 20.0],
    dtype=mx.float32,
)


@dataclass(frozen=True)
class ModelConfig:
    """Serializable architecture configuration."""

    schema_version: int = 1
    architecture: str = "entity-set-value-v1"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "entity-set-value-v1":
            raise ValueError("unsupported model configuration")
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
    def from_dict(cls, values: dict[str, object]) -> ModelConfig:
        config = cls(**values)
        config.validate()
        return config


class SetAttentionBlock(nn.Module):
    """Pre-norm self-attention over a masked entity set."""

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


class EntitySetValueModel(nn.Module):
    """Predict decomposed final base score from public game entities."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        config = config or ModelConfig()
        config.validate()
        self.config = config
        initialize_entity_encoder(self, config)
        hidden = config.hidden_dim
        combined_dim = hidden * 11
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, TARGET_DIM),
        )

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        combined = encode_entity_state(
            self,
            self.config,
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
        )
        return nn.softplus(self.trunk(combined) - 2.0)

    def predict_components(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        """Return score components in points rather than normalized units."""
        return (
            self(
                board_entities,
                board_mask,
                market_entities,
                market_mask,
                global_features,
            )
            * TARGET_SCALES
        )


def value_loss(model: EntitySetValueModel, batch: object) -> mx.array:
    """Weighted normalized MSE with an explicit total-score consistency term."""
    predictions = model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
    )
    normalized_targets = batch.targets / TARGET_SCALES
    component_loss = mx.mean(mx.square(predictions - normalized_targets))
    predicted_total = mx.sum(predictions * TARGET_SCALES, axis=-1)
    target_total = mx.sum(batch.targets, axis=-1)
    total_loss = mx.mean(mx.square((predicted_total - target_total) / 100.0))
    return component_loss + 0.5 * total_loss


def initialize_entity_encoder(model: nn.Module, config: object) -> None:
    """Attach the shared entity encoder while preserving flat checkpoint names."""
    hidden = config.hidden_dim
    model.board_projection = nn.Sequential(
        nn.Linear(ENTITY_DIM, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
    )
    model.market_projection = nn.Sequential(
        nn.Linear(ENTITY_DIM, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
    )
    model.seat_embedding = nn.Embedding(4, hidden)
    model.board_blocks = [
        SetAttentionBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        for _ in range(config.board_blocks)
    ]
    model.market_blocks = [
        SetAttentionBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        for _ in range(config.market_blocks)
    ]
    model.global_projection = nn.Sequential(
        nn.Linear(GLOBAL_DIM, hidden * 2),
        nn.GELU(),
        nn.Linear(hidden * 2, hidden),
        nn.GELU(),
    )


def encode_entity_state(
    model: nn.Module,
    config: object,
    board_entities: mx.array,
    board_mask: mx.array,
    market_entities: mx.array,
    market_mask: mx.array,
    global_features: mx.array,
) -> mx.array:
    """Encode boards, market, and global features for a value head."""
    batch_size = board_entities.shape[0]
    hidden = config.hidden_dim
    boards = model.board_projection(board_entities)
    seats = model.seat_embedding(mx.arange(4))[None, :, None, :]
    boards = boards + seats
    boards = boards.reshape(batch_size * 4, 23, hidden)
    flat_board_mask = board_mask.reshape(batch_size * 4, 23)
    boards = boards * flat_board_mask[..., None]
    for block in model.board_blocks:
        boards = block(boards, flat_board_mask)
    board_summary = _masked_pool(boards, flat_board_mask).reshape(batch_size, -1)

    market = model.market_projection(market_entities)
    for block in model.market_blocks:
        market = block(market, market_mask)
    market_summary = _masked_pool(market, market_mask)

    global_summary = model.global_projection(global_features)
    return mx.concatenate([board_summary, market_summary, global_summary], axis=-1)


def _masked_pool(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask[..., None]
    count = mx.maximum(mx.sum(weights, axis=1), 1.0)
    mean = mx.sum(values * weights, axis=1) / count
    maximum = mx.max(mx.where(weights, values, -1e9), axis=1)
    has_values = mx.any(mask, axis=1, keepdims=True)
    maximum = mx.where(has_values, maximum, 0.0)
    return mx.concatenate([mean, maximum], axis=-1)
