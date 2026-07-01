"""Iso-parameter universal-parent ranker for ADR 0156."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    MaskedAttentionBlock,
    PerceiverCrossBlock,
    masked_pool,
)
from cascadia_mlx.r3_action_edit_mlx_cache import CONTROL_ARM as R3_CONTROL_ARM
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    r3_action_edit_loss,
    r3_action_edit_loss_components,
)
from cascadia_mlx.r4_bounded_parent_mlx_cache import (
    ARMS,
    UNIVERSAL_PARENT_CLASS_COUNT,
    UNIVERSAL_PARENT_VALUE_WIDTH,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "r4-bounded-universal-parent-independent-ranker-v1"


@dataclass(frozen=True)
class R4BoundedParentModelConfig:
    """The only model graph admitted to the ADR 0156 comparison."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    arm: str = ARMS[0]
    hidden_dim: int = 64
    attention_heads: int = 4
    parent_perceiver_latents: int = 16
    candidate_perceiver_latents: int = 8
    parent_latent_blocks: int = 1
    candidate_latent_blocks: int = 1
    cross_board_blocks: int = 1
    staged_market_blocks: int = 1
    feed_forward_multiplier: int = 2

    def validate(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION or self.architecture != ARCHITECTURE:
            raise ValueError("unsupported R4 bounded-parent model schema")
        if self.arm not in ARMS:
            raise ValueError("R4 parent model names an unknown comparison arm")
        if self.hidden_dim != 64 or self.attention_heads != 4:
            raise ValueError("ADR 0156 freezes hidden width 64 and four heads")
        if self.parent_perceiver_latents != 16 or self.candidate_perceiver_latents != 8:
            raise ValueError("ADR 0156 Perceiver latent counts drifted")
        if (
            self.parent_latent_blocks != 1
            or self.candidate_latent_blocks != 1
            or self.cross_board_blocks != 1
            or self.staged_market_blocks != 1
        ):
            raise ValueError("ADR 0156 attention-block counts drifted")
        if self.feed_forward_multiplier != 2:
            raise ValueError("ADR 0156 freezes feed-forward multiplier two")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> R4BoundedParentModelConfig:
        config = cls(**values)
        config.validate()
        return config


class UniversalParentEncoder(nn.Module):
    """Nine semantic adapters over compact board-local parent token sets."""

    def __init__(self, config: R4BoundedParentModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        scale = (2.0 / UNIVERSAL_PARENT_VALUE_WIDTH) ** 0.5
        self.token_weights = (
            mx.random.normal(
                (
                    UNIVERSAL_PARENT_CLASS_COUNT,
                    UNIVERSAL_PARENT_VALUE_WIDTH,
                    hidden,
                )
            )
            * scale
        )
        self.token_bias = mx.zeros((UNIVERSAL_PARENT_CLASS_COUNT, hidden))
        self.token_activation = nn.GELU()
        self.token_norm = nn.LayerNorm(hidden)
        self.seat_embedding = nn.Embedding(BOARD_SLOTS, hidden)
        self.market_projection = nn.Sequential(
            nn.Linear(MARKET_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.player_projection = nn.Sequential(
            nn.Linear(PLAYER_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.latents = mx.random.normal((config.parent_perceiver_latents, hidden)) * 0.02
        self.perceiver_cross = PerceiverCrossBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.perceiver_blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.parent_latent_blocks)
        ]
        self.board_summary_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cross_board_blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.cross_board_blocks)
        ]
        self.state_summary_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def __call__(self, batch: object) -> mx.array:
        values = batch.token_values
        classes = batch.token_classes
        mask = batch.token_mask
        if (
            values.ndim != 4
            or values.shape[1] != BOARD_SLOTS
            or values.shape[-1] != UNIVERSAL_PARENT_VALUE_WIDTH
            or classes.shape != values.shape[:-1]
            or mask.shape != classes.shape
        ):
            raise ValueError("R4 universal parent token tensor shape drifted")
        if batch.market_features.shape[-1] != MARKET_FEATURES:
            raise ValueError("R4 parent market feature width drifted")
        if batch.player_features.shape[-1] != PLAYER_FEATURES:
            raise ValueError("R4 parent player feature width drifted")
        if batch.global_features.shape[-1] != GLOBAL_FEATURES:
            raise ValueError("R4 parent global feature width drifted")

        class_indices = mx.maximum(classes - 1, 0)
        selected_weights = self.token_weights[class_indices]
        selected_bias = self.token_bias[class_indices]
        normalized_values = values.astype(mx.float32) / 64.0
        tokens = (
            mx.sum(
                normalized_values[..., :, None] * selected_weights,
                axis=-2,
            )
            + selected_bias
        )
        tokens = self.token_norm(self.token_activation(tokens))
        seat_ids = mx.broadcast_to(
            mx.arange(BOARD_SLOTS, dtype=mx.int32)[None, :, None],
            classes.shape,
        )
        tokens = (tokens + self.seat_embedding(seat_ids)) * mask[..., None]

        market = self.market_projection(batch.market_features)
        market_mask = batch.market_mask
        market_summary = mx.sum(market * market_mask[..., None], axis=1) / mx.maximum(
            mx.sum(market_mask, axis=1, keepdims=True),
            1.0,
        )
        players = self.player_projection(batch.player_features) * batch.player_mask[..., None]
        global_context = self.global_projection(batch.global_features)

        batch_size, _, token_capacity, hidden = tokens.shape
        flat_tokens = tokens.reshape(
            batch_size * BOARD_SLOTS,
            token_capacity,
            hidden,
        )
        flat_classes = classes.reshape(batch_size * BOARD_SLOTS, token_capacity)
        flat_mask = mask.reshape(batch_size * BOARD_SLOTS, token_capacity)
        flat_players = players.reshape(batch_size * BOARD_SLOTS, 1, hidden)
        player_mask = batch.player_mask.reshape(batch_size * BOARD_SLOTS, 1)
        summaries, summary_mask = _nine_type_summaries(
            flat_tokens,
            flat_classes,
            flat_mask,
        )
        inputs = mx.concatenate(
            [flat_players, summaries, flat_tokens],
            axis=1,
        )
        input_mask = mx.concatenate(
            [player_mask, summary_mask, flat_mask],
            axis=1,
        )
        latents = mx.broadcast_to(
            self.latents[None, :, :],
            (
                batch_size * BOARD_SLOTS,
                self.config.parent_perceiver_latents,
                hidden,
            ),
        )
        latents = self.perceiver_cross(latents, inputs, input_mask)
        latent_mask = mx.ones(
            (
                batch_size * BOARD_SLOTS,
                self.config.parent_perceiver_latents,
            ),
            dtype=mx.bool_,
        )
        for block in self.perceiver_blocks:
            latents = block(latents, latent_mask)
        board_summaries = self.board_summary_projection(masked_pool(latents, latent_mask)).reshape(
            batch_size, BOARD_SLOTS, hidden
        )
        context = mx.concatenate(
            [
                global_context[:, None, :],
                market_summary[:, None, :],
                board_summaries + players,
            ],
            axis=1,
        )
        context_mask = mx.concatenate(
            [
                mx.ones((batch_size, 2), dtype=mx.bool_),
                batch.player_mask,
            ],
            axis=1,
        )
        for block in self.cross_board_blocks:
            context = block(context, context_mask)
        return self.state_summary_projection(masked_pool(context, context_mask))


class R4BoundedParentRanker(R3ActionEditRanker):
    """The accepted R3 candidate ranker with one universal parent encoder."""

    def __init__(self, config: R4BoundedParentModelConfig | None = None):
        config = config or R4BoundedParentModelConfig()
        config.validate()
        super().__init__(
            R3ActionEditModelConfig(
                arm=R3_CONTROL_ARM,
                hidden_dim=config.hidden_dim,
                attention_heads=config.attention_heads,
                parent_perceiver_latents=config.parent_perceiver_latents,
                candidate_perceiver_latents=config.candidate_perceiver_latents,
                parent_latent_blocks=config.parent_latent_blocks,
                candidate_latent_blocks=config.candidate_latent_blocks,
                cross_board_blocks=config.cross_board_blocks,
                staged_market_blocks=config.staged_market_blocks,
                feed_forward_multiplier=config.feed_forward_multiplier,
            )
        )
        self.parent_encoder = UniversalParentEncoder(config)
        self.config = config


def r4_bounded_parent_loss_components(
    model: R4BoundedParentRanker,
    batch: object,
) -> dict[str, mx.array]:
    return r3_action_edit_loss_components(model, batch)


def r4_bounded_parent_loss(
    model: R4BoundedParentRanker,
    batch: object,
) -> mx.array:
    return r3_action_edit_loss(model, batch)


def _nine_type_summaries(
    values: mx.array,
    token_classes: mx.array,
    token_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    summaries = []
    masks = []
    for token_class in range(1, UNIVERSAL_PARENT_CLASS_COUNT + 1):
        selected = token_mask & (token_classes == token_class)
        weights = selected[..., None]
        summaries.append(
            mx.sum(values * weights, axis=1) / mx.maximum(mx.sum(weights, axis=1), 1.0)
        )
        masks.append(mx.any(selected, axis=1))
    return mx.stack(summaries, axis=1), mx.stack(masks, axis=1)


__all__ = [
    "ARCHITECTURE",
    "R4BoundedParentModelConfig",
    "R4BoundedParentRanker",
    "UniversalParentEncoder",
    "parameter_count",
    "parameter_layout_blake3",
    "parameter_tensor_blake3",
    "r4_bounded_parent_loss",
    "r4_bounded_parent_loss_components",
]
