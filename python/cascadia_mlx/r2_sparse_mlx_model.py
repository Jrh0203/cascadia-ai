"""Matched-capacity MLX architectures over one exact R2 token substrate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_OWNERSHIP_ENCODING,
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    GRAPH_RELATION_COUNT,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TARGET_DIM,
    TOKEN_FEATURES,
)

ARCHITECTURES = (
    "padded-set-transformer",
    "directional-graph-attention",
    "perceiver-fixed-latents",
)

TARGET_SCALES = mx.array(
    [23.0, 23.0, 23.0, 23.0, 23.0, 30.0, 28.0, 28.0, 28.0, 40.0, 20.0],
    dtype=mx.float32,
)


@dataclass(frozen=True)
class R2SparseMlxModelConfig:
    """Frozen architecture settings for ADR 0146."""

    schema_version: int = 1
    architecture: str = "padded-set-transformer"
    hidden_dim: int = 64
    attention_heads: int = 4
    set_blocks: int = 2
    graph_message_blocks: int = 2
    graph_global_blocks: int = 1
    graph_relation_dim: int = 16
    perceiver_latents: int = 16
    perceiver_latent_blocks: int = 1
    cross_board_blocks: int = 1
    feed_forward_multiplier: int = 2
    board_ownership_encoding: str = BOARD_OWNERSHIP_ENCODING

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported R2 sparse MLX model schema")
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown R2 sparse MLX architecture")
        if self.hidden_dim != 64:
            raise ValueError("R2 freezes hidden_dim at 64")
        if self.attention_heads != 4:
            raise ValueError("R2 freezes attention_heads at 4")
        if self.set_blocks != 2:
            raise ValueError("R2 freezes two Set Transformer blocks")
        if self.graph_message_blocks != 2 or self.graph_global_blocks != 1:
            raise ValueError("R2 freezes two graph blocks and one global attention block")
        if self.graph_relation_dim != 16:
            raise ValueError("R2 freezes graph_relation_dim at 16")
        if self.perceiver_latents != 16 or self.perceiver_latent_blocks != 1:
            raise ValueError("R2 freezes 16 latents and one latent attention block")
        if self.cross_board_blocks != 1:
            raise ValueError("R2 freezes one explicit player/global context block")
        if self.feed_forward_multiplier != 2:
            raise ValueError("R2 freezes the feed-forward multiplier at two")
        if self.board_ownership_encoding != BOARD_OWNERSHIP_ENCODING:
            raise ValueError("R2 freezes explicit relative-seat board ownership")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> R2SparseMlxModelConfig:
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


class _GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, relation_dim: int):
        super().__init__()
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.neighbor_norm = nn.LayerNorm(hidden_dim)
        self.relation_embedding = nn.Embedding(GRAPH_RELATION_COUNT, relation_dim)
        self.direction_projection = nn.Linear(6, relation_dim)
        self.message_projection = nn.Sequential(
            nn.Linear(hidden_dim + relation_dim, hidden_dim),
            nn.GELU(),
        )
        self.update_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def __call__(
        self,
        values: mx.array,
        token_mask: mx.array,
        neighbors: mx.array,
        neighbor_mask: mx.array,
        relations: mx.array,
        direction_features: mx.array,
    ) -> mx.array:
        batch_size, tokens, hidden_dim = values.shape
        normalized = self.neighbor_norm(values)
        offsets = mx.arange(batch_size, dtype=mx.int32)[:, None, None] * tokens
        flat_indices = (neighbors + offsets).reshape(-1)
        gathered = mx.take(
            normalized.reshape(batch_size * tokens, hidden_dim),
            flat_indices,
            axis=0,
        )
        gathered = gathered.reshape(*neighbors.shape, hidden_dim)
        relation = self.relation_embedding(relations) + self.direction_projection(
            direction_features
        )
        messages = self.message_projection(mx.concatenate([gathered, relation], axis=-1))
        weights = neighbor_mask[..., None]
        degree = mx.maximum(mx.sum(weights, axis=2), 1.0)
        aggregate = mx.sum(messages * weights, axis=2) / degree
        update = self.update_projection(
            mx.concatenate([self.source_norm(values), aggregate], axis=-1)
        )
        return (values + update) * token_mask[..., None]


class _PerceiverCrossBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, multiplier: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiHeadAttention(hidden_dim, heads, bias=True)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * multiplier),
            nn.GELU(),
            nn.Linear(hidden_dim * multiplier, hidden_dim),
        )

    def __call__(
        self,
        latents: mx.array,
        inputs: mx.array,
        input_mask: mx.array,
    ) -> mx.array:
        attention_mask = mx.where(input_mask[:, None, None, :], 0.0, -1e9)
        latents = latents + self.attention(
            self.query_norm(latents),
            self.input_norm(inputs),
            self.input_norm(inputs),
            mask=attention_mask,
        )
        return latents + self.feed_forward(self.feed_forward_norm(latents))


class _CommonStateEncoder(nn.Module):
    """Shared adapters for exact board-local tokens and public context."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        board_token_capacity: int = BOARD_TOKEN_CAPACITY,
    ):
        super().__init__()
        if (
            not isinstance(board_token_capacity, int)
            or isinstance(board_token_capacity, bool)
            or board_token_capacity <= 0
        ):
            raise ValueError("R2 common encoder token capacity is invalid")
        self.board_token_capacity = board_token_capacity
        self.token_projection = nn.Sequential(
            nn.Linear(TOKEN_FEATURES, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.market_projection = nn.Sequential(
            nn.Linear(MARKET_FEATURES, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.player_projection = nn.Sequential(
            nn.Linear(PLAYER_FEATURES, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def __call__(
        self,
        token_features: mx.array,
        token_mask: mx.array,
        market_features: mx.array,
        market_mask: mx.array,
        player_features: mx.array,
        player_mask: mx.array,
        global_features: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if token_features.shape[1:] != (
            BOARD_SLOTS,
            self.board_token_capacity,
            TOKEN_FEATURES,
        ):
            raise ValueError("R2 token tensor shape drifted")
        encoded_tokens = self.token_projection(token_features) * token_mask[..., None]
        encoded_market = self.market_projection(market_features)
        market_summary = _masked_mean(encoded_market, market_mask)
        encoded_players = self.player_projection(player_features) * player_mask[..., None]
        global_summary = self.global_projection(global_features)
        return encoded_tokens, encoded_players, market_summary, global_summary


class R2SparseValueModel(nn.Module):
    """One state encoder and one architecture-specific trunk per decision."""

    def __init__(self, config: R2SparseMlxModelConfig | None = None):
        super().__init__()
        config = config or R2SparseMlxModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.common_encoder = _CommonStateEncoder(hidden)

        if config.architecture == "padded-set-transformer":
            self.set_blocks = [
                _MaskedAttentionBlock(
                    hidden,
                    config.attention_heads,
                    config.feed_forward_multiplier,
                )
                for _ in range(config.set_blocks)
            ]
        elif config.architecture == "directional-graph-attention":
            self.graph_blocks = [
                _GraphMessageBlock(hidden, config.graph_relation_dim)
                for _ in range(config.graph_message_blocks)
            ]
            self.graph_global_blocks = [
                _MaskedAttentionBlock(
                    hidden,
                    config.attention_heads,
                    config.feed_forward_multiplier,
                )
                for _ in range(config.graph_global_blocks)
            ]
        elif config.architecture == "perceiver-fixed-latents":
            self.latents = mx.random.normal((config.perceiver_latents, hidden)) * 0.02
            self.perceiver_cross = _PerceiverCrossBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            self.perceiver_blocks = [
                _MaskedAttentionBlock(
                    hidden,
                    config.attention_heads,
                    config.feed_forward_multiplier,
                )
                for _ in range(config.perceiver_latent_blocks)
            ]
        self.board_summary_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cross_board_blocks = [
            _MaskedAttentionBlock(
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
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, TARGET_DIM),
        )

    def encode_state(self, batch: object) -> mx.array:
        """Encode the public state exactly once for one value decision."""
        tokens, players, market, global_context = self.common_encoder(
            batch.token_features,
            batch.token_mask,
            batch.market_features,
            batch.market_mask,
            batch.player_features,
            batch.player_mask,
            batch.global_features,
        )
        batch_size = tokens.shape[0]
        hidden = self.config.hidden_dim
        flat_tokens = tokens.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            hidden,
        )
        flat_types = batch.token_types.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
        )
        flat_mask = batch.token_mask.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
        )
        flat_players = players.reshape(batch_size * BOARD_SLOTS, 1, hidden)
        player_mask = batch.player_mask.reshape(batch_size * BOARD_SLOTS, 1)
        type_summaries, type_mask = _type_summary_tokens(
            flat_tokens,
            flat_types,
            flat_mask,
        )
        architecture = self.config.architecture
        if architecture == "padded-set-transformer":
            values = mx.concatenate(
                [flat_players, type_summaries, flat_tokens],
                axis=1,
            )
            mask = mx.concatenate([player_mask, type_mask, flat_mask], axis=1)
            for block in self.set_blocks:
                values = block(values, mask)
            board_pooled = _masked_pool(values[:, :5], mask[:, :5])
        elif architecture == "directional-graph-attention":
            values = flat_tokens
            flat_neighbors = batch.graph_neighbors.reshape(
                batch_size * BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY,
                batch.graph_neighbors.shape[-1],
            )
            flat_neighbor_mask = batch.graph_neighbor_mask.reshape(
                batch_size * BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY,
                batch.graph_neighbor_mask.shape[-1],
            )
            flat_relations = batch.graph_relations.reshape(
                batch_size * BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY,
                batch.graph_relations.shape[-1],
            )
            flat_directions = batch.graph_direction_features.reshape(
                batch_size * BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY,
                batch.graph_direction_features.shape[-2],
                batch.graph_direction_features.shape[-1],
            )
            for block in self.graph_blocks:
                values = block(
                    values,
                    flat_mask,
                    flat_neighbors,
                    flat_neighbor_mask,
                    flat_relations,
                    flat_directions,
                )
            type_summaries, type_mask = _type_summary_tokens(
                values,
                flat_types,
                flat_mask,
            )
            values = mx.concatenate([flat_players, type_summaries, values], axis=1)
            mask = mx.concatenate([player_mask, type_mask, flat_mask], axis=1)
            for block in self.graph_global_blocks:
                values = block(values, mask)
            board_pooled = _masked_pool(values[:, :5], mask[:, :5])
        elif architecture == "perceiver-fixed-latents":
            inputs = mx.concatenate(
                [flat_players, type_summaries, flat_tokens],
                axis=1,
            )
            input_mask = mx.concatenate([player_mask, type_mask, flat_mask], axis=1)
            latents = mx.broadcast_to(
                self.latents[None, :, :],
                (
                    batch_size * BOARD_SLOTS,
                    self.config.perceiver_latents,
                    hidden,
                ),
            )
            latents = self.perceiver_cross(latents, inputs, input_mask)
            latent_mask = mx.ones(
                (batch_size * BOARD_SLOTS, self.config.perceiver_latents),
                dtype=mx.bool_,
            )
            for block in self.perceiver_blocks:
                latents = block(latents, latent_mask)
            board_pooled = _masked_pool(latents, latent_mask)
        else:
            raise ValueError("unknown R2 sparse MLX architecture")

        board_summaries = self.board_summary_projection(board_pooled).reshape(
            batch_size,
            BOARD_SLOTS,
            hidden,
        )
        player_boards = board_summaries + players
        context = mx.concatenate(
            [
                global_context[:, None, :],
                market[:, None, :],
                player_boards,
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
        return self.state_summary_projection(_masked_pool(context, context_mask))

    def __call__(self, batch: object) -> mx.array:
        state = self.encode_state(batch)
        return nn.softplus(self.value_head(state) - 2.0)

    def predict_components(self, batch: object) -> mx.array:
        return self(batch) * TARGET_SCALES


def r2_sparse_value_loss(model: R2SparseValueModel, batch: object) -> mx.array:
    predictions = model(batch)
    normalized_targets = batch.targets / TARGET_SCALES
    component_loss = mx.mean(mx.square(predictions - normalized_targets))
    predicted_total = mx.sum(predictions * TARGET_SCALES, axis=-1)
    target_total = mx.sum(batch.targets, axis=-1)
    total_loss = mx.mean(mx.square((predicted_total - target_total) / 100.0))
    return component_loss + 0.5 * total_loss


def parameter_count(model: R2SparseValueModel) -> int:
    return sum(int(value.size) for _, value in tree_flatten(model.trainable_parameters()))


def architecture_parameter_counts() -> dict[str, int]:
    """Return counts from identical initialization-independent model shapes."""
    return {
        architecture: parameter_count(
            R2SparseValueModel(R2SparseMlxModelConfig(architecture=architecture))
        )
        for architecture in ARCHITECTURES
    }


def _type_summary_tokens(
    values: mx.array,
    token_types: mx.array,
    token_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    summaries = []
    masks = []
    for token_type in range(1, 5):
        selected = token_mask & (token_types == token_type)
        summaries.append(_masked_mean(values, selected))
        masks.append(mx.any(selected, axis=1))
    return mx.stack(summaries, axis=1), mx.stack(masks, axis=1)


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask[..., None]
    count = mx.maximum(mx.sum(weights, axis=1), 1.0)
    return mx.sum(values * weights, axis=1) / count


def _masked_pool(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask[..., None]
    mean = _masked_mean(values, mask)
    maximum = mx.max(mx.where(weights, values, -1e9), axis=1)
    maximum = mx.where(mx.any(mask, axis=1, keepdims=True), maximum, 0.0)
    return mx.concatenate([mean, maximum], axis=-1)


# Reusable public building blocks for representation experiments that must
# preserve the accepted R2 Perceiver trunk without carrying its value head.
MaskedAttentionBlock = _MaskedAttentionBlock
PerceiverCrossBlock = _PerceiverCrossBlock
CommonStateEncoder = _CommonStateEncoder
type_summary_tokens = _type_summary_tokens
masked_pool = _masked_pool
