"""Signed score-to-go models for generic sets and edge-aware hex graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM, TARGET_DIM
from cascadia_mlx.model import (
    TARGET_SCALES,
    SetAttentionBlock,
    _masked_pool,
    encode_entity_state,
    initialize_entity_encoder,
)

ENTITY_SET_SCORE_TO_GO_V1 = "entity-set-score-to-go-v1"
EDGE_AWARE_HEX_SCORE_TO_GO_V2 = "edge-aware-hex-score-to-go-v2"
HEX_DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
PAIRWISE_TEMPERATURE = 2.0
PAIRWISE_LOSS_WEIGHT = 0.25


@dataclass(frozen=True)
class ScoreToGoModelConfig:
    schema_version: int = 1
    architecture: str = ENTITY_SET_SCORE_TO_GO_V1
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    graph_blocks: int = 0
    market_blocks: int = 1
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture not in {
            ENTITY_SET_SCORE_TO_GO_V1,
            EDGE_AWARE_HEX_SCORE_TO_GO_V2,
        }:
            raise ValueError("unsupported score-to-go model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.board_blocks < 0 or self.graph_blocks < 0 or self.market_blocks < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive")
        if self.architecture == ENTITY_SET_SCORE_TO_GO_V1:
            if self.graph_blocks != 0:
                raise ValueError("entity-set score-to-go cannot use graph blocks")
        elif self.board_blocks != 0 or self.graph_blocks <= 0:
            raise ValueError("edge-aware score-to-go requires graph blocks and no board blocks")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ScoreToGoModelConfig:
        config = cls(**values)
        config.validate()
        return config


class HexGraphBlock(nn.Module):
    """Residual edge-conditioned message passing over one Cascadia board."""

    def __init__(self, hidden_dim: int, multiplier: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + 11, hidden_dim * multiplier),
            nn.GELU(),
            nn.Linear(hidden_dim * multiplier, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * multiplier),
            nn.GELU(),
            nn.Linear(hidden_dim * multiplier, hidden_dim),
        )

    def __call__(
        self,
        values: mx.array,
        mask: mx.array,
        adjacency: tuple[mx.array, ...],
        edge_matches: tuple[mx.array, ...],
    ) -> mx.array:
        normalized = self.norm(values)
        aggregate = mx.zeros_like(values)
        degree = mx.zeros((*values.shape[:-1], 1), dtype=values.dtype)
        direction_basis = mx.eye(6, dtype=values.dtype)
        for direction in range(6):
            neighbors = mx.matmul(adjacency[direction], normalized)
            present = mx.sum(adjacency[direction], axis=-1, keepdims=True)
            direction_features = mx.broadcast_to(
                direction_basis[direction],
                (*values.shape[:-1], 6),
            )
            message_input = mx.concatenate(
                [neighbors, edge_matches[direction], direction_features],
                axis=-1,
            )
            aggregate += self.message(message_input) * present
            degree += present
        aggregate = aggregate / mx.maximum(degree, 1.0)
        updated = values + self.node_update(mx.concatenate([normalized, aggregate], axis=-1))
        return updated * mask[..., None]


class ScoreToGoValueModel(nn.Module):
    """Predict signed decomposed future score increments."""

    def __init__(self, config: ScoreToGoModelConfig | None = None):
        super().__init__()
        config = config or ScoreToGoModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        if config.architecture == ENTITY_SET_SCORE_TO_GO_V1:
            initialize_entity_encoder(self, config)
        else:
            self.board_projection = nn.Sequential(
                nn.Linear(ENTITY_DIM, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.seat_embedding = nn.Embedding(4, hidden)
            self.graph_blocks = [
                HexGraphBlock(hidden, config.feed_forward_multiplier)
                for _ in range(config.graph_blocks)
            ]
            self.market_projection = nn.Sequential(
                nn.Linear(ENTITY_DIM, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
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
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        if self.config.architecture == ENTITY_SET_SCORE_TO_GO_V1:
            encoded = encode_entity_state(
                self,
                self.config,
                board_entities,
                board_mask,
                market_entities,
                market_mask,
                global_features,
            )
        else:
            encoded = self._encode_hex_graph(
                board_entities,
                board_mask,
                market_entities,
                market_mask,
                global_features,
            )
        return self.trunk(encoded)

    def _encode_hex_graph(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        batch_size = board_entities.shape[0]
        hidden = self.config.hidden_dim
        values = self.board_projection(board_entities)
        values = values + self.seat_embedding(mx.arange(4))[None, :, None, :]
        values = values.reshape(batch_size * 4, 23, hidden)
        flat_mask = board_mask.reshape(batch_size * 4, 23)
        values = values * flat_mask[..., None]
        adjacency, edge_matches = hex_graph_relations(board_entities, board_mask)
        for block in self.graph_blocks:
            values = block(values, flat_mask, adjacency, edge_matches)
        board_summary = _masked_pool(values, flat_mask).reshape(batch_size, -1)

        market = self.market_projection(market_entities)
        for block in self.market_blocks:
            market = block(market, market_mask)
        market_summary = _masked_pool(market, market_mask)
        global_summary = self.global_projection(global_features)
        return mx.concatenate([board_summary, market_summary, global_summary], axis=-1)

    def predict_components(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
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


def score_to_go_loss(model: ScoreToGoValueModel, batch: object) -> mx.array:
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
    loss = component_loss + 0.5 * total_loss
    if model.config.architecture == EDGE_AWARE_HEX_SCORE_TO_GO_V2:
        predicted_final = mx.sum(
            batch.current_targets + predictions * TARGET_SCALES,
            axis=-1,
        )
        target_final = mx.sum(batch.final_targets, axis=-1)
        loss += PAIRWISE_LOSS_WEIGHT * within_round_pairwise_loss(
            predicted_final,
            target_final,
            batch.game_index,
            batch.turn,
        )
    return loss


def hex_graph_relations(
    board_entities: mx.array,
    board_mask: mx.array,
) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
    """Derive directed adjacency and matching oriented terrain attributes."""
    batch_size = board_entities.shape[0]
    q = mx.round(board_entities[..., 0] * 24.0).reshape(batch_size * 4, 23)
    r = mx.round(board_entities[..., 1] * 24.0).reshape(batch_size * 4, 23)
    mask = board_mask.reshape(batch_size * 4, 23)
    source_q = q[:, None, :]
    source_r = r[:, None, :]
    target_q = q[:, :, None]
    target_r = r[:, :, None]
    pair_mask = mask[:, :, None] & mask[:, None, :]

    terrain_a = board_entities[..., 2:7]
    terrain_b = board_entities[..., 7:12]
    has_b = 1.0 - board_entities[..., 12]
    rotation = mx.argmax(board_entities[..., 13:19], axis=-1)
    edge_terrains = []
    for direction in range(6):
        uses_a = (((direction + 6 - rotation) % 6) < 3).astype(board_entities.dtype)
        uses_a = 1.0 - has_b + has_b * uses_a
        terrain = terrain_a * uses_a[..., None] + terrain_b * (1.0 - uses_a[..., None])
        edge_terrains.append(terrain.reshape(batch_size * 4, 23, 5))

    adjacency = []
    edge_matches = []
    for direction, (dq, dr) in enumerate(HEX_DIRECTIONS):
        connected = ((source_q - target_q == dq) & (source_r - target_r == dr) & pair_mask).astype(
            board_entities.dtype
        )
        neighbor_edge = mx.matmul(connected, edge_terrains[(direction + 3) % 6])
        adjacency.append(connected)
        edge_matches.append(edge_terrains[direction] * neighbor_edge)
    return tuple(adjacency), tuple(edge_matches)


def within_round_pairwise_loss(
    predicted_final: mx.array,
    target_final: mx.array,
    game_index: mx.array,
    turn: mx.array,
) -> mx.array:
    """Soft pairwise final-score loss within one game and personal-turn round."""
    count = predicted_final.shape[0]
    indices = mx.arange(count)
    personal_round = turn // 4
    pair_mask = (
        (game_index[:, None] == game_index[None, :])
        & (personal_round[:, None] == personal_round[None, :])
        & (indices[:, None] < indices[None, :])
    )
    target_logit = (target_final[:, None] - target_final[None, :]) / PAIRWISE_TEMPERATURE
    target_probability = mx.sigmoid(target_logit)
    student_logit = (predicted_final[:, None] - predicted_final[None, :]) / PAIRWISE_TEMPERATURE
    pair_loss = mx.logaddexp(0.0, student_logit) - target_probability * student_logit
    weights = pair_mask.astype(predicted_final.dtype)
    return mx.sum(pair_loss * weights) / mx.maximum(mx.sum(weights), 1.0)
