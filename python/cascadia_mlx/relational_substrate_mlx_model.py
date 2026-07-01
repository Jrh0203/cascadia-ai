"""Iso-parameter relational substrate ranker for ADR 0161."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    MaskedAttentionBlock,
    PerceiverCrossBlock,
    masked_pool,
)
from cascadia_mlx.r3_action_edit_mlx_cache import CONTROL_ARM as R3_CONTROL_ARM
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditEncoding,
    R3ActionEditModelConfig,
    R3ActionEditRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    r3_action_edit_loss,
    r3_action_edit_loss_components,
)
from cascadia_mlx.relational_substrate_mlx_cache import (
    ARMS,
    PARENT_CLASS_COUNT,
    RELATIONAL_CLASS_COUNT,
    RELATIONAL_VALUE_WIDTH,
    S5_FEATURES,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "relational-substrate-iso-independent-ranker-v1"


@dataclass(frozen=True)
class RelationalSubstrateModelConfig:
    """The only model graph admitted to the ADR 0161 tournament."""

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
    relational_value_width: int = RELATIONAL_VALUE_WIDTH
    relational_classes: int = RELATIONAL_CLASS_COUNT
    derivative_width: int = S5_FEATURES

    def validate(self) -> None:
        if (
            self.schema_version != MODEL_SCHEMA_VERSION
            or self.architecture != ARCHITECTURE
        ):
            raise ValueError("unsupported relational substrate model schema")
        if self.arm not in ARMS:
            raise ValueError("relational substrate model names an unknown arm")
        if self.hidden_dim != 64 or self.attention_heads != 4:
            raise ValueError("ADR 0161 freezes hidden width 64 and four heads")
        if (
            self.parent_perceiver_latents != 16
            or self.candidate_perceiver_latents != 8
            or self.parent_latent_blocks != 1
            or self.candidate_latent_blocks != 1
            or self.cross_board_blocks != 1
            or self.staged_market_blocks != 1
            or self.feed_forward_multiplier != 2
        ):
            raise ValueError("ADR 0161 attention topology drifted")
        if (
            self.relational_value_width != RELATIONAL_VALUE_WIDTH
            or self.relational_classes != RELATIONAL_CLASS_COUNT
            or self.derivative_width != S5_FEATURES
        ):
            raise ValueError("ADR 0161 factual feature widths drifted")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> RelationalSubstrateModelConfig:
        config = cls(**values)
        config.validate()
        return config


class RelationalParentEncoder(nn.Module):
    """Native R2 and relational adapters feeding one shared parent trunk."""

    def __init__(self, config: RelationalSubstrateModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.r2_token_projection = nn.Sequential(
            nn.Linear(TOKEN_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        scale = (2.0 / RELATIONAL_VALUE_WIDTH) ** 0.5
        self.relational_weights = (
            mx.random.normal(
                (
                    RELATIONAL_CLASS_COUNT,
                    RELATIONAL_VALUE_WIDTH,
                    hidden,
                )
            )
            * scale
        )
        self.relational_bias = mx.zeros((RELATIONAL_CLASS_COUNT, hidden))
        self.relational_activation = nn.GELU()
        self.relational_norm = nn.LayerNorm(hidden)
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
        self.latents = (
            mx.random.normal((config.parent_perceiver_latents, hidden)) * 0.02
        )
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
        r2_features = batch.r2_token_features
        r2_types = batch.r2_token_types
        r2_mask = batch.r2_token_mask
        relational_values = batch.relational_values
        relational_classes = batch.relational_classes
        relational_mask = batch.relational_mask
        if (
            r2_features.ndim != 4
            or r2_features.shape[1] != BOARD_SLOTS
            or r2_features.shape[-1] != TOKEN_FEATURES
            or r2_types.shape != r2_features.shape[:-1]
            or r2_mask.shape != r2_types.shape
        ):
            raise ValueError("native R2 parent tensor shape drifted")
        if (
            relational_values.ndim != 4
            or relational_values.shape[1] != BOARD_SLOTS
            or relational_values.shape[-1] != RELATIONAL_VALUE_WIDTH
            or relational_classes.shape != relational_values.shape[:-1]
            or relational_mask.shape != relational_classes.shape
        ):
            raise ValueError("relational parent tensor shape drifted")
        if batch.market_features.shape[-1] != MARKET_FEATURES:
            raise ValueError("relational parent market width drifted")
        if batch.player_features.shape[-1] != PLAYER_FEATURES:
            raise ValueError("relational parent player width drifted")
        if batch.global_features.shape[-1] != GLOBAL_FEATURES:
            raise ValueError("relational parent global width drifted")

        groups = r2_features.shape[0]
        hidden = self.config.hidden_dim
        if r2_features.shape[2]:
            r2_tokens = self.r2_token_projection(r2_features) * r2_mask[..., None]
        else:
            r2_tokens = mx.zeros((groups, BOARD_SLOTS, 0, hidden))

        if relational_values.shape[2]:
            class_indices = mx.maximum(relational_classes - 1, 0)
            selected_weights = self.relational_weights[class_indices]
            selected_bias = self.relational_bias[class_indices]
            normalized = relational_values.astype(mx.float32) / 64.0
            relational_tokens = self.relational_norm(
                self.relational_activation(
                    mx.sum(
                        normalized[..., :, None] * selected_weights,
                        axis=-2,
                    )
                    + selected_bias
                )
            )
            seat_ids = mx.broadcast_to(
                mx.arange(BOARD_SLOTS, dtype=mx.int32)[None, :, None],
                relational_classes.shape,
            )
            relational_tokens = (
                relational_tokens + self.seat_embedding(seat_ids)
            ) * relational_mask[..., None]
        else:
            relational_tokens = mx.zeros((groups, BOARD_SLOTS, 0, hidden))

        market_tokens = self.market_projection(batch.market_features)
        market = mx.sum(
            market_tokens * batch.market_mask[..., None],
            axis=1,
        ) / mx.maximum(
            mx.sum(batch.market_mask[..., None], axis=1),
            1.0,
        )
        players = (
            self.player_projection(batch.player_features)
            * batch.player_mask[..., None]
        )
        global_context = self.global_projection(batch.global_features)

        flat_groups = groups * BOARD_SLOTS
        r2_capacity = r2_tokens.shape[2]
        relational_capacity = relational_tokens.shape[2]
        flat_r2 = r2_tokens.reshape(flat_groups, r2_capacity, hidden)
        flat_r2_types = r2_types.reshape(flat_groups, r2_capacity)
        flat_r2_mask = r2_mask.reshape(flat_groups, r2_capacity)
        flat_relational = relational_tokens.reshape(
            flat_groups,
            relational_capacity,
            hidden,
        )
        flat_relational_classes = relational_classes.reshape(
            flat_groups,
            relational_capacity,
        )
        flat_relational_mask = relational_mask.reshape(
            flat_groups,
            relational_capacity,
        )
        relational_summaries, relational_summary_mask = _class_summaries(
            flat_relational,
            flat_relational_classes,
            flat_relational_mask,
            RELATIONAL_CLASS_COUNT,
        )
        r2_summaries, r2_summary_mask = _class_summaries(
            flat_r2,
            flat_r2_types,
            flat_r2_mask,
            PARENT_CLASS_COUNT - RELATIONAL_CLASS_COUNT,
        )
        summaries = mx.concatenate(
            [relational_summaries, r2_summaries],
            axis=1,
        )
        summary_mask = mx.concatenate(
            [relational_summary_mask, r2_summary_mask],
            axis=1,
        )
        flat_players = players.reshape(flat_groups, 1, hidden)
        player_mask = batch.player_mask.reshape(flat_groups, 1)
        inputs = mx.concatenate(
            [
                flat_players,
                summaries,
                flat_r2,
                flat_relational,
            ],
            axis=1,
        )
        input_mask = mx.concatenate(
            [
                player_mask,
                summary_mask,
                flat_r2_mask,
                flat_relational_mask,
            ],
            axis=1,
        )
        latents = mx.broadcast_to(
            self.latents[None, :, :],
            (
                flat_groups,
                self.config.parent_perceiver_latents,
                hidden,
            ),
        )
        latents = self.perceiver_cross(latents, inputs, input_mask)
        latent_mask = mx.ones(
            (flat_groups, self.config.parent_perceiver_latents),
            dtype=mx.bool_,
        )
        for block in self.perceiver_blocks:
            latents = block(latents, latent_mask)
        boards = self.board_summary_projection(
            masked_pool(latents, latent_mask)
        ).reshape(groups, BOARD_SLOTS, hidden)
        context = mx.concatenate(
            [
                global_context[:, None, :],
                market[:, None, :],
                boards + players,
            ],
            axis=1,
        )
        context_mask = mx.concatenate(
            [
                mx.ones((groups, 2), dtype=mx.bool_),
                batch.player_mask,
            ],
            axis=1,
        )
        for block in self.cross_board_blocks:
            context = block(context, context_mask)
        return self.state_summary_projection(masked_pool(context, context_mask))


class RelationalSubstrateRanker(R3ActionEditRanker):
    """The accepted R3 ranker with matched relational and derivative adapters."""

    def __init__(self, config: RelationalSubstrateModelConfig | None = None):
        config = config or RelationalSubstrateModelConfig()
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
        hidden = config.hidden_dim
        self.parent_encoder = RelationalParentEncoder(config)
        self.derivative_projection = nn.Sequential(
            nn.Linear(S5_FEATURES, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(hidden * 11, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
        )
        self.config = config

    def encode_candidates(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditEncoding:
        parent = self.encode_parent(batch) if parent_state is None else parent_state
        selected = candidate_slice or slice(None)
        base = batch.base
        candidate_mask = base.candidate_mask[:, selected]
        spatial = self.candidate_encoder(
            batch.candidate_token_features[:, selected],
            batch.candidate_token_mask[:, selected],
        )
        action = self.action_projection(base.action_features[:, selected])
        prior = self.prior_projection(base.prior_features[:, selected])

        staged_market = self.staged_market_projection(
            base.staged_market_entities[:, selected]
        )
        groups, candidates, market_width, hidden = staged_market.shape
        flat_market = staged_market.reshape(
            groups * candidates,
            market_width,
            hidden,
        )
        flat_market_mask = base.staged_market_mask[:, selected].reshape(
            groups * candidates,
            market_width,
        )
        for block in self.staged_market_blocks:
            flat_market = block(flat_market, flat_market_mask)
        staged_market_summary = self.staged_market_summary(
            masked_pool(flat_market, flat_market_mask)
        ).reshape(groups, candidates, hidden)

        supply = self.supply_projection(batch.supply_vector)
        supply_candidates = mx.broadcast_to(
            supply[:, None, :],
            (groups, candidates, hidden),
        )
        staged_supply = self.staged_supply_projection(
            batch.staged_supply_vector[:, selected]
        )
        relation = self.relation_projection(
            mx.concatenate(
                [
                    self.archetype_embedding(
                        batch.selected_archetype[:, selected]
                    ),
                    self.frontier_projection(
                        batch.frontier_features[:, selected]
                    ),
                ],
                axis=-1,
            )
        )
        derivative = self.derivative_projection(
            batch.derivative_features[:, selected]
        )
        parent_candidates = mx.broadcast_to(
            parent[:, None, :],
            (groups, candidates, hidden),
        )
        fused = self.candidate_fusion(
            mx.concatenate(
                [
                    parent_candidates,
                    spatial,
                    action,
                    prior,
                    staged_market_summary,
                    supply_candidates,
                    staged_supply,
                    relation,
                    derivative,
                    parent_candidates * spatial,
                    action * spatial,
                ],
                axis=-1,
            )
        )
        output = self.output_trunk(fused) * candidate_mask[..., None]
        return R3ActionEditEncoding(
            hidden=output,
            candidate_mask=candidate_mask,
        )


def relational_substrate_loss_components(
    model: RelationalSubstrateRanker,
    batch: object,
) -> dict[str, mx.array]:
    return r3_action_edit_loss_components(model, batch)


def relational_substrate_loss(
    model: RelationalSubstrateRanker,
    batch: object,
) -> mx.array:
    return r3_action_edit_loss(model, batch)


def _class_summaries(
    values: mx.array,
    token_classes: mx.array,
    token_mask: mx.array,
    class_count: int,
) -> tuple[mx.array, mx.array]:
    if values.shape[1] == 0:
        return (
            mx.zeros((values.shape[0], class_count, values.shape[-1])),
            mx.zeros((values.shape[0], class_count), dtype=mx.bool_),
        )
    summaries = []
    masks = []
    for token_class in range(1, class_count + 1):
        selected = token_mask & (token_classes == token_class)
        weights = selected[..., None]
        summaries.append(
            mx.sum(values * weights, axis=1)
            / mx.maximum(mx.sum(weights, axis=1), 1.0)
        )
        masks.append(mx.any(selected, axis=1))
    return mx.stack(summaries, axis=1), mx.stack(masks, axis=1)


__all__ = [
    "ARCHITECTURE",
    "RelationalParentEncoder",
    "RelationalSubstrateModelConfig",
    "RelationalSubstrateRanker",
    "parameter_count",
    "parameter_layout_blake3",
    "parameter_tensor_blake3",
    "relational_substrate_loss",
    "relational_substrate_loss_components",
]
