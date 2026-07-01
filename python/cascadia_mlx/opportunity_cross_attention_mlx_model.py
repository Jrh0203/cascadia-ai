"""Matched query-conditioning ablations over exact Cascadia opportunity memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.r2_sparse_mlx_cache import BOARD_SLOTS, TOKEN_FEATURES
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditEncoding,
    R3ActionEditPrediction,
    r3_action_edit_loss,
    r3_action_edit_loss_components,
)
from cascadia_mlx.relational_substrate_mlx_cache import (
    CONTROL_ARM as RELATIONAL_CONTROL_ARM,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateModelConfig,
    RelationalSubstrateRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    EXACT_TOKEN_COUNT,
    SUPPLY_TOKEN_DIM,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "exact-r2-opportunity-query-factorial-v1"
ARMS = (
    "c0-parent-conditioned",
    "t1-supply-query",
    "t2-frontier-query",
    "t3-combined-query",
)
_QUERY_GATES = {
    ARMS[0]: (0.0, 0.0),
    ARMS[1]: (1.0, 0.0),
    ARMS[2]: (0.0, 1.0),
    ARMS[3]: (1.0, 1.0),
}


@dataclass(frozen=True)
class OpportunityCrossAttentionModelConfig:
    """One graph with arm identity expressed only through query routing."""

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
    supply_token_count: int = EXACT_TOKEN_COUNT
    supply_token_dim: int = SUPPLY_TOKEN_DIM
    board_slots: int = BOARD_SLOTS
    r2_token_features: int = TOKEN_FEATURES

    def validate(self) -> None:
        if (
            self.schema_version != MODEL_SCHEMA_VERSION
            or self.architecture != ARCHITECTURE
        ):
            raise ValueError("unsupported opportunity cross-attention schema")
        if self.arm not in ARMS:
            raise ValueError("opportunity model names an unknown arm")
        if self.hidden_dim != 64 or self.attention_heads != 4:
            raise ValueError("opportunity model freezes width 64 and four heads")
        if (
            self.parent_perceiver_latents != 16
            or self.candidate_perceiver_latents != 8
            or self.parent_latent_blocks != 1
            or self.candidate_latent_blocks != 1
            or self.cross_board_blocks != 1
            or self.staged_market_blocks != 1
            or self.feed_forward_multiplier != 2
        ):
            raise ValueError("opportunity model base topology drifted")
        if (
            self.supply_token_count != EXACT_TOKEN_COUNT
            or self.supply_token_dim != SUPPLY_TOKEN_DIM
            or self.board_slots != BOARD_SLOTS
            or self.r2_token_features != TOKEN_FEATURES
        ):
            raise ValueError("opportunity model factual tensor widths drifted")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> OpportunityCrossAttentionModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class OpportunityCrossAttentionContext:
    """Candidate and parent queries plus their exact-memory readouts."""

    candidate_query: mx.array
    parent_query: mx.array
    supply_context: mx.array
    frontier_context: mx.array


class MaskedQueryCrossAttention(nn.Module):
    """Read masked memory without adding the query as a residual shortcut."""

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiHeadAttention(
            hidden_dim,
            heads,
            bias=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def __call__(
        self,
        query: mx.array,
        memory: mx.array,
        memory_mask: mx.array,
    ) -> mx.array:
        if (
            query.ndim != 3
            or memory.ndim != 3
            or memory_mask.shape != memory.shape[:-1]
            or query.shape[0] != memory.shape[0]
            or query.shape[-1] != memory.shape[-1]
        ):
            raise ValueError("query-memory attention tensor shape drifted")
        attention_mask = mx.where(
            memory_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        normalized_memory = self.memory_norm(memory)
        return self.output_norm(
            self.attention(
                self.query_norm(query),
                normalized_memory,
                normalized_memory,
                mask=attention_mask,
            )
        )


class OpportunityCrossAttentionRanker(RelationalSubstrateRanker):
    """Exact-R2 ranker with matched supply and board-memory query ablations."""

    def __init__(
        self,
        config: OpportunityCrossAttentionModelConfig | None = None,
    ):
        config = config or OpportunityCrossAttentionModelConfig()
        config.validate()
        super().__init__(
            RelationalSubstrateModelConfig(
                arm=RELATIONAL_CONTROL_ARM,
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
        self.config = config
        self.memory_seat_embedding = nn.Embedding(BOARD_SLOTS, hidden)
        self.supply_token_projection = nn.Sequential(
            nn.Linear(SUPPLY_TOKEN_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.supply_position_embedding = nn.Embedding(
            EXACT_TOKEN_COUNT,
            hidden,
        )
        self.candidate_query_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.parent_query_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.supply_cross_attention = MaskedQueryCrossAttention(
            hidden,
            config.attention_heads,
        )
        self.frontier_cross_attention = MaskedQueryCrossAttention(
            hidden,
            config.attention_heads,
        )
        self.context_projection = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.context_delta = nn.Linear(hidden, hidden)
        self.context_delta.weight = mx.zeros_like(self.context_delta.weight)
        self.context_delta.bias = mx.zeros_like(self.context_delta.bias)

    @property
    def query_gates(self) -> tuple[float, float]:
        return _QUERY_GATES[self.config.arm]

    def freeze_base_for_adapter_training(
        self,
    ) -> OpportunityCrossAttentionRanker:
        """Freeze the warm-started ranker and train only opportunity adapters."""
        self.freeze()
        for module in (
            self.memory_seat_embedding,
            self.supply_token_projection,
            self.supply_position_embedding,
            self.candidate_query_projection,
            self.parent_query_projection,
            self.supply_cross_attention,
            self.frontier_cross_attention,
            self.context_projection,
            self.context_delta,
        ):
            module.unfreeze()
        return self

    def encode_base_candidates(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditEncoding:
        """Expose the warm-start-compatible exact-R2 candidate encoding."""
        return super().encode_candidates(
            batch,
            candidate_slice=candidate_slice,
            parent_state=parent_state,
        )

    def opportunity_context(
        self,
        batch: object,
        encoding: R3ActionEditEncoding,
        parent_state: mx.array,
    ) -> OpportunityCrossAttentionContext:
        """Read exact supply and R2 memories with the arm's frozen queries."""
        groups, candidates, hidden = encoding.hidden.shape
        if (
            parent_state.shape != (groups, hidden)
            or encoding.candidate_mask.shape != (groups, candidates)
        ):
            raise ValueError("opportunity context does not align with candidates")

        candidate_query = self.candidate_query_projection(encoding.hidden)
        parent_query = mx.broadcast_to(
            self.parent_query_projection(parent_state)[:, None, :],
            candidate_query.shape,
        )
        supply_gate, frontier_gate = self.query_gates
        supply_query = parent_query + supply_gate * (
            candidate_query - parent_query
        )
        frontier_query = parent_query + frontier_gate * (
            candidate_query - parent_query
        )

        supply_memory, supply_mask = self._supply_memory(batch)
        frontier_memory, frontier_mask = self._frontier_memory(batch)
        return OpportunityCrossAttentionContext(
            candidate_query=candidate_query,
            parent_query=parent_query,
            supply_context=self.supply_cross_attention(
                supply_query,
                supply_memory,
                supply_mask,
            ),
            frontier_context=self.frontier_cross_attention(
                frontier_query,
                frontier_memory,
                frontier_mask,
            ),
        )

    def encode_candidates(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditEncoding:
        parent = self.encode_parent(batch) if parent_state is None else parent_state
        encoding = self.encode_base_candidates(
            batch,
            candidate_slice=candidate_slice,
            parent_state=parent,
        )
        context = self.opportunity_context(batch, encoding, parent)
        features = self.context_projection(
            mx.concatenate(
                [
                    encoding.hidden,
                    context.supply_context,
                    context.frontier_context,
                    encoding.hidden * context.supply_context,
                    encoding.hidden * context.frontier_context,
                ],
                axis=-1,
            )
        )
        contextualized = (
            encoding.hidden + self.context_delta(features)
        ) * encoding.candidate_mask[..., None]
        return R3ActionEditEncoding(
            hidden=contextualized,
            candidate_mask=encoding.candidate_mask,
        )

    def predict(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditPrediction:
        return super().predict(
            batch,
            candidate_slice=candidate_slice,
            parent_state=parent_state,
        )

    def _supply_memory(
        self,
        batch: object,
    ) -> tuple[mx.array, mx.array]:
        supply_tokens = batch.supply_tokens
        supply_mask = batch.supply_mask
        if (
            supply_tokens.ndim != 3
            or supply_tokens.shape[1:] != (
                EXACT_TOKEN_COUNT,
                SUPPLY_TOKEN_DIM,
            )
            or supply_mask.shape != supply_tokens.shape[:-1]
        ):
            raise ValueError("exact supply token tensor shape drifted")
        positions = mx.arange(EXACT_TOKEN_COUNT, dtype=mx.int32)
        memory = self.supply_token_projection(supply_tokens)
        memory = (
            memory + self.supply_position_embedding(positions)[None, :, :]
        )
        return memory * supply_mask[..., None], supply_mask

    def _frontier_memory(
        self,
        batch: object,
    ) -> tuple[mx.array, mx.array]:
        features = batch.parent.r2_token_features
        mask = batch.parent.r2_token_mask
        if (
            features.ndim != 4
            or features.shape[1] != BOARD_SLOTS
            or features.shape[-1] != TOKEN_FEATURES
            or mask.shape != features.shape[:-1]
        ):
            raise ValueError("exact R2 opportunity memory shape drifted")
        groups, _, capacity, _ = features.shape
        memory = self.parent_encoder.r2_token_projection(features)
        seats = mx.arange(BOARD_SLOTS, dtype=mx.int32)
        memory = (
            memory
            + self.memory_seat_embedding(seats)[None, :, None, :]
        ) * mask[..., None]
        return (
            memory.reshape(groups, BOARD_SLOTS * capacity, -1),
            mask.reshape(groups, BOARD_SLOTS * capacity),
        )


def opportunity_cross_attention_loss_components(
    model: OpportunityCrossAttentionRanker,
    batch: object,
) -> dict[str, mx.array]:
    return r3_action_edit_loss_components(model, batch)


def opportunity_cross_attention_loss(
    model: OpportunityCrossAttentionRanker,
    batch: object,
) -> mx.array:
    return r3_action_edit_loss(model, batch)


__all__ = [
    "ARCHITECTURE",
    "ARMS",
    "MaskedQueryCrossAttention",
    "OpportunityCrossAttentionContext",
    "OpportunityCrossAttentionModelConfig",
    "OpportunityCrossAttentionRanker",
    "opportunity_cross_attention_loss",
    "opportunity_cross_attention_loss_components",
    "parameter_count",
    "parameter_layout_blake3",
    "parameter_tensor_blake3",
]
