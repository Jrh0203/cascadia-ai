"""Matched S4 candidate-set models for the failed-radius-one rescue test."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.r2_sparse_mlx_model import (
    MaskedAttentionBlock,
    PerceiverCrossBlock,
)
from cascadia_mlx.r3_action_edit_mlx_cache import ARMS as R3_ARMS
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditEncoding,
    R3ActionEditPrediction,
    R3ActionEditRanker,
    r3_action_edit_loss,
)
from cascadia_mlx.s4_candidate_context import (
    ANCHOR_LIMIT,
    RELATION_NEIGHBOR_LIMIT,
)
from cascadia_mlx.s4_candidate_context_cache import CandidateContextBatch
from cascadia_mlx.s4_candidate_relation_census import RELATIONS

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "s4-matched-candidate-context-v1"
INDUCING_LATENTS = 16
S4_ARMS = (
    "c0-independent",
    "t1-inducing-16",
    "t2-exact-relations",
    "t3-combined",
)
_ARM_GATES = {
    S4_ARMS[0]: (0.0, 0.0),
    S4_ARMS[1]: (1.0, 0.0),
    S4_ARMS[2]: (0.0, 1.0),
    S4_ARMS[3]: (1.0, 1.0),
}


@dataclass(frozen=True)
class S4CandidateSetModelConfig:
    """One graph with frozen treatment gates for the four-arm S4 comparison."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    arm: str = S4_ARMS[0]
    r3_arm: str = R3_ARMS[3]
    hidden_dim: int = 64
    attention_heads: int = 4
    parent_perceiver_latents: int = 16
    candidate_perceiver_latents: int = 8
    parent_latent_blocks: int = 1
    candidate_latent_blocks: int = 1
    cross_board_blocks: int = 1
    staged_market_blocks: int = 1
    feed_forward_multiplier: int = 2
    anchor_limit: int = ANCHOR_LIMIT
    inducing_latents: int = INDUCING_LATENTS
    relation_neighbor_limit: int = RELATION_NEIGHBOR_LIMIT
    relation_count: int = len(RELATIONS)

    def validate(self) -> None:
        if (
            self.schema_version != MODEL_SCHEMA_VERSION
            or self.architecture != ARCHITECTURE
        ):
            raise ValueError("unsupported S4 candidate-set model schema")
        if self.arm not in S4_ARMS:
            raise ValueError("S4 model names an unknown comparison arm")
        if self.r3_arm != R3_ARMS[3]:
            raise ValueError("S4 freezes the failed R3 radius-one rescue substrate")
        if self.hidden_dim != 64 or self.attention_heads != 4:
            raise ValueError("S4 freezes hidden width 64 and four heads")
        if (
            self.parent_perceiver_latents != 16
            or self.candidate_perceiver_latents != 8
            or self.parent_latent_blocks != 1
            or self.candidate_latent_blocks != 1
            or self.cross_board_blocks != 1
            or self.staged_market_blocks != 1
            or self.feed_forward_multiplier != 2
        ):
            raise ValueError("S4 R3 substrate configuration drifted")
        if (
            self.anchor_limit != ANCHOR_LIMIT
            or self.inducing_latents != INDUCING_LATENTS
            or self.relation_neighbor_limit != RELATION_NEIGHBOR_LIMIT
            or self.relation_count != len(RELATIONS)
        ):
            raise ValueError("S4 candidate-context dimensions drifted")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> S4CandidateSetModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class S4CandidateContextMlxBatch:
    """Exact context routing tensors converted once from the immutable cache."""

    rows: mx.array
    candidate_counts: mx.array
    anchor_candidate_indices: mx.array
    anchor_mask: mx.array
    relation_neighbor_anchor_slots: mx.array
    relation_neighbor_mask: mx.array
    relation_anchor_sibling_counts: mx.array

    def query_slice(self, selected: slice) -> S4CandidateContextMlxBatch:
        return S4CandidateContextMlxBatch(
            rows=self.rows,
            candidate_counts=self.candidate_counts,
            anchor_candidate_indices=self.anchor_candidate_indices,
            anchor_mask=self.anchor_mask,
            relation_neighbor_anchor_slots=(
                self.relation_neighbor_anchor_slots[:, selected]
            ),
            relation_neighbor_mask=self.relation_neighbor_mask[:, selected],
            relation_anchor_sibling_counts=(
                self.relation_anchor_sibling_counts[:, selected]
            ),
        )


@dataclass(frozen=True)
class S4CandidateSetBatch:
    """One R3 batch bound to its exact S4 context sidecar."""

    r3: object
    context: S4CandidateContextMlxBatch

    def __getattr__(self, name: str) -> object:
        return getattr(self.r3, name)


@dataclass(frozen=True)
class S4PreparedCandidateContext:
    """Reusable parent, anchors, and inducing summary for chunked scoring."""

    parent_state: mx.array
    anchor_hidden: mx.array
    anchor_mask: mx.array
    inducing_latents: mx.array


def mlx_candidate_context(
    context: CandidateContextBatch,
) -> S4CandidateContextMlxBatch:
    """Convert one verified NumPy context batch into MLX routing tensors."""
    return S4CandidateContextMlxBatch(
        rows=mx.array(context.rows.astype("int32", copy=False)),
        candidate_counts=mx.array(
            context.candidate_counts.astype("int32", copy=False)
        ),
        anchor_candidate_indices=mx.array(
            context.anchor_candidate_indices.astype("int32", copy=False)
        ),
        anchor_mask=mx.array(context.anchor_mask),
        relation_neighbor_anchor_slots=mx.array(
            context.relation_neighbor_anchor_slots.astype("int32", copy=False)
        ),
        relation_neighbor_mask=mx.array(context.relation_neighbor_mask),
        relation_anchor_sibling_counts=mx.array(
            context.relation_anchor_sibling_counts.astype("float32", copy=False)
        ),
    )


class S4CandidateSetRanker(R3ActionEditRanker):
    """R3 ranker plus gated global-set and exact-relation residual context."""

    def __init__(self, config: S4CandidateSetModelConfig | None = None):
        config = config or S4CandidateSetModelConfig()
        config.validate()
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config
        hidden = config.hidden_dim

        self.context_latents = (
            mx.random.normal((config.inducing_latents, hidden)) * 0.02
        )
        self.anchor_to_latents = PerceiverCrossBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.latent_block = MaskedAttentionBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.query_to_latents = PerceiverCrossBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.inducing_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.inducing_delta = nn.Linear(hidden, hidden)

        self.relation_embedding = nn.Embedding(config.relation_count, hidden)
        self.relation_pair_projection = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.relation_neighbor_score = nn.Linear(hidden, 1)
        self.relation_count_projection = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.relation_segment_gate = nn.Linear(hidden * 3, 1)
        self.relation_output_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.relation_delta = nn.Linear(hidden, hidden)

        self.inducing_delta.weight = mx.zeros_like(self.inducing_delta.weight)
        self.inducing_delta.bias = mx.zeros_like(self.inducing_delta.bias)
        self.relation_delta.weight = mx.zeros_like(self.relation_delta.weight)
        self.relation_delta.bias = mx.zeros_like(self.relation_delta.bias)

    @property
    def treatment_gates(self) -> tuple[float, float]:
        return _ARM_GATES[self.config.arm]

    def prepare_context(
        self,
        batch: object,
        context: S4CandidateContextMlxBatch,
        *,
        parent_state: mx.array | None = None,
    ) -> S4PreparedCandidateContext:
        """Encode the parent and 256 stable anchors exactly once per decision."""
        self._validate_context(batch, context)
        parent = self.encode_parent(batch) if parent_state is None else parent_state
        anchor_batch = _gather_anchor_batch(
            batch,
            context.anchor_candidate_indices,
            context.anchor_mask,
        )
        anchor_encoding = super().encode_candidates(
            anchor_batch,
            parent_state=parent,
        )
        inducing = self.encode_inducing_latents(
            anchor_encoding.hidden,
            context.anchor_mask,
        )
        return S4PreparedCandidateContext(
            parent_state=parent,
            anchor_hidden=anchor_encoding.hidden,
            anchor_mask=context.anchor_mask,
            inducing_latents=inducing,
        )

    def encode_inducing_latents(
        self,
        anchor_hidden: mx.array,
        anchor_mask: mx.array,
    ) -> mx.array:
        """Summarize an unordered anchor set with 16 fixed learned queries."""
        groups, anchors, hidden = anchor_hidden.shape
        if (
            anchors != self.config.anchor_limit
            or hidden != self.config.hidden_dim
            or anchor_mask.shape != (groups, anchors)
        ):
            raise ValueError("S4 inducing-anchor tensor contract differs")
        latents = mx.broadcast_to(
            self.context_latents[None, :, :],
            (groups, self.config.inducing_latents, hidden),
        )
        latents = self.anchor_to_latents(latents, anchor_hidden, anchor_mask)
        latent_mask = mx.ones(
            (groups, self.config.inducing_latents),
            dtype=mx.bool_,
        )
        return self.latent_block(latents, latent_mask)

    def contextualize_encoding(
        self,
        encoding: R3ActionEditEncoding,
        context: S4CandidateContextMlxBatch,
        prepared: S4PreparedCandidateContext,
    ) -> R3ActionEditEncoding:
        """Apply both context paths, with arm identity expressed only by gates."""
        groups, candidates, hidden = encoding.hidden.shape
        if (
            context.relation_neighbor_anchor_slots.shape
            != (
                groups,
                candidates,
                self.config.relation_count,
                self.config.relation_neighbor_limit,
            )
            or context.relation_neighbor_mask.shape
            != context.relation_neighbor_anchor_slots.shape
            or context.relation_anchor_sibling_counts.shape
            != (groups, candidates, self.config.relation_count)
            or prepared.anchor_hidden.shape
            != (groups, self.config.anchor_limit, hidden)
            or prepared.anchor_mask.shape
            != (groups, self.config.anchor_limit)
            or prepared.inducing_latents.shape
            != (groups, self.config.inducing_latents, hidden)
        ):
            raise ValueError("S4 prepared context does not align with its queries")

        latent_mask = mx.ones(
            (groups, self.config.inducing_latents),
            dtype=mx.bool_,
        )
        attended = self.query_to_latents(
            encoding.hidden,
            prepared.inducing_latents,
            latent_mask,
        )
        inducing_features = self.inducing_projection(
            mx.concatenate(
                [
                    encoding.hidden,
                    attended,
                    encoding.hidden * attended,
                ],
                axis=-1,
            )
        )
        inducing_delta = self.inducing_delta(inducing_features)

        relation_context = self._relation_context(
            encoding.hidden,
            encoding.candidate_mask,
            prepared.anchor_hidden,
            context,
        )
        relation_delta = self.relation_delta(relation_context)
        inducing_gate, relation_gate = self.treatment_gates
        contextualized = (
            encoding.hidden
            + inducing_gate * inducing_delta
            + relation_gate * relation_delta
        ) * encoding.candidate_mask[..., None]
        return R3ActionEditEncoding(
            hidden=contextualized,
            candidate_mask=encoding.candidate_mask,
        )

    def predict(
        self,
        batch: object,
        context: S4CandidateContextMlxBatch | None = None,
        *,
        candidate_slice: slice | None = None,
        prepared_context: S4PreparedCandidateContext | None = None,
    ) -> R3ActionEditPrediction:
        if isinstance(batch, S4CandidateSetBatch):
            if context is not None:
                raise ValueError("S4 context was provided twice")
            context = batch.context
            batch = batch.r3
        if context is None:
            raise ValueError("S4 prediction requires exact candidate context")
        selected = candidate_slice or slice(None)
        prepared = (
            self.prepare_context(batch, context)
            if prepared_context is None
            else prepared_context
        )
        encoding = super().encode_candidates(
            batch,
            candidate_slice=candidate_slice,
            parent_state=prepared.parent_state,
        )
        contextualized = self.contextualize_encoding(
            encoding,
            context.query_slice(selected),
            prepared,
        )
        return super().predict_from_encoding(
            batch,
            contextualized,
            candidate_slice=candidate_slice,
        )

    def __call__(self, batch: object) -> R3ActionEditPrediction:
        return self.predict(batch)

    def _relation_context(
        self,
        query_hidden: mx.array,
        query_mask: mx.array,
        anchor_hidden: mx.array,
        context: S4CandidateContextMlxBatch,
    ) -> mx.array:
        groups, candidates, hidden = query_hidden.shape
        relation_count = self.config.relation_count
        neighbor_limit = self.config.relation_neighbor_limit
        neighbor_hidden = _gather_anchor_neighbors(
            anchor_hidden,
            context.relation_neighbor_anchor_slots,
        )
        query = mx.broadcast_to(
            query_hidden[:, :, None, None, :],
            (
                groups,
                candidates,
                relation_count,
                neighbor_limit,
                hidden,
            ),
        )
        relation_ids = mx.arange(relation_count, dtype=mx.int32)
        relation_embeddings = self.relation_embedding(relation_ids)
        relation = mx.broadcast_to(
            relation_embeddings[None, None, :, None, :],
            query.shape,
        )
        pair = self.relation_pair_projection(
            mx.concatenate(
                [
                    query,
                    neighbor_hidden,
                    query - neighbor_hidden,
                    query * neighbor_hidden + relation,
                ],
                axis=-1,
            )
        )
        mask = context.relation_neighbor_mask & query_mask[:, :, None, None]
        logits = self.relation_neighbor_score(pair).squeeze(-1)
        weights = mx.softmax(mx.where(mask, logits, -1e9), axis=-1)
        weights = weights * mask.astype(weights.dtype)
        weights = weights / mx.maximum(
            mx.sum(weights, axis=-1, keepdims=True),
            1e-8,
        )
        relation_summary = mx.sum(pair * weights[..., None], axis=-2)

        sibling_counts = context.relation_anchor_sibling_counts
        normalized_counts = mx.log1p(sibling_counts) / math.log1p(
            self.config.anchor_limit
        )
        count_features = self.relation_count_projection(
            normalized_counts[..., None]
        )
        relation_tokens = (
            relation_summary
            + relation_embeddings[None, None, :, :]
            + count_features
        )
        query_by_relation = mx.broadcast_to(
            query_hidden[:, :, None, :],
            relation_tokens.shape,
        )
        segment_gate = mx.sigmoid(
            self.relation_segment_gate(
                mx.concatenate(
                    [
                        query_by_relation,
                        relation_tokens,
                        query_by_relation * relation_tokens,
                    ],
                    axis=-1,
                )
            ).squeeze(-1)
        )
        relation_mask = (sibling_counts > 0) & query_mask[:, :, None]
        segment_weights = segment_gate * relation_mask.astype(segment_gate.dtype)
        aggregate = mx.sum(
            relation_tokens * segment_weights[..., None],
            axis=-2,
        ) / mx.maximum(
            mx.sum(segment_weights, axis=-1, keepdims=True),
            1.0,
        )
        return self.relation_output_projection(
            mx.concatenate(
                [
                    query_hidden,
                    aggregate,
                    query_hidden * aggregate,
                ],
                axis=-1,
            )
        ) * query_mask[..., None]

    def _validate_context(
        self,
        batch: object,
        context: S4CandidateContextMlxBatch,
    ) -> None:
        groups, candidates = batch.base.candidate_mask.shape
        if (
            context.rows.shape != (groups,)
            or context.candidate_counts.shape != (groups,)
            or context.anchor_candidate_indices.shape
            != (groups, self.config.anchor_limit)
            or context.anchor_mask.shape != (groups, self.config.anchor_limit)
            or context.relation_neighbor_anchor_slots.shape
            != (
                groups,
                candidates,
                self.config.relation_count,
                self.config.relation_neighbor_limit,
            )
            or context.relation_neighbor_mask.shape
            != context.relation_neighbor_anchor_slots.shape
            or context.relation_anchor_sibling_counts.shape
            != (groups, candidates, self.config.relation_count)
        ):
            raise ValueError("S4 context batch tensor contract differs")


def s4_candidate_set_loss(
    model: S4CandidateSetRanker,
    batch: S4CandidateSetBatch,
) -> mx.array:
    """Keep the accepted R3 objective exactly unchanged."""
    return r3_action_edit_loss(model, batch)  # type: ignore[arg-type]


def _gather_anchor_batch(
    batch: object,
    anchor_indices: mx.array,
    anchor_mask: mx.array,
) -> SimpleNamespace:
    base = batch.base
    anchor_base = SimpleNamespace(
        candidate_mask=anchor_mask,
        action_features=_gather_group_candidates(
            base.action_features,
            anchor_indices,
        ),
        prior_features=_gather_group_candidates(
            base.prior_features,
            anchor_indices,
        ),
        staged_market_entities=_gather_group_candidates(
            base.staged_market_entities,
            anchor_indices,
        ),
        staged_market_mask=_gather_group_candidates(
            base.staged_market_mask,
            anchor_indices,
        ),
    )
    return SimpleNamespace(
        base=anchor_base,
        parent=batch.parent,
        candidate_token_features=_gather_group_candidates(
            batch.candidate_token_features,
            anchor_indices,
        ),
        candidate_token_mask=_gather_group_candidates(
            batch.candidate_token_mask,
            anchor_indices,
        ),
        supply_vector=batch.supply_vector,
        staged_supply_vector=_gather_group_candidates(
            batch.staged_supply_vector,
            anchor_indices,
        ),
        selected_archetype=_gather_group_candidates(
            batch.selected_archetype,
            anchor_indices,
        ),
        frontier_features=_gather_group_candidates(
            batch.frontier_features,
            anchor_indices,
        ),
    )


def _gather_group_candidates(
    values: mx.array,
    indices: mx.array,
) -> mx.array:
    groups, candidates = values.shape[:2]
    if indices.shape[0] != groups:
        raise ValueError("S4 gather indices do not align with candidate groups")
    flat_indices = (
        indices + mx.arange(groups, dtype=mx.int32)[:, None] * candidates
    ).reshape(-1)
    gathered = mx.take(
        values.reshape(groups * candidates, *values.shape[2:]),
        flat_indices,
        axis=0,
    )
    return gathered.reshape(groups, indices.shape[1], *values.shape[2:])


def _gather_anchor_neighbors(
    anchor_hidden: mx.array,
    neighbor_slots: mx.array,
) -> mx.array:
    groups, anchors, hidden = anchor_hidden.shape
    if neighbor_slots.shape[0] != groups:
        raise ValueError("S4 neighbor slots do not align with anchor groups")
    offsets = mx.arange(groups, dtype=mx.int32).reshape(
        groups,
        *([1] * (neighbor_slots.ndim - 1)),
    )
    flat_indices = (neighbor_slots + offsets * anchors).reshape(-1)
    gathered = mx.take(
        anchor_hidden.reshape(groups * anchors, hidden),
        flat_indices,
        axis=0,
    )
    return gathered.reshape(*neighbor_slots.shape, hidden)
