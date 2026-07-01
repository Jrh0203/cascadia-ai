"""Selected-prefix pointer rankers over the accepted exact-R2 state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import blake3
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    DRAFT_FACTOR_DIM,
    STAGED_PUBLIC_DIM,
    STAGES,
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
    TILE_FACTOR_DIM,
)
from cascadia_mlx.r2_sparse_mlx_cache import BOARD_TOKEN_CAPACITY
from cascadia_mlx.relational_substrate_mlx_cache import (
    CONTROL_ARM as RELATIONAL_CONTROL_ARM,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalParentEncoder,
    RelationalSubstrateModelConfig,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "exact-r2-selected-prefix-pointer-ranker-v1"
HIDDEN_DIM = 64
ATTENTION_HEADS = 4
ROTATION_COUNT = 6
DESTINATION_KIND_COUNT = 3

DRAFT_OBSERVABLE_DIM = DRAFT_FACTOR_DIM + STAGED_PUBLIC_DIM
TILE_QUERY_DIM = DRAFT_OBSERVABLE_DIM
WILDLIFE_QUERY_DIM = DRAFT_OBSERVABLE_DIM + TILE_FACTOR_DIM
STAGE_QUERY_DIMS = {
    "draft": 1,
    "tile": TILE_QUERY_DIM,
    "wildlife": WILDLIFE_QUERY_DIM,
}
STAGE_ITEM_DIMS = {
    "draft": DRAFT_OBSERVABLE_DIM,
    "tile": 0,
    "wildlife": 0,
}

DESTINATION_NONE = 0
DESTINATION_NEW_TILE = 1
DESTINATION_EXISTING_TILE = 2


@dataclass(frozen=True)
class RelationalPointerModelConfig:
    """Frozen graph for the first selected-prefix pointer pilot."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    stage: str = "tile"
    hidden_dim: int = HIDDEN_DIM
    attention_heads: int = ATTENTION_HEADS
    parent_perceiver_latents: int = 16
    parent_latent_blocks: int = 1
    cross_board_blocks: int = 1
    feed_forward_multiplier: int = 2

    def validate(self) -> None:
        if (
            self.schema_version != MODEL_SCHEMA_VERSION
            or self.architecture != ARCHITECTURE
        ):
            raise ValueError("unsupported relational pointer model schema")
        if self.stage not in STAGES:
            raise ValueError("relational pointer model names an unknown stage")
        if self.hidden_dim != HIDDEN_DIM or self.attention_heads != ATTENTION_HEADS:
            raise ValueError("relational pointer pilot freezes width 64 and four heads")
        if (
            self.parent_perceiver_latents != 16
            or self.parent_latent_blocks != 1
            or self.cross_board_blocks != 1
            or self.feed_forward_multiplier != 2
        ):
            raise ValueError("relational pointer parent topology drifted")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> RelationalPointerModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class PointerParentEncoding:
    """One parent summary and addressable active-board token memory."""

    summary: mx.array
    active_tokens: mx.array
    active_mask: mx.array
    active_types: mx.array


class RelationalPointerRanker(nn.Module):
    """Score legal factors by pointing into exact selected-prefix state."""

    def __init__(
        self,
        config: RelationalPointerModelConfig | None = None,
    ):
        super().__init__()
        config = config or RelationalPointerModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.parent_encoder = RelationalParentEncoder(
            RelationalSubstrateModelConfig(
                arm=RELATIONAL_CONTROL_ARM,
                hidden_dim=hidden,
                attention_heads=config.attention_heads,
                parent_perceiver_latents=config.parent_perceiver_latents,
                parent_latent_blocks=config.parent_latent_blocks,
                cross_board_blocks=config.cross_board_blocks,
                feed_forward_multiplier=config.feed_forward_multiplier,
            )
        )
        self.rotation_embedding = nn.Embedding(ROTATION_COUNT, hidden)
        self.destination_kind_embedding = nn.Embedding(
            DESTINATION_KIND_COUNT,
            hidden,
        )
        self.none_destination = mx.random.normal((hidden,)) * 0.02
        self.query_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.token_context_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.interaction = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        stage = config.stage
        if stage == "draft":
            self.draft_item_projection = nn.Sequential(
                nn.Linear(DRAFT_OBSERVABLE_DIM, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
        else:
            self.prefix_projection = nn.Sequential(
                nn.Linear(STAGE_QUERY_DIMS[stage], hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
        if stage == "tile":
            self.tile_pointer_projection = nn.Sequential(
                nn.Linear(hidden * 2, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
        elif stage == "wildlife":
            self.selected_tile_projection = nn.Sequential(
                nn.Linear(hidden * 3, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.wildlife_pointer_projection = nn.Sequential(
                nn.Linear(hidden * 3, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )

    def freeze_parent_for_pointer_training(self) -> RelationalPointerRanker:
        """Train only pointer-specific modules after the verified C0 warm start."""
        self.parent_encoder.freeze()
        return self

    def encode_parent(self, parent: object) -> PointerParentEncoding:
        """Encode the exact parent once and retain addressable active-board tokens."""
        summary = mx.stop_gradient(self.parent_encoder(parent))
        tokens = mx.stop_gradient(
            self.parent_encoder.r2_token_projection(parent.r2_token_features)
            * parent.r2_token_mask[..., None]
        )
        return PointerParentEncoding(
            summary=summary,
            active_tokens=tokens[:, 0],
            active_mask=parent.r2_token_mask[:, 0],
            active_types=parent.r2_token_types[:, 0],
        )

    def __call__(
        self,
        batch: object,
        *,
        parent_encoding: PointerParentEncoding | None = None,
    ) -> mx.array:
        encoding = (
            self.encode_parent(batch.parent)
            if parent_encoding is None
            else parent_encoding
        )
        query_parent = batch.query_parent_indices
        state = encoding.summary[query_parent]
        active_tokens = encoding.active_tokens[query_parent]
        active_mask = encoding.active_mask[query_parent]
        item_mask = batch.item_mask
        if (
            state.ndim != 2
            or active_tokens.ndim != 3
            or active_tokens.shape[1] != BOARD_TOKEN_CAPACITY
            or active_mask.shape != active_tokens.shape[:-1]
            or item_mask.ndim != 2
        ):
            raise ValueError("relational pointer batch shape drifted")

        stage = self.config.stage
        if stage == "draft":
            if batch.item_features.shape[-1] != DRAFT_OBSERVABLE_DIM:
                raise ValueError("draft pointer item width drifted")
            prefix = mx.zeros_like(state)
            items = self.draft_item_projection(batch.item_features)
        else:
            if batch.query_features.shape[-1] != STAGE_QUERY_DIMS[stage]:
                raise ValueError(f"{stage} pointer query width drifted")
            prefix = self.prefix_projection(batch.query_features)
            if stage == "tile":
                pointer = _batched_gather(
                    active_tokens,
                    batch.item_pointer_indices,
                )
                pointer_mask = _batched_gather_mask(
                    active_mask,
                    batch.item_pointer_indices,
                )
                if not _same_static_shape(pointer_mask, item_mask):
                    raise ValueError("tile pointer mask shape differs from item mask")
                rotation = self.rotation_embedding(batch.item_rotations)
                items = self.tile_pointer_projection(
                    mx.concatenate([pointer, rotation], axis=-1)
                )
                item_mask = item_mask & pointer_mask
            else:
                selected_pointer = _batched_gather(
                    active_tokens,
                    batch.query_tile_pointer_indices[:, None],
                )[:, 0]
                selected_mask = _batched_gather_mask(
                    active_mask,
                    batch.query_tile_pointer_indices[:, None],
                )[:, 0]
                selected_rotation = self.rotation_embedding(
                    batch.query_tile_rotations
                )
                selected_tile = self.selected_tile_projection(
                    mx.concatenate(
                        [selected_pointer, selected_rotation, prefix],
                        axis=-1,
                    )
                )
                existing = _batched_gather(
                    active_tokens,
                    batch.item_pointer_indices,
                )
                existing_mask = _batched_gather_mask(
                    active_mask,
                    batch.item_pointer_indices,
                )
                kinds = batch.item_kinds
                if kinds.shape != item_mask.shape:
                    raise ValueError("wildlife destination kind shape drifted")
                none = mx.broadcast_to(
                    self.none_destination[None, None, :],
                    existing.shape,
                )
                new_tile = mx.broadcast_to(
                    selected_tile[:, None, :],
                    existing.shape,
                )
                destination = mx.where(
                    (kinds == DESTINATION_NONE)[..., None],
                    none,
                    mx.where(
                        (kinds == DESTINATION_NEW_TILE)[..., None],
                        new_tile,
                        existing,
                    ),
                )
                destination_valid = (
                    (kinds == DESTINATION_NONE)
                    | ((kinds == DESTINATION_NEW_TILE) & selected_mask[:, None])
                    | ((kinds == DESTINATION_EXISTING_TILE) & existing_mask)
                )
                kind = self.destination_kind_embedding(kinds)
                selected = mx.broadcast_to(
                    selected_tile[:, None, :],
                    existing.shape,
                )
                items = self.wildlife_pointer_projection(
                    mx.concatenate([destination, kind, selected], axis=-1)
                )
                item_mask = item_mask & destination_valid

        query = self.query_projection(mx.concatenate([state, prefix], axis=-1))
        query_items = mx.broadcast_to(query[:, None, :], items.shape)
        state_items = mx.broadcast_to(state[:, None, :], items.shape)
        contextual_items = self.token_context_projection(
            mx.concatenate([items, query_items, state_items], axis=-1)
        )
        hidden = self.interaction(
            mx.concatenate(
                [
                    contextual_items,
                    query_items,
                    state_items,
                    contextual_items * query_items,
                    mx.abs(contextual_items - query_items),
                ],
                axis=-1,
            )
        )
        hidden = hidden * item_mask[..., None]
        denominator = mx.maximum(mx.sum(item_mask, axis=1, keepdims=True), 1)
        mean = mx.sum(hidden, axis=1) / denominator
        maximum = mx.max(
            mx.where(item_mask[..., None], hidden, -1e9),
            axis=1,
        )
        maximum = mx.where(mx.any(item_mask, axis=1, keepdims=True), maximum, 0.0)
        scores = self.output(
            mx.concatenate(
                [
                    hidden,
                    mx.broadcast_to(mean[:, None, :], hidden.shape),
                    mx.broadcast_to(maximum[:, None, :], hidden.shape),
                    hidden - mean[:, None, :],
                ],
                axis=-1,
            )
        ).reshape(item_mask.shape)
        return mx.where(item_mask, scores, -1e9)


def relational_pointer_loss(
    model: RelationalPointerRanker,
    batch: object,
    parent_encoding: PointerParentEncoding | None = None,
) -> mx.array:
    """Use the historical calibrated objective with pointer-native logits."""
    scores = model(batch, parent_encoding=parent_encoding)
    expected_rank = batch.expected_rank
    expected_rank_mask = batch.expected_rank_mask
    target = batch.target
    regression_target = -mx.log1p(expected_rank)
    delta = mx.abs(scores - regression_target)
    smooth_l1 = mx.where(delta < 1.0, 0.5 * delta * delta, delta - 0.5)
    regression = _masked_query_mean(smooth_l1, expected_rank_mask)

    target_logits = mx.where(
        expected_rank_mask,
        -(expected_rank - 1.0) / TARGET_SCALE,
        -1e9,
    )
    target_probability = mx.softmax(target_logits, axis=-1)
    student_logits = mx.where(
        expected_rank_mask,
        scores / STUDENT_TEMPERATURE,
        -1e9,
    )
    log_probability = student_logits - mx.logsumexp(
        student_logits,
        axis=-1,
        keepdims=True,
    )
    listwise = -mx.sum(
        mx.where(
            expected_rank_mask,
            target_probability * log_probability,
            0.0,
        ),
        axis=-1,
    )
    listwise = mx.mean(listwise)

    negative = batch.item_mask & ~target
    positive_count = mx.sum(target, axis=-1)
    negative_count = mx.sum(negative, axis=-1)
    positive_loss = mx.sum(
        mx.where(target, nn.softplus(-scores), 0.0),
        axis=-1,
    ) / mx.maximum(positive_count, 1)
    negative_loss = mx.sum(
        mx.where(negative, nn.softplus(scores), 0.0),
        axis=-1,
    ) / mx.maximum(negative_count, 1)
    boundary_valid = (positive_count > 0) & (negative_count > 0)
    boundary = mx.sum(
        mx.where(boundary_valid, positive_loss + negative_loss, 0.0)
    ) / mx.maximum(mx.sum(boundary_valid), 1)
    return regression + listwise + boundary


def parameter_count(model: RelationalPointerRanker) -> int:
    return sum(
        int(value.size)
        for _name, value in tree_flatten(model.trainable_parameters())
    )


def trainable_parameter_names(model: RelationalPointerRanker) -> tuple[str, ...]:
    return tuple(name for name, _value in tree_flatten(model.trainable_parameters()))


def parameter_layout_blake3(
    model: RelationalPointerRanker,
    *,
    trainable_only: bool = False,
) -> str:
    """Hash the exact ordered parameter layout."""
    tree = model.trainable_parameters() if trainable_only else model.parameters()
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(tree)
    ]
    return blake3.blake3(
        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parameter_tensor_blake3(
    model: RelationalPointerRanker,
    *,
    parent_only: bool = False,
    trainable_only: bool = False,
) -> str:
    """Hash parameter names, layouts, and bytes in stable tree order."""
    if parent_only and trainable_only:
        raise ValueError("parameter hash cannot be parent-only and trainable-only")
    tree = model.trainable_parameters() if trainable_only else model.parameters()
    values = tree_flatten(tree)
    if parent_only:
        values = [
            (name, value)
            for name, value in values
            if name.startswith("parent_encoder.")
        ]
    digest = blake3.blake3()
    for name, value in values:
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode()
        )
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()


def _batched_gather(memory: mx.array, indices: mx.array) -> mx.array:
    if (
        memory.ndim != 3
        or indices.ndim != 2
        or memory.shape[0] != indices.shape[0]
    ):
        raise ValueError("pointer gather shape drifted")
    batch, width, hidden = memory.shape
    if width != BOARD_TOKEN_CAPACITY:
        raise ValueError("pointer gather memory capacity drifted")
    offsets = mx.arange(batch, dtype=mx.int32)[:, None] * width
    flat = (indices + offsets).reshape(-1)
    return mx.take(memory.reshape(batch * width, hidden), flat, axis=0).reshape(
        *indices.shape,
        hidden,
    )


def _batched_gather_mask(mask: mx.array, indices: mx.array) -> mx.array:
    if (
        mask.ndim != 2
        or indices.ndim != 2
        or mask.shape[0] != indices.shape[0]
        or mask.shape[1] != BOARD_TOKEN_CAPACITY
    ):
        raise ValueError("pointer mask gather shape drifted")
    offsets = mx.arange(mask.shape[0], dtype=mx.int32)[:, None] * mask.shape[1]
    flat = (indices + offsets).reshape(-1)
    return mx.take(mask.reshape(-1), flat, axis=0).reshape(indices.shape)


def _same_static_shape(left: mx.array, right: mx.array) -> bool:
    return tuple(left.shape) == tuple(right.shape)


def _masked_query_mean(values: mx.array, mask: mx.array) -> mx.array:
    per_query = mx.sum(mx.where(mask, values, 0.0), axis=-1) / mx.maximum(
        mx.sum(mask, axis=-1),
        1,
    )
    valid = mx.any(mask, axis=-1)
    return mx.sum(mx.where(valid, per_query, 0.0)) / mx.maximum(
        mx.sum(valid),
        1,
    )
