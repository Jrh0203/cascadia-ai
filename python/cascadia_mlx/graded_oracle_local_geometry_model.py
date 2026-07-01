"""Rotation-canonical local geometry treatment for complete-action ranking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ACTION_ROTATION_SLICE,
    GRADED_ACTION_TILE_Q_INDEX,
    GRADED_ACTION_TILE_R_INDEX,
    GRADED_ACTION_WILDLIFE_Q_INDEX,
    GRADED_ACTION_WILDLIFE_R_INDEX,
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PRIOR_SCHEMA,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    GradedOracleModelConfig,
    GradedOraclePrediction,
    GradedOracleRanker,
)
from cascadia_mlx.model import _masked_pool

LOCAL_GEOMETRY_ARCHITECTURE = "complete-action-graded-local-geometry-v1"
LOCAL_GEOMETRY_MODEL_SCHEMA_VERSION = 1
LOCAL_GEOMETRY_RELATION_SCHEMA = "active-board-local-13-v1"
LOCAL_GEOMETRY_RELATIONS = 13
LOCAL_GEOMETRY_CANONICAL_ENTITY_DIM = ENTITY_DIM - 2
LOCAL_GEOMETRY_RELATION_DIM = LOCAL_GEOMETRY_CANONICAL_ENTITY_DIM + 1
LOCAL_GEOMETRY_CONTEXT_DIM = LOCAL_GEOMETRY_RELATIONS * LOCAL_GEOMETRY_RELATION_DIM
LOCAL_GEOMETRY_CANONICAL_ACTION_DIM = GRADED_ORACLE_ACTION_DIM - 10
LOCAL_GEOMETRY_CORRECTION_RANGE = 12.0
_COORDINATE_TOLERANCE = 1e-5
_NORMALIZED_HEX_DIRECTIONS = (
    (1.0 / 24.0, 0.0),
    (1.0 / 24.0, -1.0 / 24.0),
    (0.0, -1.0 / 24.0),
    (-1.0 / 24.0, 0.0),
    (-1.0 / 24.0, 1.0 / 24.0),
    (0.0, 1.0 / 24.0),
)


@dataclass(frozen=True)
class LocalGeometryModelConfig:
    """Serializable ADR 0088 architecture."""

    schema_version: int = LOCAL_GEOMETRY_MODEL_SCHEMA_VERSION
    architecture: str = LOCAL_GEOMETRY_ARCHITECTURE
    prior_feature_schema: str = GRADED_ORACLE_PRIOR_SCHEMA
    relation_schema: str = LOCAL_GEOMETRY_RELATION_SCHEMA
    hidden_dim: int = 192
    attention_heads: int = 6
    board_blocks: int = 3
    market_blocks: int = 2
    feed_forward_multiplier: int = 4
    local_hidden_dim: int = 192

    def validate(self) -> None:
        if (
            self.schema_version != LOCAL_GEOMETRY_MODEL_SCHEMA_VERSION
            or self.architecture != LOCAL_GEOMETRY_ARCHITECTURE
            or self.prior_feature_schema != GRADED_ORACLE_PRIOR_SCHEMA
            or self.relation_schema != LOCAL_GEOMETRY_RELATION_SCHEMA
        ):
            raise ValueError("unsupported local-geometry model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.board_blocks < 0 or self.market_blocks < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0 or self.local_hidden_dim <= 0:
            raise ValueError("feed-forward and local hidden dimensions must be positive")

    def base_config(self) -> GradedOracleModelConfig:
        self.validate()
        return GradedOracleModelConfig(
            hidden_dim=self.hidden_dim,
            attention_heads=self.attention_heads,
            board_blocks=self.board_blocks,
            market_blocks=self.market_blocks,
            feed_forward_multiplier=self.feed_forward_multiplier,
        )

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> LocalGeometryModelConfig:
        config = cls(**values)
        config.validate()
        return config


class LocalGeometryRanker(nn.Module):
    """ADR 0081 base ranker plus an exact local-board relation correction."""

    def __init__(self, config: LocalGeometryModelConfig | None = None):
        super().__init__()
        config = config or LocalGeometryModelConfig()
        config.validate()
        self.config = config

        # Instantiate the paired base first so ADR 0081 seeds reproduce its tensors.
        self.base = GradedOracleRanker(config.base_config())
        local_hidden = config.local_hidden_dim
        local_input = (
            LOCAL_GEOMETRY_CONTEXT_DIM
            + LOCAL_GEOMETRY_CANONICAL_ACTION_DIM
            + GRADED_ORACLE_PRIOR_DIM
            + GLOBAL_DIM
        )
        self.local_projection = nn.Sequential(
            nn.Linear(local_input, local_hidden * 2),
            nn.GELU(),
            nn.LayerNorm(local_hidden * 2),
            nn.Linear(local_hidden * 2, local_hidden),
            nn.GELU(),
        )
        self.local_output = nn.Sequential(
            nn.Linear(local_hidden * 4, local_hidden * 3),
            nn.GELU(),
            nn.LayerNorm(local_hidden * 3),
            nn.Linear(local_hidden * 3, local_hidden),
            nn.GELU(),
        )
        self.local_residual_head = nn.Linear(local_hidden, 1)
        self.local_residual_head.weight = mx.zeros_like(self.local_residual_head.weight)
        self.local_residual_head.bias = mx.zeros_like(self.local_residual_head.bias)

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        screen_value: mx.array,
        candidate_mask: mx.array,
    ) -> GradedOraclePrediction:
        base = self.base(
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            public_supply,
            action_features,
            prior_features,
            staged_market_entities,
            staged_market_mask,
            staged_public_supply,
            screen_value,
            candidate_mask,
        )
        groups, candidates = screen_value.shape
        local_context = candidate_local_geometry(
            board_entities,
            board_mask,
            action_features,
            candidate_mask,
        )
        canonical_action = canonical_local_action_features(action_features)
        repeated_global = mx.broadcast_to(
            global_features[:, None, :],
            (groups, candidates, GLOBAL_DIM),
        )
        local = self.local_projection(
            mx.concatenate(
                [
                    local_context,
                    canonical_action,
                    prior_features,
                    repeated_global,
                ],
                axis=-1,
            )
        )
        local = local * candidate_mask[..., None]
        pooled = _masked_pool(local, candidate_mask)
        hidden = self.config.local_hidden_dim
        mean = mx.broadcast_to(pooled[:, None, :hidden], local.shape)
        maximum = mx.broadcast_to(pooled[:, None, hidden:], local.shape)
        output = self.local_output(
            mx.concatenate([local, mean, maximum, local - mean], axis=-1)
        )
        correction = (
            LOCAL_GEOMETRY_CORRECTION_RANGE
            * mx.tanh(self.local_residual_head(output).reshape(groups, candidates))
            * candidate_mask
        )
        residual = mx.clip(
            base.residuals + correction,
            -GRADED_ORACLE_RESIDUAL_RANGE,
            GRADED_ORACLE_RESIDUAL_RANGE,
        )
        residual = residual * candidate_mask
        return GradedOraclePrediction(
            scores=screen_value + residual,
            residuals=residual,
            standard_errors=base.standard_errors,
        )


def canonical_local_action_features(action_features: mx.array) -> mx.array:
    """Drop absolute position and orientation from the local correction path."""
    canonical = mx.concatenate(
        [
            action_features[..., :GRADED_ACTION_TILE_Q_INDEX],
            action_features[..., 42:43],
            action_features[..., 45:],
        ],
        axis=-1,
    )
    if canonical.shape[-1] != LOCAL_GEOMETRY_CANONICAL_ACTION_DIM:
        raise AssertionError("local-geometry canonical action dimension drifted")
    return canonical


def candidate_local_geometry(
    board_entities: mx.array,
    board_mask: mx.array,
    action_features: mx.array,
    candidate_mask: mx.array,
) -> mx.array:
    """Encode exact active-board relations in the candidate tile's local frame."""
    active_board = board_entities[:, 0]
    active_mask = board_mask[:, 0]
    rotations = mx.argmax(action_features[..., GRADED_ACTION_ROTATION_SLICE], axis=-1)

    tile_neighbors = _canonical_direction_masks(
        active_board,
        active_mask,
        action_features[..., GRADED_ACTION_TILE_Q_INDEX],
        action_features[..., GRADED_ACTION_TILE_R_INDEX],
        rotations,
        candidate_mask,
    )
    wildlife_present = action_features[..., 42] > 0.5
    wildlife_target = _coordinate_match(
        active_board,
        active_mask,
        action_features[..., GRADED_ACTION_WILDLIFE_Q_INDEX],
        action_features[..., GRADED_ACTION_WILDLIFE_R_INDEX],
        candidate_mask & wildlife_present,
    )
    wildlife_neighbors = _canonical_direction_masks(
        active_board,
        active_mask,
        action_features[..., GRADED_ACTION_WILDLIFE_Q_INDEX],
        action_features[..., GRADED_ACTION_WILDLIFE_R_INDEX],
        rotations,
        candidate_mask & wildlife_present,
    )
    relation_masks = mx.concatenate(
        [
            tile_neighbors,
            wildlife_target[:, :, None, :],
            wildlife_neighbors,
        ],
        axis=2,
    )
    relation_weights = relation_masks.astype(active_board.dtype)
    selected = mx.einsum("gcrn,gne->gcre", relation_weights, active_board)
    present = mx.clip(mx.sum(relation_weights, axis=-1), 0.0, 1.0)

    selected_rotation = mx.argmax(selected[..., 13:19], axis=-1)
    relative_rotation = (
        selected_rotation + 6 - rotations[:, :, None]
    ) % 6
    relative_rotation_features = mx.eye(6)[relative_rotation] * present[..., None]
    canonical_entities = mx.concatenate(
        [
            selected[..., 2:13],
            selected[..., 19:],
            relative_rotation_features,
        ],
        axis=-1,
    )
    if canonical_entities.shape[-1] != LOCAL_GEOMETRY_CANONICAL_ENTITY_DIM:
        raise AssertionError("local-geometry canonical entity dimension drifted")
    context = mx.concatenate(
        [canonical_entities, present[..., None]],
        axis=-1,
    ).reshape(
        action_features.shape[0],
        action_features.shape[1],
        LOCAL_GEOMETRY_CONTEXT_DIM,
    )
    return context * candidate_mask[..., None]


def _canonical_direction_masks(
    active_board: mx.array,
    active_mask: mx.array,
    origin_q: mx.array,
    origin_r: mx.array,
    rotations: mx.array,
    candidate_mask: mx.array,
) -> mx.array:
    delta_q = active_board[:, None, :, 0] - origin_q[..., None]
    delta_r = active_board[:, None, :, 1] - origin_r[..., None]
    global_masks = mx.stack(
        [
            (mx.abs(delta_q - direction_q) < _COORDINATE_TOLERANCE)
            & (mx.abs(delta_r - direction_r) < _COORDINATE_TOLERANCE)
            for direction_q, direction_r in _NORMALIZED_HEX_DIRECTIONS
        ],
        axis=2,
    )
    local_to_global = (
        mx.arange(6)[None, None, :] + rotations[:, :, None]
    ) % 6
    permutation = mx.eye(6)[local_to_global]
    local_masks = mx.einsum(
        "gcld,gcdn->gcln",
        permutation,
        global_masks.astype(active_board.dtype),
    )
    return (
        (local_masks > 0.5)
        & active_mask[:, None, None, :]
        & candidate_mask[:, :, None, None]
    )


def _coordinate_match(
    active_board: mx.array,
    active_mask: mx.array,
    origin_q: mx.array,
    origin_r: mx.array,
    candidate_mask: mx.array,
) -> mx.array:
    return (
        (mx.abs(active_board[:, None, :, 0] - origin_q[..., None]) < _COORDINATE_TOLERANCE)
        & (mx.abs(active_board[:, None, :, 1] - origin_r[..., None]) < _COORDINATE_TOLERANCE)
        & active_mask[:, None, :]
        & candidate_mask[:, :, None]
    )


def load_promoted_local_geometry_model(
    model_dir: str | Path,
) -> LocalGeometryRanker:
    """Load a promoted ADR 0088 model with its serialized architecture."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model.json").read_text())
    if manifest.get("status") != "promoted" or manifest.get("kind") != (
        "graded-oracle-local-geometry-ranking"
    ):
        raise ValueError("unsupported promoted local-geometry model")
    model = LocalGeometryRanker(
        LocalGeometryModelConfig.from_dict(manifest["model_config"])
    )
    model.load_weights(str(model_dir / manifest["model"]["file"]))
    model.eval()
    return model
