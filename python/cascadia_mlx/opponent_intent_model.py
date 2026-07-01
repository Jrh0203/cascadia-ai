"""Matched MLX ablations for policy-held-out opponent intent and survival."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import blake3
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.model import SetAttentionBlock
from cascadia_mlx.opponent_intent_dataset import (
    HISTORY_FEATURE_DIM,
    HISTORY_LENGTH,
    MARKET_SLOTS,
    OPPONENT_COUNT,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "compact-opponent-intent-survival-v1"
ARMS = (
    "a0-public-state",
    "a1-recent-history",
    "a2-next-draft-auxiliary",
    "a3-joint-intent-survival",
)
ARM_GATES = {
    ARMS[0]: (0.0, 0.0, 0.0),
    ARMS[1]: (1.0, 0.0, 0.0),
    ARMS[2]: (1.0, 1.0, 0.0),
    ARMS[3]: (1.0, 1.0, 1.0),
}

DISPOSITION_LOSS_WEIGHT = 1.0
PAIR_SURVIVAL_LOSS_WEIGHT = 0.25
FINAL_SLOT_LOSS_WEIGHT = 0.10
NEXT_DRAFT_AUXILIARY_WEIGHT = 0.25


@dataclass(frozen=True)
class OpponentIntentModelConfig:
    """Serializable graph with arm identity expressed only through gates."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    arm: str = ARMS[0]
    hidden_dim: int = 64
    attention_heads: int = 4
    board_blocks: int = 1
    market_blocks: int = 1
    history_blocks: int = 2
    feed_forward_multiplier: int = 2

    def validate(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION or self.architecture != ARCHITECTURE:
            raise ValueError("unsupported O1 model schema")
        if self.arm not in ARMS:
            raise ValueError("unknown O1 model arm")
        if self.hidden_dim != 64 or self.attention_heads != 4:
            raise ValueError("O1 model freezes width 64 with four heads")
        if (
            self.board_blocks != 1
            or self.market_blocks != 1
            or self.history_blocks != 2
            or self.feed_forward_multiplier != 2
        ):
            raise ValueError("O1 model topology drifted")

    @property
    def gates(self) -> tuple[float, float, float]:
        self.validate()
        return ARM_GATES[self.arm]

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> OpponentIntentModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class OpponentIntentPrediction:
    """All primary and authorized auxiliary logits."""

    disposition_logits: mx.array
    pair_survival_logits: mx.array
    final_slot_logits: mx.array
    tile_slot_logits: mx.array
    wildlife_slot_logits: mx.array
    draft_kind_logits: mx.array
    drafted_wildlife_logits: mx.array
    replace_three_logits: mx.array


class MaskedCrossAttention(nn.Module):
    """Cross-attend to a masked memory without a query residual shortcut."""

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


class OpponentIntentSurvivalModel(nn.Module):
    """Predict ordered opponent drafts and exact future market access."""

    def __init__(
        self,
        config: OpponentIntentModelConfig | None = None,
    ):
        super().__init__()
        config = config or OpponentIntentModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim

        self.board_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.relative_seat_embedding = nn.Embedding(4, hidden)
        self.board_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.board_blocks)
        ]
        self.board_summary_projection = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )

        self.market_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.market_slot_embedding = nn.Embedding(MARKET_SLOTS, hidden)
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
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )

        self.history_projection = nn.Sequential(
            nn.Linear(HISTORY_FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.history_position_embedding = nn.Embedding(HISTORY_LENGTH, hidden)
        self.history_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.history_blocks)
        ]
        self.history_summary_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.opponent_embedding = nn.Embedding(OPPONENT_COUNT, hidden)
        self.history_cross_attention = MaskedCrossAttention(
            hidden,
            config.attention_heads,
        )
        self.intent_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.tile_slot_head = nn.Linear(hidden, 4)
        self.wildlife_slot_head = nn.Linear(hidden, 4)
        self.draft_kind_head = nn.Linear(hidden, 2)
        self.drafted_wildlife_head = nn.Linear(hidden, 5)
        self.replace_three_head = nn.Linear(hidden, 2)

        self.intent_cross_attention = MaskedCrossAttention(
            hidden,
            config.attention_heads,
        )
        self.survival_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.disposition_head = nn.Linear(hidden, 4)
        self.pair_survival_head = nn.Linear(hidden, 2)
        self.final_slot_head = nn.Linear(hidden, 4)

    def __call__(self, batch: object) -> OpponentIntentPrediction:
        history_gate, auxiliary_gate, joint_gate = self.config.gates
        batch_size = batch.board_entities.shape[0]
        hidden = self.config.hidden_dim

        boards = self.board_projection(batch.board_entities)
        boards = boards + self.relative_seat_embedding(mx.arange(4))[None, :, None, :]
        boards = boards.reshape(batch_size * 4, 23, hidden)
        flat_board_mask = batch.board_mask.reshape(batch_size * 4, 23)
        boards = boards * flat_board_mask[..., None]
        for block in self.board_blocks:
            boards = block(boards, flat_board_mask)
        board_summary = _masked_pool(boards, flat_board_mask).reshape(
            batch_size,
            hidden * 8,
        )
        board_summary = self.board_summary_projection(board_summary)

        market = self.market_projection(batch.market_entities)
        market = market + self.market_slot_embedding(mx.arange(MARKET_SLOTS))[None, :, :]
        market = market * batch.market_mask[..., None]
        for block in self.market_blocks:
            market = block(market, batch.market_mask)
        market_summary = _masked_pool(market, batch.market_mask)
        global_summary = self.global_projection(batch.global_features)
        state = self.state_projection(
            mx.concatenate(
                [board_summary, market_summary, global_summary],
                axis=-1,
            )
        )

        history = self.history_projection(batch.history_features)
        history = history + self.history_position_embedding(mx.arange(HISTORY_LENGTH))[None, :, :]
        history = history * batch.history_mask[..., None]
        for block in self.history_blocks:
            history = block(history, batch.history_mask)
        history_summary = self.history_summary_projection(_masked_pool(history, batch.history_mask))
        gated_history_summary = history_summary * history_gate

        opponent_query = (
            state[:, None, :]
            + gated_history_summary[:, None, :]
            + self.opponent_embedding(mx.arange(OPPONENT_COUNT))[None, :, :]
        )
        history_readout = self.history_cross_attention(
            opponent_query,
            history,
            batch.history_mask,
        )
        intent = self.intent_projection(
            mx.concatenate(
                [
                    opponent_query,
                    history_readout * auxiliary_gate,
                ],
                axis=-1,
            )
        )

        survival_query = market + state[:, None, :] + gated_history_summary[:, None, :]
        intent_mask = mx.ones(
            (batch_size, OPPONENT_COUNT),
            dtype=mx.bool_,
        )
        intent_readout = self.intent_cross_attention(
            survival_query,
            intent,
            intent_mask,
        )
        survival = self.survival_projection(
            mx.concatenate(
                [survival_query, intent_readout * joint_gate],
                axis=-1,
            )
        )
        return OpponentIntentPrediction(
            disposition_logits=self.disposition_head(survival),
            pair_survival_logits=self.pair_survival_head(survival),
            final_slot_logits=self.final_slot_head(survival),
            tile_slot_logits=self.tile_slot_head(intent),
            wildlife_slot_logits=self.wildlife_slot_head(intent),
            draft_kind_logits=self.draft_kind_head(intent),
            drafted_wildlife_logits=self.drafted_wildlife_head(intent),
            replace_three_logits=self.replace_three_head(intent),
        )


def opponent_intent_loss(
    model: OpponentIntentSurvivalModel,
    batch: object,
) -> mx.array:
    """Primary survival loss plus arm-gated next-draft auxiliary losses."""
    prediction = model(batch)
    disposition = _cross_entropy(
        prediction.disposition_logits,
        batch.disposition_targets,
    )
    survivor_mask = batch.disposition_targets == 3
    pair_survival = _masked_cross_entropy(
        prediction.pair_survival_logits,
        batch.pair_survival_targets,
        survivor_mask,
    )
    final_slot = _masked_cross_entropy(
        prediction.final_slot_logits,
        batch.final_slot_targets,
        survivor_mask,
    )
    auxiliary = mx.mean(
        mx.stack(
            [
                _cross_entropy(
                    prediction.tile_slot_logits,
                    batch.tile_slot_targets,
                ),
                _cross_entropy(
                    prediction.wildlife_slot_logits,
                    batch.wildlife_slot_targets,
                ),
                _cross_entropy(
                    prediction.draft_kind_logits,
                    batch.draft_kind_targets,
                ),
                _cross_entropy(
                    prediction.drafted_wildlife_logits,
                    batch.drafted_wildlife_targets,
                ),
                _cross_entropy(
                    prediction.replace_three_logits,
                    batch.replace_three_targets,
                ),
            ]
        )
    )
    auxiliary_gate = model.config.gates[1]
    return (
        DISPOSITION_LOSS_WEIGHT * disposition
        + PAIR_SURVIVAL_LOSS_WEIGHT * pair_survival
        + FINAL_SLOT_LOSS_WEIGHT * final_slot
        + NEXT_DRAFT_AUXILIARY_WEIGHT * auxiliary_gate * auxiliary
    )


def parameter_count(model: OpponentIntentSurvivalModel) -> int:
    return sum(int(value.size) for _name, value in tree_flatten(model.trainable_parameters()))


def parameter_layout_blake3(model: OpponentIntentSurvivalModel) -> str:
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(model.trainable_parameters())
    ]
    return blake3.blake3(
        json.dumps(
            layout,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def parameter_tensor_blake3(model: OpponentIntentSurvivalModel) -> str:
    digest = blake3.blake3()
    for name, value in tree_flatten(model.trainable_parameters()):
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()


def _cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    log_probabilities = logits - mx.logsumexp(
        logits,
        axis=-1,
        keepdims=True,
    )
    selected = mx.take_along_axis(
        log_probabilities,
        targets[..., None],
        axis=-1,
    )[..., 0]
    return -mx.mean(selected)


def _masked_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> mx.array:
    safe_targets = mx.where(mask, targets, 0)
    log_probabilities = logits - mx.logsumexp(
        logits,
        axis=-1,
        keepdims=True,
    )
    selected = mx.take_along_axis(
        log_probabilities,
        safe_targets[..., None],
        axis=-1,
    )[..., 0]
    return -mx.sum(mx.where(mask, selected, 0.0)) / mx.maximum(
        mx.sum(mask),
        1,
    )


def _masked_pool(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask[..., None]
    count = mx.maximum(mx.sum(weights, axis=1), 1.0)
    mean = mx.sum(values * weights, axis=1) / count
    maximum = mx.max(mx.where(weights, values, -1e9), axis=1)
    has_values = mx.any(mask, axis=1, keepdims=True)
    maximum = mx.where(has_values, maximum, 0.0)
    return mx.concatenate([mean, maximum], axis=-1)


__all__ = [
    "ARMS",
    "ARM_GATES",
    "OpponentIntentModelConfig",
    "OpponentIntentPrediction",
    "OpponentIntentSurvivalModel",
    "opponent_intent_loss",
    "parameter_count",
    "parameter_layout_blake3",
    "parameter_tensor_blake3",
]
