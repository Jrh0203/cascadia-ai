"""MLX listwise action ranker distilled from fair search."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.model import SetAttentionBlock, _masked_pool


@dataclass(frozen=True)
class RankingModelConfig:
    """Serializable search-distillation architecture."""

    schema_version: int = 1
    architecture: str = "entity-set-ranker-v1"
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    feed_forward_multiplier: int = 3

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != "entity-set-ranker-v1":
            raise ValueError("unsupported ranking model configuration")
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
    def from_dict(cls, values: dict[str, object]) -> RankingModelConfig:
        config = cls(**values)
        config.validate()
        return config


class EntitySetRanker(nn.Module):
    """Score grouped legal afterstates with a compact shared encoder."""

    def __init__(self, config: RankingModelConfig | None = None):
        super().__init__()
        config = config or RankingModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.board_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.market_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.seat_embedding = nn.Embedding(4, hidden)
        self.board_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.board_blocks)
        ]
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
            nn.Linear(hidden * 2, 1),
        )

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
    ) -> mx.array:
        groups, candidates = global_features.shape[:2]
        flat_count = groups * candidates
        hidden = self.config.hidden_dim

        boards = self.board_projection(board_entities.reshape(flat_count, 4, 23, -1))
        seats = self.seat_embedding(mx.arange(4))[None, :, None, :]
        boards = boards + seats
        boards = boards.reshape(flat_count * 4, 23, hidden)
        flat_board_mask = board_mask.reshape(flat_count * 4, 23)
        boards = boards * flat_board_mask[..., None]
        for block in self.board_blocks:
            boards = block(boards, flat_board_mask)
        board_summary = _masked_pool(boards, flat_board_mask).reshape(flat_count, -1)

        market = self.market_projection(market_entities.reshape(flat_count, 4, -1))
        flat_market_mask = market_mask.reshape(flat_count, 4)
        for block in self.market_blocks:
            market = block(market, flat_market_mask)
        market_summary = _masked_pool(market, flat_market_mask)

        global_summary = self.global_projection(global_features.reshape(flat_count, -1))
        combined = mx.concatenate([board_summary, market_summary, global_summary], axis=-1)
        return self.trunk(combined).reshape(groups, candidates)


def ranking_loss(
    model: EntitySetRanker,
    batch: object,
    *,
    teacher_temperature: float = 1.0,
) -> mx.array:
    """Uncertainty-weighted listwise cross-entropy over complete decisions."""
    scores = model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
    )
    masked_scores = mx.where(batch.candidate_mask, scores, -1e9)
    teacher_logits = mx.where(
        batch.candidate_mask,
        batch.teacher_mean / teacher_temperature,
        -1e9,
    )
    teacher_probabilities = mx.softmax(teacher_logits, axis=-1)
    log_probabilities = masked_scores - mx.logsumexp(masked_scores, axis=-1, keepdims=True)
    cross_entropy = -mx.sum(teacher_probabilities * log_probabilities, axis=-1)

    candidate_count = mx.maximum(mx.sum(batch.candidate_mask, axis=-1), 1)
    mean_uncertainty = (
        mx.sum(
            mx.where(batch.candidate_mask, batch.teacher_stddev, 0.0),
            axis=-1,
        )
        / candidate_count
    )
    weights = 1.0 / (1.0 + mean_uncertainty)
    return mx.sum(cross_entropy * weights) / mx.sum(weights)
