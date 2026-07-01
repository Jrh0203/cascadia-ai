"""Low-capacity candidate-set residual over exact parent hidden states."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.imitation_model import (
    _masked_standardize,
    distributional_imitation_loss_from_scores,
)
from cascadia_mlx.imitation_parent_hidden_dataset import IMITATION_PARENT_HIDDEN_DIM
from cascadia_mlx.model import _masked_pool

PARENT_HIDDEN_ARCHITECTURE_V5 = "exact-parent-hidden-set-residual-v5"


@dataclass(frozen=True)
class ParentHiddenModelConfig:
    schema_version: int = 1
    architecture: str = PARENT_HIDDEN_ARCHITECTURE_V5
    candidate_dim: int = 128
    residual_dim: int = 256

    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.architecture != PARENT_HIDDEN_ARCHITECTURE_V5
            or self.candidate_dim != 128
            or self.residual_dim != 256
        ):
            raise ValueError("ADR 0070 freezes the parent-hidden residual architecture")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ParentHiddenModelConfig:
        config = cls(**values)
        config.validate()
        return config


class ParentHiddenSetResidual(nn.Module):
    """Permutation-equivariant correction anchored to exact parent order."""

    def __init__(self, config: ParentHiddenModelConfig | None = None):
        super().__init__()
        config = config or ParentHiddenModelConfig()
        config.validate()
        self.config = config
        self.hidden_norm = nn.LayerNorm(IMITATION_PARENT_HIDDEN_DIM)
        self.candidate_projection = nn.Sequential(
            nn.Linear(IMITATION_PARENT_HIDDEN_DIM + 5, config.candidate_dim),
            nn.GELU(),
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, config.candidate_dim),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(config.candidate_dim * 4, config.residual_dim),
            nn.GELU(),
            nn.LayerNorm(config.residual_dim),
            nn.Linear(config.residual_dim, config.candidate_dim),
            nn.GELU(),
            nn.Linear(config.candidate_dim, 1),
        )
        output = self.residual.layers[-1]
        output.weight = mx.zeros_like(output.weight)
        output.bias = mx.zeros_like(output.bias)

    def __call__(
        self,
        parent_hidden: mx.array,
        parent_immediate: mx.array,
        parent_remaining: mx.array,
        parent_total: mx.array,
        parent_rank: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        if parent_hidden.shape[-1] != IMITATION_PARENT_HIDDEN_DIM:
            raise ValueError("parent-hidden input width does not match the exact parent")
        standardized_total = _masked_standardize(parent_total, candidate_mask)
        reciprocal_rank = 1.0 / mx.maximum(parent_rank, 1.0)
        scalar_features = mx.stack(
            [
                parent_immediate / 100.0,
                parent_remaining / 100.0,
                parent_total / 100.0,
                standardized_total,
                reciprocal_rank,
            ],
            axis=-1,
        )
        candidate = self.candidate_projection(
            mx.concatenate([self.hidden_norm(parent_hidden), scalar_features], axis=-1)
        )
        candidate = candidate * candidate_mask[..., None]
        pooled = _masked_pool(candidate, candidate_mask)
        width = self.config.candidate_dim
        mean = mx.broadcast_to(pooled[:, None, :width], candidate.shape)
        maximum = mx.broadcast_to(pooled[:, None, width:], candidate.shape)
        residual = self.residual(
            mx.concatenate([candidate, mean, maximum, candidate - mean], axis=-1)
        ).squeeze(-1)
        return standardized_total + residual


def score_parent_hidden_actions(model: ParentHiddenSetResidual, batch: object) -> mx.array:
    return model(
        batch.parent_hidden,
        batch.parent_immediate,
        batch.parent_remaining,
        batch.parent_total,
        batch.parent_rank,
        batch.candidate_mask,
    )


def parent_hidden_distributional_loss(
    model: ParentHiddenSetResidual,
    batch: object,
) -> mx.array:
    scores = score_parent_hidden_actions(model, batch)
    return distributional_imitation_loss_from_scores(scores, batch)
