"""Intent-conditioned exact-R2 reranker for ADR 0188."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import blake3
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.o1_ranking_intent_cache import ARMS, INTENT_FEATURE_DIM
from cascadia_mlx.r3_action_edit_mlx_cache import CONTROL_ARM
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditEncoding,
    R3ActionEditModelConfig,
    R3ActionEditRanker,
    r3_action_edit_loss,
    r3_action_edit_loss_components,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "o1-intent-conditioned-exact-r2-reranker-v1"
HIDDEN_DIM = 64
INTENT_INTERMEDIATE_DIM = 128


@dataclass(frozen=True)
class O1RankingModelConfig:
    """One matched graph whose arm changes only the routed input tensor."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    arm: str = ARMS[0]
    hidden_dim: int = HIDDEN_DIM
    intent_feature_dim: int = INTENT_FEATURE_DIM
    intent_intermediate_dim: int = INTENT_INTERMEDIATE_DIM
    base_arm: str = CONTROL_ARM

    def validate(self) -> None:
        if (
            self.schema_version != MODEL_SCHEMA_VERSION
            or self.architecture != ARCHITECTURE
        ):
            raise ValueError("unsupported O1 ranking model schema")
        if self.arm not in ARMS:
            raise ValueError("O1 ranking model names an unknown arm")
        if (
            self.hidden_dim != HIDDEN_DIM
            or self.intent_feature_dim != INTENT_FEATURE_DIM
            or self.intent_intermediate_dim != INTENT_INTERMEDIATE_DIM
            or self.base_arm != CONTROL_ARM
        ):
            raise ValueError("ADR 0188 model dimensions or exact-R2 base drifted")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> O1RankingModelConfig:
        config = cls(**values)
        config.validate()
        return config


class O1IntentConditionedRanker(R3ActionEditRanker):
    """Frozen exact-R2 ranker with a zero-initialized intent residual."""

    def __init__(self, config: O1RankingModelConfig | None = None):
        config = config or O1RankingModelConfig()
        config.validate()
        super().__init__(R3ActionEditModelConfig(arm=CONTROL_ARM))
        self.config = config
        hidden = config.hidden_dim
        intermediate = config.intent_intermediate_dim
        self.intent_projection = nn.Sequential(
            nn.Linear(config.intent_feature_dim, intermediate),
            nn.GELU(),
            nn.LayerNorm(intermediate),
            nn.Linear(intermediate, hidden),
            nn.GELU(),
        )
        self.intent_fusion = nn.Sequential(
            nn.Linear(hidden * 3, intermediate),
            nn.GELU(),
            nn.LayerNorm(intermediate),
            nn.Linear(intermediate, hidden),
            nn.GELU(),
        )
        self.intent_delta = nn.Linear(hidden, hidden)
        self.intent_delta.weight = mx.zeros_like(self.intent_delta.weight)
        self.intent_delta.bias = mx.zeros_like(self.intent_delta.bias)

    def freeze_base_for_adapter_training(self) -> O1IntentConditionedRanker:
        """Freeze every warm-started parameter and train only the O1 adapter."""
        self.freeze()
        for module in (
            self.intent_projection,
            self.intent_fusion,
            self.intent_delta,
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
        """Expose exact-R2 candidate hidden states for parity tests."""
        return super().encode_candidates(
            batch,
            candidate_slice=candidate_slice,
            parent_state=parent_state,
        )

    def encode_candidates(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditEncoding:
        encoding = self.encode_base_candidates(
            batch,
            candidate_slice=candidate_slice,
            parent_state=parent_state,
        )
        selected = candidate_slice or slice(None)
        intent_features = batch.intent_features[:, selected]
        if intent_features.shape != (
            encoding.hidden.shape[0],
            encoding.hidden.shape[1],
            self.config.intent_feature_dim,
        ):
            raise ValueError("O1 intent features do not align with candidate encoding")
        intent = self.intent_projection(intent_features)
        fused = self.intent_fusion(
            mx.concatenate(
                [
                    encoding.hidden,
                    intent,
                    encoding.hidden * intent,
                ],
                axis=-1,
            )
        )
        contextualized = (
            encoding.hidden + self.intent_delta(fused)
        ) * encoding.candidate_mask[..., None]
        return R3ActionEditEncoding(
            hidden=contextualized,
            candidate_mask=encoding.candidate_mask,
        )


def o1_ranking_loss(
    model: O1IntentConditionedRanker,
    batch: object,
) -> mx.array:
    """Reuse the frozen ADR 0150 objective without new auxiliaries."""
    return r3_action_edit_loss(model, batch)


def o1_ranking_loss_components(
    model: O1IntentConditionedRanker,
    batch: object,
) -> dict[str, mx.array]:
    return r3_action_edit_loss_components(model, batch)


def parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    tree = model.trainable_parameters() if trainable_only else model.parameters()
    return sum(int(value.size) for _, value in tree_flatten(tree))


def parameter_layout_blake3(
    model: nn.Module,
    *,
    trainable_only: bool = True,
) -> str:
    tree = model.trainable_parameters() if trainable_only else model.parameters()
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(tree)
    ]
    payload = json.dumps(
        layout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def parameter_tensor_blake3(
    model: nn.Module,
    *,
    trainable_only: bool = True,
) -> str:
    tree = model.trainable_parameters() if trainable_only else model.parameters()
    digest = blake3.blake3()
    for name, value in tree_flatten(tree):
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()
