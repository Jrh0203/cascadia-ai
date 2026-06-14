"""Differentiable MLX value network for rollout-return fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.legacy_nnue import (
    LEGACY_NNUE_FEATURES,
    LEGACY_NNUE_HIDDEN1,
    LEGACY_NNUE_HIDDEN2,
    LegacySparseNnue,
)

ROLLOUT_VALUE_ARCHITECTURE = "legacy-sparse-nnue-rollout-return-v1"
ROLLOUT_VALUE_HUBER_DELTA = 4.0
ROLLOUT_ROOT_HUBER_DELTA = 1.0
ROLLOUT_ROOT_SELECTED_WEIGHT = 0.50
ROLLOUT_ROOT_TEACHER_WEIGHT = 0.25
ROLLOUT_ROOT_TEACHER_TEMPERATURE = 1.0
VALUE_TENSOR_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3")


@dataclass(frozen=True)
class RolloutValueNnueConfig:
    schema_version: int = 1
    architecture: str = ROLLOUT_VALUE_ARCHITECTURE
    features: int = LEGACY_NNUE_FEATURES
    hidden1: int = LEGACY_NNUE_HIDDEN1
    hidden2: int = LEGACY_NNUE_HIDDEN2

    def validate(self) -> None:
        if self.schema_version != 1 or self.architecture != ROLLOUT_VALUE_ARCHITECTURE:
            raise ValueError("unsupported rollout-value NNUE configuration")
        if self.features <= 0 or self.hidden1 <= 0 or self.hidden2 <= 0:
            raise ValueError("rollout-value NNUE dimensions must be positive")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> RolloutValueNnueConfig:
        config = cls(**values)
        config.validate()
        return config


class RolloutValueNnue(nn.Module):
    """The qualified sparse NNUE value path expressed with trainable primitives."""

    def __init__(
        self,
        config: RolloutValueNnueConfig | None = None,
        *,
        tensors: dict[str, mx.array] | None = None,
    ):
        super().__init__()
        config = config or RolloutValueNnueConfig()
        config.validate()
        self.config = config
        if tensors is None:
            scale1 = (2.0 / max(config.features, 1)) ** 0.5
            scale2 = (2.0 / config.hidden1) ** 0.5
            self.w1 = mx.random.normal((config.features, config.hidden1)) * scale1
            self.b1 = mx.zeros((config.hidden1,))
            self.w2 = mx.random.normal((config.hidden1, config.hidden2)) * scale2
            self.b2 = mx.zeros((config.hidden2,))
            self.w3 = mx.zeros((config.hidden2,))
            self.b3 = mx.zeros((1,))
        else:
            expected = {
                "w1": (config.features, config.hidden1),
                "b1": (config.hidden1,),
                "w2": (config.hidden1, config.hidden2),
                "b2": (config.hidden2,),
                "w3": (config.hidden2,),
                "b3": (1,),
            }
            if set(tensors) != set(expected):
                raise ValueError("rollout-value tensors do not match the value-path contract")
            for name, shape in expected.items():
                if tuple(tensors[name].shape) != shape:
                    raise ValueError(f"rollout-value tensor {name} has invalid shape")
                setattr(self, name, mx.array(tensors[name]))

    @classmethod
    def from_parent(cls, parent_model_dir: str | Path) -> RolloutValueNnue:
        parent = LegacySparseNnue.load(parent_model_dir)
        return cls(tensors={name: parent.tensors[name] for name in VALUE_TENSOR_NAMES})

    def __call__(self, feature_indices: mx.array, feature_mask: mx.array) -> mx.array:
        if feature_indices.ndim != 2 or feature_mask.shape != feature_indices.shape:
            raise ValueError("rollout-value sparse inputs must be matching rank-two arrays")
        gathered = mx.take(self.w1, feature_indices, axis=0)
        h1 = self.b1 + mx.sum(
            gathered * feature_mask[..., None].astype(gathered.dtype),
            axis=1,
        )
        h1 = mx.maximum(h1, 0.0)
        h2 = mx.maximum(h1 @ self.w2 + self.b2, 0.0)
        return h2 @ self.w3 + self.b3[0]


def rollout_value_loss(model: RolloutValueNnue, batch: object) -> mx.array:
    predictions = model(batch.feature_indices, batch.feature_mask)
    return nn.losses.huber_loss(
        predictions,
        batch.target_remaining,
        delta=ROLLOUT_VALUE_HUBER_DELTA,
        reduction="mean",
    )


def rollout_root_scores(model: RolloutValueNnue, batch: object) -> mx.array:
    groups, candidates, feature_width = batch.feature_indices.shape
    remaining = model(
        batch.feature_indices.reshape(groups * candidates, feature_width),
        batch.feature_mask.reshape(groups * candidates, feature_width),
    ).reshape(groups, candidates)
    return batch.immediate_score + remaining


def rollout_root_ranking_loss(model: RolloutValueNnue, batch: object) -> mx.array:
    """Optimize within-decision score differences and selected-action ordering."""
    mask = batch.candidate_mask
    mask_float = mask.astype(mx.float32)
    predictions = rollout_root_scores(model, batch)
    targets = batch.immediate_score + batch.target_remaining
    candidate_count = mx.maximum(mx.sum(mask_float, axis=-1, keepdims=True), 1.0)

    prediction_mean = mx.sum(mx.where(mask, predictions, 0.0), axis=-1, keepdims=True)
    prediction_mean /= candidate_count
    target_mean = mx.sum(mx.where(mask, targets, 0.0), axis=-1, keepdims=True)
    target_mean /= candidate_count
    centered_error = (predictions - prediction_mean) - (targets - target_mean)
    centered = mx.sum(mask_float * _root_huber(centered_error)) / mx.sum(mask_float)

    logits = mx.where(
        mask,
        predictions / ROLLOUT_ROOT_TEACHER_TEMPERATURE,
        -1e9,
    )
    log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected = batch.selected.astype(predictions.dtype)
    selected_listwise = -mx.mean(mx.sum(selected * log_probabilities, axis=-1))

    teacher_logits = mx.where(
        mask,
        targets / ROLLOUT_ROOT_TEACHER_TEMPERATURE,
        -1e9,
    )
    teacher_probabilities = mx.softmax(teacher_logits, axis=-1)
    teacher_listwise = -mx.mean(mx.sum(teacher_probabilities * log_probabilities, axis=-1))
    return (
        centered
        + ROLLOUT_ROOT_SELECTED_WEIGHT * selected_listwise
        + ROLLOUT_ROOT_TEACHER_WEIGHT * teacher_listwise
    )


def _root_huber(errors: mx.array) -> mx.array:
    absolute = mx.abs(errors)
    return mx.where(
        absolute <= ROLLOUT_ROOT_HUBER_DELTA,
        0.5 * mx.square(errors),
        ROLLOUT_ROOT_HUBER_DELTA * (absolute - 0.5 * ROLLOUT_ROOT_HUBER_DELTA),
    )
