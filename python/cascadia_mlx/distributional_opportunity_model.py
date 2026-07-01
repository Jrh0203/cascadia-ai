"""Matched distributional heads for R12 counterfactual opportunity value."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import blake3
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.action_ranking_model import (
    encode_action_afterstates,
    initialize_action_afterstate_encoder,
)
from cascadia_mlx.counterfactual_advantage_dataset import (
    COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.model import SetAttentionBlock

ARMS = (
    "c0-homoscedastic-mean",
    "g1-heteroscedastic-gaussian",
    "q2-quantile",
    "e3-crps-atoms",
)
ATOM_COUNT = 12
CORRECTION_SCALE = 8.0
ATOM_OFFSET_SCALE = 8.0
MINIMUM_GAUSSIAN_SCALE = 0.25
TEACHER_TEMPERATURE = 0.5
HARD_TOP_WEIGHT = 0.50
SOFT_LISTWISE_WEIGHT = 0.25
DISTRIBUTION_WEIGHT = 0.25
AUXILIARY_REGULARIZATION = 0.01
QUANTILE_CROSSING_WEIGHT = 0.05


@dataclass(frozen=True)
class DistributionalOpportunityModelConfig:
    """One parameter graph shared by every matched uncertainty arm."""

    schema_version: int = 1
    architecture: str = "r12-distributional-opportunity-set-ranker-v1"
    arm: str = ARMS[0]
    hidden_dim: int = 96
    attention_heads: int = 4
    board_blocks: int = 2
    market_blocks: int = 1
    candidate_blocks: int = 2
    feed_forward_multiplier: int = 3
    atom_count: int = ATOM_COUNT

    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.architecture != "r12-distributional-opportunity-set-ranker-v1"
            or self.arm not in ARMS
            or self.atom_count != ATOM_COUNT
        ):
            raise ValueError("unsupported distributional-opportunity configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if min(self.board_blocks, self.market_blocks, self.candidate_blocks) < 0:
            raise ValueError("attention block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, object],
    ) -> DistributionalOpportunityModelConfig:
        config = cls(**values)
        config.validate()
        return config


class DistributionalOpportunityRanker(nn.Module):
    """Predict candidate means and matched distribution parameters."""

    def __init__(
        self,
        config: DistributionalOpportunityModelConfig | None = None,
    ):
        super().__init__()
        config = config or DistributionalOpportunityModelConfig()
        config.validate()
        self.config = config
        initialize_action_afterstate_encoder(self, config)
        hidden = config.hidden_dim
        self.public_supply_projection = nn.Sequential(
            nn.Linear(COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden * 13, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.candidate_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.candidate_blocks)
        ]
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1 + ATOM_COUNT),
        )
        output = self.output_head.layers[-1]
        output.weight = mx.zeros_like(output.weight)
        output.bias = mx.zeros_like(output.bias)

    def __call__(self, batch: object) -> tuple[mx.array, mx.array]:
        groups, candidates = batch.global_features.shape[:2]
        encoded = encode_action_afterstates(
            self,
            self.config,
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.action_features,
        )
        supply = self.public_supply_projection(batch.public_supply)
        supply = mx.repeat(supply[:, None, :], candidates, axis=1).reshape(
            groups * candidates,
            self.config.hidden_dim,
        )
        values = self.candidate_projection(mx.concatenate([encoded, supply], axis=-1))
        values = values.reshape(groups, candidates, self.config.hidden_dim)
        values = values * batch.candidate_mask[..., None]
        for block in self.candidate_blocks:
            values = block(values, batch.candidate_mask)
        output = self.output_head(values)
        scores = batch.immediate_score + CORRECTION_SCALE * mx.tanh(output[..., 0])
        means = _center_candidates(scores, batch.candidate_mask)
        return means, output[..., 1:]

    def distribution(
        self,
        batch: object,
        *,
        homoscedastic_offsets: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return centered means, ordered evaluation atoms, and uncertainty."""
        means, auxiliary = self(batch)
        if self.config.arm == "c0-homoscedastic-mean":
            if homoscedastic_offsets is None:
                raise ValueError("homoscedastic control requires frozen offsets")
            atoms = means[..., None] + homoscedastic_offsets
        elif self.config.arm == "g1-heteroscedastic-gaussian":
            scale = MINIMUM_GAUSSIAN_SCALE + nn.softplus(auxiliary[..., 0])
            atoms = means[..., None] + scale[..., None] * normal_atom_locations()
        else:
            offsets = ATOM_OFFSET_SCALE * mx.tanh(auxiliary)
            offsets = offsets - mx.mean(offsets, axis=-1, keepdims=True)
            atoms = means[..., None] + offsets
            atoms = mx.sort(atoms, axis=-1)
        uncertainty = atoms[..., -1] - atoms[..., 0]
        return means, atoms, uncertainty


def distributional_opportunity_loss(
    model: DistributionalOpportunityRanker,
    batch: object,
    *,
    homoscedastic_offsets: mx.array | None = None,
) -> mx.array:
    """Hold the expected-score objective fixed while varying distribution loss."""
    means, auxiliary = model(batch)
    targets = mx.mean(batch.target_centered_samples, axis=-1)
    mask = batch.candidate_mask
    centered = _masked_mean(_huber(means - targets), mask)

    masked_predictions = mx.where(mask, means, -1e9)
    log_probabilities = masked_predictions - mx.logsumexp(
        masked_predictions,
        axis=-1,
        keepdims=True,
    )
    target_max = mx.max(mx.where(mask, targets, -1e9), axis=-1, keepdims=True)
    hard_top_mask = mask & (mx.abs(targets - target_max) < 1e-6)
    hard_top_count = mx.maximum(mx.sum(hard_top_mask, axis=-1, keepdims=True), 1)
    hard_top = -mx.sum(
        hard_top_mask / hard_top_count * log_probabilities,
        axis=-1,
    )
    teacher_logits = mx.where(mask, targets / TEACHER_TEMPERATURE, -1e9)
    soft_listwise = -mx.sum(
        mx.softmax(teacher_logits, axis=-1) * log_probabilities,
        axis=-1,
    )
    ranking = mx.mean(HARD_TOP_WEIGHT * hard_top + SOFT_LISTWISE_WEIGHT * soft_listwise)
    mean_objective = centered + ranking

    samples = batch.target_centered_samples
    arm = model.config.arm
    if arm == "c0-homoscedastic-mean":
        if homoscedastic_offsets is None:
            raise ValueError("homoscedastic control requires frozen offsets")
        atoms = means[..., None] + homoscedastic_offsets
        distribution = _empirical_crps(atoms, samples)
        distribution += AUXILIARY_REGULARIZATION * mx.mean(mx.square(auxiliary))
    elif arm == "g1-heteroscedastic-gaussian":
        scale = MINIMUM_GAUSSIAN_SCALE + nn.softplus(auxiliary[..., 0])
        standardized = (samples - means[..., None]) / scale[..., None]
        nll = 0.5 * mx.square(standardized) + mx.log(scale[..., None])
        distribution = mx.mean(nll) + AUXILIARY_REGULARIZATION * mx.mean(
            mx.square(auxiliary[..., 1:])
        )
    else:
        offsets = ATOM_OFFSET_SCALE * mx.tanh(auxiliary)
        offsets = offsets - mx.mean(offsets, axis=-1, keepdims=True)
        atoms = means[..., None] + offsets
        if arm == "q2-quantile":
            errors = samples[..., :, None] - atoms[..., None, :]
            taus = quantile_levels()
            pinball = mx.maximum(
                taus * errors,
                (taus - 1.0) * errors,
            )
            crossings = nn.relu(atoms[..., :-1] - atoms[..., 1:])
            distribution = mx.mean(pinball) + QUANTILE_CROSSING_WEIGHT * mx.mean(crossings)
        elif arm == "e3-crps-atoms":
            distribution = _empirical_crps(atoms, samples)
        else:
            raise ValueError(f"unknown distributional arm: {arm}")
    return mean_objective + DISTRIBUTION_WEIGHT * distribution


def quantile_levels() -> mx.array:
    return (mx.arange(ATOM_COUNT, dtype=mx.float32) + 0.5) / ATOM_COUNT


def normal_atom_locations() -> mx.array:
    return mx.array(
        (
            -1.731664396,
            -1.150349380,
            -0.812217801,
            -0.548522282,
            -0.318639364,
            -0.104633455,
            0.104633455,
            0.318639364,
            0.548522282,
            0.812217801,
            1.150349380,
            1.731664396,
        ),
        dtype=mx.float32,
    )


def parameter_count(model: DistributionalOpportunityRanker) -> int:
    return sum(int(value.size) for _, value in tree_flatten(model.trainable_parameters()))


def parameter_layout_blake3(model: DistributionalOpportunityRanker) -> str:
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(model.trainable_parameters())
    ]
    payload = json.dumps(
        layout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def parameter_tensor_blake3(model: DistributionalOpportunityRanker) -> str:
    digest = blake3.blake3()
    for name, value in tree_flatten(model.trainable_parameters()):
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()


def _center_candidates(values: mx.array, mask: mx.array) -> mx.array:
    count = mx.maximum(mx.sum(mask, axis=-1, keepdims=True), 1)
    mean = mx.sum(mx.where(mask, values, 0.0), axis=-1, keepdims=True) / count
    return mx.where(mask, values - mean, 0.0)


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    return mx.sum(mx.where(mask, values, 0.0)) / mx.maximum(mx.sum(mask), 1)


def _empirical_crps(atoms: mx.array, samples: mx.array) -> mx.array:
    sample_distance = mx.mean(mx.abs(atoms[..., :, None] - samples[..., None, :]))
    atom_distance = mx.mean(mx.abs(atoms[..., :, None] - atoms[..., None, :]))
    return sample_distance - 0.5 * atom_distance


def _huber(errors: mx.array) -> mx.array:
    absolute = mx.abs(errors)
    return mx.where(absolute <= 1.0, 0.5 * mx.square(errors), absolute - 0.5)


__all__ = [
    "ARMS",
    "ATOM_COUNT",
    "DistributionalOpportunityModelConfig",
    "DistributionalOpportunityRanker",
    "distributional_opportunity_loss",
    "normal_atom_locations",
    "parameter_count",
    "parameter_layout_blake3",
    "parameter_tensor_blake3",
    "quantile_levels",
]
