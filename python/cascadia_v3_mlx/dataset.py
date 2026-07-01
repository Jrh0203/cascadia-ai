"""Deterministic packed sparse batches for V3 engineering and training."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .contracts import V3MlxConfig
from .model import SparseBatch


@dataclass(frozen=True)
class SparseWidths:
    own_base: int = 48
    field_base: int = 96
    own_opportunities: int = 96
    field_opportunities: int = 192
    own_opportunity_factors: int = 96
    field_opportunity_factors: int = 192

    def validate(self) -> None:
        if (
            min(
                self.own_base,
                self.field_base,
                self.own_opportunities,
                self.field_opportunities,
                self.own_opportunity_factors,
                self.field_opportunity_factors,
            )
            <= 0
        ):
            raise ValueError("sparse widths must be positive")


def _rows(
    rng: np.random.Generator,
    batch_size: int,
    padded_width: int,
    feature_rows: int,
    *,
    maximum_count: int,
) -> tuple[mx.array, mx.array, mx.array]:
    active = rng.integers(max(1, padded_width // 2), padded_width + 1, size=batch_size)
    indices = np.zeros((batch_size, padded_width), dtype=np.int32)
    counts = np.zeros((batch_size, padded_width), dtype=np.float32)
    mask = np.zeros((batch_size, padded_width), dtype=np.bool_)
    for row, count in enumerate(active):
        selected = rng.choice(feature_rows, size=int(count), replace=False)
        selected.sort()
        indices[row, :count] = selected
        counts[row, :count] = rng.integers(1, maximum_count + 1, size=int(count))
        mask[row, :count] = True
    return mx.array(indices), mx.array(counts), mx.array(mask)


def synthetic_batch(
    config: V3MlxConfig,
    batch_size: int,
    seed: int,
    widths: SparseWidths | None = None,
) -> SparseBatch:
    config.validate()
    widths = widths or SparseWidths()
    widths.validate()
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(seed)
    own_base = _rows(
        rng,
        batch_size,
        widths.own_base,
        config.base_feature_rows,
        maximum_count=1,
    )
    field_base = _rows(
        rng,
        batch_size,
        widths.field_base,
        config.base_feature_rows,
        maximum_count=3,
    )
    own_opportunities = _rows(
        rng,
        batch_size,
        widths.own_opportunities,
        config.opportunity_feature_rows,
        maximum_count=4,
    )
    field_opportunities = _rows(
        rng,
        batch_size,
        widths.field_opportunities,
        config.opportunity_feature_rows,
        maximum_count=8,
    )
    own_opportunity_factors = _rows(
        rng,
        batch_size,
        widths.own_opportunity_factors,
        config.opportunity_training_factor_rows,
        maximum_count=16,
    )
    field_opportunity_factors = _rows(
        rng,
        batch_size,
        widths.field_opportunity_factors,
        config.opportunity_training_factor_rows,
        maximum_count=32,
    )
    batch = SparseBatch(
        own_base_indices=own_base[0],
        own_base_counts=own_base[1],
        own_base_mask=own_base[2],
        field_base_indices=field_base[0],
        field_base_counts=field_base[1],
        field_base_mask=field_base[2],
        own_opportunity_indices=own_opportunities[0],
        own_opportunity_counts=own_opportunities[1],
        own_opportunity_mask=own_opportunities[2],
        field_opportunity_indices=field_opportunities[0],
        field_opportunity_counts=field_opportunities[1],
        field_opportunity_mask=field_opportunities[2],
        own_opportunity_factor_indices=own_opportunity_factors[0],
        own_opportunity_factor_counts=own_opportunity_factors[1],
        own_opportunity_factor_mask=own_opportunity_factors[2],
        field_opportunity_factor_indices=field_opportunity_factors[0],
        field_opportunity_factor_counts=field_opportunity_factors[1],
        field_opportunity_factor_mask=field_opportunity_factors[2],
        phase_buckets=mx.array(rng.integers(0, 8, size=batch_size, dtype=np.int32)),
        targets=mx.array(rng.normal(75.0, 8.0, size=batch_size).astype(np.float32)),
        confidence_weights=mx.array(rng.uniform(0.25, 4.0, size=batch_size).astype(np.float32)),
    )
    batch.validate(config)
    return batch


def slice_batch(batch: SparseBatch, start: int, stop: int) -> SparseBatch:
    values = {
        field: getattr(batch, field)[start:stop] for field in SparseBatch.__dataclass_fields__
    }
    return SparseBatch(**values)
