"""MLX implementation of the Cascadia V3 SFNNv13-style network."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .contracts import (
    DENSE_SCALE,
    FC0_NONLINEAR,
    FC0_OUTPUTS,
    FC1_INPUTS,
    FC1_OUTPUTS,
    FEATURE_SCALE,
    GLOBAL_BASE,
    PHASE_BUCKETS,
    POOL_HALF,
    TRANSFORM_WIDTH,
    V3MlxConfig,
)

ACCUMULATOR_HEADROOM_LIMIT = 64.0
ACCUMULATOR_HEADROOM_COEFFICIENT = 0.1


@dataclass(frozen=True, slots=True)
class SparseBatch:
    own_base_indices: mx.array
    own_base_counts: mx.array
    own_base_mask: mx.array
    field_base_indices: mx.array
    field_base_counts: mx.array
    field_base_mask: mx.array
    own_opportunity_indices: mx.array
    own_opportunity_counts: mx.array
    own_opportunity_mask: mx.array
    field_opportunity_indices: mx.array
    field_opportunity_counts: mx.array
    field_opportunity_mask: mx.array
    own_opportunity_factor_indices: mx.array
    own_opportunity_factor_counts: mx.array
    own_opportunity_factor_mask: mx.array
    field_opportunity_factor_indices: mx.array
    field_opportunity_factor_counts: mx.array
    field_opportunity_factor_mask: mx.array
    phase_buckets: mx.array
    targets: mx.array
    confidence_weights: mx.array

    def validate(self, config: V3MlxConfig) -> int:
        arrays = (
            (
                self.own_base_indices,
                self.own_base_counts,
                self.own_base_mask,
                config.base_feature_rows,
            ),
            (
                self.field_base_indices,
                self.field_base_counts,
                self.field_base_mask,
                config.base_feature_rows,
            ),
            (
                self.own_opportunity_indices,
                self.own_opportunity_counts,
                self.own_opportunity_mask,
                config.opportunity_feature_rows,
            ),
            (
                self.field_opportunity_indices,
                self.field_opportunity_counts,
                self.field_opportunity_mask,
                config.opportunity_feature_rows,
            ),
            (
                self.own_opportunity_factor_indices,
                self.own_opportunity_factor_counts,
                self.own_opportunity_factor_mask,
                config.opportunity_training_factor_rows,
            ),
            (
                self.field_opportunity_factor_indices,
                self.field_opportunity_factor_counts,
                self.field_opportunity_factor_mask,
                config.opportunity_training_factor_rows,
            ),
        )
        batch_size = int(self.phase_buckets.shape[0])
        for indices, counts, mask, width in arrays:
            if indices.ndim != 2 or counts.shape != indices.shape or mask.shape != indices.shape:
                raise ValueError("V3 sparse batch rows must be matching rank-two tensors")
            if int(indices.shape[0]) != batch_size:
                raise ValueError("V3 sparse tensors disagree on batch size")
            if indices.dtype != mx.int32 or counts.dtype != mx.float32 or mask.dtype != mx.bool_:
                raise ValueError("V3 sparse tensor dtypes drifted")
            if int(mx.max(mx.where(mask, indices, mx.zeros_like(indices))).item()) >= width:
                raise ValueError("V3 sparse feature index is out of range")
        if self.phase_buckets.shape != (batch_size,) or self.phase_buckets.dtype != mx.int32:
            raise ValueError("V3 phase buckets must be int32 [batch]")
        if self.targets.shape != (batch_size,) or self.confidence_weights.shape != (batch_size,):
            raise ValueError("V3 target tensors must be [batch]")
        if self.targets.dtype != mx.float32 or self.confidence_weights.dtype != mx.float32:
            raise ValueError("V3 targets and confidence weights must be float32")
        return batch_size


@dataclass(frozen=True, slots=True)
class CsrRows:
    offsets: mx.array
    indices: mx.array
    counts: mx.array
    row_indices: mx.array
    gradient_positions: mx.array
    gradient_features: mx.array
    gradient_offsets: mx.array

    def validate(self, rows: int, feature_rows: int) -> None:
        if (
            self.offsets.shape != (rows + 1,)
            or self.indices.ndim != 1
            or self.counts.shape != self.indices.shape
            or self.row_indices.shape != self.indices.shape
            or self.gradient_positions.shape != self.indices.shape
            or self.gradient_features.ndim != 1
            or self.gradient_offsets.shape != (self.gradient_features.size + 1,)
        ):
            raise ValueError("V3 CSR tensors have invalid shapes")
        if (
            self.offsets.dtype != mx.int32
            or self.indices.dtype != mx.int32
            or self.counts.dtype != mx.float32
            or self.row_indices.dtype != mx.int32
            or self.gradient_positions.dtype != mx.int32
            or self.gradient_features.dtype != mx.int32
            or self.gradient_offsets.dtype != mx.int32
        ):
            raise ValueError("V3 CSR tensor dtypes drifted")
        if self.indices.size and int(mx.max(self.indices).item()) >= feature_rows:
            raise ValueError("V3 CSR feature index is out of range")


@dataclass(frozen=True, slots=True)
class CsrBatch:
    own_base: CsrRows
    field_base: CsrRows
    own_opportunities: CsrRows
    field_opportunities: CsrRows
    own_opportunity_factors: CsrRows
    field_opportunity_factors: CsrRows
    phase_buckets: mx.array
    targets: mx.array
    confidence_weights: mx.array

    def validate(self, config: V3MlxConfig) -> int:
        rows = int(self.phase_buckets.shape[0])
        self.own_base.validate(rows, config.base_feature_rows)
        self.field_base.validate(rows, config.base_feature_rows)
        self.own_opportunities.validate(rows, config.opportunity_feature_rows)
        self.field_opportunities.validate(rows, config.opportunity_feature_rows)
        self.own_opportunity_factors.validate(rows, config.opportunity_training_factor_rows)
        self.field_opportunity_factors.validate(rows, config.opportunity_training_factor_rows)
        if self.phase_buckets.dtype != mx.int32:
            raise ValueError("V3 CSR phases must be int32")
        if self.targets.shape != (rows,) or self.confidence_weights.shape != (rows,):
            raise ValueError("V3 CSR targets must be [rows]")
        return rows


_CSR_BAG_1024 = mx.fast.metal_kernel(
    name="cascadia_v3_csr_embedding_bag_1024_v1",
    input_names=["offsets", "indices", "counts", "weight", "bias"],
    output_names=["out"],
    source=r"""
        uint lane = thread_position_in_grid.x * 4;
        uint row = thread_position_in_grid.y;
        float4 value = *((device const float4*)(bias + lane));
        uint start = uint(offsets[row]);
        uint end = uint(offsets[row + 1]);
        for (uint position = start; position < end; ++position) {
            uint feature = uint(indices[position]);
            float count = counts[position];
            value += count * *((device const float4*)(weight + feature * 1024 + lane));
        }
        *((device float4*)(out + row * 1024 + lane)) = value;
    """,
)

_CSR_BAG_1024_BACKWARD = mx.fast.metal_kernel(
    name="cascadia_v3_csr_embedding_bag_1024_backward_deterministic_v1",
    input_names=[
        "counts",
        "row_indices",
        "gradient_positions",
        "gradient_features",
        "gradient_offsets",
        "cotangent",
    ],
    output_names=["weight_gradient"],
    source=r"""
        uint lane = thread_position_in_grid.x * 4;
        uint group = thread_position_in_grid.y;
        uint feature = uint(gradient_features[group]);
        uint start = uint(gradient_offsets[group]);
        uint end = uint(gradient_offsets[group + 1]);
        for (uint offset = 0; offset < 4; ++offset) {
            float value = 0.0f;
            for (uint cursor = start; cursor < end; ++cursor) {
                uint position = uint(gradient_positions[cursor]);
                uint row = uint(row_indices[position]);
                value += counts[position] * cotangent[row * 1024 + lane + offset];
            }
            weight_gradient[feature * 1024 + lane + offset] = value;
        }
    """,
)

_CSR_BAG_8 = mx.fast.metal_kernel(
    name="cascadia_v3_csr_embedding_bag_8_v2",
    input_names=["offsets", "indices", "counts", "weight", "bias"],
    output_names=["out"],
    source=r"""
        uint lane = thread_position_in_grid.x * 4;
        uint row = thread_position_in_grid.y;
        float4 value = *((device const float4*)(bias + lane));
        uint start = uint(offsets[row]);
        uint end = uint(offsets[row + 1]);
        for (uint position = start; position < end; ++position) {
            uint feature = uint(indices[position]);
            if (feature >= 10066) {
                continue;
            }
            float count = counts[position];
            value += count * *((device const float4*)(weight + feature * 8 + lane));
        }
        *((device float4*)(out + row * 8 + lane)) = value;
    """,
)

_CSR_BAG_8_BACKWARD = mx.fast.metal_kernel(
    name="cascadia_v3_csr_embedding_bag_8_backward_deterministic_v1",
    input_names=[
        "counts",
        "row_indices",
        "gradient_positions",
        "gradient_features",
        "gradient_offsets",
        "cotangent",
    ],
    output_names=["weight_gradient"],
    source=r"""
        uint lane = thread_position_in_grid.x * 4;
        uint group = thread_position_in_grid.y;
        uint feature = uint(gradient_features[group]);
        if (feature >= 10066) {
            return;
        }
        uint start = uint(gradient_offsets[group]);
        uint end = uint(gradient_offsets[group + 1]);
        for (uint offset = 0; offset < 4; ++offset) {
            float value = 0.0f;
            for (uint cursor = start; cursor < end; ++cursor) {
                uint position = uint(gradient_positions[cursor]);
                uint row = uint(row_indices[position]);
                value += counts[position] * cotangent[row * 8 + lane + offset];
            }
            weight_gradient[feature * 8 + lane + offset] = value;
        }
    """,
)


def _csr_bag_forward(
    weight: mx.array,
    bias: mx.array,
    offsets: mx.array,
    indices: mx.array,
    counts: mx.array,
    row_indices: mx.array,
    gradient_positions: mx.array,
    gradient_features: mx.array,
    gradient_offsets: mx.array,
) -> mx.array:
    del row_indices, gradient_positions, gradient_features, gradient_offsets
    rows = int(offsets.shape[0]) - 1
    width = int(weight.shape[1])
    if width == 1_024:
        kernel, lanes = _CSR_BAG_1024, 256
    elif width == 8:
        kernel, lanes = _CSR_BAG_8, 2
    else:
        raise ValueError(f"unsupported V3 CSR embedding width {width}")
    return kernel(
        inputs=[offsets, indices, counts, weight, bias],
        output_shapes=[(rows, width)],
        output_dtypes=[mx.float32],
        # Keep each Metal grid dimension bounded independently. A flattened
        # ``rows * lanes`` launch overflows MLX/Metal's u32 grid contract at
        # campaign batches even though both logical dimensions are valid.
        grid=(lanes, rows, 1),
        threadgroup=(min(lanes, 256), 1, 1),
    )[0]


_csr_embedding_bag = mx.custom_function(_csr_bag_forward)


@_csr_embedding_bag.vjp
def _csr_embedding_bag_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array | None, ...]:
    del output
    (
        weight,
        _bias,
        _offsets,
        indices,
        counts,
        row_indices,
        gradient_positions,
        gradient_features,
        gradient_offsets,
    ) = primals
    width = int(weight.shape[1])
    if width == 1_024:
        kernel, lanes = _CSR_BAG_1024_BACKWARD, 256
    elif width == 8:
        kernel, lanes = _CSR_BAG_8_BACKWARD, 2
    else:
        raise ValueError(f"unsupported V3 CSR gradient width {width}")
    if int(indices.size) == 0 or int(gradient_features.size) == 0:
        gradient = mx.zeros_like(weight)
    else:
        gradient = kernel(
            inputs=[
                counts,
                row_indices,
                gradient_positions,
                gradient_features,
                gradient_offsets,
                cotangent,
            ],
            output_shapes=[weight.shape],
            output_dtypes=[mx.float32],
            grid=(lanes, int(gradient_features.size), 1),
            threadgroup=(min(lanes, 256), 1, 1),
            init_value=0.0,
        )[0]
    return (
        gradient,
        mx.sum(cotangent, axis=0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _fake_quantize(value: mx.array, scale: int, minimum: int, maximum: int) -> mx.array:
    quantized = mx.clip(mx.round(value * scale), minimum, maximum) / scale
    return value + mx.stop_gradient(quantized - value)


def _fake_floor_positive(value: mx.array, scale: int, maximum_units: int) -> mx.array:
    quantized = mx.clip(mx.floor(value * scale), 0, maximum_units) / scale
    return value + mx.stop_gradient(quantized - value)


def _fake_round_away(
    value: mx.array,
    scale: int,
    minimum_units: int,
    maximum_units: int,
) -> mx.array:
    scaled = value * scale
    rounded = mx.where(scaled >= 0, mx.floor(scaled + 0.5), mx.ceil(scaled - 0.5))
    quantized = mx.clip(rounded, minimum_units, maximum_units) / scale
    return value + mx.stop_gradient(quantized - value)


def _embedding_bag(
    weight: mx.array,
    indices: mx.array,
    counts: mx.array,
    mask: mx.array,
) -> mx.array:
    selected = mx.take(weight, indices, axis=0)
    multiplier = counts * mask.astype(mx.float32)
    return mx.sum(selected * multiplier[..., None], axis=1)


class V3Nnue(nn.Module):
    """Shared sparse transformer with own/field product pooling and eight heads."""

    def __init__(self, config: V3MlxConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.base_embedding = nn.Embedding(config.base_feature_rows, TRANSFORM_WIDTH)
        self.opportunity_embedding = nn.Embedding(
            config.opportunity_feature_rows,
            TRANSFORM_WIDTH,
        )
        self.opportunity_factor_embedding = nn.Embedding(
            config.opportunity_training_factor_rows,
            TRANSFORM_WIDTH,
        )
        self.transformer_bias = mx.zeros((TRANSFORM_WIDTH,), dtype=mx.float32)
        self.direct_potential = nn.Embedding(config.base_feature_rows, PHASE_BUCKETS)
        self.fc0 = [nn.Linear(TRANSFORM_WIDTH, FC0_OUTPUTS) for _ in range(PHASE_BUCKETS)]
        self.fc1 = [nn.Linear(FC1_INPUTS, FC1_OUTPUTS) for _ in range(PHASE_BUCKETS)]
        self.fc2 = [nn.Linear(FC1_OUTPUTS, 1) for _ in range(PHASE_BUCKETS)]

    def _weights(self) -> tuple[mx.array, mx.array, mx.array]:
        if not self.config.qat:
            return (
                self.base_embedding.weight,
                self.opportunity_embedding.weight,
                self.opportunity_factor_embedding.weight,
            )
        return (
            _fake_quantize(self.base_embedding.weight, FEATURE_SCALE, -32768, 32767),
            _fake_quantize(self.opportunity_embedding.weight, FEATURE_SCALE, -128, 127),
            _fake_quantize(
                self.opportunity_factor_embedding.weight,
                FEATURE_SCALE,
                -128,
                127,
            ),
        )

    def _accumulate(
        self,
        base_indices: mx.array,
        base_counts: mx.array,
        base_mask: mx.array,
        opportunity_indices: mx.array,
        opportunity_counts: mx.array,
        opportunity_mask: mx.array,
        factor_indices: mx.array,
        factor_counts: mx.array,
        factor_mask: mx.array,
    ) -> mx.array:
        base_weight, opportunity_weight, factor_weight = self._weights()
        transformer_bias = self.transformer_bias
        if self.config.qat:
            transformer_bias = _fake_quantize(
                transformer_bias,
                FEATURE_SCALE,
                -32768,
                32767,
            )
        return (
            transformer_bias
            + _embedding_bag(base_weight, base_indices, base_counts, base_mask)
            + _embedding_bag(
                opportunity_weight,
                opportunity_indices,
                opportunity_counts,
                opportunity_mask,
            )
            + _embedding_bag(factor_weight, factor_indices, factor_counts, factor_mask)
        )

    def _accumulate_sparse(
        self, batch: SparseBatch
    ) -> tuple[mx.array, mx.array, mx.array]:
        own = self._accumulate(
            batch.own_base_indices,
            batch.own_base_counts,
            batch.own_base_mask,
            batch.own_opportunity_indices,
            batch.own_opportunity_counts,
            batch.own_opportunity_mask,
            batch.own_opportunity_factor_indices,
            batch.own_opportunity_factor_counts,
            batch.own_opportunity_factor_mask,
        )
        field = self._accumulate(
            batch.field_base_indices,
            batch.field_base_counts,
            batch.field_base_mask,
            batch.field_opportunity_indices,
            batch.field_opportunity_counts,
            batch.field_opportunity_mask,
            batch.field_opportunity_factor_indices,
            batch.field_opportunity_factor_counts,
            batch.field_opportunity_factor_mask,
        )
        direct_weight = self.direct_potential.weight
        if self.config.qat:
            direct_weight = _fake_quantize(
                direct_weight,
                self.config.output_scale,
                -32768,
                32767,
            )
        direct_by_phase = _embedding_bag(
            direct_weight,
            batch.own_base_indices,
            batch.own_base_counts,
            batch.own_base_mask & (batch.own_base_indices < GLOBAL_BASE),
        )
        return own, field, direct_by_phase

    def __call__(self, batch: SparseBatch) -> mx.array:
        own, field, direct = self._accumulate_sparse(batch)
        return self._dense(own, field, direct, batch.phase_buckets)

    def _accumulate_csr(self, batch: CsrBatch) -> tuple[mx.array, mx.array, mx.array]:
        batch.validate(self.config)
        base_weight, opportunity_weight, factor_weight = self._weights()
        transformer_bias = self.transformer_bias
        if self.config.qat:
            transformer_bias = _fake_quantize(
                transformer_bias,
                FEATURE_SCALE,
                -32768,
                32767,
            )
        zero = mx.zeros_like(transformer_bias)
        own = _csr_embedding_bag(
            base_weight,
            transformer_bias,
            batch.own_base.offsets,
            batch.own_base.indices,
            batch.own_base.counts,
            batch.own_base.row_indices,
            batch.own_base.gradient_positions,
            batch.own_base.gradient_features,
            batch.own_base.gradient_offsets,
        ) + _csr_embedding_bag(
            opportunity_weight,
            zero,
            batch.own_opportunities.offsets,
            batch.own_opportunities.indices,
            batch.own_opportunities.counts,
            batch.own_opportunities.row_indices,
            batch.own_opportunities.gradient_positions,
            batch.own_opportunities.gradient_features,
            batch.own_opportunities.gradient_offsets,
        ) + _csr_embedding_bag(
            factor_weight,
            zero,
            batch.own_opportunity_factors.offsets,
            batch.own_opportunity_factors.indices,
            batch.own_opportunity_factors.counts,
            batch.own_opportunity_factors.row_indices,
            batch.own_opportunity_factors.gradient_positions,
            batch.own_opportunity_factors.gradient_features,
            batch.own_opportunity_factors.gradient_offsets,
        )
        field = _csr_embedding_bag(
            base_weight,
            transformer_bias,
            batch.field_base.offsets,
            batch.field_base.indices,
            batch.field_base.counts,
            batch.field_base.row_indices,
            batch.field_base.gradient_positions,
            batch.field_base.gradient_features,
            batch.field_base.gradient_offsets,
        ) + _csr_embedding_bag(
            opportunity_weight,
            zero,
            batch.field_opportunities.offsets,
            batch.field_opportunities.indices,
            batch.field_opportunities.counts,
            batch.field_opportunities.row_indices,
            batch.field_opportunities.gradient_positions,
            batch.field_opportunities.gradient_features,
            batch.field_opportunities.gradient_offsets,
        ) + _csr_embedding_bag(
            factor_weight,
            zero,
            batch.field_opportunity_factors.offsets,
            batch.field_opportunity_factors.indices,
            batch.field_opportunity_factors.counts,
            batch.field_opportunity_factors.row_indices,
            batch.field_opportunity_factors.gradient_positions,
            batch.field_opportunity_factors.gradient_features,
            batch.field_opportunity_factors.gradient_offsets,
        )
        direct_weight = self.direct_potential.weight
        if self.config.qat:
            direct_weight = _fake_quantize(
                direct_weight,
                self.config.output_scale,
                -32768,
                32767,
            )
        direct = _csr_embedding_bag(
            direct_weight,
            mx.zeros((8,), dtype=mx.float32),
            batch.own_base.offsets,
            batch.own_base.indices,
            batch.own_base.counts,
            batch.own_base.row_indices,
            batch.own_base.gradient_positions,
            batch.own_base.gradient_features,
            batch.own_base.gradient_offsets,
        )
        return own, field, direct

    def call_csr(self, batch: CsrBatch) -> mx.array:
        own, field, direct = self._accumulate_csr(batch)
        return self._dense(own, field, direct, batch.phase_buckets)

    def _dense(
        self,
        own: mx.array,
        field: mx.array,
        direct_by_phase: mx.array,
        phase_buckets: mx.array,
    ) -> mx.array:
        own = mx.clip(own, 0.0, 1.0)
        field = mx.clip(field, 0.0, 1.0)
        pooled = mx.concatenate(
            (
                own[:, :POOL_HALF] * own[:, POOL_HALF:],
                field[:, :POOL_HALF] * field[:, POOL_HALF:],
            ),
            axis=-1,
        )
        if self.config.qat:
            pooled = _fake_floor_positive(pooled, FEATURE_SCALE, FEATURE_SCALE - 1)
        phase_outputs = []
        for phase in range(PHASE_BUCKETS):
            fc0_weight = self.fc0[phase].weight
            fc1_weight = self.fc1[phase].weight
            fc2_weight = self.fc2[phase].weight
            if self.config.qat:
                fc0_weight = _fake_quantize(fc0_weight, DENSE_SCALE, -128, 127)
                fc1_weight = _fake_quantize(fc1_weight, DENSE_SCALE, -128, 127)
                fc2_weight = _fake_quantize(fc2_weight, DENSE_SCALE, -128, 127)
                fc0_bias = _fake_quantize(
                    self.fc0[phase].bias,
                    DENSE_SCALE * FEATURE_SCALE,
                    -(2**31),
                    2**31 - 1,
                )
                fc1_bias = _fake_quantize(
                    self.fc1[phase].bias,
                    DENSE_SCALE * FEATURE_SCALE,
                    -(2**31),
                    2**31 - 1,
                )
                fc2_bias = _fake_quantize(
                    self.fc2[phase].bias,
                    DENSE_SCALE * FEATURE_SCALE,
                    -(2**31),
                    2**31 - 1,
                )
            else:
                fc0_bias = self.fc0[phase].bias
                fc1_bias = self.fc1[phase].bias
                fc2_bias = self.fc2[phase].bias
            first = pooled @ fc0_weight.T + fc0_bias
            if self.config.qat:
                first = _fake_round_away(first, FEATURE_SCALE, -(2**31), 2**31 - 1)
            nonlinear = mx.clip(
                first[:, :FC0_NONLINEAR],
                0.0,
                (FEATURE_SCALE - 1) / FEATURE_SCALE,
            )
            if self.config.qat:
                squared = _fake_floor_positive(
                    nonlinear * nonlinear * FEATURE_SCALE / (FEATURE_SCALE - 1),
                    FEATURE_SCALE,
                    FEATURE_SCALE - 1,
                )
            else:
                squared = nonlinear * nonlinear
            second_input = mx.concatenate((squared, nonlinear), axis=-1)
            second = second_input @ fc1_weight.T + fc1_bias
            if self.config.qat:
                second = _fake_round_away(second, FEATURE_SCALE, 0, FEATURE_SCALE - 1)
            second = mx.clip(second, 0.0, (FEATURE_SCALE - 1) / FEATURE_SCALE)
            value = second @ fc2_weight.T + fc2_bias
            if self.config.qat:
                value = _fake_round_away(
                    value,
                    self.config.output_scale,
                    -(2**31),
                    2**31 - 1,
                )
            skip = first[:, FC0_OUTPUTS - 1]
            if self.config.qat:
                skip = _fake_round_away(
                    skip,
                    self.config.output_scale,
                    -(2**31),
                    2**31 - 1,
                )
            value = value[:, 0] + skip + direct_by_phase[:, phase]
            phase_outputs.append(value)
        all_outputs = mx.stack(phase_outputs, axis=1)
        return mx.take_along_axis(all_outputs, phase_buckets[:, None], axis=1)[:, 0]


def accumulator_headroom_penalty(own: mx.array, field: mx.array) -> mx.array:
    def excess(values: mx.array) -> mx.array:
        maximum = mx.max(mx.abs(values), axis=1)
        return mx.maximum(maximum - ACCUMULATOR_HEADROOM_LIMIT, 0.0)

    return ACCUMULATOR_HEADROOM_COEFFICIENT * 0.5 * (
        mx.mean(mx.square(excess(own))) + mx.mean(mx.square(excess(field)))
    )


def v3_loss(model: V3Nnue, batch: SparseBatch | CsrBatch) -> mx.array:
    if isinstance(batch, CsrBatch):
        own, field, direct = model._accumulate_csr(batch)
    else:
        own, field, direct = model._accumulate_sparse(batch)
    predictions = model._dense(own, field, direct, batch.phase_buckets)
    residual = mx.abs(predictions - batch.targets) / 100.0
    weighted = batch.confidence_weights * mx.power(residual, 2.4)
    return mx.mean(weighted) + accumulator_headroom_penalty(own, field)
