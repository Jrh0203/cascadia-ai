"""Quantized V3 bundle export and integer reference inference."""

from __future__ import annotations

import json
import os
import struct
import uuid
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import blake3
import mlx.core as mx
import numpy as np

from .contracts import (
    ARCHITECTURE_ID,
    BASE_FEATURE_ROWS,
    DENSE_SCALE,
    FC0_NONLINEAR,
    FC0_OUTPUTS,
    FC1_INPUTS,
    FC1_OUTPUTS,
    FEATURE_SCALE,
    FEATURE_SCHEMA_ID,
    MODEL_FORMAT_VERSION,
    OUTPUT_SCALE,
    PHASE_BUCKETS,
    POOL_HALF,
    TRANSFORM_WIDTH,
)
from .model import V3Nnue

MODEL_MAGIC = b"CSV3Q01\0"


_QUANTIZED_FORWARD = mx.fast.metal_kernel(
    name="cascadia_v3_quantized_sfnnv13_forward_v1",
    input_names=[
        "own",
        "field",
        "phases",
        "direct",
        "fc0_weights",
        "fc0_biases",
        "fc1_weights",
        "fc1_biases",
        "fc2_weights",
        "fc2_biases",
    ],
    output_names=["output"],
    source=r"""
        uint row = thread_position_in_grid.x;
        int phase = phases[row];
        thread int first[32];
        for (int output_index = 0; output_index < 32; ++output_index) {
            int value = fc0_biases[phase * 32 + output_index];
            for (int input_index = 0; input_index < 1024; ++input_index) {
                int left;
                int right;
                if (input_index < 512) {
                    left = clamp(int(own[row * 1024 + input_index]), 0, 256);
                    right = clamp(int(own[row * 1024 + input_index + 512]), 0, 256);
                } else {
                    int index = input_index - 512;
                    left = clamp(int(field[row * 1024 + index]), 0, 256);
                    right = clamp(int(field[row * 1024 + index + 512]), 0, 256);
                }
                int activation = clamp((left * right) / 256, 0, 255);
                int weight_index = (phase * 1024 + input_index) * 32 + output_index;
                value += activation * int(fc0_weights[weight_index]);
            }
            first[output_index] = value >= 0 ? (value + 32) / 64 : (value - 32) / 64;
        }
        thread short second_input[62];
        for (int index = 0; index < 31; ++index) {
            int clipped = clamp(first[index], 0, 255);
            second_input[index] = short((clipped * clipped) / 255);
            second_input[31 + index] = short(clipped);
        }
        thread short second[32];
        for (int output_index = 0; output_index < 32; ++output_index) {
            int value = fc1_biases[phase * 32 + output_index];
            for (int input_index = 0; input_index < 62; ++input_index) {
                int weight_index = (phase * 62 + input_index) * 32 + output_index;
                value += int(second_input[input_index]) * int(fc1_weights[weight_index]);
            }
            int rounded = value >= 0 ? (value + 32) / 64 : (value - 32) / 64;
            second[output_index] = short(clamp(rounded, 0, 255));
        }
        int dense = fc2_biases[phase];
        for (int index = 0; index < 32; ++index) {
            dense += int(second[index]) * int(fc2_weights[phase * 32 + index]);
        }
        dense = dense >= 0 ? (dense + 512) / 1024 : (dense - 512) / 1024;
        int skip_numerator = first[31] * 16;
        int skip = skip_numerator >= 0
            ? (skip_numerator + 128) / 256
            : (skip_numerator - 128) / 256;
        output[row] = dense + skip + direct[row * 8 + phase];
    """,
)


@dataclass(frozen=True)
class QuantizedParameters:
    transformer_bias: np.ndarray
    base_transformer: np.ndarray
    opportunity_transformer: np.ndarray
    direct_potential: np.ndarray
    fc0_weights: np.ndarray
    fc0_biases: np.ndarray
    fc1_weights: np.ndarray
    fc1_biases: np.ndarray
    fc2_weights: np.ndarray
    fc2_biases: np.ndarray


def _quantize(value: mx.array, scale: int, dtype: np.dtype) -> np.ndarray:
    limits = np.iinfo(dtype)
    return np.clip(
        np.rint(np.asarray(value, dtype=np.float64) * scale), limits.min, limits.max
    ).astype(dtype)


def _coalesced_opportunity_weight(
    model: V3Nnue,
    factor_offsets: list[int] | None,
    factor_indices: list[int] | None,
) -> np.ndarray:
    direct = np.asarray(model.opportunity_embedding.weight, dtype=np.float32).copy()
    if factor_offsets is None and factor_indices is None:
        return direct
    if factor_offsets is None or factor_indices is None:
        raise ValueError("opportunity factor offsets and indices must be supplied together")
    if len(factor_offsets) != model.config.opportunity_feature_rows + 1:
        raise ValueError("opportunity factor offsets disagree with inference rows")
    if factor_offsets[0] != 0 or factor_offsets[-1] != len(factor_indices):
        raise ValueError("opportunity factor map is noncanonical")
    factors = np.asarray(model.opportunity_factor_embedding.weight, dtype=np.float32)
    for row, (start, end) in enumerate(pairwise(factor_offsets)):
        selected = factor_indices[start:end]
        if not selected or any(index < 0 or index >= factors.shape[0] for index in selected):
            raise ValueError(f"opportunity row {row} has an invalid factor map")
        direct[row] += factors[selected].sum(axis=0)
    return direct


def coalesce_training_factors_in_place(
    model: V3Nnue,
    factor_offsets: list[int],
    factor_indices: list[int],
) -> None:
    """Bake virtual training factors into serving rows without changing float semantics."""
    effective = _coalesced_opportunity_weight(model, factor_offsets, factor_indices)
    model.opportunity_embedding.weight = mx.array(effective)
    model.opportunity_factor_embedding.weight = mx.zeros_like(
        model.opportunity_factor_embedding.weight
    )


def quantize_model(
    model: V3Nnue,
    factor_offsets: list[int] | None = None,
    factor_indices: list[int] | None = None,
) -> QuantizedParameters:
    model.config.validate()
    mx.eval(model.parameters())
    return QuantizedParameters(
        transformer_bias=_quantize(model.transformer_bias, FEATURE_SCALE, np.int16),
        base_transformer=_quantize(model.base_embedding.weight, FEATURE_SCALE, np.int16),
        opportunity_transformer=_quantize(
            mx.array(_coalesced_opportunity_weight(model, factor_offsets, factor_indices)),
            FEATURE_SCALE,
            np.int8,
        ),
        direct_potential=_quantize(model.direct_potential.weight, OUTPUT_SCALE, np.int16),
        fc0_weights=np.stack(
            [_quantize(layer.weight.T, DENSE_SCALE, np.int8) for layer in model.fc0]
        ),
        fc0_biases=np.stack(
            [_quantize(layer.bias, DENSE_SCALE * FEATURE_SCALE, np.int32) for layer in model.fc0]
        ),
        fc1_weights=np.stack(
            [_quantize(layer.weight.T, DENSE_SCALE, np.int8) for layer in model.fc1]
        ),
        fc1_biases=np.stack(
            [_quantize(layer.bias, DENSE_SCALE * FEATURE_SCALE, np.int32) for layer in model.fc1]
        ),
        fc2_weights=np.stack(
            [_quantize(layer.weight[0], DENSE_SCALE, np.int8) for layer in model.fc2]
        ),
        fc2_biases=np.stack(
            [_quantize(layer.bias[0], DENSE_SCALE * FEATURE_SCALE, np.int32) for layer in model.fc2]
        ),
    )


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def export_quantized_bundle(
    model: V3Nnue,
    output: Path,
    feature_manifest: Path,
    *,
    training_origin: str,
    checkpoint_id: str,
    training_run_manifest_blake3: str | None = None,
) -> dict[str, object]:
    feature = json.loads(feature_manifest.read_text())
    if (
        feature.get("schema_id") != FEATURE_SCHEMA_ID
        or feature.get("base_feature_rows") != BASE_FEATURE_ROWS
        or feature.get("opportunity_feature_rows") != model.config.opportunity_feature_rows
    ):
        raise ValueError("feature manifest does not match the V3 model")
    output.mkdir(parents=True, exist_ok=True)
    factor_offsets = feature.get("opportunity_training_factor_offsets")
    factor_indices = feature.get("opportunity_training_factor_indices")
    if (
        feature.get("opportunity_training_factor_rows")
        != model.config.opportunity_training_factor_rows
    ):
        raise ValueError("feature manifest training-factor width differs from the model")
    parameters = quantize_model(model, factor_offsets, factor_indices)
    final_weights = output / "weights.v3q"
    temporary = output / f".weights.v3q.{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as handle:
        handle.write(MODEL_MAGIC)
        handle.write(
            struct.pack(
                "<HIIH4B3i",
                MODEL_FORMAT_VERSION,
                BASE_FEATURE_ROWS,
                model.config.opportunity_feature_rows,
                TRANSFORM_WIDTH,
                PHASE_BUCKETS,
                FC0_OUTPUTS,
                FC1_INPUTS,
                FC1_OUTPUTS,
                FEATURE_SCALE,
                DENSE_SCALE,
                OUTPUT_SCALE,
            )
        )
        for value in (
            parameters.transformer_bias,
            parameters.base_transformer,
            parameters.opportunity_transformer,
            parameters.direct_potential,
            parameters.fc0_weights,
            parameters.fc0_biases,
            parameters.fc1_weights,
            parameters.fc1_biases,
            parameters.fc2_weights,
            parameters.fc2_biases,
        ):
            handle.write(np.asarray(value).tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final_weights)
    manifest = {
        "schema_version": MODEL_FORMAT_VERSION,
        "architecture_id": ARCHITECTURE_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_blake3": feature["canonical_blake3"],
        "opportunity_catalog_blake3": feature["opportunity_catalog_blake3"],
        "base_feature_rows": BASE_FEATURE_ROWS,
        "opportunity_feature_rows": model.config.opportunity_feature_rows,
        "opportunity_training_factor_rows": model.config.opportunity_training_factor_rows,
        "opportunity_training_factor_blake3": feature[
            "opportunity_training_factor_blake3"
        ],
        "training_factors_coalesced": True,
        "transformer_width": TRANSFORM_WIDTH,
        "accumulator_storage": "int32_exact",
        "activation_input": "int16_clipped_0_feature_scale",
        "phase_buckets": PHASE_BUCKETS,
        "fc0_outputs": FC0_OUTPUTS,
        "fc1_inputs": FC1_INPUTS,
        "fc1_outputs": FC1_OUTPUTS,
        "scales": {
            "feature_transformer": FEATURE_SCALE,
            "dense": DENSE_SCALE,
            "output": OUTPUT_SCALE,
        },
        "training_origin": training_origin,
        "training_run_manifest_blake3": training_run_manifest_blake3,
        "checkpoint_id": checkpoint_id,
        "weights_file": final_weights.name,
        "weights_blake3": _checksum(final_weights),
        "serving_compatible": True,
    }
    _write_json_atomic(output / "model.json", manifest)
    return manifest


def _rounded_div(value: np.ndarray | np.int64, divisor: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    numerator = np.where(value >= 0, value + divisor // 2, value - divisor // 2)
    return np.trunc(numerator.astype(np.float64) / divisor).astype(np.int64)


def quantized_forward_from_accumulators(
    parameters: QuantizedParameters,
    own: np.ndarray,
    field: np.ndarray,
    phases: np.ndarray,
    direct: np.ndarray,
) -> np.ndarray:
    own = np.asarray(own, dtype=np.int64)
    field = np.asarray(field, dtype=np.int64)
    phases = np.asarray(phases, dtype=np.int64)
    if own.shape != field.shape or own.ndim != 2 or own.shape[1] != TRANSFORM_WIDTH:
        raise ValueError("quantized accumulator shapes are invalid")
    own_clipped = np.clip(own, 0, FEATURE_SCALE)
    field_clipped = np.clip(field, 0, FEATURE_SCALE)
    pooled = np.concatenate(
        (
            np.clip(
                own_clipped[:, :POOL_HALF] * own_clipped[:, POOL_HALF:] // FEATURE_SCALE,
                0,
                FEATURE_SCALE - 1,
            ),
            np.clip(
                field_clipped[:, :POOL_HALF] * field_clipped[:, POOL_HALF:] // FEATURE_SCALE,
                0,
                FEATURE_SCALE - 1,
            ),
        ),
        axis=1,
    )
    outputs = np.empty((own.shape[0],), dtype=np.int32)
    for row, phase in enumerate(phases):
        first = _rounded_div(
            pooled[row] @ parameters.fc0_weights[phase].astype(np.int64)
            + parameters.fc0_biases[phase],
            DENSE_SCALE,
        )
        nonlinear = np.clip(first[:FC0_NONLINEAR], 0, FEATURE_SCALE - 1)
        second_input = np.concatenate((nonlinear * nonlinear // (FEATURE_SCALE - 1), nonlinear))
        second = np.clip(
            _rounded_div(
                second_input @ parameters.fc1_weights[phase].astype(np.int64)
                + parameters.fc1_biases[phase],
                DENSE_SCALE,
            ),
            0,
            FEATURE_SCALE - 1,
        )
        value = _rounded_div(
            second @ parameters.fc2_weights[phase].astype(np.int64) + parameters.fc2_biases[phase],
            DENSE_SCALE * FEATURE_SCALE // OUTPUT_SCALE,
        )
        skip = _rounded_div(first[FC0_OUTPUTS - 1] * OUTPUT_SCALE, FEATURE_SCALE)
        outputs[row] = np.int32(value + skip + direct[row, phase])
    return outputs


def quantized_forward_mlx_from_accumulators(
    parameters: QuantizedParameters,
    own: np.ndarray,
    field: np.ndarray,
    phases: np.ndarray,
    direct: np.ndarray,
) -> np.ndarray:
    """Run the exported integer graph as a native MLX Metal kernel."""
    own = np.ascontiguousarray(own, dtype=np.int16)
    field = np.ascontiguousarray(field, dtype=np.int16)
    phases = np.ascontiguousarray(phases, dtype=np.int32)
    direct = np.ascontiguousarray(direct, dtype=np.int32)
    if own.shape != field.shape or own.ndim != 2 or own.shape[1] != TRANSFORM_WIDTH:
        raise ValueError("quantized MLX accumulator shapes are invalid")
    rows = own.shape[0]
    if phases.shape != (rows,) or direct.shape != (rows, PHASE_BUCKETS):
        raise ValueError("quantized MLX phase or direct-potential shapes are invalid")
    output = _QUANTIZED_FORWARD(
        inputs=[
            mx.array(own),
            mx.array(field),
            mx.array(phases),
            mx.array(direct),
            mx.array(np.ascontiguousarray(parameters.fc0_weights)),
            mx.array(np.ascontiguousarray(parameters.fc0_biases)),
            mx.array(np.ascontiguousarray(parameters.fc1_weights)),
            mx.array(np.ascontiguousarray(parameters.fc1_biases)),
            mx.array(np.ascontiguousarray(parameters.fc2_weights)),
            mx.array(np.ascontiguousarray(parameters.fc2_biases)),
        ],
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
        grid=(rows, 1, 1),
        threadgroup=(min(rows, 256), 1, 1),
    )[0]
    mx.eval(output)
    return np.asarray(output, dtype=np.int32)
