"""Strict conversion and sparse MLX inference for the qualified legacy NNUE."""

from __future__ import annotations

import json
import os
import shutil
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

LEGACY_NNUE_MAGIC = b"NNUE"
LEGACY_NNUE_VERSION = 1
LEGACY_NNUE_FEATURES = 11_231
LEGACY_NNUE_HIDDEN1 = 512
LEGACY_NNUE_HIDDEN2 = 64
LEGACY_NNUE_SOURCE_BYTES = 23_134_992
LEGACY_NNUE_SOURCE_BLAKE3 = "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"
LEGACY_NNUE_ARCHITECTURE = "legacy-sparse-nnue-v4opp-mlx-v1"
LEGACY_NNUE_ARTIFACT_SCHEMA = 1
LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA = 2
CORRECTED_NNUE_ARTIFACT_SCHEMA = 3

CORRECTED_NNUE_MAGIC = b"NNUC"
CORRECTED_NNUE_CONTAINER_VERSION = 1
CORRECTED_NNUE_SCHEMA_ID = "legacy-mid-v4-fixed-v1"
CORRECTED_NNUE_SCHEMA_TAG = b"MIDTAIL-CORR-V1\0"
CORRECTED_NNUE_ARCHITECTURE = "legacy-sparse-nnue-midtail-corr-v1-mlx-v1"
CORRECTED_NNUE_HEADER = struct.Struct("<4sII16sIII")
NNUE_SUPPORTED_HEAD_VERSIONS = frozenset({1, 2, 3, 4})
NNUE_SPLIT_HEADS = 11

HISTORICAL_NNUE_SUPPORTED_FEATURES = frozenset({5_197, 5_566, 7_670, 10_561, 10_862, 11_231})
HISTORICAL_NNUE_SCHEMA_IDS = {
    5_197: "legacy-nnue-v1-5197",
    5_566: "legacy-nnue-v1-v4opp-5566",
    7_670: "legacy-nnue-v1-7670",
    10_561: "legacy-nnue-v2-10561",
    10_862: "legacy-mid-10862",
    11_231: "legacy-mid-v4opp-11231",
}

CORRECTED_NNUE_BASE_START = 0
CORRECTED_NNUE_BASE_END = 10_561
CORRECTED_NNUE_OPPONENT_START = 10_561
CORRECTED_NNUE_OPPONENT_END = 10_930
CORRECTED_NNUE_TERRAIN_START = 10_930
CORRECTED_NNUE_TERRAIN_END = 11_080
CORRECTED_NNUE_WILDLIFE_START = 11_080
CORRECTED_NNUE_WILDLIFE_END = 11_230
CORRECTED_NNUE_OVERFLOW_START = 11_230
CORRECTED_NNUE_OVERFLOW_END = 11_231


@dataclass(frozen=True)
class NnueFeatureBlock:
    """A named half-open first-layer row range."""

    name: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


CORRECTED_NNUE_ROW_LAYOUT = (
    NnueFeatureBlock("historical_v2_base", CORRECTED_NNUE_BASE_START, CORRECTED_NNUE_BASE_END),
    NnueFeatureBlock(
        "opponent_detail",
        CORRECTED_NNUE_OPPONENT_START,
        CORRECTED_NNUE_OPPONENT_END,
    ),
    NnueFeatureBlock(
        "extended_tile_terrain_counts",
        CORRECTED_NNUE_TERRAIN_START,
        CORRECTED_NNUE_TERRAIN_END,
    ),
    NnueFeatureBlock(
        "extended_tile_wildlife_capacity_counts",
        CORRECTED_NNUE_WILDLIFE_START,
        CORRECTED_NNUE_WILDLIFE_END,
    ),
    NnueFeatureBlock(
        "overflow_used",
        CORRECTED_NNUE_OVERFLOW_START,
        CORRECTED_NNUE_OVERFLOW_END,
    ),
)

_RUST_EXACT_H1 = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_h1_float4_v2",
    input_names=["offsets", "indices", "w1", "b1"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        uint row = elem / 128;
        uint lane = elem - row * 128;
        uint hidden = lane * 4;
        uint start = uint(offsets[row]);
        uint end = uint(offsets[row + 1]);
        float4 value = *((device const float4*)(b1 + hidden));
        for (uint position = start; position < end; ++position) {
            uint feature = uint(indices[position]);
            value = value + *((device const float4*)(w1 + feature * 512 + hidden));
        }
        *((device float4*)(out + row * 512 + hidden)) = max(value, float4(0.0f));
    """,
)

_RUST_EXACT_H2 = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_h2_float8_v3",
    input_names=["h1", "w2", "b2"],
    output_names=["out"],
    header="#pragma clang fp contract(off)\n",
    source=r"""
        uint elem = thread_position_in_grid.x;
        uint row = elem / 8;
        uint lane = elem - row * 8;
        uint hidden = lane * 8;
        float4 value0 = *((device const float4*)(b2 + hidden));
        float4 value1 = *((device const float4*)(b2 + hidden + 4));
        for (uint input = 0; input < 512; ++input) {
            float activation = h1[row * 512 + input];
            if (activation > 0.0f) {
                uint weight_offset = input * 64 + hidden;
                float4 product0 = activation *
                    *((device const float4*)(w2 + weight_offset));
                float4 product1 = activation *
                    *((device const float4*)(w2 + weight_offset + 4));
                value0 = value0 + product0;
                value1 = value1 + product1;
            }
        }
        uint output_offset = row * 64 + hidden;
        *((device float4*)(out + output_offset)) = max(value0, float4(0.0f));
        *((device float4*)(out + output_offset + 4)) = max(value1, float4(0.0f));
    """,
)

_RUST_EXACT_OUTPUT = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_output_v1",
    input_names=["h2", "w3", "b3"],
    output_names=["out"],
    header="#pragma clang fp contract(off)\n",
    source=r"""
        uint row = thread_position_in_grid.x;
        float value = b3[0];
        for (uint hidden = 0; hidden < 64; ++hidden) {
            float product = h2[row * 64 + hidden] * w3[hidden];
            value = value + product;
        }
        out[row] = value;
    """,
)

_RUST_EXACT_SPLIT2_OUTPUT = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_split2_output_v1",
    input_names=["h2", "w_wildlife", "b_wildlife", "w_habitat", "b_habitat"],
    output_names=["out"],
    header="#pragma clang fp contract(off)\n",
    source=r"""
        uint row = thread_position_in_grid.x;
        float wildlife = b_wildlife[0];
        float habitat = b_habitat[0];
        for (uint hidden = 0; hidden < 64; ++hidden) {
            wildlife = wildlife + h2[row * 64 + hidden] * w_wildlife[hidden];
            habitat = habitat + h2[row * 64 + hidden] * w_habitat[hidden];
        }
        out[row] = wildlife + habitat;
    """,
)

_RUST_EXACT_SPLIT11_OUTPUT = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_split11_output_v1",
    input_names=["h2", "w_heads", "b_heads"],
    output_names=["out"],
    header="#pragma clang fp contract(off)\n",
    source=r"""
        uint row = thread_position_in_grid.x;
        float sum = 0.0f;
        for (uint head_index = 0; head_index < 11; ++head_index) {
            float head = b_heads[head_index];
            for (uint hidden = 0; hidden < 64; ++hidden) {
                head = head +
                    h2[row * 64 + hidden] * w_heads[head_index * 64 + hidden];
            }
            sum = sum + head;
        }
        out[row] = sum;
    """,
)


class LegacyNnueError(ValueError):
    """Raised when a source model, sparse batch, or artifact is invalid."""


@dataclass(frozen=True)
class LegacyNnueWeights:
    version: int
    feature_count: int
    hidden1: int
    hidden2: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w3_policy: np.ndarray
    b3_policy: np.ndarray
    container_magic: bytes = LEGACY_NNUE_MAGIC
    container_version: int | None = None
    schema_id: str | None = None
    has_policy: bool = True
    has_split_value_heads: bool = False
    w3_wildlife: np.ndarray | None = None
    b3_wildlife: np.ndarray | None = None
    w3_habitat: np.ndarray | None = None
    b3_habitat: np.ndarray | None = None
    has_split11_heads: bool = False
    w3_heads: np.ndarray | None = None
    b3_heads: np.ndarray | None = None
    has_heteroscedastic: bool = False
    w3_var: np.ndarray | None = None
    b3_var: np.ndarray | None = None

    def tensors(self) -> dict[str, mx.array]:
        tensors = {
            "w1": mx.array(self.w1),
            "b1": mx.array(self.b1),
            "w2": mx.array(self.w2),
            "b2": mx.array(self.b2),
            "w3": mx.array(self.w3),
            "b3": mx.array(self.b3),
            "w3_policy": mx.array(self.w3_policy),
            "b3_policy": mx.array(self.b3_policy),
        }
        if self.has_split_value_heads:
            tensors.update(
                {
                    "w3_wildlife": mx.array(_required_tensor(self.w3_wildlife, "w3_wildlife")),
                    "b3_wildlife": mx.array(_required_tensor(self.b3_wildlife, "b3_wildlife")),
                    "w3_habitat": mx.array(_required_tensor(self.w3_habitat, "w3_habitat")),
                    "b3_habitat": mx.array(_required_tensor(self.b3_habitat, "b3_habitat")),
                }
            )
        if self.has_split11_heads:
            tensors.update(
                {
                    "w3_heads": mx.array(_required_tensor(self.w3_heads, "w3_heads")),
                    "b3_heads": mx.array(_required_tensor(self.b3_heads, "b3_heads")),
                }
            )
        if self.has_heteroscedastic:
            tensors.update(
                {
                    "w3_var": mx.array(_required_tensor(self.w3_var, "w3_var")),
                    "b3_var": mx.array(_required_tensor(self.b3_var, "b3_var")),
                }
            )
        return tensors

    @property
    def is_corrected(self) -> bool:
        return (
            self.container_magic == CORRECTED_NNUE_MAGIC
            and self.container_version == CORRECTED_NNUE_CONTAINER_VERSION
            and self.schema_id == CORRECTED_NNUE_SCHEMA_ID
        )


def _required_tensor(tensor: np.ndarray | None, name: str) -> np.ndarray:
    if tensor is None:
        raise LegacyNnueError(f"NNUE tensor {name} is required by the head version")
    return tensor


def _head_trailing_float_count(version: int, hidden2: int) -> int:
    if version not in NNUE_SUPPORTED_HEAD_VERSIONS:
        raise LegacyNnueError(f"unsupported NNUE head version {version}")
    count = hidden2 + 1
    if version >= 2:
        count += 2 * (hidden2 + 1)
    if version >= 3:
        count += NNUE_SPLIT_HEADS * hidden2 + NNUE_SPLIT_HEADS
    if version >= 4:
        count += hidden2 + 1
    return count


def _fixed_after_first_layer_float_count(hidden1: int, hidden2: int) -> int:
    return hidden1 + hidden1 * hidden2 + hidden2 + hidden2 + 1


def _historical_read_plan(
    byte_count: int,
    version: int,
    hidden1: int,
    hidden2: int,
    expected_features: int | None,
) -> tuple[int, bool]:
    if version not in NNUE_SUPPORTED_HEAD_VERSIONS:
        raise LegacyNnueError(f"unsupported legacy NNUE head version {version}")
    fixed_bytes = _fixed_after_first_layer_float_count(hidden1, hidden2) * 4
    full_trailing_bytes = _head_trailing_float_count(version, hidden2) * 4
    row_bytes = hidden1 * 4
    candidates: list[tuple[int, bool]] = []
    for trailing_bytes, has_policy in ((full_trailing_bytes, True), (0, False)):
        first_layer_bytes = byte_count - 8 - fixed_bytes - trailing_bytes
        if first_layer_bytes > 0 and first_layer_bytes % row_bytes == 0:
            candidates.append((first_layer_bytes // row_bytes, has_policy))
    if expected_features is not None:
        candidates = [candidate for candidate in candidates if candidate[0] == expected_features]
    if len(candidates) != 1:
        raise LegacyNnueError("legacy NNUE file size does not match a supported payload")
    feature_count, has_policy = candidates[0]
    if expected_features is None and feature_count not in HISTORICAL_NNUE_SUPPORTED_FEATURES:
        raise LegacyNnueError(f"legacy NNUE has unsupported first-layer width {feature_count}")
    return feature_count, has_policy


def _parse_weight_payload(
    raw: bytes,
    *,
    payload_offset: int,
    version: int,
    feature_count: int,
    hidden1: int,
    hidden2: int,
    has_policy: bool,
    container_magic: bytes,
    container_version: int | None,
    schema_id: str | None,
) -> LegacyNnueWeights:
    if (len(raw) - payload_offset) % 4:
        raise LegacyNnueError("NNUE payload byte count is not float32 aligned")
    values = np.frombuffer(raw, dtype="<f4", offset=payload_offset)
    offset = 0

    def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        end = offset + count
        if end > len(values):
            raise LegacyNnueError("NNUE source ended inside a tensor")
        tensor = np.array(values[offset:end], dtype=np.float32, copy=True).reshape(shape)
        offset = end
        return tensor

    w1 = take(feature_count * hidden1, (feature_count, hidden1))
    b1 = take(hidden1, (hidden1,))
    w2 = take(hidden1 * hidden2, (hidden1, hidden2))
    b2 = take(hidden2, (hidden2,))
    w3 = take(hidden2, (hidden2,))
    b3 = take(1, (1,))
    if has_policy:
        w3_policy = take(hidden2, (hidden2,))
        b3_policy = take(1, (1,))
    else:
        w3_policy = np.zeros(hidden2, dtype=np.float32)
        b3_policy = np.zeros(1, dtype=np.float32)

    has_split = has_policy and version >= 2
    if has_split:
        w3_wildlife = take(hidden2, (hidden2,))
        b3_wildlife = take(1, (1,))
        w3_habitat = take(hidden2, (hidden2,))
        b3_habitat = take(1, (1,))
    else:
        w3_wildlife = b3_wildlife = w3_habitat = b3_habitat = None

    has_split11 = has_policy and version >= 3
    if has_split11:
        w3_heads = take(NNUE_SPLIT_HEADS * hidden2, (NNUE_SPLIT_HEADS, hidden2))
        b3_heads = take(NNUE_SPLIT_HEADS, (NNUE_SPLIT_HEADS,))
    else:
        w3_heads = b3_heads = None

    has_heteroscedastic = has_policy and version >= 4
    if has_heteroscedastic:
        w3_var = take(hidden2, (hidden2,))
        b3_var = take(1, (1,))
    else:
        w3_var = b3_var = None

    if offset != len(values):
        raise LegacyNnueError(
            f"NNUE source has {len(values) - offset} unparsed trailing float32 values"
        )
    tensors = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
        "w3_policy": w3_policy,
        "b3_policy": b3_policy,
        "w3_wildlife": w3_wildlife,
        "b3_wildlife": b3_wildlife,
        "w3_habitat": w3_habitat,
        "b3_habitat": b3_habitat,
        "w3_heads": w3_heads,
        "b3_heads": b3_heads,
        "w3_var": w3_var,
        "b3_var": b3_var,
    }
    for name, tensor in tensors.items():
        if tensor is not None and not np.all(np.isfinite(tensor)):
            raise LegacyNnueError(f"NNUE tensor {name} contains non-finite values")
    return LegacyNnueWeights(
        version=version,
        feature_count=feature_count,
        hidden1=hidden1,
        hidden2=hidden2,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        w3_policy=w3_policy,
        b3_policy=b3_policy,
        container_magic=container_magic,
        container_version=container_version,
        schema_id=schema_id,
        has_policy=has_policy,
        has_split_value_heads=has_split,
        w3_wildlife=w3_wildlife,
        b3_wildlife=b3_wildlife,
        w3_habitat=w3_habitat,
        b3_habitat=b3_habitat,
        has_split11_heads=has_split11,
        w3_heads=w3_heads,
        b3_heads=b3_heads,
        has_heteroscedastic=has_heteroscedastic,
        w3_var=w3_var,
        b3_var=b3_var,
    )


def checksum_file(path: str | Path) -> str:
    path = Path(path)
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_legacy_nnue(
    path: str | Path,
    *,
    hidden1: int = LEGACY_NNUE_HIDDEN1,
    hidden2: int = LEGACY_NNUE_HIDDEN2,
    expected_features: int | None = LEGACY_NNUE_FEATURES,
) -> LegacyNnueWeights:
    """Parse either frozen historical `NNUE` or corrected `NNUC` weights.

    Historical rows retain their source semantics. Use
    :func:`parse_corrected_nnue` when historical inputs should be migrated to
    `legacy-mid-v4-fixed-v1`.
    """

    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 4:
        raise LegacyNnueError("NNUE source is too short for a container magic")
    magic = raw[:4]
    if magic == LEGACY_NNUE_MAGIC:
        if len(raw) < 8:
            raise LegacyNnueError("legacy NNUE source is too short for its header")
        version = struct.unpack_from("<I", raw, 4)[0]
        feature_count, has_policy = _historical_read_plan(
            len(raw),
            version,
            hidden1,
            hidden2,
            expected_features,
        )
        return _parse_weight_payload(
            raw,
            payload_offset=8,
            version=version,
            feature_count=feature_count,
            hidden1=hidden1,
            hidden2=hidden2,
            has_policy=has_policy,
            container_magic=LEGACY_NNUE_MAGIC,
            container_version=None,
            schema_id=HISTORICAL_NNUE_SCHEMA_IDS.get(feature_count),
        )

    if magic != CORRECTED_NNUE_MAGIC:
        raise LegacyNnueError(f"NNUE source has unknown magic {magic!r}")
    if len(raw) < CORRECTED_NNUE_HEADER.size:
        raise LegacyNnueError("corrected NNUE source is too short for its 40-byte header")
    (
        _,
        container_version,
        version,
        schema_tag,
        feature_count,
        stored_hidden1,
        stored_hidden2,
    ) = CORRECTED_NNUE_HEADER.unpack_from(raw)
    if container_version != CORRECTED_NNUE_CONTAINER_VERSION:
        raise LegacyNnueError(f"unsupported corrected NNUE container version {container_version}")
    if version not in NNUE_SUPPORTED_HEAD_VERSIONS:
        raise LegacyNnueError(f"unsupported corrected NNUE head version {version}")
    if schema_tag != CORRECTED_NNUE_SCHEMA_TAG:
        raise LegacyNnueError("corrected NNUE schema tag does not match legacy-mid-v4-fixed-v1")
    if expected_features is not None and feature_count != expected_features:
        raise LegacyNnueError(
            f"corrected NNUE feature count {feature_count} != expected {expected_features}"
        )
    if stored_hidden1 != hidden1 or stored_hidden2 != hidden2:
        raise LegacyNnueError(
            "corrected NNUE architecture mismatch: "
            f"file={feature_count}x{stored_hidden1}x{stored_hidden2}, "
            f"expected={expected_features or feature_count}x{hidden1}x{hidden2}"
        )
    expected_size = (
        CORRECTED_NNUE_HEADER.size
        + feature_count * stored_hidden1 * 4
        + _fixed_after_first_layer_float_count(stored_hidden1, stored_hidden2) * 4
        + _head_trailing_float_count(version, stored_hidden2) * 4
    )
    if len(raw) != expected_size:
        raise LegacyNnueError(
            f"corrected NNUE file size mismatch: expected {expected_size}, found {len(raw)}"
        )
    return _parse_weight_payload(
        raw,
        payload_offset=CORRECTED_NNUE_HEADER.size,
        version=version,
        feature_count=feature_count,
        hidden1=stored_hidden1,
        hidden2=stored_hidden2,
        has_policy=True,
        container_magic=CORRECTED_NNUE_MAGIC,
        container_version=container_version,
        schema_id=CORRECTED_NNUE_SCHEMA_ID,
    )


def corrected_feature_for_historical(
    source_feature: int,
    *,
    source_feature_count: int,
) -> int | None:
    """Return a corrected row for a recognized historical row, or `None` if discarded."""

    if source_feature_count not in HISTORICAL_NNUE_SUPPORTED_FEATURES:
        raise LegacyNnueError(
            f"historical NNUE has unsupported first-layer width {source_feature_count}"
        )
    if source_feature < 0 or source_feature >= source_feature_count:
        raise LegacyNnueError(
            f"historical feature {source_feature} is outside 0..{source_feature_count}"
        )
    if source_feature_count in {5_197, 7_670, 10_561}:
        return source_feature
    if source_feature_count == 10_862:
        return source_feature if source_feature < CORRECTED_NNUE_BASE_END else None
    if source_feature_count == 5_566:
        if source_feature < 5_197:
            return source_feature
        return CORRECTED_NNUE_OPPONENT_START + source_feature - 5_197
    if source_feature < CORRECTED_NNUE_BASE_END:
        return source_feature
    if 10_862 <= source_feature < LEGACY_NNUE_FEATURES:
        return CORRECTED_NNUE_OPPONENT_START + source_feature - 10_862
    return None


def remap_historical_features_to_corrected(
    features: list[int],
    *,
    source_feature_count: int = LEGACY_NNUE_FEATURES,
    reject_discarded: bool = True,
) -> list[int]:
    """Map sparse historical features while preserving order and multiplicity."""

    corrected: list[int] = []
    for feature in features:
        destination = corrected_feature_for_historical(
            feature,
            source_feature_count=source_feature_count,
        )
        if destination is None:
            if reject_discarded:
                raise LegacyNnueError(
                    f"historical feature {feature} is in a discarded schema-defect range"
                )
            continue
        corrected.append(destination)
    return corrected


def migrate_historical_nnue_to_corrected(weights: LegacyNnueWeights) -> LegacyNnueWeights:
    """Apply the Rust `FixedLegacyFirstLayerLayout` migration exactly in memory."""

    if weights.container_magic != LEGACY_NNUE_MAGIC:
        raise LegacyNnueError("corrected migration requires a historical NNUE container")
    if weights.feature_count not in HISTORICAL_NNUE_SUPPORTED_FEATURES:
        raise LegacyNnueError(
            f"historical NNUE has unsupported first-layer width {weights.feature_count}"
        )
    migrated_w1 = np.zeros(
        (LEGACY_NNUE_FEATURES, weights.hidden1),
        dtype=np.float32,
    )
    for source_feature in range(weights.feature_count):
        destination = corrected_feature_for_historical(
            source_feature,
            source_feature_count=weights.feature_count,
        )
        if destination is not None:
            migrated_w1[destination] = weights.w1[source_feature]

    def copy_optional(tensor: np.ndarray | None) -> np.ndarray | None:
        return None if tensor is None else tensor.copy()

    return LegacyNnueWeights(
        version=weights.version,
        feature_count=LEGACY_NNUE_FEATURES,
        hidden1=weights.hidden1,
        hidden2=weights.hidden2,
        w1=migrated_w1,
        b1=weights.b1.copy(),
        w2=weights.w2.copy(),
        b2=weights.b2.copy(),
        w3=weights.w3.copy(),
        b3=weights.b3.copy(),
        w3_policy=weights.w3_policy.copy(),
        b3_policy=weights.b3_policy.copy(),
        container_magic=CORRECTED_NNUE_MAGIC,
        container_version=CORRECTED_NNUE_CONTAINER_VERSION,
        schema_id=CORRECTED_NNUE_SCHEMA_ID,
        has_policy=weights.has_policy,
        has_split_value_heads=weights.has_split_value_heads,
        w3_wildlife=copy_optional(weights.w3_wildlife),
        b3_wildlife=copy_optional(weights.b3_wildlife),
        w3_habitat=copy_optional(weights.w3_habitat),
        b3_habitat=copy_optional(weights.b3_habitat),
        has_split11_heads=weights.has_split11_heads,
        w3_heads=copy_optional(weights.w3_heads),
        b3_heads=copy_optional(weights.b3_heads),
        has_heteroscedastic=weights.has_heteroscedastic,
        w3_var=copy_optional(weights.w3_var),
        b3_var=copy_optional(weights.b3_var),
    )


def parse_corrected_nnue(
    path: str | Path,
    *,
    hidden1: int = LEGACY_NNUE_HIDDEN1,
    hidden2: int = LEGACY_NNUE_HIDDEN2,
) -> LegacyNnueWeights:
    """Load a corrected container or migrate a recognized historical layout."""

    path = Path(path)
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as error:
        raise LegacyNnueError(f"cannot read NNUE source: {error}") from error
    if magic == CORRECTED_NNUE_MAGIC:
        weights = parse_legacy_nnue(path, hidden1=hidden1, hidden2=hidden2)
        if not weights.is_corrected:
            raise LegacyNnueError("corrected NNUE source did not identify the corrected schema")
        return weights
    if magic != LEGACY_NNUE_MAGIC:
        raise LegacyNnueError(f"NNUE source has unknown magic {magic!r}")
    historical = parse_legacy_nnue(
        path,
        hidden1=hidden1,
        hidden2=hidden2,
        expected_features=None,
    )
    return migrate_historical_nnue_to_corrected(historical)


def convert_legacy_nnue(source: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    source_bytes = source.stat().st_size
    source_blake3 = checksum_file(source)
    if source_bytes != LEGACY_NNUE_SOURCE_BYTES:
        raise LegacyNnueError(
            f"legacy NNUE source bytes {source_bytes} != expected {LEGACY_NNUE_SOURCE_BYTES}"
        )
    if source_blake3 != LEGACY_NNUE_SOURCE_BLAKE3:
        raise LegacyNnueError("legacy NNUE source checksum does not match the qualified model")
    weights = parse_legacy_nnue(source)

    if output.exists():
        manifest = load_legacy_nnue_manifest(output)
        if (
            manifest["schema_version"] != LEGACY_NNUE_ARTIFACT_SCHEMA
            or manifest["source"]["blake3"] != source_blake3
            or manifest["source"]["bytes"] != source_bytes
        ):
            raise LegacyNnueError("existing MLX NNUE artifact belongs to another source")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        model_path = temporary / "model.safetensors"
        mx.save_safetensors(str(model_path), weights.tensors())
        manifest = {
            "schema_version": LEGACY_NNUE_ARTIFACT_SCHEMA,
            "architecture": LEGACY_NNUE_ARCHITECTURE,
            "source": {
                "path": str(source),
                "bytes": source_bytes,
                "blake3": source_blake3,
                "version": weights.version,
            },
            "dimensions": {
                "features": weights.feature_count,
                "hidden1": weights.hidden1,
                "hidden2": weights.hidden2,
                "outputs": 1,
            },
            "files": {
                "model.safetensors": {
                    "bytes": model_path.stat().st_size,
                    "blake3": checksum_file(model_path),
                }
            },
        }
        manifest_path = temporary / "model.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_legacy_nnue_manifest(output)


def convert_corrected_nnue(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Convert a schema-tagged corrected checkpoint into an integrity-checked MLX artifact."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    weights = parse_legacy_nnue(source)
    if not weights.is_corrected:
        raise LegacyNnueError("corrected MLX conversion requires an NNUC corrected checkpoint")
    source_bytes = source.stat().st_size
    source_blake3 = checksum_file(source)

    if output.exists():
        manifest = load_legacy_nnue_manifest(output)
        if (
            manifest["schema_version"] != CORRECTED_NNUE_ARTIFACT_SCHEMA
            or manifest["source"]["blake3"] != source_blake3
            or manifest["source"]["bytes"] != source_bytes
        ):
            raise LegacyNnueError("existing corrected MLX artifact belongs to another source")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        model_path = temporary / "model.safetensors"
        mx.save_safetensors(str(model_path), weights.tensors())
        manifest = {
            "schema_version": CORRECTED_NNUE_ARTIFACT_SCHEMA,
            "architecture": CORRECTED_NNUE_ARCHITECTURE,
            "source": {
                "bytes": source_bytes,
                "blake3": source_blake3,
                "magic": CORRECTED_NNUE_MAGIC.decode("ascii"),
                "container_version": weights.container_version,
                "head_version": weights.version,
                "schema_id": weights.schema_id,
                "schema_tag_hex": CORRECTED_NNUE_SCHEMA_TAG.hex(),
            },
            "dimensions": {
                "features": weights.feature_count,
                "hidden1": weights.hidden1,
                "hidden2": weights.hidden2,
                "outputs": 1,
            },
            "row_layout": [
                {
                    "name": block.name,
                    "start": block.start,
                    "end": block.end,
                    "width": block.width,
                }
                for block in CORRECTED_NNUE_ROW_LAYOUT
            ],
            "files": {
                "model.safetensors": {
                    "bytes": model_path.stat().st_size,
                    "blake3": checksum_file(model_path),
                }
            },
        }
        manifest_path = temporary / "model.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_legacy_nnue_manifest(output)


def load_legacy_nnue_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    try:
        manifest = json.loads((root / "model.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyNnueError(f"cannot read MLX NNUE manifest: {error}") from error
    dimensions = {
        "features": LEGACY_NNUE_FEATURES,
        "hidden1": LEGACY_NNUE_HIDDEN1,
        "hidden2": LEGACY_NNUE_HIDDEN2,
        "outputs": 1,
    }
    schema_version = manifest.get("schema_version")
    if manifest.get("dimensions") != dimensions:
        raise LegacyNnueError("MLX NNUE manifest contract is invalid")
    source = manifest.get("source", {})
    if schema_version in (LEGACY_NNUE_ARTIFACT_SCHEMA, LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA):
        if manifest.get("architecture") != LEGACY_NNUE_ARCHITECTURE:
            raise LegacyNnueError("MLX NNUE manifest contract is invalid")
        if (
            source.get("bytes") != LEGACY_NNUE_SOURCE_BYTES
            or source.get("blake3") != LEGACY_NNUE_SOURCE_BLAKE3
            or source.get("version") != LEGACY_NNUE_VERSION
        ):
            raise LegacyNnueError("MLX NNUE manifest source identity is invalid")
    elif schema_version == CORRECTED_NNUE_ARTIFACT_SCHEMA:
        head_version = source.get("head_version")
        expected_source_bytes = (
            CORRECTED_NNUE_HEADER.size
            + LEGACY_NNUE_FEATURES * LEGACY_NNUE_HIDDEN1 * 4
            + _fixed_after_first_layer_float_count(
                LEGACY_NNUE_HIDDEN1,
                LEGACY_NNUE_HIDDEN2,
            )
            * 4
            + _head_trailing_float_count(head_version, LEGACY_NNUE_HIDDEN2) * 4
            if head_version in NNUE_SUPPORTED_HEAD_VERSIONS
            else None
        )
        expected_layout = [
            {
                "name": block.name,
                "start": block.start,
                "end": block.end,
                "width": block.width,
            }
            for block in CORRECTED_NNUE_ROW_LAYOUT
        ]
        if (
            manifest.get("architecture") != CORRECTED_NNUE_ARCHITECTURE
            or source.get("magic") != CORRECTED_NNUE_MAGIC.decode("ascii")
            or source.get("container_version") != CORRECTED_NNUE_CONTAINER_VERSION
            or head_version not in NNUE_SUPPORTED_HEAD_VERSIONS
            or source.get("schema_id") != CORRECTED_NNUE_SCHEMA_ID
            or source.get("schema_tag_hex") != CORRECTED_NNUE_SCHEMA_TAG.hex()
            or source.get("bytes") != expected_source_bytes
            or not isinstance(source.get("blake3"), str)
            or len(source["blake3"]) != 64
            or manifest.get("row_layout") != expected_layout
        ):
            raise LegacyNnueError("corrected MLX NNUE manifest source identity is invalid")
    else:
        raise LegacyNnueError("MLX NNUE manifest contract is invalid")
    if schema_version == LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA:
        _validate_derivation(manifest.get("derivation"))
    model_path = root / "model.safetensors"
    expected = manifest.get("files", {}).get("model.safetensors", {})
    if (
        not model_path.is_file()
        or model_path.stat().st_size != expected.get("bytes")
        or checksum_file(model_path) != expected.get("blake3")
    ):
        raise LegacyNnueError("MLX NNUE safetensors integrity check failed")
    return manifest


def package_derived_legacy_nnue(
    parent_root: str | Path,
    output: str | Path,
    value_tensors: dict[str, mx.array],
    derivation: dict[str, Any],
) -> dict[str, Any]:
    """Atomically package a fine-tuned value path with the parent's policy tensors."""
    parent_root = Path(parent_root).resolve()
    output = Path(output).resolve()
    parent_manifest = load_legacy_nnue_manifest(parent_root)
    if parent_manifest["schema_version"] != LEGACY_NNUE_ARTIFACT_SCHEMA:
        raise LegacyNnueError("rollout-return derivation requires the qualified base artifact")
    _validate_derivation(derivation)
    parent_tensors = dict(mx.load(str(parent_root / "model.safetensors")))
    expected_value_names = {"w1", "b1", "w2", "b2", "w3", "b3"}
    if set(value_tensors) != expected_value_names:
        raise LegacyNnueError("derived artifact value tensor names are invalid")
    tensors = dict(parent_tensors)
    tensors.update({name: mx.array(tensor) for name, tensor in value_tensors.items()})
    LegacySparseNnue(tensors)
    for name, tensor in tensors.items():
        mx.eval(tensor)
        if not np.all(np.isfinite(np.asarray(tensor))):
            raise LegacyNnueError(f"derived MLX NNUE tensor {name} contains non-finite values")
    if not np.array_equal(
        np.asarray(tensors["w3_policy"]),
        np.asarray(parent_tensors["w3_policy"]),
    ) or not np.array_equal(
        np.asarray(tensors["b3_policy"]),
        np.asarray(parent_tensors["b3_policy"]),
    ):
        raise LegacyNnueError("derived MLX NNUE policy tensors changed")

    if output.exists():
        manifest = load_legacy_nnue_manifest(output)
        if (
            manifest.get("schema_version") != LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA
            or manifest.get("derivation") != derivation
        ):
            raise LegacyNnueError("existing derived artifact has another identity")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        model_path = temporary / "model.safetensors"
        mx.save_safetensors(str(model_path), tensors)
        manifest = {
            "schema_version": LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA,
            "architecture": LEGACY_NNUE_ARCHITECTURE,
            "source": parent_manifest["source"],
            "dimensions": parent_manifest["dimensions"],
            "parent": {
                "manifest_blake3": checksum_file(parent_root / "model.json"),
                "model_blake3": checksum_file(parent_root / "model.safetensors"),
            },
            "derivation": derivation,
            "files": {
                "model.safetensors": {
                    "bytes": model_path.stat().st_size,
                    "blake3": checksum_file(model_path),
                }
            },
        }
        manifest_path = temporary / "model.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_legacy_nnue_manifest(output)


def _validate_derivation(value: object) -> None:
    if not isinstance(value, dict):
        raise LegacyNnueError("derived MLX NNUE derivation metadata is missing")
    digest_fields = (
        "parent_manifest_blake3",
        "parent_model_blake3",
        "train_dataset_manifest_blake3",
        "validation_dataset_manifest_blake3",
        "run_manifest_blake3",
        "checkpoint_manifest_blake3",
    )
    if (
        value.get("kind")
        not in {
            "mlx-rollout-return-finetune-v1",
            "mlx-joint-return-ranking-finetune-v1",
        }
        or not isinstance(value.get("selected_checkpoint"), str)
        or not value["selected_checkpoint"]
        or any(
            not isinstance(value.get(field), str) or len(value[field]) != 64
            for field in digest_fields
        )
    ):
        raise LegacyNnueError("derived MLX NNUE derivation metadata is invalid")


class LegacySparseNnue:
    """Sparse-binary NNUE computation expressed with standard MLX primitives."""

    def __init__(self, tensors: dict[str, mx.array]):
        core_shapes = {
            "w1": (LEGACY_NNUE_FEATURES, LEGACY_NNUE_HIDDEN1),
            "b1": (LEGACY_NNUE_HIDDEN1,),
            "w2": (LEGACY_NNUE_HIDDEN1, LEGACY_NNUE_HIDDEN2),
            "b2": (LEGACY_NNUE_HIDDEN2,),
            "w3": (LEGACY_NNUE_HIDDEN2,),
            "b3": (1,),
            "w3_policy": (LEGACY_NNUE_HIDDEN2,),
            "b3_policy": (1,),
        }
        split_shapes = {
            "w3_wildlife": (LEGACY_NNUE_HIDDEN2,),
            "b3_wildlife": (1,),
            "w3_habitat": (LEGACY_NNUE_HIDDEN2,),
            "b3_habitat": (1,),
        }
        split11_shapes = {
            "w3_heads": (NNUE_SPLIT_HEADS, LEGACY_NNUE_HIDDEN2),
            "b3_heads": (NNUE_SPLIT_HEADS,),
        }
        heteroscedastic_shapes = {
            "w3_var": (LEGACY_NNUE_HIDDEN2,),
            "b3_var": (1,),
        }
        names = set(tensors)
        if not set(core_shapes).issubset(names):
            raise LegacyNnueError("MLX NNUE tensor names do not match the artifact contract")
        self.has_split_value_heads = bool(names & set(split_shapes))
        self.has_split11_heads = bool(names & set(split11_shapes))
        self.has_heteroscedastic = bool(names & set(heteroscedastic_shapes))
        expected_shapes = dict(core_shapes)
        for present, shapes in (
            (self.has_split_value_heads, split_shapes),
            (self.has_split11_heads, split11_shapes),
            (self.has_heteroscedastic, heteroscedastic_shapes),
        ):
            if present:
                expected_shapes.update(shapes)
        if names != set(expected_shapes):
            raise LegacyNnueError("MLX NNUE optional head tensors are incomplete or unknown")
        if self.has_split11_heads and not self.has_split_value_heads:
            raise LegacyNnueError("MLX NNUE split-11 heads require the split-value head block")
        if self.has_heteroscedastic and not self.has_split11_heads:
            raise LegacyNnueError("MLX NNUE heteroscedastic head requires the split-11 head block")
        for name, shape in expected_shapes.items():
            if tuple(tensors[name].shape) != shape:
                raise LegacyNnueError(f"MLX NNUE tensor {name} has invalid shape")
        self.tensors = tensors

    @classmethod
    def load(cls, root: str | Path) -> LegacySparseNnue:
        root = Path(root)
        manifest = load_legacy_nnue_manifest(root)
        tensors = dict(mx.load(str(root / "model.safetensors")))
        model = cls(tensors)
        if manifest["schema_version"] == CORRECTED_NNUE_ARTIFACT_SCHEMA:
            head_version = manifest["source"]["head_version"]
            if (
                model.has_split_value_heads != (head_version >= 2)
                or model.has_split11_heads != (head_version >= 3)
                or model.has_heteroscedastic != (head_version >= 4)
            ):
                raise LegacyNnueError("corrected MLX NNUE tensor heads do not match the manifest")
        for name, tensor in tensors.items():
            mx.eval(tensor)
            if not np.all(np.isfinite(np.asarray(tensor))):
                raise LegacyNnueError(f"MLX NNUE tensor {name} contains non-finite values")
        return model

    def __call__(self, feature_indices: mx.array, feature_mask: mx.array) -> mx.array:
        if feature_indices.ndim != 2 or feature_mask.shape != feature_indices.shape:
            raise LegacyNnueError("sparse NNUE batch must have matching rank-two arrays")
        gathered = mx.take(self.tensors["w1"], feature_indices, axis=0)
        h1 = self.tensors["b1"] + mx.sum(
            gathered * feature_mask[..., None].astype(gathered.dtype),
            axis=1,
        )
        h1 = mx.maximum(h1, 0.0)
        h2 = mx.maximum(h1 @ self.tensors["w2"] + self.tensors["b2"], 0.0)
        if getattr(self, "has_split11_heads", False):
            heads = h2 @ self.tensors["w3_heads"].T + self.tensors["b3_heads"]
            return mx.sum(heads, axis=1)
        if getattr(self, "has_split_value_heads", False):
            wildlife = h2 @ self.tensors["w3_wildlife"] + self.tensors["b3_wildlife"][0]
            habitat = h2 @ self.tensors["w3_habitat"] + self.tensors["b3_habitat"][0]
            return wildlife + habitat
        return h2 @ self.tensors["w3"] + self.tensors["b3"][0]


class LegacyRustExactSparseNnue(LegacySparseNnue):
    """MLX Metal inference with the qualified Rust forward pass's operation order."""

    @classmethod
    def load(cls, root: str | Path) -> LegacyRustExactSparseNnue:
        root = Path(root)
        manifest = load_legacy_nnue_manifest(root)
        tensors = dict(mx.load(str(root / "model.safetensors")))
        model = cls(tensors)
        if manifest["schema_version"] == CORRECTED_NNUE_ARTIFACT_SCHEMA:
            head_version = manifest["source"]["head_version"]
            if (
                model.has_split_value_heads != (head_version >= 2)
                or model.has_split11_heads != (head_version >= 3)
                or model.has_heteroscedastic != (head_version >= 4)
            ):
                raise LegacyNnueError("corrected MLX NNUE tensor heads do not match the manifest")
        for name, tensor in tensors.items():
            mx.eval(tensor)
            if not np.all(np.isfinite(np.asarray(tensor))):
                raise LegacyNnueError(f"MLX NNUE tensor {name} contains non-finite values")
        return model

    def all_hidden_and_output(
        self,
        feature_offsets: mx.array,
        feature_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if feature_offsets.ndim != 1 or feature_indices.ndim != 1:
            raise LegacyNnueError("exact sparse NNUE inputs must be rank-one arrays")
        rows = feature_offsets.shape[0] - 1
        if rows <= 0:
            raise LegacyNnueError("exact sparse NNUE batch cannot be empty")
        h1 = _RUST_EXACT_H1(
            inputs=[
                feature_offsets,
                feature_indices,
                self.tensors["w1"],
                self.tensors["b1"],
            ],
            output_shapes=[(rows, LEGACY_NNUE_HIDDEN1)],
            output_dtypes=[mx.float32],
            grid=(rows * (LEGACY_NNUE_HIDDEN1 // 4), 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        h2, output = self._hidden_and_output_from_h1(h1, rows)
        return h1, h2, output

    def _hidden_and_output_from_h1(
        self,
        h1: mx.array,
        rows: int,
    ) -> tuple[mx.array, mx.array]:
        h2 = _RUST_EXACT_H2(
            inputs=[h1, self.tensors["w2"], self.tensors["b2"]],
            output_shapes=[(rows, LEGACY_NNUE_HIDDEN2)],
            output_dtypes=[mx.float32],
            grid=(rows * (LEGACY_NNUE_HIDDEN2 // 8), 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        output_arguments = {
            "output_shapes": [(rows,)],
            "output_dtypes": [mx.float32],
            "grid": (rows, 1, 1),
            "threadgroup": (min(rows, 256), 1, 1),
        }
        if self.has_split11_heads:
            output = _RUST_EXACT_SPLIT11_OUTPUT(
                inputs=[h2, self.tensors["w3_heads"], self.tensors["b3_heads"]],
                **output_arguments,
            )[0]
        elif self.has_split_value_heads:
            output = _RUST_EXACT_SPLIT2_OUTPUT(
                inputs=[
                    h2,
                    self.tensors["w3_wildlife"],
                    self.tensors["b3_wildlife"],
                    self.tensors["w3_habitat"],
                    self.tensors["b3_habitat"],
                ],
                **output_arguments,
            )[0]
        else:
            output = _RUST_EXACT_OUTPUT(
                inputs=[h2, self.tensors["w3"], self.tensors["b3"]],
                **output_arguments,
            )[0]
        return h2, output

    def hidden_and_output(
        self,
        feature_offsets: mx.array,
        feature_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        _, h2, output = self.all_hidden_and_output(feature_offsets, feature_indices)
        return h2, output

    def __call__(self, feature_offsets: mx.array, feature_indices: mx.array) -> mx.array:
        return self.hidden_and_output(feature_offsets, feature_indices)[1]


def pack_sparse_features(
    feature_sets: list[list[int]],
    *,
    feature_count: int = LEGACY_NNUE_FEATURES,
) -> tuple[mx.array, mx.array]:
    if not feature_sets:
        raise LegacyNnueError("sparse NNUE batch cannot be empty")
    maximum = max((len(features) for features in feature_sets), default=0)
    width = max(maximum, 1)
    indices = np.zeros((len(feature_sets), width), dtype=np.int32)
    mask = np.zeros((len(feature_sets), width), dtype=np.bool_)
    for row, features in enumerate(feature_sets):
        if any(index < 0 or index >= feature_count for index in features):
            raise LegacyNnueError(f"sparse NNUE row {row} contains an out-of-range index")
        if features:
            indices[row, : len(features)] = features
            mask[row, : len(features)] = True
    return mx.array(indices), mx.array(mask)


def pack_sparse_csr(
    feature_sets: list[list[int]],
    *,
    feature_count: int = LEGACY_NNUE_FEATURES,
) -> tuple[mx.array, mx.array]:
    if not feature_sets:
        raise LegacyNnueError("sparse NNUE batch cannot be empty")
    offsets = np.zeros(len(feature_sets) + 1, dtype=np.uint32)
    total = 0
    for row, features in enumerate(feature_sets):
        if any(index < 0 or index >= feature_count for index in features):
            raise LegacyNnueError(f"sparse NNUE row {row} contains an out-of-range index")
        total += len(features)
        if total > np.iinfo(np.uint32).max:
            raise LegacyNnueError("sparse NNUE batch contains too many feature indices")
        offsets[row + 1] = total
    indices = np.empty(total, dtype=np.uint16)
    position = 0
    for features in feature_sets:
        end = position + len(features)
        indices[position:end] = features
        position = end
    return mx.array(offsets), mx.array(indices)


def reference_forward(weights: LegacyNnueWeights, features: list[int]) -> np.float32:
    if any(index < 0 or index >= weights.feature_count for index in features):
        raise LegacyNnueError("reference sparse feature set contains an out-of-range index")
    h1 = weights.b1.copy()
    for index in features:
        h1 += weights.w1[index]
    np.maximum(h1, np.float32(0.0), out=h1)
    h2 = weights.b2.copy()
    for index, activation in enumerate(h1):
        if activation > 0:
            h2 += activation * weights.w2[index]
    np.maximum(h2, np.float32(0.0), out=h2)
    if weights.has_split11_heads:
        w3_heads = _required_tensor(weights.w3_heads, "w3_heads")
        b3_heads = _required_tensor(weights.b3_heads, "b3_heads")
        total = np.float32(0.0)
        for head_index in range(NNUE_SPLIT_HEADS):
            head = np.float32(b3_heads[head_index])
            for index, activation in enumerate(h2):
                head = np.float32(head + activation * w3_heads[head_index, index])
            total = np.float32(total + head)
        return total
    if weights.has_split_value_heads:
        w3_wildlife = _required_tensor(weights.w3_wildlife, "w3_wildlife")
        b3_wildlife = _required_tensor(weights.b3_wildlife, "b3_wildlife")
        w3_habitat = _required_tensor(weights.w3_habitat, "w3_habitat")
        b3_habitat = _required_tensor(weights.b3_habitat, "b3_habitat")
        wildlife = np.float32(b3_wildlife[0])
        habitat = np.float32(b3_habitat[0])
        for index, activation in enumerate(h2):
            wildlife = np.float32(wildlife + activation * w3_wildlife[index])
            habitat = np.float32(habitat + activation * w3_habitat[index])
        return np.float32(wildlife + habitat)
    result = np.float32(weights.b3[0])
    for index, activation in enumerate(h2):
        result = np.float32(result + activation * weights.w3[index])
    return result
