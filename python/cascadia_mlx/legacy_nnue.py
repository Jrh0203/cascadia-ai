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

_RUST_EXACT_H1 = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_h1_v1",
    input_names=["offsets", "indices", "w1", "b1"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        uint row = elem / 512;
        uint hidden = elem - row * 512;
        uint start = uint(offsets[row]);
        uint end = uint(offsets[row + 1]);
        float value = b1[hidden];
        for (uint position = start; position < end; ++position) {
            uint feature = uint(indices[position]);
            value = value + w1[feature * 512 + hidden];
        }
        out[elem] = value > 0.0f ? value : 0.0f;
    """,
)

_RUST_EXACT_H2 = mx.fast.metal_kernel(
    name="legacy_nnue_rust_exact_h2_v1",
    input_names=["h1", "w2", "b2"],
    output_names=["out"],
    header="#pragma clang fp contract(off)\n",
    source=r"""
        uint elem = thread_position_in_grid.x;
        uint row = elem / 64;
        uint hidden = elem - row * 64;
        float value = b2[hidden];
        for (uint input = 0; input < 512; ++input) {
            float activation = h1[row * 512 + input];
            if (activation > 0.0f) {
                float product = activation * w2[input * 64 + hidden];
                value = value + product;
            }
        }
        out[elem] = value > 0.0f ? value : 0.0f;
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

    def tensors(self) -> dict[str, mx.array]:
        return {
            "w1": mx.array(self.w1),
            "b1": mx.array(self.b1),
            "w2": mx.array(self.w2),
            "b2": mx.array(self.b2),
            "w3": mx.array(self.w3),
            "b3": mx.array(self.b3),
            "w3_policy": mx.array(self.w3_policy),
            "b3_policy": mx.array(self.b3_policy),
        }


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
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8 or raw[:4] != LEGACY_NNUE_MAGIC:
        raise LegacyNnueError("legacy NNUE source has invalid magic")
    version = struct.unpack_from("<I", raw, 4)[0]
    if version != LEGACY_NNUE_VERSION:
        raise LegacyNnueError(f"unsupported legacy NNUE version {version}")

    fixed_floats = hidden1 + hidden1 * hidden2 + hidden2 + hidden2 + 1
    policy_floats = hidden2 + 1
    payload_bytes = len(raw) - 8
    fixed_bytes = (fixed_floats + policy_floats) * 4
    if payload_bytes <= fixed_bytes:
        raise LegacyNnueError("legacy NNUE source is too short")
    first_layer_bytes = payload_bytes - fixed_bytes
    row_bytes = hidden1 * 4
    if first_layer_bytes % row_bytes:
        raise LegacyNnueError("legacy NNUE first-layer byte count is not row aligned")
    feature_count = first_layer_bytes // row_bytes
    if expected_features is not None and feature_count != expected_features:
        raise LegacyNnueError(
            f"legacy NNUE feature count {feature_count} != expected {expected_features}"
        )

    values = np.frombuffer(raw, dtype="<f4", offset=8)
    offset = 0

    def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        end = offset + count
        if end > len(values):
            raise LegacyNnueError("legacy NNUE source ended inside a tensor")
        tensor = np.array(values[offset:end], dtype=np.float32, copy=True).reshape(shape)
        offset = end
        return tensor

    w1 = take(feature_count * hidden1, (feature_count, hidden1))
    b1 = take(hidden1, (hidden1,))
    w2 = take(hidden1 * hidden2, (hidden1, hidden2))
    b2 = take(hidden2, (hidden2,))
    w3 = take(hidden2, (hidden2,))
    b3 = take(1, (1,))
    w3_policy = take(hidden2, (hidden2,))
    b3_policy = take(1, (1,))
    if offset != len(values):
        raise LegacyNnueError("legacy NNUE source has unparsed trailing values")
    for name, tensor in {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
        "w3_policy": w3_policy,
        "b3_policy": b3_policy,
    }.items():
        if not np.all(np.isfinite(tensor)):
            raise LegacyNnueError(f"legacy NNUE tensor {name} contains non-finite values")
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
    )


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


def load_legacy_nnue_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    try:
        manifest = json.loads((root / "model.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyNnueError(f"cannot read MLX NNUE manifest: {error}") from error
    if (
        manifest.get("schema_version")
        not in (LEGACY_NNUE_ARTIFACT_SCHEMA, LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA)
        or manifest.get("architecture") != LEGACY_NNUE_ARCHITECTURE
        or manifest.get("dimensions")
        != {
            "features": LEGACY_NNUE_FEATURES,
            "hidden1": LEGACY_NNUE_HIDDEN1,
            "hidden2": LEGACY_NNUE_HIDDEN2,
            "outputs": 1,
        }
    ):
        raise LegacyNnueError("MLX NNUE manifest contract is invalid")
    source = manifest.get("source", {})
    if (
        source.get("bytes") != LEGACY_NNUE_SOURCE_BYTES
        or source.get("blake3") != LEGACY_NNUE_SOURCE_BLAKE3
        or source.get("version") != LEGACY_NNUE_VERSION
    ):
        raise LegacyNnueError("MLX NNUE manifest source identity is invalid")
    if manifest["schema_version"] == LEGACY_NNUE_DERIVED_ARTIFACT_SCHEMA:
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
        expected_shapes = {
            "w1": (LEGACY_NNUE_FEATURES, LEGACY_NNUE_HIDDEN1),
            "b1": (LEGACY_NNUE_HIDDEN1,),
            "w2": (LEGACY_NNUE_HIDDEN1, LEGACY_NNUE_HIDDEN2),
            "b2": (LEGACY_NNUE_HIDDEN2,),
            "w3": (LEGACY_NNUE_HIDDEN2,),
            "b3": (1,),
            "w3_policy": (LEGACY_NNUE_HIDDEN2,),
            "b3_policy": (1,),
        }
        if set(tensors) != set(expected_shapes):
            raise LegacyNnueError("MLX NNUE tensor names do not match the artifact contract")
        for name, shape in expected_shapes.items():
            if tuple(tensors[name].shape) != shape:
                raise LegacyNnueError(f"MLX NNUE tensor {name} has invalid shape")
        self.tensors = tensors

    @classmethod
    def load(cls, root: str | Path) -> LegacySparseNnue:
        root = Path(root)
        load_legacy_nnue_manifest(root)
        tensors = dict(mx.load(str(root / "model.safetensors")))
        model = cls(tensors)
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
        return h2 @ self.tensors["w3"] + self.tensors["b3"][0]


class LegacyRustExactSparseNnue(LegacySparseNnue):
    """MLX Metal inference with the qualified Rust forward pass's operation order."""

    @classmethod
    def load(cls, root: str | Path) -> LegacyRustExactSparseNnue:
        root = Path(root)
        load_legacy_nnue_manifest(root)
        tensors = dict(mx.load(str(root / "model.safetensors")))
        model = cls(tensors)
        for name, tensor in tensors.items():
            mx.eval(tensor)
            if not np.all(np.isfinite(np.asarray(tensor))):
                raise LegacyNnueError(f"MLX NNUE tensor {name} contains non-finite values")
        return model

    def hidden_and_output(
        self,
        feature_offsets: mx.array,
        feature_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
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
            grid=(rows * LEGACY_NNUE_HIDDEN1, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        h2 = _RUST_EXACT_H2(
            inputs=[h1, self.tensors["w2"], self.tensors["b2"]],
            output_shapes=[(rows, LEGACY_NNUE_HIDDEN2)],
            output_dtypes=[mx.float32],
            grid=(rows * LEGACY_NNUE_HIDDEN2, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        output = _RUST_EXACT_OUTPUT(
            inputs=[h2, self.tensors["w3"], self.tensors["b3"]],
            output_shapes=[(rows,)],
            output_dtypes=[mx.float32],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 256), 1, 1),
        )[0]
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
    offsets = np.zeros(len(feature_sets) + 1, dtype=np.int32)
    total = 0
    for row, features in enumerate(feature_sets):
        if any(index < 0 or index >= feature_count for index in features):
            raise LegacyNnueError(f"sparse NNUE row {row} contains an out-of-range index")
        total += len(features)
        if total > np.iinfo(np.int32).max:
            raise LegacyNnueError("sparse NNUE batch contains too many feature indices")
        offsets[row + 1] = total
    indices = np.empty(total, dtype=np.int32)
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
    result = np.float32(weights.b3[0])
    for index, activation in enumerate(h2):
        result = np.float32(result + activation * weights.w3[index])
    return result
