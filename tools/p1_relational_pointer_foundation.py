#!/usr/bin/env python3
"""Prove exact selected-prefix pointer semantics over the sparse R2 state."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import numpy as np

SCHEMA_VERSION = 1
EXPERIMENT_ID = "p1-relational-hierarchical-pointer-foundation-v1"
PROTOCOL_ID = "exact-r2-selected-prefix-pointer-alignment-v1"
ADR_ID = "0174"

STAGES = ("draft", "tile", "wildlife")
DRAFT_FACTOR_DIM = 117
TILE_FACTOR_DIM = 8
WILDLIFE_FACTOR_DIM = 3
STAGED_PUBLIC_DIM = 158
TILE_CONTEXT_TILE_OFFSET = DRAFT_FACTOR_DIM + STAGED_PUBLIC_DIM
COORDINATE_SCALE = 24.0
CHAMPION_FRONTIER_FLAG = 1 << 1
R2_OCCUPIED_TOKEN = 1
R2_FRONTIER_TOKEN = 2
HISTORICAL_DENSE_CELLS_PER_BOARD = 441
COMPACT_REFERENCE_CELLS_PER_BOARD = 121
MAXIMUM_EXACT_SPARSE_TOKENS_PER_BOARD = 121
MAXIMUM_DRAFT_POINTERS = 20
MAXIMUM_FRONTIER_POINTERS = 31
MAXIMUM_WILDLIFE_DESTINATION_POINTERS = 25
EXPECTED_SPLIT_COUNTS = {
    "train": (560, 2_135_111),
    "validation": (240, 860_203),
}
REQUIRED_FACTOR_ARRAYS = {
    "action_hash",
    "action_source_flags",
    "draft_action_item",
    "draft_item_features",
    "draft_item_hash",
    "draft_query_group",
    "draft_query_offsets",
    "group_action_offsets",
    "group_id",
    "tile_action_item",
    "tile_item_features",
    "tile_item_hash",
    "tile_query_context",
    "tile_query_group",
    "tile_query_offsets",
    "wildlife_action_item",
    "wildlife_item_features",
    "wildlife_item_hash",
    "wildlife_query_context",
    "wildlife_query_group",
    "wildlife_query_offsets",
}
REQUIRED_R3_TENSORS = {
    "action_hashes",
    "candidate_offsets",
    "group_ids",
    "parent_token_payload",
    "parent_token_types",
    "source_candidate_counts",
    "source_candidate_indices",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTOR_CACHE = (
    REPO_ROOT
    / "artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache"
)
DEFAULT_R3_CACHE = (
    REPO_ROOT
    / "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache"
    / "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
DEFAULT_D6_METADATA = (
    REPO_ROOT / "python/cascadia_mlx/d6_contract_metadata.v1.json"
)


class PointerFoundationError(ValueError):
    """The pointer foundation input or result is malformed."""


@dataclass(frozen=True)
class R3Split:
    """The exact R3 tensors needed to bind pointer items to sparse R2."""

    manifest: dict[str, Any]
    tensors: dict[str, np.memmap]
    group_rows: dict[int, int]


def canonical_blake3(value: object) -> str:
    """Hash one strict canonical JSON value."""
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_blake3(path: Path) -> str:
    """Stream a file into BLAKE3."""
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PointerFoundationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PointerFoundationError(f"{label} root must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require_blake3(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PointerFoundationError(f"{label} must be a lowercase BLAKE3 digest")
    return value


def _distribution(values: np.ndarray | list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or not len(array):
        raise PointerFoundationError("distribution input must be nonempty and one-dimensional")
    return {
        "count": len(array),
        "minimum": int(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": int(array.max()),
    }


def _decode_coordinates(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise PointerFoundationError(f"{label} coordinates must have shape (N, 2)")
    scaled = array * COORDINATE_SCALE
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-5, rtol=0.0):
        raise PointerFoundationError(f"{label} contains a non-integral normalized coordinate")
    if np.any((rounded < -127) | (rounded > 127)):
        raise PointerFoundationError(f"{label} coordinate escapes signed byte range")
    return rounded.astype(np.int16)


def _hash16(payload: bytes) -> bytes:
    return blake3.blake3(payload).digest(length=16)


def _factor_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype=np.float32).tobytes()


def _pointer_key(
    draft_hash: bytes,
    tile_coord: tuple[int, int],
    rotation: int,
    wildlife_kind: int,
    wildlife_coord: tuple[int, int],
) -> bytes:
    if len(draft_hash) != 16:
        raise PointerFoundationError("draft hash must contain 16 bytes")
    if not 0 <= rotation < 6 or not 0 <= wildlife_kind < 3:
        raise PointerFoundationError("pointer rotation or wildlife kind is invalid")
    values = np.asarray(
        [
            tile_coord[0],
            tile_coord[1],
            rotation,
            wildlife_kind,
            wildlife_coord[0],
            wildlife_coord[1],
        ],
        dtype=np.int16,
    )
    return draft_hash + values.tobytes()


def _load_d6(path: Path) -> dict[str, Any]:
    raw = _read_json(path, "D6 metadata")
    if (
        raw.get("schema_version") != 1
        or raw.get("contract_id") != "cascadia-game-exact-d6-v1"
        or len(raw.get("coordinate_matrices", [])) != 12
        or len(raw.get("dual_tile_rotation_tables", [])) != 12
        or len(raw.get("single_tile_rotation_tables", [])) != 12
        or len(raw.get("inverse_table", [])) != 12
    ):
        raise PointerFoundationError("D6 metadata does not match the exact contract")
    _require_blake3(raw.get("scientific_blake3"), "D6 scientific identity")
    return raw


def _d6_pointer_checks(
    coordinates: np.ndarray,
    *,
    rotations: np.ndarray | None,
    dual_terrain: np.ndarray | None,
    d6: dict[str, Any],
) -> tuple[int, int]:
    coords = np.asarray(coordinates, dtype=np.int16)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise PointerFoundationError("D6 pointer coordinates must have shape (N, 2)")
    if rotations is None:
        if dual_terrain is not None:
            raise PointerFoundationError("D6 dual-terrain mask requires rotations")
    else:
        rotations = np.asarray(rotations, dtype=np.int16)
        dual_terrain = np.asarray(dual_terrain, dtype=np.bool_)
        if rotations.shape != (len(coords),) or dual_terrain.shape != (len(coords),):
            raise PointerFoundationError("D6 rotation inputs do not align with coordinates")
        if np.any((rotations < 0) | (rotations >= 6)):
            raise PointerFoundationError("D6 rotation escapes [0, 5]")
        if np.any(~dual_terrain & (rotations != 0)):
            raise PointerFoundationError("single-terrain pointer rotation is not canonical zero")

    matrices = np.asarray(d6["coordinate_matrices"], dtype=np.int16)
    inverses = np.asarray(d6["inverse_table"], dtype=np.int64)
    dual_tables = np.asarray(d6["dual_tile_rotation_tables"], dtype=np.int16)
    single_tables = np.asarray(d6["single_tile_rotation_tables"], dtype=np.int16)
    checks = 0
    failures = 0
    for transform_id in range(12):
        transformed = coords @ matrices[transform_id].T
        restored = transformed @ matrices[inverses[transform_id]].T
        failures += int(np.any(restored != coords, axis=1).sum())
        checks += len(coords)
        if rotations is not None:
            transformed_rotation = np.where(
                dual_terrain,
                dual_tables[transform_id, rotations],
                single_tables[transform_id, rotations],
            )
            restored_rotation = np.where(
                dual_terrain,
                dual_tables[inverses[transform_id], transformed_rotation],
                single_tables[inverses[transform_id], transformed_rotation],
            )
            failures += int((restored_rotation != rotations).sum())
            checks += len(coords)
    return checks, failures


def _load_r3_split(root: Path, split: str) -> R3Split:
    manifest_path = root / "cache.json"
    manifest = _read_json(manifest_path, "R3 cache manifest")
    raw_split = manifest.get("splits", {}).get(split)
    if not isinstance(raw_split, dict):
        raise PointerFoundationError(f"R3 cache lacks split {split}")
    files = raw_split.get("files")
    if not isinstance(files, dict) or not REQUIRED_R3_TENSORS.issubset(files):
        raise PointerFoundationError("R3 cache lacks required pointer tensors")
    tensors: dict[str, np.memmap] = {}
    for name in sorted(REQUIRED_R3_TENSORS):
        specification = files[name]
        if not isinstance(specification, dict):
            raise PointerFoundationError(f"R3 tensor specification is malformed: {name}")
        path = root / str(specification.get("file", ""))
        shape = specification.get("shape")
        dtype_name = specification.get("dtype")
        if (
            path.parent != root
            or not path.is_file()
            or not isinstance(shape, list)
            or not isinstance(dtype_name, str)
        ):
            raise PointerFoundationError(f"R3 tensor path or shape is invalid: {name}")
        dtype = np.dtype(dtype_name)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if path.stat().st_size != expected_bytes or specification.get("bytes") != expected_bytes:
            raise PointerFoundationError(f"R3 tensor byte count drifted: {name}")
        if file_blake3(path) != specification.get("blake3"):
            raise PointerFoundationError(f"R3 tensor checksum mismatch: {name}")
        tensors[name] = np.memmap(path, mode="r", dtype=dtype, shape=tuple(shape))
    group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
    if len(set(int(group_id) for group_id in group_ids)) != len(group_ids):
        raise PointerFoundationError("R3 group IDs are duplicated")
    return R3Split(
        manifest=manifest,
        tensors=tensors,
        group_rows={int(group_id): row for row, group_id in enumerate(group_ids)},
    )


def _validate_factor_manifest(root: Path, split: str) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json", f"{split} factor-cache manifest")
    expected_groups, expected_actions = EXPECTED_SPLIT_COUNTS[split]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("cache_schema") != "hierarchical-factor-retrieval-cache-v1"
        or manifest.get("groups") != expected_groups
        or manifest.get("candidates") != expected_actions
        or manifest.get("all_factor_bijections") is not True
        or manifest.get("all_prefix_invariants") is not True
    ):
        raise PointerFoundationError(f"{split} factor-cache manifest drifted")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PointerFoundationError(f"{split} factor cache has no shards")
    return manifest


def _source_identity(
    *,
    bundle_id: str,
    d6_path: Path,
) -> dict[str, Any]:
    return {
        "bundle_id": _require_blake3(bundle_id, "bundle ID"),
        "script_blake3": file_blake3(Path(__file__)),
        "d6_metadata_blake3": file_blake3(d6_path),
    }


def audit_split(
    *,
    split: str,
    factor_cache: Path,
    r3_cache: Path,
    d6_metadata: Path,
    bundle_id: str,
    runtime_host: str | None = None,
) -> dict[str, Any]:
    """Audit one complete open split without writing another cache."""
    if split not in EXPECTED_SPLIT_COUNTS:
        raise PointerFoundationError("split must be train or validation")
    started = time.perf_counter()
    factor_root = factor_cache / split
    factor_manifest = _validate_factor_manifest(factor_root, split)
    r3 = _load_r3_split(r3_cache, split)
    r3_split = r3.manifest["splits"][split]
    if (
        r3_split.get("groups") != factor_manifest["groups"]
        or r3_split.get("source_candidates") != factor_manifest["candidates"]
        or r3_split.get("dataset_manifest_blake3")
        != factor_manifest.get("dataset_manifest_blake3")
    ):
        raise PointerFoundationError("factor and R3 split identities do not align")
    d6 = _load_d6(d6_metadata)

    checks = {
        "groups": 0,
        "source_actions": 0,
        "anchored_actions": 0,
        "eligible_actions": 0,
        "r3_action_hash_checks": 0,
        "r3_action_hash_failures": 0,
        "prefix_hash_checks": 0,
        "prefix_hash_failures": 0,
        "tile_pointer_checks": 0,
        "tile_pointer_missing": 0,
        "tile_pointer_ambiguous": 0,
        "wildlife_pointer_checks": 0,
        "wildlife_existing": 0,
        "wildlife_new_tile": 0,
        "wildlife_none": 0,
        "wildlife_pointer_missing": 0,
        "wildlife_pointer_ambiguous": 0,
        "action_map_checks": 0,
        "action_map_failures": 0,
        "pointer_bijection_checks": 0,
        "pointer_collisions": 0,
        "d6_checks": 0,
        "d6_failures": 0,
    }
    query_widths: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGES}
    active_board_tokens: list[int] = []
    all_board_tokens: list[int] = []
    frontier_tokens: list[int] = []
    occupied_tokens: list[int] = []
    wildlife_destination_support: list[int] = []
    seen_group_ids: set[int] = set()

    for shard_entry in sorted(
        factor_manifest["shards"],
        key=lambda value: int(value["shard_index"]),
    ):
        shard_path = factor_root / str(shard_entry["cache_file"])
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != shard_entry.get("cache_bytes")
            or file_blake3(shard_path) != shard_entry.get("cache_blake3")
        ):
            raise PointerFoundationError(f"factor shard checksum mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as arrays:
            if not REQUIRED_FACTOR_ARRAYS.issubset(arrays.files):
                raise PointerFoundationError(f"factor shard tensor set is incomplete: {shard_path}")
            group_ids = np.asarray(arrays["group_id"], dtype=np.uint64)
            group_offsets = np.asarray(arrays["group_action_offsets"], dtype=np.int64)
            if len(group_offsets) != len(group_ids) + 1:
                raise PointerFoundationError("factor group offsets do not align with group IDs")
            for stage in STAGES:
                offsets = np.asarray(arrays[f"{stage}_query_offsets"], dtype=np.int64)
                if len(offsets) < 2 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
                    raise PointerFoundationError(f"{stage} query offsets are invalid")
                query_widths[stage].append(np.diff(offsets))

            draft_features = np.asarray(arrays["draft_item_features"], dtype=np.float32)
            draft_hashes = np.asarray(arrays["draft_item_hash"], dtype=np.uint8)
            tile_context = np.asarray(arrays["tile_query_context"], dtype=np.float32)
            tile_features = np.asarray(arrays["tile_item_features"], dtype=np.float32)
            tile_hashes = np.asarray(arrays["tile_item_hash"], dtype=np.uint8)
            tile_offsets = np.asarray(arrays["tile_query_offsets"], dtype=np.int64)
            tile_query_groups = np.asarray(arrays["tile_query_group"], dtype=np.int64)
            wildlife_context = np.asarray(
                arrays["wildlife_query_context"],
                dtype=np.float32,
            )
            wildlife_features = np.asarray(
                arrays["wildlife_item_features"],
                dtype=np.float32,
            )
            wildlife_hashes = np.asarray(arrays["wildlife_item_hash"], dtype=np.uint8)
            wildlife_offsets = np.asarray(
                arrays["wildlife_query_offsets"],
                dtype=np.int64,
            )
            wildlife_query_groups = np.asarray(
                arrays["wildlife_query_group"],
                dtype=np.int64,
            )
            if (
                len(tile_context) != len(draft_features)
                or len(wildlife_context) != len(tile_features)
                or len(tile_offsets) != len(tile_context) + 1
                or len(wildlife_offsets) != len(wildlife_context) + 1
            ):
                raise PointerFoundationError("selected-prefix query ordering drifted")

            tile_coordinates = _decode_coordinates(
                tile_features[:, :2],
                "tile pointer",
            )
            tile_rotations = np.argmax(tile_features[:, 2:8], axis=-1).astype(np.int16)
            tile_query_index = np.repeat(
                np.arange(len(tile_context), dtype=np.int64),
                np.diff(tile_offsets),
            )
            if len(tile_query_index) != len(tile_features):
                raise PointerFoundationError("tile query-to-item expansion drifted")
            dual_queries = np.argmax(tile_context[:, 17:23], axis=-1) != 5
            d6_checks, d6_failures = _d6_pointer_checks(
                tile_coordinates,
                rotations=tile_rotations,
                dual_terrain=dual_queries[tile_query_index],
                d6=d6,
            )
            checks["d6_checks"] += d6_checks
            checks["d6_failures"] += d6_failures

            wildlife_coordinates = _decode_coordinates(
                wildlife_features[:, 1:3],
                "wildlife pointer",
            )
            wildlife_present = wildlife_features[:, 0] > 0.5
            d6_checks, d6_failures = _d6_pointer_checks(
                wildlife_coordinates[wildlife_present],
                rotations=None,
                dual_terrain=None,
                d6=d6,
            )
            checks["d6_checks"] += d6_checks
            checks["d6_failures"] += d6_failures

            for local_group, raw_group_id in enumerate(group_ids):
                group_id = int(raw_group_id)
                if group_id in seen_group_ids or group_id not in r3.group_rows:
                    raise PointerFoundationError("group identity is duplicated or absent from R3")
                seen_group_ids.add(group_id)
                r3_row = r3.group_rows[group_id]
                token_types = np.asarray(
                    r3.tensors["parent_token_types"][r3_row],
                    dtype=np.uint8,
                )
                token_payload = np.asarray(
                    r3.tensors["parent_token_payload"][r3_row],
                    dtype=np.int8,
                )
                active_types = token_types[0]
                active_payload = token_payload[0]
                frontier = active_payload[active_types == R2_FRONTIER_TOKEN, :2].astype(
                    np.int16
                )
                occupied = active_payload[active_types == R2_OCCUPIED_TOKEN, :2].astype(
                    np.int16
                )
                if (
                    len({tuple(value) for value in frontier}) != len(frontier)
                    or len({tuple(value) for value in occupied}) != len(occupied)
                ):
                    raise PointerFoundationError("R2 active-board coordinates are duplicated")
                active_board_tokens.append(int(np.count_nonzero(active_types)))
                all_board_tokens.append(int(np.count_nonzero(token_types)))
                frontier_tokens.append(len(frontier))
                occupied_tokens.append(len(occupied))
                wildlife_destination_support.append(len(occupied) + 2)
                checks["groups"] += 1

                action_start = int(group_offsets[local_group])
                action_end = int(group_offsets[local_group + 1])
                action_count = action_end - action_start
                checks["source_actions"] += action_count
                if int(r3.tensors["source_candidate_counts"][r3_row]) != action_count:
                    raise PointerFoundationError("R3 source action count differs from factor cache")
                r3_start = int(r3.tensors["candidate_offsets"][r3_row])
                r3_end = int(r3.tensors["candidate_offsets"][r3_row + 1])
                source_indices = np.asarray(
                    r3.tensors["source_candidate_indices"][r3_start:r3_end],
                    dtype=np.int64,
                )
                factor_action_hashes = np.asarray(
                    arrays["action_hash"][action_start:action_end],
                    dtype=np.uint8,
                )
                r3_action_hashes = np.asarray(
                    r3.tensors["action_hashes"][r3_start:r3_end],
                    dtype=np.uint8,
                )
                aligned_hashes = factor_action_hashes[source_indices]
                checks["r3_action_hash_checks"] += len(source_indices)
                checks["r3_action_hash_failures"] += int(
                    np.any(aligned_hashes != r3_action_hashes, axis=1).sum()
                )

            for query_index, context in enumerate(tile_context):
                draft = _factor_bytes(context[:DRAFT_FACTOR_DIM])
                checks["prefix_hash_checks"] += 1
                checks["prefix_hash_failures"] += int(
                    _hash16(draft) != bytes(draft_hashes[query_index])
                )
                group_local = int(tile_query_groups[query_index])
                if not 0 <= group_local < len(group_ids):
                    raise PointerFoundationError("tile query names an invalid group")
                r3_row = r3.group_rows[int(group_ids[group_local])]
                types = np.asarray(r3.tensors["parent_token_types"][r3_row, 0])
                payload = np.asarray(r3.tensors["parent_token_payload"][r3_row, 0])
                frontier = payload[types == R2_FRONTIER_TOKEN, :2].astype(np.int16)
                for item_index in range(
                    int(tile_offsets[query_index]),
                    int(tile_offsets[query_index + 1]),
                ):
                    tile = _factor_bytes(tile_features[item_index, :TILE_FACTOR_DIM])
                    checks["prefix_hash_checks"] += 1
                    checks["prefix_hash_failures"] += int(
                        _hash16(draft + tile) != bytes(tile_hashes[item_index])
                    )
                    matches = int(
                        np.all(frontier == tile_coordinates[item_index], axis=1).sum()
                    )
                    checks["tile_pointer_checks"] += 1
                    checks["tile_pointer_missing"] += int(matches == 0)
                    checks["tile_pointer_ambiguous"] += int(matches > 1)

            for query_index, context in enumerate(wildlife_context):
                draft = _factor_bytes(context[:DRAFT_FACTOR_DIM])
                tile = _factor_bytes(
                    context[
                        TILE_CONTEXT_TILE_OFFSET : TILE_CONTEXT_TILE_OFFSET
                        + TILE_FACTOR_DIM
                    ]
                )
                checks["prefix_hash_checks"] += 1
                checks["prefix_hash_failures"] += int(
                    _hash16(draft + tile) != bytes(tile_hashes[query_index])
                )
                tile_coord = _decode_coordinates(
                    context[
                        None,
                        TILE_CONTEXT_TILE_OFFSET : TILE_CONTEXT_TILE_OFFSET + 2,
                    ],
                    "selected-prefix tile pointer",
                )[0]
                group_local = int(wildlife_query_groups[query_index])
                if not 0 <= group_local < len(group_ids):
                    raise PointerFoundationError("wildlife query names an invalid group")
                r3_row = r3.group_rows[int(group_ids[group_local])]
                types = np.asarray(r3.tensors["parent_token_types"][r3_row, 0])
                payload = np.asarray(r3.tensors["parent_token_payload"][r3_row, 0])
                occupied = payload[types == R2_OCCUPIED_TOKEN, :2].astype(np.int16)
                for item_index in range(
                    int(wildlife_offsets[query_index]),
                    int(wildlife_offsets[query_index + 1]),
                ):
                    wildlife = _factor_bytes(
                        wildlife_features[item_index, :WILDLIFE_FACTOR_DIM]
                    )
                    checks["prefix_hash_checks"] += 1
                    checks["prefix_hash_failures"] += int(
                        _hash16(draft + tile + wildlife)
                        != bytes(wildlife_hashes[item_index])
                    )
                    checks["wildlife_pointer_checks"] += 1
                    if not wildlife_present[item_index]:
                        checks["wildlife_none"] += 1
                        continue
                    coord = wildlife_coordinates[item_index]
                    if np.array_equal(coord, tile_coord):
                        checks["wildlife_new_tile"] += 1
                        continue
                    matches = int(np.all(occupied == coord, axis=1).sum())
                    checks["wildlife_existing"] += int(matches == 1)
                    checks["wildlife_pointer_missing"] += int(matches == 0)
                    checks["wildlife_pointer_ambiguous"] += int(matches > 1)

            draft_map = np.asarray(arrays["draft_action_item"], dtype=np.int64)
            tile_map = np.asarray(arrays["tile_action_item"], dtype=np.int64)
            wildlife_map = np.asarray(arrays["wildlife_action_item"], dtype=np.int64)
            source_flags = np.asarray(arrays["action_source_flags"], dtype=np.int64)
            for local_group in range(len(group_ids)):
                start = int(group_offsets[local_group])
                end = int(group_offsets[local_group + 1])
                anchored = (source_flags[start:end] & CHAMPION_FRONTIER_FLAG) != 0
                mapped = (
                    (draft_map[start:end] >= 0)
                    & (tile_map[start:end] >= 0)
                    & (wildlife_map[start:end] >= 0)
                )
                expected_mapped = ~anchored
                checks["anchored_actions"] += int(anchored.sum())
                checks["eligible_actions"] += int(expected_mapped.sum())
                checks["action_map_checks"] += end - start
                checks["action_map_failures"] += int((mapped != expected_mapped).sum())
                pointer_keys: set[bytes] = set()
                for action_index in np.flatnonzero(mapped) + start:
                    draft_item = int(draft_map[action_index])
                    tile_item = int(tile_map[action_index])
                    wildlife_item = int(wildlife_map[action_index])
                    tile_coord = tuple(int(value) for value in tile_coordinates[tile_item])
                    rotation = int(tile_rotations[tile_item])
                    if not wildlife_present[wildlife_item]:
                        wildlife_kind = 0
                        wildlife_coord = (0, 0)
                    else:
                        wildlife_coord = tuple(
                            int(value) for value in wildlife_coordinates[wildlife_item]
                        )
                        wildlife_kind = 1 if wildlife_coord == tile_coord else 2
                    pointer_keys.add(
                        _pointer_key(
                            bytes(draft_hashes[draft_item]),
                            tile_coord,
                            rotation,
                            wildlife_kind,
                            wildlife_coord,
                        )
                    )
                checks["pointer_bijection_checks"] += int(mapped.sum())
                checks["pointer_collisions"] += int(mapped.sum()) - len(pointer_keys)

    expected_groups, expected_actions = EXPECTED_SPLIT_COUNTS[split]
    r3_retained = int(r3.manifest["splits"][split]["retained_candidates"])
    stage_distributions = {
        stage: _distribution(np.concatenate(query_widths[stage]))
        for stage in STAGES
    }
    token_distributions = {
        "active_board_exact_sparse_tokens": _distribution(active_board_tokens),
        "all_four_boards_exact_sparse_tokens": _distribution(all_board_tokens),
        "active_board_frontier_tokens": _distribution(frontier_tokens),
        "active_board_occupied_tokens": _distribution(occupied_tokens),
        "wildlife_destination_pointer_support": _distribution(
            wildlife_destination_support
        ),
    }
    compactness = {
        "uses_441_dense_cells": False,
        "historical_dense_cells_per_board": HISTORICAL_DENSE_CELLS_PER_BOARD,
        "compact_reference_cells_per_board": COMPACT_REFERENCE_CELLS_PER_BOARD,
        "maximum_exact_sparse_tokens_per_board": token_distributions[
            "active_board_exact_sparse_tokens"
        ]["maximum"],
        "maximum_exact_sparse_tokens_fraction_of_441": (
            token_distributions["active_board_exact_sparse_tokens"]["maximum"]
            / HISTORICAL_DENSE_CELLS_PER_BOARD
        ),
        "maximum_exact_sparse_tokens_fraction_of_121": (
            token_distributions["active_board_exact_sparse_tokens"]["maximum"]
            / COMPACT_REFERENCE_CELLS_PER_BOARD
        ),
        "tile_pointer_support": {
            "frontier_tokens_maximum": token_distributions[
                "active_board_frontier_tokens"
            ]["maximum"],
            "rotation_choices": 6,
            "flattened_tile_rows_are_not_model_tokens": True,
        },
        "wildlife_pointer_support": {
            "occupied_plus_new_tile_plus_none_maximum": token_distributions[
                "wildlife_destination_pointer_support"
            ]["maximum"],
        },
    }
    gates = {
        "complete_coverage": (
            checks["groups"] == expected_groups
            and len(seen_group_ids) == expected_groups
            and checks["source_actions"] == expected_actions
            and checks["anchored_actions"] + checks["eligible_actions"]
            == expected_actions
            and checks["r3_action_hash_checks"] == r3_retained
        ),
        "cross_cache_action_identity": checks["r3_action_hash_failures"] == 0,
        "selected_prefix_identity": checks["prefix_hash_failures"] == 0,
        "tile_pointer_exact": (
            checks["tile_pointer_checks"] == int(factor_manifest["items"]["tile"])
            and checks["tile_pointer_missing"] == 0
            and checks["tile_pointer_ambiguous"] == 0
        ),
        "wildlife_pointer_exact": (
            checks["wildlife_pointer_checks"]
            == int(factor_manifest["items"]["wildlife"])
            and checks["wildlife_pointer_missing"] == 0
            and checks["wildlife_pointer_ambiguous"] == 0
        ),
        "action_mapping_exact": (
            checks["action_map_checks"] == expected_actions
            and checks["action_map_failures"] == 0
        ),
        "complete_action_pointer_bijection": (
            checks["pointer_bijection_checks"] == checks["eligible_actions"]
            and checks["pointer_collisions"] == 0
        ),
        "d6_pointer_roundtrip": (
            checks["d6_checks"] > 0 and checks["d6_failures"] == 0
        ),
        "compact_exact_sparse_state": (
            compactness["maximum_exact_sparse_tokens_per_board"]
            <= MAXIMUM_EXACT_SPARSE_TOKENS_PER_BOARD
            and compactness["uses_441_dense_cells"] is False
        ),
        "bounded_pointer_support": (
            stage_distributions["draft"]["maximum"] <= MAXIMUM_DRAFT_POINTERS
            and token_distributions["active_board_frontier_tokens"]["maximum"]
            <= MAXIMUM_FRONTIER_POINTERS
            and token_distributions["wildlife_destination_pointer_support"]["maximum"]
            <= MAXIMUM_WILDLIFE_DESTINATION_POINTERS
        ),
    }
    passed = all(gates.values())
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "complete-open-split-pointer-alignment",
        "split": split,
        "inputs": {
            "factor_cache_payload_blake3": factor_manifest["payload_blake3"],
            "factor_dataset_manifest_blake3": factor_manifest[
                "dataset_manifest_blake3"
            ],
            "r3_cache_id": r3.manifest["cache_id"],
            "r3_dataset_manifest_blake3": r3_split["dataset_manifest_blake3"],
            "d6_contract_id": d6["contract_id"],
            "d6_scientific_blake3": d6["scientific_blake3"],
        },
        "source": _source_identity(
            bundle_id=bundle_id,
            d6_path=d6_metadata,
        ),
        "checks": checks,
        "stage_query_widths": stage_distributions,
        "token_and_pointer_support": token_distributions,
        "compactness": compactness,
        "gates": gates,
        "passed": passed,
        "classification": (
            "p1_relational_pointer_foundation_passed"
            if passed
            else "p1_relational_pointer_foundation_failed"
        ),
    }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_identity": identity,
        "scientific_blake3": canonical_blake3(identity),
        "runtime": {
            "host": runtime_host or socket.gethostname().split(".")[0],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
        },
    }


def _validate_report(report: dict[str, Any]) -> dict[str, Any]:
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or not isinstance(identity, dict)
        or report.get("scientific_blake3") != canonical_blake3(identity)
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("protocol_id") != PROTOCOL_ID
        or identity.get("adr") != ADR_ID
    ):
        raise PointerFoundationError("pointer report envelope is invalid")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("host"), str):
        raise PointerFoundationError("pointer report runtime identity is absent")
    return identity


def classify_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Require one cross-host-identical pair for each complete open split."""
    if len(reports) != 4:
        raise PointerFoundationError("classification requires exactly four reports")
    by_split: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        split: [] for split in EXPECTED_SPLIT_COUNTS
    }
    for report in reports:
        identity = _validate_report(report)
        split = identity.get("split")
        if split not in by_split:
            raise PointerFoundationError("pointer report names an unknown split")
        by_split[split].append((report, identity))

    split_results: dict[str, Any] = {}
    sources = []
    structural = True
    cross_host_consistent = True
    scientific_passed = True
    for split, entries in by_split.items():
        if len(entries) != 2:
            structural = False
            continue
        hosts = [entry[0]["runtime"]["host"] for entry in entries]
        if len(set(hosts)) != 2:
            cross_host_consistent = False
        digests = [entry[0]["scientific_blake3"] for entry in entries]
        if digests[0] != digests[1]:
            cross_host_consistent = False
        identities = [entry[1] for entry in entries]
        scientific_passed &= all(identity.get("passed") is True for identity in identities)
        sources.extend(identity.get("source") for identity in identities)
        split_results[split] = {
            "scientific_blake3": digests[0],
            "hosts": sorted(hosts),
            "identity": identities[0],
        }
    if len(sources) != 4 or any(source != sources[0] for source in sources[1:]):
        cross_host_consistent = False
    gates = {
        "structural": structural,
        "cross_host_consistent": cross_host_consistent,
        "all_split_gates_passed": scientific_passed,
    }
    if not structural:
        classification = "p1_relational_pointer_foundation_structurally_invalid"
    elif not cross_host_consistent:
        classification = "p1_relational_pointer_foundation_cross_host_inconsistent"
    elif not scientific_passed:
        classification = "p1_relational_pointer_foundation_failed"
    else:
        classification = "p1_relational_pointer_foundation_passed"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "terminal-classification",
        "splits": split_results,
        "source": sources[0] if sources else None,
        "gates": gates,
        "passed": classification == "p1_relational_pointer_foundation_passed",
        "classification": classification,
        "authorized_successor": (
            "matched-mlx-selected-prefix-pointer-pilot"
            if classification == "p1_relational_pointer_foundation_passed"
            else None
        ),
        "claim_boundary": (
            "Exact pointer semantics and compact support only; no learned-quality, "
            "gameplay, or 100-point claim."
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_identity": identity,
        "scientific_blake3": canonical_blake3(identity),
        "runtime": {
            "host": socket.gethostname().split(".")[0],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--split", choices=sorted(EXPECTED_SPLIT_COUNTS), required=True)
    audit.add_argument("--factor-cache", type=Path, default=DEFAULT_FACTOR_CACHE)
    audit.add_argument("--r3-cache", type=Path, default=DEFAULT_R3_CACHE)
    audit.add_argument("--d6-metadata", type=Path, default=DEFAULT_D6_METADATA)
    audit.add_argument("--bundle-id", required=True)
    audit.add_argument("--host")
    audit.add_argument("--output", type=Path, required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--report", type=Path, action="append", required=True)
    classify.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "audit":
        report = audit_split(
            split=args.split,
            factor_cache=args.factor_cache,
            r3_cache=args.r3_cache,
            d6_metadata=args.d6_metadata,
            bundle_id=args.bundle_id,
            runtime_host=args.host,
        )
    else:
        report = classify_reports(
            [_read_json(path, f"pointer report {path}") for path in args.report]
        )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
