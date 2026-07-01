"""Content-addressed exact-relation context cache for S4 candidate models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GradedOracleDataset,
    GradedOracleGroupHeader,
    GradedOracleGroupRef,
    inspect_graded_oracle_candidate_records,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache
from cascadia_mlx.s4_candidate_context import (
    ANCHOR_LIMIT,
    MISSING_INDEX,
    RELATION_NEIGHBOR_LIMIT,
    CandidateContextIndex,
    build_candidate_context_index,
    verify_candidate_context_index,
)
from cascadia_mlx.s4_candidate_relation_census import RELATIONS

SCHEMA_VERSION = 1
CACHE_SCHEMA = "s4-candidate-context-cache-v1"
EXPERIMENT_ID = "s4-candidate-context-cache-v1"
FOUNDATION_EXPERIMENT_ID = "s4-candidate-relation-foundation-v1"
FOUNDATION_REPORT_ID = (
    "2b977892c9b899d2fb9b38cfeb1b2e10c9a4f778650cf68dbadc78b28a33c7fc"
)
CONTAINER_MAGIC = b"CSD2S4C\0"
CONTAINER_PREFIX = struct.Struct("<8sQ")
ARRAY_ALIGNMENT = 64
EXPECTED_SPLIT_GROUPS = {"train": 560, "validation": 240}
EXPECTED_SPLIT_CANDIDATES = {"validation": 860_203}
SUPPORTED_DTYPES = frozenset({"|u1", "<u2", "<u4", "<u8"})
SPLIT_ARRAY_NAMES = (
    "rows",
    "group_ids",
    "candidate_offsets",
    "selected_indices",
    "action_hashes",
    "anchor_indices",
    "relation_neighbor_indices",
    "relation_neighbor_counts",
    "relation_anchor_sibling_counts",
)


class S4CandidateContextCacheError(ValueError):
    """The S4 context cache or one of its immutable inputs is inconsistent."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(_canonical_json(value)).hexdigest()


def _file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _array_blake3(value: np.ndarray) -> str:
    raw = np.asarray(value).view(np.uint8).reshape(-1)
    digest = blake3.blake3()
    for start in range(0, len(raw), 1 << 20):
        digest.update(memoryview(raw[start : start + (1 << 20)]))
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise S4CandidateContextCacheError(
            f"{label} is unreadable: {path}"
        ) from error
    if not isinstance(value, dict):
        raise S4CandidateContextCacheError(f"{label} is not an object: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _normalize_host(host: str) -> str:
    lowered = host.lower()
    for known in ("john1", "john2", "john3", "john4"):
        if known in lowered:
            return known
    return host


def _align(value: int) -> int:
    return (value + ARRAY_ALIGNMENT - 1) // ARRAY_ALIGNMENT * ARRAY_ALIGNMENT


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    dtype = array.dtype
    if dtype.byteorder == "=" and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
        array = np.asarray(array, dtype=dtype, order="C")
    if dtype.str not in SUPPORTED_DTYPES:
        raise S4CandidateContextCacheError(
            f"unsupported S4 context tensor dtype: {dtype.str}"
        )
    return array


def _array_descriptors(
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], int]:
    if not arrays:
        raise S4CandidateContextCacheError("S4 context container has no tensors")
    canonical: dict[str, np.ndarray] = {}
    descriptors: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(arrays):
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise S4CandidateContextCacheError(
                f"invalid S4 context tensor name: {name}"
            )
        array = _canonical_array(arrays[name])
        canonical[name] = array
        offset = _align(offset)
        descriptors.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "bytes": array.nbytes,
                "blake3": _array_blake3(array),
            }
        )
        offset += array.nbytes
    return descriptors, canonical, offset


@dataclass(frozen=True)
class ContextContainer:
    """One verified deterministic tensor container."""

    path: Path
    header: dict[str, Any]
    arrays: dict[str, np.memmap]

    @property
    def container_id(self) -> str:
        return str(self.header["container_id"])

    @property
    def scientific_identity(self) -> dict[str, Any]:
        value = self.header["scientific_identity"]
        assert isinstance(value, dict)
        return value


def write_context_container(
    path: Path,
    *,
    scientific_identity: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> str:
    """Write a deterministic, checksummed, memory-mappable tensor container."""
    descriptors, canonical, payload_bytes = _array_descriptors(arrays)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "scientific_identity": scientific_identity,
        "arrays": descriptors,
    }
    container_id = _canonical_blake3(identity)
    header = {
        **identity,
        "container_id": container_id,
        "payload_bytes": payload_bytes,
        "array_alignment": ARRAY_ALIGNMENT,
    }
    encoded_header = _canonical_json(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(CONTAINER_PREFIX.pack(CONTAINER_MAGIC, len(encoded_header)))
            handle.write(encoded_header)
            payload_position = 0
            by_name = {entry["name"]: entry for entry in descriptors}
            for name in sorted(canonical):
                descriptor = by_name[name]
                target = int(descriptor["offset"])
                if target < payload_position:
                    raise S4CandidateContextCacheError(
                        "S4 context tensor offsets overlap"
                    )
                handle.write(b"\0" * (target - payload_position))
                canonical[name].tofile(handle)
                payload_position = target + canonical[name].nbytes
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    read_context_container(path, verify_hashes=True)
    return container_id


def read_context_container(
    path: Path,
    *,
    verify_hashes: bool = True,
) -> ContextContainer:
    """Open and validate one deterministic S4 context tensor container."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(CONTAINER_PREFIX.size)
            if len(prefix) != CONTAINER_PREFIX.size:
                raise S4CandidateContextCacheError(
                    f"S4 context container is truncated: {path}"
                )
            magic, header_bytes = CONTAINER_PREFIX.unpack(prefix)
            if magic != CONTAINER_MAGIC or header_bytes <= 0 or header_bytes > 1 << 20:
                raise S4CandidateContextCacheError(
                    f"S4 context container header is invalid: {path}"
                )
            encoded_header = handle.read(header_bytes)
    except OSError as error:
        raise S4CandidateContextCacheError(
            f"S4 context container is unreadable: {path}"
        ) from error
    try:
        header = json.loads(encoded_header)
    except json.JSONDecodeError as error:
        raise S4CandidateContextCacheError(
            f"S4 context container JSON is invalid: {path}"
        ) from error
    if not isinstance(header, dict) or _canonical_json(header) != encoded_header:
        raise S4CandidateContextCacheError(
            f"S4 context container header is not canonical: {path}"
        )
    descriptors = header.get("arrays")
    scientific_identity = header.get("scientific_identity")
    if (
        header.get("schema_version") != SCHEMA_VERSION
        or header.get("cache_schema") != CACHE_SCHEMA
        or header.get("array_alignment") != ARRAY_ALIGNMENT
        or not isinstance(descriptors, list)
        or not isinstance(scientific_identity, dict)
    ):
        raise S4CandidateContextCacheError(
            f"S4 context container envelope is unsupported: {path}"
        )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "scientific_identity": scientific_identity,
        "arrays": descriptors,
    }
    if _canonical_blake3(identity) != header.get("container_id"):
        raise S4CandidateContextCacheError(
            f"S4 context container identity differs: {path}"
        )

    payload_base = CONTAINER_PREFIX.size + len(encoded_header)
    payload_bytes = header.get("payload_bytes")
    if not isinstance(payload_bytes, int) or payload_bytes < 0:
        raise S4CandidateContextCacheError(
            f"S4 context container payload length is invalid: {path}"
        )
    if path.stat().st_size != payload_base + payload_bytes:
        raise S4CandidateContextCacheError(
            f"S4 context container byte length differs: {path}"
        )

    arrays: dict[str, np.memmap] = {}
    prior_end = 0
    names: list[str] = []
    payload = np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        shape=(payload_bytes,),
        offset=payload_base,
    )
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise S4CandidateContextCacheError(
                f"S4 context tensor descriptor is invalid: {path}"
            )
        name = descriptor.get("name")
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offset = descriptor.get("offset")
        byte_count = descriptor.get("bytes")
        if (
            not isinstance(name, str)
            or not isinstance(dtype_name, str)
            or dtype_name not in SUPPORTED_DTYPES
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offset, int)
            or offset < 0
            or offset % ARRAY_ALIGNMENT
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(descriptor.get("blake3"), str)
            or len(descriptor["blake3"]) != 64
        ):
            raise S4CandidateContextCacheError(
                f"S4 context tensor descriptor fields are invalid: {path}"
            )
        dtype = np.dtype(dtype_name)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if (
            name in arrays
            or offset != _align(prior_end)
            or byte_count != expected_bytes
            or offset + byte_count > payload_bytes
        ):
            raise S4CandidateContextCacheError(
                f"S4 context tensor layout is invalid: {name}"
            )
        if offset > prior_end and np.any(payload[prior_end:offset] != 0):
            raise S4CandidateContextCacheError(
                f"S4 context tensor padding differs before: {name}"
            )
        array = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            shape=tuple(shape),
            offset=payload_base + offset,
            order="C",
        )
        if verify_hashes and _array_blake3(array) != descriptor.get("blake3"):
            raise S4CandidateContextCacheError(
                f"S4 context tensor checksum differs: {name}"
            )
        arrays[name] = array
        names.append(name)
        prior_end = offset + byte_count
    if names != sorted(names) or prior_end != payload_bytes:
        raise S4CandidateContextCacheError(
            f"S4 context tensor ordering or trailing payload is invalid: {path}"
        )
    return ContextContainer(path=path, header=header, arrays=arrays)


def _dataset_locations(
    dataset: GradedOracleDataset,
) -> dict[int, tuple[np.memmap, GradedOracleGroupRef, GradedOracleGroupHeader]]:
    locations: dict[
        int,
        tuple[np.memmap, GradedOracleGroupRef, GradedOracleGroupHeader],
    ] = {}
    for raw, ref, header in dataset.raw_group_headers():
        if header.group_id in locations:
            raise S4CandidateContextCacheError(
                f"graded-oracle group ID is duplicated: {header.group_id}"
            )
        locations[header.group_id] = (raw, ref, header)
    return locations


def _selected_rows(total: int, modulus: int, remainder: int) -> np.ndarray:
    if modulus <= 0 or remainder < 0 or remainder >= modulus:
        raise S4CandidateContextCacheError(
            "S4 context row shard modulus/remainder is invalid"
        )
    return np.arange(remainder, total, modulus, dtype=np.uint32)


def _validate_foundation_report(path: Path) -> dict[str, Any]:
    report = _read_json(path, "S4 foundation aggregate")
    validation = report.get("aggregate", {}).get("validation", {})
    context_128 = validation.get("contexts", {}).get("128", {})
    context_256 = validation.get("contexts", {}).get("256", {})
    if (
        report.get("experiment_id") != FOUNDATION_EXPERIMENT_ID
        or report.get("report_id") != FOUNDATION_REPORT_ID
        or report.get("train_groups") != EXPECTED_SPLIT_GROUPS["train"]
        or report.get("validation_groups") != EXPECTED_SPLIT_GROUPS["validation"]
        or context_128.get("union_all_candidates_linked_to_anchor_fraction", 1.0)
        >= 0.95
        or context_256.get("confidence_set_coverage", 0.0) < 0.99
        or context_256.get("mean_retained_r4800_regret", 1.0) >= 0.15
        or context_256.get("union_winner_linked_to_anchor_fraction", 0.0) < 0.98
        or context_256.get("union_all_candidates_linked_to_anchor_fraction", 0.0)
        < 0.95
    ):
        raise S4CandidateContextCacheError(
            "S4 foundation aggregate does not authorize the frozen context"
        )
    return report


def _validate_bundle_manifest(path: Path) -> str:
    manifest = _read_json(path, "S4 context source bundle")
    identity = manifest.get("identity")
    bundle_id = manifest.get("bundle_id")
    if (
        not isinstance(identity, dict)
        or identity.get("experiment_id") != EXPERIMENT_ID
        or not isinstance(bundle_id, str)
        or len(bundle_id) != 64
        or _canonical_blake3(identity) != bundle_id
        or path.parent.name != bundle_id
    ):
        raise S4CandidateContextCacheError(
            "S4 context source bundle identity is invalid"
        )
    return bundle_id


def _authorized_inputs(
    *,
    train_dataset_root: Path,
    validation_dataset_root: Path,
    cache_root: Path,
    s1_cache_root: Path,
    authorization_path: Path,
) -> tuple[
    R3ActionEditMlxCache,
    GradedOracleDataset,
    GradedOracleDataset,
    str,
]:
    authorization = _read_json(authorization_path, "ADR 0150 authorization")
    identity = authorization.get("identity")
    if authorization.get("approved") is not True or not isinstance(identity, dict):
        raise S4CandidateContextCacheError("ADR 0150 authorization is invalid")
    expected_open_data = identity.get("open_data_verification")
    if not isinstance(expected_open_data, dict):
        raise S4CandidateContextCacheError(
            "ADR 0150 open-data proof is absent"
        )
    proof_id = open_data_verification_id(expected_open_data)
    if proof_id != identity.get("open_data_verification_id"):
        raise S4CandidateContextCacheError(
            "ADR 0150 open-data proof digest differs"
        )
    cache = R3ActionEditMlxCache(
        cache_root,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    s1_cache = S1ExactSupplyCache(
        s1_cache_root,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    observed_open_data = open_data_verification_identity(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset_root,
        validation_dataset=validation_dataset_root,
    )
    if observed_open_data != expected_open_data:
        raise S4CandidateContextCacheError(
            "S4 context cache open-data identity differs from authorization"
        )
    train_dataset = GradedOracleDataset(
        train_dataset_root,
        verify_checksums=False,
    )
    validation_dataset = GradedOracleDataset(
        validation_dataset_root,
        verify_checksums=False,
    )
    return cache, train_dataset, validation_dataset, proof_id


def _build_split_shard(
    *,
    split: str,
    cache: R3ActionEditMlxCache,
    dataset: GradedOracleDataset,
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    source = cache.splits[split]
    tensors = source.tensors
    source_offsets = np.asarray(tensors["candidate_offsets"], dtype=np.uint64)
    source_group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
    locations = _dataset_locations(dataset)
    candidate_counts = source_offsets[rows.astype(np.int64) + 1] - source_offsets[
        rows
    ]
    total_candidates = int(np.sum(candidate_counts, dtype=np.uint64))
    group_count = len(rows)

    group_ids = np.empty(group_count, dtype=np.uint64)
    candidate_offsets = np.zeros(group_count + 1, dtype=np.uint64)
    candidate_offsets[1:] = np.cumsum(candidate_counts, dtype=np.uint64)
    selected_indices = np.empty(group_count, dtype=np.uint16)
    action_hashes = np.empty((total_candidates, 32), dtype=np.uint8)
    anchor_indices = np.full(
        (group_count, ANCHOR_LIMIT),
        MISSING_INDEX,
        dtype=np.uint16,
    )
    relation_neighbor_indices = np.full(
        (
            total_candidates,
            len(RELATIONS),
            RELATION_NEIGHBOR_LIMIT,
        ),
        MISSING_INDEX,
        dtype=np.uint16,
    )
    relation_neighbor_counts = np.zeros(
        (total_candidates, len(RELATIONS)),
        dtype=np.uint8,
    )
    relation_anchor_sibling_counts = np.zeros(
        (total_candidates, len(RELATIONS)),
        dtype=np.uint16,
    )

    for local_row, row_value in enumerate(rows):
        row = int(row_value)
        group_id = int(source_group_ids[row])
        if group_id not in locations:
            raise S4CandidateContextCacheError(
                f"{split} dataset is missing group {group_id}"
            )
        raw, ref, _header = locations[group_id]
        full_candidates = inspect_graded_oracle_candidate_records(raw, ref)
        source_start = int(source_offsets[row])
        source_end = int(source_offsets[row + 1])
        source_indices = np.asarray(
            tensors["source_candidate_indices"][source_start:source_end],
            dtype=np.int64,
        )
        candidates = full_candidates[source_indices]
        expected_hashes = np.asarray(
            tensors["action_hashes"][source_start:source_end],
            dtype=np.uint8,
        )
        if not np.array_equal(candidates["action_hash"], expected_hashes):
            raise S4CandidateContextCacheError(
                f"{split} action identity drifted at row {row}"
            )
        selected_source = int(tensors["selected_source_indices"][row])
        selected_matches = np.flatnonzero(source_indices == selected_source)
        if len(selected_matches) != 1:
            raise S4CandidateContextCacheError(
                f"{split} selected action is absent at row {row}"
            )
        context = build_candidate_context_index(
            candidates,
            np.asarray(
                tensors["control_after_hashes"][source_start:source_end],
                dtype=np.uint8,
            ),
        )
        verify_candidate_context_index(
            context,
            candidates,
            np.asarray(
                tensors["control_after_hashes"][source_start:source_end],
                dtype=np.uint8,
            ),
        )

        destination_start = int(candidate_offsets[local_row])
        destination_end = int(candidate_offsets[local_row + 1])
        group_ids[local_row] = group_id
        selected_indices[local_row] = int(selected_matches[0])
        action_hashes[destination_start:destination_end] = expected_hashes
        anchor_indices[local_row] = context.anchor_indices
        relation_neighbor_indices[destination_start:destination_end] = (
            context.relation_neighbor_indices
        )
        relation_neighbor_counts[destination_start:destination_end] = (
            context.relation_neighbor_counts
        )
        relation_anchor_sibling_counts[destination_start:destination_end] = (
            context.relation_anchor_sibling_counts
        )

    return {
        f"{split}/rows": rows.astype(np.uint32, copy=False),
        f"{split}/group_ids": group_ids,
        f"{split}/candidate_offsets": candidate_offsets,
        f"{split}/selected_indices": selected_indices,
        f"{split}/action_hashes": action_hashes,
        f"{split}/anchor_indices": anchor_indices,
        f"{split}/relation_neighbor_indices": relation_neighbor_indices,
        f"{split}/relation_neighbor_counts": relation_neighbor_counts,
        f"{split}/relation_anchor_sibling_counts": (
            relation_anchor_sibling_counts
        ),
    }


def build_context_shard(
    *,
    train_dataset_root: Path,
    validation_dataset_root: Path,
    cache_root: Path,
    s1_cache_root: Path,
    authorization_path: Path,
    foundation_report_path: Path,
    bundle_manifest_path: Path,
    modulus: int,
    remainder: int,
    output: Path,
) -> dict[str, Any]:
    """Build one deterministic modulo shard of exact S4 context tensors."""
    source_bundle_id = _validate_bundle_manifest(bundle_manifest_path)
    foundation = _validate_foundation_report(foundation_report_path)
    cache, train_dataset, validation_dataset, proof_id = _authorized_inputs(
        train_dataset_root=train_dataset_root,
        validation_dataset_root=validation_dataset_root,
        cache_root=cache_root,
        s1_cache_root=s1_cache_root,
        authorization_path=authorization_path,
    )
    arrays: dict[str, np.ndarray] = {}
    row_summary: dict[str, list[int]] = {}
    group_summary: dict[str, str] = {}
    candidate_summary: dict[str, int] = {}
    for split, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        rows = _selected_rows(cache.splits[split].groups, modulus, remainder)
        split_arrays = _build_split_shard(
            split=split,
            cache=cache,
            dataset=dataset,
            rows=rows,
        )
        arrays.update(split_arrays)
        row_summary[split] = rows.astype(int).tolist()
        group_summary[split] = _array_blake3(split_arrays[f"{split}/group_ids"])
        candidate_summary[split] = len(split_arrays[f"{split}/action_hashes"])

    scientific_identity = {
        "schema_version": SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "kind": "shard",
        "source_bundle_id": source_bundle_id,
        "foundation_report_id": foundation["report_id"],
        "open_data_verification_id": proof_id,
        "r3_cache_id": cache.cache_id,
        "anchor_limit": ANCHOR_LIMIT,
        "relation_neighbor_limit": RELATION_NEIGHBOR_LIMIT,
        "relations": list(RELATIONS),
        "row_shard": {"modulus": modulus, "remainder": remainder},
        "rows": row_summary,
        "group_ids_blake3": group_summary,
        "candidates": candidate_summary,
    }
    container_id = write_context_container(
        output,
        scientific_identity=scientific_identity,
        arrays=arrays,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "container_id": container_id,
        "output": str(output),
        "bytes": output.stat().st_size,
        "blake3": _file_blake3(output),
        "host": _normalize_host(socket.gethostname().split(".")[0]),
        "row_shard": {"modulus": modulus, "remainder": remainder},
        "groups": {split: len(rows) for split, rows in row_summary.items()},
        "candidates": candidate_summary,
    }


def _validate_shard_set(
    containers: list[ContextContainer],
) -> tuple[int, dict[str, Any]]:
    if not containers:
        raise S4CandidateContextCacheError(
            "S4 context merge requires shard containers"
        )
    identities = [container.scientific_identity for container in containers]
    if any(
        identity.get("kind") != "shard"
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("foundation_report_id") != FOUNDATION_REPORT_ID
        or identity.get("anchor_limit") != ANCHOR_LIMIT
        or identity.get("relation_neighbor_limit") != RELATION_NEIGHBOR_LIMIT
        or identity.get("relations") != list(RELATIONS)
        for identity in identities
    ):
        raise S4CandidateContextCacheError(
            "S4 context shard scientific contract differs"
        )
    common_fields = (
        "source_bundle_id",
        "foundation_report_id",
        "open_data_verification_id",
        "r3_cache_id",
    )
    for field in common_fields:
        values = {identity.get(field) for identity in identities}
        if len(values) != 1:
            raise S4CandidateContextCacheError(
                f"S4 context shard {field} differs"
            )
    moduli = {
        identity.get("row_shard", {}).get("modulus")
        for identity in identities
    }
    if len(moduli) != 1:
        raise S4CandidateContextCacheError(
            "S4 context shard moduli differ"
        )
    modulus = next(iter(moduli))
    if not isinstance(modulus, int) or modulus <= 0 or len(containers) != modulus:
        raise S4CandidateContextCacheError(
            "S4 context merge requires exactly one shard per remainder"
        )
    remainders = {
        identity.get("row_shard", {}).get("remainder")
        for identity in identities
    }
    if remainders != set(range(modulus)):
        raise S4CandidateContextCacheError(
            "S4 context shard remainders are incomplete"
        )
    common = {field: identities[0][field] for field in common_fields}
    return modulus, common


def _merge_split(
    containers: list[ContextContainer],
    split: str,
) -> dict[str, np.ndarray]:
    expected_groups = EXPECTED_SPLIT_GROUPS[split]
    locations: dict[int, tuple[ContextContainer, int]] = {}
    for container in containers:
        rows = np.asarray(container.arrays[f"{split}/rows"], dtype=np.uint32)
        for local_row, row_value in enumerate(rows):
            row = int(row_value)
            if row in locations:
                raise S4CandidateContextCacheError(
                    f"S4 context {split} row {row} overlaps"
                )
            locations[row] = (container, local_row)
    if set(locations) != set(range(expected_groups)):
        raise S4CandidateContextCacheError(
            f"S4 context {split} row coverage is incomplete"
        )

    counts = np.empty(expected_groups, dtype=np.uint64)
    for row in range(expected_groups):
        container, local_row = locations[row]
        offsets = container.arrays[f"{split}/candidate_offsets"]
        counts[row] = int(offsets[local_row + 1]) - int(offsets[local_row])
    total_candidates = int(np.sum(counts, dtype=np.uint64))
    expected_candidates = EXPECTED_SPLIT_CANDIDATES.get(split)
    if expected_candidates is not None and total_candidates != expected_candidates:
        raise S4CandidateContextCacheError(
            f"S4 context {split} candidate coverage differs"
        )

    rows = np.arange(expected_groups, dtype=np.uint32)
    group_ids = np.empty(expected_groups, dtype=np.uint64)
    candidate_offsets = np.zeros(expected_groups + 1, dtype=np.uint64)
    candidate_offsets[1:] = np.cumsum(counts, dtype=np.uint64)
    selected_indices = np.empty(expected_groups, dtype=np.uint16)
    action_hashes = np.empty((total_candidates, 32), dtype=np.uint8)
    anchor_indices = np.empty(
        (expected_groups, ANCHOR_LIMIT),
        dtype=np.uint16,
    )
    relation_neighbor_indices = np.empty(
        (
            total_candidates,
            len(RELATIONS),
            RELATION_NEIGHBOR_LIMIT,
        ),
        dtype=np.uint16,
    )
    relation_neighbor_counts = np.empty(
        (total_candidates, len(RELATIONS)),
        dtype=np.uint8,
    )
    relation_anchor_sibling_counts = np.empty(
        (total_candidates, len(RELATIONS)),
        dtype=np.uint16,
    )

    for row in range(expected_groups):
        container, local_row = locations[row]
        source_offsets = container.arrays[f"{split}/candidate_offsets"]
        source_start = int(source_offsets[local_row])
        source_end = int(source_offsets[local_row + 1])
        destination_start = int(candidate_offsets[row])
        destination_end = int(candidate_offsets[row + 1])
        group_ids[row] = container.arrays[f"{split}/group_ids"][local_row]
        selected_indices[row] = container.arrays[
            f"{split}/selected_indices"
        ][local_row]
        anchor_indices[row] = container.arrays[
            f"{split}/anchor_indices"
        ][local_row]
        source_slice = slice(source_start, source_end)
        destination_slice = slice(destination_start, destination_end)
        action_hashes[destination_slice] = container.arrays[
            f"{split}/action_hashes"
        ][source_slice]
        relation_neighbor_indices[destination_slice] = container.arrays[
            f"{split}/relation_neighbor_indices"
        ][source_slice]
        relation_neighbor_counts[destination_slice] = container.arrays[
            f"{split}/relation_neighbor_counts"
        ][source_slice]
        relation_anchor_sibling_counts[destination_slice] = container.arrays[
            f"{split}/relation_anchor_sibling_counts"
        ][source_slice]

    return {
        f"{split}/rows": rows,
        f"{split}/group_ids": group_ids,
        f"{split}/candidate_offsets": candidate_offsets,
        f"{split}/selected_indices": selected_indices,
        f"{split}/action_hashes": action_hashes,
        f"{split}/anchor_indices": anchor_indices,
        f"{split}/relation_neighbor_indices": relation_neighbor_indices,
        f"{split}/relation_neighbor_counts": relation_neighbor_counts,
        f"{split}/relation_anchor_sibling_counts": (
            relation_anchor_sibling_counts
        ),
    }


def merge_context_shards(
    shard_paths: list[Path],
    *,
    output_root: Path,
) -> tuple[Path, dict[str, Any], bool]:
    """Merge a complete shard set into one content-addressed cache."""
    containers = [
        read_context_container(path, verify_hashes=True)
        for path in shard_paths
    ]
    modulus, common = _validate_shard_set(containers)
    arrays: dict[str, np.ndarray] = {}
    split_summary: dict[str, Any] = {}
    for split in ("train", "validation"):
        split_arrays = _merge_split(containers, split)
        arrays.update(split_arrays)
        split_summary[split] = {
            "groups": EXPECTED_SPLIT_GROUPS[split],
            "candidates": len(split_arrays[f"{split}/action_hashes"]),
            "group_ids_blake3": _array_blake3(
                split_arrays[f"{split}/group_ids"]
            ),
            "action_hashes_blake3": _array_blake3(
                split_arrays[f"{split}/action_hashes"]
            ),
        }
    scientific_identity = {
        "schema_version": SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "kind": "complete",
        **common,
        "anchor_limit": ANCHOR_LIMIT,
        "relation_neighbor_limit": RELATION_NEIGHBOR_LIMIT,
        "relations": list(RELATIONS),
        "source_shards": modulus,
        "source_shard_ids": sorted(
            container.container_id for container in containers
        ),
        "splits": split_summary,
        "complete_open_corpus": True,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        container_path = temporary / "context.s4ctx"
        cache_id = write_context_container(
            container_path,
            scientific_identity=scientific_identity,
            arrays=arrays,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cache_schema": CACHE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "cache_id": cache_id,
            "scientific_identity": scientific_identity,
            "container": {
                "path": container_path.name,
                "bytes": container_path.stat().st_size,
                "blake3": _file_blake3(container_path),
            },
            "tensor_contract": {
                "anchor_limit": ANCHOR_LIMIT,
                "relation_neighbor_limit": RELATION_NEIGHBOR_LIMIT,
                "relations": list(RELATIONS),
                "all_complete_candidates_are_queries": True,
                "indices_are_group_relative_uint16": True,
                "missing_index": int(MISSING_INDEX),
            },
            "complete_open_corpus": True,
        }
        _write_json_atomic(temporary / "cache.json", manifest)
        destination = output_root / cache_id
        if destination.exists():
            existing = S4CandidateContextCache(
                destination,
                verify_checksums=True,
                verify_semantics=True,
            )
            if existing.cache_id != cache_id:
                raise S4CandidateContextCacheError(
                    "existing S4 context cache identity differs"
                )
            shutil.rmtree(temporary)
            return destination, existing.manifest, True
        os.replace(temporary, destination)
        cache = S4CandidateContextCache(
            destination,
            verify_checksums=True,
            verify_semantics=True,
        )
        return destination, cache.manifest, False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@dataclass(frozen=True)
class CandidateContextGroup:
    """One decision group's exact context arrays."""

    row: int
    group_id: int
    selected_index: int
    action_hashes: np.ndarray
    context: CandidateContextIndex


@dataclass(frozen=True)
class CandidateContextSplit:
    """One complete split in the merged S4 context cache."""

    rows: np.ndarray
    group_ids: np.ndarray
    candidate_offsets: np.ndarray
    selected_indices: np.ndarray
    action_hashes: np.ndarray
    anchor_indices: np.ndarray
    relation_neighbor_indices: np.ndarray
    relation_neighbor_counts: np.ndarray
    relation_anchor_sibling_counts: np.ndarray

    def group(self, row: int) -> CandidateContextGroup:
        if row < 0 or row >= len(self.rows) or int(self.rows[row]) != row:
            raise S4CandidateContextCacheError(
                f"S4 context row is absent: {row}"
            )
        start = int(self.candidate_offsets[row])
        end = int(self.candidate_offsets[row + 1])
        context = CandidateContextIndex(
            anchor_indices=self.anchor_indices[row],
            relation_neighbor_indices=self.relation_neighbor_indices[start:end],
            relation_neighbor_counts=self.relation_neighbor_counts[start:end],
            relation_anchor_sibling_counts=(
                self.relation_anchor_sibling_counts[start:end]
            ),
        )
        return CandidateContextGroup(
            row=row,
            group_id=int(self.group_ids[row]),
            selected_index=int(self.selected_indices[row]),
            action_hashes=self.action_hashes[start:end],
            context=context,
        )


@dataclass(frozen=True)
class CandidateContextBatch:
    """Padded NumPy routing tensors ready for conversion to MLX arrays."""

    rows: np.ndarray
    candidate_counts: np.ndarray
    anchor_candidate_indices: np.ndarray
    anchor_mask: np.ndarray
    relation_neighbor_anchor_slots: np.ndarray
    relation_neighbor_mask: np.ndarray
    relation_anchor_sibling_counts: np.ndarray


class S4CandidateContextCache:
    """Fail-closed loader for the complete S4 exact-relation sidecar."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = True,
        verify_semantics: bool = True,
    ):
        self.root = Path(root)
        self.manifest = _read_json(
            self.root / "cache.json",
            "S4 context cache manifest",
        )
        container_entry = self.manifest.get("container")
        if (
            self.manifest.get("schema_version") != SCHEMA_VERSION
            or self.manifest.get("cache_schema") != CACHE_SCHEMA
            or self.manifest.get("experiment_id") != EXPERIMENT_ID
            or self.manifest.get("complete_open_corpus") is not True
            or not isinstance(container_entry, dict)
        ):
            raise S4CandidateContextCacheError(
                "unsupported S4 context cache envelope"
            )
        container_path = self.root / str(container_entry.get("path", ""))
        if (
            not container_path.is_file()
            or container_path.stat().st_size != container_entry.get("bytes")
            or (
                verify_checksums
                and _file_blake3(container_path) != container_entry.get("blake3")
            )
        ):
            raise S4CandidateContextCacheError(
                "S4 context cache container differs"
            )
        self.container = read_context_container(
            container_path,
            verify_hashes=verify_checksums,
        )
        if (
            self.root.name != self.manifest.get("cache_id")
            or self.container.container_id != self.manifest.get("cache_id")
            or self.container.scientific_identity
            != self.manifest.get("scientific_identity")
        ):
            raise S4CandidateContextCacheError(
                "S4 context cache content address differs"
            )
        self.splits = {
            split: self._load_split(split)
            for split in ("train", "validation")
        }
        if verify_semantics:
            for split in self.splits.values():
                self._verify_split(split)

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    def _load_split(self, split: str) -> CandidateContextSplit:
        arrays = self.container.arrays
        missing = [
            name
            for name in SPLIT_ARRAY_NAMES
            if f"{split}/{name}" not in arrays
        ]
        if missing:
            raise S4CandidateContextCacheError(
                f"S4 context {split} tensors are missing: {missing}"
            )
        return CandidateContextSplit(
            rows=arrays[f"{split}/rows"],
            group_ids=arrays[f"{split}/group_ids"],
            candidate_offsets=arrays[f"{split}/candidate_offsets"],
            selected_indices=arrays[f"{split}/selected_indices"],
            action_hashes=arrays[f"{split}/action_hashes"],
            anchor_indices=arrays[f"{split}/anchor_indices"],
            relation_neighbor_indices=arrays[
                f"{split}/relation_neighbor_indices"
            ],
            relation_neighbor_counts=arrays[
                f"{split}/relation_neighbor_counts"
            ],
            relation_anchor_sibling_counts=arrays[
                f"{split}/relation_anchor_sibling_counts"
            ],
        )

    def _verify_split(self, split: CandidateContextSplit) -> None:
        groups = len(split.rows)
        if (
            split.rows.dtype != np.uint32
            or not np.array_equal(
                split.rows,
                np.arange(groups, dtype=np.uint32),
            )
            or split.group_ids.shape != (groups,)
            or split.candidate_offsets.shape != (groups + 1,)
            or split.candidate_offsets[0] != 0
            or np.any(np.diff(split.candidate_offsets.astype(np.int64)) <= 0)
            or split.selected_indices.shape != (groups,)
            or split.anchor_indices.shape != (groups, ANCHOR_LIMIT)
        ):
            raise S4CandidateContextCacheError(
                "S4 context split group contract differs"
            )
        candidates = int(split.candidate_offsets[-1])
        if (
            split.action_hashes.shape != (candidates, 32)
            or split.relation_neighbor_indices.shape
            != (
                candidates,
                len(RELATIONS),
                RELATION_NEIGHBOR_LIMIT,
            )
            or split.relation_neighbor_counts.shape
            != (candidates, len(RELATIONS))
            or split.relation_anchor_sibling_counts.shape
            != (candidates, len(RELATIONS))
        ):
            raise S4CandidateContextCacheError(
                "S4 context split candidate contract differs"
            )
        for row in range(groups):
            group = split.group(row)
            group.context.validate(len(group.action_hashes))
            if group.selected_index < 0 or group.selected_index >= len(
                group.action_hashes
            ):
                raise S4CandidateContextCacheError(
                    f"S4 context selected index differs at row {row}"
                )

    def bind_action_hashes(
        self,
        split: str,
        row: int,
        action_hashes: np.ndarray,
    ) -> CandidateContextGroup:
        if split not in self.splits:
            raise S4CandidateContextCacheError(
                f"unknown S4 context split: {split}"
            )
        group = self.splits[split].group(row)
        observed = np.asarray(action_hashes, dtype=np.uint8)
        if not np.array_equal(group.action_hashes, observed):
            raise S4CandidateContextCacheError(
                f"S4 context action identity drifted at {split} row {row}"
            )
        return group

    def materialize(
        self,
        split: str,
        rows: np.ndarray,
        *,
        action_hashes: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> CandidateContextBatch:
        """Bind and pad exact context, mapping neighbor candidates to anchors."""
        if split not in self.splits:
            raise S4CandidateContextCacheError(
                f"unknown S4 context split: {split}"
            )
        selected_rows = np.asarray(rows, dtype=np.int64)
        hashes = np.asarray(action_hashes, dtype=np.uint8)
        mask = np.asarray(candidate_mask, dtype=np.bool_)
        if (
            selected_rows.ndim != 1
            or not len(selected_rows)
            or hashes.ndim != 3
            or hashes.shape[0] != len(selected_rows)
            or hashes.shape[2] != 32
            or mask.shape != hashes.shape[:2]
        ):
            raise S4CandidateContextCacheError(
                "S4 context materialization batch contract differs"
            )
        groups, maximum_candidates = mask.shape
        candidate_counts = np.zeros(groups, dtype=np.uint16)
        anchor_candidate_indices = np.zeros(
            (groups, ANCHOR_LIMIT),
            dtype=np.int32,
        )
        anchor_mask = np.zeros(
            (groups, ANCHOR_LIMIT),
            dtype=np.bool_,
        )
        relation_neighbor_anchor_slots = np.zeros(
            (
                groups,
                maximum_candidates,
                len(RELATIONS),
                RELATION_NEIGHBOR_LIMIT,
            ),
            dtype=np.int32,
        )
        relation_neighbor_mask = np.zeros_like(
            relation_neighbor_anchor_slots,
            dtype=np.bool_,
        )
        relation_anchor_sibling_counts = np.zeros(
            (groups, maximum_candidates, len(RELATIONS)),
            dtype=np.uint16,
        )

        for group_index, row_value in enumerate(selected_rows):
            row = int(row_value)
            group = self.splits[split].group(row)
            candidate_count = len(group.action_hashes)
            if candidate_count > maximum_candidates:
                raise S4CandidateContextCacheError(
                    f"S4 context row {row} exceeds the padded candidate width"
                )
            expected_mask = np.zeros(maximum_candidates, dtype=np.bool_)
            expected_mask[:candidate_count] = True
            if (
                not np.array_equal(mask[group_index], expected_mask)
                or not np.array_equal(
                    hashes[group_index, :candidate_count],
                    group.action_hashes,
                )
            ):
                raise S4CandidateContextCacheError(
                    f"S4 context action identity drifted at {split} row {row}"
                )
            candidate_counts[group_index] = candidate_count
            valid_anchors = group.context.anchor_indices != MISSING_INDEX
            anchors = group.context.anchor_indices[valid_anchors].astype(
                np.int32
            )
            anchor_count = len(anchors)
            anchor_candidate_indices[group_index, :anchor_count] = anchors
            anchor_mask[group_index, :anchor_count] = True
            anchor_slots = np.full(
                candidate_count,
                MISSING_INDEX,
                dtype=np.uint16,
            )
            anchor_slots[anchors] = np.arange(
                anchor_count,
                dtype=np.uint16,
            )

            neighbor_counts = group.context.relation_neighbor_counts
            slots = np.arange(
                RELATION_NEIGHBOR_LIMIT,
                dtype=np.uint8,
            )[None, None, :]
            present = slots < neighbor_counts[..., None]
            neighbor_candidates = group.context.relation_neighbor_indices
            safe_neighbors = np.where(present, neighbor_candidates, 0)
            mapped = anchor_slots[safe_neighbors]
            if np.any(mapped[present] == MISSING_INDEX):
                raise S4CandidateContextCacheError(
                    f"S4 context neighbor is not an anchor at {split} row {row}"
                )
            relation_neighbor_anchor_slots[
                group_index,
                :candidate_count,
            ] = np.where(present, mapped, 0).astype(np.int32)
            relation_neighbor_mask[
                group_index,
                :candidate_count,
            ] = present
            relation_anchor_sibling_counts[
                group_index,
                :candidate_count,
            ] = group.context.relation_anchor_sibling_counts

        return CandidateContextBatch(
            rows=selected_rows.astype(np.int32),
            candidate_counts=candidate_counts,
            anchor_candidate_indices=anchor_candidate_indices,
            anchor_mask=anchor_mask,
            relation_neighbor_anchor_slots=relation_neighbor_anchor_slots,
            relation_neighbor_mask=relation_neighbor_mask,
            relation_anchor_sibling_counts=relation_anchor_sibling_counts,
        )


def _selected_relative_indices(
    candidate_offsets: np.ndarray,
    source_candidate_indices: np.ndarray,
    selected_source_indices: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(candidate_offsets, dtype=np.uint64)
    sources = np.asarray(source_candidate_indices, dtype=np.uint16)
    selected = np.asarray(selected_source_indices, dtype=np.uint16)
    if (
        offsets.ndim != 1
        or len(offsets) != len(selected) + 1
        or offsets[0] != 0
        or offsets[-1] != len(sources)
        or np.any(np.diff(offsets.astype(np.int64)) <= 0)
    ):
        raise S4CandidateContextCacheError(
            "R3 selected-index audit tensor contract differs"
        )
    result = np.empty(len(selected), dtype=np.uint16)
    for row, selected_source in enumerate(selected):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        matches = np.flatnonzero(sources[start:end] == selected_source)
        if len(matches) != 1:
            raise S4CandidateContextCacheError(
                f"R3 selected action is not unique at row {row}"
            )
        result[row] = int(matches[0])
    return result


def audit_context_cache_binding(
    *,
    context_cache_root: Path,
    r3_cache_root: Path,
) -> dict[str, Any]:
    """Independently bind every merged context row to the frozen R3 cache."""
    context = S4CandidateContextCache(
        context_cache_root,
        verify_checksums=True,
        verify_semantics=True,
    )
    r3 = R3ActionEditMlxCache(
        r3_cache_root,
        verify_checksums=True,
        verify_semantics=True,
        require_complete=True,
    )
    split_reports: dict[str, Any] = {}
    for split in ("train", "validation"):
        observed = context.splits[split]
        source = r3.splits[split]
        tensors = source.tensors
        source_group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        source_offsets = np.asarray(
            tensors["candidate_offsets"],
            dtype=np.uint64,
        )
        source_hashes = np.asarray(tensors["action_hashes"], dtype=np.uint8)
        source_selected = _selected_relative_indices(
            source_offsets,
            np.asarray(
                tensors["source_candidate_indices"],
                dtype=np.uint16,
            ),
            np.asarray(
                tensors["selected_source_indices"],
                dtype=np.uint16,
            ),
        )
        checks = {
            "group_ids": np.array_equal(observed.group_ids, source_group_ids),
            "candidate_offsets": np.array_equal(
                observed.candidate_offsets,
                source_offsets,
            ),
            "action_hashes": np.array_equal(
                observed.action_hashes,
                source_hashes,
            ),
            "selected_indices": np.array_equal(
                observed.selected_indices,
                source_selected,
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise S4CandidateContextCacheError(
                f"S4 context {split} binding differs: {failed}"
            )
        split_reports[split] = {
            "groups": len(observed.rows),
            "candidates": len(observed.action_hashes),
            "group_ids_blake3": _array_blake3(observed.group_ids),
            "candidate_offsets_blake3": _array_blake3(
                observed.candidate_offsets
            ),
            "action_hashes_blake3": _array_blake3(observed.action_hashes),
            "selected_indices_blake3": _array_blake3(
                observed.selected_indices
            ),
            "checks": checks,
        }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "context_cache_id": context.cache_id,
        "r3_cache_id": r3.cache_id,
        "splits": split_reports,
        "all_bindings_match": True,
    }
    return {
        **identity,
        "audit_id": _canonical_blake3(identity),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--train-dataset", type=Path, required=True)
    shard.add_argument("--validation-dataset", type=Path, required=True)
    shard.add_argument("--cache", type=Path, required=True)
    shard.add_argument("--s1-cache", type=Path, required=True)
    shard.add_argument("--authorization", type=Path, required=True)
    shard.add_argument("--foundation-report", type=Path, required=True)
    shard.add_argument("--bundle-manifest", type=Path, required=True)
    shard.add_argument("--row-modulus", type=int, required=True)
    shard.add_argument("--row-remainder", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--shard", type=Path, action="append", required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--cache", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--context-cache", type=Path, required=True)
    audit.add_argument("--r3-cache", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "shard":
        report = build_context_shard(
            train_dataset_root=args.train_dataset,
            validation_dataset_root=args.validation_dataset,
            cache_root=args.cache,
            s1_cache_root=args.s1_cache,
            authorization_path=args.authorization,
            foundation_report_path=args.foundation_report,
            bundle_manifest_path=args.bundle_manifest,
            modulus=args.row_modulus,
            remainder=args.row_remainder,
            output=args.output,
        )
    elif args.command == "merge":
        path, manifest, reused = merge_context_shards(
            args.shard,
            output_root=args.output_root,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "cache_id": manifest["cache_id"],
            "cache_path": str(path),
            "reused": reused,
            "splits": manifest["scientific_identity"]["splits"],
        }
    elif args.command == "inspect":
        cache = S4CandidateContextCache(
            args.cache,
            verify_checksums=True,
            verify_semantics=True,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "cache_id": cache.cache_id,
            "splits": {
                split: {
                    "groups": len(value.rows),
                    "candidates": len(value.action_hashes),
                }
                for split, value in cache.splits.items()
            },
        }
    else:
        report = audit_context_cache_binding(
            context_cache_root=args.context_cache,
            r3_cache_root=args.r3_cache,
        )
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
