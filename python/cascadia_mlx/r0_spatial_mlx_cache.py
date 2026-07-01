"""Verified MLX cache boundary for the R0 spatial representation tournament."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "r0-spatial-mlx-cache-v1"
EXPERIMENT_ID = "r0-spatial-mlx-tournament-v1"
CORPUS_LOCK_SCHEMA_VERSION = 1
CORPUS_LOCK_CONTRACT = "r0-frozen-60000-position-corpus-v1"
TOKEN_FIELDS = 11
MARKET_FEATURES = 31
GLOBAL_FEATURES = 96
TARGET_DIM = 11
BOARD_SLOTS = 4
MAX_ENTITIES_PER_BOARD = 23
D6_TRANSFORMS = 12
SLOT_SENTINEL = np.uint16(65535)
EXPECTED_SPLIT_RECORDS = {
    "train": 50_000,
    "validation": 10_000,
}
EXPECTED_TOTAL_RECORDS = sum(EXPECTED_SPLIT_RECORDS.values())

ARM_LOCAL_CAPACITY = {
    "exact-entity-control": 0,
    "hex-radius-6-127": 127,
    "hex-radius-5-91": 91,
    "hex-radius-4-61": 61,
    "historical-square-21x21-441": 441,
}
ARM_TOKEN_CAPACITY = {
    "exact-entity-control": 23,
    "hex-radius-6-127": 150,
    "hex-radius-5-91": 114,
    "hex-radius-4-61": 84,
    "historical-square-21x21-441": 464,
}
_DTYPES = {
    "|i1": np.dtype("i1"),
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<f4": np.dtype("<f4"),
    "<u8": np.dtype("<u8"),
}


class R0SpatialMlxCacheError(ValueError):
    """Raised when a cache cannot prove its R0 scientific identity."""


@dataclass(frozen=True)
class R0SpatialMlxBatch:
    """One dense model batch materialized from Rust-authored sparse rows."""

    spatial_tokens: mx.array
    spatial_mask: mx.array
    market_features: mx.array
    market_mask: mx.array
    global_features: mx.array
    targets: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class _TensorSpec:
    path: Path
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    bytes: int
    blake3: str

    def memmap(self) -> np.memmap:
        return np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape)


@dataclass(frozen=True)
class _Split:
    records: int
    tensors: Mapping[str, np.memmap]
    integrity: Mapping[str, int]


class R0SpatialMlxCache:
    """A content-addressed, fail-closed R0 cache with deterministic batching."""

    def __init__(
        self,
        root: str | Path,
        *,
        corpus_lock: str | Path,
        verify_checksums: bool = True,
        verify_padding: bool = True,
    ):
        self.root = Path(root)
        self.manifest_path = self.root / "cache.json"
        self.manifest = _read_json(self.manifest_path, "cache manifest")
        self._validate_manifest_envelope()
        self.arm = str(self.manifest["arm"])
        self.local_capacity = ARM_LOCAL_CAPACITY[self.arm]
        self.token_capacity = ARM_TOKEN_CAPACITY[self.arm]
        self.corpus_lock = _read_json(Path(corpus_lock), "corpus lock")
        self._validate_corpus_lock()
        self.splits = self._load_splits(verify_checksums=verify_checksums)
        if verify_padding:
            for split in ("train", "validation"):
                self._verify_split_integrity(split)

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    @property
    def corpus_lock_id(self) -> str:
        return str(self.manifest["corpus"]["lock_id"])

    @property
    def source_semantic_blake3(self) -> str:
        return str(self.manifest["semantic_integrity"]["source_semantic_blake3"])

    @property
    def d6_semantic_blake3(self) -> str:
        return str(self.manifest["semantic_integrity"]["d6_semantic_blake3"])

    @property
    def target_blake3(self) -> str:
        return str(self.manifest["semantic_integrity"]["target_blake3"])

    @property
    def exporter_executable_blake3(self) -> str:
        return str(self.manifest["exporter"]["executable_blake3"])

    def sample_count(self, split: str) -> int:
        return self._split(split).records

    def batch(
        self,
        split: str,
        indices: Sequence[int] | np.ndarray,
        *,
        transform_ids: Sequence[int] | np.ndarray | None = None,
    ) -> R0SpatialMlxBatch:
        """Materialize one batch; Python only scatters Rust-provided destination slots."""
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("batch indices must be a nonempty one-dimensional sequence")
        source = self._split(split)
        if np.any(selected < 0) or np.any(selected >= source.records):
            raise IndexError("batch index is outside the selected cache split")
        if transform_ids is None:
            transforms = np.zeros(len(selected), dtype=np.int64)
        else:
            transforms = np.asarray(transform_ids, dtype=np.int64)
            if transforms.shape != selected.shape:
                raise ValueError("transform IDs must have one entry per batch index")
            if np.any(transforms < 0) or np.any(transforms >= D6_TRANSFORMS):
                raise ValueError("D6 transform IDs must be in [0, 11]")

        slots = np.asarray(source.tensors["token_slots"][selected, transforms]).copy()
        sparse_features = np.asarray(source.tensors["token_features"][selected, transforms]).copy()
        valid = slots != SLOT_SENTINEL
        dense = np.zeros(
            (len(selected), BOARD_SLOTS, self.token_capacity, TOKEN_FIELDS),
            dtype=np.int8,
        )
        dense_mask = np.zeros(
            (len(selected), BOARD_SLOTS, self.token_capacity),
            dtype=np.bool_,
        )
        batch_rows, boards, entity_rows = np.nonzero(valid)
        destinations = slots[batch_rows, boards, entity_rows].astype(np.intp)
        dense[batch_rows, boards, destinations] = sparse_features[batch_rows, boards, entity_rows]
        dense_mask[batch_rows, boards, destinations] = True

        return R0SpatialMlxBatch(
            spatial_tokens=mx.array(dense.astype(np.int32)),
            spatial_mask=mx.array(dense_mask),
            market_features=mx.array(
                np.asarray(source.tensors["market_features"][selected]).copy()
            ),
            market_mask=mx.array(
                np.asarray(source.tensors["market_mask"][selected], dtype=np.bool_).copy()
            ),
            global_features=mx.array(
                np.asarray(source.tensors["global_features"][selected]).copy()
            ),
            targets=mx.array(np.asarray(source.tensors["targets"][selected]).copy()),
            game_index=mx.array(
                np.asarray(source.tensors["game_index"][selected], dtype=np.int64).copy()
            ),
            turn=mx.array(np.asarray(source.tensors["turn"][selected], dtype=np.int32).copy()),
        )

    def sequential_batches(
        self,
        split: str,
        batch_size: int,
        *,
        transform_id: int = 0,
    ) -> Iterator[R0SpatialMlxBatch]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not 0 <= transform_id < D6_TRANSFORMS:
            raise ValueError("D6 transform ID must be in [0, 11]")
        records = self.sample_count(split)
        for start in range(0, records, batch_size):
            indices = np.arange(start, min(start + batch_size, records), dtype=np.int64)
            yield self.batch(
                split,
                indices,
                transform_ids=np.full(len(indices), transform_id, dtype=np.int64),
            )

    def deterministic_training_batch(
        self,
        *,
        step: int,
        batch_size: int,
        seed: int,
    ) -> R0SpatialMlxBatch:
        """Return the exact training batch for one optimizer step, independent of resume."""
        if step < 0:
            raise ValueError("training step cannot be negative")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        sequence = np.random.SeedSequence([seed, step, 0x52304D4C58])
        rng = np.random.default_rng(sequence)
        indices = rng.integers(
            0,
            self.sample_count("train"),
            size=batch_size,
            dtype=np.int64,
        )
        transforms = rng.integers(0, D6_TRANSFORMS, size=batch_size, dtype=np.int64)
        return self.batch("train", indices, transform_ids=transforms)

    def _validate_manifest_envelope(self) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
        ):
            raise R0SpatialMlxCacheError("unsupported R0 MLX cache schema")
        arm = manifest.get("arm")
        if arm not in ARM_TOKEN_CAPACITY:
            raise R0SpatialMlxCacheError("cache names an unknown R0 arm")
        identity = manifest.get("scientific_identity")
        if not isinstance(identity, dict):
            raise R0SpatialMlxCacheError("cache scientific identity is missing")
        computed = _canonical_blake3(identity)
        if computed != manifest.get("cache_id"):
            raise R0SpatialMlxCacheError("cache content address does not match its identity")
        if self.root.name != computed:
            raise R0SpatialMlxCacheError("cache directory name is not its content address")
        if identity.get("arm") != arm:
            raise R0SpatialMlxCacheError("cache arm and scientific identity disagree")
        exporter = manifest.get("exporter")
        source_provenance = (
            exporter.get("source_provenance") if isinstance(exporter, dict) else None
        )
        if (
            not isinstance(exporter, dict)
            or not isinstance(source_provenance, dict)
            or identity.get("exporter_executable_blake3") != exporter.get("executable_blake3")
            or identity.get("exporter_source_v2_blake3")
            != source_provenance.get("v2_source_blake3")
        ):
            raise R0SpatialMlxCacheError(
                "cache content address does not bind its exporter provenance"
            )
        _require_blake3(
            identity.get("exporter_executable_blake3"),
            "exporter executable blake3",
        )
        _require_blake3(
            identity.get("exporter_source_v2_blake3"),
            "exporter source blake3",
        )
        contract = manifest.get("tensor_contract")
        if not isinstance(contract, dict):
            raise R0SpatialMlxCacheError("cache tensor contract is missing")
        expected_capacity = ARM_TOKEN_CAPACITY[str(arm)]
        expected_local = ARM_LOCAL_CAPACITY[str(arm)]
        if (
            contract.get("board_slots") != BOARD_SLOTS
            or contract.get("max_entities_per_board") != MAX_ENTITIES_PER_BOARD
            or contract.get("spatial_token_capacity") != expected_capacity
            or contract.get("local_capacity") != expected_local
            or contract.get("market_feature_dim") != MARKET_FEATURES
            or contract.get("global_feature_dim") != GLOBAL_FEATURES
            or contract.get("d6_transform_ids") != list(range(D6_TRANSFORMS))
        ):
            raise R0SpatialMlxCacheError("cache tensor shape contract drifted")
        semantic = manifest.get("semantic_integrity")
        if not isinstance(semantic, dict) or not all(
            semantic.get(field) is True
            for field in (
                "identity_round_trip_verified",
                "packed_round_trip_verified",
                "d6_inverse_round_trip_verified",
            )
        ):
            raise R0SpatialMlxCacheError("cache lacks complete Rust semantic proof")
        if semantic.get("packed_round_trip_records") != EXPECTED_TOTAL_RECORDS:
            raise R0SpatialMlxCacheError("cache did not round-trip the full frozen corpus")
        for field in ("source_semantic_blake3", "d6_semantic_blake3", "target_blake3"):
            _require_blake3(semantic.get(field), field)
        overflow = manifest.get("overflow_integrity")
        if not isinstance(overflow, dict) or overflow.get("exact_entities_retained") is not True:
            raise R0SpatialMlxCacheError("cache did not retain exact overflow entities")

    def _validate_corpus_lock(self) -> None:
        lock = self.corpus_lock
        if (
            lock.get("schema_version") != CORPUS_LOCK_SCHEMA_VERSION
            or lock.get("contract_id") != CORPUS_LOCK_CONTRACT
            or not isinstance(lock.get("identity"), dict)
        ):
            raise R0SpatialMlxCacheError("unsupported R0 corpus lock")
        if _canonical_blake3(lock["identity"]) != lock.get("lock_id"):
            raise R0SpatialMlxCacheError("R0 corpus lock hash drifted")
        corpus = self.manifest.get("corpus")
        if (
            not isinstance(corpus, dict)
            or corpus.get("contract_id") != CORPUS_LOCK_CONTRACT
            or corpus.get("lock_id") != lock.get("lock_id")
            or corpus.get("identity") != lock.get("identity")
        ):
            raise R0SpatialMlxCacheError("cache corpus identity does not match the supplied lock")
        identity = lock["identity"]
        if (
            identity.get("total_records") != EXPECTED_TOTAL_RECORDS
            or identity.get("train_records") != EXPECTED_SPLIT_RECORDS["train"]
            or identity.get("validation_records") != EXPECTED_SPLIT_RECORDS["validation"]
            or not isinstance(identity.get("datasets"), list)
            or len(identity["datasets"]) != 8
        ):
            raise R0SpatialMlxCacheError("corpus lock is not the frozen 60,000-row R0 corpus")

    def _load_splits(self, *, verify_checksums: bool) -> dict[str, _Split]:
        raw_splits = self.manifest.get("splits")
        identity_files = self.manifest["scientific_identity"].get("files")
        if not isinstance(raw_splits, dict) or not isinstance(identity_files, dict):
            raise R0SpatialMlxCacheError("cache split manifests are missing")
        loaded: dict[str, _Split] = {}
        for split, records in EXPECTED_SPLIT_RECORDS.items():
            raw = raw_splits.get(split)
            scientific_files = identity_files.get(split)
            if (
                not isinstance(raw, dict)
                or raw.get("records") != records
                or not isinstance(raw.get("files"), dict)
                or raw.get("files") != scientific_files
                or not isinstance(raw.get("integrity"), dict)
            ):
                raise R0SpatialMlxCacheError(f"{split} cache manifest drifted")
            tensors: dict[str, np.memmap] = {}
            expected_shapes = _expected_shapes(records)
            if set(raw["files"]) != set(expected_shapes):
                raise R0SpatialMlxCacheError(f"{split} tensor set drifted")
            for name, expected_shape in expected_shapes.items():
                specification = raw["files"][name]
                tensor = self._tensor_spec(specification, expected_shape)
                if verify_checksums and _checksum(tensor.path) != tensor.blake3:
                    raise R0SpatialMlxCacheError(
                        f"{split} tensor failed BLAKE3 verification: {name}"
                    )
                tensors[name] = tensor.memmap()
            loaded[split] = _Split(
                records=records,
                tensors=tensors,
                integrity=raw["integrity"],
            )
        return loaded

    def _tensor_spec(
        self,
        value: object,
        expected_shape: tuple[int, ...],
    ) -> _TensorSpec:
        if not isinstance(value, dict):
            raise R0SpatialMlxCacheError("tensor specification must be an object")
        dtype_name = value.get("dtype")
        if dtype_name not in _DTYPES:
            raise R0SpatialMlxCacheError("tensor uses an unsupported dtype")
        shape = value.get("shape")
        if shape != list(expected_shape):
            raise R0SpatialMlxCacheError("tensor shape drifted")
        path = self.root / str(value.get("file"))
        if path.parent != self.root or not path.is_file():
            raise R0SpatialMlxCacheError("tensor path escapes or is absent from the cache")
        dtype = _DTYPES[str(dtype_name)]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise R0SpatialMlxCacheError(f"cannot stat cache tensor: {error}") from error
        if value.get("bytes") != expected_bytes or actual_bytes != expected_bytes:
            raise R0SpatialMlxCacheError("tensor byte count does not match shape and dtype")
        digest = value.get("blake3")
        _require_blake3(digest, "tensor blake3")
        return _TensorSpec(
            path=path,
            dtype=dtype,
            shape=expected_shape,
            bytes=expected_bytes,
            blake3=str(digest),
        )

    def _verify_split_integrity(self, split: str) -> None:
        source = self._split(split)
        expected_active = 0
        expected_padding = 0
        identity_overflow_rows = 0
        identity_overflow_positions = 0
        d6_overflow_rows = 0
        d6_overflow_positions = 0
        chunk_size = 512
        for start in range(0, source.records, chunk_size):
            end = min(start + chunk_size, source.records)
            slots = np.asarray(source.tensors["token_slots"][start:end])
            features = np.asarray(source.tensors["token_features"][start:end])
            counts = np.asarray(source.tensors["board_counts"][start:end], dtype=np.int64)
            valid = slots != SLOT_SENTINEL
            if np.any(features[~valid] != 0):
                raise R0SpatialMlxCacheError(f"{split} padding contains nonzero features")
            if np.any(slots[valid] >= self.token_capacity):
                raise R0SpatialMlxCacheError(f"{split} token slot exceeds the arm shape")
            sorted_slots = np.sort(slots, axis=-1)
            duplicates = (sorted_slots[..., 1:] == sorted_slots[..., :-1]) & (
                sorted_slots[..., 1:] != SLOT_SENTINEL
            )
            if np.any(duplicates):
                raise R0SpatialMlxCacheError(f"{split} contains duplicate dense token slots")
            active_per_board = valid.sum(axis=-1)
            if not np.array_equal(
                active_per_board,
                np.broadcast_to(counts[:, None, :], active_per_board.shape),
            ):
                raise R0SpatialMlxCacheError(
                    f"{split} active tokens do not match Rust board counts"
                )
            path_codes = features[..., 4]
            if self.arm == "exact-entity-control":
                if np.any(path_codes[valid] != 1):
                    raise R0SpatialMlxCacheError("exact cache contains local or overflow tokens")
                overflow = np.zeros_like(valid)
            else:
                local = valid & (slots < self.local_capacity)
                overflow = valid & (slots >= self.local_capacity)
                if np.any(path_codes[local] != 2) or np.any(path_codes[overflow] != 3):
                    raise R0SpatialMlxCacheError(
                        f"{split} token path codes disagree with Rust destination slots"
                    )
            expected_active += int(valid.sum())
            expected_padding += int((~valid).sum())
            identity_overflow_rows += int(overflow[:, 0].sum())
            identity_overflow_positions += int(np.any(overflow[:, 0], axis=(1, 2)).sum())
            d6_overflow_rows += int(overflow.sum())
            d6_overflow_positions += int(np.any(overflow, axis=(2, 3)).sum())

        expected = source.integrity
        observed = {
            "exported_active_token_rows": expected_active,
            "exported_padding_token_rows": expected_padding,
            "identity_overflow_entity_rows": identity_overflow_rows,
            "identity_positions_with_overflow": identity_overflow_positions,
            "d6_overflow_entity_rows": d6_overflow_rows,
            "d6_positions_with_overflow": d6_overflow_positions,
        }
        for field, value in observed.items():
            if expected.get(field) != value:
                raise R0SpatialMlxCacheError(f"{split} integrity field {field} does not reconcile")
        if expected.get("source_entity_rows", -1) * D6_TRANSFORMS != expected_active:
            raise R0SpatialMlxCacheError(f"{split} source entity accounting drifted")

    def _split(self, split: str) -> _Split:
        try:
            return self.splits[split]
        except KeyError as error:
            raise ValueError("split must be train or validation") from error


def _expected_shapes(records: int) -> dict[str, tuple[int, ...]]:
    return {
        "token_slots": (records, D6_TRANSFORMS, BOARD_SLOTS, MAX_ENTITIES_PER_BOARD),
        "token_features": (
            records,
            D6_TRANSFORMS,
            BOARD_SLOTS,
            MAX_ENTITIES_PER_BOARD,
            TOKEN_FIELDS,
        ),
        "market_features": (records, 4, MARKET_FEATURES),
        "market_mask": (records, 4),
        "global_features": (records, GLOBAL_FEATURES),
        "targets": (records, TARGET_DIM),
        "game_index": (records,),
        "turn": (records,),
        "board_counts": (records, BOARD_SLOTS),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R0SpatialMlxCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise R0SpatialMlxCacheError(f"{label} must be a JSON object")
    return value


def _canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_blake3(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R0SpatialMlxCacheError(f"{field} must be a lowercase BLAKE3 digest")
