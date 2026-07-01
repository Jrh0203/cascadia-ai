"""Fail-closed MLX cache boundary for the exact R2 sparse token substrate."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.d6_contract import D6_CONTRACT

CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "r2-sparse-board-local-mlx-cache-v1"
EXPERIMENT_ID = "r2-sparse-mlx-architecture-tournament-v1"
CORPUS_LOCK_SCHEMA_VERSION = 1
CORPUS_LOCK_CONTRACT = "r2-sparse-mlx-frozen-corpus-v1"
FOUNDATION_EXPERIMENT_ID = "r2-sparse-occupied-frontier-foundation-v1"
FOUNDATION_SCIENTIFIC_BLAKE3 = (
    "186ad8934287ef0a74a166ed00cc9ebe857dcded20faa01a264974e1eb7081e6"
)
FOUNDATION_PUBLIC_POSITION_BLAKE3 = (
    "29836be57c6e0529c06b0b628c455b27f06284fe7a8c333e54024174a7e7f003"
)
FOUNDATION_PACKED_STATE_BLAKE3 = (
    "c181be2126a42b668f500666cccf41573ea079a3f2c34ab7bc3989f690fec789"
)

BOARD_SLOTS = 4
BOARD_TOKEN_CAPACITY = 92
TOKEN_CAPACITY = BOARD_SLOTS * BOARD_TOKEN_CAPACITY
TOKEN_PAYLOAD_WIDTH = 52
TOKEN_FEATURES = 60
BOARD_OWNERSHIP_ENCODING = "relative-seat-one-hot-4"
FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS = 83
FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS = 92
GRAPH_MAX_DEGREE = 24
GRAPH_RELATION_COUNT = 10
MARKET_FEATURES = 31
PLAYER_FEATURES = 23
GLOBAL_FEATURES = 96
TARGET_DIM = 11
D6_TRANSFORMS = 12

TOKEN_TYPE_OCCUPIED = 1
TOKEN_TYPE_FRONTIER = 2
TOKEN_TYPE_COMPONENT = 3
TOKEN_TYPE_MOTIF = 4
TOKEN_TYPE_NAMES = {
    TOKEN_TYPE_OCCUPIED: "occupied",
    TOKEN_TYPE_FRONTIER: "frontier",
    TOKEN_TYPE_COMPONENT: "habitat_component",
    TOKEN_TYPE_MOTIF: "wildlife_motif",
}

EXPECTED_SPLIT_RECORDS = {"train": 50_000, "validation": 10_000}
EXPECTED_TOTAL_RECORDS = sum(EXPECTED_SPLIT_RECORDS.values())
EXPECTED_LAYER_MAXIMA = [91, 107, 69, 79, 340]
EXPECTED_TYPE_TOKEN_TOTALS = [3_090_000, 4_155_914, 2_257_600, 2_365_940]
EXPECTED_ACTIVE_TOKENS = sum(EXPECTED_TYPE_TOKEN_TOTALS)

_DTYPES = {
    "|i1": np.dtype("i1"),
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<u4": np.dtype("<u4"),
    "<u8": np.dtype("<u8"),
    "<f4": np.dtype("<f4"),
}
_COORDINATE_MATRICES = np.asarray(D6_CONTRACT.coordinate_matrices, dtype=np.int16)
_DIRECTION_TABLES = np.asarray(D6_CONTRACT.direction_tables, dtype=np.intp)
_DUAL_ROTATION_TABLES = np.asarray(D6_CONTRACT.dual_tile_rotation_tables, dtype=np.int8)
_SINGLE_ROTATION_TABLES = np.asarray(D6_CONTRACT.single_tile_rotation_tables, dtype=np.int8)


class R2SparseMlxCacheError(ValueError):
    """Raised when an R2 cache cannot prove exact identity and shape."""


@dataclass(frozen=True)
class R2SparseMlxBatch:
    """One exact padded batch shared by every R2 architecture."""

    token_features: mx.array
    token_types: mx.array
    token_mask: mx.array
    graph_neighbors: mx.array
    graph_neighbor_mask: mx.array
    graph_relations: mx.array
    graph_direction_features: mx.array
    market_features: mx.array
    market_mask: mx.array
    player_features: mx.array
    player_mask: mx.array
    global_features: mx.array
    targets: mx.array
    game_index: mx.array
    turn: mx.array
    transform_ids: mx.array


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
    edges: int
    tensors: Mapping[str, np.memmap]
    integrity: Mapping[str, Any]


class R2SparseMlxCache:
    """Content-addressed R2 cache with deterministic batches and exact D6."""

    def __init__(
        self,
        root: str | Path,
        *,
        corpus_lock: str | Path,
        verify_checksums: bool = True,
        verify_semantics: bool = True,
    ):
        self.root = Path(root)
        self.manifest_path = self.root / "cache.json"
        self.manifest = _read_json(self.manifest_path, "cache manifest")
        self._validate_manifest_envelope()
        self.corpus_lock = _read_json(Path(corpus_lock), "corpus lock")
        self._validate_corpus_lock()
        self.splits = self._load_splits(verify_checksums=verify_checksums)
        if verify_semantics:
            for split in EXPECTED_SPLIT_RECORDS:
                self._verify_split_integrity(split)
            self._verify_corpus_integrity()

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    @property
    def corpus_lock_id(self) -> str:
        return str(self.manifest["corpus"]["lock_id"])

    @property
    def exporter_executable_blake3(self) -> str:
        return str(self.manifest["exporter"]["executable_blake3"])

    @property
    def target_blake3(self) -> str:
        return str(self.manifest["semantic_integrity"]["target_blake3"])

    @property
    def identity_semantic_blake3(self) -> str:
        return str(
            self.manifest["semantic_integrity"]["identity_encoded_semantic_blake3"]
        )

    @property
    def d6_semantic_blake3(self) -> str:
        return str(
            self.manifest["semantic_integrity"]["d6_regenerated_semantic_blake3"]
        )

    @property
    def active_token_statistics(self) -> dict[str, dict[str, Any]]:
        return {
            split: {
                "records": source.records,
                "padded_capacity_per_position": TOKEN_CAPACITY,
                "board_slots": BOARD_SLOTS,
                "padded_capacity_per_board": BOARD_TOKEN_CAPACITY,
                "active_tokens_total": int(source.integrity["active_tokens"]),
                "active_tokens_mean": (
                    float(source.integrity["active_tokens"]) / source.records
                ),
                "active_tokens_max": int(source.integrity["max_active_tokens"]),
                "active_tokens_max_per_board": int(
                    source.integrity["max_active_tokens_per_board"]
                ),
                "padding_tokens_total": int(source.integrity["padding_tokens"]),
                "type_tokens": {
                    TOKEN_TYPE_NAMES[index + 1]: {
                        "total": int(total),
                        "mean_per_position": float(total) / source.records,
                        "fraction_of_active": float(total)
                        / max(int(source.integrity["active_tokens"]), 1),
                        "maximum_per_position": int(
                            source.integrity["layer_maxima"][index]
                        ),
                    }
                    for index, total in enumerate(
                        source.integrity["type_token_totals"]
                    )
                },
                "foundation_per_board_p99_active_tokens": (
                    FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
                ),
                "foundation_per_board_max_active_tokens": (
                    FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
                ),
            }
            for split, source in self.splits.items()
        }

    def sample_count(self, split: str) -> int:
        return self._split(split).records

    def batch(
        self,
        split: str,
        indices: Sequence[int] | np.ndarray,
        *,
        transform_ids: Sequence[int] | np.ndarray | None = None,
    ) -> R2SparseMlxBatch:
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
            if np.any((transforms < 0) | (transforms >= D6_TRANSFORMS)):
                raise ValueError("D6 transform IDs must be in [0, 11]")

        token_types = np.asarray(source.tensors["token_types"][selected]).copy()
        token_seats = np.asarray(source.tensors["token_seats"][selected]).copy()
        payload = np.asarray(source.tensors["token_payload"][selected]).copy()
        token_mask = token_types != 0
        flat_types = token_types.reshape(len(selected), TOKEN_CAPACITY)
        flat_payload = payload.reshape(
            len(selected),
            TOKEN_CAPACITY,
            TOKEN_PAYLOAD_WIDTH,
        )
        _transform_payload_in_place(flat_payload, flat_types, transforms)
        token_features = _materialize_token_features(
            token_types,
            token_seats,
            payload,
            token_mask,
        )
        (
            graph_neighbors,
            graph_neighbor_mask,
            graph_relations,
            graph_direction_features,
        ) = self._materialize_graph(source, selected, transforms, token_mask)

        return R2SparseMlxBatch(
            token_features=mx.array(token_features),
            token_types=mx.array(token_types.astype(np.int32)),
            token_mask=mx.array(token_mask),
            graph_neighbors=mx.array(graph_neighbors),
            graph_neighbor_mask=mx.array(graph_neighbor_mask),
            graph_relations=mx.array(graph_relations),
            graph_direction_features=mx.array(graph_direction_features),
            market_features=mx.array(
                np.asarray(source.tensors["market_features"][selected]).copy()
            ),
            market_mask=mx.array(
                np.asarray(
                    source.tensors["market_mask"][selected],
                    dtype=np.bool_,
                ).copy()
            ),
            player_features=mx.array(
                np.asarray(source.tensors["player_features"][selected]).copy()
            ),
            player_mask=mx.array(
                np.asarray(
                    source.tensors["player_mask"][selected],
                    dtype=np.bool_,
                ).copy()
            ),
            global_features=mx.array(
                np.asarray(source.tensors["global_features"][selected]).copy()
            ),
            targets=mx.array(np.asarray(source.tensors["targets"][selected]).copy()),
            game_index=mx.array(
                np.asarray(
                    source.tensors["game_index"][selected],
                    dtype=np.int64,
                ).copy()
            ),
            turn=mx.array(
                np.asarray(
                    source.tensors["turn"][selected],
                    dtype=np.int32,
                ).copy()
            ),
            transform_ids=mx.array(transforms.astype(np.int32)),
        )

    def sequential_batches(
        self,
        split: str,
        batch_size: int,
        *,
        transform_id: int = 0,
    ) -> Iterator[R2SparseMlxBatch]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not 0 <= transform_id < D6_TRANSFORMS:
            raise ValueError("D6 transform ID must be in [0, 11]")
        for start in range(0, self.sample_count(split), batch_size):
            indices = np.arange(
                start,
                min(start + batch_size, self.sample_count(split)),
                dtype=np.int64,
            )
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
    ) -> R2SparseMlxBatch:
        if step < 0:
            raise ValueError("training step cannot be negative")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        sequence = np.random.SeedSequence([seed, step, 0x5232535041525345])
        rng = np.random.default_rng(sequence)
        indices = rng.integers(
            0,
            self.sample_count("train"),
            size=batch_size,
            dtype=np.int64,
        )
        transforms = rng.integers(
            0,
            D6_TRANSFORMS,
            size=batch_size,
            dtype=np.int64,
        )
        return self.batch("train", indices, transform_ids=transforms)

    def _materialize_graph(
        self,
        source: _Split,
        selected: np.ndarray,
        transforms: np.ndarray,
        token_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = len(selected)
        neighbors = np.zeros(
            (
                batch_size,
                BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY,
                GRAPH_MAX_DEGREE,
            ),
            dtype=np.int32,
        )
        neighbor_mask = np.zeros_like(neighbors, dtype=np.bool_)
        relations = np.zeros_like(neighbors, dtype=np.int32)
        direction_bits = np.zeros_like(neighbors, dtype=np.uint8)
        record_offsets = source.tensors["graph_record_offsets"]
        token_offsets = source.tensors["graph_token_offsets"]
        targets = source.tensors["graph_targets"]
        raw_relations = source.tensors["graph_relations"]
        raw_direction_bits = source.tensors["graph_direction_bits"]

        source_token_ids = np.arange(TOKEN_CAPACITY, dtype=np.int32)
        for batch_row, record_index in enumerate(selected):
            edge_start = int(record_offsets[record_index])
            edge_end = int(record_offsets[record_index + 1])
            local_offsets = np.asarray(token_offsets[record_index], dtype=np.int64)
            if local_offsets[0] != 0 or local_offsets[-1] != edge_end - edge_start:
                raise R2SparseMlxCacheError("graph record and token offsets disagree")
            degrees = np.diff(local_offsets)
            if np.any((degrees < 0) | (degrees > GRAPH_MAX_DEGREE)):
                raise R2SparseMlxCacheError("graph degree exceeds the frozen bound")
            edge_count = edge_end - edge_start
            if edge_count == 0:
                continue

            # The relation topology is authored once by Rust. This expands the
            # frozen CSR rows into the fixed MLX batch shape without deriving
            # neighborhoods or relation semantics in Python.
            source_tokens = np.repeat(source_token_ids, degrees)
            edge_ranks = np.arange(edge_count, dtype=np.int64) - np.repeat(
                local_offsets[:-1],
                degrees,
            )
            boards = source_tokens // BOARD_TOKEN_CAPACITY
            local_sources = source_tokens % BOARD_TOKEN_CAPACITY
            row_targets = np.asarray(
                targets[edge_start:edge_end],
                dtype=np.int32,
            )
            target_boards = row_targets // BOARD_TOKEN_CAPACITY
            local_targets = row_targets % BOARD_TOKEN_CAPACITY
            if (
                len(source_tokens) != edge_count
                or np.any(edge_ranks < 0)
                or np.any(edge_ranks >= GRAPH_MAX_DEGREE)
                or np.any(target_boards != boards)
                or np.any(
                    ~token_mask[batch_row, boards, local_sources]
                )
                or np.any(
                    ~token_mask[batch_row, target_boards, local_targets]
                )
            ):
                raise R2SparseMlxCacheError(
                    "cached graph crosses boards or references padding"
                )
            output_index = (
                batch_row,
                boards,
                local_sources,
                edge_ranks,
            )
            neighbors[output_index] = local_targets
            relations[output_index] = np.asarray(
                raw_relations[edge_start:edge_end],
                dtype=np.int32,
            )
            direction_bits[output_index] = np.asarray(
                raw_direction_bits[edge_start:edge_end],
                dtype=np.uint8,
            )
            neighbor_mask[output_index] = True

        transformed_direction_bits = _transform_direction_bits(
            direction_bits,
            transforms,
        )
        direction_features = (
            (transformed_direction_bits[..., None] >> np.arange(6, dtype=np.uint8))
            & 1
        ).astype(np.float32)
        direction_features *= neighbor_mask[..., None]
        return neighbors, neighbor_mask, relations, direction_features

    def _validate_manifest_envelope(self) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
        ):
            raise R2SparseMlxCacheError("unsupported R2 MLX cache schema")
        identity = manifest.get("scientific_identity")
        if not isinstance(identity, dict):
            raise R2SparseMlxCacheError("cache scientific identity is missing")
        computed = _canonical_blake3(identity)
        if computed != manifest.get("cache_id"):
            raise R2SparseMlxCacheError("cache content address does not match its identity")
        if self.root.name != computed:
            raise R2SparseMlxCacheError("cache directory name is not its content address")
        contract = manifest.get("tensor_contract")
        if not isinstance(contract, dict):
            raise R2SparseMlxCacheError("cache tensor contract is missing")
        d6 = contract.get("d6")
        if (
            contract.get("token_layout") != "board-major-4x92"
            or contract.get("board_slots") != BOARD_SLOTS
            or contract.get("board_token_capacity") != BOARD_TOKEN_CAPACITY
            or contract.get("token_capacity") != TOKEN_CAPACITY
            or contract.get("token_payload_width") != TOKEN_PAYLOAD_WIDTH
            or contract.get("board_ownership_encoding")
            != BOARD_OWNERSHIP_ENCODING
            or contract.get("foundation_per_board_p99_active_tokens")
            != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
            or contract.get("foundation_per_board_max_active_tokens")
            != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
            or contract.get("board_local_type_order")
            != [
                "occupied",
                "frontier",
                "habitat_component",
                "wildlife_motif",
            ]
            or contract.get("foundation_type_token_totals")
            != EXPECTED_TYPE_TOKEN_TOTALS
            or contract.get("foundation_active_tokens") != EXPECTED_ACTIVE_TOKENS
            or contract.get("graph_max_degree") != GRAPH_MAX_DEGREE
            or contract.get("graph_relation_count") != GRAPH_RELATION_COUNT
            or contract.get("market_feature_dim") != MARKET_FEATURES
            or contract.get("player_feature_dim") != PLAYER_FEATURES
            or contract.get("global_feature_dim") != GLOBAL_FEATURES
            or contract.get("target_dim") != TARGET_DIM
            or contract.get("truncation") != "forbidden"
            or not isinstance(d6, dict)
            or d6.get("contract_id") != D6_CONTRACT.contract_id
            or d6.get("scientific_blake3") != D6_CONTRACT.scientific_blake3
            or d6.get("coordinate_matrices")
            != [[list(row) for row in matrix] for matrix in D6_CONTRACT.coordinate_matrices]
            or d6.get("direction_tables")
            != [list(row) for row in D6_CONTRACT.direction_tables]
            or d6.get("dual_tile_rotation_tables")
            != [list(row) for row in D6_CONTRACT.dual_tile_rotation_tables]
            or d6.get("single_tile_rotation_tables")
            != [list(row) for row in D6_CONTRACT.single_tile_rotation_tables]
        ):
            raise R2SparseMlxCacheError("cache tensor or D6 contract drifted")
        semantic = manifest.get("semantic_integrity")
        if not isinstance(semantic, dict) or not all(
            semantic.get(field) is True
            for field in (
                "exact_public_reconstruction_verified",
                "canonical_packed_round_trip_verified",
                "exact_no_truncation_verified",
                "padding_zero_verified",
                "graph_degree_bound_verified",
                "board_local_layout_verified",
                "derived_tokens_cached_after_regeneration",
            )
        ):
            raise R2SparseMlxCacheError("cache lacks complete semantic proof")
        if semantic.get("test_or_final_data_opened") is not False:
            raise R2SparseMlxCacheError("cache opened prohibited test or final data")
        for field in (
            "identity_encoded_semantic_blake3",
            "d6_regenerated_semantic_blake3",
            "public_position_blake3",
            "packed_state_blake3",
            "target_blake3",
        ):
            _require_blake3(semantic.get(field), field)
        if (
            semantic.get("public_position_blake3")
            != FOUNDATION_PUBLIC_POSITION_BLAKE3
            or semantic.get("packed_state_blake3") != FOUNDATION_PACKED_STATE_BLAKE3
            or semantic.get("d6_transform_inverse_checks")
            != EXPECTED_TOTAL_RECORDS * D6_TRANSFORMS
            or semantic.get("type_token_totals") != EXPECTED_TYPE_TOKEN_TOTALS
            or semantic.get("active_tokens") != EXPECTED_ACTIVE_TOKENS
            or semantic.get("per_board_p99_active_tokens")
            != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
            or semantic.get("per_board_max_active_tokens")
            != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
        ):
            raise R2SparseMlxCacheError("cache semantic stream differs from the foundation")
        exporter = manifest.get("exporter")
        if not isinstance(exporter, dict):
            raise R2SparseMlxCacheError("cache exporter identity is missing")
        _require_blake3(exporter.get("executable_blake3"), "exporter executable blake3")
        if identity.get("exporter_executable_blake3") != exporter.get(
            "executable_blake3"
        ):
            raise R2SparseMlxCacheError("cache identity does not bind its exporter")

    def _validate_corpus_lock(self) -> None:
        lock = self.corpus_lock
        if (
            lock.get("schema_version") != CORPUS_LOCK_SCHEMA_VERSION
            or lock.get("contract_id") != CORPUS_LOCK_CONTRACT
            or not isinstance(lock.get("identity"), dict)
            or _canonical_blake3(lock["identity"]) != lock.get("lock_id")
        ):
            raise R2SparseMlxCacheError("unsupported R2 MLX corpus lock")
        identity = lock["identity"]
        if (
            identity.get("foundation_experiment_id") != FOUNDATION_EXPERIMENT_ID
            or identity.get("foundation_scientific_blake3")
            != FOUNDATION_SCIENTIFIC_BLAKE3
            or identity.get("foundation_public_position_blake3")
            != FOUNDATION_PUBLIC_POSITION_BLAKE3
            or identity.get("foundation_packed_state_blake3")
            != FOUNDATION_PACKED_STATE_BLAKE3
            or identity.get("total_records") != EXPECTED_TOTAL_RECORDS
            or identity.get("train_records") != EXPECTED_SPLIT_RECORDS["train"]
            or identity.get("validation_records")
            != EXPECTED_SPLIT_RECORDS["validation"]
            or identity.get("layer_maxima") != EXPECTED_LAYER_MAXIMA
            or identity.get("type_token_totals") != EXPECTED_TYPE_TOKEN_TOTALS
            or identity.get("active_tokens") != EXPECTED_ACTIVE_TOKENS
            or identity.get("per_board_p99_active_tokens")
            != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
            or identity.get("per_board_max_active_tokens")
            != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
            or not isinstance(identity.get("datasets"), list)
            or len(identity["datasets"]) != 8
        ):
            raise R2SparseMlxCacheError("corpus lock is not the frozen R2 foundation")
        if self.manifest.get("corpus") != lock:
            raise R2SparseMlxCacheError("cache corpus identity differs from the supplied lock")

    def _load_splits(self, *, verify_checksums: bool) -> dict[str, _Split]:
        raw_splits = self.manifest.get("splits")
        identity_files = self.manifest["scientific_identity"].get("files")
        if not isinstance(raw_splits, dict) or not isinstance(identity_files, dict):
            raise R2SparseMlxCacheError("cache split manifests are missing")
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
                raise R2SparseMlxCacheError(f"{split} cache manifest drifted")
            edges = raw["integrity"].get("graph_edges")
            if not isinstance(edges, int) or isinstance(edges, bool) or edges < 0:
                raise R2SparseMlxCacheError(f"{split} graph edge count is invalid")
            expected_shapes = _expected_shapes(records, edges)
            if set(raw["files"]) != set(expected_shapes):
                raise R2SparseMlxCacheError(f"{split} tensor set drifted")
            tensors = {}
            for name, expected_shape in expected_shapes.items():
                specification = raw["files"][name]
                tensor = self._tensor_spec(specification, expected_shape)
                if verify_checksums and _checksum(tensor.path) != tensor.blake3:
                    raise R2SparseMlxCacheError(
                        f"{split} tensor failed BLAKE3 verification: {name}"
                    )
                tensors[name] = tensor.memmap()
            loaded[split] = _Split(
                records=records,
                edges=edges,
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
            raise R2SparseMlxCacheError("tensor specification must be an object")
        dtype_name = value.get("dtype")
        if dtype_name not in _DTYPES:
            raise R2SparseMlxCacheError("tensor uses an unsupported dtype")
        if value.get("shape") != list(expected_shape):
            raise R2SparseMlxCacheError("tensor shape drifted")
        path = self.root / str(value.get("file"))
        if path.parent != self.root or not path.is_file():
            raise R2SparseMlxCacheError("tensor path escapes or is absent from the cache")
        dtype = _DTYPES[str(dtype_name)]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        if value.get("bytes") != expected_bytes or path.stat().st_size != expected_bytes:
            raise R2SparseMlxCacheError("tensor byte count does not match shape and dtype")
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
        observed_active = 0
        observed_padding = 0
        observed_max_active = 0
        observed_max_active_per_board = 0
        observed_layer_maxima = np.zeros(4, dtype=np.int64)
        observed_type_totals = np.zeros(4, dtype=np.int64)
        chunk_size = 512
        for start in range(0, source.records, chunk_size):
            end = min(start + chunk_size, source.records)
            types = np.asarray(source.tensors["token_types"][start:end])
            seats = np.asarray(source.tensors["token_seats"][start:end])
            payload = np.asarray(source.tensors["token_payload"][start:end])
            counts = np.asarray(
                source.tensors["board_type_counts"][start:end],
                dtype=np.int64,
            )
            mask = types != 0
            expected_mask, expected_types = _layout_from_board_type_counts(counts)
            if not np.array_equal(mask, expected_mask):
                raise R2SparseMlxCacheError(f"{split} layer counts do not match token types")
            if not np.array_equal(types, expected_types):
                raise R2SparseMlxCacheError(
                    f"{split} board-local type ordering is noncanonical"
                )
            expected_seats = np.broadcast_to(
                np.arange(BOARD_SLOTS, dtype=np.uint8)[None, :, None],
                seats.shape,
            )
            if np.any(seats[mask] != expected_seats[mask]):
                raise R2SparseMlxCacheError(
                    f"{split} active token board ownership is invalid"
                )
            if (
                np.any(types[~mask] != 0)
                or np.any(seats[~mask] != 0)
                or np.any(payload[~mask] != 0)
            ):
                raise R2SparseMlxCacheError(f"{split} padding contains nonzero data")
            active_per_row = mask.sum(axis=2)
            active_per_position = active_per_row.sum(axis=1)
            if np.any(active_per_row != counts.sum(axis=2)):
                raise R2SparseMlxCacheError(f"{split} token accounting drifted")
            observed_active += int(mask.sum())
            observed_padding += int((~mask).sum())
            observed_max_active = max(
                observed_max_active,
                int(active_per_position.max()),
            )
            observed_max_active_per_board = max(
                observed_max_active_per_board,
                int(active_per_row.max()),
            )
            position_type_counts = counts.sum(axis=1)
            observed_layer_maxima = np.maximum(
                observed_layer_maxima,
                position_type_counts.max(axis=0),
            )
            observed_type_totals += position_type_counts.sum(axis=0)

        record_offsets = np.asarray(source.tensors["graph_record_offsets"], dtype=np.uint64)
        token_offsets = np.asarray(source.tensors["graph_token_offsets"], dtype=np.uint32)
        if (
            record_offsets[0] != 0
            or record_offsets[-1] != source.edges
            or np.any(record_offsets[1:] < record_offsets[:-1])
            or np.any(token_offsets[:, 0] != 0)
        ):
            raise R2SparseMlxCacheError(f"{split} graph offsets are noncanonical")
        record_edge_counts = np.diff(record_offsets).astype(np.uint64)
        if np.any(token_offsets[:, -1].astype(np.uint64) != record_edge_counts):
            raise R2SparseMlxCacheError(f"{split} graph offset levels disagree")
        degrees = np.diff(token_offsets, axis=1)
        if np.any(degrees > GRAPH_MAX_DEGREE):
            raise R2SparseMlxCacheError(f"{split} graph degree exceeds the hard bound")
        all_counts = np.asarray(
            source.tensors["board_type_counts"],
            dtype=np.int64,
        )
        all_mask, _ = _layout_from_board_type_counts(all_counts)
        flat_mask = all_mask.reshape(source.records, TOKEN_CAPACITY)
        if np.any(degrees[~flat_mask] != 0):
            raise R2SparseMlxCacheError(f"{split} padding token has graph edges")
        targets = np.asarray(source.tensors["graph_targets"], dtype=np.int64)
        relations = np.asarray(source.tensors["graph_relations"], dtype=np.int64)
        direction_bits = np.asarray(
            source.tensors["graph_direction_bits"],
            dtype=np.uint8,
        )
        if (
            np.any(targets < 0)
            or np.any(targets >= TOKEN_CAPACITY)
            or np.any(relations < 1)
            or np.any(relations >= GRAPH_RELATION_COUNT)
            or np.any(direction_bits & np.uint8(0xC0))
        ):
            raise R2SparseMlxCacheError(f"{split} graph edge value is invalid")
        for record_index in range(source.records):
            start = int(record_offsets[record_index])
            end = int(record_offsets[record_index + 1])
            if start == end:
                continue
            record_targets = targets[start:end]
            source_tokens = np.repeat(
                np.arange(TOKEN_CAPACITY, dtype=np.int64),
                degrees[record_index],
            )
            if (
                len(source_tokens) != len(record_targets)
                or np.any(~flat_mask[record_index, record_targets])
                or np.any(
                    source_tokens // BOARD_TOKEN_CAPACITY
                    != record_targets // BOARD_TOKEN_CAPACITY
                )
            ):
                raise R2SparseMlxCacheError(
                    f"{split} graph crosses boards or targets padding"
                )

        expected = source.integrity
        observed = {
            "active_tokens": observed_active,
            "padding_tokens": observed_padding,
            "graph_edges": source.edges,
            "max_active_tokens": observed_max_active,
            "max_active_tokens_per_board": observed_max_active_per_board,
            "max_graph_degree": int(degrees.max(initial=0)),
            "layer_maxima": observed_layer_maxima.tolist(),
            "type_token_totals": observed_type_totals.tolist(),
        }
        for field, value in observed.items():
            if expected.get(field) != value:
                raise R2SparseMlxCacheError(
                    f"{split} integrity field {field} does not reconcile"
                )

    def _verify_corpus_integrity(self) -> None:
        type_totals = [
            sum(
                int(source.integrity["type_token_totals"][index])
                for source in self.splits.values()
            )
            for index in range(4)
        ]
        active = sum(
            int(source.integrity["active_tokens"])
            for source in self.splits.values()
        )
        max_per_board = max(
            int(source.integrity["max_active_tokens_per_board"])
            for source in self.splits.values()
        )
        if (
            type_totals != EXPECTED_TYPE_TOKEN_TOTALS
            or active != EXPECTED_ACTIVE_TOKENS
            or max_per_board != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
        ):
            raise R2SparseMlxCacheError(
                "cache type totals or board-local census differ from ADR 0145"
            )

    def _split(self, split: str) -> _Split:
        try:
            return self.splits[split]
        except KeyError as error:
            raise ValueError("split must be train or validation") from error


def _transform_payload_in_place(
    payload: np.ndarray,
    token_types: np.ndarray,
    transform_ids: np.ndarray,
) -> None:
    for row, transform_id in enumerate(transform_ids):
        matrix = _COORDINATE_MATRICES[transform_id]
        direction_table = _DIRECTION_TABLES[transform_id]

        occupied = np.flatnonzero(token_types[row] == TOKEN_TYPE_OCCUPIED)
        if occupied.size:
            transformed = _transform_coordinates(
                payload[row, occupied],
                matrix,
                0,
                1,
            )
            payload[row, occupied, 0] = transformed[:, 0]
            payload[row, occupied, 1] = transformed[:, 1]
            dual = payload[row, occupied, 3] != 5
            rotations = payload[row, occupied, 4].astype(np.intp)
            payload[row, occupied, 4] = np.where(
                dual,
                _DUAL_ROTATION_TABLES[transform_id, rotations],
                _SINGLE_ROTATION_TABLES[transform_id, rotations],
            )
            payload[row, occupied, 5:11] = _permute_direction_values(
                payload[row, occupied, 5:11],
                direction_table,
            )

        frontier = np.flatnonzero(token_types[row] == TOKEN_TYPE_FRONTIER)
        if frontier.size:
            transformed = _transform_coordinates(
                payload[row, frontier],
                matrix,
                0,
                1,
            )
            payload[row, frontier, 0] = transformed[:, 0]
            payload[row, frontier, 1] = transformed[:, 1]
            presence = _transform_direction_bits(
                payload[row, frontier, 2].astype(np.uint8),
                np.full(frontier.size, transform_id, dtype=np.int64),
            )
            payload[row, frontier, 2] = presence.astype(np.int8)
            payload[row, frontier, 3:9] = _permute_direction_values(
                payload[row, frontier, 3:9],
                direction_table,
            )
            payload[row, frontier, 15] = _opposite_pair_bits(presence).astype(np.int8)
            touch_counts = payload[row, frontier, 16].astype(np.intp)
            for local_row, touch_count in enumerate(touch_counts):
                for touch in range(touch_count):
                    slot = 20 + touch * 4
                    bits = payload[row, frontier[local_row], slot].astype(np.uint8)
                    payload[row, frontier[local_row], slot] = np.int8(
                        _transform_one_bitset(int(bits), direction_table)
                    )

        components = np.flatnonzero(token_types[row] == TOKEN_TYPE_COMPONENT)
        for token in components:
            members = int(payload[row, token, 2])
            if members:
                coordinates = payload[row, token, 6 : 6 + members * 2].reshape(members, 2)
                transformed = coordinates.astype(np.int16) @ matrix.T
                order = np.lexsort((transformed[:, 1], transformed[:, 0]))
                payload[row, token, 6 : 6 + members * 2] = transformed[order].astype(
                    np.int8
                ).reshape(-1)

        motifs = np.flatnonzero(token_types[row] == TOKEN_TYPE_MOTIF)
        if motifs.size:
            transformed = _transform_coordinates(
                payload[row, motifs],
                matrix,
                0,
                1,
            )
            payload[row, motifs, 0] = transformed[:, 0]
            payload[row, motifs, 1] = transformed[:, 1]
            payload[row, motifs, 3:9] = _permute_direction_values(
                payload[row, motifs, 3:9],
                direction_table,
            )
            transformed_bits = _transform_direction_bits(
                payload[row, motifs, 14].astype(np.uint8),
                np.full(motifs.size, transform_id, dtype=np.int64),
            )
            payload[row, motifs, 14] = transformed_bits.astype(np.int8)


def _transform_coordinates(
    rows: np.ndarray,
    matrix: np.ndarray,
    q_slot: int,
    r_slot: int,
) -> np.ndarray:
    coordinates = rows[:, [q_slot, r_slot]].astype(np.int16)
    transformed = coordinates @ matrix.T
    if np.any((transformed < -128) | (transformed > 127)):
        raise R2SparseMlxCacheError("D6 coordinate transform exceeds the i8 cache domain")
    return transformed.astype(np.int8)


def _permute_direction_values(values: np.ndarray, direction_table: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[:, direction_table] = values
    return result


def _transform_direction_bits(
    bitsets: np.ndarray,
    transform_ids: np.ndarray,
) -> np.ndarray:
    bitsets = np.asarray(bitsets, dtype=np.uint8)
    transforms = np.asarray(transform_ids, dtype=np.int64)
    if bitsets.shape[0] != transforms.shape[0]:
        raise ValueError("bitset transform IDs must match the leading batch dimension")
    result = np.zeros_like(bitsets, dtype=np.uint8)
    for row, transform_id in enumerate(transforms):
        table = _DIRECTION_TABLES[transform_id]
        flattened = bitsets[row].reshape(-1)
        transformed = np.fromiter(
            (_transform_one_bitset(int(value), table) for value in flattened),
            dtype=np.uint8,
            count=flattened.size,
        )
        result[row] = transformed.reshape(bitsets[row].shape)
    return result


def _transform_one_bitset(value: int, direction_table: np.ndarray) -> int:
    transformed = 0
    for source_direction in range(6):
        if value & (1 << source_direction):
            transformed |= 1 << int(direction_table[source_direction])
    return transformed


def _opposite_pair_bits(presence: np.ndarray) -> np.ndarray:
    result = np.zeros_like(presence, dtype=np.uint8)
    for pair in range(3):
        present = ((presence >> pair) & 1) & ((presence >> (pair + 3)) & 1)
        result |= present.astype(np.uint8) << pair
    return result


def _materialize_token_features(
    token_types: np.ndarray,
    token_seats: np.ndarray,
    payload: np.ndarray,
    token_mask: np.ndarray,
) -> np.ndarray:
    shape = token_types.shape
    flat_types = token_types.reshape(-1)
    flat_seats = token_seats.reshape(-1)
    flat_payload = payload.reshape(-1, TOKEN_PAYLOAD_WIDTH)
    flat_mask = token_mask.reshape(-1)
    features = np.zeros((flat_types.size, TOKEN_FEATURES), dtype=np.float32)
    active = np.flatnonzero(flat_mask)
    features[active, flat_types[active].astype(np.intp) - 1] = 1.0
    features[active, 4 + flat_seats[active].astype(np.intp)] = 1.0
    features[:, 8:] = flat_payload.astype(np.float32) / 64.0
    features *= flat_mask[:, None]
    return features.reshape(*shape, TOKEN_FEATURES)


def _layout_from_board_type_counts(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 3 or counts.shape[1:] != (BOARD_SLOTS, 4):
        raise ValueError("board type counts must have shape [records, 4, 4]")
    if np.any(counts < 0) or np.any(counts.sum(axis=2) > BOARD_TOKEN_CAPACITY):
        raise R2SparseMlxCacheError("board type counts exceed the exact 4x92 layout")
    positions = np.arange(BOARD_TOKEN_CAPACITY)[None, None, :]
    mask = np.zeros(
        (counts.shape[0], BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        dtype=np.bool_,
    )
    token_types = np.zeros_like(mask, dtype=np.uint8)
    cursor = np.zeros((counts.shape[0], BOARD_SLOTS), dtype=np.int64)
    for type_index in range(4):
        end = cursor + counts[:, :, type_index]
        selected = (positions >= cursor[:, :, None]) & (positions < end[:, :, None])
        mask |= selected
        token_types[selected] = type_index + 1
        cursor = end
    return mask, token_types


def _expected_shapes(records: int, edges: int) -> dict[str, tuple[int, ...]]:
    return {
        "token_types": (records, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "token_seats": (records, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "token_payload": (
            records,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            TOKEN_PAYLOAD_WIDTH,
        ),
        "board_type_counts": (records, BOARD_SLOTS, 4),
        "graph_record_offsets": (records + 1,),
        "graph_token_offsets": (records, TOKEN_CAPACITY + 1),
        "graph_targets": (edges,),
        "graph_relations": (edges,),
        "graph_direction_bits": (edges,),
        "market_features": (records, 4, MARKET_FEATURES),
        "market_mask": (records, 4),
        "player_features": (records, BOARD_SLOTS, PLAYER_FEATURES),
        "player_mask": (records, BOARD_SLOTS),
        "global_features": (records, GLOBAL_FEATURES),
        "targets": (records, TARGET_DIM),
        "game_index": (records,),
        "turn": (records,),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R2SparseMlxCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise R2SparseMlxCacheError(f"{label} must be a JSON object")
    return value


def _canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
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
        raise R2SparseMlxCacheError(f"{field} must be a lowercase BLAKE3 digest")


# Public primitives for experiments that consume the accepted exact-R2 cache
# without duplicating its normalization or D6 transformation semantics.
materialize_token_features = _materialize_token_features
transform_token_payload_in_place = _transform_payload_in_place
