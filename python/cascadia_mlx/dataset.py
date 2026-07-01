"""Streaming decoder for the canonical Cascadia v2 dataset format."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

DATASET_SCHEMA_VERSION = 1
FEATURE_SCHEMA = "compact-entity-v2"
TARGET_SCHEMA = "base-score-components-v1"
SHARD_MAGIC = b"CSD2REC\0"
SHARD_HEADER_SIZE = 80
RECORD_SIZE = 864
BOARD_SLOTS = 4
MAX_BOARD_TILES = 23
ENTITY_DIM = 31
GLOBAL_DIM = 96
TARGET_DIM = 11

_HEADER = struct.Struct("<8sHHHHIIQBBBB5s7s32s")
POSITION_RECORD_DTYPE = np.dtype(
    {
        "names": [
            "game_index",
            "turn",
            "active_seat",
            "player_count",
            "total_turns",
            "board_counts",
            "nature_tokens",
            "scoring_cards",
            "habitat_bonuses",
            "wildlife_counts",
            "habitat_sizes",
            "board_entities",
            "market_entities",
            "targets",
        ],
        "formats": [
            "<u8",
            "u1",
            "u1",
            "u1",
            "u1",
            ("u1", (4,)),
            ("u1", (4,)),
            ("u1", (5,)),
            "u1",
            ("u1", (4, 5)),
            ("u1", (4, 5)),
            ("u1", (4, 23, 8)),
            ("u1", (4, 8)),
            ("<u2", (11,)),
        ],
        "offsets": [0, 8, 9, 10, 11, 12, 16, 20, 25, 32, 52, 72, 808, 840],
        "itemsize": RECORD_SIZE,
    }
)

# Kept for older dataset adapters that imported the private name before the
# canonical position dtype became a shared public contract.
_RECORD_DTYPE = POSITION_RECORD_DTYPE


class DatasetError(ValueError):
    """Raised when a dataset is incompatible or fails integrity checks."""


@dataclass(frozen=True)
class Batch:
    """One decoded model batch, already resident as MLX arrays."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    targets: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class Shard:
    """Validated shard metadata."""

    path: Path
    record_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        """Memory-map the fixed-width records without copying the shard."""
        return np.memmap(
            self.path,
            mode="r",
            dtype=_RECORD_DTYPE,
            offset=SHARD_HEADER_SIZE,
            shape=(self.record_count,),
        )


class Dataset:
    """A manifest plus lazily memory-mapped, checksum-verified shards."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "dataset.json"
        try:
            self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read dataset manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def sample_count(self) -> int:
        return int(self.manifest["total_records"])

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[Batch]:
        """Stream decoded batches one shard at a time."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        shard_indices = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            records = self.shards[int(shard_index)].records()
            indices = np.arange(len(records))
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                yield decode_records(records[indices[start : start + batch_size]])

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported dataset schema version")
        if manifest.get("feature_schema") != FEATURE_SCHEMA:
            raise DatasetError("unsupported feature schema")
        if manifest.get("target_schema") != TARGET_SCHEMA:
            raise DatasetError("unsupported target schema")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("manifest shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("manifest record total does not match its shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("manifest game total does not match its shards")

    def _load_shard(self, entry: dict[str, Any], verify_checksum: bool) -> Shard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
        except OSError as error:
            raise DatasetError(f"cannot stat shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"shard checksum mismatch: {path}")
        try:
            header = path.read_bytes()[:SHARD_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read shard {path}: {error}") from error
        if len(header) != SHARD_HEADER_SIZE:
            raise DatasetError(f"truncated shard header: {path}")
        (
            magic,
            schema,
            header_size,
            record_size,
            target_dim,
            record_count,
            game_count,
            first_game_index,
            _split,
            _strategy,
            _players,
            _bonuses,
            _cards,
            _reserved,
            feature_hash,
        ) = _HEADER.unpack(header)
        if magic != SHARD_MAGIC:
            raise DatasetError(f"invalid shard magic: {path}")
        if (
            schema != DATASET_SCHEMA_VERSION
            or header_size != SHARD_HEADER_SIZE
            or record_size != RECORD_SIZE
            or target_dim != TARGET_DIM
            or feature_hash != blake3.blake3(FEATURE_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"shard header disagrees with manifest: {path}")
        expected_size = SHARD_HEADER_SIZE + record_count * RECORD_SIZE
        if stat.st_size != expected_size:
            raise DatasetError(f"shard record count does not match file size: {path}")
        return Shard(path, record_count, game_count, first_game_index)


def decode_records(records: np.ndarray) -> Batch:
    """Vectorize compact records into the model's stable entity schema."""
    records = np.asarray(records)
    board_raw = records["board_entities"]
    market_raw = records["market_entities"]
    batch_size = len(records)

    board_counts = records["board_counts"].astype(np.int32)
    tile_indices = np.arange(MAX_BOARD_TILES)[None, None, :]
    board_mask = tile_indices < board_counts[:, :, None]

    board_q = board_raw[..., 0].view(np.int8).astype(np.float32)[..., None] / 24.0
    board_r = board_raw[..., 1].view(np.int8).astype(np.float32)[..., None] / 24.0
    board_features = np.concatenate(
        [
            board_q,
            board_r,
            _one_hot(board_raw[..., 2], 5),
            _one_hot_with_none(board_raw[..., 3], 5),
            _one_hot(board_raw[..., 4], 6),
            _mask_bits(board_raw[..., 5], 5),
            _one_hot_with_none(board_raw[..., 6], 5),
            board_raw[..., 7, None].astype(np.float32),
        ],
        axis=-1,
    )
    board_features *= board_mask[..., None]

    market_features, market_mask = decode_market_entities(market_raw)

    turns = records["turn"].astype(np.float32)
    total_turns = records["total_turns"].astype(np.float32)
    phase = (turns / np.maximum(total_turns, 1.0))[:, None]
    remaining = ((total_turns - turns) / np.maximum(total_turns, 1.0))[:, None]
    market_wildlife = market_raw[..., 3]
    diversity = np.array(
        [len(set(int(value) for value in row if value < 5)) / 4.0 for row in market_wildlife],
        dtype=np.float32,
    )[:, None]
    global_features = np.concatenate(
        [
            phase,
            remaining,
            _one_hot(records["player_count"] - 1, 4),
            records["nature_tokens"].astype(np.float32) / 20.0,
            board_counts.astype(np.float32) / 23.0,
            records["wildlife_counts"].astype(np.float32).reshape(batch_size, -1) / 20.0,
            records["habitat_sizes"].astype(np.float32).reshape(batch_size, -1) / 23.0,
            _one_hot(market_wildlife, 5).reshape(batch_size, -1),
            _one_hot(records["scoring_cards"], 4).reshape(batch_size, -1),
            records["habitat_bonuses"].astype(np.float32)[:, None],
            diversity,
        ],
        axis=-1,
    )

    if board_features.shape[-1] != ENTITY_DIM:
        raise AssertionError("board feature dimension drifted")
    if market_features.shape[-1] != ENTITY_DIM:
        raise AssertionError("market feature dimension drifted")
    if global_features.shape[-1] != GLOBAL_DIM:
        raise AssertionError("global feature dimension drifted")

    return Batch(
        board_entities=mx.array(board_features),
        board_mask=mx.array(board_mask),
        market_entities=mx.array(market_features),
        market_mask=mx.array(market_mask),
        global_features=mx.array(global_features),
        targets=mx.array(records["targets"].astype(np.float32)),
        game_index=mx.array(records["game_index"].astype(np.int64)),
        turn=mx.array(records["turn"].astype(np.int32)),
    )


def decode_market_entities(market_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode compact market rows with any leading batch dimensions."""
    market_raw = np.asarray(market_raw)
    market_mask = (market_raw[..., 0] < 5) | (market_raw[..., 3] < 5)
    prefix = market_raw.shape[:-1]
    zeros = np.zeros((*prefix, 2), dtype=np.float32)
    rotation_zeros = np.zeros((*prefix, 6), dtype=np.float32)
    market_features = np.concatenate(
        [
            zeros,
            _one_hot(market_raw[..., 0], 5),
            _one_hot_with_none(market_raw[..., 1], 5),
            rotation_zeros,
            _mask_bits(market_raw[..., 2], 5),
            _one_hot_with_none(market_raw[..., 3], 5),
            market_raw[..., 4, None].astype(np.float32),
        ],
        axis=-1,
    )
    market_features *= market_mask[..., None]
    if market_features.shape[-1] != ENTITY_DIM:
        raise AssertionError("market feature dimension drifted")
    return market_features, market_mask


def decode_record_bytes(payload: bytes | bytearray | memoryview, count: int) -> Batch:
    """Decode an inference frame containing exactly ``count`` compact records."""
    expected = count * RECORD_SIZE
    if len(payload) != expected:
        raise DatasetError(f"record payload has {len(payload)} bytes, expected {expected}")
    records = np.frombuffer(payload, dtype=_RECORD_DTYPE, count=count)
    return decode_records(records)


def _one_hot(values: np.ndarray, classes: int) -> np.ndarray:
    valid = values < classes
    clipped = np.where(valid, values, 0)
    return np.eye(classes, dtype=np.float32)[clipped] * valid[..., None]


def _one_hot_with_none(values: np.ndarray, classes: int) -> np.ndarray:
    mapped = np.where(values < classes, values, classes)
    return np.eye(classes + 1, dtype=np.float32)[mapped]


def _mask_bits(values: np.ndarray, bits: int) -> np.ndarray:
    shifts = np.arange(bits, dtype=np.uint8)
    return ((values[..., None] >> shifts) & 1).astype(np.float32)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
