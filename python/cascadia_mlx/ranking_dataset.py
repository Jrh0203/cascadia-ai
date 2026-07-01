"""Grouped search-teacher dataset decoding for MLX policy distillation."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.dataset import (
    _RECORD_DTYPE,
    FEATURE_SCHEMA,
    Batch,
    DatasetError,
    decode_records,
)

RANKING_DATASET_SCHEMA_VERSION = 1
RANKING_TARGET_SCHEMA = "search-ranking-v1"
RANKING_SHARD_MAGIC = b"CSD2RKG\0"
RANKING_HEADER_SIZE = 112
RANKING_RECORD_SIZE = 920

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_RANKING_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "candidate_index",
            "candidate_count",
            "immediate_rank",
            "immediate_score",
            "teacher_mean",
            "teacher_stddev",
            "action_hash",
            "position",
        ],
        "formats": [
            "<u8",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<f4",
            "<f4",
            ("u1", (32,)),
            _RECORD_DTYPE,
        ],
        "offsets": [0, 8, 10, 12, 14, 16, 20, 24, 56],
        "itemsize": RANKING_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class RankingBatch:
    """One padded batch of complete candidate groups."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    candidate_mask: mx.array
    teacher_mean: mx.array
    teacher_stddev: mx.array
    immediate_rank: mx.array
    immediate_score: mx.array
    group_id: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class RankingShard:
    """Validated fixed-width ranking shard."""

    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=_RANKING_DTYPE,
            offset=RANKING_HEADER_SIZE,
            shape=(self.record_count,),
        )

    def groups(self) -> tuple[np.ndarray, ...]:
        records = self.records()
        if len(records) == 0:
            return ()
        boundaries = np.flatnonzero(records["group_id"][1:] != records["group_id"][:-1]) + 1
        groups = tuple(np.split(np.arange(len(records)), boundaries))
        if len(groups) != self.group_count:
            raise DatasetError(f"ranking group count mismatch: {self.path}")
        for indices in groups:
            group = records[indices]
            count = len(group)
            if (
                np.any(group["candidate_count"] != count)
                or not np.array_equal(group["candidate_index"], np.arange(count))
                or np.any(group["immediate_rank"] == 0)
            ):
                raise DatasetError(f"inconsistent ranking group: {self.path}")
        return groups


class RankingDataset:
    """Manifest-backed grouped action-ranking dataset."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "dataset.json"
        try:
            self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read ranking manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def group_count(self) -> int:
        return int(self.manifest["total_groups"])

    @property
    def candidate_count(self) -> int:
        return int(self.manifest["total_records"])

    def batches(
        self,
        group_batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[RankingBatch]:
        """Stream complete decision groups without crossing candidate boundaries."""
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        rng = np.random.default_rng(seed)
        shard_indices = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            shard = self.shards[int(shard_index)]
            records = shard.records()
            groups = list(shard.groups())
            if shuffle:
                rng.shuffle(groups)
            for start in range(0, len(groups), group_batch_size):
                yield decode_ranking_groups(records, groups[start : start + group_batch_size])

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != RANKING_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported ranking dataset schema version")
        if manifest.get("feature_schema") != FEATURE_SCHEMA:
            raise DatasetError("unsupported ranking feature schema")
        if manifest.get("target_schema") != RANKING_TARGET_SCHEMA:
            raise DatasetError("unsupported ranking target schema")
        if manifest.get("record_size") != RANKING_RECORD_SIZE:
            raise DatasetError("unsupported ranking record size")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("ranking manifest shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("ranking manifest record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("ranking manifest group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("ranking manifest game total does not match shards")

    def _load_shard(self, entry: dict[str, Any], verify_checksum: bool) -> RankingShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:RANKING_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read ranking shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"ranking shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"ranking shard checksum mismatch: {path}")
        if len(header) != RANKING_HEADER_SIZE:
            raise DatasetError(f"truncated ranking shard header: {path}")
        (
            magic,
            schema,
            header_size,
            record_size,
            _reserved,
            record_count,
            group_count,
            game_count,
            _split,
            _players,
            _bonuses,
            _reserved_byte,
            first_game_index,
            feature_hash,
            target_hash,
            _reserved_tail,
        ) = _HEADER.unpack(header)
        if (
            magic != RANKING_SHARD_MAGIC
            or schema != RANKING_DATASET_SCHEMA_VERSION
            or header_size != RANKING_HEADER_SIZE
            or record_size != RANKING_RECORD_SIZE
            or feature_hash != blake3.blake3(FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(RANKING_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible ranking shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"ranking shard header disagrees with manifest: {path}")
        expected_size = RANKING_HEADER_SIZE + record_count * RANKING_RECORD_SIZE
        if stat.st_size != expected_size:
            raise DatasetError(f"ranking shard record count does not match file size: {path}")
        return RankingShard(path, record_count, group_count, game_count, first_game_index)


def decode_ranking_groups(
    records: np.ndarray,
    groups: Sequence[np.ndarray],
) -> RankingBatch:
    """Pad complete groups and decode their nested canonical afterstates."""
    if not groups:
        raise ValueError("ranking batch must contain at least one group")
    group_count = len(groups)
    max_candidates = max(len(group) for group in groups)
    positions = np.zeros(group_count * max_candidates, dtype=_RECORD_DTYPE)
    candidate_mask = np.zeros((group_count, max_candidates), dtype=np.bool_)
    teacher_mean = np.zeros((group_count, max_candidates), dtype=np.float32)
    teacher_stddev = np.ones((group_count, max_candidates), dtype=np.float32)
    immediate_rank = np.zeros((group_count, max_candidates), dtype=np.float32)
    immediate_score = np.zeros((group_count, max_candidates), dtype=np.float32)
    group_ids = np.zeros(group_count, dtype=np.uint64)

    for group_index, indices in enumerate(groups):
        group = records[indices]
        count = len(group)
        start = group_index * max_candidates
        positions[start : start + count] = group["position"]
        candidate_mask[group_index, :count] = True
        teacher_mean[group_index, :count] = group["teacher_mean"]
        teacher_stddev[group_index, :count] = group["teacher_stddev"]
        immediate_rank[group_index, :count] = group["immediate_rank"]
        immediate_score[group_index, :count] = group["immediate_score"]
        group_ids[group_index] = group["group_id"][0]

    decoded: Batch = decode_records(positions)
    shape_prefix = (group_count, max_candidates)
    return RankingBatch(
        board_entities=decoded.board_entities.reshape(*shape_prefix, 4, 23, -1),
        board_mask=decoded.board_mask.reshape(*shape_prefix, 4, 23),
        market_entities=decoded.market_entities.reshape(*shape_prefix, 4, -1),
        market_mask=decoded.market_mask.reshape(*shape_prefix, 4),
        global_features=decoded.global_features.reshape(*shape_prefix, -1),
        candidate_mask=mx.array(candidate_mask),
        teacher_mean=mx.array(teacher_mean),
        teacher_stddev=mx.array(teacher_stddev),
        immediate_rank=mx.array(immediate_rank),
        immediate_score=mx.array(immediate_score),
        group_id=mx.array(group_ids.astype(np.int64)),
        game_index=decoded.game_index.reshape(*shape_prefix),
        turn=decoded.turn.reshape(*shape_prefix),
    )


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
