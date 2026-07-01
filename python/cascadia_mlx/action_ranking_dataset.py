"""Grouped action-delta ranking data for MLX policy distillation."""

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
    MAX_BOARD_TILES,
    Batch,
    DatasetError,
    _mask_bits,
    _one_hot,
    _one_hot_with_none,
    decode_records,
)

ACTION_RANKING_DATASET_SCHEMA_VERSION = 1
ACTION_FEATURE_SCHEMA = "compact-action-delta-v1"
ACTION_RANKING_TARGET_SCHEMA = "search-action-ranking-v1"
ACTION_RANKING_SHARD_MAGIC = b"CSD2ARK\0"
ACTION_RANKING_HEADER_SIZE = 112
ACTION_FEATURE_SIZE = 52
ACTION_POSITION_RECORD_SIZE = 916
ACTION_RANKING_RECORD_SIZE = 972
ACTION_BOARD_ENTITY_DIM = 33
ACTION_DIM = 63

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_ACTION_DTYPE = np.dtype(
    {
        "names": [
            "draft_kind",
            "tile_slot",
            "wildlife_slot",
            "tile_terrain_a",
            "tile_terrain_b",
            "tile_wildlife_mask",
            "tile_keystone",
            "drafted_wildlife",
            "tile_q",
            "tile_r",
            "rotation",
            "wildlife_present",
            "wildlife_q",
            "wildlife_r",
            "replace_three_of_a_kind",
            "paid_wipe_count",
            "paid_wipe_slot_mask",
            "paid_wipe_total_slots",
            "immediate_rank",
            "immediate_score",
            "immediate_deltas",
        ],
        "formats": [
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            "u1",
            "u1",
            "<u2",
            "<u2",
            ("<i2", (11,)),
        ],
        "offsets": [*range(18), 18, 20, 22],
        "itemsize": ACTION_FEATURE_SIZE,
    }
)
_ACTION_POSITION_DTYPE = np.dtype(
    {
        "names": ["position", "action"],
        "formats": [_RECORD_DTYPE, _ACTION_DTYPE],
        "offsets": [0, 864],
        "itemsize": ACTION_POSITION_RECORD_SIZE,
    }
)
_ACTION_RANKING_DTYPE = np.dtype(
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
            "input",
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
            _ACTION_POSITION_DTYPE,
        ],
        "offsets": [0, 8, 10, 12, 14, 16, 20, 24, 56],
        "itemsize": ACTION_RANKING_RECORD_SIZE,
    }
)

_DELTA_SCALES = np.array(
    [23.0, 23.0, 23.0, 23.0, 23.0, 30.0, 28.0, 28.0, 28.0, 40.0, 20.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ActionRankingBatch:
    """One padded batch of complete action-delta candidate groups."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    action_features: mx.array
    candidate_mask: mx.array
    teacher_mean: mx.array
    teacher_stddev: mx.array
    immediate_rank: mx.array
    immediate_score: mx.array
    group_id: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class ActionRankingShard:
    """Validated fixed-width action-ranking shard."""

    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=_ACTION_RANKING_DTYPE,
            offset=ACTION_RANKING_HEADER_SIZE,
            shape=(self.record_count,),
        )

    def groups(self) -> tuple[np.ndarray, ...]:
        records = self.records()
        if len(records) == 0:
            return ()
        boundaries = np.flatnonzero(records["group_id"][1:] != records["group_id"][:-1]) + 1
        groups = tuple(np.split(np.arange(len(records)), boundaries))
        if len(groups) != self.group_count:
            raise DatasetError(f"action-ranking group count mismatch: {self.path}")
        for indices in groups:
            group = records[indices]
            count = len(group)
            if (
                np.any(group["candidate_count"] != count)
                or not np.array_equal(group["candidate_index"], np.arange(count))
                or np.any(group["immediate_rank"] == 0)
                or np.any(group["input"]["action"]["immediate_rank"] != group["immediate_rank"])
                or np.any(group["input"]["action"]["immediate_score"] != group["immediate_score"])
            ):
                raise DatasetError(f"inconsistent action-ranking group: {self.path}")
        return groups


class ActionRankingDataset:
    """Manifest-backed grouped action-delta ranking dataset."""

    DATASET_SCHEMA_VERSION = ACTION_RANKING_DATASET_SCHEMA_VERSION
    TARGET_SCHEMA = ACTION_RANKING_TARGET_SCHEMA
    SHARD_MAGIC = ACTION_RANKING_SHARD_MAGIC
    REQUIRE_SOURCE = True

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "dataset.json"
        try:
            self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read action-ranking manifest: {error}") from error
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
    ) -> Iterator[ActionRankingBatch]:
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
                yield decode_action_ranking_groups(
                    records,
                    groups[start : start + group_batch_size],
                )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != self.DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported action-ranking dataset schema version")
        if manifest.get("feature_schema") != ACTION_FEATURE_SCHEMA:
            raise DatasetError("unsupported action feature schema")
        if manifest.get("position_feature_schema") != FEATURE_SCHEMA:
            raise DatasetError("unsupported action-ranking position feature schema")
        if manifest.get("target_schema") != self.TARGET_SCHEMA:
            raise DatasetError("unsupported action-ranking target schema")
        if manifest.get("record_size") != ACTION_RANKING_RECORD_SIZE:
            raise DatasetError("unsupported action-ranking record size")
        if manifest.get("action_feature_size") != ACTION_FEATURE_SIZE:
            raise DatasetError("unsupported raw action feature size")
        source = manifest.get("source")
        if self.REQUIRE_SOURCE and (
            not isinstance(source, dict) or not source.get("manifest_blake3")
        ):
            raise DatasetError("action-ranking manifest requires source provenance")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("action-ranking manifest shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("action-ranking manifest record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("action-ranking manifest group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("action-ranking manifest game total does not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ActionRankingShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:ACTION_RANKING_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read action-ranking shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"action-ranking shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"action-ranking shard checksum mismatch: {path}")
        if len(header) != ACTION_RANKING_HEADER_SIZE:
            raise DatasetError(f"truncated action-ranking shard header: {path}")
        (
            magic,
            schema,
            header_size,
            record_size,
            action_feature_size,
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
            magic != self.SHARD_MAGIC
            or schema != self.DATASET_SCHEMA_VERSION
            or header_size != ACTION_RANKING_HEADER_SIZE
            or record_size != ACTION_RANKING_RECORD_SIZE
            or action_feature_size != ACTION_FEATURE_SIZE
            or feature_hash != blake3.blake3(ACTION_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(self.TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible action-ranking shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"action-ranking shard header disagrees with manifest: {path}")
        expected_size = ACTION_RANKING_HEADER_SIZE + record_count * ACTION_RANKING_RECORD_SIZE
        if stat.st_size != expected_size:
            raise DatasetError(
                f"action-ranking shard record count does not match file size: {path}"
            )
        return ActionRankingShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )


def decode_action_ranking_groups(
    records: np.ndarray,
    groups: Sequence[np.ndarray],
) -> ActionRankingBatch:
    """Pad complete groups and decode canonical afterstates plus explicit actions."""
    if not groups:
        raise ValueError("action-ranking batch must contain at least one group")
    group_count = len(groups)
    max_candidates = max(len(group) for group in groups)
    inputs = np.zeros(group_count * max_candidates, dtype=_ACTION_POSITION_DTYPE)
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
        inputs[start : start + count] = group["input"]
        candidate_mask[group_index, :count] = True
        teacher_mean[group_index, :count] = group["teacher_mean"]
        teacher_stddev[group_index, :count] = group["teacher_stddev"]
        immediate_rank[group_index, :count] = group["immediate_rank"]
        immediate_score[group_index, :count] = group["immediate_score"]
        group_ids[group_index] = group["group_id"][0]

    decoded, action_features = decode_action_positions(inputs)
    shape_prefix = (group_count, max_candidates)
    return ActionRankingBatch(
        board_entities=decoded.board_entities.reshape(*shape_prefix, 4, 23, -1),
        board_mask=decoded.board_mask.reshape(*shape_prefix, 4, 23),
        market_entities=decoded.market_entities.reshape(*shape_prefix, 4, -1),
        market_mask=decoded.market_mask.reshape(*shape_prefix, 4),
        global_features=decoded.global_features.reshape(*shape_prefix, -1),
        action_features=action_features.reshape(*shape_prefix, -1),
        candidate_mask=mx.array(candidate_mask),
        teacher_mean=mx.array(teacher_mean),
        teacher_stddev=mx.array(teacher_stddev),
        immediate_rank=mx.array(immediate_rank),
        immediate_score=mx.array(immediate_score),
        group_id=mx.array(group_ids.astype(np.int64)),
        game_index=decoded.game_index.reshape(*shape_prefix),
        turn=decoded.turn.reshape(*shape_prefix),
    )


def decode_action_positions(records: np.ndarray) -> tuple[Batch, mx.array]:
    """Decode action-position records and append changed-entity markers."""
    records = np.asarray(records)
    positions = records["position"]
    actions = records["action"]
    decoded = decode_records(positions)
    count = len(records)

    board_raw = positions["board_entities"]
    board_counts = positions["board_counts"].astype(np.int32)
    board_indices = np.arange(MAX_BOARD_TILES)[None, :]
    seat_zero_mask = board_indices < board_counts[:, 0, None]
    board_q = board_raw[:, 0, :, 0].view(np.int8)
    board_r = board_raw[:, 0, :, 1].view(np.int8)
    tile_changed = (
        seat_zero_mask
        & (board_q == actions["tile_q"][:, None])
        & (board_r == actions["tile_r"][:, None])
    )
    wildlife_changed = (
        seat_zero_mask
        & (actions["wildlife_present"][:, None] == 1)
        & (board_q == actions["wildlife_q"][:, None])
        & (board_r == actions["wildlife_r"][:, None])
    )
    markers = np.zeros((count, 4, MAX_BOARD_TILES, 2), dtype=np.float32)
    markers[:, 0, :, 0] = tile_changed
    markers[:, 0, :, 1] = wildlife_changed
    board_entities = mx.concatenate([decoded.board_entities, mx.array(markers)], axis=-1)
    if board_entities.shape[-1] != ACTION_BOARD_ENTITY_DIM:
        raise AssertionError("action-ranking board feature dimension drifted")

    return (
        Batch(
            board_entities=board_entities,
            board_mask=decoded.board_mask,
            market_entities=decoded.market_entities,
            market_mask=decoded.market_mask,
            global_features=decoded.global_features,
            targets=decoded.targets,
            game_index=decoded.game_index,
            turn=decoded.turn,
        ),
        decode_action_features(actions),
    )


def decode_action_features(actions: np.ndarray) -> mx.array:
    """Decode the stable 52-byte explicit-action feature record."""
    actions = np.asarray(actions)
    presence = actions["wildlife_present"].astype(np.float32)[:, None]
    action_features = np.concatenate(
        [
            _one_hot(actions["draft_kind"], 2),
            _one_hot(actions["tile_slot"], 4),
            _one_hot(actions["wildlife_slot"], 4),
            _one_hot(actions["tile_terrain_a"], 5),
            _one_hot_with_none(actions["tile_terrain_b"], 5),
            _mask_bits(actions["tile_wildlife_mask"], 5),
            actions["tile_keystone"].astype(np.float32)[:, None],
            _one_hot(actions["drafted_wildlife"], 5),
            actions["tile_q"].astype(np.float32)[:, None] / 24.0,
            actions["tile_r"].astype(np.float32)[:, None] / 24.0,
            _one_hot(actions["rotation"], 6),
            presence,
            actions["wildlife_q"].astype(np.float32)[:, None] / 24.0 * presence,
            actions["wildlife_r"].astype(np.float32)[:, None] / 24.0 * presence,
            actions["replace_three_of_a_kind"].astype(np.float32)[:, None],
            actions["paid_wipe_count"].astype(np.float32)[:, None] / 20.0,
            _mask_bits(actions["paid_wipe_slot_mask"], 4),
            actions["paid_wipe_total_slots"].astype(np.float32)[:, None] / 80.0,
            actions["immediate_deltas"].astype(np.float32) / _DELTA_SCALES,
            actions["immediate_rank"].astype(np.float32)[:, None] / 32.0,
            actions["immediate_score"].astype(np.float32)[:, None] / 100.0,
        ],
        axis=-1,
    )
    if action_features.shape[-1] != ACTION_DIM:
        raise AssertionError("decoded action feature dimension drifted")
    return mx.array(action_features)


def decode_action_position_bytes(
    payload: bytes | bytearray | memoryview,
    count: int,
) -> tuple[Batch, mx.array]:
    """Decode an inference frame containing exactly ``count`` action-position records."""
    expected = count * ACTION_POSITION_RECORD_SIZE
    if len(payload) != expected:
        raise DatasetError(f"action payload has {len(payload)} bytes, expected {expected}")
    records = np.frombuffer(payload, dtype=_ACTION_POSITION_DTYPE, count=count)
    return decode_action_positions(records)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
