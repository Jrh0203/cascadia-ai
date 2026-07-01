"""Paired anchor/challenger data for confidence-gated policy distillation."""

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

from cascadia_mlx.action_ranking_dataset import (
    _ACTION_POSITION_DTYPE,
    ACTION_POSITION_RECORD_SIZE,
    decode_action_positions,
)
from cascadia_mlx.dataset import DatasetError

CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION = 1
CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA = "paired-observable-action-afterstates-v1"
CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA = "paired-c90-lower-bound-v1"
CONSERVATIVE_ADVANTAGE_SHARD_MAGIC = b"CSD2CAV\0"
CONSERVATIVE_ADVANTAGE_HEADER_SIZE = 128
CONSERVATIVE_ADVANTAGE_RECORD_SIZE = 92 + 2 * ACTION_POSITION_RECORD_SIZE

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s24s")
_RECORD_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "candidate_index",
            "candidate_count",
            "selected",
            "reserved",
            "mean_advantage",
            "advantage_standard_error",
            "lower_bound",
            "anchor_hash",
            "candidate_hash",
            "anchor",
            "candidate",
        ],
        "formats": [
            "<u8",
            "<u2",
            "<u2",
            "u1",
            ("u1", (3,)),
            "<f4",
            "<f4",
            "<f4",
            ("u1", (32,)),
            ("u1", (32,)),
            _ACTION_POSITION_DTYPE,
            _ACTION_POSITION_DTYPE,
        ],
        "offsets": [0, 8, 10, 12, 13, 16, 20, 24, 28, 60, 92, 1008],
        "itemsize": CONSERVATIVE_ADVANTAGE_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class ConservativeAdvantageBatch:
    """One padded batch of complete anchor/challenger decision groups."""

    anchor_board_entities: mx.array
    anchor_board_mask: mx.array
    anchor_market_entities: mx.array
    anchor_market_mask: mx.array
    anchor_global_features: mx.array
    anchor_action_features: mx.array
    candidate_board_entities: mx.array
    candidate_board_mask: mx.array
    candidate_market_entities: mx.array
    candidate_market_mask: mx.array
    candidate_global_features: mx.array
    candidate_action_features: mx.array
    candidate_mask: mx.array
    lower_bound: mx.array
    mean_advantage: mx.array
    advantage_standard_error: mx.array
    selected: mx.array
    group_id: mx.array


@dataclass(frozen=True)
class ConservativeAdvantageShard:
    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=_RECORD_DTYPE,
            offset=CONSERVATIVE_ADVANTAGE_HEADER_SIZE,
            shape=(self.record_count,),
        )

    def groups(self) -> tuple[np.ndarray, ...]:
        records = self.records()
        if len(records) == 0:
            return ()
        boundaries = np.flatnonzero(records["group_id"][1:] != records["group_id"][:-1]) + 1
        groups = tuple(np.split(np.arange(len(records)), boundaries))
        if len(groups) != self.group_count:
            raise DatasetError(f"conservative-advantage group count mismatch: {self.path}")
        for indices in groups:
            group = records[indices]
            count = len(group)
            if (
                np.any(group["candidate_count"] != count)
                or not np.array_equal(group["candidate_index"], np.arange(count))
                or np.sum(group["selected"]) > 1
                or not np.all(group["anchor_hash"] == group["anchor_hash"][0])
                or np.any(np.all(group["anchor_hash"] == group["candidate_hash"], axis=1))
                or np.any(~np.isfinite(group["lower_bound"]))
                or np.any(~np.isfinite(group["mean_advantage"]))
                or np.any(~np.isfinite(group["advantage_standard_error"]))
                or np.any(group["advantage_standard_error"] < 0)
                or np.any((group["selected"] != 0) & (group["lower_bound"] <= 0))
            ):
                raise DatasetError(f"inconsistent conservative-advantage group: {self.path}")
        return groups


class ConservativeAdvantageDataset:
    """Manifest-backed paired c90 lower-bound dataset."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read conservative-advantage manifest: {error}") from error
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
    ) -> Iterator[ConservativeAdvantageBatch]:
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
                yield decode_conservative_advantage_groups(
                    records,
                    groups[start : start + group_batch_size],
                )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported conservative-advantage dataset schema")
        if manifest.get("feature_schema") != CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA:
            raise DatasetError("unsupported conservative-advantage feature schema")
        if manifest.get("target_schema") != CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA:
            raise DatasetError("unsupported conservative-advantage target schema")
        if manifest.get("record_size") != CONSERVATIVE_ADVANTAGE_RECORD_SIZE:
            raise DatasetError("unsupported conservative-advantage record size")
        if manifest.get("action_position_record_size") != ACTION_POSITION_RECORD_SIZE:
            raise DatasetError("unsupported conservative-advantage action record size")
        teacher = manifest.get("teacher")
        if (
            not isinstance(teacher, dict)
            or teacher.get("determinizations") != 8
            or teacher.get("confidence_percent") != 90
        ):
            raise DatasetError("conservative-advantage teacher is not frozen R8 c90")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("conservative-advantage shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("conservative-advantage record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("conservative-advantage group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("conservative-advantage game total does not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ConservativeAdvantageShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:CONSERVATIVE_ADVANTAGE_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(
                f"cannot read conservative-advantage shard {path}: {error}"
            ) from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"conservative-advantage shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"conservative-advantage shard checksum mismatch: {path}")
        if len(header) != CONSERVATIVE_ADVANTAGE_HEADER_SIZE:
            raise DatasetError(f"truncated conservative-advantage shard header: {path}")
        (
            magic,
            schema,
            header_size,
            record_size,
            action_record_size,
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
            magic != CONSERVATIVE_ADVANTAGE_SHARD_MAGIC
            or schema != CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION
            or header_size != CONSERVATIVE_ADVANTAGE_HEADER_SIZE
            or record_size != CONSERVATIVE_ADVANTAGE_RECORD_SIZE
            or action_record_size != ACTION_POSITION_RECORD_SIZE
            or feature_hash
            != blake3.blake3(CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible conservative-advantage shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(
                f"conservative-advantage shard header disagrees with manifest: {path}"
            )
        expected_size = (
            CONSERVATIVE_ADVANTAGE_HEADER_SIZE + record_count * CONSERVATIVE_ADVANTAGE_RECORD_SIZE
        )
        if stat.st_size != expected_size:
            raise DatasetError(
                f"conservative-advantage shard record count does not match size: {path}"
            )
        return ConservativeAdvantageShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )


def decode_conservative_advantage_groups(
    records: np.ndarray,
    groups: Sequence[np.ndarray],
) -> ConservativeAdvantageBatch:
    if not groups:
        raise ValueError("conservative-advantage batch must contain at least one group")
    group_count = len(groups)
    max_candidates = max(len(group) for group in groups)
    flat_count = group_count * max_candidates
    anchors = np.zeros(flat_count, dtype=_ACTION_POSITION_DTYPE)
    candidates = np.zeros(flat_count, dtype=_ACTION_POSITION_DTYPE)
    mask = np.zeros((group_count, max_candidates), dtype=np.bool_)
    lower_bound = np.zeros((group_count, max_candidates), dtype=np.float32)
    mean_advantage = np.zeros((group_count, max_candidates), dtype=np.float32)
    standard_error = np.zeros((group_count, max_candidates), dtype=np.float32)
    selected = np.zeros((group_count, max_candidates), dtype=np.bool_)
    group_ids = np.zeros(group_count, dtype=np.uint64)
    for group_index, indices in enumerate(groups):
        group = records[indices]
        count = len(group)
        start = group_index * max_candidates
        anchors[start : start + count] = group["anchor"]
        candidates[start : start + count] = group["candidate"]
        mask[group_index, :count] = True
        lower_bound[group_index, :count] = group["lower_bound"]
        mean_advantage[group_index, :count] = group["mean_advantage"]
        standard_error[group_index, :count] = group["advantage_standard_error"]
        selected[group_index, :count] = group["selected"] != 0
        group_ids[group_index] = group["group_id"][0]

    anchor_positions, anchor_actions = decode_action_positions(anchors)
    candidate_positions, candidate_actions = decode_action_positions(candidates)
    shape = (group_count, max_candidates)
    return ConservativeAdvantageBatch(
        anchor_board_entities=anchor_positions.board_entities.reshape(*shape, 4, 23, -1),
        anchor_board_mask=anchor_positions.board_mask.reshape(*shape, 4, 23),
        anchor_market_entities=anchor_positions.market_entities.reshape(*shape, 4, -1),
        anchor_market_mask=anchor_positions.market_mask.reshape(*shape, 4),
        anchor_global_features=anchor_positions.global_features.reshape(*shape, -1),
        anchor_action_features=anchor_actions.reshape(*shape, -1),
        candidate_board_entities=candidate_positions.board_entities.reshape(*shape, 4, 23, -1),
        candidate_board_mask=candidate_positions.board_mask.reshape(*shape, 4, 23),
        candidate_market_entities=candidate_positions.market_entities.reshape(*shape, 4, -1),
        candidate_market_mask=candidate_positions.market_mask.reshape(*shape, 4),
        candidate_global_features=candidate_positions.global_features.reshape(*shape, -1),
        candidate_action_features=candidate_actions.reshape(*shape, -1),
        candidate_mask=mx.array(mask),
        lower_bound=mx.array(lower_bound),
        mean_advantage=mx.array(mean_advantage),
        advantage_standard_error=mx.array(standard_error),
        selected=mx.array(selected),
        group_id=mx.array(group_ids.astype(np.int64)),
    )


def decode_conservative_advantage_pair_bytes(
    payload: bytes | bytearray | memoryview,
    count: int,
) -> tuple[Any, mx.array, Any, mx.array]:
    expected = count * 2 * ACTION_POSITION_RECORD_SIZE
    if len(payload) != expected:
        raise DatasetError(f"advantage payload has {len(payload)} bytes, expected {expected}")
    positions = np.frombuffer(payload, dtype=_ACTION_POSITION_DTYPE, count=count * 2)
    anchors = positions[0::2]
    candidates = positions[1::2]
    anchor_positions, anchor_actions = decode_action_positions(anchors)
    candidate_positions, candidate_actions = decode_action_positions(candidates)
    return anchor_positions, anchor_actions, candidate_positions, candidate_actions


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
