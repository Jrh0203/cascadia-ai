"""Public-redetermination beam values for observable action afterstates."""

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

PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION = 1
PUBLIC_BEAM_VALUE_FEATURE_SCHEMA = "observable-action-afterstate-v1"
PUBLIC_BEAM_VALUE_TARGET_SCHEMA = "public-redetermined-b16-w2-r8x2-terminal-v1"
PUBLIC_BEAM_VALUE_SHARD_MAGIC = b"CSD2PBV\0"
PUBLIC_BEAM_VALUE_HEADER_SIZE = 128
PUBLIC_BEAM_VALUE_RECORD_SIZE = 96 + ACTION_POSITION_RECORD_SIZE

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s24s")
_RECORD_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "candidate_index",
            "candidate_count",
            "current_base_score",
            "reserved",
            "batch_a_mean",
            "batch_b_mean",
            "batch_a_stddev",
            "batch_b_stddev",
            "public_position_hash",
            "action_hash",
            "input",
        ],
        "formats": [
            "<u8",
            "<u2",
            "<u2",
            "<u2",
            ("u1", (2,)),
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            ("u1", (32,)),
            ("u1", (32,)),
            _ACTION_POSITION_DTYPE,
        ],
        "offsets": [0, 8, 10, 12, 14, 16, 20, 24, 28, 32, 64, 96],
        "itemsize": PUBLIC_BEAM_VALUE_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class PublicBeamValueBatch:
    """One padded batch of complete public-value candidate groups."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    action_features: mx.array
    candidate_mask: mx.array
    target_mean: mx.array
    batch_a_mean: mx.array
    batch_b_mean: mx.array
    batch_a_stddev: mx.array
    batch_b_stddev: mx.array
    current_base_score: mx.array
    immediate_score: mx.array
    group_id: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class PublicBeamValueShard:
    """A validated fixed-width public beam-value shard."""

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
            offset=PUBLIC_BEAM_VALUE_HEADER_SIZE,
            shape=(self.record_count,),
        )

    def groups(self) -> tuple[np.ndarray, ...]:
        records = self.records()
        if len(records) == 0:
            return ()
        boundaries = np.flatnonzero(records["group_id"][1:] != records["group_id"][:-1]) + 1
        groups = tuple(np.split(np.arange(len(records)), boundaries))
        if len(groups) != self.group_count:
            raise DatasetError(f"public beam-value group count mismatch: {self.path}")
        for indices in groups:
            group = records[indices]
            count = len(group)
            hashes = {bytes(value) for value in group["action_hash"]}
            derived_current = group["input"]["action"]["immediate_score"].astype(np.int32)
            derived_current -= np.sum(
                group["input"]["action"]["immediate_deltas"].astype(np.int32),
                axis=-1,
            )
            if (
                np.any(group["candidate_count"] != count)
                or not np.array_equal(group["candidate_index"], np.arange(count))
                or not np.all(group["public_position_hash"] == group["public_position_hash"][0])
                or np.any(group["current_base_score"] != group["current_base_score"][0])
                or np.any(derived_current != group["current_base_score"])
                or len(hashes) != count
                or np.any(~np.isfinite(group["batch_a_mean"]))
                or np.any(~np.isfinite(group["batch_b_mean"]))
                or np.any(~np.isfinite(group["batch_a_stddev"]))
                or np.any(~np.isfinite(group["batch_b_stddev"]))
                or np.any(group["batch_a_stddev"] < 0)
                or np.any(group["batch_b_stddev"] < 0)
                or np.any(group["input"]["action"]["immediate_rank"] == 0)
                or np.any(group["input"]["action"]["replace_three_of_a_kind"] != 0)
                or np.any(group["input"]["action"]["paid_wipe_count"] != 0)
                or np.any(group["input"]["action"]["paid_wipe_total_slots"] != 0)
            ):
                raise DatasetError(f"inconsistent public beam-value group: {self.path}")
        return groups


class PublicBeamValueDataset:
    """Manifest-backed public beam-state value data."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read public beam-value manifest: {error}") from error
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
    ) -> Iterator[PublicBeamValueBatch]:
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
                yield decode_public_beam_value_groups(
                    records,
                    groups[start : start + group_batch_size],
                )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported public beam-value dataset schema")
        if manifest.get("feature_schema") != PUBLIC_BEAM_VALUE_FEATURE_SCHEMA:
            raise DatasetError("unsupported public beam-value feature schema")
        if manifest.get("target_schema") != PUBLIC_BEAM_VALUE_TARGET_SCHEMA:
            raise DatasetError("unsupported public beam-value target schema")
        if manifest.get("record_size") != PUBLIC_BEAM_VALUE_RECORD_SIZE:
            raise DatasetError("unsupported public beam-value record size")
        if manifest.get("action_position_record_size") != ACTION_POSITION_RECORD_SIZE:
            raise DatasetError("unsupported public beam-value action record size")
        teacher = manifest.get("teacher")
        if (
            not isinstance(teacher, dict)
            or teacher.get("final_personal_turns") != 5
            or teacher.get("recorded_personal_turns") != [5, 4, 3, 2]
            or teacher.get("determinizations_per_batch") != 8
            or teacher.get("batches") != 2
            or teacher.get("wildlife_candidates") != 2
            or teacher.get("beam_width") != 16
            or teacher.get("seed_schema") != "public-state-hash-domain-separated-v1"
        ):
            raise DatasetError("public beam-value teacher is not the frozen R8x2 B16 W2 probe")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("public beam-value shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("public beam-value record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("public beam-value group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("public beam-value game total does not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> PublicBeamValueShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:PUBLIC_BEAM_VALUE_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read public beam-value shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"public beam-value shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"public beam-value shard checksum mismatch: {path}")
        if len(header) != PUBLIC_BEAM_VALUE_HEADER_SIZE:
            raise DatasetError(f"truncated public beam-value shard header: {path}")
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
            magic != PUBLIC_BEAM_VALUE_SHARD_MAGIC
            or schema != PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION
            or header_size != PUBLIC_BEAM_VALUE_HEADER_SIZE
            or record_size != PUBLIC_BEAM_VALUE_RECORD_SIZE
            or action_record_size != ACTION_POSITION_RECORD_SIZE
            or feature_hash != blake3.blake3(PUBLIC_BEAM_VALUE_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(PUBLIC_BEAM_VALUE_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible public beam-value shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"public beam-value header disagrees with manifest: {path}")
        expected_size = PUBLIC_BEAM_VALUE_HEADER_SIZE + record_count * PUBLIC_BEAM_VALUE_RECORD_SIZE
        if stat.st_size != expected_size:
            raise DatasetError(f"public beam-value record count disagrees with size: {path}")
        return PublicBeamValueShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )


def decode_public_beam_value_groups(
    records: np.ndarray,
    groups: Sequence[np.ndarray],
) -> PublicBeamValueBatch:
    if not groups:
        raise ValueError("public beam-value batch must contain at least one group")
    group_count = len(groups)
    max_candidates = max(len(group) for group in groups)
    inputs = np.zeros(group_count * max_candidates, dtype=_ACTION_POSITION_DTYPE)
    candidate_mask = np.zeros((group_count, max_candidates), dtype=np.bool_)
    batch_a_mean = np.zeros((group_count, max_candidates), dtype=np.float32)
    batch_b_mean = np.zeros((group_count, max_candidates), dtype=np.float32)
    batch_a_stddev = np.zeros((group_count, max_candidates), dtype=np.float32)
    batch_b_stddev = np.zeros((group_count, max_candidates), dtype=np.float32)
    current_base_score = np.zeros((group_count, max_candidates), dtype=np.float32)
    immediate_score = np.zeros((group_count, max_candidates), dtype=np.float32)
    group_ids = np.zeros(group_count, dtype=np.uint64)

    for group_index, indices in enumerate(groups):
        group = records[indices]
        count = len(group)
        start = group_index * max_candidates
        inputs[start : start + count] = group["input"]
        candidate_mask[group_index, :count] = True
        batch_a_mean[group_index, :count] = group["batch_a_mean"]
        batch_b_mean[group_index, :count] = group["batch_b_mean"]
        batch_a_stddev[group_index, :count] = group["batch_a_stddev"]
        batch_b_stddev[group_index, :count] = group["batch_b_stddev"]
        current_base_score[group_index, :count] = group["current_base_score"]
        immediate_score[group_index, :count] = group["input"]["action"]["immediate_score"]
        group_ids[group_index] = group["group_id"][0]

    decoded, action_features = decode_action_positions(inputs)
    shape_prefix = (group_count, max_candidates)
    return PublicBeamValueBatch(
        board_entities=decoded.board_entities.reshape(*shape_prefix, 4, 23, -1),
        board_mask=decoded.board_mask.reshape(*shape_prefix, 4, 23),
        market_entities=decoded.market_entities.reshape(*shape_prefix, 4, -1),
        market_mask=decoded.market_mask.reshape(*shape_prefix, 4),
        global_features=decoded.global_features.reshape(*shape_prefix, -1),
        action_features=action_features.reshape(*shape_prefix, -1),
        candidate_mask=mx.array(candidate_mask),
        target_mean=mx.array((batch_a_mean + batch_b_mean) / 2.0),
        batch_a_mean=mx.array(batch_a_mean),
        batch_b_mean=mx.array(batch_b_mean),
        batch_a_stddev=mx.array(batch_a_stddev),
        batch_b_stddev=mx.array(batch_b_stddev),
        current_base_score=mx.array(current_base_score),
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
