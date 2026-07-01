"""Streaming decoder for exact-MLX rollout-return datasets."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.dataset import DatasetError

ROLLOUT_VALUE_DATASET_SCHEMA_VERSION = 1
ROLLOUT_VALUE_FEATURE_SCHEMA = "legacy-mid-v4opp-sparse-u16-v1"
ROLLOUT_VALUE_TARGET_SCHEMA = "terminal-base-score-to-go-v1"
ROLLOUT_VALUE_SHARD_MAGIC = b"CSD2NNV\0"
ROLLOUT_VALUE_HEADER_SIZE = 160
ROLLOUT_VALUE_RECORD_PREFIX_SIZE = 40
ROLLOUT_VALUE_FEATURE_COUNT = 11_231

TRAJECTORY_KIND = 0
ROOT_ESTIMATE_KIND = 1

_HEADER = struct.Struct("<8sHHHHIIII8sQ32s32s32s16s")
_RECORD_PREFIX = struct.Struct("<BBBBHHIQQfff")
_SPLIT_CODES = {"train": 0, "validation": 1, "test": 2, "final": 3}


@dataclass(frozen=True)
class RolloutValueBatch:
    feature_indices: mx.array
    feature_mask: mx.array
    target_remaining: mx.array
    immediate_score: mx.array
    game_index: np.ndarray
    decision_index: np.ndarray
    personal_turn: np.ndarray
    selected: np.ndarray
    rollout_seed: np.ndarray
    target_stddev: np.ndarray
    samples: np.ndarray

    @property
    def size(self) -> int:
        return int(self.game_index.shape[0])

    def exact_csr(self) -> tuple[mx.array, mx.array]:
        mask = np.asarray(self.feature_mask, dtype=np.bool_)
        padded = np.asarray(self.feature_indices, dtype=np.int32)
        counts = mask.sum(axis=1, dtype=np.int64)
        offsets = np.zeros(self.size + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:], dtype=np.int64)
        indices = padded[mask].astype(np.int32, copy=False)
        return mx.array(offsets), mx.array(indices)


@dataclass(frozen=True)
class RolloutValueRootBatch:
    feature_indices: mx.array
    feature_mask: mx.array
    candidate_mask: mx.array
    target_remaining: mx.array
    immediate_score: mx.array
    selected: mx.array
    target_stddev: mx.array
    samples: mx.array
    game_index: np.ndarray
    decision_index: np.ndarray
    personal_turn: np.ndarray

    @property
    def group_count(self) -> int:
        return int(self.game_index.shape[0])


@dataclass(frozen=True)
class _DecodedRecord:
    kind: int
    decision_index: int
    personal_turn: int
    selected: bool
    samples: int
    game_index: int
    rollout_seed: int
    immediate_score: float
    target_remaining: float
    target_stddev: float
    features: np.ndarray


class RolloutValueShard:
    """A validated variable-record shard with compact random-access offsets."""

    def __init__(
        self,
        path: Path,
        entry: dict[str, Any],
        *,
        split: str,
        teacher: dict[str, Any],
        verify_checksum: bool,
    ):
        self.path = path
        self.entry = entry
        self.split = split
        self.teacher = teacher
        try:
            stat = path.stat()
        except OSError as error:
            raise DatasetError(f"cannot stat rollout-value shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"rollout-value shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"rollout-value shard checksum mismatch: {path}")
        self._bytes = np.memmap(path, mode="r", dtype=np.uint8)
        self._validate_header()
        self.record_offsets, self.record_kinds = self._index_and_validate_records()

    @property
    def record_count(self) -> int:
        return int(self.record_offsets.shape[0])

    def indices(self, kind: Literal["trajectory", "root"]) -> np.ndarray:
        code = TRAJECTORY_KIND if kind == "trajectory" else ROOT_ESTIMATE_KIND
        return np.flatnonzero(self.record_kinds == code)

    def decode(self, record_index: int) -> _DecodedRecord:
        offset = int(self.record_offsets[record_index])
        values = _RECORD_PREFIX.unpack_from(self._bytes, offset)
        feature_len = values[4]
        feature_offset = offset + ROLLOUT_VALUE_RECORD_PREFIX_SIZE
        features = np.frombuffer(
            self._bytes,
            dtype="<u2",
            count=feature_len,
            offset=feature_offset,
        )
        return _DecodedRecord(
            kind=values[0],
            decision_index=values[1],
            personal_turn=values[2],
            selected=bool(values[3]),
            samples=values[6],
            game_index=values[7],
            rollout_seed=values[8],
            immediate_score=values[9],
            target_remaining=values[10],
            target_stddev=values[11],
            features=features,
        )

    def _validate_header(self) -> None:
        if self._bytes.size < ROLLOUT_VALUE_HEADER_SIZE:
            raise DatasetError(f"truncated rollout-value shard header: {self.path}")
        (
            magic,
            schema,
            header_size,
            prefix_size,
            feature_count,
            record_count,
            trajectory_count,
            root_count,
            game_count,
            split_and_reserved,
            first_game_index,
            feature_hash,
            target_hash,
            teacher_hash,
            _reserved,
        ) = _HEADER.unpack_from(self._bytes, 0)
        expected_teacher_hash = blake3.blake3(
            json.dumps(self.teacher, separators=(",", ":"), ensure_ascii=False).encode()
        ).digest()
        if (
            magic != ROLLOUT_VALUE_SHARD_MAGIC
            or schema != ROLLOUT_VALUE_DATASET_SCHEMA_VERSION
            or header_size != ROLLOUT_VALUE_HEADER_SIZE
            or prefix_size != ROLLOUT_VALUE_RECORD_PREFIX_SIZE
            or feature_count != ROLLOUT_VALUE_FEATURE_COUNT
            or feature_hash != blake3.blake3(ROLLOUT_VALUE_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(ROLLOUT_VALUE_TARGET_SCHEMA.encode()).digest()
            or teacher_hash != expected_teacher_hash
        ):
            raise DatasetError(f"incompatible rollout-value shard header: {self.path}")
        if split_and_reserved[0] != _SPLIT_CODES[self.split]:
            raise DatasetError(f"rollout-value shard split mismatch: {self.path}")
        if (
            record_count != int(self.entry["record_count"])
            or trajectory_count + root_count != record_count
            or game_count != 1
            or int(self.entry["game_count"]) != 1
            or first_game_index != int(self.entry["first_game_index"])
        ):
            raise DatasetError(f"rollout-value shard header disagrees with manifest: {self.path}")
        self._header_record_count = record_count
        self._header_trajectory_count = trajectory_count
        self._header_root_count = root_count
        self._game_index = first_game_index

    def _index_and_validate_records(self) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.empty(self._header_record_count, dtype=np.uint64)
        kinds = np.empty(self._header_record_count, dtype=np.uint8)
        root_counts = np.zeros(80, dtype=np.uint16)
        selected_root_counts = np.zeros(80, dtype=np.uint8)
        position = ROLLOUT_VALUE_HEADER_SIZE
        trajectory_count = 0
        root_count = 0
        for record_index in range(self._header_record_count):
            if position + ROLLOUT_VALUE_RECORD_PREFIX_SIZE > self._bytes.size:
                raise DatasetError(f"truncated rollout-value record prefix: {self.path}")
            values = _RECORD_PREFIX.unpack_from(self._bytes, position)
            (
                kind,
                decision_index,
                personal_turn,
                selected,
                feature_len,
                reserved,
                samples,
                game_index,
                rollout_seed,
                immediate_score,
                target_remaining,
                target_stddev,
            ) = values
            record_end = position + ROLLOUT_VALUE_RECORD_PREFIX_SIZE + feature_len * 2
            if record_end > self._bytes.size:
                raise DatasetError(f"truncated rollout-value feature payload: {self.path}")
            features = np.frombuffer(
                self._bytes,
                dtype="<u2",
                count=feature_len,
                offset=position + ROLLOUT_VALUE_RECORD_PREFIX_SIZE,
            )
            if (
                kind not in (TRAJECTORY_KIND, ROOT_ESTIMATE_KIND)
                or decision_index >= 80
                or personal_turn < 1
                or personal_turn > 20
                or selected not in (0, 1)
                or feature_len == 0
                or reserved != 0
                or samples == 0
                or game_index != self._game_index
                or np.any(features >= ROLLOUT_VALUE_FEATURE_COUNT)
                or not np.isfinite(immediate_score)
                or not np.isfinite(target_remaining)
                or not np.isfinite(target_stddev)
                or target_stddev < 0
            ):
                raise DatasetError(f"invalid rollout-value record: {self.path}")
            root_personal_turn = decision_index // 4 + 1
            if kind == TRAJECTORY_KIND:
                trajectory_count += 1
                if (
                    selected != 1
                    or samples != 1
                    or target_stddev != 0
                    or personal_turn < root_personal_turn
                ):
                    raise DatasetError(f"invalid rollout trajectory record: {self.path}")
            else:
                root_count += 1
                if rollout_seed != 0 or personal_turn != root_personal_turn:
                    raise DatasetError(f"invalid rollout root record: {self.path}")
                root_counts[decision_index] += 1
                selected_root_counts[decision_index] += selected
            offsets[record_index] = position
            kinds[record_index] = kind
            position = record_end
        if position != self._bytes.size:
            raise DatasetError(f"rollout-value shard has trailing bytes: {self.path}")
        if (
            trajectory_count != self._header_trajectory_count
            or root_count != self._header_root_count
            or np.any(root_counts == 0)
            or np.any(selected_root_counts != 1)
        ):
            raise DatasetError(f"rollout-value record totals are invalid: {self.path}")
        return offsets, kinds


class RolloutValueDataset:
    """Checksummed rollout-return evidence with streaming sparse batches."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read rollout-value manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            RolloutValueShard(
                self.root / str(entry["file"]),
                entry,
                split=self.split,
                teacher=self.manifest["teacher"],
                verify_checksum=verify_checksums,
            )
            for entry in self.manifest["shards"]
        )
        self._root_group_cache: tuple[tuple[_DecodedRecord, ...], ...] | None = None

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def trajectory_count(self) -> int:
        return int(self.manifest["trajectory_records"])

    @property
    def root_count(self) -> int:
        return int(self.manifest["root_estimate_records"])

    @property
    def manifest_blake3(self) -> str:
        return _checksum(self.root / "dataset.json")

    def batches(
        self,
        batch_size: int,
        *,
        kind: Literal["trajectory", "root"] = "trajectory",
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[RolloutValueBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        if shuffle:
            kind_indices = [shard.indices(kind) for shard in self.shards]
            cumulative = np.zeros(len(kind_indices) + 1, dtype=np.int64)
            np.cumsum([len(indices) for indices in kind_indices], out=cumulative[1:])
            order = rng.permutation(cumulative[-1])
            for start in range(0, len(order), batch_size):
                global_indices = order[start : start + batch_size]
                shard_indices = np.searchsorted(
                    cumulative[1:],
                    global_indices,
                    side="right",
                )
                records = []
                for global_index, shard_index in zip(
                    global_indices,
                    shard_indices,
                    strict=True,
                ):
                    local_index = int(global_index - cumulative[shard_index])
                    record_index = int(kind_indices[shard_index][local_index])
                    records.append(self.shards[int(shard_index)].decode(record_index))
                yield _batch(records)
            return
        for shard in self.shards:
            indices = shard.indices(kind)
            for start in range(0, len(indices), batch_size):
                records = [
                    shard.decode(int(index)) for index in indices[start : start + batch_size]
                ]
                yield _batch(records)

    def root_groups(
        self,
        group_batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[RolloutValueRootBatch]:
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        groups = self._root_groups()
        order = np.arange(len(groups), dtype=np.int64)
        if shuffle:
            order = np.random.default_rng(seed).permutation(order)
        for start in range(0, len(order), group_batch_size):
            yield _root_group_batch(
                [groups[int(index)] for index in order[start : start + group_batch_size]]
            )

    def _root_groups(self) -> tuple[tuple[_DecodedRecord, ...], ...]:
        if self._root_group_cache is not None:
            return self._root_group_cache
        grouped: dict[tuple[int, int], list[_DecodedRecord]] = {}
        for shard in self.shards:
            for record_index in shard.indices("root"):
                record = shard.decode(int(record_index))
                key = (record.game_index, record.decision_index)
                grouped.setdefault(key, []).append(record)
        groups = tuple(tuple(records) for records in grouped.values())
        expected = int(self.manifest["completed_games"]) * 80
        if len(groups) != expected:
            raise DatasetError(
                f"rollout-value root group count {len(groups)} does not match {expected}"
            )
        self._root_group_cache = groups
        return groups

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != ROLLOUT_VALUE_DATASET_SCHEMA_VERSION
            or manifest.get("feature_schema") != ROLLOUT_VALUE_FEATURE_SCHEMA
            or manifest.get("target_schema") != ROLLOUT_VALUE_TARGET_SCHEMA
            or manifest.get("record_prefix_size") != ROLLOUT_VALUE_RECORD_PREFIX_SIZE
            or manifest.get("split") not in _SPLIT_CODES
        ):
            raise DatasetError("unsupported rollout-value dataset manifest")
        teacher = manifest.get("teacher")
        if (
            not isinstance(teacher, dict)
            or teacher.get("feature_count") != ROLLOUT_VALUE_FEATURE_COUNT
            or teacher.get("candidate_limit") != 32
            or int(teacher.get("rollouts", 0)) <= 0
            or int(teacher.get("trace_modulus", 0)) <= 0
            or teacher.get("lmr") is not True
            or teacher.get("diverse_prefilter") is not True
            or len(str(teacher.get("parent_model_manifest_blake3", ""))) != 64
            or len(str(teacher.get("weights_blake3", ""))) != 64
        ):
            raise DatasetError("invalid rollout-value teacher contract")
        shards = manifest.get("shards")
        completed = int(manifest.get("completed_games", -1))
        requested = int(manifest.get("requested_games", -1))
        if (
            not isinstance(shards, list)
            or completed < 0
            or requested < completed
            or len(shards) != completed
            or sum(int(shard["game_count"]) for shard in shards) != completed
            or sum(int(shard["record_count"]) for shard in shards)
            != int(manifest.get("total_records", -1))
        ):
            raise DatasetError("rollout-value manifest totals do not match shards")
        first_game_index = int(manifest["first_game_index"])
        for index, shard in enumerate(shards):
            if int(shard["first_game_index"]) != first_game_index + index:
                raise DatasetError("rollout-value shard game sequence is invalid")
        trajectory = sum(_header_kind_count(self.root / shard["file"], 20) for shard in shards)
        roots = sum(_header_kind_count(self.root / shard["file"], 24) for shard in shards)
        if (
            trajectory != int(manifest.get("trajectory_records", -1))
            or roots != int(manifest.get("root_estimate_records", -1))
            or trajectory + roots != int(manifest.get("total_records", -1))
        ):
            raise DatasetError("rollout-value manifest kind totals do not match shards")


def _batch(records: list[_DecodedRecord]) -> RolloutValueBatch:
    if not records:
        raise ValueError("cannot build an empty rollout-value batch")
    width = max(max(len(record.features) for record in records), 1)
    indices = np.zeros((len(records), width), dtype=np.int32)
    mask = np.zeros((len(records), width), dtype=np.bool_)
    for row, record in enumerate(records):
        feature_len = len(record.features)
        indices[row, :feature_len] = record.features
        mask[row, :feature_len] = True
    return RolloutValueBatch(
        feature_indices=mx.array(indices),
        feature_mask=mx.array(mask),
        target_remaining=mx.array(
            np.asarray([record.target_remaining for record in records], dtype=np.float32)
        ),
        immediate_score=mx.array(
            np.asarray([record.immediate_score for record in records], dtype=np.float32)
        ),
        game_index=np.asarray([record.game_index for record in records], dtype=np.uint64),
        decision_index=np.asarray([record.decision_index for record in records], dtype=np.uint8),
        personal_turn=np.asarray([record.personal_turn for record in records], dtype=np.uint8),
        selected=np.asarray([record.selected for record in records], dtype=np.bool_),
        rollout_seed=np.asarray([record.rollout_seed for record in records], dtype=np.uint64),
        target_stddev=np.asarray([record.target_stddev for record in records], dtype=np.float32),
        samples=np.asarray([record.samples for record in records], dtype=np.uint32),
    )


def _root_group_batch(groups: list[tuple[_DecodedRecord, ...]]) -> RolloutValueRootBatch:
    if not groups:
        raise ValueError("cannot build an empty rollout-value root batch")
    candidate_width = max(len(group) for group in groups)
    feature_width = max(max(len(record.features) for record in group) for group in groups)
    shape = (len(groups), candidate_width)
    indices = np.zeros((*shape, feature_width), dtype=np.int32)
    feature_mask = np.zeros((*shape, feature_width), dtype=np.bool_)
    candidate_mask = np.zeros(shape, dtype=np.bool_)
    target_remaining = np.zeros(shape, dtype=np.float32)
    immediate_score = np.zeros(shape, dtype=np.float32)
    selected = np.zeros(shape, dtype=np.bool_)
    target_stddev = np.zeros(shape, dtype=np.float32)
    samples = np.zeros(shape, dtype=np.float32)
    game_index = np.empty(len(groups), dtype=np.uint64)
    decision_index = np.empty(len(groups), dtype=np.uint8)
    personal_turn = np.empty(len(groups), dtype=np.uint8)
    for group_index, group in enumerate(groups):
        first = group[0]
        game_index[group_index] = first.game_index
        decision_index[group_index] = first.decision_index
        personal_turn[group_index] = first.personal_turn
        for candidate_index, record in enumerate(group):
            if (
                record.game_index != first.game_index
                or record.decision_index != first.decision_index
                or record.personal_turn != first.personal_turn
            ):
                raise DatasetError("rollout-value root group metadata is inconsistent")
            feature_len = len(record.features)
            indices[group_index, candidate_index, :feature_len] = record.features
            feature_mask[group_index, candidate_index, :feature_len] = True
            candidate_mask[group_index, candidate_index] = True
            target_remaining[group_index, candidate_index] = record.target_remaining
            immediate_score[group_index, candidate_index] = record.immediate_score
            selected[group_index, candidate_index] = record.selected
            target_stddev[group_index, candidate_index] = record.target_stddev
            samples[group_index, candidate_index] = record.samples
    if not np.all(selected.sum(axis=1) == 1):
        raise DatasetError("rollout-value root batch must contain one selected action per group")
    return RolloutValueRootBatch(
        feature_indices=mx.array(indices),
        feature_mask=mx.array(feature_mask),
        candidate_mask=mx.array(candidate_mask),
        target_remaining=mx.array(target_remaining),
        immediate_score=mx.array(immediate_score),
        selected=mx.array(selected),
        target_stddev=mx.array(target_stddev),
        samples=mx.array(samples),
        game_index=game_index,
        decision_index=decision_index,
        personal_turn=personal_turn,
    )


def _header_kind_count(path: Path, offset: int) -> int:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(4)
    except OSError as error:
        raise DatasetError(f"cannot read rollout-value shard header {path}: {error}") from error
    if len(payload) != 4:
        raise DatasetError(f"truncated rollout-value shard header: {path}")
    return int.from_bytes(payload, "little")


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(records: RolloutValueBatch) -> str:
    """Stable test/debug digest that includes sparse multiplicity and metadata."""
    digest = hashlib.sha256()
    digest.update(np.asarray(records.feature_indices).astype("<i4", copy=False).tobytes())
    digest.update(np.asarray(records.feature_mask).astype(np.uint8, copy=False).tobytes())
    digest.update(np.asarray(records.target_remaining).astype("<f4", copy=False).tobytes())
    digest.update(records.game_index.astype("<u8", copy=False).tobytes())
    digest.update(records.decision_index.tobytes())
    return digest.hexdigest()
