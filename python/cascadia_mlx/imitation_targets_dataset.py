"""Hash-aligned MCE evidence layered over grouped canonical actions."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.dataset import DatasetError
from cascadia_mlx.imitation_dataset import (
    ImitationBatch,
    ImitationDataset,
    ImitationGroup,
    decode_imitation_groups,
)

IMITATION_TARGETS_DATASET_SCHEMA_VERSION = 1
IMITATION_TARGETS_FEATURE_SCHEMA = "canonical-action-hash-alignment-v1"
IMITATION_TARGETS_TARGET_SCHEMA = "mce-score-distribution-v1"
IMITATION_TARGETS_SHARD_MAGIC = b"CSD2IMV\0"
IMITATION_TARGETS_HEADER_SIZE = 112
IMITATION_TARGETS_RECORD_SIZE = 56

SOURCE_TEACHER_FRONTIER = 1 << 0
SOURCE_PATTERN_FRONTIER = 1 << 1
SOURCE_IMMEDIATE_TOP = 1 << 2
SOURCE_DETERMINISTIC_NEGATIVE = 1 << 3

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_TARGET_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "candidate_index",
            "candidate_count",
            "action_hash",
            "teacher_mean",
            "teacher_stddev",
            "teacher_samples",
            "source_flags",
            "selected",
        ],
        "formats": [
            "<u8",
            "<u2",
            "<u2",
            ("u1", (32,)),
            "<f4",
            "<f4",
            "<u2",
            "u1",
            "u1",
        ],
        "offsets": [0, 8, 10, 12, 44, 48, 52, 54, 55],
        "itemsize": IMITATION_TARGETS_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class ImitationTargetGroup:
    group_id: int
    records: np.ndarray


@dataclass
class ImitationTargetShard:
    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def groups(self) -> tuple[ImitationTargetGroup, ...]:
        records = np.frombuffer(
            self.path.read_bytes(),
            dtype=_TARGET_DTYPE,
            offset=IMITATION_TARGETS_HEADER_SIZE,
            count=self.record_count,
        )
        groups: list[ImitationTargetGroup] = []
        start = 0
        while start < len(records):
            count = int(records["candidate_count"][start])
            end = start + count
            if count < 2 or end > len(records):
                raise DatasetError(f"inconsistent imitation-target group: {self.path}")
            group_records = records[start:end]
            group_id = int(group_records["group_id"][0])
            indices = group_records["candidate_index"].astype(np.int64)
            samples = group_records["teacher_samples"].astype(np.int64)
            means = group_records["teacher_mean"].astype(np.float64)
            selected = group_records["selected"].astype(np.bool_)
            if (
                np.any(group_records["group_id"] != group_id)
                or np.any(group_records["candidate_count"] != count)
                or not np.array_equal(indices, np.arange(count))
                or np.unique(group_records["action_hash"], axis=0).shape[0] != count
                or np.count_nonzero(selected) != 1
                or np.any(group_records["source_flags"] == 0)
                or np.any(~np.isfinite(means))
                or np.any(~np.isfinite(group_records["teacher_stddev"]))
                or np.any(group_records["teacher_stddev"] < 0)
                or np.any((samples == 0) & (means != 0))
                or np.any((samples == 0) & (group_records["teacher_stddev"] != 0))
            ):
                raise DatasetError(f"invalid imitation-target evidence group {group_id}")
            scored = samples > 0
            if not np.any(scored) or means[selected][0] != np.max(means[scored]):
                raise DatasetError(f"selected action is not a scored maximum in group {group_id}")
            groups.append(ImitationTargetGroup(group_id, group_records))
            start = end
        if len(groups) != self.group_count or start != self.record_count:
            raise DatasetError(f"imitation-target shard totals do not match: {self.path}")
        return tuple(groups)


class ImitationEvidenceDataset:
    """Complete grouped actions enriched with full-frontier MCE distributions."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read imitation-target manifest: {error}") from error
        self._validate_manifest()
        source = self.manifest["source"]
        self.source = ImitationDataset(
            Path(str(source["path"])),
            verify_checksums=verify_checksums,
        )
        self._validate_source_identity()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )
        if len(self.shards) != len(self.source.shards):
            raise DatasetError("imitation evidence requires complete paired source shards")
        for source_shard, target_shard in zip(self.source.shards, self.shards, strict=True):
            if (
                source_shard.first_game_index != target_shard.first_game_index
                or source_shard.game_count != target_shard.game_count
                or source_shard.group_count != target_shard.group_count
                or source_shard.record_count != target_shard.record_count
            ):
                raise DatasetError("imitation source and target shard ranges differ")
            self._validate_group_alignment(source_shard.groups(), target_shard.groups())

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
    ) -> Iterator[ImitationBatch]:
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        rng = np.random.default_rng(seed)
        shard_indices = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            source_groups = list(self.source.shards[int(shard_index)].groups())
            target_by_group = {
                group.group_id: group for group in self.shards[int(shard_index)].groups()
            }
            if shuffle:
                rng.shuffle(source_groups)
            for start in range(0, len(source_groups), group_batch_size):
                groups = source_groups[start : start + group_batch_size]
                batch = decode_imitation_groups(groups)
                yield self._attach_targets(
                    batch,
                    groups,
                    [target_by_group[group.group_id] for group in groups],
                )

    def _attach_targets(
        self,
        batch: ImitationBatch,
        source_groups: list[ImitationGroup],
        target_groups: list[ImitationTargetGroup],
    ) -> ImitationBatch:
        shape = tuple(batch.candidate_mask.shape)
        means = np.zeros(shape, dtype=np.float32)
        stddev = np.zeros(shape, dtype=np.float32)
        samples = np.zeros(shape, dtype=np.float32)
        scored = np.zeros(shape, dtype=np.bool_)
        selected = np.zeros(shape, dtype=np.bool_)
        source_flags = np.zeros(shape, dtype=np.uint8)
        for group_index, (source, target) in enumerate(
            zip(source_groups, target_groups, strict=True)
        ):
            self._validate_group_alignment((source,), (target,))
            count = len(source.candidates)
            records = target.records
            means[group_index, :count] = records["teacher_mean"]
            stddev[group_index, :count] = records["teacher_stddev"]
            samples[group_index, :count] = records["teacher_samples"]
            scored[group_index, :count] = records["teacher_samples"] > 0
            selected[group_index, :count] = records["selected"] != 0
            source_flags[group_index, :count] = records["source_flags"]
        return replace(
            batch,
            teacher_mean=mx.array(means),
            teacher_stddev=mx.array(stddev),
            teacher_samples=mx.array(samples),
            teacher_scored=mx.array(scored),
            selected=mx.array(selected),
            source_flags=mx.array(source_flags),
        )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != IMITATION_TARGETS_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported imitation-target schema version")
        if manifest.get("feature_schema") != IMITATION_TARGETS_FEATURE_SCHEMA:
            raise DatasetError("unsupported imitation-target feature schema")
        if manifest.get("target_schema") != IMITATION_TARGETS_TARGET_SCHEMA:
            raise DatasetError("unsupported imitation-target target schema")
        if manifest.get("record_size") != IMITATION_TARGETS_RECORD_SIZE:
            raise DatasetError("unsupported imitation-target record size")
        if not isinstance(manifest.get("source"), dict):
            raise DatasetError("imitation-target manifest requires source identity")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("imitation-target manifest shards must be a list")
        if int(manifest.get("completed_games", -1)) != int(manifest.get("requested_games", -2)):
            raise DatasetError("imitation evidence dataset must be complete")
        if int(manifest.get("teacher_estimates", -1)) != int(
            manifest.get("aligned_teacher_estimates", -2)
        ):
            raise DatasetError(
                "imitation evidence dataset does not retain the full teacher frontier"
            )
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("imitation-target record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("imitation-target group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("imitation-target game total does not match shards")

    def _validate_source_identity(self) -> None:
        source = self.manifest["source"]
        actual = self.source.manifest
        expected = {
            "dataset_id": actual.get("dataset_id"),
            "feature_schema": actual.get("feature_schema"),
            "target_schema": actual.get("target_schema"),
            "first_game_index": actual.get("first_game_index"),
            "requested_games": actual.get("requested_games"),
        }
        if any(source.get(key) != value for key, value in expected.items()):
            raise DatasetError("imitation-target source identity does not match source dataset")
        if self.source.split != self.split:
            raise DatasetError("imitation source and target splits differ")
        if self.source.manifest.get("teacher") != self.manifest.get("teacher"):
            raise DatasetError("imitation source and target teachers differ")
        sampler = self.source.manifest.get("candidates", {}).get("deterministic_sampler")
        if sampler != "teacher-frontier-pattern-immediate-blake3-action-json-v1":
            raise DatasetError(
                "imitation evidence source does not guarantee teacher-frontier recall"
            )
        if self.source.group_count != self.group_count:
            raise DatasetError("imitation source and target group totals differ")
        if self.source.candidate_count != self.candidate_count:
            raise DatasetError("imitation source and target candidate totals differ")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ImitationTargetShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:IMITATION_TARGETS_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read imitation-target shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"imitation-target shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"imitation-target shard checksum mismatch: {path}")
        if len(header) != IMITATION_TARGETS_HEADER_SIZE:
            raise DatasetError(f"truncated imitation-target shard header: {path}")
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
            magic != IMITATION_TARGETS_SHARD_MAGIC
            or schema != IMITATION_TARGETS_DATASET_SCHEMA_VERSION
            or header_size != IMITATION_TARGETS_HEADER_SIZE
            or record_size != IMITATION_TARGETS_RECORD_SIZE
            or feature_hash != blake3.blake3(IMITATION_TARGETS_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(IMITATION_TARGETS_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible imitation-target shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"imitation-target shard header disagrees with manifest: {path}")
        expected_size = IMITATION_TARGETS_HEADER_SIZE + record_count * IMITATION_TARGETS_RECORD_SIZE
        if stat.st_size != expected_size:
            raise DatasetError(f"imitation-target record count mismatch: {path}")
        return ImitationTargetShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )

    @staticmethod
    def _validate_group_alignment(
        source_groups: tuple[ImitationGroup, ...],
        target_groups: tuple[ImitationTargetGroup, ...],
    ) -> None:
        if len(source_groups) != len(target_groups):
            raise DatasetError("imitation source and target group counts differ")
        for source, target in zip(source_groups, target_groups, strict=True):
            if (
                source.group_id != target.group_id
                or len(source.candidates) != len(target.records)
                or not np.array_equal(
                    source.candidates["action_hash"],
                    target.records["action_hash"],
                )
                or int(np.flatnonzero(target.records["selected"])[0]) != source.selected_index
            ):
                raise DatasetError(
                    f"imitation source and target actions differ for group {source.group_id}"
                )


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
