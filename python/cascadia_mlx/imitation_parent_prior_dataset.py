"""Exact-MLX parent priors aligned with full-frontier MCE evidence."""

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
    ImitationGroup,
    decode_imitation_groups,
)
from cascadia_mlx.imitation_targets_dataset import ImitationEvidenceDataset

IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION = 1
IMITATION_PARENT_PRIOR_FEATURE_SCHEMA = "canonical-action-parent-prior-v1"
IMITATION_PARENT_PRIOR_TARGET_SCHEMA = "exact-mlx-afterstate-value-v1"
IMITATION_PARENT_PRIOR_SHARD_MAGIC = b"CSD2IMP\0"
IMITATION_PARENT_PRIOR_HEADER_SIZE = 112
IMITATION_PARENT_PRIOR_RECORD_SIZE = 56

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_PRIOR_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "candidate_index",
            "candidate_count",
            "action_hash",
            "parent_immediate",
            "parent_remaining",
        ],
        "formats": [
            "<u8",
            "<u2",
            "<u2",
            ("u1", (32,)),
            "<f4",
            "<f4",
        ],
        "offsets": [0, 8, 10, 12, 44, 48],
        "itemsize": IMITATION_PARENT_PRIOR_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class ImitationParentPriorGroup:
    group_id: int
    records: np.ndarray


@dataclass
class ImitationParentPriorShard:
    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def groups(self) -> tuple[ImitationParentPriorGroup, ...]:
        records = np.frombuffer(
            self.path.read_bytes(),
            dtype=_PRIOR_DTYPE,
            offset=IMITATION_PARENT_PRIOR_HEADER_SIZE,
            count=self.record_count,
        )
        groups: list[ImitationParentPriorGroup] = []
        start = 0
        while start < len(records):
            count = int(records["candidate_count"][start])
            end = start + count
            if count < 2 or end > len(records):
                raise DatasetError(f"inconsistent parent-prior group: {self.path}")
            group_records = records[start:end]
            group_id = int(group_records["group_id"][0])
            indices = group_records["candidate_index"].astype(np.int64)
            totals = group_records["parent_immediate"].astype(np.float64) + group_records[
                "parent_remaining"
            ].astype(np.float64)
            if (
                np.any(group_records["group_id"] != group_id)
                or np.any(group_records["candidate_count"] != count)
                or not np.array_equal(indices, np.arange(count))
                or np.unique(group_records["action_hash"], axis=0).shape[0] != count
                or np.any(~np.isfinite(totals))
            ):
                raise DatasetError(f"invalid parent-prior group {group_id}")
            groups.append(ImitationParentPriorGroup(group_id, group_records))
            start = end
        if len(groups) != self.group_count or start != self.record_count:
            raise DatasetError(f"parent-prior shard totals do not match: {self.path}")
        return tuple(groups)


class ImitationParentEvidenceDataset:
    """MCE action evidence with an exact frozen-parent score for every action."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read parent-prior manifest: {error}") from error
        self._validate_manifest()
        self.evidence = ImitationEvidenceDataset(
            Path(str(self.manifest["source"]["path"])),
            verify_checksums=verify_checksums,
        )
        self._validate_source_identity()
        self._validate_model_identity()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )
        if len(self.shards) != len(self.evidence.shards):
            raise DatasetError("parent-prior evidence requires complete paired shards")
        for index, (source_shard, prior_shard) in enumerate(
            zip(self.evidence.shards, self.shards, strict=True)
        ):
            if (
                source_shard.first_game_index != prior_shard.first_game_index
                or source_shard.game_count != prior_shard.game_count
                or source_shard.group_count != prior_shard.group_count
                or source_shard.record_count != prior_shard.record_count
            ):
                raise DatasetError("parent-prior and MCE shard ranges differ")
            source_groups = self.evidence.source.shards[index].groups()
            self._validate_group_alignment(source_groups, prior_shard.groups())

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
            index = int(shard_index)
            source_groups = list(self.evidence.source.shards[index].groups())
            targets = {group.group_id: group for group in self.evidence.shards[index].groups()}
            priors = {group.group_id: group for group in self.shards[index].groups()}
            if shuffle:
                rng.shuffle(source_groups)
            for start in range(0, len(source_groups), group_batch_size):
                groups = source_groups[start : start + group_batch_size]
                batch = decode_imitation_groups(groups)
                batch = self.evidence._attach_targets(
                    batch,
                    groups,
                    [targets[group.group_id] for group in groups],
                )
                yield self._attach_priors(
                    batch,
                    groups,
                    [priors[group.group_id] for group in groups],
                )

    def _attach_priors(
        self,
        batch: ImitationBatch,
        source_groups: list[ImitationGroup],
        prior_groups: list[ImitationParentPriorGroup],
    ) -> ImitationBatch:
        shape = tuple(batch.candidate_mask.shape)
        immediate = np.zeros(shape, dtype=np.float32)
        remaining = np.zeros(shape, dtype=np.float32)
        total = np.zeros(shape, dtype=np.float32)
        rank = np.zeros(shape, dtype=np.float32)
        for group_index, (source, prior) in enumerate(
            zip(source_groups, prior_groups, strict=True)
        ):
            self._validate_group_alignment((source,), (prior,))
            count = len(source.candidates)
            records = prior.records
            immediate[group_index, :count] = records["parent_immediate"]
            remaining[group_index, :count] = records["parent_remaining"]
            values = records["parent_immediate"].astype(np.float32) + records[
                "parent_remaining"
            ].astype(np.float32)
            total[group_index, :count] = values
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(count, dtype=np.float32)
            ranks[order] = np.arange(1, count + 1, dtype=np.float32)
            rank[group_index, :count] = ranks
        return replace(
            batch,
            parent_immediate=mx.array(immediate),
            parent_remaining=mx.array(remaining),
            parent_total=mx.array(total),
            parent_rank=mx.array(rank),
        )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported parent-prior schema version")
        if manifest.get("feature_schema") != IMITATION_PARENT_PRIOR_FEATURE_SCHEMA:
            raise DatasetError("unsupported parent-prior feature schema")
        if manifest.get("target_schema") != IMITATION_PARENT_PRIOR_TARGET_SCHEMA:
            raise DatasetError("unsupported parent-prior target schema")
        if manifest.get("record_size") != IMITATION_PARENT_PRIOR_RECORD_SIZE:
            raise DatasetError("unsupported parent-prior record size")
        if not isinstance(manifest.get("source"), dict):
            raise DatasetError("parent-prior manifest requires source identity")
        if not isinstance(manifest.get("model"), dict):
            raise DatasetError("parent-prior manifest requires model identity")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("parent-prior manifest shards must be a list")
        if int(manifest.get("completed_games", -1)) != int(manifest.get("requested_games", -2)):
            raise DatasetError("parent-prior dataset must be complete")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("parent-prior record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("parent-prior group total does not match shards")

    def _validate_source_identity(self) -> None:
        source = self.manifest["source"]
        actual = self.evidence.manifest
        if (
            source.get("dataset_id") != actual.get("dataset_id")
            or source.get("feature_schema") != actual.get("feature_schema")
            or source.get("target_schema") != actual.get("target_schema")
            or source.get("action_dataset_id") != actual.get("source", {}).get("dataset_id")
            or source.get("first_game_index") != actual.get("first_game_index")
            or source.get("requested_games") != actual.get("requested_games")
            or self.evidence.split != self.split
        ):
            raise DatasetError("parent-prior source identity does not match MCE evidence")

    def _validate_model_identity(self) -> None:
        model = self.manifest["model"]
        root = Path(str(model["path"]))
        if _checksum(root / "model.json") != model.get("manifest_blake3") or _checksum(
            root / "model.safetensors"
        ) != model.get("tensors_blake3"):
            raise DatasetError("parent-prior model identity does not match local files")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ImitationParentPriorShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:IMITATION_PARENT_PRIOR_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read parent-prior shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"parent-prior shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"parent-prior shard checksum mismatch: {path}")
        if len(header) != IMITATION_PARENT_PRIOR_HEADER_SIZE:
            raise DatasetError(f"truncated parent-prior shard header: {path}")
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
            magic != IMITATION_PARENT_PRIOR_SHARD_MAGIC
            or schema != IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION
            or header_size != IMITATION_PARENT_PRIOR_HEADER_SIZE
            or record_size != IMITATION_PARENT_PRIOR_RECORD_SIZE
            or feature_hash
            != blake3.blake3(IMITATION_PARENT_PRIOR_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(IMITATION_PARENT_PRIOR_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible parent-prior shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"parent-prior header disagrees with manifest: {path}")
        expected_size = (
            IMITATION_PARENT_PRIOR_HEADER_SIZE + record_count * IMITATION_PARENT_PRIOR_RECORD_SIZE
        )
        if stat.st_size != expected_size:
            raise DatasetError(f"parent-prior record count mismatch: {path}")
        return ImitationParentPriorShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )

    @staticmethod
    def _validate_group_alignment(
        source_groups: tuple[ImitationGroup, ...],
        prior_groups: tuple[ImitationParentPriorGroup, ...],
    ) -> None:
        if len(source_groups) != len(prior_groups):
            raise DatasetError("parent-prior source and prior group counts differ")
        for source, prior in zip(source_groups, prior_groups, strict=True):
            if (
                source.group_id != prior.group_id
                or len(source.candidates) != len(prior.records)
                or not np.array_equal(
                    source.candidates["action_hash"],
                    prior.records["action_hash"],
                )
            ):
                raise DatasetError(f"parent-prior actions differ for group {source.group_id}")


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
