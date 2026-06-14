"""Exact-MLX parent hidden states aligned with full-frontier MCE evidence."""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.dataset import DatasetError
from cascadia_mlx.imitation_dataset import ImitationBatch, ImitationGroup
from cascadia_mlx.imitation_parent_prior_dataset import (
    ImitationParentEvidenceDataset,
)

IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION = 1
IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA = "canonical-action-parent-hidden-v1"
IMITATION_PARENT_HIDDEN_TARGET_SCHEMA = "exact-mlx-afterstate-hidden64-value-v1"
IMITATION_PARENT_HIDDEN_SHARD_MAGIC = b"CSD2IMH\0"
IMITATION_PARENT_HIDDEN_HEADER_SIZE = 112
IMITATION_PARENT_HIDDEN_DIM = 64
IMITATION_PARENT_HIDDEN_RECORD_SIZE = 312

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_HIDDEN_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "candidate_index",
            "candidate_count",
            "action_hash",
            "parent_immediate",
            "parent_remaining",
            "parent_hidden",
        ],
        "formats": [
            "<u8",
            "<u2",
            "<u2",
            ("u1", (32,)),
            "<f4",
            "<f4",
            ("<f4", (IMITATION_PARENT_HIDDEN_DIM,)),
        ],
        "offsets": [0, 8, 10, 12, 44, 48, 52],
        "itemsize": IMITATION_PARENT_HIDDEN_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class ImitationParentHiddenGroup:
    group_id: int
    records: np.ndarray


@dataclass
class ImitationParentHiddenShard:
    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def groups(self) -> tuple[ImitationParentHiddenGroup, ...]:
        records = np.frombuffer(
            self.path.read_bytes(),
            dtype=_HIDDEN_DTYPE,
            offset=IMITATION_PARENT_HIDDEN_HEADER_SIZE,
            count=self.record_count,
        )
        groups: list[ImitationParentHiddenGroup] = []
        start = 0
        while start < len(records):
            count = int(records["candidate_count"][start])
            end = start + count
            if count < 2 or end > len(records):
                raise DatasetError(f"inconsistent parent-hidden group: {self.path}")
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
                or np.any(~np.isfinite(group_records["parent_hidden"]))
            ):
                raise DatasetError(f"invalid parent-hidden group {group_id}")
            groups.append(ImitationParentHiddenGroup(group_id, group_records))
            start = end
        if len(groups) != self.group_count or start != self.record_count:
            raise DatasetError(f"parent-hidden shard totals do not match: {self.path}")
        return tuple(groups)


class ImitationParentHiddenEvidenceDataset(ImitationParentEvidenceDataset):
    """MCE evidence with exact parent score and hidden state for every action."""

    def _attach_priors(
        self,
        batch: ImitationBatch,
        source_groups: list[ImitationGroup],
        prior_groups: list[ImitationParentHiddenGroup],
    ) -> ImitationBatch:
        shape = tuple(batch.candidate_mask.shape)
        immediate = np.zeros(shape, dtype=np.float32)
        remaining = np.zeros(shape, dtype=np.float32)
        total = np.zeros(shape, dtype=np.float32)
        rank = np.zeros(shape, dtype=np.float32)
        hidden = np.zeros((*shape, IMITATION_PARENT_HIDDEN_DIM), dtype=np.float32)
        for group_index, (source, prior) in enumerate(
            zip(source_groups, prior_groups, strict=True)
        ):
            self._validate_group_alignment((source,), (prior,))
            count = len(source.candidates)
            records = prior.records
            immediate[group_index, :count] = records["parent_immediate"]
            remaining[group_index, :count] = records["parent_remaining"]
            hidden[group_index, :count] = records["parent_hidden"]
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
            parent_hidden=mx.array(hidden),
        )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported parent-hidden schema version")
        if manifest.get("feature_schema") != IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA:
            raise DatasetError("unsupported parent-hidden feature schema")
        if manifest.get("target_schema") != IMITATION_PARENT_HIDDEN_TARGET_SCHEMA:
            raise DatasetError("unsupported parent-hidden target schema")
        if manifest.get("record_size") != IMITATION_PARENT_HIDDEN_RECORD_SIZE:
            raise DatasetError("unsupported parent-hidden record size")
        if not isinstance(manifest.get("source"), dict):
            raise DatasetError("parent-hidden manifest requires source identity")
        if not isinstance(manifest.get("model"), dict):
            raise DatasetError("parent-hidden manifest requires model identity")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("parent-hidden manifest shards must be a list")
        if int(manifest.get("completed_games", -1)) != int(manifest.get("requested_games", -2)):
            raise DatasetError("parent-hidden dataset must be complete")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("parent-hidden record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("parent-hidden group total does not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ImitationParentHiddenShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:IMITATION_PARENT_HIDDEN_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read parent-hidden shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"parent-hidden shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"parent-hidden shard checksum mismatch: {path}")
        if len(header) != IMITATION_PARENT_HIDDEN_HEADER_SIZE:
            raise DatasetError(f"truncated parent-hidden shard header: {path}")
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
            magic != IMITATION_PARENT_HIDDEN_SHARD_MAGIC
            or schema != IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION
            or header_size != IMITATION_PARENT_HIDDEN_HEADER_SIZE
            or record_size != IMITATION_PARENT_HIDDEN_RECORD_SIZE
            or feature_hash
            != blake3.blake3(IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(IMITATION_PARENT_HIDDEN_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible parent-hidden shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"parent-hidden header disagrees with manifest: {path}")
        expected_size = (
            IMITATION_PARENT_HIDDEN_HEADER_SIZE + record_count * IMITATION_PARENT_HIDDEN_RECORD_SIZE
        )
        if stat.st_size != expected_size:
            raise DatasetError(f"parent-hidden record count mismatch: {path}")
        return ImitationParentHiddenShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
