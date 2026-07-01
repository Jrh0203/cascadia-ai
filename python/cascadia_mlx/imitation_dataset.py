"""Shared-state canonical action-imitation data for MLX."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
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
    _mask_bits,
    _one_hot,
    _one_hot_with_none,
    decode_records,
)

IMITATION_DATASET_SCHEMA_VERSION = 1
IMITATION_FEATURE_SCHEMA = "compact-state-action-v1"
IMITATION_TARGET_SCHEMA = "canonical-action-imitation-v1"
IMITATION_SHARD_MAGIC = b"CSD2IMT\0"
IMITATION_HEADER_SIZE = 112
IMITATION_GROUP_HEADER_SIZE = 880
IMITATION_CANDIDATE_RECORD_SIZE = 68
PROPOSAL_ACTION_FEATURE_SIZE = 32
PROPOSAL_ACTION_DIM = 52
PROPOSAL_ACTION_IMMEDIATE_RANK_INDEX = 50
PROPOSAL_ACTION_IMMEDIATE_SCORE_INDEX = 51

_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_PROPOSAL_ACTION_DTYPE = np.dtype(
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
        ],
        "offsets": [*range(18), 18, 20],
        "itemsize": PROPOSAL_ACTION_FEATURE_SIZE,
    }
)
_IMITATION_GROUP_HEADER_DTYPE = np.dtype(
    {
        "names": ["group_id", "candidate_count", "selected_index", "position"],
        "formats": ["<u8", "<u2", "<u2", _RECORD_DTYPE],
        "offsets": [0, 8, 10, 16],
        "itemsize": IMITATION_GROUP_HEADER_SIZE,
    }
)
_IMITATION_CANDIDATE_DTYPE = np.dtype(
    {
        "names": [
            "immediate_rank",
            "immediate_score",
            "action_hash",
            "action",
        ],
        "formats": [
            "<u2",
            "<u2",
            ("u1", (32,)),
            _PROPOSAL_ACTION_DTYPE,
        ],
        "offsets": [0, 2, 4, 36],
        "itemsize": IMITATION_CANDIDATE_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class ImitationBatch:
    """Complete decisions with one shared public state and padded actions."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    action_features: mx.array
    candidate_mask: mx.array
    teacher_mean: mx.array
    teacher_stddev: mx.array
    teacher_samples: mx.array
    teacher_scored: mx.array
    selected: mx.array
    source_flags: mx.array
    immediate_rank: mx.array
    immediate_score: mx.array
    parent_immediate: mx.array
    parent_remaining: mx.array
    parent_total: mx.array
    parent_rank: mx.array
    parent_hidden: mx.array
    group_id: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class ImitationGroup:
    group_id: int
    selected_index: int
    position: np.ndarray
    candidates: np.ndarray


@dataclass
class ImitationShard:
    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int

    def groups(self) -> tuple[ImitationGroup, ...]:
        raw = np.frombuffer(self.path.read_bytes(), dtype=np.uint8)
        offset = IMITATION_HEADER_SIZE
        groups: list[ImitationGroup] = []
        candidate_total = 0
        for _ in range(self.group_count):
            if offset + IMITATION_GROUP_HEADER_SIZE > len(raw):
                raise DatasetError(f"truncated imitation group header: {self.path}")
            header = np.ndarray(
                (1,),
                dtype=_IMITATION_GROUP_HEADER_DTYPE,
                buffer=raw,
                offset=offset,
            )
            count = int(header["candidate_count"][0])
            selected_index = int(header["selected_index"][0])
            if count < 2 or selected_index >= count:
                raise DatasetError(f"inconsistent imitation group header: {self.path}")
            offset += IMITATION_GROUP_HEADER_SIZE
            candidate_bytes = count * IMITATION_CANDIDATE_RECORD_SIZE
            if offset + candidate_bytes > len(raw):
                raise DatasetError(f"truncated imitation candidates: {self.path}")
            candidates = np.ndarray(
                (count,),
                dtype=_IMITATION_CANDIDATE_DTYPE,
                buffer=raw,
                offset=offset,
            )
            group_id = int(header["group_id"][0])
            if (
                np.any(candidates["immediate_rank"] == 0)
                or np.any(candidates["action"]["immediate_rank"] != candidates["immediate_rank"])
                or np.any(candidates["action"]["immediate_score"] != candidates["immediate_score"])
                or np.unique(candidates["action_hash"], axis=0).shape[0] != count
            ):
                raise DatasetError(f"inconsistent imitation group {group_id}: {self.path}")
            groups.append(
                ImitationGroup(
                    group_id=group_id,
                    selected_index=selected_index,
                    position=header["position"],
                    candidates=candidates,
                )
            )
            candidate_total += count
            offset += candidate_bytes
        if candidate_total != self.record_count or offset != len(raw):
            raise DatasetError(f"imitation shard totals do not match payload: {self.path}")
        return tuple(groups)


class ImitationDataset:
    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read imitation manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )
        for shard in self.shards:
            shard.groups()

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
            shard = self.shards[int(shard_index)]
            groups = list(shard.groups())
            if shuffle:
                rng.shuffle(groups)
            for start in range(0, len(groups), group_batch_size):
                yield decode_imitation_groups(groups[start : start + group_batch_size])

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != IMITATION_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported imitation dataset schema version")
        if manifest.get("feature_schema") != IMITATION_FEATURE_SCHEMA:
            raise DatasetError("unsupported imitation feature schema")
        if manifest.get("position_feature_schema") != FEATURE_SCHEMA:
            raise DatasetError("unsupported imitation position schema")
        if manifest.get("target_schema") != IMITATION_TARGET_SCHEMA:
            raise DatasetError("unsupported imitation target schema")
        if manifest.get("group_header_size") != IMITATION_GROUP_HEADER_SIZE:
            raise DatasetError("unsupported imitation group header size")
        if manifest.get("candidate_record_size") != IMITATION_CANDIDATE_RECORD_SIZE:
            raise DatasetError("unsupported imitation candidate record size")
        if manifest.get("action_feature_size") != PROPOSAL_ACTION_FEATURE_SIZE:
            raise DatasetError("unsupported imitation action feature size")
        if not isinstance(manifest.get("teacher"), dict) or not manifest["teacher"].get(
            "weights_blake3"
        ):
            raise DatasetError("imitation manifest requires teacher provenance")
        if not isinstance(manifest.get("candidates"), dict):
            raise DatasetError("imitation manifest requires candidate metadata")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("imitation manifest shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("imitation record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("imitation group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("imitation game total does not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ImitationShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:IMITATION_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read imitation shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"imitation shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"imitation shard checksum mismatch: {path}")
        if len(header) != IMITATION_HEADER_SIZE:
            raise DatasetError(f"truncated imitation shard header: {path}")
        (
            magic,
            schema,
            header_size,
            group_header_size,
            candidate_record_size,
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
            magic != IMITATION_SHARD_MAGIC
            or schema != IMITATION_DATASET_SCHEMA_VERSION
            or header_size != IMITATION_HEADER_SIZE
            or group_header_size != IMITATION_GROUP_HEADER_SIZE
            or candidate_record_size != IMITATION_CANDIDATE_RECORD_SIZE
            or feature_hash != blake3.blake3(IMITATION_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(IMITATION_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible imitation shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"imitation shard header disagrees with manifest: {path}")
        expected_size = (
            IMITATION_HEADER_SIZE
            + group_count * IMITATION_GROUP_HEADER_SIZE
            + record_count * IMITATION_CANDIDATE_RECORD_SIZE
        )
        if stat.st_size != expected_size:
            raise DatasetError(f"imitation shard record count mismatch: {path}")
        return ImitationShard(
            path,
            record_count,
            group_count,
            game_count,
            first_game_index,
        )


def decode_imitation_groups(
    groups: Sequence[ImitationGroup],
) -> ImitationBatch:
    if not groups:
        raise ValueError("imitation batch must contain at least one group")
    group_count = len(groups)
    max_candidates = max(len(group.candidates) for group in groups)
    positions = np.zeros(group_count, dtype=_RECORD_DTYPE)
    actions = np.zeros((group_count, max_candidates), dtype=_PROPOSAL_ACTION_DTYPE)
    candidate_mask = np.zeros((group_count, max_candidates), dtype=np.bool_)
    teacher_mean = np.zeros((group_count, max_candidates), dtype=np.float32)
    teacher_stddev = np.zeros((group_count, max_candidates), dtype=np.float32)
    teacher_samples = np.zeros((group_count, max_candidates), dtype=np.float32)
    teacher_scored = np.zeros((group_count, max_candidates), dtype=np.bool_)
    selected = np.zeros((group_count, max_candidates), dtype=np.bool_)
    source_flags = np.zeros((group_count, max_candidates), dtype=np.uint8)
    immediate_rank = np.zeros((group_count, max_candidates), dtype=np.float32)
    immediate_score = np.zeros((group_count, max_candidates), dtype=np.float32)
    parent_immediate = np.zeros((group_count, max_candidates), dtype=np.float32)
    parent_remaining = np.zeros((group_count, max_candidates), dtype=np.float32)
    parent_total = np.zeros((group_count, max_candidates), dtype=np.float32)
    parent_rank = np.zeros((group_count, max_candidates), dtype=np.float32)
    parent_hidden = np.zeros((group_count, max_candidates, 64), dtype=np.float32)
    group_ids = np.zeros(group_count, dtype=np.uint64)

    for group_index, group in enumerate(groups):
        count = len(group.candidates)
        positions[group_index] = group.position[0]
        actions[group_index, :count] = group.candidates["action"]
        candidate_mask[group_index, :count] = True
        teacher_mean[group_index, group.selected_index] = 1.0
        teacher_scored[group_index, group.selected_index] = True
        selected[group_index, group.selected_index] = True
        immediate_rank[group_index, :count] = group.candidates["immediate_rank"]
        immediate_score[group_index, :count] = group.candidates["immediate_score"]
        group_ids[group_index] = group.group_id

    decoded = decode_records(positions)
    return ImitationBatch(
        board_entities=decoded.board_entities,
        board_mask=decoded.board_mask,
        market_entities=decoded.market_entities,
        market_mask=decoded.market_mask,
        global_features=decoded.global_features,
        action_features=decode_proposal_actions(actions),
        candidate_mask=mx.array(candidate_mask),
        teacher_mean=mx.array(teacher_mean),
        teacher_stddev=mx.array(teacher_stddev),
        teacher_samples=mx.array(teacher_samples),
        teacher_scored=mx.array(teacher_scored),
        selected=mx.array(selected),
        source_flags=mx.array(source_flags),
        immediate_rank=mx.array(immediate_rank),
        immediate_score=mx.array(immediate_score),
        parent_immediate=mx.array(parent_immediate),
        parent_remaining=mx.array(parent_remaining),
        parent_total=mx.array(parent_total),
        parent_rank=mx.array(parent_rank),
        parent_hidden=mx.array(parent_hidden),
        group_id=mx.array(group_ids.astype(np.int64)),
        game_index=decoded.game_index,
        turn=decoded.turn,
    )


def decode_proposal_actions(actions: np.ndarray) -> mx.array:
    actions = np.asarray(actions)
    presence = actions["wildlife_present"].astype(np.float32)[..., None]
    features = np.concatenate(
        [
            _one_hot(actions["draft_kind"], 2),
            _one_hot(actions["tile_slot"], 4),
            _one_hot(actions["wildlife_slot"], 4),
            _one_hot(actions["tile_terrain_a"], 5),
            _one_hot_with_none(actions["tile_terrain_b"], 5),
            _mask_bits(actions["tile_wildlife_mask"], 5),
            actions["tile_keystone"].astype(np.float32)[..., None],
            _one_hot(actions["drafted_wildlife"], 5),
            actions["tile_q"].astype(np.float32)[..., None] / 24.0,
            actions["tile_r"].astype(np.float32)[..., None] / 24.0,
            _one_hot(actions["rotation"], 6),
            presence,
            actions["wildlife_q"].astype(np.float32)[..., None] / 24.0 * presence,
            actions["wildlife_r"].astype(np.float32)[..., None] / 24.0 * presence,
            actions["replace_three_of_a_kind"].astype(np.float32)[..., None],
            actions["paid_wipe_count"].astype(np.float32)[..., None] / 20.0,
            _mask_bits(actions["paid_wipe_slot_mask"], 4),
            actions["paid_wipe_total_slots"].astype(np.float32)[..., None] / 80.0,
            actions["immediate_rank"].astype(np.float32)[..., None] / 4096.0,
            actions["immediate_score"].astype(np.float32)[..., None] / 100.0,
        ],
        axis=-1,
    )
    if features.shape[-1] != PROPOSAL_ACTION_DIM:
        raise AssertionError("decoded proposal action dimension drifted")
    if not np.array_equal(
        features[..., PROPOSAL_ACTION_IMMEDIATE_RANK_INDEX],
        actions["immediate_rank"].astype(np.float32) / 4096.0,
    ):
        raise AssertionError("decoded immediate-rank feature index drifted")
    if not np.array_equal(
        features[..., PROPOSAL_ACTION_IMMEDIATE_SCORE_INDEX],
        actions["immediate_score"].astype(np.float32) / 100.0,
    ):
        raise AssertionError("decoded immediate-score feature index drifted")
    return mx.array(features)


def decode_imitation_inference_bytes(
    payload: bytes | bytearray | memoryview,
    count: int,
) -> tuple[Batch, mx.array]:
    expected = 864 + count * PROPOSAL_ACTION_FEATURE_SIZE
    if len(payload) != expected:
        raise DatasetError(f"imitation payload has {len(payload)} bytes, expected {expected}")
    position = np.frombuffer(payload, dtype=_RECORD_DTYPE, count=1)
    actions = np.frombuffer(
        payload,
        dtype=_PROPOSAL_ACTION_DTYPE,
        count=count,
        offset=864,
    )
    return decode_records(position), decode_proposal_actions(actions[None, :])


def rotate_imitation_batch(
    batch: ImitationBatch,
    rotations: int | Sequence[int],
) -> ImitationBatch:
    """Rotate every board and candidate action by exact 60-degree steps."""
    group_count = batch.action_features.shape[0]
    steps = np.asarray(rotations, dtype=np.int32)
    if steps.ndim == 0:
        steps = np.full(group_count, int(steps), dtype=np.int32)
    if steps.shape != (group_count,) or np.any((steps < 0) | (steps >= 6)):
        raise ValueError("rotations must provide one value in [0, 5] per group")
    steps_mx = mx.array(steps)

    boards = batch.board_entities
    board_q, board_r = _rotate_axial(boards[..., 0], boards[..., 1], steps_mx)
    board_rotations = _rotate_one_hot(
        boards[..., 13:19],
        steps_mx,
        batch.board_mask,
    )
    rotated_boards = mx.concatenate(
        [
            board_q[..., None],
            board_r[..., None],
            boards[..., 2:13],
            board_rotations,
            boards[..., 19:],
        ],
        axis=-1,
    )

    actions = batch.action_features
    tile_q, tile_r = _rotate_axial(actions[..., 32], actions[..., 33], steps_mx)
    wildlife_q, wildlife_r = _rotate_axial(
        actions[..., 41],
        actions[..., 42],
        steps_mx,
    )
    action_rotations = _rotate_one_hot(
        actions[..., 34:40],
        steps_mx,
        batch.candidate_mask,
    )
    rotated_actions = mx.concatenate(
        [
            actions[..., :32],
            tile_q[..., None],
            tile_r[..., None],
            action_rotations,
            actions[..., 40:41],
            wildlife_q[..., None],
            wildlife_r[..., None],
            actions[..., 43:],
        ],
        axis=-1,
    )
    return replace(
        batch,
        board_entities=rotated_boards,
        action_features=rotated_actions,
    )


def randomly_rotate_imitation_batch(
    batch: ImitationBatch,
    seed: int,
) -> ImitationBatch:
    rng = np.random.default_rng(seed)
    rotations = rng.integers(0, 6, size=batch.action_features.shape[0])
    return rotate_imitation_batch(batch, rotations)


def _rotate_axial(
    q: mx.array,
    r: mx.array,
    steps: mx.array,
) -> tuple[mx.array, mx.array]:
    q_options = mx.stack([q, q + r, r, -q, -q - r, -r], axis=-1)
    r_options = mx.stack([r, -q, -q - r, -r, q, q + r], axis=-1)
    weight_shape = (steps.shape[0],) + (1,) * (q.ndim - 1) + (6,)
    weights = mx.eye(6)[steps].reshape(weight_shape)
    return mx.sum(q_options * weights, axis=-1), mx.sum(r_options * weights, axis=-1)


def _rotate_one_hot(
    values: mx.array,
    steps: mx.array,
    mask: mx.array,
) -> mx.array:
    step_shape = (steps.shape[0],) + (1,) * (values.ndim - 2)
    indices = (mx.argmax(values, axis=-1) + steps.reshape(step_shape)) % 6
    return mx.eye(6)[indices] * mask[..., None]


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
