"""Streaming decoder for signed score-to-go value datasets."""

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
    RECORD_SIZE,
    TARGET_DIM,
    DatasetError,
    decode_records,
)

SCORE_TO_GO_DATASET_SCHEMA_VERSION = 1
SCORE_TO_GO_TARGET_SCHEMA = "signed-score-to-go-components-v1"
SCORE_TO_GO_SHARD_MAGIC = b"CSD2STG\0"
SCORE_TO_GO_HEADER_SIZE = 128
SCORE_TO_GO_RECORD_SIZE = RECORD_SIZE + TARGET_DIM * 4

_HEADER = struct.Struct("<8sHHHHHII4sQ32s32s26s")
_RECORD = np.dtype(
    {
        "names": ["position", "current", "residual"],
        "formats": [_RECORD_DTYPE, ("<u2", (TARGET_DIM,)), ("<i2", (TARGET_DIM,))],
        "offsets": [0, RECORD_SIZE, RECORD_SIZE + TARGET_DIM * 2],
        "itemsize": SCORE_TO_GO_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class ScoreToGoBatch:
    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    targets: mx.array
    current_targets: mx.array
    final_targets: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class ScoreToGoShard:
    path: Path
    record_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=_RECORD,
            offset=SCORE_TO_GO_HEADER_SIZE,
            shape=(self.record_count,),
        )


class ScoreToGoDataset:
    """Checksummed score-to-go shards with exact residual identity validation."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read score-to-go manifest: {error}") from error
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
    ) -> Iterator[ScoreToGoBatch]:
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
                yield decode_score_to_go_records(records[indices[start : start + batch_size]])

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != SCORE_TO_GO_DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported score-to-go dataset schema")
        if manifest.get("feature_schema") != FEATURE_SCHEMA:
            raise DatasetError("unsupported score-to-go feature schema")
        if manifest.get("target_schema") != SCORE_TO_GO_TARGET_SCHEMA:
            raise DatasetError("unsupported score-to-go target schema")
        if manifest.get("record_size") != SCORE_TO_GO_RECORD_SIZE:
            raise DatasetError("unsupported score-to-go record size")
        if manifest.get("position_record_size") != RECORD_SIZE:
            raise DatasetError("unsupported score-to-go position record size")
        if manifest.get("target_dim") != TARGET_DIM:
            raise DatasetError("unsupported score-to-go target dimension")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("score-to-go shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("score-to-go record total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("score-to-go game total does not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> ScoreToGoShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:SCORE_TO_GO_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read score-to-go shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"score-to-go shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"score-to-go shard checksum mismatch: {path}")
        if len(header) != SCORE_TO_GO_HEADER_SIZE:
            raise DatasetError(f"truncated score-to-go shard header: {path}")
        (
            magic,
            schema,
            header_size,
            record_size,
            position_record_size,
            target_dim,
            record_count,
            game_count,
            split_and_reserved,
            first_game_index,
            feature_hash,
            target_hash,
            _reserved,
        ) = _HEADER.unpack(header)
        if (
            magic != SCORE_TO_GO_SHARD_MAGIC
            or schema != SCORE_TO_GO_DATASET_SCHEMA_VERSION
            or header_size != SCORE_TO_GO_HEADER_SIZE
            or record_size != SCORE_TO_GO_RECORD_SIZE
            or position_record_size != RECORD_SIZE
            or target_dim != TARGET_DIM
            or feature_hash != blake3.blake3(FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(SCORE_TO_GO_TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible score-to-go shard header: {path}")
        split_codes = {"train": 0, "validation": 1, "test": 2, "final": 3}
        if split_and_reserved[0] != split_codes[self.split]:
            raise DatasetError(f"score-to-go shard split mismatch: {path}")
        if (
            record_count != int(entry["record_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"score-to-go shard header disagrees with manifest: {path}")
        expected_size = SCORE_TO_GO_HEADER_SIZE + record_count * SCORE_TO_GO_RECORD_SIZE
        if stat.st_size != expected_size:
            raise DatasetError(f"score-to-go record count does not match file size: {path}")

        records = np.memmap(
            path,
            mode="r",
            dtype=_RECORD,
            offset=SCORE_TO_GO_HEADER_SIZE,
            shape=(record_count,),
        )
        final_targets = records["position"]["targets"].astype(np.int32)
        current_targets = records["current"].astype(np.int32)
        residual_targets = records["residual"].astype(np.int32)
        if not np.array_equal(current_targets + residual_targets, final_targets):
            raise DatasetError(f"score-to-go target identity failed: {path}")
        return ScoreToGoShard(path, record_count, game_count, first_game_index)


def decode_score_to_go_records(records: np.ndarray) -> ScoreToGoBatch:
    records = np.asarray(records)
    base = decode_records(records["position"])
    current = records["current"].astype(np.float32)
    residual = records["residual"].astype(np.float32)
    final = records["position"]["targets"].astype(np.float32)
    if not np.array_equal(current + residual, final):
        raise DatasetError("score-to-go batch target identity failed")
    return ScoreToGoBatch(
        board_entities=base.board_entities,
        board_mask=base.board_mask,
        market_entities=base.market_entities,
        market_mask=base.market_mask,
        global_features=base.global_features,
        targets=mx.array(residual),
        current_targets=mx.array(current),
        final_targets=mx.array(final),
        game_index=base.game_index,
        turn=base.turn,
    )


def rotate_score_to_go_batch(
    batch: ScoreToGoBatch,
    rotations: int | Sequence[int],
) -> ScoreToGoBatch:
    """Rotate every public board by exact 60-degree steps."""
    batch_size = batch.board_entities.shape[0]
    steps = np.asarray(rotations, dtype=np.int32)
    if steps.ndim == 0:
        steps = np.full(batch_size, int(steps), dtype=np.int32)
    if steps.shape != (batch_size,) or np.any((steps < 0) | (steps >= 6)):
        raise ValueError("rotations must provide one value in [0, 5] per position")
    steps_mx = mx.array(steps)
    boards = batch.board_entities
    board_q, board_r = _rotate_axial(boards[..., 0], boards[..., 1], steps_mx)
    board_rotations = _rotate_one_hot(
        boards[..., 13:19],
        steps_mx,
        batch.board_mask,
    )
    return replace(
        batch,
        board_entities=mx.concatenate(
            [
                board_q[..., None],
                board_r[..., None],
                boards[..., 2:13],
                board_rotations,
                boards[..., 19:],
            ],
            axis=-1,
        ),
    )


def randomly_rotate_score_to_go_batch(batch: ScoreToGoBatch, seed: int) -> ScoreToGoBatch:
    rng = np.random.default_rng(seed)
    return rotate_score_to_go_batch(
        batch,
        rng.integers(0, 6, size=batch.board_entities.shape[0]),
    )


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


def _rotate_one_hot(values: mx.array, steps: mx.array, mask: mx.array) -> mx.array:
    step_shape = (steps.shape[0],) + (1,) * (values.ndim - 2)
    indices = (mx.argmax(values, axis=-1) + steps.reshape(step_shape)) % 6
    return mx.eye(6)[indices] * mask[..., None]


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
