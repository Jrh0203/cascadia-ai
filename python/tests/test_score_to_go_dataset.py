from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.dataset import RECORD_SIZE, TARGET_DIM
from cascadia_mlx.score_to_go_dataset import (
    SCORE_TO_GO_DATASET_SCHEMA_VERSION,
    SCORE_TO_GO_HEADER_SIZE,
    SCORE_TO_GO_RECORD_SIZE,
    SCORE_TO_GO_SHARD_MAGIC,
    SCORE_TO_GO_TARGET_SCHEMA,
    ScoreToGoDataset,
    rotate_score_to_go_batch,
)


def _position(final: np.ndarray) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, 7, 60, 0, 4, 80)
    record[12:16] = bytes([20, 20, 20, 20])
    record[20:25] = bytes(5)
    struct.pack_into("<11H", record, 840, *final.tolist())
    return bytes(record)


def test_score_to_go_dataset_decodes_signed_residuals(tmp_path: Path) -> None:
    root = tmp_path / "score-to-go"
    root.mkdir()
    current = np.array([5, 5, 5, 5, 5, 4, 6, 7, 8, 9, 2], dtype=np.uint16)
    residual = np.array([1, 0, 2, 0, 1, -1, 3, 0, 1, 2, 1], dtype=np.int16)
    final = (current.astype(np.int32) + residual.astype(np.int32)).astype(np.uint16)
    payload = (
        _position(final)
        + struct.pack("<11H", *current.tolist())
        + struct.pack("<11h", *residual.tolist())
    )
    header = struct.pack(
        "<8sHHHHHII4sQ32s32s26s",
        SCORE_TO_GO_SHARD_MAGIC,
        SCORE_TO_GO_DATASET_SCHEMA_VERSION,
        SCORE_TO_GO_HEADER_SIZE,
        SCORE_TO_GO_RECORD_SIZE,
        RECORD_SIZE,
        TARGET_DIM,
        1,
        1,
        bytes([0, 4, 0, 0]),
        7,
        blake3.blake3(b"compact-entity-v2").digest(),
        blake3.blake3(SCORE_TO_GO_TARGET_SCHEMA.encode()).digest(),
        bytes(26),
    )
    shard = root / "shard-00000.stg"
    shard.write_bytes(header + payload)
    manifest = {
        "schema_version": SCORE_TO_GO_DATASET_SCHEMA_VERSION,
        "feature_schema": "compact-entity-v2",
        "target_schema": SCORE_TO_GO_TARGET_SCHEMA,
        "record_size": SCORE_TO_GO_RECORD_SIZE,
        "position_record_size": RECORD_SIZE,
        "target_dim": TARGET_DIM,
        "split": "train",
        "completed_games": 1,
        "total_records": 1,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 7,
                "game_count": 1,
                "record_count": 1,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))

    batch = next(ScoreToGoDataset(root).batches(1))

    assert np.asarray(batch.targets).tolist() == [residual.astype(np.float32).tolist()]
    assert np.asarray(batch.current_targets).tolist() == [current.astype(np.float32).tolist()]
    assert np.asarray(batch.final_targets).tolist() == [final.astype(np.float32).tolist()]


def test_score_to_go_rotation_updates_coordinates_and_tile_orientation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "score-to-go"
    root.mkdir()
    current = np.zeros(TARGET_DIM, dtype=np.uint16)
    residual = np.zeros(TARGET_DIM, dtype=np.int16)
    position = bytearray(_position(current))
    position[12:16] = bytes([1, 0, 0, 0])
    position[72:80] = bytes([1, 0, 0, 255, 1, 1, 255, 1])
    payload = (
        bytes(position)
        + struct.pack("<11H", *current.tolist())
        + struct.pack("<11h", *residual.tolist())
    )
    header = struct.pack(
        "<8sHHHHHII4sQ32s32s26s",
        SCORE_TO_GO_SHARD_MAGIC,
        SCORE_TO_GO_DATASET_SCHEMA_VERSION,
        SCORE_TO_GO_HEADER_SIZE,
        SCORE_TO_GO_RECORD_SIZE,
        RECORD_SIZE,
        TARGET_DIM,
        1,
        1,
        bytes([0, 4, 0, 0]),
        7,
        blake3.blake3(b"compact-entity-v2").digest(),
        blake3.blake3(SCORE_TO_GO_TARGET_SCHEMA.encode()).digest(),
        bytes(26),
    )
    shard = root / "shard-00000.stg"
    shard.write_bytes(header + payload)
    manifest = {
        "schema_version": SCORE_TO_GO_DATASET_SCHEMA_VERSION,
        "feature_schema": "compact-entity-v2",
        "target_schema": SCORE_TO_GO_TARGET_SCHEMA,
        "record_size": SCORE_TO_GO_RECORD_SIZE,
        "position_record_size": RECORD_SIZE,
        "target_dim": TARGET_DIM,
        "split": "train",
        "completed_games": 1,
        "total_records": 1,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 7,
                "game_count": 1,
                "record_count": 1,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    batch = next(ScoreToGoDataset(root).batches(1))

    rotated = rotate_score_to_go_batch(batch, 1)
    board = np.asarray(rotated.board_entities)[0, 0, 0]

    np.testing.assert_allclose(board[:2], [1.0 / 24.0, -1.0 / 24.0])
    assert np.argmax(board[13:19]) == 2
    np.testing.assert_array_equal(
        np.asarray(rotated.targets),
        np.asarray(batch.targets),
    )
