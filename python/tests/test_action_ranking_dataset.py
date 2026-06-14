from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.action_ranking_dataset import (
    ACTION_BOARD_ENTITY_DIM,
    ACTION_DIM,
    ACTION_FEATURE_SCHEMA,
    ACTION_FEATURE_SIZE,
    ACTION_RANKING_DATASET_SCHEMA_VERSION,
    ACTION_RANKING_HEADER_SIZE,
    ACTION_RANKING_RECORD_SIZE,
    ACTION_RANKING_SHARD_MAGIC,
    ACTION_RANKING_TARGET_SCHEMA,
    ActionRankingDataset,
)
from cascadia_mlx.dataset import FEATURE_SCHEMA, RECORD_SIZE


def _position(game_index: int, turn: int) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, game_index, turn, 0, 4, 80)
    record[12:16] = bytes([2, 1, 1, 1])
    record[20:25] = bytes(5)
    record[72:80] = bytes([1, 254, 0, 255, 0, 1, 255, 0])
    record[80:88] = bytes([0, 0, 1, 255, 0, 2, 3, 0])
    record[808:840] = bytes([255] * 32)
    return bytes(record)


def _action() -> bytes:
    action = bytearray(ACTION_FEATURE_SIZE)
    struct.pack_into(
        "<BBBBBBBBbbBBbbBBBBHH11h",
        action,
        0,
        0,
        1,
        1,
        0,
        255,
        0b00101,
        1,
        3,
        1,
        -2,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        41,
        1,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
    )
    return bytes(action)


def _candidate(group_id: int, index: int, count: int) -> bytes:
    record = bytearray(ACTION_RANKING_RECORD_SIZE)
    struct.pack_into(
        "<QHHHHff",
        record,
        0,
        group_id,
        index,
        count,
        2,
        41,
        50.0 - index,
        1.5,
    )
    record[24:56] = bytes([index] * 32)
    record[56 : 56 + RECORD_SIZE] = _position(7, 5)
    record[56 + RECORD_SIZE :] = _action()
    return bytes(record)


def _write_dataset(root: Path) -> None:
    root.mkdir()
    records = _candidate(11, 0, 2) + _candidate(11, 1, 2)
    header = struct.pack(
        "<8sHHHHIIIBBBBQ32s32s8s",
        ACTION_RANKING_SHARD_MAGIC,
        ACTION_RANKING_DATASET_SCHEMA_VERSION,
        ACTION_RANKING_HEADER_SIZE,
        ACTION_RANKING_RECORD_SIZE,
        ACTION_FEATURE_SIZE,
        2,
        1,
        1,
        0,
        4,
        0,
        0,
        7,
        blake3.blake3(ACTION_FEATURE_SCHEMA.encode()).digest(),
        blake3.blake3(ACTION_RANKING_TARGET_SCHEMA.encode()).digest(),
        bytes(8),
    )
    shard = root / "shard-00000.car"
    shard.write_bytes(header + records)
    manifest = {
        "schema_version": ACTION_RANKING_DATASET_SCHEMA_VERSION,
        "dataset_id": "action-ranking-test",
        "feature_schema": ACTION_FEATURE_SCHEMA,
        "position_feature_schema": FEATURE_SCHEMA,
        "target_schema": ACTION_RANKING_TARGET_SCHEMA,
        "record_size": ACTION_RANKING_RECORD_SIZE,
        "action_feature_size": ACTION_FEATURE_SIZE,
        "split": "train",
        "source": {"manifest_blake3": "source"},
        "completed_games": 1,
        "total_groups": 1,
        "total_records": 2,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 7,
                "game_count": 1,
                "group_count": 1,
                "record_count": 2,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))


def test_action_ranking_dataset_marks_changed_entities_and_decodes_action(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "ranking")
    dataset = ActionRankingDataset(tmp_path / "ranking")
    batch = next(dataset.batches(2))

    board = np.asarray(batch.board_entities)
    actions = np.asarray(batch.action_features)
    assert board.shape == (1, 2, 4, 23, ACTION_BOARD_ENTITY_DIM)
    assert actions.shape == (1, 2, ACTION_DIM)
    assert board[0, 0, 0, 0, -2:].tolist() == [1.0, 0.0]
    assert board[0, 0, 0, 1, -2:].tolist() == [0.0, 1.0]
    assert np.asarray(batch.immediate_rank).tolist() == [[2.0, 2.0]]
    assert np.asarray(batch.immediate_score).tolist() == [[41.0, 41.0]]
    np.testing.assert_allclose(actions[0, 0, -2:], [2 / 32, 0.41])
