from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.action_ranking_dataset import (
    ACTION_BOARD_ENTITY_DIM,
    ACTION_DIM,
    ACTION_FEATURE_SIZE,
)
from cascadia_mlx.conservative_advantage_dataset import (
    CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION,
    CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA,
    CONSERVATIVE_ADVANTAGE_HEADER_SIZE,
    CONSERVATIVE_ADVANTAGE_RECORD_SIZE,
    CONSERVATIVE_ADVANTAGE_SHARD_MAGIC,
    CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA,
    ConservativeAdvantageDataset,
)
from cascadia_mlx.dataset import RECORD_SIZE


def _position(game_index: int, turn: int) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, game_index, turn, 0, 4, 80)
    record[12:16] = bytes([2, 1, 1, 1])
    record[20:25] = bytes(5)
    record[72:80] = bytes([1, 254, 0, 255, 0, 1, 255, 0])
    record[80:88] = bytes([0, 0, 1, 255, 0, 2, 3, 0])
    record[808:840] = bytes([255] * 32)
    return bytes(record)


def _action(rank: int) -> bytes:
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
        rank,
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


def _input(rank: int) -> bytes:
    return _position(7, 65) + _action(rank)


def _record(index: int, selected: bool, lower_bound: float) -> bytes:
    record = bytearray(CONSERVATIVE_ADVANTAGE_RECORD_SIZE)
    struct.pack_into(
        "<QHHB3xfff",
        record,
        0,
        11,
        index,
        2,
        selected,
        lower_bound + 0.5,
        0.25,
        lower_bound,
    )
    record[28:60] = bytes([1] * 32)
    record[60:92] = bytes([index + 2] * 32)
    record[92:1008] = _input(1)
    record[1008:] = _input(index + 2)
    return bytes(record)


def _write_dataset(root: Path) -> None:
    root.mkdir()
    records = _record(0, True, 0.75) + _record(1, False, -0.25)
    header = struct.pack(
        "<8sHHHHIIIBBBBQ32s32s24s",
        CONSERVATIVE_ADVANTAGE_SHARD_MAGIC,
        CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION,
        CONSERVATIVE_ADVANTAGE_HEADER_SIZE,
        CONSERVATIVE_ADVANTAGE_RECORD_SIZE,
        916,
        2,
        1,
        1,
        0,
        4,
        0,
        0,
        7,
        blake3.blake3(CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA.encode()).digest(),
        blake3.blake3(CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA.encode()).digest(),
        bytes(24),
    )
    shard = root / "shard-00000.cav"
    shard.write_bytes(header + records)
    manifest = {
        "schema_version": CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION,
        "feature_schema": CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA,
        "target_schema": CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA,
        "record_size": CONSERVATIVE_ADVANTAGE_RECORD_SIZE,
        "action_position_record_size": 916,
        "split": "train",
        "teacher": {"determinizations": 8, "confidence_percent": 90},
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


def test_conservative_advantage_dataset_decodes_paired_actions(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "advantage")
    dataset = ConservativeAdvantageDataset(tmp_path / "advantage")
    batch = next(dataset.batches(2))

    assert np.asarray(batch.anchor_board_entities).shape == (
        1,
        2,
        4,
        23,
        ACTION_BOARD_ENTITY_DIM,
    )
    assert np.asarray(batch.candidate_action_features).shape == (1, 2, ACTION_DIM)
    assert np.asarray(batch.lower_bound).tolist() == [[0.75, -0.25]]
    assert np.asarray(batch.selected).tolist() == [[True, False]]
