from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.dataset import FEATURE_SCHEMA, RECORD_SIZE
from cascadia_mlx.ranking_dataset import (
    RANKING_DATASET_SCHEMA_VERSION,
    RANKING_HEADER_SIZE,
    RANKING_RECORD_SIZE,
    RANKING_SHARD_MAGIC,
    RANKING_TARGET_SCHEMA,
    RankingDataset,
)


def _position(game_index: int, turn: int) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, game_index, turn, 0, 4, 80)
    record[12:16] = bytes([3, 3, 3, 3])
    record[20:25] = bytes(5)
    record[808:840] = bytes([255] * 32)
    return bytes(record)


def _candidate(group_id: int, index: int, count: int) -> bytes:
    record = bytearray(RANKING_RECORD_SIZE)
    struct.pack_into(
        "<QHHHHff",
        record,
        0,
        group_id,
        index,
        count,
        index + 1,
        40 + index,
        50.0 - index,
        1.5,
    )
    record[24:56] = bytes([index] * 32)
    record[56:] = _position(7, 5)
    return bytes(record)


def _write_dataset(root: Path) -> Path:
    root.mkdir()
    records = _candidate(11, 0, 2) + _candidate(11, 1, 2) + _candidate(12, 0, 1)
    header = struct.pack(
        "<8sHHHHIIIBBBBQ32s32s8s",
        RANKING_SHARD_MAGIC,
        RANKING_DATASET_SCHEMA_VERSION,
        RANKING_HEADER_SIZE,
        RANKING_RECORD_SIZE,
        0,
        3,
        2,
        1,
        0,
        4,
        0,
        0,
        7,
        blake3.blake3(FEATURE_SCHEMA.encode()).digest(),
        blake3.blake3(RANKING_TARGET_SCHEMA.encode()).digest(),
        bytes(8),
    )
    shard = root / "shard-00000.csr"
    shard.write_bytes(header + records)
    manifest = {
        "schema_version": RANKING_DATASET_SCHEMA_VERSION,
        "dataset_id": "ranking-test",
        "feature_schema": FEATURE_SCHEMA,
        "target_schema": RANKING_TARGET_SCHEMA,
        "record_size": RANKING_RECORD_SIZE,
        "split": "train",
        "completed_games": 1,
        "total_groups": 2,
        "total_records": 3,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 7,
                "game_count": 1,
                "group_count": 2,
                "record_count": 3,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return shard


def test_ranking_dataset_preserves_complete_groups(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "ranking")
    dataset = RankingDataset(tmp_path / "ranking")
    batch = next(dataset.batches(2))

    assert dataset.group_count == 2
    assert dataset.candidate_count == 3
    assert batch.candidate_mask.shape == (2, 2)
    assert np.asarray(batch.candidate_mask).tolist() == [[True, True], [True, False]]
    assert np.asarray(batch.teacher_mean).tolist() == [[50.0, 49.0], [50.0, 0.0]]
    assert batch.board_entities.shape[:2] == (2, 2)
    assert np.asarray(batch.game_index)[0, 0] == 7
