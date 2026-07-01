from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.dataset import (
    DATASET_SCHEMA_VERSION,
    ENTITY_DIM,
    FEATURE_SCHEMA,
    GLOBAL_DIM,
    RECORD_SIZE,
    SHARD_HEADER_SIZE,
    SHARD_MAGIC,
    TARGET_DIM,
    TARGET_SCHEMA,
    Dataset,
    DatasetError,
)


def _write_dataset(root: Path, *, partial_market: bool = False) -> Path:
    root.mkdir()
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, 17, 5, 1, 4, 80)
    record[12:16] = bytes([3, 3, 3, 3])
    record[16:20] = bytes([1, 2, 3, 4])
    record[20:25] = bytes([0, 0, 0, 0, 0])
    record[32:52] = bytes(range(20))
    record[52:72] = bytes(range(1, 21))
    entity = bytes([0, 0, 0, 255, 0, 1, 255, 1])
    for board in range(4):
        for tile in range(3):
            offset = 72 + (board * 23 + tile) * 8
            record[offset : offset + 8] = entity
    markets = (
        [
            bytes([255] * 8),
            bytes([0, 255, 1, 255, 1, 0, 0, 0]),
            bytes([255, 255, 0, 2, 0, 0, 0, 0]),
            bytes([0, 255, 1, 0, 1, 0, 0, 0]),
        ]
        if partial_market
        else [bytes([0, 255, 1, 0, 1, 0, 0, 0])] * 4
    )
    for slot, market in enumerate(markets):
        offset = 808 + slot * 8
        record[offset : offset + 8] = market
    struct.pack_into("<11H", record, 840, *range(1, 12))

    feature_hash = blake3.blake3(FEATURE_SCHEMA.encode()).digest()
    header = struct.pack(
        "<8sHHHHIIQBBBB5s7s32s",
        SHARD_MAGIC,
        DATASET_SCHEMA_VERSION,
        SHARD_HEADER_SIZE,
        RECORD_SIZE,
        TARGET_DIM,
        1,
        1,
        17,
        0,
        0,
        4,
        0,
        bytes(5),
        bytes(7),
        feature_hash,
    )
    shard = root / "shard-00000.csd"
    shard.write_bytes(header + record)
    checksum = blake3.blake3(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": "test",
        "feature_schema": FEATURE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "split": "train",
        "strategy": "random-v1",
        "first_game_index": 17,
        "requested_games": 1,
        "completed_games": 1,
        "total_records": 1,
        "created_unix_seconds": 0,
        "updated_unix_seconds": 0,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 17,
                "game_count": 1,
                "record_count": 1,
                "byte_count": shard.stat().st_size,
                "blake3": checksum,
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return shard


def test_dataset_decodes_fixed_record_schema(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset")
    dataset = Dataset(tmp_path / "dataset")
    batch = next(dataset.batches(8))

    assert dataset.sample_count == 1
    assert batch.board_entities.shape == (1, 4, 23, ENTITY_DIM)
    assert batch.market_entities.shape == (1, 4, ENTITY_DIM)
    assert batch.global_features.shape == (1, GLOBAL_DIM)
    assert batch.targets.shape == (1, TARGET_DIM)
    assert np.asarray(batch.board_mask).sum() == 12
    assert np.asarray(batch.targets).tolist() == [list(range(1, 12))]


def test_dataset_decodes_empty_and_partial_market_slots(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset", partial_market=True)
    batch = next(Dataset(tmp_path / "dataset").batches(8))
    market = np.asarray(batch.market_entities)[0]
    mask = np.asarray(batch.market_mask)[0]

    assert mask.tolist() == [False, True, True, True]
    assert market[0].sum() == 0.0
    assert market[1, 29] == 1.0
    assert market[2, 2:7].sum() == 0.0
    assert market[2, 19:24].sum() == 0.0
    assert market[2, 26] == 1.0


def test_dataset_shuffle_is_deterministic(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset")
    dataset = Dataset(tmp_path / "dataset")

    left = list(dataset.batches(1, shuffle=True, seed=9))
    right = list(dataset.batches(1, shuffle=True, seed=9))
    assert [np.asarray(batch.game_index).tolist() for batch in left] == [
        np.asarray(batch.game_index).tolist() for batch in right
    ]


def test_dataset_rejects_tampered_shard(tmp_path: Path) -> None:
    shard = _write_dataset(tmp_path / "dataset")
    content = bytearray(shard.read_bytes())
    content[-1] ^= 1
    shard.write_bytes(content)

    with pytest.raises(DatasetError, match="checksum"):
        Dataset(tmp_path / "dataset")
