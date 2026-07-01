from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.action_ranking_dataset import ACTION_FEATURE_SIZE
from cascadia_mlx.dataset import RECORD_SIZE, DatasetError
from cascadia_mlx.public_beam_value_dataset import (
    PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION,
    PUBLIC_BEAM_VALUE_FEATURE_SCHEMA,
    PUBLIC_BEAM_VALUE_HEADER_SIZE,
    PUBLIC_BEAM_VALUE_RECORD_SIZE,
    PUBLIC_BEAM_VALUE_SHARD_MAGIC,
    PUBLIC_BEAM_VALUE_TARGET_SCHEMA,
    PublicBeamValueDataset,
)


def _position(game_index: int, turn: int) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, game_index, turn, 0, 4, 80)
    record[12:16] = bytes([2, 1, 1, 1])
    record[20:25] = bytes(5)
    record[72:80] = bytes([1, 254, 0, 255, 0, 1, 255, 0])
    record[80:88] = bytes([0, 0, 1, 255, 0, 2, 3, 0])
    record[808:840] = bytes([255] * 32)
    return bytes(record)


def _action(rank: int, score: int) -> bytes:
    action = bytearray(ACTION_FEATURE_SIZE)
    struct.pack_into("<HH", action, 18, rank, score)
    struct.pack_into("<h", action, 22, score - 61)
    return bytes(action)


def _record(index: int) -> bytes:
    record = bytearray(PUBLIC_BEAM_VALUE_RECORD_SIZE)
    struct.pack_into(
        "<QHHH2xffff",
        record,
        0,
        17,
        index,
        2,
        61,
        92.0 + index,
        92.5 + index,
        1.25,
        1.5,
    )
    record[32:64] = bytes([7] * 32)
    record[64:96] = bytes([index + 1] * 32)
    record[96 : 96 + RECORD_SIZE] = _position(40_000, 64)
    record[96 + RECORD_SIZE :] = _action(index + 1, 64 + index)
    return bytes(record)


def _write_dataset(root: Path) -> Path:
    root.mkdir()
    records = _record(0) + _record(1)
    header = struct.pack(
        "<8sHHHHIIIBBBBQ32s32s24s",
        PUBLIC_BEAM_VALUE_SHARD_MAGIC,
        PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION,
        PUBLIC_BEAM_VALUE_HEADER_SIZE,
        PUBLIC_BEAM_VALUE_RECORD_SIZE,
        916,
        2,
        1,
        1,
        0,
        4,
        0,
        0,
        40_000,
        blake3.blake3(PUBLIC_BEAM_VALUE_FEATURE_SCHEMA.encode()).digest(),
        blake3.blake3(PUBLIC_BEAM_VALUE_TARGET_SCHEMA.encode()).digest(),
        bytes(24),
    )
    shard = root / "shard-00000.pbv"
    shard.write_bytes(header + records)
    manifest = {
        "schema_version": PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION,
        "dataset_id": "public-beam-value-test",
        "feature_schema": PUBLIC_BEAM_VALUE_FEATURE_SCHEMA,
        "target_schema": PUBLIC_BEAM_VALUE_TARGET_SCHEMA,
        "record_size": PUBLIC_BEAM_VALUE_RECORD_SIZE,
        "action_position_record_size": 916,
        "split": "train",
        "teacher": {
            "final_personal_turns": 5,
            "recorded_personal_turns": [5, 4, 3, 2],
            "determinizations_per_batch": 8,
            "batches": 2,
            "wildlife_candidates": 2,
            "beam_width": 16,
            "seed_schema": "public-state-hash-domain-separated-v1",
        },
        "completed_games": 1,
        "total_groups": 1,
        "total_records": 2,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 40_000,
                "game_count": 1,
                "group_count": 1,
                "record_count": 2,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return shard


def test_public_beam_value_dataset_round_trips_targets_and_inputs(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "public-value")
    dataset = PublicBeamValueDataset(tmp_path / "public-value")
    batch = next(dataset.batches(4))

    assert dataset.group_count == 1
    assert dataset.candidate_count == 2
    assert np.asarray(batch.candidate_mask).tolist() == [[True, True]]
    np.testing.assert_allclose(np.asarray(batch.target_mean), [[92.25, 93.25]])
    np.testing.assert_allclose(np.asarray(batch.current_base_score), [[61.0, 61.0]])
    np.testing.assert_allclose(np.asarray(batch.immediate_score), [[64.0, 65.0]])
    assert np.asarray(batch.action_features).shape == (1, 2, 63)
    assert np.asarray(batch.game_index).tolist() == [[40_000, 40_000]]


def test_public_beam_value_dataset_rejects_checksum_drift(tmp_path: Path) -> None:
    root = tmp_path / "public-value"
    shard = _write_dataset(root)
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(DatasetError, match="checksum"):
        PublicBeamValueDataset(root)
