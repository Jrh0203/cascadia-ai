from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.action_ranking_dataset import ACTION_FEATURE_SIZE
from cascadia_mlx.counterfactual_advantage_dataset import (
    COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SELECTION,
    COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION,
    COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA,
    COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE,
    COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE,
    COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC,
    COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING,
    COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA,
    COUNTERFACTUAL_ADVANTAGE_TEACHER,
    CounterfactualAdvantageDataset,
)
from cascadia_mlx.dataset import RECORD_SIZE, DatasetError


def _position(game_index: int, turn: int, active_seat: int) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, game_index, turn, active_seat, 4, 80)
    record[12:16] = bytes([2, 1, 1, 1])
    record[20:25] = bytes(5)
    record[72:80] = bytes([1, 254, 0, 255, 0, 1, 255, 0])
    record[80:88] = bytes([0, 0, 1, 255, 0, 2, 3, 0])
    record[808:840] = bytes([255] * 32)
    return bytes(record)


def _action(rank: int, score: int, current: int) -> bytes:
    action = bytearray(ACTION_FEATURE_SIZE)
    struct.pack_into("<HH", action, 18, rank, score)
    struct.pack_into("<h", action, 22, score - current)
    return bytes(action)


def _record(game_index: int, group_index: int) -> bytes:
    turn = group_index * 20
    current = 61 + group_index
    record = bytearray(COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE)
    struct.pack_into("<QBBB", record, 0, 100 + group_index, 1, 4, 12)
    struct.pack_into("<H", record, 16, current)
    record[38 + 15] = 81 - turn
    record[68 : 68 + RECORD_SIZE] = _position(game_index, turn, turn % 4)
    for sample in range(12):
        record[932 + sample * 32 : 932 + (sample + 1) * 32] = bytes([sample + 1]) * 32

    candidate_start = 1444
    for candidate in range(4):
        start = candidate_start + candidate * 1308
        record[start : start + 32] = bytes([candidate + 1]) * 32
        struct.pack_into("<ff", record, start + 32, 70.0 + candidate, 1.0)
        score = current + candidate + 1
        action_position = _position(game_index, turn + 1, turn % 4) + _action(
            candidate + 1, score, current
        )
        record[start + 40 : start + 956] = action_position
        for sample in range(12):
            terminal = 90 + group_index + candidate + sample % 2
            struct.pack_into("<H", record, start + 956 + sample * 22, terminal)
    return bytes(record)


def write_counterfactual_advantage_dataset(
    root: Path,
    *,
    split: str,
    game_index: int,
) -> Path:
    root.mkdir()
    teacher = {
        "strategy_id": COUNTERFACTUAL_ADVANTAGE_TEACHER,
        "immediate_candidates": 8,
        "habitat_candidates": 6,
        "determinizations": 4,
        "greedy_plies": 4,
        "candidate_count": 4,
        "groups_per_game": 4,
        "samples_per_candidate": 12,
        "sample_seed_domain": "cascadia-v2-counterfactual-advantage-v1",
        "candidate_selection": COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SELECTION,
        "stabilization_conditioning": COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING,
    }
    records = b"".join(_record(game_index, group_index) for group_index in range(4))
    header = struct.pack(
        "<8sHHHHHHHHIIBBBBIQ32s32s32s16s",
        COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC,
        COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION,
        COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE,
        COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE,
        916,
        11,
        4,
        16,
        30,
        4,
        1,
        {"train": 0, "validation": 1}[split],
        4,
        4,
        12,
        4,
        game_index,
        blake3.blake3(COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA.encode()).digest(),
        blake3.blake3(COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA.encode()).digest(),
        blake3.blake3(json.dumps(teacher, separators=(",", ":")).encode()).digest(),
        bytes(16),
    )
    shard = root / "shard-00000.cfa"
    shard.write_bytes(header + records)
    manifest = {
        "schema_version": COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION,
        "dataset_id": f"counterfactual-advantage-test-{split}-{game_index}",
        "feature_schema": COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA,
        "target_schema": COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA,
        "record_size": COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE,
        "action_position_record_size": 916,
        "target_dim": 11,
        "maximum_candidates": 4,
        "maximum_samples": 16,
        "game": {
            "player_count": 4,
            "mode": "Standard",
            "scoring_cards": {
                "bear": "A",
                "elk": "A",
                "salmon": "A",
                "hawk": "A",
                "fox": "A",
            },
            "habitat_bonuses": False,
        },
        "split": split,
        "teacher": teacher,
        "first_game_index": game_index,
        "requested_games": 1,
        "completed_games": 1,
        "total_groups": 4,
        "total_candidates": 16,
        "total_continuations": 192,
        "collection_milliseconds": 1,
        "created_unix_seconds": 1,
        "updated_unix_seconds": 1,
        "provenance": {},
        "shards": [
            {
                "file": shard.name,
                "first_game_index": game_index,
                "game_count": 1,
                "record_count": 4,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return shard


def test_counterfactual_advantage_dataset_decodes_r12_groups(tmp_path: Path) -> None:
    root = tmp_path / "counterfactual"
    write_counterfactual_advantage_dataset(root, split="train", game_index=9_996)

    dataset = CounterfactualAdvantageDataset(root)
    batch = next(dataset.batches(4))

    assert dataset.group_count == 4
    assert dataset.candidate_count == 16
    assert np.asarray(batch.candidate_mask).shape == (4, 4)
    assert np.asarray(batch.action_features).shape == (4, 4, 63)
    assert np.asarray(batch.public_supply).shape == (4, 30)
    assert np.asarray(batch.selected_index).tolist() == [1, 1, 1, 1]
    np.testing.assert_allclose(
        np.asarray(batch.target_mean)[0],
        [90.5, 91.5, 92.5, 93.5],
    )
    np.testing.assert_allclose(
        np.asarray(batch.target_standard_error)[0],
        [0.15075567] * 4,
        rtol=1e-5,
    )
    assert np.asarray(batch.target_total_samples).shape == (4, 4, 12)
    assert np.asarray(batch.target_centered_samples).shape == (4, 4, 12)
    assert np.asarray(batch.target_component_samples).shape == (4, 4, 12, 11)
    np.testing.assert_allclose(
        np.asarray(batch.target_centered_samples).mean(axis=1),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(batch.target_total_samples),
        np.asarray(batch.target_component_samples).sum(axis=-1),
    )


def test_counterfactual_advantage_dataset_rejects_checksum_drift(tmp_path: Path) -> None:
    root = tmp_path / "counterfactual"
    shard = write_counterfactual_advantage_dataset(root, split="train", game_index=9_996)
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(DatasetError, match="checksum"):
        CounterfactualAdvantageDataset(root)


def test_counterfactual_advantage_dataset_rejects_non_r12_manifest(tmp_path: Path) -> None:
    root = tmp_path / "counterfactual"
    write_counterfactual_advantage_dataset(root, split="train", game_index=9_996)
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["teacher"]["samples_per_candidate"] = 8
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DatasetError, match="frozen R12"):
        CounterfactualAdvantageDataset(root)
