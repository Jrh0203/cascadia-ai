from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.dataset import FEATURE_SCHEMA, RECORD_SIZE, DatasetError
from cascadia_mlx.imitation_dataset import (
    IMITATION_CANDIDATE_RECORD_SIZE,
    IMITATION_DATASET_SCHEMA_VERSION,
    IMITATION_FEATURE_SCHEMA,
    IMITATION_GROUP_HEADER_SIZE,
    IMITATION_HEADER_SIZE,
    IMITATION_SHARD_MAGIC,
    IMITATION_TARGET_SCHEMA,
    PROPOSAL_ACTION_DIM,
    PROPOSAL_ACTION_FEATURE_SIZE,
    ImitationDataset,
    rotate_imitation_batch,
)
from cascadia_mlx.imitation_parent_hidden_dataset import (
    IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION,
    IMITATION_PARENT_HIDDEN_DIM,
    IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA,
    IMITATION_PARENT_HIDDEN_HEADER_SIZE,
    IMITATION_PARENT_HIDDEN_RECORD_SIZE,
    IMITATION_PARENT_HIDDEN_SHARD_MAGIC,
    IMITATION_PARENT_HIDDEN_TARGET_SCHEMA,
    ImitationParentHiddenEvidenceDataset,
)
from cascadia_mlx.imitation_parent_prior_dataset import (
    IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION,
    IMITATION_PARENT_PRIOR_FEATURE_SCHEMA,
    IMITATION_PARENT_PRIOR_HEADER_SIZE,
    IMITATION_PARENT_PRIOR_RECORD_SIZE,
    IMITATION_PARENT_PRIOR_SHARD_MAGIC,
    IMITATION_PARENT_PRIOR_TARGET_SCHEMA,
    ImitationParentEvidenceDataset,
)
from cascadia_mlx.imitation_targets_dataset import (
    IMITATION_TARGETS_DATASET_SCHEMA_VERSION,
    IMITATION_TARGETS_FEATURE_SCHEMA,
    IMITATION_TARGETS_HEADER_SIZE,
    IMITATION_TARGETS_RECORD_SIZE,
    IMITATION_TARGETS_SHARD_MAGIC,
    IMITATION_TARGETS_TARGET_SCHEMA,
    SOURCE_IMMEDIATE_TOP,
    SOURCE_TEACHER_FRONTIER,
    ImitationEvidenceDataset,
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
    action = bytearray(PROPOSAL_ACTION_FEATURE_SIZE)
    struct.pack_into(
        "<BBBBBBBBbbBBbbBBBBHH",
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
        score,
    )
    return bytes(action)


def _candidate(index: int) -> bytes:
    rank = index + 1
    score = 41 - index
    record = bytearray(IMITATION_CANDIDATE_RECORD_SIZE)
    struct.pack_into("<HH", record, 0, rank, score)
    record[4:36] = bytes([index] * 32)
    record[36:] = _action(rank, score)
    return bytes(record)


def _write_dataset(root: Path, *, shard_count: int = 1) -> None:
    root.mkdir()
    shards = []
    for shard_index in range(shard_count):
        game_index = 7 + shard_index
        group_header = bytearray(IMITATION_GROUP_HEADER_SIZE)
        struct.pack_into("<QHH", group_header, 0, 11 + shard_index, 2, 0)
        group_header[16:] = _position(game_index, 5)
        records = bytes(group_header) + _candidate(0) + _candidate(1)
        header = struct.pack(
            "<8sHHHHIIIBBBBQ32s32s8s",
            IMITATION_SHARD_MAGIC,
            IMITATION_DATASET_SCHEMA_VERSION,
            IMITATION_HEADER_SIZE,
            IMITATION_GROUP_HEADER_SIZE,
            IMITATION_CANDIDATE_RECORD_SIZE,
            2,
            1,
            1,
            0,
            4,
            0,
            0,
            game_index,
            blake3.blake3(IMITATION_FEATURE_SCHEMA.encode()).digest(),
            blake3.blake3(IMITATION_TARGET_SCHEMA.encode()).digest(),
            bytes(8),
        )
        shard = root / f"shard-{shard_index:05}.cim"
        shard.write_bytes(header + records)
        shards.append(
            {
                "file": shard.name,
                "first_game_index": game_index,
                "game_count": 1,
                "group_count": 1,
                "record_count": 2,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": IMITATION_DATASET_SCHEMA_VERSION,
        "dataset_id": "imitation-test",
        "feature_schema": IMITATION_FEATURE_SCHEMA,
        "position_feature_schema": FEATURE_SCHEMA,
        "target_schema": IMITATION_TARGET_SCHEMA,
        "group_header_size": IMITATION_GROUP_HEADER_SIZE,
        "candidate_record_size": IMITATION_CANDIDATE_RECORD_SIZE,
        "action_feature_size": PROPOSAL_ACTION_FEATURE_SIZE,
        "split": "train",
        "teacher": {"weights_blake3": "teacher"},
        "candidates": {
            "group_limit": 2,
            "deterministic_sampler": ("teacher-frontier-pattern-immediate-blake3-action-json-v1"),
        },
        "first_game_index": 7,
        "requested_games": shard_count,
        "completed_games": shard_count,
        "total_groups": shard_count,
        "total_records": shard_count * 2,
        "shards": shards,
    }
    (root / "dataset.json").write_text(json.dumps(manifest))


def _write_targets(root: Path, source_root: Path, *, corrupt_hash: bool = False) -> None:
    root.mkdir()
    source_manifest = json.loads((source_root / "dataset.json").read_text())
    shards = []
    for shard_index in range(source_manifest["requested_games"]):
        game_index = 7 + shard_index
        records = bytearray()
        for index, (mean, stddev, samples) in enumerate(((90.0, 3.0, 20), (88.0, 4.0, 10))):
            action_hash = bytes([index] * 32)
            if corrupt_hash and shard_index == 0 and index == 1:
                action_hash = bytes([9] * 32)
            records.extend(
                struct.pack(
                    "<QHH32sffHBB",
                    11 + shard_index,
                    index,
                    2,
                    action_hash,
                    mean,
                    stddev,
                    samples,
                    SOURCE_TEACHER_FRONTIER | SOURCE_IMMEDIATE_TOP,
                    int(index == 0),
                )
            )
        header = struct.pack(
            "<8sHHHHIIIBBBBQ32s32s8s",
            IMITATION_TARGETS_SHARD_MAGIC,
            IMITATION_TARGETS_DATASET_SCHEMA_VERSION,
            IMITATION_TARGETS_HEADER_SIZE,
            IMITATION_TARGETS_RECORD_SIZE,
            0,
            2,
            1,
            1,
            0,
            4,
            0,
            0,
            game_index,
            blake3.blake3(IMITATION_TARGETS_FEATURE_SCHEMA.encode()).digest(),
            blake3.blake3(IMITATION_TARGETS_TARGET_SCHEMA.encode()).digest(),
            bytes(8),
        )
        shard = root / f"shard-{shard_index:05}.imv"
        shard.write_bytes(header + records)
        shards.append(
            {
                "file": shard.name,
                "first_game_index": game_index,
                "game_count": 1,
                "group_count": 1,
                "record_count": 2,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        )
    shard_count = source_manifest["requested_games"]
    manifest = {
        "schema_version": IMITATION_TARGETS_DATASET_SCHEMA_VERSION,
        "dataset_id": "imitation-target-test",
        "feature_schema": IMITATION_TARGETS_FEATURE_SCHEMA,
        "target_schema": IMITATION_TARGETS_TARGET_SCHEMA,
        "record_size": IMITATION_TARGETS_RECORD_SIZE,
        "split": "train",
        "teacher": source_manifest["teacher"],
        "source": {
            "path": str(source_root.resolve()),
            "dataset_id": source_manifest["dataset_id"],
            "feature_schema": source_manifest["feature_schema"],
            "target_schema": source_manifest["target_schema"],
            "first_game_index": 7,
            "requested_games": shard_count,
        },
        "first_game_index": 7,
        "requested_games": shard_count,
        "completed_games": shard_count,
        "total_groups": shard_count,
        "total_records": shard_count * 2,
        "teacher_estimates": shard_count * 2,
        "aligned_teacher_estimates": shard_count * 2,
        "shards": shards,
    }
    (root / "dataset.json").write_text(json.dumps(manifest))


def _write_parent_priors(root: Path, targets_root: Path, model_root: Path) -> None:
    root.mkdir()
    model_root.mkdir()
    model_manifest = {"schema_version": 1, "architecture": "test-parent"}
    (model_root / "model.json").write_text(json.dumps(model_manifest))
    (model_root / "model.safetensors").write_bytes(b"test tensors")
    targets = json.loads((targets_root / "dataset.json").read_text())
    shards = []
    for shard_index in range(targets["requested_games"]):
        game_index = 7 + shard_index
        records = bytearray()
        for index, (immediate, remaining) in enumerate(((40.0, 55.0), (39.0, 52.0))):
            records.extend(
                struct.pack(
                    "<QHH32sff4x",
                    11 + shard_index,
                    index,
                    2,
                    bytes([index] * 32),
                    immediate,
                    remaining,
                )
            )
        header = struct.pack(
            "<8sHHHHIIIBBBBQ32s32s8s",
            IMITATION_PARENT_PRIOR_SHARD_MAGIC,
            IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION,
            IMITATION_PARENT_PRIOR_HEADER_SIZE,
            IMITATION_PARENT_PRIOR_RECORD_SIZE,
            0,
            2,
            1,
            1,
            0,
            4,
            0,
            0,
            game_index,
            blake3.blake3(IMITATION_PARENT_PRIOR_FEATURE_SCHEMA.encode()).digest(),
            blake3.blake3(IMITATION_PARENT_PRIOR_TARGET_SCHEMA.encode()).digest(),
            bytes(8),
        )
        shard = root / f"shard-{shard_index:05}.imp"
        shard.write_bytes(header + records)
        shards.append(
            {
                "file": shard.name,
                "first_game_index": game_index,
                "game_count": 1,
                "group_count": 1,
                "record_count": 2,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION,
        "dataset_id": "parent-prior-test",
        "feature_schema": IMITATION_PARENT_PRIOR_FEATURE_SCHEMA,
        "target_schema": IMITATION_PARENT_PRIOR_TARGET_SCHEMA,
        "record_size": IMITATION_PARENT_PRIOR_RECORD_SIZE,
        "split": "train",
        "teacher": targets["teacher"],
        "source": {
            "path": str(targets_root.resolve()),
            "dataset_id": targets["dataset_id"],
            "feature_schema": targets["feature_schema"],
            "target_schema": targets["target_schema"],
            "action_dataset_id": targets["source"]["dataset_id"],
            "first_game_index": 7,
            "requested_games": targets["requested_games"],
        },
        "model": {
            "path": str(model_root.resolve()),
            "architecture": "test-parent",
            "manifest_blake3": blake3.blake3((model_root / "model.json").read_bytes()).hexdigest(),
            "tensors_blake3": blake3.blake3(
                (model_root / "model.safetensors").read_bytes()
            ).hexdigest(),
        },
        "first_game_index": 7,
        "requested_games": targets["requested_games"],
        "completed_games": targets["requested_games"],
        "total_groups": targets["total_groups"],
        "total_records": targets["total_records"],
        "shards": shards,
    }
    (root / "dataset.json").write_text(json.dumps(manifest))


def _write_parent_hidden(root: Path, targets_root: Path, model_root: Path) -> None:
    root.mkdir()
    if not model_root.exists():
        model_root.mkdir()
        model_manifest = {"schema_version": 1, "architecture": "test-parent"}
        (model_root / "model.json").write_text(json.dumps(model_manifest))
        (model_root / "model.safetensors").write_bytes(b"test tensors")
    targets = json.loads((targets_root / "dataset.json").read_text())
    shards = []
    for shard_index in range(targets["requested_games"]):
        game_index = 7 + shard_index
        records = bytearray()
        for index, (immediate, remaining) in enumerate(((40.0, 55.0), (39.0, 52.0))):
            hidden = np.arange(IMITATION_PARENT_HIDDEN_DIM, dtype=np.float32) + index * 100
            records.extend(
                struct.pack(
                    "<QHH32sff64f4x",
                    11 + shard_index,
                    index,
                    2,
                    bytes([index] * 32),
                    immediate,
                    remaining,
                    *hidden.tolist(),
                )
            )
        header = struct.pack(
            "<8sHHHHIIIBBBBQ32s32s8s",
            IMITATION_PARENT_HIDDEN_SHARD_MAGIC,
            IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION,
            IMITATION_PARENT_HIDDEN_HEADER_SIZE,
            IMITATION_PARENT_HIDDEN_RECORD_SIZE,
            0,
            2,
            1,
            1,
            0,
            4,
            0,
            0,
            game_index,
            blake3.blake3(IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA.encode()).digest(),
            blake3.blake3(IMITATION_PARENT_HIDDEN_TARGET_SCHEMA.encode()).digest(),
            bytes(8),
        )
        shard = root / f"shard-{shard_index:05}.imh"
        shard.write_bytes(header + records)
        shards.append(
            {
                "file": shard.name,
                "first_game_index": game_index,
                "game_count": 1,
                "group_count": 1,
                "record_count": 2,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION,
        "dataset_id": "parent-hidden-test",
        "feature_schema": IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA,
        "target_schema": IMITATION_PARENT_HIDDEN_TARGET_SCHEMA,
        "record_size": IMITATION_PARENT_HIDDEN_RECORD_SIZE,
        "split": "train",
        "teacher": targets["teacher"],
        "source": {
            "path": str(targets_root.resolve()),
            "dataset_id": targets["dataset_id"],
            "feature_schema": targets["feature_schema"],
            "target_schema": targets["target_schema"],
            "action_dataset_id": targets["source"]["dataset_id"],
            "first_game_index": 7,
            "requested_games": targets["requested_games"],
        },
        "model": {
            "path": str(model_root.resolve()),
            "architecture": "test-parent",
            "manifest_blake3": blake3.blake3((model_root / "model.json").read_bytes()).hexdigest(),
            "tensors_blake3": blake3.blake3(
                (model_root / "model.safetensors").read_bytes()
            ).hexdigest(),
        },
        "first_game_index": 7,
        "requested_games": targets["requested_games"],
        "completed_games": targets["requested_games"],
        "total_groups": targets["total_groups"],
        "total_records": targets["total_records"],
        "shards": shards,
    }
    (root / "dataset.json").write_text(json.dumps(manifest))


def test_imitation_dataset_decodes_one_shared_state_and_explicit_actions(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path / "imitation")
    dataset = ImitationDataset(tmp_path / "imitation")
    batch = next(dataset.batches(2))

    actions = np.asarray(batch.action_features)
    assert np.asarray(batch.board_entities).shape == (1, 4, 23, 31)
    assert actions.shape == (1, 2, PROPOSAL_ACTION_DIM)
    assert np.asarray(batch.candidate_mask).tolist() == [[True, True]]
    assert np.asarray(batch.teacher_mean).tolist() == [[1.0, 0.0]]
    assert np.asarray(batch.immediate_rank).tolist() == [[1.0, 2.0]]
    assert np.asarray(batch.immediate_score).tolist() == [[41.0, 40.0]]
    np.testing.assert_allclose(actions[0, 0, -2:], [1 / 4096, 0.41])
    np.testing.assert_allclose(actions[0, 1, -2:], [2 / 4096, 0.40])


def test_imitation_rotation_updates_coordinates_and_orientations(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path / "imitation")
    batch = next(ImitationDataset(tmp_path / "imitation").batches(2))
    rotated = rotate_imitation_batch(batch, 1)

    boards = np.asarray(rotated.board_entities)
    actions = np.asarray(rotated.action_features)
    np.testing.assert_allclose(boards[0, 0, 0, :2], [-1 / 24, -1 / 24])
    assert int(np.argmax(boards[0, 0, 0, 13:19])) == 1
    np.testing.assert_allclose(actions[0, 0, 32:34], [-1 / 24, -1 / 24])
    assert int(np.argmax(actions[0, 0, 34:40])) == 1
    np.testing.assert_allclose(actions[0, :, 43:], np.asarray(batch.action_features)[0, :, 43:])

    round_trip = batch
    for _ in range(6):
        round_trip = rotate_imitation_batch(round_trip, 1)
    np.testing.assert_allclose(
        np.asarray(round_trip.board_entities),
        np.asarray(batch.board_entities),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(round_trip.action_features),
        np.asarray(batch.action_features),
        atol=1e-6,
    )


def test_imitation_evidence_attaches_full_frontier_targets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    targets = tmp_path / "targets"
    _write_dataset(source)
    _write_targets(targets, source)

    dataset = ImitationEvidenceDataset(targets)
    batch = next(dataset.batches(2))

    assert np.asarray(batch.teacher_scored).tolist() == [[True, True]]
    assert np.asarray(batch.selected).tolist() == [[True, False]]
    assert np.asarray(batch.teacher_samples).tolist() == [[20.0, 10.0]]
    assert np.asarray(batch.teacher_mean).tolist() == [[90.0, 88.0]]
    assert np.asarray(batch.teacher_stddev).tolist() == [[3.0, 4.0]]


def test_imitation_evidence_rejects_action_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    targets = tmp_path / "targets"
    _write_dataset(source)
    _write_targets(targets, source, corrupt_hash=True)

    with np.testing.assert_raises(DatasetError):
        ImitationEvidenceDataset(targets)


def test_parent_prior_evidence_attaches_exact_scores_and_ranks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    targets = tmp_path / "targets"
    priors = tmp_path / "priors"
    model = tmp_path / "model"
    _write_dataset(source)
    _write_targets(targets, source)
    _write_parent_priors(priors, targets, model)

    batch = next(ImitationParentEvidenceDataset(priors).batches(2))

    assert np.asarray(batch.parent_immediate).tolist() == [[40.0, 39.0]]
    assert np.asarray(batch.parent_remaining).tolist() == [[55.0, 52.0]]
    assert np.asarray(batch.parent_total).tolist() == [[95.0, 91.0]]
    assert np.asarray(batch.parent_rank).tolist() == [[1.0, 2.0]]


def test_parent_hidden_evidence_attaches_exact_scores_ranks_and_hidden(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    targets = tmp_path / "targets"
    hidden = tmp_path / "hidden"
    model = tmp_path / "model"
    _write_dataset(source)
    _write_targets(targets, source)
    _write_parent_hidden(hidden, targets, model)

    batch = next(ImitationParentHiddenEvidenceDataset(hidden).batches(2))

    assert np.asarray(batch.parent_immediate).tolist() == [[40.0, 39.0]]
    assert np.asarray(batch.parent_remaining).tolist() == [[55.0, 52.0]]
    assert np.asarray(batch.parent_total).tolist() == [[95.0, 91.0]]
    assert np.asarray(batch.parent_rank).tolist() == [[1.0, 2.0]]
    hidden_values = np.asarray(batch.parent_hidden)
    assert hidden_values.shape == (1, 2, IMITATION_PARENT_HIDDEN_DIM)
    np.testing.assert_array_equal(hidden_values[0, 0], np.arange(64, dtype=np.float32))
    np.testing.assert_array_equal(hidden_values[0, 1], np.arange(64, dtype=np.float32) + 100)


def test_imitation_evidence_streams_many_shards_without_open_memmaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    targets = tmp_path / "targets"
    _write_dataset(source, shard_count=140)
    _write_targets(targets, source)

    def reject_memmap(*_args, **_kwargs):
        raise AssertionError("shard readers must not retain open memmaps")

    monkeypatch.setattr(np, "memmap", reject_memmap)
    dataset = ImitationEvidenceDataset(targets)
    assert dataset.group_count == 140
    assert sum(len(np.asarray(batch.group_id)) for batch in dataset.batches(16)) == 140
