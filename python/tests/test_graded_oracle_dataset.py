from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.dataset import RECORD_SIZE, DatasetError
from cascadia_mlx.graded_oracle_dataset import (
    _ACTION_DTYPE,
    _CANDIDATE_DTYPE,
    _GROUP_HEADER_DTYPE,
    GRADED_FIDELITY_R1200,
    GRADED_FIDELITY_R4800,
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_ACTION_FEATURE_SIZE,
    GRADED_ORACLE_CANDIDATE_RECORD_SIZE,
    GRADED_ORACLE_DATASET_SCHEMA_VERSION,
    GRADED_ORACLE_FEATURE_SCHEMA,
    GRADED_ORACLE_GROUP_HEADER_SIZE,
    GRADED_ORACLE_HEADER_SIZE,
    GRADED_ORACLE_MAX_WILDLIFE_WIPES,
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PRIOR_FEATURES,
    GRADED_ORACLE_PRIOR_SCHEMA,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
    GRADED_ORACLE_SHARD_MAGIC,
    GRADED_ORACLE_TARGET_SCHEMA,
    GRADED_SOURCE_COMPLETE_LEGAL,
    GRADED_SOURCE_R1200,
    GRADED_SOURCE_R4800,
    GradedOracleDataset,
    GradedOracleGroupRef,
    _pack_groups,
    decode_graded_oracle_groups,
    decode_graded_prior_features,
    inspect_graded_oracle_candidate_records,
    inspect_graded_oracle_group,
    inspect_graded_oracle_group_header,
    rotate_graded_oracle_batch,
)


def _position(game_index: int) -> bytes:
    record = bytearray(RECORD_SIZE)
    struct.pack_into("<QBBBB", record, 0, game_index, 12, 0, 4, 80)
    record[12:16] = bytes([3, 3, 3, 3])
    record[16:20] = bytes([2, 1, 0, 0])
    record[20:25] = bytes(5)
    record[72:80] = bytes([0, 0, 1, 255, 0, 1, 255, 0])
    record[808:840] = bytes([255] * 32)
    return bytes(record)


def write_graded_oracle_dataset(root: Path) -> Path:
    root.mkdir()
    raw_seed = 61_003
    group = np.zeros(1, dtype=_GROUP_HEADER_DTYPE)
    group["group_id"] = 1234
    group["raw_seed"] = raw_seed
    group["candidate_count"] = 2
    group["selected_index"] = 1
    group["champion_index"] = 0
    group["completed_turns"] = 12
    group["current_player"] = 0
    group["personal_turn"] = 4
    group["phase"] = 0
    group["public_state_hash"] = np.arange(32, dtype=np.uint8)
    group["position"] = np.frombuffer(_position(raw_seed), dtype=group["position"].dtype)
    group["public_supply"] = np.arange(
        GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
        dtype=np.uint8,
    )

    candidates = np.zeros(2, dtype=_CANDIDATE_DTYPE)
    candidates["action_hash"][0] = 1
    candidates["action_hash"][1] = 2
    candidates["canonical_index"] = [0, 1]
    candidates["screen_rank"] = [2, 1]
    candidates["source_flags"] = [
        GRADED_SOURCE_COMPLETE_LEGAL | GRADED_SOURCE_R1200,
        GRADED_SOURCE_COMPLETE_LEGAL | GRADED_SOURCE_R1200 | GRADED_SOURCE_R4800,
    ]
    candidates["fidelity_mask"] = [
        GRADED_FIDELITY_R1200,
        GRADED_FIDELITY_R1200 | GRADED_FIDELITY_R4800,
    ]
    candidates["model_immediate_score"] = [8.0, 9.0]
    candidates["model_remaining_value"] = [82.0, 83.0]
    candidates["screen_value"] = [90.0, 92.0]
    candidates["uniform_market_survival_proxy"] = [0.2, 0.4]
    candidates["visible_wildlife_count"] = [1, 2]
    candidates["public_bag_wildlife_count"] = [18, 17]
    for index in range(2):
        action = candidates["action"][index]
        action["draft_kind"] = index
        action["tile_slot"] = index
        action["wildlife_slot"] = index
        action["tile_id"] = 20 + index
        action["tile_terrain_a"] = index
        action["tile_terrain_b"] = 255
        action["tile_wildlife_mask"] = 1 << index
        action["tile_keystone"] = 1
        action["drafted_wildlife"] = index
        action["tile_q"] = -1 + index
        action["tile_r"] = 2
        action["rotation"] = index
        action["wildlife_present"] = 1
        action["wildlife_q"] = index
        action["wildlife_r"] = 1
        action["wipe_count"] = 2
        action["wipe_masks"][:2] = [0b0011, 0b1100]
        action["staged_active_nature_tokens"] = 3
        action["staged_market_entities"][:] = 255
        action["staged_market_entities"][0] = [0, 255, 1, 0, 1, 0, 0, 0]
        action["staged_public_supply"] = np.arange(
            GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
            dtype=np.uint8,
        )
        action["immediate_score"] = 70 + index
        action["immediate_deltas"] = np.arange(11, dtype=np.int16)
    candidates["r1200"]["mean"] = [93.0, 94.0]
    candidates["r1200"]["stddev"] = [2.0, 1.5]
    candidates["r1200"]["samples"] = [1200, 1200]
    candidates["r4800"]["mean"][1] = 95.0
    candidates["r4800"]["stddev"][1] = 1.0
    candidates["r4800"]["samples"][1] = 4800

    payload = group.tobytes() + candidates.tobytes()
    header = struct.pack(
        "<8sHHHHIIIBBBBQ32s32s8s",
        GRADED_ORACLE_SHARD_MAGIC,
        GRADED_ORACLE_DATASET_SCHEMA_VERSION,
        GRADED_ORACLE_HEADER_SIZE,
        GRADED_ORACLE_GROUP_HEADER_SIZE,
        GRADED_ORACLE_CANDIDATE_RECORD_SIZE,
        2,
        1,
        1,
        1,
        4,
        0,
        0,
        raw_seed,
        blake3.blake3(GRADED_ORACLE_FEATURE_SCHEMA.encode()).digest(),
        blake3.blake3(GRADED_ORACLE_TARGET_SCHEMA.encode()).digest(),
        bytes(8),
    )
    shard = root / f"seed-{raw_seed}.gov"
    shard.write_bytes(header + payload)
    manifest = {
        "schema_version": GRADED_ORACLE_DATASET_SCHEMA_VERSION,
        "dataset_id": "graded-oracle-validation-test",
        "feature_schema": GRADED_ORACLE_FEATURE_SCHEMA,
        "position_feature_schema": "compact-entity-v2",
        "target_schema": GRADED_ORACLE_TARGET_SCHEMA,
        "group_header_size": GRADED_ORACLE_GROUP_HEADER_SIZE,
        "candidate_record_size": GRADED_ORACLE_CANDIDATE_RECORD_SIZE,
        "action_feature_size": GRADED_ORACLE_ACTION_FEATURE_SIZE,
        "public_supply_size": GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
        "maximum_wildlife_wipes": GRADED_ORACLE_MAX_WILDLIFE_WIPES,
        "game": {},
        "split": "validation",
        "seeds": [raw_seed],
        "requested_games": 1,
        "completed_games": 1,
        "total_groups": 1,
        "total_records": 2,
        "teacher": {},
        "audit_inputs": [{}],
        "created_unix_seconds": 1,
        "updated_unix_seconds": 1,
        "provenance": {},
        "shards": [
            {
                "file": shard.name,
                "first_game_index": raw_seed,
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


def test_graded_oracle_dataset_decodes_lossless_complete_groups(tmp_path: Path) -> None:
    root = tmp_path / "graded"
    write_graded_oracle_dataset(root)

    dataset = GradedOracleDataset(root)
    batch = next(dataset.batches(2, maximum_actions_per_batch=8))

    assert dataset.group_count == 1
    assert dataset.candidate_count == 2
    assert np.asarray(batch.action_features).shape == (1, 2, GRADED_ORACLE_ACTION_DIM)
    assert np.asarray(batch.prior_features).shape == (1, 2, GRADED_ORACLE_PRIOR_DIM)
    assert np.asarray(batch.staged_market_entities).shape == (1, 2, 4, 31)
    assert np.asarray(batch.staged_public_supply).shape == (1, 2, 30)
    assert np.asarray(batch.public_supply).shape == (1, 30)
    assert np.asarray(batch.selected).tolist() == [[False, True]]
    assert np.asarray(batch.champion).tolist() == [[True, False]]
    assert np.asarray(batch.r1200_samples).tolist() == [[1200.0, 1200.0]]
    assert np.asarray(batch.r4800_mask).tolist() == [[False, True]]
    assert np.asarray(batch.active_nature_tokens).tolist() == [2]
    assert np.asarray(batch.draft_kind).tolist() == [[0, 1]]
    assert np.asarray(batch.wipe_count).tolist() == [[2, 2]]
    assert batch.action_hash[0, 0, 0] == 1
    assert batch.action_hash[0, 1, 0] == 2


def test_graded_oracle_raw_candidate_view_is_exact_and_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "graded"
    write_graded_oracle_dataset(root)
    dataset = GradedOracleDataset(root)
    shard = dataset.shards[0]

    candidates = inspect_graded_oracle_candidate_records(
        shard.bytes(),
        shard.groups[0],
    )

    assert candidates.shape == (2,)
    assert candidates["screen_rank"].tolist() == [2, 1]
    assert candidates["action"]["tile_id"].tolist() == [20, 21]
    assert candidates.flags.writeable is False
    with pytest.raises(ValueError):
        candidates["screen_rank"][0] = 99


def test_graded_oracle_subset_decode_remaps_selected_and_champion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "graded"
    write_graded_oracle_dataset(root)
    dataset = GradedOracleDataset(root)
    shard = dataset.shards[0]
    batch = decode_graded_oracle_groups(
        shard.bytes(),
        (shard.groups[0],),
        candidate_indices=([0, 1],),
    )

    np.testing.assert_array_equal(np.asarray(batch.selected_index), [1])
    np.testing.assert_array_equal(np.asarray(batch.champion_index), [0])
    np.testing.assert_array_equal(batch.action_hash[0, :, 0], [1, 2])

    with pytest.raises(DatasetError, match="omitted"):
        decode_graded_oracle_groups(
            shard.bytes(),
            (shard.groups[0],),
            candidate_indices=([1],),
        )


def test_graded_oracle_subset_can_explicitly_omit_evaluation_only_actions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "graded"
    write_graded_oracle_dataset(root)
    dataset = GradedOracleDataset(root)
    shard = dataset.shards[0]

    selected_only = decode_graded_oracle_groups(
        shard.bytes(),
        (shard.groups[0],),
        candidate_indices=([1],),
        require_champion_action=False,
    )
    np.testing.assert_array_equal(np.asarray(selected_only.selected_index), [0])
    np.testing.assert_array_equal(np.asarray(selected_only.champion_index), [-1])
    np.testing.assert_array_equal(np.asarray(selected_only.selected), [[True]])
    np.testing.assert_array_equal(np.asarray(selected_only.champion), [[False]])

    champion_only = decode_graded_oracle_groups(
        shard.bytes(),
        (shard.groups[0],),
        candidate_indices=([0],),
        require_selected_action=False,
    )
    np.testing.assert_array_equal(np.asarray(champion_only.selected_index), [-1])
    np.testing.assert_array_equal(np.asarray(champion_only.champion_index), [0])
    np.testing.assert_array_equal(np.asarray(champion_only.selected), [[False]])
    np.testing.assert_array_equal(np.asarray(champion_only.champion), [[True]])


def test_group_header_inspection_copies_metadata_without_action_hash_views(
    tmp_path: Path,
) -> None:
    root = tmp_path / "graded"
    write_graded_oracle_dataset(root)
    dataset = GradedOracleDataset(root)
    shard = dataset.shards[0]
    raw = shard.bytes()
    ref = shard.groups[0]

    header = inspect_graded_oracle_group_header(raw, ref)
    identity = inspect_graded_oracle_group(raw, ref)

    assert header.group_id == identity.group_id == 1234
    assert header.selected_draft_kind == identity.selected_draft_kind == 1
    assert header.public_state_hash.base is None
    assert not hasattr(header, "action_hashes")
    assert identity.action_hashes.shape == (2, 32)


def test_graded_oracle_rotation_is_group_consistent_and_invertible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "graded"
    write_graded_oracle_dataset(root)
    batch = next(GradedOracleDataset(root).batches(1))

    rotated = rotate_graded_oracle_batch(batch, 1)
    restored = rotate_graded_oracle_batch(rotated, 5)

    np.testing.assert_allclose(
        np.asarray(restored.board_entities),
        np.asarray(batch.board_entities),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(restored.action_features),
        np.asarray(batch.action_features),
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(rotated.candidate_mask),
        np.asarray(batch.candidate_mask),
    )
    np.testing.assert_array_equal(
        np.asarray(rotated.staged_public_supply),
        np.asarray(batch.staged_public_supply),
    )


def test_graded_oracle_dataset_rejects_checksum_drift(tmp_path: Path) -> None:
    root = tmp_path / "graded"
    shard = write_graded_oracle_dataset(root)
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(DatasetError, match="checksum"):
        GradedOracleDataset(root)


def test_graded_oracle_fixed_width_layout_matches_rust_contract() -> None:
    assert _ACTION_DTYPE.itemsize == GRADED_ORACLE_ACTION_FEATURE_SIZE
    assert _CANDIDATE_DTYPE.itemsize == GRADED_ORACLE_CANDIDATE_RECORD_SIZE
    assert _GROUP_HEADER_DTYPE.itemsize == GRADED_ORACLE_GROUP_HEADER_SIZE


def test_graded_oracle_prior_is_observable_and_provenance_invariant() -> None:
    candidate = np.zeros(1, dtype=_CANDIDATE_DTYPE)
    candidate["screen_rank"] = 17
    candidate["model_immediate_score"] = 8.0
    candidate["model_remaining_value"] = 82.0
    candidate["screen_value"] = 90.0
    candidate["uniform_market_survival_proxy"] = 0.25
    candidate["visible_wildlife_count"] = 3
    candidate["public_bag_wildlife_count"] = 18

    baseline = decode_graded_prior_features(candidate)
    mutated = candidate.copy()
    mutated["source_flags"] = np.iinfo(np.uint16).max
    mutated["fidelity_mask"] = np.iinfo(np.uint16).max

    assert GRADED_ORACLE_PRIOR_SCHEMA == "observable-screen-priors-v1"
    assert GRADED_ORACLE_PRIOR_FEATURES == (
        "model_immediate_score",
        "model_remaining_value",
        "screen_value",
        "screen_rank_scaled",
        "screen_inverse_rank",
        "uniform_market_survival_proxy",
        "visible_wildlife_count",
        "public_bag_wildlife_count",
    )
    assert GRADED_ORACLE_PRIOR_DIM == 8
    np.testing.assert_array_equal(
        decode_graded_prior_features(mutated),
        baseline,
    )


def test_oversized_complete_group_runs_alone_without_truncation() -> None:
    small = GradedOracleGroupRef(0, 0, 4_000)
    oversized = GradedOracleGroupRef(0, 0, 10_854)
    batches = list(
        _pack_groups(
            (small, oversized, small),
            group_batch_size=64,
            maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
            maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        )
    )
    assert [[ref.candidate_count for ref in batch] for batch in batches] == [
        [4_000],
        [10_854],
        [4_000],
    ]


def test_group_ceiling_rejects_unbounded_corrupt_width() -> None:
    too_large = GradedOracleGroupRef(
        0,
        0,
        GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS + 1,
    )
    with pytest.raises(DatasetError, match="maximum_group_actions"):
        list(
            _pack_groups(
                (too_large,),
                group_batch_size=64,
                maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
                maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
            )
        )
