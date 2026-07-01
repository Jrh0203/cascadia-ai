from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.opponent_intent_dataset import (
    OPPONENT_INTENT_RECORD_DTYPE,
    RECORD_SIZE,
    SHARD_HEADER_SIZE,
    SHARD_MAGIC,
    OpponentIntentDataset,
    OpponentIntentDatasetError,
    decode_opponent_intent_inputs,
    decode_opponent_intent_records,
    model_input_arrays,
)


def write_opponent_intent_dataset(
    root: Path,
    *,
    split: str,
    game_index: int,
    records: int = 8,
    cohort_id: str = "test-cohort",
) -> Path:
    root.mkdir(parents=True)
    values = np.zeros(records, dtype=OPPONENT_INTENT_RECORD_DTYPE)
    values["game_index"] = game_index
    values["focal_turn"] = np.arange(records, dtype=np.uint8)
    values["focal_seat"] = np.arange(records, dtype=np.uint8) % 4
    values["seat_policy_codes"] = np.asarray([1, 2, 3, 1], dtype=np.uint8)
    position = values["position"]
    position["game_index"] = game_index
    position["turn"] = values["focal_turn"] + 1
    position["active_seat"] = values["focal_seat"]
    position["player_count"] = 4
    position["total_turns"] = 80
    position["board_counts"] = 1
    position["nature_tokens"] = np.asarray([1, 2, 3, 4], dtype=np.uint8)
    position["scoring_cards"] = 0
    position["wildlife_counts"][:, :, 0] = np.asarray(
        [1, 2, 3, 4],
        dtype=np.uint8,
    )
    position["habitat_sizes"][:, :, 0] = np.asarray(
        [5, 6, 7, 8],
        dtype=np.uint8,
    )
    for seat in range(4):
        position["board_entities"][:, seat, 0, 0] = seat
        position["board_entities"][:, seat, 0, 2] = seat
        position["board_entities"][:, seat, 0, 3] = 255
        position["board_entities"][:, seat, 0, 4] = 0
        position["board_entities"][:, seat, 0, 5] = 1
        position["board_entities"][:, seat, 0, 6] = 255
        position["board_entities"][:, seat, 0, 7] = 1
    for slot in range(4):
        position["market_entities"][:, slot, 0] = slot
        position["market_entities"][:, slot, 1] = 255
        position["market_entities"][:, slot, 2] = 1 << slot
        position["market_entities"][:, slot, 3] = slot
        position["market_entities"][:, slot, 4] = slot == 0
    position["targets"] = 999

    values["history_count"] = 1
    history = values["history"]
    history[:, 0]["valid"] = 1
    history[:, 0]["age"] = 0
    history[:, 0]["relative_seat"] = 0
    _fill_actions(history[:, 0]["action"])

    targets = values["opponent_targets"]
    targets["relative_seat"] = np.asarray([1, 2, 3], dtype=np.uint8)
    targets["policy_code"] = np.asarray([2, 3, 1], dtype=np.uint8)
    targets["selected_tile_id"] = np.asarray([20, 21, 22], dtype=np.uint8)
    for opponent in range(3):
        _fill_actions(targets[:, opponent]["action"], slot=opponent)

    survival = values["survival_targets"]
    survival["initial_tile_id"] = np.asarray(
        [20, 21, 22, 23],
        dtype=np.uint8,
    )
    survival["initial_wildlife"] = np.asarray(
        [0, 1, 2, 3],
        dtype=np.uint8,
    )
    survival["disposition"] = np.asarray([1, 2, 3, 4], dtype=np.uint8)
    survival["pair_survives"] = np.asarray([0, 0, 0, 1], dtype=np.uint8)
    survival["final_slot"] = np.asarray(
        [255, 255, 255, 3],
        dtype=np.uint8,
    )
    values["final_scores"] = np.asarray([80, 81, 82, 83], dtype=np.uint16)

    split_code = {"train": 0, "validation": 1, "test": 2, "final": 3}[split]
    header = bytearray(SHARD_HEADER_SIZE)
    struct.pack_into(
        "<8sHIBQIQ",
        header,
        0,
        SHARD_MAGIC,
        1,
        RECORD_SIZE,
        split_code,
        game_index,
        1,
        records,
    )
    shard = root / "shard-00000.o1i"
    shard.write_bytes(bytes(header) + values.tobytes())
    checksum = blake3.blake3(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_id": f"test-{split}-{game_index}",
        "feature_schema": "compact-public-state-recent-actions-no-policy-id-v1",
        "target_schema": "three-opponent-next-action-four-tile-survival-v1",
        "record_size": RECORD_SIZE,
        "split": split,
        "first_game_index": game_index,
        "requested_games": 1,
        "completed_games": 1,
        "total_records": records,
        "windows_per_game": 76,
        "history_length": 12,
        "cohort": {
            "cohort_id": cohort_id,
            "policy_pool": ["greedy"],
            "required_policy": None,
        },
        "policy_identity_observable": False,
        "game_index_observable": False,
        "strategy_switch_targets_available": False,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": game_index,
                "game_count": 1,
                "group_count": records,
                "record_count": records,
                "byte_count": shard.stat().st_size,
                "blake3": checksum,
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return root


def _fill_actions(actions: np.ndarray, *, slot: int = 0) -> None:
    actions["draft_kind"] = slot % 2
    actions["tile_slot"] = slot
    actions["wildlife_slot"] = slot
    actions["tile_terrain_a"] = slot
    actions["tile_terrain_b"] = 255
    actions["tile_wildlife_mask"] = 1 << slot
    actions["tile_keystone"] = slot == 0
    actions["drafted_wildlife"] = slot
    actions["rotation"] = slot
    actions["wildlife_present"] = 1


def test_exact_dtype_and_policy_blind_decode(tmp_path: Path) -> None:
    root = write_opponent_intent_dataset(
        tmp_path / "validation",
        split="validation",
        game_index=100,
    )
    dataset = OpponentIntentDataset(root)
    records = dataset.shards[0].records()
    batch = decode_opponent_intent_records(records[:4])

    assert OPPONENT_INTENT_RECORD_DTYPE.itemsize == 1_312
    assert batch.board_entities.shape == (4, 4, 23, 31)
    assert batch.history_features.shape == (4, 12, 55)
    assert batch.disposition_targets.shape == (4, 4)
    assert np.array_equal(
        np.asarray(batch.disposition_targets[0]),
        np.asarray([0, 1, 2, 3]),
    )

    changed = records[:4].copy()
    changed["game_index"] += 10_000
    changed["seat_policy_codes"] = 5
    changed["position"]["game_index"] += 20_000
    changed["position"]["targets"] = 1
    changed["opponent_targets"]["policy_code"] = 5
    changed["opponent_targets"]["selected_tile_id"] = 99
    changed["opponent_targets"]["action"]["tile_slot"] = 3
    changed["survival_targets"]["initial_tile_id"] = 99
    changed["survival_targets"]["disposition"] = 4
    changed["final_scores"] = 200

    original_inputs = model_input_arrays(records[:4])
    changed_inputs = model_input_arrays(changed)
    assert all(
        np.array_equal(original, mutated)
        for original, mutated in zip(
            original_inputs,
            changed_inputs,
            strict=True,
        )
    )


def test_focal_seat_is_rotated_to_relative_zero(tmp_path: Path) -> None:
    root = write_opponent_intent_dataset(
        tmp_path / "train",
        split="train",
        game_index=200,
    )
    records = OpponentIntentDataset(root).shards[0].records()
    batch = decode_opponent_intent_records(records[2:3])
    global_features = np.asarray(batch.global_features[0])

    assert np.allclose(
        global_features[6:10],
        np.asarray([3, 4, 1, 2], dtype=np.float32) / 20.0,
    )
    board_entities = np.asarray(batch.board_entities[0])
    assert np.argmax(board_entities[:, 0, 2:7], axis=-1).tolist() == [
        2,
        3,
        0,
        1,
    ]


def test_target_free_decoder_accepts_public_candidate_afterstates(
    tmp_path: Path,
) -> None:
    root = write_opponent_intent_dataset(
        tmp_path / "target-free",
        split="validation",
        game_index=300,
    )
    records = OpponentIntentDataset(root).shards[0].records()[:2].copy()
    records["opponent_targets"] = np.zeros_like(records["opponent_targets"])
    records["survival_targets"] = np.zeros_like(records["survival_targets"])
    records["final_scores"] = 0

    inputs = decode_opponent_intent_inputs(records)

    assert inputs.board_entities.shape == (2, 4, 23, 31)
    assert inputs.market_entities.shape == (2, 4, 31)
    assert inputs.history_features.shape == (2, 12, 55)
    with pytest.raises(OpponentIntentDatasetError, match="disposition"):
        decode_opponent_intent_records(records)
