from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).with_name("feature_schema_activation_census.py")
_SPEC = importlib.util.spec_from_file_location("feature_schema_activation_census", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
census = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = census
_SPEC.loader.exec_module(census)


def _write_graded_fixture(
    root: Path,
    *,
    dataset_id: str = "synthetic-graded",
    split: str = "train",
    seed: int = 9000,
) -> Path:
    root.mkdir()
    shard = root / f"seed-{seed}.gov"
    payload = bytearray()
    groups = []
    personal_turns = (1, 3, 8, 15)
    for seat, personal_turn in enumerate(personal_turns):
        group = np.zeros(1, dtype=census._GRADED_GROUP_DTYPE)
        group["group_id"] = seed * 10 + seat
        group["raw_seed"] = seed
        group["candidate_count"] = 2
        group["selected_index"] = 0
        group["champion_index"] = 1
        group["completed_turns"] = (personal_turn - 1) * 4 + seat
        group["current_player"] = seat
        group["personal_turn"] = personal_turn
        group["phase"] = min((personal_turn - 1) * 3 // 20, 2)
        position = group["position"][0]
        position["game_index"] = seed
        position["turn"] = group["completed_turns"][0]
        position["active_seat"] = seat
        position["player_count"] = 4
        position["total_turns"] = 80
        position["board_counts"] = 1
        position["nature_tokens"] = np.arange(4, dtype=np.uint8)
        position["scoring_cards"] = 0
        position["wildlife_counts"] = seat
        position["habitat_sizes"] = seat + 1
        position["board_entities"] = 255
        position["board_entities"][:, 0, :] = 0
        position["market_entities"] = 255
        position["market_entities"][:, 0] = 0
        position["market_entities"][:, 2] = 1
        position["market_entities"][:, 3] = np.arange(4, dtype=np.uint8)
        group["public_supply"] = np.arange(30, dtype=np.uint8)

        candidates = np.zeros(2, dtype=census._GRADED_CANDIDATE_DTYPE)
        for index in range(2):
            candidates["action_hash"][index] = np.frombuffer(
                census.blake3.blake3(f"{seed}:{seat}:{index}".encode()).digest(),
                dtype=np.uint8,
            )
            candidates["canonical_index"][index] = index
            candidates["screen_rank"][index] = 1
            candidates["model_immediate_score"][index] = 5.0
            candidates["model_remaining_value"][index] = 50.0
            candidates["screen_value"][index] = 55.0
            candidates["uniform_market_survival_proxy"][index] = 0.5
            candidates["visible_wildlife_count"][index] = 2
            candidates["public_bag_wildlife_count"][index] = 10
            action = candidates["action"][index]
            action["same_slot_independent"] = 1
            action["draft_kind"] = 0
            action["tile_slot"] = 0
            action["wildlife_slot"] = 0
            action["tile_id"] = 7
            action["tile_terrain_a"] = 0
            action["tile_terrain_b"] = 255
            action["tile_wildlife_mask"] = 3
            action["tile_keystone"] = 1
            action["drafted_wildlife"] = 0
            action["tile_q"] = seat
            action["tile_r"] = -seat
            action["rotation"] = 0
            action["wildlife_present"] = 1
            action["wildlife_q"] = seat
            action["wildlife_r"] = -seat
            action["staged_market_entities"] = position["market_entities"]
            action["staged_public_supply"] = group["public_supply"][0]
            action["immediate_score"] = 12
            action["immediate_deltas"] = 1
        groups.append((group.tobytes(), candidates.tobytes()))

    header = census._GRADED_HEADER.pack(
        census.GRADED_MAGIC,
        1,
        census.GRADED_HEADER_SIZE,
        census.GRADED_GROUP_HEADER_SIZE,
        census.GRADED_CANDIDATE_SIZE,
        8,
        4,
        1,
        0,
        4,
        0,
        0,
        seed,
        census.blake3.blake3(census.GRADED_FEATURE_SCHEMA.encode()).digest(),
        census.blake3.blake3(census.GRADED_TARGET_SCHEMA.encode()).digest(),
        b"\0" * 8,
    )
    payload.extend(header)
    for group, candidates in groups:
        payload.extend(group)
        payload.extend(candidates)
    shard.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "feature_schema": census.GRADED_FEATURE_SCHEMA,
        "position_feature_schema": census.POSITION_FEATURE_SCHEMA,
        "target_schema": census.GRADED_TARGET_SCHEMA,
        "group_header_size": census.GRADED_GROUP_HEADER_SIZE,
        "candidate_record_size": census.GRADED_CANDIDATE_SIZE,
        "action_feature_size": census.GRADED_ACTION_STORAGE_SIZE,
        "public_supply_size": census.GRADED_PUBLIC_SUPPLY_SIZE,
        "maximum_wildlife_wipes": census.GRADED_MAX_WIPES,
        "split": split,
        "total_records": 8,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": seed,
                "game_count": 1,
                "group_count": 4,
                "record_count": 8,
                "byte_count": len(payload),
                "blake3": census.checksum(shard),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return root


def _run_census(root: Path, output: Path, *, limit: int | None = None) -> dict:
    args = [
        "census",
        "--train-root",
        str(root),
        "--output",
        str(output),
        "--batch-rows",
        "2",
        "--collision-sample-modulus",
        "1",
    ]
    if limit is not None:
        args.extend(["--row-limit", str(limit)])
    assert census.main(args) == 0
    return json.loads(output.read_text())


def _block(report: dict, block_id: str) -> dict:
    return next(
        block
        for block in report["scientific"]["blocks"]
        if block["block_id"] == block_id
    )


def test_manifest_is_deterministic_complete_and_marks_future_schemas() -> None:
    first = census.build_manifest()
    second = census.build_manifest()
    assert first == second
    assert census.scientific_blake3(
        {key: value for key, value in first.items() if key != "scientific_blake3"}
    ) == first["scientific_blake3"]
    schemas = {schema["name"]: schema for schema in first["schemas"]}
    assert schemas["legacy-mid-v4opp-11231"]["blocks"][-1]["ownership"] == {
        "kind": "feature_index_range",
        "start": 10_862,
        "stop": 11_231,
    }
    future = schemas["relational-opportunity-graph-v0"]["blocks"][0]
    assert future["implementation_status"] == "unimplemented"
    assert future["measurement_status"] == "unmeasurable"
    assert first["experiment_contract"]["minimum_candidate_rows"] == 1_000_000


def test_input_manifest_scientific_identity_excludes_runtime_metadata() -> None:
    first = {
        "dataset_id": "stable",
        "created_unix_seconds": 1,
        "dataset_root": "/machine-a/data",
        "execution": {"hostname": "machine-a"},
        "shards": [{"file": "seed-1.gov", "blake3": "abc"}],
    }
    second = {
        "dataset_id": "stable",
        "created_unix_seconds": 2,
        "dataset_root": "/machine-b/data",
        "execution": {"hostname": "machine-b"},
        "shards": [{"file": "seed-1.gov", "blake3": "abc"}],
    }
    assert census.manifest_scientific_blake3(first) == census.manifest_scientific_blake3(
        second
    )


def test_accumulator_reports_dead_constant_rare_and_exact_aliases() -> None:
    spec = census.BlockSpec(
        block_id="synthetic.block",
        schema="synthetic",
        name="synthetic",
        ownership={"kind": "tensor"},
        semantic_owner="test",
        value_domain="binary",
        expected_d6_behavior="invariant",
        perspective_convention="focal",
        incremental_dependencies=(),
        compatibility="test",
        row_domain="test",
        width=4,
    )
    accumulator = census.BlockAccumulator(
        spec,
        rare_threshold=0.5,
        exact_alias_cell_limit=100,
    )
    accumulator.update(
        np.asarray(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=np.float32,
        ),
        phases=["opening", "early", "middle"],
        seats=["0", "1", "2"],
    )
    result = accumulator.finish()
    assert result["dead_channels"] == [0]
    assert result["constant_channels"] == [0, 1, 2]
    assert result["rare_channels"] == [3]
    assert [1, 2] in result["alias_analysis"]["exact_alias_groups"]
    assert result["phase_seat"]["opening"]["0"]["rows"] == 1


def test_graded_fixture_accounts_for_phase_seat_collisions_and_row_limit(
    tmp_path: Path,
) -> None:
    root = _write_graded_fixture(tmp_path / "graded")
    report = _run_census(root, tmp_path / "report.json", limit=5)
    action = _block(report, "graded.action.tile_coordinates")["census"]
    assert action["rows"] == 5
    assert action["phase_seat"]["opening"]["0"]["rows"] == 2
    assert action["phase_seat"]["early"]["1"]["rows"] == 2
    assert action["phase_seat"]["middle"]["2"]["rows"] == 1
    assert action["phase_seat"]["late"]["3"]["rows"] == 0
    collision = report["scientific"]["representation_collisions"][
        "graded_candidate_bundle"
    ]
    assert collision["verified_representation_collisions"] > 0
    assert report["scientific"]["closed_domains"]["hidden_teacher_values_used_as_features"] is False


def test_checksum_rejection_happens_before_interpretation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _write_graded_fixture(tmp_path / "graded")
    shard = next(root.glob("*.gov"))
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)
    status = census.main(
        [
            "census",
            "--train-root",
            str(root),
            "--output",
            str(tmp_path / "report.json"),
            "--row-limit",
            "1",
        ]
    )
    assert status == 2
    assert "checksum/size mismatch" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_report_hash_and_merge_are_deterministic_and_reject_overlap(
    tmp_path: Path,
) -> None:
    left_root = _write_graded_fixture(
        tmp_path / "left",
        dataset_id="left",
        seed=9100,
    )
    right_root = _write_graded_fixture(
        tmp_path / "right",
        dataset_id="right",
        seed=9200,
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left = _run_census(left_root, left_path, limit=3)
    right = _run_census(right_root, right_path, limit=3)
    assert census.scientific_blake3(left["scientific"]) == left["scientific_blake3"]
    assert census.scientific_blake3(right["scientific"]) == right["scientific_blake3"]

    merged_lr = census.merge_reports([left_path, right_path])
    merged_rl = census.merge_reports([right_path, left_path])
    assert merged_lr["scientific_blake3"] == merged_rl["scientific_blake3"]
    assert merged_lr["provenance"] == merged_rl["provenance"]
    assert len(merged_lr["provenance"]["input_manifests"]) == 2
    assert _block(merged_lr, "graded.action.tile_coordinates")["census"]["rows"] == 6
    wipe_aliases = _block(merged_lr, "graded.action.wipe_masks")["census"][
        "alias_analysis"
    ]
    assert wipe_aliases["mode"] == "exact_per_shard_intersection"
    assert wipe_aliases["exact_alias_groups"] == [list(range(80))]
    assert (
        merged_lr["scientific"]["representation_collisions"]["merge_status"]
        == "per_shard_byte_verified_cross_shard_unknown"
    )
    with pytest.raises(census.CensusError, match="duplicate evidence"):
        census.merge_reports([left_path, left_path])


def test_legacy_sparse_stream_requires_manifest_and_counts_all_seats(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    shard = root / "rows.jsonl"
    rows = [
        {"features": [0, 4_851], "phase": phase, "focal_seat": seat}
        for seat, phase in enumerate(census.PHASES)
    ]
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 1,
        "dataset_id": "legacy-synthetic",
        "feature_schema": "legacy-mid-v4opp-11231",
        "feature_count": 11_231,
        "split": "validation",
        "rows": len(rows),
        "shards": [
            {
                "file": shard.name,
                "row_count": len(rows),
                "blake3": census.checksum(shard),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "legacy-report.json"
    assert (
        census.main(
            [
                "census",
                "--legacy-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    cell = _block(report, "legacy.cell_core")["census"]
    assert cell["constant_channels"] == [0]
    for seat, phase in enumerate(census.PHASES):
        assert cell["phase_seat"][phase][str(seat)]["rows"] == 1
