from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import pytest
import s1_semantic_supply_merge as merge


def _catalog() -> list[dict]:
    terrains = ["Mountain", "Forest", "Prairie", "Wetland", "River"]
    catalog = []
    for archetype_id in range(75):
        terrain = terrains[archetype_id % len(terrains)]
        catalog.append(
            {
                "id": archetype_id,
                "archetype": {
                    "primary_terrain": terrain,
                    "secondary_terrain": None,
                    "directed_edges": [terrain] * 6,
                    "wildlife": 1 << (archetype_id % 5),
                    "keystone": True,
                },
                "standard_tile_count": 2 if archetype_id < 10 else 1,
            }
        )
    return catalog


def _supply_bytes(catalog_hash: bytes, counts: list[int], turn: int) -> bytes:
    wildlife = [max(0, 20 - turn // 5)] * 5
    drawable = max(0, sum(counts) - 2)
    encoded = bytearray(merge.SUPPLY_MAGIC)
    encoded.extend(struct.pack("<H", 1))
    encoded.extend(catalog_hash)
    encoded.extend(struct.pack("<5H", *wildlife))
    encoded.extend(struct.pack("<HHH", sum(counts), drawable, len(counts)))
    encoded.extend(struct.pack(f"<{len(counts)}H", *counts))
    return bytes(encoded)


def _report(split: str, shard_index: int, shard_count: int = 2) -> dict:
    catalog = _catalog()
    catalog_hash = blake3.blake3(merge.catalog_bytes(catalog)).digest()
    first_game_index = 300_000 if split == "train" else 310_000
    requested_games = 4
    selected = [
        first_game_index + offset
        for offset in range(requested_games)
        if offset % shard_count == shard_index
    ]
    records = []
    for game_index in selected:
        for turn in range(80):
            counts = [definition["standard_tile_count"] for definition in catalog]
            remaining_to_remove = min(turn + 4, 83)
            for index in range(len(counts)):
                removed = min(counts[index], remaining_to_remove)
                counts[index] -= removed
                remaining_to_remove -= removed
                if remaining_to_remove == 0:
                    break
            supply_bytes = _supply_bytes(catalog_hash, counts, turn)
            drawable = max(0, sum(counts) - 2)
            refill_hashes = [
                (
                    merge.refill_hash(catalog_hash, counts, slots)
                    if slots <= min(4, drawable)
                    else None
                )
                for slots in range(1, 5)
            ]
            records.append(
                {
                    "game_index": game_index,
                    "turn": turn,
                    "active_player": turn % 4,
                    "public_state_blake3": blake3.blake3(
                        f"{game_index}:{turn}".encode()
                    ).hexdigest(),
                    "semantic_supply_blake3": blake3.blake3(supply_bytes).hexdigest(),
                    "semantic_supply_bytes_hex": supply_bytes.hex(),
                    "unseen_tile_count": sum(counts),
                    "drawable_tile_count": drawable,
                    "excluded_tile_count": sum(counts) - drawable,
                    "wildlife_bag_counts": [max(0, 20 - turn // 5)] * 5,
                    "archetype_counts": counts,
                    "market_archetype_ids": [0, 1, 2, 3],
                    "refill_distribution_blake3_by_slots": refill_hashes,
                }
            )
    report = {
        "schema_version": 1,
        "experiment_id": merge.EXPERIMENT_ID,
        "semantic_supply_schema_version": 1,
        "semantic_supply_schema": merge.SEMANTIC_SCHEMA,
        "archetype_schema": merge.ARCHETYPE_SCHEMA,
        "catalog_blake3": catalog_hash.hex(),
        "request": {
            "split": split,
            "strategy": "random-v1",
            "first_game_index": first_game_index,
            "requested_games": requested_games,
        },
        "shard": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "partition_rule": ("(game_index - first_game_index) % shard_count == shard_index"),
            "selected_game_indices": selected,
        },
        "provenance": {
            "source": {
                "git_revision": "unavailable",
                "git_dirty": True,
                "git_status_blake3": "0" * 64,
                "v2_source_blake3": "1" * 64,
            },
            "executable_blake3": "2" * 64,
        },
        "catalog": catalog,
        "legacy_collision_witness": {
            "left_standard_tile_ids": [0, 23],
            "right_standard_tile_ids": [2, 20],
            "left_archetype_ids": [0, 0],
            "right_archetype_ids": [0, 1],
            "shared_legacy_tile_marginals": [0] * 25,
            "exact_archetype_multisets_differ": True,
        },
        "summary": {
            "selected_games": len(selected),
            "positions": len(records),
        },
        "records": records,
        "scientific_blake3": "",
    }
    report["scientific_blake3"] = merge.shard_scientific_digest(report)
    return report


def _write_reports(tmp_path: Path) -> list[Path]:
    paths = []
    for split in ("train", "validation"):
        for shard_index in range(2):
            path = tmp_path / f"{split}-{shard_index}.json"
            path.write_text(json.dumps(_report(split, shard_index)))
            paths.append(path)
    return paths


def test_merge_is_complete_and_input_order_independent(tmp_path: Path) -> None:
    paths = _write_reports(tmp_path)
    forward = merge.merge_reports(paths, 2)
    reverse = merge.merge_reports(list(reversed(paths)), 2)
    assert forward == reverse
    assert forward["classification"] == "exact_semantic_supply_census_complete"
    assert forward["positions"] == 8 * 80
    assert forward["coverage"]["train"]["complete_game_interval"] is True
    assert forward["coverage"]["validation"]["complete_game_interval"] is True
    assert forward["all_exact_checks_passed"] is True


def test_merge_rejects_supply_byte_or_refill_hash_tampering(tmp_path: Path) -> None:
    paths = _write_reports(tmp_path)
    report = json.loads(paths[0].read_text())
    report["records"][0]["semantic_supply_bytes_hex"] = "00"
    paths[0].write_text(json.dumps(report))
    with pytest.raises(merge.MergeError, match=r"magic|BLAKE3"):
        merge.merge_reports(paths, 2)

    paths = _write_reports(tmp_path)
    report = json.loads(paths[0].read_text())
    report["records"][0]["refill_distribution_blake3_by_slots"][0] = "0" * 64
    report["scientific_blake3"] = merge.shard_scientific_digest(report)
    paths[0].write_text(json.dumps(report))
    with pytest.raises(merge.MergeError, match="refill distribution"):
        merge.merge_reports(paths, 2)


def test_merge_rejects_missing_or_overlapping_partition_evidence(tmp_path: Path) -> None:
    paths = _write_reports(tmp_path)
    with pytest.raises(merge.MergeError, match="one train and one validation"):
        merge.merge_reports(paths[:-1], 2)

    report = json.loads(paths[1].read_text())
    report["shard"]["shard_index"] = 0
    report["scientific_blake3"] = merge.shard_scientific_digest(report)
    paths[1].write_text(json.dumps(report))
    with pytest.raises(merge.MergeError, match=r"selected game indices|one shard"):
        merge.merge_reports(paths, 2)


def test_catalog_payload_is_cryptographically_bound() -> None:
    catalog = _catalog()
    original = blake3.blake3(merge.catalog_bytes(catalog)).hexdigest()
    catalog[0]["archetype"]["wildlife"] = 3
    changed = blake3.blake3(merge.catalog_bytes(catalog)).hexdigest()
    assert original != changed
