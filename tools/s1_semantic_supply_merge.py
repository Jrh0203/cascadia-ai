#!/usr/bin/env python3
"""Validate and deterministically merge source-frozen S1 census shards."""

# ruff: noqa: B905, UP045 - cluster tools must run under macOS system Python 3.9.

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Optional

import blake3

SCHEMA_VERSION = 1
EXPERIMENT_ID = "exact-semantic-supply-v1"
SEMANTIC_SCHEMA = "exact-semantic-supply-v1"
ARCHETYPE_SCHEMA = "canonical-public-tile-archetype-v1"
SUPPLY_MAGIC = b"CSSSUP1\0"
REFILL_MAGIC = b"CSSRFL1\0"
CATALOG_MAGIC = b"CSSCAT1\0"
TERRAINS = {
    "Mountain": 0,
    "Forest": 1,
    "Prairie": 2,
    "Wetland": 3,
    "River": 4,
}


class MergeError(RuntimeError):
    """Raised when S1 evidence is incomplete, inconsistent, or corrupted."""


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MergeError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MergeError(f"{label} must be an array")
    return value


def _require_int(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise MergeError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{label} must be a nonempty string")
    return value


def _decode_hex(value: object, label: str, expected_bytes: Optional[int] = None) -> bytes:
    text = _require_string(value, label)
    try:
        decoded = bytes.fromhex(text)
    except ValueError as error:
        raise MergeError(f"{label} must be lowercase hexadecimal") from error
    if text != decoded.hex():
        raise MergeError(f"{label} must use canonical lowercase hexadecimal")
    if expected_bytes is not None and len(decoded) != expected_bytes:
        raise MergeError(f"{label} must contain {expected_bytes} bytes")
    return decoded


def _terrain_code(value: object, label: str) -> int:
    if value not in TERRAINS:
        raise MergeError(f"{label} has an unknown terrain")
    return TERRAINS[str(value)]


def catalog_bytes(catalog: list[Any]) -> bytes:
    encoded = bytearray(CATALOG_MAGIC)
    encoded.extend(struct.pack("<HH", SCHEMA_VERSION, len(catalog)))
    for index, raw_definition in enumerate(catalog):
        definition = _require_dict(raw_definition, f"catalog[{index}]")
        if _require_int(definition.get("id"), f"catalog[{index}].id") != index:
            raise MergeError("catalog IDs must be contiguous and sorted")
        archetype = _require_dict(
            definition.get("archetype"),
            f"catalog[{index}].archetype",
        )
        primary = _terrain_code(
            archetype.get("primary_terrain"),
            f"catalog[{index}].archetype.primary_terrain",
        )
        secondary_value = archetype.get("secondary_terrain")
        secondary = (
            255
            if secondary_value is None
            else _terrain_code(
                secondary_value,
                f"catalog[{index}].archetype.secondary_terrain",
            )
        )
        edges = _require_list(
            archetype.get("directed_edges"),
            f"catalog[{index}].archetype.directed_edges",
        )
        if len(edges) != 6:
            raise MergeError("every archetype must contain six directed edges")
        wildlife = _require_int(
            archetype.get("wildlife"),
            f"catalog[{index}].archetype.wildlife",
        )
        if wildlife > 0b1_1111:
            raise MergeError("archetype wildlife mask exceeds five public species")
        keystone = archetype.get("keystone")
        if not isinstance(keystone, bool):
            raise MergeError("archetype keystone flag must be boolean")
        standard_count = _require_int(
            definition.get("standard_tile_count"),
            f"catalog[{index}].standard_tile_count",
            minimum=1,
        )
        encoded.extend(struct.pack("<HBB", index, primary, secondary))
        encoded.extend(
            bytes(
                _terrain_code(
                    terrain,
                    f"catalog[{index}].archetype.directed_edges",
                )
                for terrain in edges
            )
        )
        encoded.extend(struct.pack("<BBH", wildlife, int(keystone), standard_count))
    return bytes(encoded)


def parse_supply_bytes(
    encoded: bytes,
    *,
    expected_catalog_hash: bytes,
    expected_count_len: int,
) -> dict[str, Any]:
    minimum = 8 + 2 + 32 + 10 + 2 + 2 + 2
    if len(encoded) < minimum or encoded[:8] != SUPPLY_MAGIC:
        raise MergeError("semantic supply bytes have invalid magic or length")
    version = struct.unpack_from("<H", encoded, 8)[0]
    if version != SCHEMA_VERSION:
        raise MergeError("semantic supply bytes use an unsupported schema")
    catalog_hash = encoded[10:42]
    if catalog_hash != expected_catalog_hash:
        raise MergeError("semantic supply bytes reference a different catalog")
    offset = 42
    wildlife_bag = list(struct.unpack_from("<5H", encoded, offset))
    offset += 10
    unseen_tile_count, drawable_tile_count, count_len = struct.unpack_from("<HHH", encoded, offset)
    offset += 6
    if count_len != expected_count_len:
        raise MergeError("semantic supply count vector length differs from the catalog")
    expected_length = offset + count_len * 2
    if len(encoded) != expected_length:
        raise MergeError("semantic supply bytes are truncated or contain trailing data")
    counts = list(struct.unpack_from(f"<{count_len}H", encoded, offset))
    if sum(counts) != unseen_tile_count:
        raise MergeError("semantic supply archetype counts do not conserve unseen tiles")
    if drawable_tile_count > unseen_tile_count:
        raise MergeError("semantic supply drawable count exceeds unseen tiles")
    if any(count > 20 for count in wildlife_bag):
        raise MergeError("semantic supply wildlife count exceeds official multiplicity")
    return {
        "wildlife_bag_counts": wildlife_bag,
        "unseen_tile_count": unseen_tile_count,
        "drawable_tile_count": drawable_tile_count,
        "excluded_tile_count": unseen_tile_count - drawable_tile_count,
        "archetype_counts": counts,
    }


def refill_hash(catalog_hash: bytes, counts: list[int], slots: int) -> str:
    encoded = bytearray(REFILL_MAGIC)
    encoded.extend(struct.pack("<HB", SCHEMA_VERSION, slots))
    encoded.extend(catalog_hash)
    encoded.extend(struct.pack("<H", len(counts)))
    encoded.extend(struct.pack(f"<{len(counts)}H", *counts))
    return blake3.blake3(bytes(encoded)).hexdigest()


def _update_len_prefixed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def shard_scientific_digest(report: dict[str, Any]) -> str:
    request = _require_dict(report.get("request"), "request")
    shard = _require_dict(report.get("shard"), "shard")
    records = _require_list(report.get("records"), "records")
    digest = blake3.blake3()
    digest.update(b"cascadia-exact-semantic-supply-census-v1\0")
    _update_len_prefixed(
        digest,
        _require_string(report.get("catalog_blake3"), "catalog_blake3").encode(),
    )
    _update_len_prefixed(digest, _require_string(request.get("split"), "request.split").encode())
    _update_len_prefixed(
        digest,
        _require_string(request.get("strategy"), "request.strategy").encode(),
    )
    digest.update(
        struct.pack(
            "<Q",
            _require_int(request.get("first_game_index"), "first_game_index"),
        )
    )
    digest.update(
        struct.pack(
            "<Q",
            _require_int(request.get("requested_games"), "requested_games"),
        )
    )
    digest.update(struct.pack("<Q", _require_int(shard.get("shard_index"), "shard_index")))
    digest.update(struct.pack("<Q", _require_int(shard.get("shard_count"), "shard_count")))
    digest.update(struct.pack("<Q", len(records)))
    for index, raw_record in enumerate(records):
        record = _require_dict(raw_record, f"records[{index}]")
        digest.update(struct.pack("<Q", _require_int(record.get("game_index"), "game_index")))
        digest.update(struct.pack("<H", _require_int(record.get("turn"), "turn")))
        digest.update(struct.pack("<Q", _require_int(record.get("active_player"), "active_player")))
        _update_len_prefixed(
            digest,
            _require_string(record.get("public_state_blake3"), "public_state_blake3").encode(),
        )
        _update_len_prefixed(
            digest,
            _require_string(
                record.get("semantic_supply_blake3"),
                "semantic_supply_blake3",
            ).encode(),
        )
    return digest.hexdigest()


def validate_report(report: dict[str, Any], expected_shard_count: int) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise MergeError("unsupported S1 census report schema")
    if report.get("experiment_id") != EXPERIMENT_ID:
        raise MergeError("unexpected S1 experiment ID")
    if report.get("semantic_supply_schema_version") != SCHEMA_VERSION:
        raise MergeError("unexpected semantic supply schema version")
    if report.get("semantic_supply_schema") != SEMANTIC_SCHEMA:
        raise MergeError("unexpected semantic supply schema")
    if report.get("archetype_schema") != ARCHETYPE_SCHEMA:
        raise MergeError("unexpected tile archetype schema")

    catalog = _require_list(report.get("catalog"), "catalog")
    catalog_hash = _decode_hex(report.get("catalog_blake3"), "catalog_blake3", 32)
    if blake3.blake3(catalog_bytes(catalog)).digest() != catalog_hash:
        raise MergeError("catalog payload does not match catalog_blake3")
    standard_counts = [
        _require_int(
            _require_dict(definition, f"catalog[{index}]").get("standard_tile_count"),
            f"catalog[{index}].standard_tile_count",
            minimum=1,
        )
        for index, definition in enumerate(catalog)
    ]
    if sum(standard_counts) != 85:
        raise MergeError("semantic catalog does not conserve the 85 standard tiles")

    request = _require_dict(report.get("request"), "request")
    split = _require_string(request.get("split"), "request.split")
    if split not in {"train", "validation"}:
        raise MergeError("S1 census must remain on open train or validation data")
    first_game_index = _require_int(request.get("first_game_index"), "first_game_index")
    requested_games = _require_int(request.get("requested_games"), "requested_games", minimum=1)
    _require_string(request.get("strategy"), "request.strategy")

    shard = _require_dict(report.get("shard"), "shard")
    shard_index = _require_int(shard.get("shard_index"), "shard_index")
    shard_count = _require_int(shard.get("shard_count"), "shard_count", minimum=1)
    if shard_count != expected_shard_count or shard_index >= shard_count:
        raise MergeError("S1 shard identity does not match the expected partition")
    if (
        shard.get("partition_rule")
        != "(game_index - first_game_index) % shard_count == shard_index"
    ):
        raise MergeError("S1 shard uses an unknown partition rule")
    selected = _require_list(shard.get("selected_game_indices"), "selected_game_indices")
    expected_selected = [
        first_game_index + offset
        for offset in range(requested_games)
        if offset % shard_count == shard_index
    ]
    if selected != expected_selected:
        raise MergeError("S1 selected game indices do not match modulo ownership")

    records = _require_list(report.get("records"), "records")
    if len(records) != len(expected_selected) * 80:
        raise MergeError("S1 record count is not exactly 80 positions per selected game")
    for game_offset, game_index in enumerate(expected_selected):
        rows = records[game_offset * 80 : (game_offset + 1) * 80]
        for turn, raw_record in enumerate(rows):
            record = _require_dict(raw_record, f"records[{game_offset * 80 + turn}]")
            if record.get("game_index") != game_index or record.get("turn") != turn:
                raise MergeError("S1 records are not in exact game/turn order")
            active_player = _require_int(record.get("active_player"), "active_player")
            if active_player != turn % 4:
                raise MergeError("S1 active player does not match four-player turn order")
            _decode_hex(record.get("public_state_blake3"), "public_state_blake3", 32)
            supply_bytes = _decode_hex(
                record.get("semantic_supply_bytes_hex"),
                "semantic_supply_bytes_hex",
            )
            supply_hash = _decode_hex(
                record.get("semantic_supply_blake3"),
                "semantic_supply_blake3",
                32,
            )
            if blake3.blake3(supply_bytes).digest() != supply_hash:
                raise MergeError("semantic supply bytes do not match their BLAKE3")
            parsed = parse_supply_bytes(
                supply_bytes,
                expected_catalog_hash=catalog_hash,
                expected_count_len=len(catalog),
            )
            for field in (
                "wildlife_bag_counts",
                "unseen_tile_count",
                "drawable_tile_count",
                "excluded_tile_count",
                "archetype_counts",
            ):
                if record.get(field) != parsed[field]:
                    raise MergeError(f"record {field} differs from canonical supply bytes")
            if any(
                count > maximum
                for count, maximum in zip(parsed["archetype_counts"], standard_counts)
            ):
                raise MergeError("record archetype count exceeds catalog multiplicity")
            market_ids = _require_list(record.get("market_archetype_ids"), "market_archetype_ids")
            if len(market_ids) != 4 or any(
                value is not None
                and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value < len(catalog)
                )
                for value in market_ids
            ):
                raise MergeError("market archetype links are malformed")
            refill_hashes = _require_list(
                record.get("refill_distribution_blake3_by_slots"),
                "refill_distribution_blake3_by_slots",
            )
            if len(refill_hashes) != 4:
                raise MergeError("record must expose four bounded refill horizons")
            maximum_slots = min(4, parsed["drawable_tile_count"])
            for slots, value in enumerate(refill_hashes, start=1):
                if slots <= maximum_slots:
                    if value != refill_hash(catalog_hash, parsed["archetype_counts"], slots):
                        raise MergeError("refill distribution hash is inconsistent")
                elif value is not None:
                    raise MergeError("refill hash exists beyond the remaining supply")

    if shard_scientific_digest(report) != report.get("scientific_blake3"):
        raise MergeError("S1 shard scientific digest does not match its records")

    provenance = _require_dict(report.get("provenance"), "provenance")
    source = _require_dict(provenance.get("source"), "provenance.source")
    source_hash = _decode_hex(
        source.get("v2_source_blake3"),
        "provenance.source.v2_source_blake3",
        32,
    ).hex()
    executable_hash = _decode_hex(
        provenance.get("executable_blake3"),
        "provenance.executable_blake3",
        32,
    ).hex()
    witness = _require_dict(report.get("legacy_collision_witness"), "legacy_collision_witness")
    if witness.get("exact_archetype_multisets_differ") is not True:
        raise MergeError("legacy collision witness does not assert exact separation")
    if len(_require_list(witness.get("shared_legacy_tile_marginals"), "collision marginals")) != 25:
        raise MergeError("legacy collision witness must contain the 25 tile marginals")
    left = sorted(_require_list(witness.get("left_archetype_ids"), "left archetypes"))
    right = sorted(_require_list(witness.get("right_archetype_ids"), "right archetypes"))
    if left == right:
        raise MergeError("legacy collision witness does not differ in exact archetypes")

    return {
        "report": report,
        "split": split,
        "first_game_index": first_game_index,
        "requested_games": requested_games,
        "strategy": request["strategy"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_games": selected,
        "records": records,
        "catalog": catalog,
        "catalog_blake3": catalog_hash.hex(),
        "source_blake3": source_hash,
        "executable_blake3": executable_hash,
        "collision_witness": witness,
    }


def merge_reports(paths: list[Path], expected_shard_count: int) -> dict[str, Any]:
    if expected_shard_count <= 0:
        raise MergeError("expected shard count must be positive")
    if len(paths) != expected_shard_count * 2:
        raise MergeError("S1 merge requires one train and one validation shard per host")
    validated = []
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise MergeError(f"cannot read S1 shard {path}: {error}") from error
        validated.append(validate_report(_require_dict(value, str(path)), expected_shard_count))
    validated.sort(key=lambda item: (item["split"], item["shard_index"]))

    common_fields = ("catalog_blake3", "source_blake3", "executable_blake3")
    for field in common_fields:
        values = {item[field] for item in validated}
        if len(values) != 1:
            raise MergeError(f"S1 shards disagree on {field}")
    catalog = validated[0]["catalog"]
    if any(item["catalog"] != catalog for item in validated):
        raise MergeError("S1 shards disagree on canonical catalog contents")
    witness = validated[0]["collision_witness"]
    if any(item["collision_witness"] != witness for item in validated):
        raise MergeError("S1 shards disagree on the legacy collision witness")

    coverage: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        split_shards = [item for item in validated if item["split"] == split]
        if [item["shard_index"] for item in split_shards] != list(range(expected_shard_count)):
            raise MergeError(f"S1 {split} coverage lacks one shard per host")
        first_values = {item["first_game_index"] for item in split_shards}
        game_values = {item["requested_games"] for item in split_shards}
        strategy_values = {item["strategy"] for item in split_shards}
        if len(first_values) != 1 or len(game_values) != 1 or len(strategy_values) != 1:
            raise MergeError(f"S1 {split} shard requests are inconsistent")
        requested_games = next(iter(game_values))
        selected = sorted(
            game_index for item in split_shards for game_index in item["selected_games"]
        )
        first_game_index = next(iter(first_values))
        if selected != list(range(first_game_index, first_game_index + requested_games)):
            raise MergeError(f"S1 {split} shards do not cover the exact requested interval")
        split_records = [record for item in split_shards for record in item["records"]]
        split_records.sort(key=lambda record: (record["game_index"], record["turn"]))
        all_records.extend(split_records)
        coverage[split] = {
            "first_game_index": first_game_index,
            "requested_games": requested_games,
            "strategy": next(iter(strategy_values)),
            "positions": len(split_records),
            "complete_game_interval": True,
        }

    archetype_count = len(catalog)
    archetype_summary = []
    for archetype_id in range(archetype_count):
        values = [record["archetype_counts"][archetype_id] for record in all_records]
        archetype_summary.append(
            {
                "archetype_id": archetype_id,
                "minimum_unseen_count": min(values),
                "maximum_unseen_count": max(values),
                "zero_count_positions": sum(value == 0 for value in values),
                "total_unseen_occurrences": sum(values),
            }
        )
    unique_supply_states = len({record["semantic_supply_blake3"] for record in all_records})
    shard_receipts = [
        {
            "split": item["split"],
            "shard_index": item["shard_index"],
            "selected_games": len(item["selected_games"]),
            "positions": len(item["records"]),
            "scientific_blake3": item["report"]["scientific_blake3"],
        }
        for item in validated
    ]
    digest = blake3.blake3()
    digest.update(b"cascadia-exact-semantic-supply-merged-v1\0")
    digest.update(bytes.fromhex(validated[0]["catalog_blake3"]))
    for receipt in shard_receipts:
        digest.update(receipt["split"].encode())
        digest.update(struct.pack("<Q", receipt["shard_index"]))
        digest.update(bytes.fromhex(receipt["scientific_blake3"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "classification": "exact_semantic_supply_census_complete",
        "semantic_supply_schema": SEMANTIC_SCHEMA,
        "archetype_schema": ARCHETYPE_SCHEMA,
        "catalog_blake3": validated[0]["catalog_blake3"],
        "source_blake3": validated[0]["source_blake3"],
        "executable_blake3": validated[0]["executable_blake3"],
        "expected_shard_count": expected_shard_count,
        "shard_receipts": shard_receipts,
        "coverage": coverage,
        "positions": len(all_records),
        "unique_supply_states": unique_supply_states,
        "minimum_unseen_tiles": min(record["unseen_tile_count"] for record in all_records),
        "maximum_unseen_tiles": max(record["unseen_tile_count"] for record in all_records),
        "minimum_drawable_tiles": min(record["drawable_tile_count"] for record in all_records),
        "maximum_drawable_tiles": max(record["drawable_tile_count"] for record in all_records),
        "minimum_excluded_tiles": min(record["excluded_tile_count"] for record in all_records),
        "maximum_excluded_tiles": max(record["excluded_tile_count"] for record in all_records),
        "archetype_summary": archetype_summary,
        "legacy_collision_witness": witness,
        "all_exact_checks_passed": True,
        "scientific_blake3": digest.hexdigest(),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--expected-shard-count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        merged = merge_reports(args.shard, args.expected_shard_count)
        write_json(args.output, merged)
        print(json.dumps(merged, sort_keys=True))
        return 0
    except MergeError as error:
        print(f"S1 semantic supply merge error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
