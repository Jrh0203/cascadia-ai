#!/usr/bin/env python3
"""Build the exact 100k compact index and packing plans from validated replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.r2_map_dataset import (
    COMPACT_INDEX_SCHEMA,
    _aggregate_source_manifests,
    compact_packing_plan,
    validate_compact_index,
)


class FastIndexError(RuntimeError):
    pass


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if compact:
        temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    else:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _inspect(exporter: Path, shard: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(exporter), "inspect-r2-map-index-metadata", "--shard", str(shard)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if (
        value.get("schema_version") != 1
        or value.get("schema_id") != "r2-map-compact-index-metadata-v1"
        or len(value.get("dataset_manifest", {}).get("sources", [])) != 1
    ):
        raise FastIndexError(f"metadata schema differs for {shard}")
    return value


def build(
    *,
    shard_root: Path,
    exporter: Path,
    aggregate_receipt: Path,
    index_path: Path,
    plan_path: Path,
    receipt_path: Path,
    workers: int,
    image_id: str,
) -> dict[str, Any]:
    started = time.time_ns()
    aggregate = json.loads(aggregate_receipt.read_text())
    if (
        aggregate.get("result") != "pass"
        or aggregate.get("games") != 100_000
        or aggregate.get("primary_example_count") != 8_000_000
        or aggregate.get("replay_shards") != 420
        or aggregate.get("completion_audits") != 30
    ):
        raise FastIndexError("aggregate corpus receipt is not the exact validated 100k gate")
    shards = sorted(shard_root.glob("*.r2sh"))
    if len(shards) != 420:
        raise FastIndexError("flat replay shard count differs")
    expected = {item["file_name"]: item for item in aggregate["shards"]}
    if set(expected) != {path.name for path in shards}:
        raise FastIndexError("flat replay shard names differ from aggregate receipt")
    for shard in shards:
        item = expected[shard.name]
        if shard.stat().st_size != item["bytes"]:
            raise FastIndexError(f"flat replay byte count differs: {shard}")

    values: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_inspect, exporter, shard): shard for shard in shards}
        completed = 0
        for future in as_completed(futures):
            values.append(future.result())
            completed += 1
            if completed % 42 == 0 or completed == len(shards):
                print(
                    json.dumps(
                        {
                            "event": "index-progress",
                            "completed_shards": completed,
                            "total_shards": len(shards),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    manifests = [value["dataset_manifest"] for value in values]
    manifest = _aggregate_source_manifests(manifests)
    games_with_widths = [game for value in values for game in value["games"]]
    games_with_widths.sort(key=lambda game: (game["global_game_index"], game["game_id"]))
    if [game["global_game_index"] for game in games_with_widths] != list(range(100_000)):
        raise FastIndexError("compact index does not cover global game indices exactly once")
    catalog: dict[str, dict[str, tuple[int, ...]]] = {}
    games: list[dict[str, Any]] = []
    for value in games_with_widths:
        widths = tuple(value["candidate_widths"])
        if value["split"] == "train":
            catalog.setdefault(value["source_file_name"], {})[value["game_id"]] = widths
        games.append(value)
    index: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": COMPACT_INDEX_SCHEMA,
        "dataset_manifest": manifest,
        "games": games,
    }
    index["index_blake3"] = _canonical_blake3(index)
    _atomic_json(index_path, index, compact=True)
    validated = validate_compact_index(index_path, shard_root=shard_root)
    if (
        validated["dataset_manifest"]["game_count"] != 100_000
        or validated["dataset_manifest"]["example_count"] != 8_000_000
    ):
        raise FastIndexError("validated compact index totals differ")

    plans = [
        compact_packing_plan(
            validated,
            catalog,
            group_batch_size=cap,
            maximum_candidates_per_batch=16_384,
            seed=20_260_618,
            epochs=12,
        )
        for cap in (16, 32, 64, 128)
    ]
    selected = min(
        plans,
        key=lambda plan: (
            plan["totals"]["steps"],
            plan["totals"]["padded_draft_candidates"],
            -plan["group_batch_size"],
        ),
    )
    repeated = compact_packing_plan(
        validated,
        catalog,
        group_batch_size=selected["group_batch_size"],
        maximum_candidates_per_batch=16_384,
        seed=20_260_618,
        epochs=12,
    )
    if repeated != selected:
        raise FastIndexError("selected packing plan is not deterministic")
    _atomic_json(plan_path, selected)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    finished = time.time_ns()
    receipt: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.bootstrap-packing-selection.v1",
        "schema_version": 1,
        "result": "pass",
        "dataset_blake3": manifest["dataset_blake3"],
        "index_blake3": validated["index_blake3"],
        "index_sha256": index_sha,
        "aggregate_receipt_sha256": hashlib.sha256(
            aggregate_receipt.read_bytes()
        ).hexdigest(),
        "image_id": image_id,
        "metadata_workers": workers,
        "games": manifest["game_count"],
        "examples": manifest["example_count"],
        "replay_shards": len(shards),
        "candidate_widths": sum(len(widths) for source in catalog.values() for widths in source.values()),
        "plans": [
            {
                "group_batch_size": plan["group_batch_size"],
                "steps": plan["totals"]["steps"],
                "draft_groups": plan["totals"]["draft_groups"],
                "draft_candidates": plan["totals"]["draft_candidates"],
                "padded_draft_candidates": plan["totals"]["padded_draft_candidates"],
            }
            for plan in plans
        ],
        "selection_rule": "minimum exact 12-epoch steps, then minimum padding, then larger cap",
        "selected_group_batch_size": selected["group_batch_size"],
        "selected_schedule_steps": selected["totals"]["steps"],
        "selected_plan_sha256": plan_sha,
        "started_unix_ms": started // 1_000_000,
        "finished_unix_ms": finished // 1_000_000,
        "wall_seconds": (finished - started) / 1_000_000_000,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--aggregate-receipt", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--selected-plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                shard_root=args.shard_root,
                exporter=args.exporter,
                aggregate_receipt=args.aggregate_receipt,
                index_path=args.index,
                plan_path=args.selected_plan,
                receipt_path=args.receipt,
                workers=args.workers,
                image_id=args.image_id,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
