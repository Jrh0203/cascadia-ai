#!/usr/bin/env python3
"""Aggregate a verified 10K-game V3 expert-cycle corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import blake3


class CycleCorpusError(ValueError):
    """The expert-cycle corpus is incomplete or internally inconsistent."""


def _digest(path: Path, algorithm: str) -> str:
    value = blake3.blake3() if algorithm == "blake3" else hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate(
    *,
    cycle: int,
    collection: Path,
    verification: Path,
    accepted_root: Path,
    campaign_state: Path,
) -> dict[str, Any]:
    if not 1 <= cycle <= 10:
        raise CycleCorpusError("cycle is outside 1..=10")
    collected = json.loads(collection.read_text())
    verified = json.loads(verification.read_text())
    state = json.loads(campaign_state.read_text())
    if (
        collected.get("passed") is not True
        or collected.get("cycle") != cycle
        or collected.get("work_items") != 100
        or collected.get("totals", {}).get("games") != 10_000
        or verified.get("passed") is not True
        or verified.get("work_items") != 100
        or verified.get("totals", {}).get("records") != 10_000
        or verified.get("totals", {}).get("expanded_training_entries") != 200_000
        or state.get("phase") != f"cycle-{cycle:02d}-collecting"
        or state.get("protected_seed_values_opened") is not False
    ):
        raise CycleCorpusError("cycle collection, verification, or campaign state differs")
    directories = [
        path
        for path in accepted_root.iterdir()
        if path.is_dir() and path.name != ".receipts"
    ]
    if len(directories) != 100:
        raise CycleCorpusError("accepted cycle artifact count differs from 100")
    files = []
    intervals = []
    for directory in sorted(directories):
        shards = sorted(directory.glob("*.v3g"))
        receipts = sorted(directory.glob("*.receipt.json"))
        if len(shards) != 1 or len(receipts) != 1:
            raise CycleCorpusError(f"cycle artifact set is incomplete: {directory.name}")
        shard = shards[0]
        receipt = json.loads(receipts[0].read_text())
        first = int(receipt.get("first_game_index", -1))
        games = int(receipt.get("games", -1))
        if (
            receipt.get("component") != "expert-iteration"
            or receipt.get("cycle") != cycle
            or games != 100
            or receipt.get("records") != games
            or receipt.get("newest_model_seats_per_expert_game") != 1
            or receipt.get("bytes") != shard.stat().st_size
            or receipt.get("blake3") != _digest(shard, "blake3")
        ):
            raise CycleCorpusError(f"cycle receipt differs: {directory.name}")
        intervals.append((first, first + games))
        files.append(
            {
                "item": directory.name,
                "path": str(shard.resolve()),
                "first_game_index": first,
                "games": games,
                "bytes": shard.stat().st_size,
                "blake3": receipt["blake3"],
                "sha256": _digest(shard, "sha256"),
            }
        )
    intervals.sort()
    expected_first = 2_000_000_000 + cycle * 10_000
    expected = [
        (first, first + 100)
        for first in range(expected_first, expected_first + 10_000, 100)
    ]
    if intervals != expected:
        raise CycleCorpusError("cycle game-index intervals are not exact and contiguous")
    result = {
        "schema_id": "cascadia-v3-expert-cycle-corpus-v1",
        "passed": True,
        "scientific_eligible": True,
        "cycle": cycle,
        "games": 10_000,
        "training_entries": 200_000,
        "focal_entries_only": True,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "collection_completion": {
            "path": str(collection.resolve()),
            "sha256": _digest(collection, "sha256"),
        },
        "verification_completion": {
            "path": str(verification.resolve()),
            "sha256": _digest(verification, "sha256"),
        },
        "protected_seed_values_opened": False,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = aggregate(
            cycle=args.cycle,
            collection=args.collection,
            verification=args.verification,
            accepted_root=args.accepted_root,
            campaign_state=args.campaign_state,
        )
    except (CycleCorpusError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    _write_atomic(args.output, value)
    print(json.dumps({"passed": True, "cycle": args.cycle, "games": 10_000}))


if __name__ == "__main__":
    main()
