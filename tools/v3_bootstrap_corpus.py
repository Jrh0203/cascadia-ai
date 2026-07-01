#!/usr/bin/env python3
"""Prove the V3 bootstrap corpus complete, disjoint, and replay-verified."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3

EXPECTED_COMPONENTS = {
    "greedy": 100_000,
    "v1-direct": 200_000,
    "mixed-frozen": 100_000,
    "rare-softmax": 100_000,
}


class CorpusError(ValueError):
    """The imported bootstrap corpus is incomplete or scientifically invalid."""


def _blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def expected_items(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        plan.get("schema_id") != "cascadia-v3-bacalhau-collection-plan-v1"
        or plan.get("phase") != "bootstrap_collecting"
        or plan.get("games") != 500_000
        or plan.get("work_items") != 250
        or plan.get("scheduler_owns_placement") is not True
        or plan.get("manual_host_sharding") is not False
    ):
        raise CorpusError("bootstrap collection plan identity or topology is invalid")
    result = {}
    for item in plan.get("items", []):
        key = item.get("key")
        metadata = item.get("application_metadata", {})
        if not isinstance(key, str) or key in result:
            raise CorpusError("collection plan item keys are invalid")
        try:
            result[key] = {
                "component": str(metadata["component"]),
                "games": int(metadata["games"]),
                "first_game_index": int(metadata["first_game_index"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise CorpusError(f"collection plan metadata is invalid for {key}") from error
    if len(result) != 250:
        raise CorpusError("bootstrap plan does not contain exactly 250 items")
    return result


def _validate_intervals(items: list[dict[str, Any]]) -> None:
    intervals = sorted(
        (
            int(item["first_game_index"]),
            int(item["first_game_index"]) + int(item["games"]),
            str(item["key"]),
        )
        for item in items
    )
    for left, right in pairwise(intervals):
        if left[1] > right[0]:
            raise CorpusError(f"game-index domains overlap: {left[2]} and {right[2]}")


def aggregate(
    *,
    plan_path: Path,
    accepted_root: Path,
    verification_path: Path,
    readiness_sha256: str,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    expected = expected_items(plan)
    verification = json.loads(verification_path.read_text())
    totals = verification.get("totals", {})
    if (
        verification.get("schema_id") != "cascadia-v3-cluster-stage-completion-v1"
        or verification.get("passed") is not True
        or verification.get("work_items") != 250
        or verification.get("succeeded") != 250
        or totals.get("records") != 500_000
        or totals.get("expanded_training_entries") != 40_000_000
    ):
        raise CorpusError("distributed replay verification is incomplete")
    verification_inputs = verification.get("inputs")
    if not isinstance(verification_inputs, list) or len(verification_inputs) != 250:
        raise CorpusError("distributed verification omits its immutable input identities")
    imported = [
        path
        for path in accepted_root.iterdir()
        if path.is_dir() and path.name != ".receipts"
    ]
    if {path.name for path in imported} != set(expected):
        missing = sorted(set(expected) - {path.name for path in imported})
        extra = sorted({path.name for path in imported} - set(expected))
        raise CorpusError(f"imported item set differs; missing={missing}, extra={extra}")
    files = []
    component_games: Counter[str] = Counter()
    policy_seat_games: Counter[str] = Counter()
    intervals = []
    image = plan["image"]
    provenance_root = accepted_root / ".receipts"
    for key, expected_item in sorted(expected.items()):
        directory = accepted_root / key
        shards = sorted(directory.glob("*.v3g"))
        receipts = sorted(directory.glob("*.receipt.json"))
        manifests = sorted(directory.glob("manifest.json"))
        provenance_path = provenance_root / f"{key}.json"
        if (
            len(shards) != 1
            or len(receipts) != 1
            or len(manifests) != 1
            or not provenance_path.is_file()
        ):
            raise CorpusError(f"artifact set is incomplete for {key}")
        shard = shards[0]
        receipt = json.loads(receipts[0].read_text())
        provenance = json.loads(provenance_path.read_text())
        if (
            receipt.get("schema_id") != "cascadia-v3-collection-shard-receipt-v1"
            or receipt.get("scientific_eligible") is not True
            or receipt.get("approved_readiness_sha256") != readiness_sha256
            or receipt.get("component") != expected_item["component"]
            or receipt.get("games") != expected_item["games"]
            or receipt.get("records") != expected_item["games"]
            or receipt.get("first_game_index") != expected_item["first_game_index"]
            or receipt.get("bytes") != shard.stat().st_size
            or receipt.get("blake3") != _blake3(shard)
            or receipt.get("seed_domain") != "scheduler-assigned-game-index-v1"
        ):
            raise CorpusError(f"scientific shard receipt differs for {key}")
        seat_counts = receipt.get("policy_seat_games")
        if not isinstance(seat_counts, dict) or sum(seat_counts.values()) != receipt["games"] * 4:
            raise CorpusError(f"policy-seat accounting differs for {key}")
        if (
            provenance.get("schema_id") != "cascadia.cluster.accepted-result.v1"
            or provenance.get("request_id") != accepted_root.name
            or provenance.get("item_id") != key
            or provenance.get("image_digest") != image
            or not 1 <= provenance.get("attempts", 0) <= 3
        ):
            raise CorpusError(f"scheduler provenance differs for {key}")
        component_games[receipt["component"]] += receipt["games"]
        policy_seat_games.update(seat_counts)
        interval = {
            "key": key,
            "first_game_index": receipt["first_game_index"],
            "games": receipt["games"],
        }
        intervals.append(interval)
        files.append(
            {
                "key": key,
                "component": receipt["component"],
                "first_game_index": receipt["first_game_index"],
                "games": receipt["games"],
                "path": str(shard.resolve()),
                "bytes": shard.stat().st_size,
                "blake3": receipt["blake3"],
                "sha256": _sha256(shard),
                "attempts": provenance["attempts"],
                "accepted_execution_id": provenance["accepted_execution_id"],
            }
        )
    _validate_intervals(intervals)
    if dict(component_games) != EXPECTED_COMPONENTS:
        raise CorpusError(f"component totals differ: {dict(component_games)}")
    try:
        verified_sources = {
            (
                item.get("source_shard"),
                item.get("source_sha256"),
                int(item.get("source_bytes", -1)),
            )
            for item in verification_inputs
            if isinstance(item, dict)
        }
    except (TypeError, ValueError) as error:
        raise CorpusError("verification input identities are malformed") from error
    imported_sources = {
        (Path(item["path"]).name, item["sha256"], item["bytes"])
        for item in files
    }
    if verified_sources != imported_sources:
        raise CorpusError("verification inputs differ from the imported replay corpus")
    result = {
        "schema_id": "cascadia-v3-bootstrap-corpus-v1",
        "passed": True,
        "scientific_eligible": True,
        "approved_readiness_sha256": readiness_sha256,
        "collection_image_digest": image,
        "collection_plan": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256(plan_path),
        },
        "verification": {
            "path": str(verification_path.resolve()),
            "sha256": _sha256(verification_path),
            "request_id": verification["request_id"],
        },
        "work_items": 250,
        "games": 500_000,
        "expanded_training_entries": 40_000_000,
        "component_games": dict(sorted(component_games.items())),
        "policy_seat_games": dict(sorted(policy_seat_games.items())),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "protected_seed_values_opened": False,
    }
    result["canonical_sha256"] = hashlib.sha256(_canonical_without_hash(result)).hexdigest()
    return result


def _canonical_without_hash(value: dict[str, Any]) -> bytes:
    copy = dict(value)
    copy.pop("canonical_sha256", None)
    return json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--approved-readiness-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = aggregate(
            plan_path=args.plan,
            accepted_root=args.accepted_root,
            verification_path=args.verification,
            readiness_sha256=args.approved_readiness_sha256,
        )
    except (CorpusError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "games", "total_bytes")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
