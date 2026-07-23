#!/usr/bin/env python3
"""Freeze unresolved wildlife count vectors into a fleet taskset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.wildlife_catalog_sharding import SCHEMA


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_taskset(scoring_cards: str, catalog_path: Path) -> dict[str, Any]:
    if scoring_cards == "AAAAA":
        from tools import aaaaa_wildlife_catalog as catalog
        from tools.aaaaa_wildlife_exact import count_vectors

        accepted_schemas = {catalog.SCHEMA, catalog.LEGACY_SCHEMA}
    elif scoring_cards == "CBDDB":
        from tools import cbddb_wildlife_catalog as catalog
        from tools.cbddb_wildlife_exact import count_vectors

        accepted_schemas = {catalog.SCHEMA}
    else:
        raise ValueError(f"unsupported scoring cards: {scoring_cards}")

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if payload.get("schema") not in accepted_schemas:
        raise ValueError(f"unsupported catalog schema in {catalog_path}")
    canonical = [counts for counts, _ in count_vectors()]
    canonical_set = set(canonical)
    completed: set[tuple[int, int, int, int, int]] = set()
    observed: set[tuple[int, int, int, int, int]] = set()
    for result in payload.get("results", []):
        counts = tuple(int(value) for value in result["counts"])
        if counts in observed:
            raise ValueError(f"catalog contains duplicate counts {counts}")
        observed.add(counts)
        if counts not in canonical_set:
            raise ValueError(f"catalog contains noncanonical counts {counts}")
        tokens, breakdown = catalog.validate_witness(counts, result["tokens"])
        if len(tokens) != 20 or sum(breakdown) != int(result["optimum"]):
            raise ValueError(f"catalog witness mismatch for {counts}")
        if result.get("proof_complete"):
            completed.add(counts)
    unresolved = [list(counts) for counts in canonical if counts not in completed]
    source = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "scoring_cards": scoring_cards,
        "task_count": len(unresolved),
        "completed_count": len(completed),
        "allocation_count": len(canonical),
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": file_sha256(catalog_path),
        "source_sha256": file_sha256(source),
        "counts": unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-cards", choices=("AAAAA", "CBDDB"), required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_taskset(args.scoring_cards, args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"taskset={args.scoring_cards} unresolved={payload['task_count']} "
        f"completed={payload['completed_count']}/{payload['allocation_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
