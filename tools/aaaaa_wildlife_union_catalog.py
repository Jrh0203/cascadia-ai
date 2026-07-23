#!/usr/bin/env python3
"""Union validated AAAAA catalog ledgers by canonical count vector."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import aaaaa_wildlife_catalog as catalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def union(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one catalog is required")
    canonical = [counts for counts, _ in catalog.count_vectors()]
    by_counts: dict[tuple[int, ...], list[tuple[Path, dict[str, Any]]]] = {}
    source_hashes = {}
    token_count = None
    count_cap = None
    for path in paths:
        payload = json.loads(path.read_bytes())
        if payload.get("schema") != catalog.SCHEMA:
            raise ValueError(f"{path}: unexpected catalog schema")
        if payload.get("allocation_count") != len(canonical):
            raise ValueError(f"{path}: allocation count mismatch")
        identity = (payload.get("token_count"), payload.get("count_cap"))
        if token_count is None:
            token_count, count_cap = identity
        elif identity != (token_count, count_cap):
            raise ValueError(f"{path}: token/count-cap mismatch")
        seen = set()
        for row in payload.get("results", []):
            counts = tuple(int(value) for value in row["counts"])
            if counts in seen:
                raise ValueError(f"{path}: duplicate count vector {counts}")
            seen.add(counts)
            if counts not in set(canonical):
                raise ValueError(f"{path}: noncanonical count vector {counts}")
            tokens, breakdown = catalog.validate_witness(counts, row["tokens"])
            if (
                breakdown != row["score_breakdown"]
                or sum(breakdown) != row["optimum"]
            ):
                raise ValueError(f"{path}: witness mismatch for {counts}")
            normalized = copy.deepcopy(row)
            normalized["tokens"] = tokens
            by_counts.setdefault(counts, []).append((path, normalized))
        source_hashes[str(path)] = _sha256(path)
    missing = set(canonical) - set(by_counts)
    if missing:
        raise ValueError(f"catalog union is missing {len(missing)} count vectors")

    results = []
    conflicts = []
    for counts in canonical:
        choices = by_counts[counts]
        exact = [(path, row) for path, row in choices if row.get("proof_complete")]
        if exact:
            optima = {row["optimum"] for _, row in exact}
            if len(optima) != 1:
                conflicts.append({"counts": list(counts), "optima": sorted(optima)})
                continue
            chosen_path, chosen = min(exact, key=lambda item: str(item[0]))
        else:
            chosen_path, chosen = max(
                choices,
                key=lambda item: (item[1]["optimum"], str(item[0])),
            )
        chosen = copy.deepcopy(chosen)
        chosen["union_source_path"] = str(chosen_path)
        chosen["union_source_sha256"] = source_hashes[str(chosen_path)]
        results.append(chosen)
    if conflicts:
        raise ValueError(f"conflicting exact optima: {conflicts[:5]}")

    return {
        "schema": catalog.SCHEMA,
        "scoring_cards": "AAAAA",
        "token_count": token_count,
        "count_cap": count_cap,
        "allocation_count": len(canonical),
        "completed_count": sum(row.get("proof_complete", False) for row in results),
        "proof_complete": all(row.get("proof_complete", False) for row in results),
        "union_sources": source_hashes,
        "external_certificates": [],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = union(args.catalog)
    _write_atomic(args.output, json.dumps(payload, indent=2) + "\n")
    _write_atomic(args.markdown, catalog.render_markdown(payload) + "\n")
    print(
        json.dumps(
            {
                "allocations": payload["allocation_count"],
                "completed": payload["completed_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
