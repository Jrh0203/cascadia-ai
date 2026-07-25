#!/usr/bin/env python3
"""Freeze a branch-count slice of the all-wildlife proof tail."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules

TASKSET_SCHEMA = "all-wildlife-near-complete-taskset-v1"
SPECIES_INDEX = {species: index for index, species in enumerate(rules.SPECIES)}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_candidate(
    row: dict[str, Any],
    *,
    index: int,
    ruleset: str,
) -> None:
    if row.get("index") != index or row.get("ruleset") != ruleset:
        raise ValueError(f"candidate identity mismatch at index {index}")
    breakdown = list(rules.score_tokens(row["tokens"], ruleset))
    if breakdown != row.get("score_breakdown") or sum(breakdown) != row.get("score"):
        raise ValueError(f"candidate score mismatch at index {index}")
    counts = [0] * 5
    for token in row["tokens"]:
        counts[SPECIES_INDEX[token["wildlife"]]] += 1
    if counts != row.get("counts"):
        raise ValueError(f"candidate count mismatch at index {index}")
    occupied = {
        (int(token["q"]), int(token["r"]))
        for token in row["tokens"]
    }
    if len(occupied) != rules.TOKEN_COUNT or len(rules.components(occupied)) != 1:
        raise ValueError(f"candidate board is disconnected at index {index}")


def build_taskset(
    catalog_path: Path,
    candidate_path: Path,
    *,
    minimum_branches: int,
    maximum_branches: int,
    comparison_candidate_path: Path | None = None,
) -> dict[str, Any]:
    if minimum_branches < 1 or maximum_branches < minimum_branches:
        raise ValueError("branch limits must satisfy 1 <= minimum <= maximum")

    catalog = _read(catalog_path)
    candidate = _read(candidate_path)
    if catalog.get("schema") not in {
        "all-wildlife-optimal-catalog-v1",
        "all-wildlife-optimal-catalog-v2",
    }:
        raise ValueError("unexpected catalog schema")
    if candidate.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected candidate schema")

    results = catalog.get("results", [])
    candidates = candidate.get("candidates", [])
    if len(results) != len(candidates):
        raise ValueError("catalog/candidate row-count mismatch")

    comparison = None
    if comparison_candidate_path is not None:
        comparison = _read(comparison_candidate_path)
        if comparison.get("schema") != "all-wildlife-merged-candidates-v1":
            raise ValueError("unexpected comparison-candidate schema")
        if len(comparison.get("candidates", [])) != len(candidates):
            raise ValueError("comparison-candidate row-count mismatch")

    tasks: list[dict[str, Any]] = []
    expected_rulesets = rules.rulesets()
    for index, result in enumerate(results):
        unresolved = result.get("unresolved_counts", [])
        branch_count = len(unresolved)
        if not minimum_branches <= branch_count <= maximum_branches:
            continue
        ruleset = expected_rulesets[index]
        if result.get("index") != index or result.get("ruleset") != ruleset:
            raise ValueError(f"catalog identity mismatch at index {index}")
        if result.get("proof_complete"):
            raise ValueError(f"complete row has unresolved branches at index {index}")
        incumbent = int(result["optimum"])
        base_row = candidates[index]
        _validate_candidate(base_row, index=index, ruleset=ruleset)
        incumbent = max(incumbent, int(base_row["score"]))
        if comparison is not None:
            comparison_row = comparison["candidates"][index]
            _validate_candidate(comparison_row, index=index, ruleset=ruleset)
            incumbent = max(incumbent, int(comparison_row["score"]))
        unresolved = [
            counts
            for counts in unresolved
            if rules.count_upper(tuple(counts), ruleset) > incumbent
        ]
        for counts in unresolved:
            if len(counts) != 5 or sum(counts) != 20 or max(counts) > 6:
                raise ValueError(f"invalid count vector at index {index}: {counts}")
        tasks.append(
            {
                "index": index,
                "ruleset": ruleset,
                "incumbent": incumbent,
                "threshold": incumbent + 1,
                "counts": unresolved,
            }
        )

    payload: dict[str, Any] = {
        "schema": TASKSET_SCHEMA,
        "source_catalog": str(catalog_path),
        "candidate_catalog": str(candidate_path),
        "minimum_unresolved_branches": minimum_branches,
        "maximum_unresolved_branches": maximum_branches,
        "selection_rule": (
            "Select every incomplete ruleset whose current sound-bound unresolved "
            f"branch count is in [{minimum_branches}, {maximum_branches}]."
        ),
        "task_count": len(tasks),
        "count_query_count": sum(len(task["counts"]) for task in tasks),
        "tasks": tasks,
    }
    if comparison_candidate_path is not None:
        payload["comparison_candidate_catalog"] = str(comparison_candidate_path)
        payload["selection_rule"] += " Stronger comparison incumbents are used when present."
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--comparison-candidates", type=Path)
    parser.add_argument("--min-branches", type=int, required=True)
    parser.add_argument("--max-branches", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_taskset(
        args.catalog,
        args.candidates,
        minimum_branches=args.min_branches,
        maximum_branches=args.max_branches,
        comparison_candidate_path=args.comparison_candidates,
    )
    _write_atomic(args.output, payload)
    print(
        f"taskset={payload['task_count']} "
        f"count_queries={payload['count_query_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
