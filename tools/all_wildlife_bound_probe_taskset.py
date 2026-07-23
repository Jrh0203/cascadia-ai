#!/usr/bin/env python3
"""Freeze explicit count-vector cases for bounded-maximization fleet probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules

SCHEMA = "all-wildlife-bound-probe-taskset-v1"
COUNT_VECTORS = frozenset(rules.count_vectors())


def parse_case(value: str) -> tuple[int, tuple[int, int, int, int, int]]:
    try:
        index_text, counts_text = value.split(":", 1)
        index = int(index_text)
        counts = tuple(int(field) for field in counts_text.split(","))
    except ValueError as error:
        raise ValueError(f"invalid case {value!r}") from error
    if index < 0 or index >= len(rules.rulesets()) or counts not in COUNT_VECTORS:
        raise ValueError(f"invalid case {value!r}")
    return index, counts  # type: ignore[return-value]


def _current_bounds(row: dict[str, Any], ruleset: str) -> dict[tuple[int, int, int, int, int], int]:
    unresolved = [tuple(counts) for counts in row.get("unresolved_counts", [])]
    if len(unresolved) != len(set(unresolved)) or any(
        counts not in COUNT_VECTORS for counts in unresolved
    ):
        raise ValueError(f"{ruleset}: invalid unresolved counts")
    analytical = {counts: rules.count_upper(counts, ruleset) for counts in unresolved}
    stored = row.get("unresolved_count_upper_bounds")
    if stored is None:
        bounds = analytical
    else:
        if not isinstance(stored, list) or len(stored) != len(unresolved):
            raise ValueError(f"{ruleset}: invalid unresolved count upper bounds")
        incumbent = int(row["optimum"])
        bounds = {}
        for counts, value in zip(unresolved, stored, strict=True):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= incumbent
                or value > analytical[counts]
            ):
                raise ValueError(f"{ruleset} {counts}: invalid current upper")
            bounds[counts] = value
    expected = max([int(row["optimum"]), *bounds.values()])
    if int(row.get("sound_upper", expected)) != expected:
        raise ValueError(f"{ruleset}: current bounds disagree with sound upper")
    return bounds


def build_taskset(
    catalog_path: Path,
    cases: list[str] | None = None,
    *,
    top_frontier_above: int | None = None,
) -> dict[str, Any]:
    encoded = catalog_path.read_bytes()
    catalog = json.loads(encoded)
    if catalog.get("schema") != "all-wildlife-optimal-catalog-v1" or len(
        catalog.get("results", [])
    ) != len(rules.rulesets()):
        raise ValueError("unexpected catalog schema or row count")
    if (cases is None) == (top_frontier_above is None):
        raise ValueError("select exactly one of explicit cases or top-frontier mode")
    if cases is not None:
        parsed = [parse_case(case) for case in cases]
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError("cases must be nonempty and unique")
        selection = {"mode": "explicit"}
    else:
        parsed = []
        for index, (ruleset, row) in enumerate(
            zip(rules.rulesets(), catalog["results"], strict=True)
        ):
            if row.get("index") != index or row.get("ruleset") != ruleset:
                raise ValueError(f"catalog identity mismatch at index {index}")
            bounds = _current_bounds(row, ruleset)
            frontier = max([int(row["optimum"]), *bounds.values()])
            if frontier <= int(top_frontier_above):
                continue
            parsed.extend((index, counts) for counts, upper in bounds.items() if upper == frontier)
        if not parsed:
            raise ValueError("top-frontier selection is empty")
        selection = {
            "mode": "top_frontier",
            "minimum_upper_exclusive": int(top_frontier_above),
        }
    tasks = []
    for task_index, (index, counts) in enumerate(parsed):
        row = catalog["results"][index]
        ruleset = rules.rulesets()[index]
        if row.get("index") != index or row.get("ruleset") != ruleset:
            raise ValueError(f"catalog identity mismatch at index {index}")
        bounds = _current_bounds(row, ruleset)
        if counts not in bounds:
            raise ValueError(f"{ruleset} {counts} is not currently unresolved")
        tasks.append(
            {
                "task_index": task_index,
                "ruleset_index": index,
                "ruleset": ruleset,
                "counts": list(counts),
                "incumbent": int(row["optimum"]),
                "analytical_upper": rules.count_upper(counts, ruleset),
                "current_upper": bounds[counts],
            }
        )
    return {
        "schema": SCHEMA,
        "catalog_path": str(catalog_path),
        "catalog_sha256": hashlib.sha256(encoded).hexdigest(),
        "selection": selection,
        "task_count": len(tasks),
        "tasks": tasks,
    }


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
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", action="append")
    selection.add_argument("--top-frontier-above", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_taskset(
        args.catalog,
        args.case,
        top_frontier_above=args.top_frontier_above,
    )
    _write_atomic(args.output, payload)
    print(f"tasks={payload['task_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
