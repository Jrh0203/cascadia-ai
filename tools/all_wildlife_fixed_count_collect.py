#!/usr/bin/env python3
"""Validate and summarize incremental fixed-count wildlife candidate chunks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules

SCHEMA = "all-wildlife-fixed-count-candidates-v1"
SUMMARY_SCHEMA = "all-wildlife-fixed-count-summary-v1"
COUNT_VECTORS = rules.count_vectors()
RULESETS = rules.rulesets()
TOTAL_CELLS = len(COUNT_VECTORS) * len(RULESETS)


def _paths(directories: list[Path]) -> list[Path]:
    return sorted(
        path
        for directory in directories
        for path in directory.glob("chunk_*.json")
    )


def _chunk_index(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("chunk_"))
    except ValueError as error:
        raise ValueError(f"{path}: invalid chunk filename") from error


def _tokens(candidate: dict[str, Any], cell_index: int) -> list[dict[str, Any]]:
    encoded = candidate.get("tokens")
    if not isinstance(encoded, list) or len(encoded) != rules.TOKEN_COUNT:
        raise ValueError(f"cell {cell_index}: invalid token count")
    tokens = []
    for token in encoded:
        if (
            not isinstance(token, list)
            or len(token) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in token)
        ):
            raise ValueError(f"cell {cell_index}: invalid compact token")
        q, r, species = token
        if species < 0 or species >= len(rules.SPECIES):
            raise ValueError(f"cell {cell_index}: invalid wildlife index")
        tokens.append({"q": q, "r": r, "wildlife": rules.SPECIES[species]})
    return tokens


def _validate_candidate(
    candidate: dict[str, Any],
    cell_index: int,
    *,
    deep: bool,
) -> tuple[str, int, bool]:
    ruleset_index, count_index = divmod(cell_index, len(COUNT_VECTORS))
    ruleset = RULESETS[ruleset_index]
    counts = COUNT_VECTORS[count_index]
    score = int(candidate.get("score", -1))
    stored_upper = candidate.get("count_upper")
    canonical_upper = rules.count_upper(counts, ruleset)
    breakdown = candidate.get("score_breakdown")
    if (
        candidate.get("cell_index") != cell_index
        or candidate.get("ruleset_index") != ruleset_index
        or candidate.get("count_index") != count_index
        or candidate.get("ruleset") != ruleset
        or tuple(candidate.get("counts", ())) != counts
        or isinstance(stored_upper, bool)
        or not isinstance(stored_upper, int)
        or stored_upper < canonical_upper
        or stored_upper < score
        or not isinstance(breakdown, list)
        or len(breakdown) != len(rules.SPECIES)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in breakdown)
        or sum(breakdown) != score
        or candidate.get("upper_bound_matched") != (score == stored_upper)
        or not isinstance(candidate.get("states_evaluated"), int)
        or candidate["states_evaluated"] < 1
    ):
        raise ValueError(f"cell {cell_index}: candidate metadata mismatch")

    tokens = _tokens(candidate, cell_index)
    occupied = {(token["q"], token["r"]) for token in tokens}
    observed = Counter(token["wildlife"] for token in tokens)
    if (
        len(occupied) != rules.TOKEN_COUNT
        or len(rules.components(occupied)) != 1
        or tuple(observed[species] for species in rules.SPECIES) != counts
    ):
        raise ValueError(f"cell {cell_index}: invalid connected board")
    if deep and tuple(breakdown) != rules.score_tokens(tokens, ruleset):
        raise ValueError(f"cell {cell_index}: independent score mismatch")
    return ruleset, score, score == canonical_upper


def collect(
    directories: list[Path],
    *,
    chunk_size: int = 256,
    deep: bool = False,
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    total_chunks = (TOTAL_CELLS + chunk_size - 1) // chunk_size
    seen_chunks: set[int] = set()
    seen_cells: set[int] = set()
    configuration: tuple[int, int, int] | None = None
    best_score = -1
    best_cells: list[int] = []
    upper_matches = 0
    elapsed_seconds = 0.0

    for path in _paths(directories):
        chunk_index = _chunk_index(path)
        if chunk_index < 0 or chunk_index >= total_chunks or chunk_index in seen_chunks:
            raise ValueError(f"{path}: duplicate or out-of-range chunk")
        payload = json.loads(path.read_bytes())
        start = chunk_index * chunk_size
        end = min(start + chunk_size, TOTAL_CELLS)
        current_configuration = (
            int(payload.get("seed", -1)),
            int(payload.get("restarts_per_cell", -1)),
            int(payload.get("iterations_per_restart", -1)),
        )
        if configuration is None:
            configuration = current_configuration
        if (
            payload.get("schema") != SCHEMA
            or payload.get("token_count") != rules.TOKEN_COUNT
            or payload.get("count_cap") != rules.COUNT_CAP
            or payload.get("total_cells") != TOTAL_CELLS
            or payload.get("range_start") != start
            or payload.get("range_end") != end
            or current_configuration != configuration
            or len(payload.get("candidates", ())) != end - start
        ):
            raise ValueError(f"{path}: chunk header/configuration mismatch")
        elapsed_seconds += float(payload.get("elapsed_seconds", 0.0))
        for offset, candidate in enumerate(payload["candidates"]):
            cell_index = start + offset
            if cell_index in seen_cells:
                raise ValueError(f"{path}: duplicate cell {cell_index}")
            _, score, matched = _validate_candidate(
                candidate,
                cell_index,
                deep=deep,
            )
            seen_cells.add(cell_index)
            upper_matches += int(matched)
            if score > best_score:
                best_score = score
                best_cells = [cell_index]
            elif score == best_score:
                best_cells.append(cell_index)
        seen_chunks.add(chunk_index)

    missing_chunks = sorted(set(range(total_chunks)) - seen_chunks)
    seed, restarts, iterations = configuration or (-1, -1, -1)
    return {
        "schema": SUMMARY_SCHEMA,
        "complete": len(seen_cells) == TOTAL_CELLS,
        "deep_validation": deep,
        "ruleset_count": len(RULESETS),
        "count_vector_count": len(COUNT_VECTORS),
        "total_cells": TOTAL_CELLS,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "completed_chunks": len(seen_chunks),
        "completed_cells": len(seen_cells),
        "completion_fraction": len(seen_cells) / TOTAL_CELLS,
        "missing_chunk_count": len(missing_chunks),
        "first_missing_chunks": missing_chunks[:100],
        "seed": seed,
        "restarts_per_cell": restarts,
        "iterations_per_restart": iterations,
        "canonical_upper_bound_matches": upper_matches,
        "best_score": None if best_score < 0 else best_score,
        "best_cells": best_cells,
        "summed_chunk_elapsed_seconds": elapsed_seconds,
        "directories": [str(directory) for directory in directories],
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = collect(
        args.directories,
        chunk_size=args.chunk_size,
        deep=args.deep,
    )
    if args.output:
        _write_atomic(args.output, summary)
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "completed_cells": summary["completed_cells"],
                "total_cells": summary["total_cells"],
                "completion_fraction": summary["completion_fraction"],
                "best_score": summary["best_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
