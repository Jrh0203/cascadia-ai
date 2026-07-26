#!/usr/bin/env python3
"""Materialize the best fixed-count wildlife board across complete search stages."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools.all_wildlife_fixed_count_collect import TOTAL_CELLS, collect

SCHEMA = "all-wildlife-fixed-count-best-v1"


def _load_chunk(directory: Path, chunk_index: int) -> dict[str, Any]:
    path = directory / f"chunk_{chunk_index:05d}.json"
    if not path.is_file():
        raise ValueError(f"{path}: required stage chunk is missing")
    return json.loads(path.read_bytes())


def _winner(
    candidates: list[tuple[int, str, dict[str, Any]]],
) -> tuple[int, str, dict[str, Any]]:
    return min(
        candidates,
        key=lambda item: (
            -int(item[2]["score"]),
            json.dumps(item[2]["tokens"], separators=(",", ":")),
            item[0],
        ),
    )


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def merge(
    stages: list[tuple[str, Path]],
    output_directory: Path,
    *,
    chunk_size: int = 256,
    deep: bool = False,
) -> dict[str, Any]:
    if len(stages) < 2 or len({name for name, _ in stages}) != len(stages):
        raise ValueError("provide at least two uniquely named stages")
    summaries = []
    for name, directory in stages:
        summary = collect([directory], chunk_size=chunk_size, deep=deep)
        if not summary["complete"]:
            raise ValueError(
                f"{name}: stage is incomplete "
                f"({summary['completed_cells']}/{summary['total_cells']})"
            )
        summaries.append((name, directory, summary))

    total_chunks = (TOTAL_CELLS + chunk_size - 1) // chunk_size
    selected = {name: 0 for name, _ in stages}
    strict_improvements = {name: 0 for name, _ in stages}
    best_score = -1
    best_cells: list[int] = []
    output_directory.mkdir(parents=True, exist_ok=True)

    for chunk_index in range(total_chunks):
        payloads = [
            (stage_index, name, _load_chunk(directory, chunk_index))
            for stage_index, (name, directory) in enumerate(stages)
        ]
        start = chunk_index * chunk_size
        end = min(start + chunk_size, TOTAL_CELLS)
        if any(
            payload.get("range_start") != start
            or payload.get("range_end") != end
            or len(payload.get("candidates", ())) != end - start
            for _, _, payload in payloads
        ):
            raise ValueError(f"chunk {chunk_index}: stage range mismatch")

        merged = []
        for offset, cell_index in enumerate(range(start, end)):
            options = [
                (stage_index, name, payload["candidates"][offset])
                for stage_index, name, payload in payloads
            ]
            if any(option[2].get("cell_index") != cell_index for option in options):
                raise ValueError(f"chunk {chunk_index}: cell identity mismatch")
            stage_index, stage_name, candidate = _winner(options)
            winner = dict(candidate)
            winner["source_stage"] = stage_name
            merged.append(winner)
            selected[stage_name] += 1
            if all(
                int(candidate["score"]) > int(other["score"])
                for other_index, _, other in options
                if other_index != stage_index
            ):
                strict_improvements[stage_name] += 1
            score = int(candidate["score"])
            if score > best_score:
                best_score = score
                best_cells = [cell_index]
            elif score == best_score:
                best_cells.append(cell_index)

        _write_atomic(
            output_directory / f"chunk_{chunk_index:05d}.json",
            {
                "schema": SCHEMA,
                "token_count": 20,
                "count_cap": 6,
                "total_cells": TOTAL_CELLS,
                "range_start": start,
                "range_end": end,
                "source_stages": [name for name, _ in stages],
                "candidates": merged,
            },
        )

    summary = {
        "schema": "all-wildlife-fixed-count-best-summary-v1",
        "complete": True,
        "deep_source_validation": deep,
        "total_cells": TOTAL_CELLS,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "stages": [
            {
                "name": name,
                "directory": str(directory),
                "restarts_per_cell": stage_summary["restarts_per_cell"],
                "iterations_per_restart": stage_summary["iterations_per_restart"],
            }
            for name, directory, stage_summary in summaries
        ],
        "selected_cells_by_stage": selected,
        "strict_improvements_by_stage": strict_improvements,
        "best_score": best_score,
        "best_cells": best_cells,
        "output_directory": str(output_directory),
    }
    _write_atomic(output_directory / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        nargs=2,
        action="append",
        metavar=("NAME", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    summary = merge(
        [(name, Path(directory)) for name, directory in args.stage],
        args.output_directory,
        chunk_size=args.chunk_size,
        deep=args.deep,
    )
    print(
        json.dumps(
            {
                "total_cells": summary["total_cells"],
                "best_score": summary["best_score"],
                "selected_cells_by_stage": summary["selected_cells_by_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
