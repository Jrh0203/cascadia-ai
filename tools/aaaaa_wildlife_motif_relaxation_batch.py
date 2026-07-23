#!/usr/bin/env python3
"""Run a durable batch of AAAAA motif-coordinate relaxation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_motif_relaxation as relaxation


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_case(value: str) -> tuple[tuple[int, int, int, int, int], int]:
    try:
        counts_text, target_text = value.split(":", 1)
        counts = tuple(int(item) for item in counts_text.split(","))
        target = int(target_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be B,E,S,H,F:TARGET") from error
    if len(counts) != len(base.SPECIES):
        raise argparse.ArgumentTypeError("case must contain five counts")
    if sum(counts) != base.TOKEN_COUNT or any(not 0 <= count <= base.COUNT_CAP for count in counts):
        raise argparse.ArgumentTypeError(f"invalid count vector: {counts}")
    if target > base.count_relaxation(counts):
        raise argparse.ArgumentTypeError(f"target exceeds standalone bound: {value}")
    return counts, target


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    rows = []
    for index, (counts, target) in enumerate(args.case):
        result = relaxation.solve_relaxation(
            counts,
            target,
            workers=args.workers,
            time_limit=args.time_limit,
            seed=args.seed + index,
        )
        rows.append(result)
        print(
            f"case={index + 1}/{len(args.case)} counts={counts} target={target} "
            f"status={result['status']} wall={result['wall_seconds']:.6f}s",
            flush=True,
        )
    source = Path(__file__).resolve()
    relaxation_source = Path(relaxation.__file__).resolve()
    return {
        "schema": "aaaaa-motif-coordinate-relaxation-batch-v1",
        "proof_complete": all(row["proof_complete"] for row in rows),
        "case_count": len(rows),
        "configuration": {
            "workers_per_case": args.workers,
            "time_limit_seconds_per_case": args.time_limit,
            "base_seed": args.seed,
            "cases_sequential": True,
        },
        "relaxation": {
            "whole_board_connectivity_required": False,
            "bear_pair_isolation_required": False,
            "salmon_component_separation_required": False,
            "hawk_isolation_required": False,
            "token_nonoverlap_exact": True,
            "forced_scoring_motifs_exact": True,
            "fox_positive_observations_exact": True,
        },
        "source_sha256": file_sha256(source),
        "relaxation_source_sha256": file_sha256(relaxation_source),
        "results": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.case)) != len(args.case):
        raise SystemExit("duplicate cases are not allowed")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    summary = {
        key: payload[key] for key in ("proof_complete", "case_count", "elapsed_seconds")
    }
    print(json.dumps(summary))
    return 0 if payload["proof_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
