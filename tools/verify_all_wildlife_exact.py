#!/usr/bin/env python3
"""Exhaustively validate fixed-board exact certificates for all 1,024 rulesets."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from typing import Any

from tools import all_wildlife_rules
from tools.all_wildlife_exact import solve_counts
from tools.test_all_wildlife_rules import random_connected_board


def _check(task: tuple[str, list[dict[str, Any]]]) -> tuple[str, int, int, str]:
    ruleset, board = task
    counts = tuple(
        sum(row["wildlife"] == species for row in board)
        for species in all_wildlife_rules.SPECIES
    )
    expected = sum(all_wildlife_rules.score_tokens(board, ruleset))
    result = solve_counts(
        ruleset,
        counts,  # type: ignore[arg-type]
        0,
        time_limit_seconds=30,
        workers=1,
        initial_tokens=board,
        fix_initial_tokens=True,
    )
    return ruleset, result.objective or 0, expected, result.status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    started = time.monotonic()
    board = random_connected_board(args.seed)
    tasks = [(ruleset, board) for ruleset in all_wildlife_rules.rulesets()]
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(_check, tasks, chunksize=1))
    rows.sort()
    failures = [
        {
            "ruleset": ruleset,
            "model": model_score,
            "expected": expected,
            "status": status,
        }
        for ruleset, model_score, expected, status in rows
        if status != "OPTIMAL" or model_score != expected
    ]
    canonical = json.dumps(rows, separators=(",", ":"))
    summary = {
        "schema": "all-wildlife-exact-fixed-board-verification-v1",
        "seed": args.seed,
        "workers": args.workers,
        "cases": len(rows),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
