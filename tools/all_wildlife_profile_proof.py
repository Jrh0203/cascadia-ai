#!/usr/bin/env python3
"""Solve fixed-score all-wildlife branches concurrently and durably."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules
from tools.all_wildlife_exact import solve_counts


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _solve(
    job: tuple[dict[str, Any], float],
) -> dict[str, Any]:
    task, seconds = job
    counts = tuple(task["counts"])
    profile = tuple(task["score_profile"])
    result = solve_counts(
        task["ruleset"],
        counts,  # type: ignore[arg-type]
        task["threshold"],
        time_limit_seconds=seconds,
        workers=1,
        enforce_connectivity=True,
        maximize=False,
        maximum_score=task["upper"],
        fixed_score_profile=profile,  # type: ignore[arg-type]
    )
    if result.status not in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"):
        raise RuntimeError(f"unexpected solver status: {result.status}")
    actual_breakdown = None
    if result.tokens is not None:
        actual_breakdown = rules.score_tokens(result.tokens, task["ruleset"])
        if sum(actual_breakdown) < task["threshold"]:
            raise AssertionError("fixed-profile witness is below threshold")
    return {
        "task_index": task["task_index"],
        "case_index": task["case_index"],
        "case_id": task["case_id"],
        "profile_index": task["profile_index"],
        "ruleset": task["ruleset"],
        "counts": task["counts"],
        "threshold": task["threshold"],
        "upper": task["upper"],
        "score_profile": task["score_profile"],
        "status": result.status,
        "elapsed_seconds": result.elapsed_seconds,
        "branches": result.branches,
        "conflicts": result.conflicts,
        "objective": result.objective,
        "independent_score_breakdown": list(actual_breakdown)
        if actual_breakdown
        else None,
        "tokens": result.tokens,
    }


def run(args: argparse.Namespace) -> int:
    encoded = args.taskset.read_bytes()
    taskset_sha = hashlib.sha256(encoded).hexdigest()
    taskset = json.loads(encoded)
    if taskset.get("schema") != "all-wildlife-score-profile-taskset-v1":
        raise ValueError("unexpected taskset schema")
    rules_sha = hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest()
    if taskset["rules_source_sha256"] != rules_sha:
        raise ValueError("taskset rules-source mismatch")
    indices = [int(value) for value in args.indices.split(",")]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate task indices")
    by_index = {task["task_index"]: task for task in taskset["tasks"]}
    if any(index not in by_index for index in indices):
        raise ValueError("task index out of range")

    started = time.monotonic()
    jobs = [(by_index[index], args.seconds) for index in indices]
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(args.jobs, len(jobs))) as pool:
        results = list(pool.imap_unordered(_solve, jobs, chunksize=1))
    results.sort(key=lambda row: row["task_index"])
    payload = {
        "schema": "all-wildlife-score-profile-shard-v1",
        "identity": {
            "taskset_sha256": taskset_sha,
            "rules_source_sha256": rules_sha,
            "exact_source_sha256": hashlib.sha256(
                Path("tools/all_wildlife_exact.py").read_bytes()
            ).hexdigest(),
            "exact_support_source_sha256": hashlib.sha256(
                Path("tools/cbddb_wildlife_exact.py").read_bytes()
            ).hexdigest(),
            "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "configuration": {
            "seconds_per_profile": args.seconds,
            "jobs": args.jobs,
            "solver_workers": 1,
            "connectivity_required": True,
            "seed": 20260723,
        },
        "task_indices": indices,
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "tasks": len(results),
                "exact": sum(row["status"] != "UNKNOWN" for row in results),
                "unknown": sum(row["status"] == "UNKNOWN" for row in results),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
