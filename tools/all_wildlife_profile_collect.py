#!/usr/bin/env python3
"""Combine any available all-wildlife score-profile shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules

EXACT_STATUSES = frozenset({"OPTIMAL", "FEASIBLE", "INFEASIBLE"})
ALLOWED_STATUSES = EXACT_STATUSES | {"UNKNOWN"}
TASK_FIELDS = (
    "task_index",
    "case_index",
    "case_id",
    "profile_index",
    "ruleset",
    "counts",
    "threshold",
    "upper",
    "score_profile",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _validate_witness(result: dict[str, Any], task: dict[str, Any]) -> None:
    tokens = rules.normalized_tokens(result["tokens"])
    counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if counts != tuple(task["counts"]):
        raise ValueError(f"task {task['task_index']}: witness count mismatch")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(rules.components(occupied)) != 1:
        raise ValueError(f"task {task['task_index']}: disconnected witness")
    breakdown = rules.score_tokens(tokens, task["ruleset"])
    if list(breakdown) != result["independent_score_breakdown"]:
        raise ValueError(f"task {task['task_index']}: witness score mismatch")
    if list(breakdown) != task["score_profile"]:
        raise ValueError(f"task {task['task_index']}: witness/profile mismatch")
    if sum(breakdown) < task["threshold"]:
        raise ValueError(f"task {task['task_index']}: witness below threshold")


def _validate_result(result: dict[str, Any], task: dict[str, Any]) -> None:
    for field in TASK_FIELDS:
        if result.get(field) != task[field]:
            raise ValueError(f"task {task['task_index']}: {field} mismatch")
    status = result.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"task {task['task_index']}: invalid status {status!r}")
    if status in {"OPTIMAL", "FEASIBLE"}:
        if result.get("tokens") is None:
            raise ValueError(f"task {task['task_index']}: feasible result has no witness")
        _validate_witness(result, task)


def _prefer(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_exact = candidate["status"] in EXACT_STATUSES
    current_exact = current["status"] in EXACT_STATUSES
    if candidate_exact != current_exact:
        return candidate_exact
    return float(candidate.get("elapsed_seconds", 0)) > float(
        current.get("elapsed_seconds", 0)
    )


def collect(
    taskset_path: Path,
    fleet_path: Path,
    shard_paths: list[Path],
    *,
    known_case_index: int | None = None,
    known_max_seconds: float | None = None,
    hard_case_indices: list[int] | None = None,
) -> dict[str, Any]:
    taskset = _read_json(taskset_path)
    fleet = _read_json(fleet_path)
    if taskset.get("schema") != "all-wildlife-score-profile-taskset-v1":
        raise ValueError("unexpected taskset schema")
    tasks = {int(task["task_index"]): task for task in taskset.get("tasks", [])}

    seen: dict[int, dict[str, Any]] = {}
    shard_rows: list[dict[str, Any]] = []
    for path in shard_paths:
        shard = _read_json(path)
        if shard.get("schema") != "all-wildlife-score-profile-shard-v1":
            raise ValueError(f"{path}: unexpected shard schema")
        accepted = 0
        for result in shard.get("results", []):
            index = int(result["task_index"])
            if index not in tasks:
                continue
            _validate_result(result, tasks[index])
            if index not in seen or _prefer(result, seen[index]):
                seen[index] = result
            accepted += 1
        shard_rows.append(
            {
                "path": str(path),
                "task_count": accepted,
                "elapsed_seconds": shard.get("elapsed_seconds"),
            }
        )

    case_summaries = []
    taskset_cases = {
        int(case["case_index"]): case for case in taskset.get("cases", [])
    }
    by_case: dict[int, list[dict[str, Any]]] = {}
    for result in seen.values():
        by_case.setdefault(int(result["case_index"]), []).append(result)
    for case_index in sorted(taskset_cases):
        case = taskset_cases[case_index]
        case_results = by_case.get(case_index, [])
        statuses = Counter(row["status"] for row in case_results)
        case_summaries.append(
            {
                **case,
                "received_profiles": len(case_results),
                "status_counts": dict(sorted(statuses.items())),
                "exact_profiles": sum(
                    row["status"] in EXACT_STATUSES for row in case_results
                ),
                "complete_exact": (
                    len(case_results) == case["profile_count"]
                    and all(row["status"] in EXACT_STATUSES for row in case_results)
                ),
            }
        )

    assessment: dict[str, Any] = {}
    if known_case_index is not None:
        known = next(
            (row for row in case_summaries if row["case_index"] == known_case_index),
            None,
        )
        assessment["known_case_complete"] = bool(known and known["complete_exact"])
        if known and known_max_seconds is not None and by_case.get(known_case_index):
            assessment["known_case_within_target_seconds"] = (
                max(row["elapsed_seconds"] for row in by_case[known_case_index])
                <= known_max_seconds
            )
    assessment["complete_hard_cases"] = [
        index
        for index in (hard_case_indices or [])
        if next(
            (row["complete_exact"] for row in case_summaries if row["case_index"] == index),
            False,
        )
    ]

    return {
        "schema": "all-wildlife-score-profile-collection-v2",
        "taskset": {"path": str(taskset_path), "task_count": len(tasks)},
        "fleet": {"path": str(fleet_path), "tag": fleet.get("tag")},
        "shards": shard_rows,
        "coverage": {"received": len(seen), "expected": len(tasks)},
        "cases": case_summaries,
        "totals": {
            "profiles": len(seen),
            "exact_profiles": sum(
                row["status"] in EXACT_STATUSES for row in seen.values()
            ),
            "unknown_profiles": sum(
                row["status"] == "UNKNOWN" for row in seen.values()
            ),
            "exact_infeasible_profiles": sum(
                row["status"] == "INFEASIBLE" for row in seen.values()
            ),
        },
        "assessment": assessment,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--fleet", "--fleet-ledger", dest="fleet", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--known-case-index", type=int)
    parser.add_argument("--known-max-seconds", type=float)
    parser.add_argument("--hard-case-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = collect(
        args.taskset,
        args.fleet,
        args.shard,
        known_case_index=args.known_case_index,
        known_max_seconds=args.known_max_seconds,
        hard_case_indices=args.hard_case_index,
    )
    _write_atomic(args.output, payload)
    print(json.dumps({**payload["coverage"], **payload["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
