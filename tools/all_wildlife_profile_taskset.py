#!/usr/bin/env python3
"""Freeze independent fixed-score branches for all-wildlife exact solving."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def parse_case(value: str) -> tuple[str, tuple[int, int, int, int, int], int]:
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError("case must be RULESET:b,e,s,h,f:THRESHOLD")
    ruleset = fields[0].upper()
    rules.parse_ruleset(ruleset)
    counts = tuple(int(part) for part in fields[1].split(","))
    if len(counts) != len(rules.SPECIES) or counts not in rules.count_vectors():
        raise ValueError(f"invalid count vector: {fields[1]}")
    threshold = int(fields[2])
    if threshold > rules.count_upper(counts, ruleset):
        raise ValueError("threshold exceeds count upper bound")
    return ruleset, counts, threshold  # type: ignore[return-value]


def build_taskset(cases: list[str]) -> dict[str, Any]:
    tasks = []
    case_rows = []
    seen = set()
    for case_index, encoded in enumerate(cases):
        ruleset, counts, threshold = parse_case(encoded)
        identity = (ruleset, counts, threshold)
        if identity in seen:
            raise ValueError(f"duplicate case: {encoded}")
        seen.add(identity)
        upper = rules.count_upper(counts, ruleset)
        profiles = rules.count_score_profiles(counts, ruleset, threshold, upper)
        case_id = f"{ruleset}_{'_'.join(map(str, counts))}_ge_{threshold}"
        case_rows.append(
            {
                "case_index": case_index,
                "case_id": case_id,
                "ruleset": ruleset,
                "counts": list(counts),
                "threshold": threshold,
                "upper": upper,
                "profile_count": len(profiles),
            }
        )
        for profile_index, profile in enumerate(profiles):
            tasks.append(
                {
                    "task_index": len(tasks),
                    "case_index": case_index,
                    "case_id": case_id,
                    "profile_index": profile_index,
                    "ruleset": ruleset,
                    "counts": list(counts),
                    "threshold": threshold,
                    "upper": upper,
                    "score_profile": list(profile),
                }
            )
    return {
        "schema": "all-wildlife-score-profile-taskset-v1",
        "rules_source_sha256": hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest(),
        "cases": case_rows,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_taskset(args.case)
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "cases": len(payload["cases"]),
                "tasks": payload["task_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
