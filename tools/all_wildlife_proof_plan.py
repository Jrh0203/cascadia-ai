#!/usr/bin/env python3
"""Create a deterministic, load-balanced all-ruleset exact-proof plan."""

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


def build_plan(candidates_path: Path, hosts: list[str]) -> dict[str, Any]:
    if not hosts or len(hosts) != len(set(hosts)):
        raise ValueError("hosts must be a nonempty unique list")
    encoded = candidates_path.read_bytes()
    candidates = json.loads(encoded)
    if candidates.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected candidate schema")
    work = []
    for index, ruleset in enumerate(rules.rulesets()):
        row = candidates["candidates"][index]
        if row["index"] != index or row["ruleset"] != ruleset:
            raise ValueError(f"candidate identity mismatch at index {index}")
        incumbent = int(row["score"])
        branch_count = sum(
            rules.count_upper(counts, ruleset) > incumbent
            for counts in rules.count_vectors()
        )
        # Even an already-bound ruleset has fixed startup and validation cost.
        work.append((max(1, branch_count), index, ruleset, branch_count))

    shards: list[dict[str, Any]] = [
        {"host": host, "estimated_weight": 0, "indices": []} for host in hosts
    ]
    for weight, index, _ruleset, _branch_count in sorted(
        work, key=lambda item: (-item[0], item[1])
    ):
        shard = min(
            shards,
            key=lambda item: (
                item["estimated_weight"],
                len(item["indices"]),
                hosts.index(item["host"]),
            ),
        )
        shard["indices"].append(index)
        shard["estimated_weight"] += weight

    by_index = {
        index: {
            "index": index,
            "ruleset": ruleset,
            "candidate_score": int(candidates["candidates"][index]["score"]),
            "unresolved_count_branches": branch_count,
        }
        for _weight, index, ruleset, branch_count in work
    }
    for shard in shards:
        # Hard-first ordering exposes the proof tail early while preserving LPT
        # makespan balance.
        shard["indices"].sort(
            key=lambda index: (
                -by_index[index]["unresolved_count_branches"],
                index,
            )
        )
        shard["ruleset_count"] = len(shard["indices"])
    return {
        "schema": "all-wildlife-proof-plan-v1",
        "candidate_sha256": hashlib.sha256(encoded).hexdigest(),
        "ruleset_count": len(work),
        "count_cap": rules.COUNT_CAP,
        "shards": shards,
        "rulesets": [by_index[index] for index in range(len(work))],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_plan(args.candidates, args.hosts)
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "rulesets": payload["ruleset_count"],
                "shards": [
                    {
                        "host": shard["host"],
                        "rulesets": shard["ruleset_count"],
                        "estimated_weight": shard["estimated_weight"],
                    }
                    for shard in payload["shards"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
