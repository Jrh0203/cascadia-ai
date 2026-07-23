#!/usr/bin/env python3
"""Fail-closed independent and production verification of a merged catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tools import all_wildlife_rules as rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("target/release/all_wildlife_score_oracle"),
    )
    args = parser.parse_args()

    encoded = args.catalog.read_bytes()
    payload = json.loads(encoded)
    if payload.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected catalog schema")
    candidates = payload["candidates"]
    if len(candidates) != len(rules.rulesets()):
        raise ValueError("catalog must contain exactly 1,024 candidates")
    requests = []
    for index, (ruleset, candidate) in enumerate(
        zip(rules.rulesets(), candidates, strict=True)
    ):
        if candidate["index"] != index or candidate["ruleset"] != ruleset:
            raise ValueError(f"row {index} is not canonical")
        tokens = rules.normalized_tokens(candidate["tokens"])
        counts = tuple(
            sum(row["wildlife"] == species for row in tokens)
            for species in rules.SPECIES
        )
        if counts != tuple(candidate["counts"]) or any(
            count > rules.COUNT_CAP for count in counts
        ):
            raise ValueError(f"{ruleset}: invalid counts")
        occupied = {(int(row["q"]), int(row["r"])) for row in tokens}
        if len(rules.components(occupied)) != 1:
            raise ValueError(f"{ruleset}: board is disconnected")
        independent = rules.score_tokens(tokens, ruleset)
        if list(independent) != candidate["score_breakdown"]:
            raise ValueError(f"{ruleset}: independent breakdown mismatch")
        if sum(independent) != candidate["score"]:
            raise ValueError(f"{ruleset}: independent total mismatch")
        if rules.count_upper(counts, ruleset) < candidate["score"]:
            raise ValueError(f"{ruleset}: candidate exceeds sound count bound")
        requests.append({"ruleset": ruleset, "tokens": tokens})

    completed = subprocess.run(
        [str(args.oracle)],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
    )
    responses = json.loads(completed.stdout)
    if len(responses) != len(candidates):
        raise ValueError("production oracle response count mismatch")
    for candidate, response in zip(candidates, responses, strict=True):
        if response["score"] != candidate["score"]:
            raise ValueError(f"{candidate['ruleset']}: production total mismatch")
        if response["score_breakdown"] != candidate["score_breakdown"]:
            raise ValueError(f"{candidate['ruleset']}: production breakdown mismatch")
    canonical = json.dumps(responses, sort_keys=True, separators=(",", ":"))
    print(
        json.dumps(
            {
                "schema": "all-wildlife-merged-candidate-verification-v1",
                "catalog_sha256": hashlib.sha256(encoded).hexdigest(),
                "rulesets": len(candidates),
                "production_response_sha256": hashlib.sha256(
                    canonical.encode()
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
