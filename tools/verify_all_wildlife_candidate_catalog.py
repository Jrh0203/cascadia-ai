#!/usr/bin/env python3
"""Fail-closed independent/production verification of candidate or exact catalogs."""

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
    schema = payload.get("schema")
    if schema == "all-wildlife-merged-candidates-v1":
        candidates = payload["candidates"]
        score_key = "score"
        verification_schema = "all-wildlife-merged-candidate-verification-v1"
    elif schema == "all-wildlife-optimal-catalog-v1":
        if not payload.get("proof_complete"):
            raise ValueError("refusing to verify an incomplete optimal catalog")
        candidates = payload["results"]
        if any(
            not row.get("proof_complete") or row.get("unresolved_counts")
            for row in candidates
        ):
            raise ValueError("optimal catalog contains an incomplete row")
        score_key = "optimum"
        verification_schema = "all-wildlife-optimal-catalog-verification-v1"
    else:
        raise ValueError("unexpected catalog schema")
    if len(candidates) != len(rules.rulesets()):
        raise ValueError("catalog must contain exactly 1,024 candidates")
    if schema == "all-wildlife-optimal-catalog-v1":
        holistic = max(candidate[score_key] for candidate in candidates)
        holistic_rulesets = [
            candidate["ruleset"]
            for candidate in candidates
            if candidate[score_key] == holistic
        ]
        if (
            payload.get("holistic_optimum") != holistic
            or payload.get("holistic_rulesets") != holistic_rulesets
        ):
            raise ValueError("holistic optimum summary mismatch")
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
        if sum(independent) != candidate[score_key]:
            raise ValueError(f"{ruleset}: independent total mismatch")
        if rules.count_upper(counts, ruleset) < candidate[score_key]:
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
        if response["score"] != candidate[score_key]:
            raise ValueError(f"{candidate['ruleset']}: production total mismatch")
        if response["score_breakdown"] != candidate["score_breakdown"]:
            raise ValueError(f"{candidate['ruleset']}: production breakdown mismatch")
    canonical = json.dumps(responses, sort_keys=True, separators=(",", ":"))
    print(
        json.dumps(
            {
                "schema": verification_schema,
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
