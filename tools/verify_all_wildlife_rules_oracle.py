#!/usr/bin/env python3
"""Cross-check the independent all-card scorer against the Rust production scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tools import all_wildlife_rules
from tools.test_all_wildlife_rules import random_connected_board

DEFAULT_SEEDS = (101, 202, 303, 404)


def verify(oracle: Path, seeds: tuple[int, ...]) -> dict[str, int | str]:
    boards = [random_connected_board(seed) for seed in seeds]
    requests = [
        {"ruleset": ruleset, "tokens": board}
        for board in boards
        for ruleset in all_wildlife_rules.rulesets()
    ]
    completed = subprocess.run(
        [str(oracle)],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
    )
    responses = json.loads(completed.stdout)
    if len(responses) != len(requests):
        raise AssertionError(
            f"oracle returned {len(responses)} responses for {len(requests)} cases"
        )
    for request, response in zip(requests, responses, strict=True):
        expected = list(
            all_wildlife_rules.score_tokens(request["tokens"], request["ruleset"])
        )
        actual = response["score_breakdown"]
        if actual != expected:
            raise AssertionError(
                f"{request['ruleset']} mismatch: independent={expected}, production={actual}"
            )
    canonical = json.dumps(responses, sort_keys=True, separators=(",", ":"))
    return {
        "boards": len(boards),
        "rulesets": len(all_wildlife_rules.rulesets()),
        "cases": len(requests),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("target/release/all_wildlife_score_oracle"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    print(json.dumps(verify(args.oracle, tuple(args.seeds)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
