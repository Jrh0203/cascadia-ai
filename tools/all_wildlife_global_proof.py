#!/usr/bin/env python3
"""Resumable exact global proof for one all-card wildlife ruleset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime
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


def _candidate(path: Path, index: int) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if payload.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected candidate schema")
    row = payload["candidates"][index]
    expected = rules.rulesets()[index]
    if row["index"] != index or row["ruleset"] != expected:
        raise ValueError("candidate index/ruleset mismatch")
    breakdown = rules.score_tokens(row["tokens"], expected)
    if list(breakdown) != row["score_breakdown"] or sum(breakdown) != row["score"]:
        raise ValueError("candidate score mismatch")
    return row, hashlib.sha256(encoded).hexdigest()


def _proof_complete(
    ruleset: str,
    incumbent: int,
    exclusions: dict[tuple[int, ...], int],
) -> bool:
    for counts in rules.count_vectors():
        if rules.count_upper(counts, ruleset) <= incumbent:
            continue
        # Infeasibility at threshold t proves the count optimum is at most t-1.
        if exclusions.get(counts, incumbent + 2) > incumbent + 1:
            return False
    return True


def run(args: argparse.Namespace) -> int:
    ruleset = rules.rulesets()[args.index]
    candidate, candidate_sha = _candidate(args.candidates, args.index)
    output = args.output
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    exact_sha = hashlib.sha256(
        Path("tools/all_wildlife_exact.py").read_bytes()
    ).hexdigest()
    exact_support_sha = hashlib.sha256(
        Path("tools/cbddb_wildlife_exact.py").read_bytes()
    ).hexdigest()
    rules_sha = hashlib.sha256(
        Path("tools/all_wildlife_rules.py").read_bytes()
    ).hexdigest()
    incumbent = {
        "score": int(candidate["score"]),
        "score_breakdown": candidate["score_breakdown"],
        "counts": candidate["counts"],
        "tokens": candidate["tokens"],
        "source": "merged_candidate",
    }
    attempts: list[dict[str, Any]] = []
    exclusions: dict[tuple[int, ...], int] = {}
    attempted_this_invocation: set[tuple[int, ...]] = set()

    if args.resume and output.exists():
        payload = json.loads(output.read_text())
        identity = payload["identity"]
        expected_identity = {
            "ruleset_index": args.index,
            "ruleset": ruleset,
            "candidate_sha256": candidate_sha,
            "proof_source_sha256": source_sha,
            "exact_source_sha256": exact_sha,
            "exact_support_source_sha256": exact_support_sha,
            "rules_source_sha256": rules_sha,
            "connectivity_required": not args.no_connectivity,
        }
        if identity != expected_identity:
            raise ValueError("resume identity mismatch")
        incumbent = payload["incumbent"]
        attempts = payload["attempts"]
        for attempt in attempts:
            if attempt["status"] == "INFEASIBLE":
                counts = tuple(attempt["counts"])
                threshold = int(attempt["threshold"])
                exclusions[counts] = min(exclusions.get(counts, threshold), threshold)

    started = time.monotonic()

    def payload() -> dict[str, Any]:
        complete = _proof_complete(ruleset, int(incumbent["score"]), exclusions)
        unresolved = [
            list(counts)
            for counts in rules.count_vectors()
            if rules.count_upper(counts, ruleset) > int(incumbent["score"])
            and exclusions.get(counts, int(incumbent["score"]) + 2)
            > int(incumbent["score"]) + 1
        ]
        return {
            "schema": "all-wildlife-global-proof-v1",
            "identity": {
                "ruleset_index": args.index,
                "ruleset": ruleset,
                "candidate_sha256": candidate_sha,
                "proof_source_sha256": source_sha,
                "exact_source_sha256": exact_sha,
                "exact_support_source_sha256": exact_support_sha,
                "rules_source_sha256": rules_sha,
                "connectivity_required": not args.no_connectivity,
            },
            "configuration": {
                "time_limit_seconds": args.time_limit,
                "workers": args.workers,
                "connectivity_required": not args.no_connectivity,
            },
            "updated_utc": datetime.now(UTC).isoformat(),
            "proof_complete": complete,
            "incumbent": incumbent,
            "attempts": attempts,
            "unresolved_counts": unresolved,
        }

    while time.monotonic() - started < args.total_time_limit:
        threshold = int(incumbent["score"]) + 1
        allocations = sorted(
            (
                (counts, rules.count_upper(counts, ruleset))
                for counts in rules.count_vectors()
                if rules.count_upper(counts, ruleset) >= threshold
                and exclusions.get(counts, threshold + 1) > threshold
                and counts not in attempted_this_invocation
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not allocations:
            break
        improved = False
        for counts, upper in allocations:
            remaining = args.total_time_limit - (time.monotonic() - started)
            if remaining <= 0:
                break
            limit = min(args.time_limit, remaining)
            hint = (
                incumbent["tokens"]
                if tuple(incumbent["counts"]) == counts
                else None
            )
            result = solve_counts(
                ruleset,
                counts,
                threshold,
                time_limit_seconds=limit,
                workers=args.workers,
                initial_tokens=hint,
                enforce_connectivity=not args.no_connectivity,
                maximize=False,
            )
            attempt = {
                "counts": list(counts),
                "count_upper": upper,
                "threshold": threshold,
                "status": result.status,
                "elapsed_seconds": result.elapsed_seconds,
                "branches": result.branches,
                "conflicts": result.conflicts,
                "objective": result.objective,
                "score_breakdown": list(result.score_breakdown)
                if result.score_breakdown
                else None,
                "tokens": result.tokens,
            }
            attempts.append(attempt)
            attempted_this_invocation.add(counts)
            if result.status == "INFEASIBLE":
                exclusions[counts] = min(exclusions.get(counts, threshold), threshold)
            elif (
                not args.no_connectivity
                and result.tokens is not None
                and result.score_breakdown is not None
            ):
                actual = sum(result.score_breakdown)
                if actual < threshold:
                    raise AssertionError("exact witness is below its requested threshold")
                if actual > int(incumbent["score"]):
                    incumbent = {
                        "score": actual,
                        "score_breakdown": list(result.score_breakdown),
                        "counts": list(counts),
                        "tokens": result.tokens,
                        "source": "exact_feasibility_witness",
                    }
                    improved = True
                    attempted_this_invocation.discard(counts)
            _write_atomic(output, payload())
            if improved:
                break
        if not improved and all(counts in attempted_this_invocation for counts, _ in allocations):
            break

    final = payload()
    _write_atomic(output, final)
    print(
        json.dumps(
            {
                "ruleset": ruleset,
                "score": incumbent["score"],
                "proof_complete": final["proof_complete"],
                "attempts": len(attempts),
                "unresolved": len(final["unresolved_counts"]),
            },
            sort_keys=True,
        )
    )
    return 0 if final["proof_complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--total-time-limit", type=float, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-connectivity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
