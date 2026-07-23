#!/usr/bin/env python3
"""Resumable bounded maximization for unresolved all-wildlife count vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules
from tools.all_wildlife_exact import solve_counts

SCHEMA = "all-wildlife-bound-probe-v1"
COUNT_VECTORS = frozenset(rules.count_vectors())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _validate_board(
    row: dict[str, Any],
    ruleset: str,
    score_key: str,
) -> list[dict[str, int | str]]:
    tokens = rules.normalized_tokens(row["tokens"])
    counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if counts != tuple(row["counts"]) or counts not in COUNT_VECTORS:
        raise ValueError(f"{ruleset}: invalid board counts")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(occupied) != rules.TOKEN_COUNT or len(rules.components(occupied)) != 1:
        raise ValueError(f"{ruleset}: board is not a connected 20-token layout")
    breakdown = rules.score_tokens(tokens, ruleset)
    if (
        list(breakdown) != row["score_breakdown"]
        or sum(breakdown) != int(row[score_key])
    ):
        raise ValueError(f"{ruleset}: board score mismatch")
    return tokens


def _parse_counts(values: list[str] | None) -> list[tuple[int, int, int, int, int]]:
    if not values:
        return []
    parsed = []
    for value in values:
        fields = tuple(int(field) for field in value.split(","))
        if len(fields) != len(rules.SPECIES) or fields not in COUNT_VECTORS:
            raise ValueError(f"invalid count vector {value!r}")
        parsed.append(fields)
    if len(parsed) != len(set(parsed)):
        raise ValueError("duplicate count vector")
    return parsed


def _load_catalog(
    path: Path,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    expected_ruleset = rules.rulesets()[index]
    if payload.get("schema") != "all-wildlife-optimal-catalog-v1":
        raise ValueError("unexpected catalog schema")
    results = payload.get("results", [])
    if index >= len(results):
        raise ValueError("catalog does not contain requested index")
    row = results[index]
    if row.get("index") != index or row.get("ruleset") != expected_ruleset:
        raise ValueError("catalog row identity mismatch")
    _validate_board(row, expected_ruleset, "optimum")
    unresolved = [tuple(counts) for counts in row.get("unresolved_counts", [])]
    if len(unresolved) != len(set(unresolved)) or any(
        counts not in COUNT_VECTORS for counts in unresolved
    ):
        raise ValueError("invalid unresolved count set")
    if bool(row.get("proof_complete")) != (not unresolved):
        raise ValueError("catalog completeness mismatch")
    return payload, row, hashlib.sha256(encoded).hexdigest()


def _sources() -> dict[str, str]:
    return {
        "probe_source_sha256": _sha256(Path(__file__)),
        "exact_source_sha256": _sha256(Path("tools/all_wildlife_exact.py")),
        "exact_support_source_sha256": _sha256(Path("tools/cbddb_wildlife_exact.py")),
        "rules_source_sha256": _sha256(Path("tools/all_wildlife_rules.py")),
    }


def _best_board(
    base: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    boards = [
        {
            "score": int(base["optimum"]),
            "score_breakdown": base["score_breakdown"],
            "counts": base["counts"],
            "tokens": base["tokens"],
            "source": "base_catalog",
        }
    ]
    for attempt in attempts:
        if attempt.get("tokens") is not None:
            boards.append(
                {
                    "score": int(attempt["witness_score"]),
                    "score_breakdown": attempt["score_breakdown"],
                    "counts": attempt["counts"],
                    "tokens": attempt["tokens"],
                    "source": "bounded_maximization_witness",
                }
            )
    return min(
        boards,
        key=lambda board: (
            -int(board["score"]),
            json.dumps(board["tokens"], sort_keys=True),
        ),
    )


def _summary(
    ruleset: str,
    base: dict[str, Any],
    selected: list[tuple[int, int, int, int, int]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt_by_count = {tuple(attempt["counts"]): attempt for attempt in attempts}
    best = _best_board(base, attempts)
    best_score = int(best["score"])
    unresolved = [tuple(counts) for counts in base["unresolved_counts"]]
    count_bounds = {}
    remaining = []
    for counts in unresolved:
        analytical = rules.count_upper(counts, ruleset)
        attempt = attempt_by_count.get(counts)
        refined = int(attempt["refined_upper"]) if attempt else analytical
        refined = min(analytical, refined)
        count_bounds[counts] = refined
        if refined > best_score:
            remaining.append(list(counts))
    sound_upper = max([best_score, *count_bounds.values()])
    return {
        "best_witness": best,
        "sound_upper": sound_upper,
        "gap": sound_upper - best_score,
        "proof_complete": not remaining,
        "remaining_counts": remaining,
        "attempted_count_count": len(attempts),
        "selected_count_count": len(selected),
    }


def run_probe(args: argparse.Namespace) -> int:
    _, base, catalog_sha = _load_catalog(args.catalog, args.index)
    ruleset = base["ruleset"]
    unresolved = [tuple(counts) for counts in base["unresolved_counts"]]
    explicit = _parse_counts(args.counts)
    selected = explicit or sorted(
        unresolved,
        key=lambda counts: (-rules.count_upper(counts, ruleset), counts),
    )
    if any(counts not in unresolved for counts in selected):
        raise ValueError("selected count is not unresolved in the base catalog")
    if args.max_counts is not None:
        if args.max_counts < 1:
            raise ValueError("max-counts must be positive")
        selected = selected[: args.max_counts]

    sources = _sources()
    identity = {
        "ruleset_index": args.index,
        "ruleset": ruleset,
        "base_catalog_sha256": catalog_sha,
        **sources,
        "connectivity_required": not args.no_connectivity,
    }
    attempts: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text())
        if (
            prior.get("schema") != SCHEMA
            or prior.get("identity") != identity
            or prior.get("selected_counts") != [list(counts) for counts in selected]
        ):
            raise ValueError("resume identity mismatch")
        attempts = prior["attempts"]

    attempted = {tuple(attempt["counts"]) for attempt in attempts}
    started = time.monotonic()

    def payload() -> dict[str, Any]:
        summary = _summary(ruleset, base, selected, attempts)
        return {
            "schema": SCHEMA,
            "identity": identity,
            "configuration": {
                "time_limit_seconds_per_count": args.time_limit,
                "total_time_limit_seconds": args.total_time_limit,
                "workers": args.workers,
                "connectivity_required": not args.no_connectivity,
                "maximize": True,
            },
            "base_incumbent": {
                "score": base["optimum"],
                "score_breakdown": base["score_breakdown"],
                "counts": base["counts"],
                "tokens": base["tokens"],
            },
            "selected_counts": [list(counts) for counts in selected],
            "attempts": attempts,
            **summary,
            "updated_utc": datetime.now(UTC).isoformat(),
        }

    for counts in selected:
        if counts in attempted:
            continue
        elapsed = time.monotonic() - started
        remaining_time = args.total_time_limit - elapsed
        if remaining_time <= 0:
            break
        best = _best_board(base, attempts)
        incumbent = int(best["score"])
        analytical = rules.count_upper(counts, ruleset)
        if analytical <= incumbent:
            attempts.append(
                {
                    "counts": list(counts),
                    "minimum_score": incumbent + 1,
                    "analytical_upper": analytical,
                    "status": "DOMINATED",
                    "model_objective": None,
                    "best_bound": analytical,
                    "refined_upper": analytical,
                    "witness_score": None,
                    "score_breakdown": None,
                    "tokens": None,
                    "elapsed_seconds": 0.0,
                    "branches": 0,
                    "conflicts": 0,
                }
            )
            _write_atomic(args.output, payload())
            continue
        limit = min(args.time_limit, remaining_time)
        hint = best["tokens"] if tuple(best["counts"]) == counts else None
        result = solve_counts(
            ruleset,
            counts,
            incumbent + 1,
            time_limit_seconds=limit,
            workers=args.workers,
            initial_tokens=hint,
            enforce_connectivity=not args.no_connectivity,
            maximize=True,
        )
        if result.status == "MODEL_INVALID":
            raise RuntimeError("CP-SAT rejected the exact model")
        witness_score = None
        breakdown = None
        tokens = None
        if result.tokens is not None and result.score_breakdown is not None:
            tokens = rules.normalized_tokens(result.tokens)
            breakdown = rules.score_tokens(tokens, ruleset)
            witness_score = sum(breakdown)
            if witness_score < incumbent + 1:
                raise AssertionError("maximization witness is below its threshold")
            witness_row = {
                "counts": list(counts),
                "tokens": tokens,
                "score_breakdown": list(breakdown),
                "score": witness_score,
            }
            _validate_board(witness_row, ruleset, "score")

        if result.status == "INFEASIBLE":
            refined_upper = incumbent
        else:
            if result.best_bound is None or not math.isfinite(result.best_bound):
                raise RuntimeError("maximization result omitted its objective bound")
            refined_upper = min(analytical, int(result.best_bound))
        if witness_score is not None and refined_upper < witness_score:
            raise AssertionError("solver upper bound is below its verified witness")
        attempts.append(
            {
                "counts": list(counts),
                "minimum_score": incumbent + 1,
                "analytical_upper": analytical,
                "status": result.status,
                "model_objective": result.objective,
                "best_bound": result.best_bound,
                "refined_upper": refined_upper,
                "witness_score": witness_score,
                "score_breakdown": list(breakdown) if breakdown is not None else None,
                "tokens": tokens,
                "elapsed_seconds": result.elapsed_seconds,
                "branches": result.branches,
                "conflicts": result.conflicts,
            }
        )
        _write_atomic(args.output, payload())

    final = payload()
    _write_atomic(args.output, final)
    print(
        json.dumps(
            {
                "ruleset": ruleset,
                "score": final["best_witness"]["score"],
                "sound_upper": final["sound_upper"],
                "gap": final["gap"],
                "proof_complete": final["proof_complete"],
                "attempts": len(final["attempts"]),
            },
            sort_keys=True,
        )
    )
    return 0 if final["proof_complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--counts", action="append")
    parser.add_argument("--max-counts", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--total-time-limit", type=float, default=330)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-connectivity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return run_probe(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
