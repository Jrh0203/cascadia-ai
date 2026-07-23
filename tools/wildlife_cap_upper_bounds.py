#!/usr/bin/env python3
"""Enumerate sound all-board wildlife upper bounds for a configurable count cap.

This is deliberately separate from the cap-six exact catalog solvers.  It
extends only their geometry-free scoring relaxations; it does not claim that a
maximizing count allocation is geometrically realizable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

SPECIES = ("bear", "elk", "salmon", "hawk", "fox")
TOKEN_COUNT = 20
SCHEMA = "wildlife-cap-upper-bounds-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partition_upper(token_count: int, group_scores: tuple[int, ...]) -> int:
    """Maximum additive score from groups with sizes 1..len(group_scores)-1."""
    best = [0] * (token_count + 1)
    for used in range(1, token_count + 1):
        best[used] = max(
            group_scores[size] + best[used - size]
            for size in range(1, min(used, len(group_scores) - 1) + 1)
        )
    return best[token_count]


def bear_a_standalone(token_count: int) -> int:
    pairs = min(token_count // 2, 4)
    return (0, 4, 11, 19, 27)[pairs]


def elk_ab_standalone(token_count: int) -> int:
    return _partition_upper(token_count, (0, 2, 5, 9, 13))


def salmon_a_standalone(token_count: int) -> int:
    # A valid run scores 2/5/8/12/16/20/25, with 25 for every length >= 7.
    run_scores = (0, 2, 5, 8, 12, 16, 20) + (25,) * max(0, token_count - 6)
    return _partition_upper(token_count, run_scores)


def hawk_a_standalone(token_count: int) -> int:
    if token_count <= 7:
        return (0, 2, 5, 8, 11, 14, 18, 22)[token_count]
    return 26


def bear_c_standalone(token_count: int) -> int:
    # Only components of sizes 1/2/3 score; the all-three-sizes bonus is
    # awarded at most once.
    return max(
        _bear_c_partition_score(token_count, ones, pairs, triples)
        for triples in range(token_count // 3 + 1)
        for pairs in range((token_count - 3 * triples) // 2 + 1)
        for ones in (token_count - 3 * triples - 2 * pairs,)
    )


def _bear_c_partition_score(
    token_count: int,
    singletons: int,
    pairs: int,
    triples: int,
) -> int:
    if singletons + 2 * pairs + 3 * triples != token_count:
        raise ValueError("Bear-C partition does not consume the requested tokens")
    base = 2 * singletons + 5 * pairs + 8 * triples
    return base + 3 * int(singletons > 0 and pairs > 0 and triples > 0)


def salmon_d_standalone(token_count: int) -> int:
    """Sound standalone Salmon-D bound with reusable surrounding tokens.

    A valid unbranched component of length k has at most 2k+4 distinct
    surrounding cells.  The same non-salmon token may border multiple salmon
    components and score once for each, so each component independently gets
    up to the full non-salmon population.
    """
    non_salmon = TOKEN_COUNT - token_count
    best = [0] * (token_count + 1)
    for used in range(1, token_count + 1):
        best[used] = best[used - 1]  # leave one salmon in a non-scoring group
        for size in range(3, used + 1):
            component_score = size + min(non_salmon, 2 * size + 4)
            best[used] = max(best[used], component_score + best[used - size])
    return best[token_count]


def _validate_counts(counts: tuple[int, int, int, int, int]) -> None:
    if len(counts) != len(SPECIES):
        raise ValueError(f"expected five species counts, got {counts}")
    if sum(counts) != TOKEN_COUNT or any(count < 0 for count in counts):
        raise ValueError(f"invalid counts: {counts}")


def aaaaa_count_upper(
    counts: tuple[int, int, int, int, int],
    *,
    incidence_aware: bool,
) -> int:
    """Sound AAAAA upper bound for one count allocation."""
    _validate_counts(counts)
    bear, elk, salmon, hawk, fox = counts
    non_fox = (
        bear_a_standalone(bear)
        + elk_ab_standalone(elk)
        + salmon_a_standalone(salmon)
        + hawk_a_standalone(hawk)
    )
    if incidence_aware:
        # A token has six neighboring cells.  Thus one non-fox token can be
        # observed by at most six foxes, which first matters when fox == 7.
        fox_score = fox if fox >= 2 else 0
        fox_score += sum(min(fox, 6 * count) for count in counts[:4] if count)
    else:
        observed_types = sum(count > 0 for count in counts[:4]) + int(fox >= 2)
        fox_score = fox * observed_types
    return non_fox + fox_score


def cbddb_count_upper(counts: tuple[int, int, int, int, int]) -> int:
    """Sound geometry-free CBDDB upper bound for one count allocation."""
    _validate_counts(counts)
    bear, elk, salmon, hawk, fox = counts
    distinct_between = sum(count > 0 for count in (bear, elk, salmon, fox))
    hawk_pair_score = (0, 4, 7, 9)[min(distinct_between, 3)]
    doubled_non_fox = sum(count >= 2 for count in (bear, elk, salmon, hawk))
    fox_score = (0, 3, 5, 7)[min(doubled_non_fox, 3)]
    return (
        bear_c_standalone(bear)
        + elk_ab_standalone(elk)
        + salmon_d_standalone(salmon)
        + (hawk // 2) * hawk_pair_score
        + fox * fox_score
    )


def count_vectors(
    cap: int,
    bound: Callable[[tuple[int, int, int, int, int]], int],
) -> list[tuple[tuple[int, int, int, int, int], int]]:
    if not 0 <= cap <= TOKEN_COUNT:
        raise ValueError(f"cap must be between 0 and {TOKEN_COUNT}")
    rows = []
    for counts in itertools.product(range(cap + 1), repeat=len(SPECIES)):
        if sum(counts) == TOKEN_COUNT:
            rows.append((counts, bound(counts)))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


def _maximum_summary(
    cap: int,
    bound: Callable[[tuple[int, int, int, int, int]], int],
) -> dict[str, Any]:
    rows = count_vectors(cap, bound)
    if not rows:
        return {"allocation_count": 0, "maximum": None, "maximizing_counts": []}
    maximum = rows[0][1]
    return {
        "allocation_count": len(rows),
        "maximum": maximum,
        "maximizing_counts": [list(counts) for counts, score in rows if score == maximum],
    }


def analyze_cap(cap: int) -> dict[str, Any]:
    naive = _maximum_summary(
        cap,
        lambda counts: aaaaa_count_upper(counts, incidence_aware=False),
    )
    incidence = _maximum_summary(
        cap,
        lambda counts: aaaaa_count_upper(counts, incidence_aware=True),
    )
    cbddb = _maximum_summary(cap, cbddb_count_upper)
    if not (
        naive["allocation_count"]
        == incidence["allocation_count"]
        == cbddb["allocation_count"]
    ):
        raise AssertionError("rulesets enumerated different count spaces")
    return {
        "cap": cap,
        "token_count": TOKEN_COUNT,
        "allocation_count": naive["allocation_count"],
        "aaaaa_geometry_free": {
            "maximum": naive["maximum"],
            "maximizing_counts": naive["maximizing_counts"],
        },
        "aaaaa_incidence_aware": {
            "maximum": incidence["maximum"],
            "maximizing_counts": incidence["maximizing_counts"],
            "extra_constraint": "one token can be adjacent to at most six foxes",
        },
        "cbddb_geometry_free": {
            "maximum": cbddb["maximum"],
            "maximizing_counts": cbddb["maximizing_counts"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema": SCHEMA,
        "source_sha256": sha256(Path(__file__).resolve()),
        "analysis": [analyze_cap(cap) for cap in args.cap],
        "interpretation": {
            "sound_upper_bounds": True,
            "geometric_realizability_proven": False,
            "optimal_board_score_proven": False,
        },
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
