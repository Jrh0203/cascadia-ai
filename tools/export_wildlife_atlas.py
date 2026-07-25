#!/usr/bin/env python3
"""Export the full wildlife catalog as a compact, validated web-app asset."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPECIES = ("bear", "elk", "salmon", "hawk", "fox")
VARIANTS = ("A", "B", "C", "D")
EXPECTED_RULESETS = len(VARIANTS) ** len(SPECIES)


def _validate_row(row: dict[str, Any], token_count: int, count_cap: int) -> None:
    ruleset = str(row["ruleset"])
    if len(ruleset) != len(SPECIES) or any(card not in VARIANTS for card in ruleset):
        raise ValueError(f"invalid ruleset {ruleset!r}")

    counts = [int(value) for value in row["counts"]]
    breakdown = [int(value) for value in row["score_breakdown"]]
    tokens = row["tokens"]
    if len(counts) != len(SPECIES) or len(breakdown) != len(SPECIES):
        raise ValueError(f"{ruleset}: expected five counts and five score parts")
    if sum(counts) != token_count or max(counts) > count_cap:
        raise ValueError(f"{ruleset}: invalid animal counts {counts}")
    if sum(breakdown) != int(row["optimum"]):
        raise ValueError(f"{ruleset}: score breakdown does not sum to total")
    if len(tokens) != token_count:
        raise ValueError(f"{ruleset}: expected {token_count} tokens")

    occupied: set[tuple[int, int]] = set()
    observed: Counter[str] = Counter()
    for token in tokens:
        coord = (int(token["q"]), int(token["r"]))
        wildlife = str(token["wildlife"])
        if wildlife not in SPECIES:
            raise ValueError(f"{ruleset}: unknown wildlife {wildlife!r}")
        if coord in occupied:
            raise ValueError(f"{ruleset}: overlapping coordinate {coord}")
        occupied.add(coord)
        observed[wildlife] += 1
    if [observed[species] for species in SPECIES] != counts:
        raise ValueError(f"{ruleset}: token species do not match counts")

    score = int(row["optimum"])
    upper = int(row["sound_upper"])
    exact = bool(row["proof_complete"])
    if upper < score or (exact and upper != score):
        raise ValueError(f"{ruleset}: invalid score interval [{score}, {upper}]")


def export(source: Path, output: Path) -> dict[str, Any]:
    source_document = json.loads(source.read_bytes())
    rows = source_document["results"]
    token_count = int(source_document["token_count"])
    count_cap = int(source_document["count_cap"])

    if len(rows) != EXPECTED_RULESETS:
        raise ValueError(f"expected {EXPECTED_RULESETS} rulesets, got {len(rows)}")
    if len({str(row["ruleset"]) for row in rows}) != EXPECTED_RULESETS:
        raise ValueError("rulesets are not unique")

    compact_rows = []
    expected_rulesets = [
        "".join(cards) for cards in itertools.product(VARIANTS, repeat=len(SPECIES))
    ]
    for expected_index, row in enumerate(sorted(rows, key=lambda value: int(value["index"]))):
        _validate_row(row, token_count, count_cap)
        if int(row["index"]) != expected_index:
            raise ValueError(
                f"catalog index mismatch: expected {expected_index}, got {row['index']}"
            )
        if str(row["ruleset"]) != expected_rulesets[expected_index]:
            raise ValueError(
                f"ruleset/index mismatch: expected {expected_rulesets[expected_index]}, "
                f"got {row['ruleset']}"
            )
        compact_rows.append(
            {
                "id": str(row["ruleset"]),
                "score": int(row["optimum"]),
                "upper": int(row["sound_upper"]),
                "exact": bool(row["proof_complete"]),
                "counts": [int(value) for value in row["counts"]],
                "parts": [int(value) for value in row["score_breakdown"]],
                "tokens": [
                    [
                        int(token["q"]),
                        int(token["r"]),
                        SPECIES.index(str(token["wildlife"])),
                    ]
                    for token in row["tokens"]
                ],
            }
        )

    leaders = [
        row["id"]
        for row in compact_rows
        if row["score"] == int(source_document["incumbent_holistic_maximum"])
    ]
    document = {
        "schema": "cascadia-wildlife-atlas-v1",
        "sourceSchema": str(source_document["schema"]),
        "rulesetCount": len(compact_rows),
        "tokenCount": token_count,
        "countCap": count_cap,
        "completedRulesets": sum(bool(row["exact"]) for row in compact_rows),
        "incumbentHolisticMaximum": int(
            source_document["incumbent_holistic_maximum"]
        ),
        "holisticSoundUpper": int(source_document["holistic_sound_upper"]),
        "leaders": leaders,
        "rows": compact_rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    document = export(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": document["rulesetCount"],
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
