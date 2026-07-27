#!/usr/bin/env python3
"""Reduce the validated fixed-count catalog to one best board per ruleset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules
from tools.all_wildlife_fixed_count_collect import (
    COUNT_VECTORS,
    RULESETS,
    TOTAL_CELLS,
    _validate_candidate,
)
from tools.all_wildlife_fixed_count_merge import SCHEMA as BEST_CHUNK_SCHEMA

CATALOG_SCHEMA = "all-wildlife-fixed-count-ruleset-catalog-v1"
DELIVERY_SCHEMA = "all-wildlife-fixed-count-delivery-v1"
ATLAS_SCHEMA = "cascadia-wildlife-atlas-v1"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def _candidate_key(candidate: dict[str, Any]) -> tuple[int, str, int]:
    return (
        -int(candidate["score"]),
        json.dumps(candidate["tokens"], separators=(",", ":")),
        int(candidate["count_index"]),
    )


def _expanded_tokens(candidate: dict[str, Any]) -> list[dict[str, int | str]]:
    return [
        {
            "q": int(q),
            "r": int(r),
            "wildlife": rules.SPECIES[int(species)],
        }
        for q, r, species in candidate["tokens"]
    ]


def _atlas(catalog: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "id": row["ruleset"],
            "score": row["score"],
            "upper": row["sound_upper"],
            "exact": row["proof_complete"],
            "counts": row["counts"],
            "parts": row["score_breakdown"],
            "tokens": [
                [
                    token["q"],
                    token["r"],
                    rules.SPECIES.index(token["wildlife"]),
                ]
                for token in row["tokens"]
            ],
        }
        for row in catalog["results"]
    ]
    return {
        "schema": ATLAS_SCHEMA,
        "sourceSchema": CATALOG_SCHEMA,
        "rulesetCount": len(rows),
        "tokenCount": rules.TOKEN_COUNT,
        "countCap": rules.COUNT_CAP,
        "completedRulesets": catalog["proof_complete_rulesets"],
        "incumbentHolisticMaximum": catalog["incumbent_holistic_maximum"],
        "holisticSoundUpper": catalog["holistic_sound_upper"],
        "leaders": catalog["incumbent_holistic_rulesets"],
        "rows": rows,
    }


def _markdown(catalog: dict[str, Any], catalog_path: Path, atlas_paths: list[Path]) -> str:
    lines = [
        "# Fixed-count wildlife catalog results",
        "",
        (
            f"Validated one connected 20-animal board for every one of "
            f"{catalog['ruleset_count']:,} scoring-card combinations and every "
            f"one of {catalog['count_vector_count']:,} legal count vectors with "
            f"at most six of each species."
        ),
        "",
        (
            f"Best score found: **{catalog['incumbent_holistic_maximum']}**. "
            f"Sound holistic upper: **{catalog['holistic_sound_upper']}**. "
            f"Bound-matched rulesets: **{catalog['proof_complete_rulesets']:,}"
            f"/{catalog['ruleset_count']:,}**."
        ),
        "",
        (
            "A row is marked exact only when its best board reaches the maximum "
            "sound count upper across all 826 count vectors. Other rows are "
            "best-known results from the completed shallow and production search."
        ),
        "",
        f"Machine-readable catalog: `{catalog_path}`.",
    ]
    if atlas_paths:
        lines.append(
            "Atlas assets: "
            + ", ".join(f"`{path}`" for path in atlas_paths)
            + "."
        )
    lines.extend(["", "## Holistic leaders", ""])
    leaders = set(catalog["incumbent_holistic_rulesets"])
    for row in catalog["results"]:
        if row["ruleset"] not in leaders:
            continue
        lines.extend(
            [
                f"### {row['ruleset']} — {row['score']} points",
                "",
                f"- Counts (bear/elk/salmon/hawk/fox): `{row['counts']}`",
                f"- Breakdown: `{row['score_breakdown']}`",
                f"- Sound upper: `{row['sound_upper']}`",
                f"- Source stage: `{row['source_stage']}`",
                "- Board:",
                "",
                "```json",
                json.dumps(row["tokens"], separators=(",", ":")),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## All scoring-card combinations",
            "",
            "| Rank | Cards | Score | Upper | Gap | Exact | Counts | Breakdown |",
            "| ---: | :---: | ---: | ---: | ---: | :---: | :--- | :--- |",
        ]
    )
    ranked = sorted(
        catalog["results"],
        key=lambda row: (
            -int(row["score"]),
            -int(row["proof_complete"]),
            int(row["sound_upper"]),
            row["ruleset"],
        ),
    )
    score_rank = 0
    previous_score = None
    for position, row in enumerate(ranked, start=1):
        if row["score"] != previous_score:
            score_rank = position
            previous_score = row["score"]
        lines.append(
            f"| {score_rank} | `{row['ruleset']}` | {row['score']} | "
            f"{row['sound_upper']} | {row['sound_upper'] - row['score']} | "
            f"{'yes' if row['proof_complete'] else 'no'} | "
            f"`{row['counts']}` | `{row['score_breakdown']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_report(
    catalog_directory: Path,
    output_json: Path,
    output_markdown: Path,
    atlas_outputs: list[Path],
    delivery_summary: Path,
    *,
    chunk_size: int = 256,
) -> dict[str, Any]:
    source_summary_path = catalog_directory / "summary.json"
    source_summary = json.loads(source_summary_path.read_bytes())
    if (
        source_summary.get("complete") is not True
        or source_summary.get("total_cells") != TOTAL_CELLS
        or source_summary.get("deep_source_validation") is not True
        or source_summary.get("deep_output_validation") is not True
        or source_summary.get("validated_output_cells") != TOTAL_CELLS
    ):
        raise ValueError("merged catalog summary is incomplete or not deeply validated")

    total_chunks = (TOTAL_CELLS + chunk_size - 1) // chunk_size
    winners: list[dict[str, Any] | None] = [None] * len(RULESETS)
    ruleset_uppers = [-1] * len(RULESETS)
    validated_cells = 0
    source_stages = [stage["name"] for stage in source_summary["stages"]]

    for chunk_index in range(total_chunks):
        path = catalog_directory / f"chunk_{chunk_index:05d}.json"
        payload = json.loads(path.read_bytes())
        start = chunk_index * chunk_size
        end = min(start + chunk_size, TOTAL_CELLS)
        candidates = payload.get("candidates")
        if (
            payload.get("schema") != BEST_CHUNK_SCHEMA
            or payload.get("token_count") != rules.TOKEN_COUNT
            or payload.get("count_cap") != rules.COUNT_CAP
            or payload.get("total_cells") != TOTAL_CELLS
            or payload.get("range_start") != start
            or payload.get("range_end") != end
            or payload.get("source_stages") != source_stages
            or not isinstance(candidates, list)
            or len(candidates) != end - start
        ):
            raise ValueError(f"{path}: invalid merged chunk")
        for offset, candidate in enumerate(candidates):
            cell_index = start + offset
            if (
                not isinstance(candidate, dict)
                or candidate.get("source_stage") not in source_stages
            ):
                raise ValueError(f"{path}: cell {cell_index} has an invalid source")
            _validate_candidate(candidate, cell_index, deep=True)
            ruleset_index, _ = divmod(cell_index, len(COUNT_VECTORS))
            ruleset_uppers[ruleset_index] = max(
                ruleset_uppers[ruleset_index],
                int(candidate["count_upper"]),
            )
            current = winners[ruleset_index]
            if current is None or _candidate_key(candidate) < _candidate_key(current):
                winners[ruleset_index] = candidate
            validated_cells += 1

    if validated_cells != TOTAL_CELLS or any(winner is None for winner in winners):
        raise ValueError("merged catalog does not cover every fixed-count cell")

    rows = []
    for ruleset_index, (ruleset, winner) in enumerate(
        zip(RULESETS, winners, strict=True)
    ):
        assert winner is not None
        score = int(winner["score"])
        sound_upper = ruleset_uppers[ruleset_index]
        if sound_upper < score:
            raise ValueError(f"{ruleset}: sound upper is below the selected score")
        rows.append(
            {
                "index": ruleset_index,
                "ruleset": ruleset,
                "score": score,
                "sound_upper": sound_upper,
                "proof_complete": score == sound_upper,
                "count_index": int(winner["count_index"]),
                "counts": [int(value) for value in winner["counts"]],
                "score_breakdown": [
                    int(value) for value in winner["score_breakdown"]
                ],
                "source_stage": winner["source_stage"],
                "states_evaluated": int(winner["states_evaluated"]),
                "tokens": _expanded_tokens(winner),
            }
        )

    holistic_score = max(row["score"] for row in rows)
    holistic_upper = max(row["sound_upper"] for row in rows)
    leaders = [row["ruleset"] for row in rows if row["score"] == holistic_score]
    catalog = {
        "schema": CATALOG_SCHEMA,
        "complete": True,
        "deep_validation": True,
        "token_count": rules.TOKEN_COUNT,
        "count_cap": rules.COUNT_CAP,
        "ruleset_count": len(RULESETS),
        "count_vector_count": len(COUNT_VECTORS),
        "validated_cells": validated_cells,
        "incumbent_holistic_maximum": holistic_score,
        "incumbent_holistic_rulesets": leaders,
        "holistic_sound_upper": holistic_upper,
        "holistic_gap": holistic_upper - holistic_score,
        "proof_complete_rulesets": sum(row["proof_complete"] for row in rows),
        "results": rows,
    }
    atlas = _atlas(catalog)

    _write_json_atomic(output_json, catalog)
    _write_text_atomic(
        output_markdown,
        _markdown(catalog, output_json, atlas_outputs),
    )
    for atlas_output in atlas_outputs:
        _write_json_atomic(atlas_output, atlas)

    delivery = {
        "schema": DELIVERY_SCHEMA,
        "complete": True,
        "deep_validation": True,
        "validated_cells": validated_cells,
        "ruleset_count": len(RULESETS),
        "count_vector_count": len(COUNT_VECTORS),
        "incumbent_holistic_maximum": holistic_score,
        "incumbent_holistic_rulesets": leaders,
        "holistic_sound_upper": holistic_upper,
        "proof_complete_rulesets": catalog["proof_complete_rulesets"],
        "catalog_json": str(output_json),
        "catalog_markdown": str(output_markdown),
        "atlas_outputs": [str(path) for path in atlas_outputs],
    }
    _write_json_atomic(delivery_summary, delivery)
    return delivery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_directory", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--atlas-output", type=Path, action="append", default=[])
    parser.add_argument("--delivery-summary", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    delivery = generate_report(
        args.catalog_directory,
        args.output_json,
        args.output_markdown,
        args.atlas_output,
        args.delivery_summary,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(delivery, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
