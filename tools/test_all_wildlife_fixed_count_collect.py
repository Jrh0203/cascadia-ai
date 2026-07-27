from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_fixed_count_collect import (
    SCHEMA_V1,
    SCHEMA_V2,
    TOTAL_CELLS,
    _legacy_chunk_needs_elk_d_regeneration,
    collect,
)


def _chunk(path: Path) -> None:
    counts = rules.count_vectors()[0]
    tokens = []
    species_indices = [
        species
        for species, count in enumerate(counts)
        for _ in range(count)
    ]
    for q, species in enumerate(species_indices):
        tokens.append([q, 0, species])
    expanded = [
        {"q": q, "r": r, "wildlife": rules.SPECIES[species]}
        for q, r, species in tokens
    ]
    breakdown = list(rules.score_tokens(expanded, "AAAAA"))
    score = sum(breakdown)
    upper = rules.count_upper(counts, "AAAAA")
    payload = {
        "schema": "all-wildlife-fixed-count-candidates-v1",
        "token_count": 20,
        "count_cap": 6,
        "total_cells": TOTAL_CELLS,
        "range_start": 0,
        "range_end": 1,
        "seed": 17,
        "threads": 1,
        "restarts_per_cell": 1,
        "iterations_per_restart": 10,
        "elapsed_seconds": 0.1,
        "candidates": [
            {
                "cell_index": 0,
                "ruleset_index": 0,
                "count_index": 0,
                "ruleset": "AAAAA",
                "count_upper": upper,
                "score": score,
                "score_breakdown": breakdown,
                "counts": list(counts),
                "upper_bound_matched": score == upper,
                "states_evaluated": 11,
                "tokens": tokens,
            }
        ],
    }
    path.write_text(json.dumps(payload))


def test_collect_validates_incremental_chunk_deeply(tmp_path: Path) -> None:
    _chunk(tmp_path / "chunk_00000.json")

    summary = collect([tmp_path], chunk_size=1, deep=True)

    assert summary["completed_cells"] == 1
    assert summary["completed_chunks"] == 1
    assert not summary["complete"]
    assert summary["first_missing_chunks"][0] == 1
    assert summary["seed"] == 17
    assert summary["canonical_upper_bound_matches"] in {0, 1}


def test_collect_accepts_current_chunk_schema(tmp_path: Path) -> None:
    path = tmp_path / "chunk_00000.json"
    _chunk(path)
    payload = json.loads(path.read_text())
    payload["schema"] = SCHEMA_V2
    path.write_text(json.dumps(payload))

    assert collect([tmp_path], chunk_size=1, deep=True)["completed_cells"] == 1


def test_legacy_elk_d_chunks_require_regeneration() -> None:
    elk_d_start = 192 * len(rules.count_vectors())

    assert not _legacy_chunk_needs_elk_d_regeneration(SCHEMA_V1, 0, 1)
    assert _legacy_chunk_needs_elk_d_regeneration(
        SCHEMA_V1,
        elk_d_start,
        elk_d_start + 1,
    )
    assert not _legacy_chunk_needs_elk_d_regeneration(
        SCHEMA_V2,
        elk_d_start,
        elk_d_start + 1,
    )


def test_collect_rejects_duplicate_chunk_across_directories(tmp_path: Path) -> None:
    _chunk(tmp_path / "chunk_00000.json")

    with pytest.raises(ValueError, match="duplicate"):
        collect([tmp_path, tmp_path], chunk_size=1)


def test_collect_rejects_score_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "chunk_00000.json"
    _chunk(path)
    payload = json.loads(path.read_text())
    payload["candidates"][0]["score"] += 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="metadata mismatch"):
        collect([tmp_path], chunk_size=1)


def test_collect_accepts_generator_upper_looser_than_current_python_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunk_00000.json"
    _chunk(path)
    payload = json.loads(path.read_text())
    candidate = payload["candidates"][0]
    candidate["count_upper"] += 4
    candidate["upper_bound_matched"] = candidate["score"] == candidate["count_upper"]
    path.write_text(json.dumps(payload))

    summary = collect([tmp_path], chunk_size=1, deep=True)

    assert summary["completed_cells"] == 1
