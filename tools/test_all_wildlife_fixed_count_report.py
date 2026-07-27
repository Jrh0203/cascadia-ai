from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_fixed_count_report as reporting

COUNT_VECTORS = [(6, 6, 6, 1, 1), (6, 6, 5, 2, 1)]
RULESETS = ["AAAAA", "AAAAB"]


def _tokens(counts: tuple[int, ...]) -> list[list[int]]:
    return [
        [position, 0, species]
        for species, count in enumerate(counts)
        for position in range(sum(counts[:species]), sum(counts[: species + 1]))
    ]


def _candidate(
    cell_index: int,
    *,
    score: int,
    upper: int,
    source: str,
) -> dict:
    ruleset_index, count_index = divmod(cell_index, len(COUNT_VECTORS))
    return {
        "cell_index": cell_index,
        "ruleset_index": ruleset_index,
        "count_index": count_index,
        "ruleset": RULESETS[ruleset_index],
        "counts": list(COUNT_VECTORS[count_index]),
        "score": score,
        "score_breakdown": [score, 0, 0, 0, 0],
        "count_upper": upper,
        "upper_bound_matched": score == upper,
        "states_evaluated": 100 + cell_index,
        "tokens": _tokens(COUNT_VECTORS[count_index]),
        "source_stage": source,
    }


def _catalog_directory(path: Path, *, deep: bool = True) -> Path:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "total_cells": 4,
                "deep_source_validation": deep,
                "deep_output_validation": deep,
                "validated_output_cells": 4,
                "stages": [{"name": "shallow"}, {"name": "production"}],
            }
        )
    )
    candidates = [
        _candidate(0, score=5, upper=10, source="shallow"),
        _candidate(1, score=7, upper=9, source="production"),
        _candidate(2, score=9, upper=9, source="production"),
        _candidate(3, score=8, upper=8, source="shallow"),
    ]
    (path / "chunk_00000.json").write_text(
        json.dumps(
            {
                "schema": reporting.BEST_CHUNK_SCHEMA,
                "token_count": 20,
                "count_cap": 6,
                "total_cells": 4,
                "range_start": 0,
                "range_end": 4,
                "source_stages": ["shallow", "production"],
                "candidates": candidates,
            }
        )
    )
    return path


def test_report_reduces_every_count_to_a_validated_ruleset_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(reporting, "COUNT_VECTORS", COUNT_VECTORS)
    monkeypatch.setattr(reporting, "RULESETS", RULESETS)
    monkeypatch.setattr(reporting, "TOTAL_CELLS", 4)
    validated = []
    monkeypatch.setattr(
        reporting,
        "_validate_candidate",
        lambda candidate, cell_index, *, deep: validated.append(
            (candidate["score"], cell_index, deep)
        ),
    )
    source = _catalog_directory(tmp_path / "best")
    output_json = tmp_path / "rulesets.json"
    output_markdown = tmp_path / "rulesets.md"
    atlas_one = tmp_path / "atlas-one.json"
    atlas_two = tmp_path / "atlas-two.json"
    delivery_path = tmp_path / "delivery.json"

    delivery = reporting.generate_report(
        source,
        output_json,
        output_markdown,
        [atlas_one, atlas_two],
        delivery_path,
        chunk_size=4,
    )

    catalog = json.loads(output_json.read_text())
    assert validated == [
        (5, 0, True),
        (7, 1, True),
        (9, 2, True),
        (8, 3, True),
    ]
    assert [row["score"] for row in catalog["results"]] == [7, 9]
    assert [row["sound_upper"] for row in catalog["results"]] == [10, 9]
    assert [row["proof_complete"] for row in catalog["results"]] == [False, True]
    assert catalog["incumbent_holistic_maximum"] == 9
    assert catalog["incumbent_holistic_rulesets"] == ["AAAAB"]
    assert catalog["holistic_sound_upper"] == 10
    assert delivery["validated_cells"] == 4
    assert delivery["complete"]
    assert json.loads(delivery_path.read_text()) == delivery
    assert atlas_one.read_bytes() == atlas_two.read_bytes()
    atlas = json.loads(atlas_one.read_text())
    assert atlas["leaders"] == ["AAAAB"]
    assert [row["id"] for row in atlas["rows"]] == RULESETS
    markdown = output_markdown.read_text()
    assert "Best score found: **9**" in markdown
    assert "### AAAAB — 9 points" in markdown
    assert "| `AAAAA` | 7 | 10 | 3 | no |" in markdown


def test_report_rejects_a_source_without_deep_output_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(reporting, "TOTAL_CELLS", 4)
    source = _catalog_directory(tmp_path / "best", deep=False)

    with pytest.raises(ValueError, match="not deeply validated"):
        reporting.generate_report(
            source,
            tmp_path / "rulesets.json",
            tmp_path / "rulesets.md",
            [],
            tmp_path / "delivery.json",
            chunk_size=4,
        )
