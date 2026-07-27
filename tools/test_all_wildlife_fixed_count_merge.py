from __future__ import annotations

import json
from pathlib import Path

from tools import all_wildlife_fixed_count_merge as merging


def _stage(path: Path, scores: list[int]) -> None:
    path.mkdir()
    candidates = [
        {
            "cell_index": cell_index,
            "score": score,
            "tokens": [[cell_index, 0, 0]],
        }
        for cell_index, score in enumerate(scores)
    ]
    (path / "chunk_00000.json").write_text(
        json.dumps(
            {
                "range_start": 0,
                "range_end": len(scores),
                "candidates": candidates,
            }
        )
    )


def test_merge_selects_best_stage_per_cell_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shallow = tmp_path / "shallow"
    production = tmp_path / "production"
    output = tmp_path / "best"
    _stage(shallow, [5, 10])
    _stage(production, [6, 9])
    monkeypatch.setattr(merging, "TOTAL_CELLS", 2)
    monkeypatch.setattr(
        merging,
        "collect",
        lambda *_args, **_kwargs: {
            "complete": True,
            "completed_cells": 2,
            "total_cells": 2,
            "restarts_per_cell": 1,
            "iterations_per_restart": 10,
        },
    )

    summary = merging.merge(
        [("shallow", shallow), ("production", production)],
        output,
        chunk_size=2,
    )

    chunk = json.loads((output / "chunk_00000.json").read_text())
    assert [candidate["score"] for candidate in chunk["candidates"]] == [6, 10]
    assert [candidate["source_stage"] for candidate in chunk["candidates"]] == [
        "production",
        "shallow",
    ]
    assert summary["selected_cells_by_stage"] == {
        "shallow": 1,
        "production": 1,
    }
    assert summary["strict_improvements_by_stage"] == {
        "shallow": 1,
        "production": 1,
    }
    assert not summary["deep_source_validation"]
    assert not summary["deep_output_validation"]
    assert summary["validated_output_cells"] == 2
    assert summary["validated_output_chunks"] == 1


def test_winner_breaks_score_ties_by_canonical_tokens_then_stage() -> None:
    options = [
        (0, "shallow", {"score": 7, "tokens": [[1, 0, 0]]}),
        (1, "production", {"score": 7, "tokens": [[0, 0, 0]]}),
    ]

    assert merging._winner(options)[1] == "production"


def test_deep_merge_validates_materialized_winners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shallow = tmp_path / "shallow"
    production = tmp_path / "production"
    output = tmp_path / "best"
    _stage(shallow, [5, 10])
    _stage(production, [6, 9])
    monkeypatch.setattr(merging, "TOTAL_CELLS", 2)
    monkeypatch.setattr(
        merging,
        "collect",
        lambda *_args, **_kwargs: {
            "complete": True,
            "completed_cells": 2,
            "total_cells": 2,
            "restarts_per_cell": 1,
            "iterations_per_restart": 10,
        },
    )
    validated = []
    monkeypatch.setattr(
        merging,
        "_validate_candidate",
        lambda candidate, cell_index, *, deep: validated.append(
            (candidate["score"], cell_index, deep)
        ),
    )

    summary = merging.merge(
        [("shallow", shallow), ("production", production)],
        output,
        chunk_size=2,
        deep=True,
    )

    assert validated == [(6, 0, True), (10, 1, True)]
    assert summary["deep_source_validation"]
    assert summary["deep_output_validation"]
