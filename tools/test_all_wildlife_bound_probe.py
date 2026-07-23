from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_bound_probe import run_probe
from tools.all_wildlife_exact import SolveResult


def _tokens() -> list[dict[str, object]]:
    return [
        {
            "q": index,
            "r": 0,
            "wildlife": rules.SPECIES[index % len(rules.SPECIES)],
        }
        for index in range(rules.TOKEN_COUNT)
    ]


def _catalog(path: Path) -> tuple[dict[str, object], tuple[int, ...]]:
    tokens = _tokens()
    breakdown = list(rules.score_tokens(tokens, "AAAAA"))
    counts = (4, 4, 4, 4, 4)
    row = {
        "index": 0,
        "ruleset": "AAAAA",
        "proof_complete": False,
        "optimum": sum(breakdown),
        "score_breakdown": breakdown,
        "counts": list(counts),
        "tokens": tokens,
        "unresolved_counts": [list(counts)],
    }
    payload = {
        "schema": "all-wildlife-optimal-catalog-v1",
        "results": [row],
    }
    path.write_text(json.dumps(payload))
    return row, counts


def _args(catalog: Path, output: Path, counts: tuple[int, ...]) -> argparse.Namespace:
    return argparse.Namespace(
        catalog=catalog,
        index=0,
        counts=[",".join(map(str, counts))],
        max_counts=None,
        output=output,
        time_limit=5.0,
        total_time_limit=5.0,
        workers=1,
        no_connectivity=False,
        resume=False,
    )


def test_unknown_probe_retains_sound_solver_bound(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "probe.json"
    row, counts = _catalog(catalog)
    analytical = rules.count_upper(counts, "AAAAA")
    result = SolveResult(
        status="UNKNOWN",
        objective=None,
        best_bound=analytical - 2,
        elapsed_seconds=5.0,
        branches=12,
        conflicts=3,
        tokens=None,
        score_breakdown=None,
    )
    with patch("tools.all_wildlife_bound_probe.solve_counts", return_value=result):
        assert run_probe(_args(catalog, output, counts)) == 2
    payload = json.loads(output.read_text())
    assert payload["selected_counts"] == [list(counts)]
    assert payload["selected_count_count"] == 1
    assert payload["attempted_count_count"] == 1
    assert payload["attempts"][0]["refined_upper"] == analytical - 2
    assert payload["best_witness"]["score"] == row["optimum"]
    assert payload["sound_upper"] == max(row["optimum"], analytical - 2)


def test_infeasible_probe_can_close_the_last_count(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "probe.json"
    row, counts = _catalog(catalog)
    result = SolveResult(
        status="INFEASIBLE",
        objective=None,
        best_bound=None,
        elapsed_seconds=1.0,
        branches=2,
        conflicts=1,
        tokens=None,
        score_breakdown=None,
    )
    with patch("tools.all_wildlife_bound_probe.solve_counts", return_value=result):
        assert run_probe(_args(catalog, output, counts)) == 0
    payload = json.loads(output.read_text())
    assert payload["proof_complete"]
    assert payload["sound_upper"] == row["optimum"]
    assert payload["remaining_counts"] == []


def test_probe_rejects_count_not_in_base_frontier(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "probe.json"
    _catalog(catalog)
    with pytest.raises(ValueError, match="not unresolved"):
        run_probe(_args(catalog, output, (3, 5, 4, 4, 4)))
