from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules
from tools.all_wildlife_exact import solve_counts
from tools.test_all_wildlife_rules import random_connected_board


@pytest.mark.parametrize(
    "ruleset",
    ["AAAAA", "BBBBB", "CCCCC", "DDDDD", "ABCDC", "DCBAD"],
)
def test_fixed_board_certificate_equals_independent_score(ruleset: str) -> None:
    board = random_connected_board(202)
    expected = all_wildlife_rules.score_tokens(board, ruleset)
    result = solve_counts(
        ruleset,
        (4, 4, 4, 4, 4),
        0,
        time_limit_seconds=10,
        workers=1,
        initial_tokens=board,
        fix_initial_tokens=True,
    )
    assert result.status == "OPTIMAL"
    assert result.objective == sum(expected)
    assert result.score_breakdown == expected


def test_known_cbddb_84_board() -> None:
    path = Path("docs/v3/evidence/cbddb_wildlife_candidates_deep_2026-07-23.json")
    if not path.exists():
        pytest.skip("local CBDDB candidate evidence is not present")
    payload = json.loads(path.read_text())
    candidate = max(payload["candidates"], key=lambda row: row["score"])
    result = solve_counts(
        "CBDDB",
        tuple(candidate["counts"]),
        0,
        time_limit_seconds=10,
        workers=1,
        initial_tokens=candidate["tokens"],
        fix_initial_tokens=True,
    )
    assert result.status == "OPTIMAL"
    assert result.objective == 84
    assert result.score_breakdown == (18, 0, 12, 27, 27)


def test_fixed_board_feasibility_mode_returns_real_certificate_score() -> None:
    board = random_connected_board(202)
    result = solve_counts(
        "AAAAA",
        (4, 4, 4, 4, 4),
        20,
        time_limit_seconds=10,
        workers=1,
        initial_tokens=board,
        fix_initial_tokens=True,
        maximize=False,
    )
    assert result.status == "OPTIMAL"
    assert result.objective is not None and result.objective >= 20
    assert result.score_breakdown is not None
    assert sum(result.score_breakdown) >= result.objective


def test_fixed_board_score_profile_table_preserves_exact_objective() -> None:
    board = random_connected_board(303)
    expected = all_wildlife_rules.score_tokens(board, "CBDDB")
    result = solve_counts(
        "CBDDB",
        (4, 4, 4, 4, 4),
        0,
        time_limit_seconds=10,
        workers=1,
        initial_tokens=board,
        fix_initial_tokens=True,
        use_score_profile_table=True,
    )
    assert result.status == "OPTIMAL"
    assert result.objective == sum(expected)
    assert result.score_breakdown == expected
