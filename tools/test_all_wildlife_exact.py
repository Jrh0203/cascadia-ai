from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules
from tools.all_wildlife_exact import (
    _anchor_centroid_initial_coordinates,
    _dihedral_axial,
    _in_centroid_wedge,
    solve_counts,
    species_tokens,
)
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


def test_every_radius_six_vector_has_a_centroid_wedge_image() -> None:
    for q in range(-6, 7):
        for r in range(-6, 7):
            if max(abs(q), abs(r), abs(q + r)) > 6:
                continue
            assert any(
                _in_centroid_wedge(_dihedral_axial((q, r), transform))
                for transform in range(12)
            )


def test_anchor_centroid_hint_preserves_every_card_score() -> None:
    board = random_connected_board(808)
    counts = (4, 4, 4, 4, 4)
    ordered_species = species_tokens(counts)
    coordinates = _anchor_centroid_initial_coordinates(board, ordered_species)
    canonical = [
        {
            "q": q,
            "r": r,
            "wildlife": all_wildlife_rules.SPECIES[species],
        }
        for (q, r), species in zip(coordinates, ordered_species, strict=True)
    ]
    centroid = (
        sum(q for q, _ in coordinates),
        sum(r for _, r in coordinates),
    )

    assert coordinates[0] == (0, 0)
    assert _in_centroid_wedge(centroid)
    assert coordinates[1:4] == sorted(coordinates[1:4])
    for ruleset in all_wildlife_rules.rulesets():
        assert all_wildlife_rules.score_tokens(canonical, ruleset) == (
            all_wildlife_rules.score_tokens(board, ruleset)
        )
