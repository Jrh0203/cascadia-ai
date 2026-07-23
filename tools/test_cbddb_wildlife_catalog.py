from __future__ import annotations

import unittest
from unittest import mock

from tools.cbddb_wildlife_catalog import render_markdown, solve_one, validate_witness
from tools.cbddb_wildlife_exact import SPECIES


def balanced_board() -> list[dict[str, int | str]]:
    cells = [
        (q, r) for q in range(-2, 3) for r in range(-2, 3) if max(abs(q), abs(r), abs(q + r)) <= 2
    ]
    cells.append((3, 0))
    return [
        {"q": q, "r": r, "wildlife": SPECIES[index % len(SPECIES)]}
        for index, (q, r) in enumerate(cells)
    ]


class CBDDBWildlifeCatalogTests(unittest.TestCase):
    def test_validated_witness_can_be_precertified_at_its_upper_bound(self) -> None:
        board, breakdown = validate_witness((4, 4, 4, 4, 4), balanced_board())
        score = sum(breakdown)
        with mock.patch("tools.cbddb_wildlife_catalog.count_relaxation", return_value=score):
            result = solve_one(
                {
                    "counts": [4, 4, 4, 4, 4],
                    "tokens": board,
                    "solver_workers": 1,
                    "relaxation_time_limit": 1.0,
                    "connected_time_limit": 1.0,
                    "seed": 20260723,
                }
            )
        self.assertTrue(result["proof_complete"])
        self.assertEqual(score, result["optimum"])
        self.assertEqual("witness_matches_count_relaxation", result["proof_method"])

    def test_timeout_is_retained_as_incomplete(self) -> None:
        board, breakdown = validate_witness((4, 4, 4, 4, 4), balanced_board())
        score = sum(breakdown)
        unknown = {
            "model_status": "UNKNOWN",
            "model_score": None,
            "objective": None,
            "best_bound": float(score + 5),
            "wall_seconds": 1.0,
            "branches": 10,
            "conflicts": 2,
            "tokens": [],
        }
        with (
            mock.patch("tools.cbddb_wildlife_catalog.count_relaxation", return_value=score + 5),
            mock.patch("tools.cbddb_wildlife_catalog.solve_counts", return_value=unknown) as solver,
        ):
            result = solve_one(
                {
                    "counts": [4, 4, 4, 4, 4],
                    "tokens": board,
                    "solver_workers": 1,
                    "relaxation_time_limit": 1.0,
                    "connected_time_limit": 1.0,
                    "seed": 20260723,
                }
            )
        self.assertFalse(result["proof_complete"])
        self.assertEqual("incomplete_timeout", result["proof_method"])
        self.assertEqual(2, solver.call_count)

    def test_markdown_reports_holistic_best_and_board(self) -> None:
        board, breakdown = validate_witness((4, 4, 4, 4, 4), balanced_board())
        result = {
            "counts": [4, 4, 4, 4, 4],
            "optimum": sum(breakdown),
            "score_breakdown": breakdown,
            "tokens": board,
            "proof_method": "test_certificate",
            "proof_complete": True,
        }
        markdown = render_markdown(
            {
                "proof_complete": False,
                "completed_count": 1,
                "allocation_count": 826,
                "results": [result],
            }
        )
        self.assertIn("## Holistic maximum", markdown)
        self.assertIn("B/E/S/H/F = `4/4/4/4/4`", markdown)
        self.assertIn("### 001. B/E/S/H/F = 4/4/4/4/4", markdown)


if __name__ == "__main__":
    unittest.main()
