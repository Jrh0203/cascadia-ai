import copy
import unittest
from unittest import mock

from tools.aaaaa_wildlife_catalog import render_markdown, solve_one, validate_witness
from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS


class AaaaaWildlifeCatalogTests(unittest.TestCase):
    def test_known_upper_bound_witness_is_immediately_certified(self) -> None:
        counts = (6, 4, 6, 0, 4)
        tokens, breakdown = validate_witness(counts, KNOWN_INCUMBENT_TOKENS)
        self.assertEqual([19, 13, 20, 0, 16], breakdown)
        result = solve_one(
            {
                "counts": list(counts),
                "tokens": tokens,
                "solver_workers": 1,
                "relaxation_time_limit": 1.0,
                "connected_time_limit": 1.0,
                "seed": 20260722,
            }
        )
        self.assertTrue(result["proof_complete"])
        self.assertEqual(68, result["optimum"])
        self.assertEqual("witness_matches_count_relaxation", result["proof_method"])
        self.assertEqual([], result["attempts"])

    def test_connected_relaxed_witness_advances_without_redundant_solve(self) -> None:
        incumbent = copy.deepcopy(KNOWN_INCUMBENT_TOKENS)
        incumbent[0]["wildlife"], incumbent[1]["wildlife"] = (
            incumbent[1]["wildlife"],
            incumbent[0]["wildlife"],
        )
        relaxed_result = {
            "model_status": "OPTIMAL",
            "model_score": 68,
            "objective": 68,
            "best_bound": 68.0,
            "wall_seconds": 0.01,
            "branches": 1,
            "conflicts": 0,
            "tokens": KNOWN_INCUMBENT_TOKENS,
        }
        with mock.patch(
            "tools.aaaaa_wildlife_catalog.solve_counts", return_value=relaxed_result
        ) as solver:
            result = solve_one(
                {
                    "counts": [6, 4, 6, 0, 4],
                    "tokens": incumbent,
                    "solver_workers": 1,
                    "relaxation_time_limit": 1.0,
                    "connected_time_limit": 1.0,
                    "seed": 20260722,
                }
            )
        self.assertEqual(68, result["optimum"])
        self.assertEqual(1, solver.call_count)
        self.assertFalse(solver.call_args.kwargs["enforce_connectivity"])

    def test_markdown_contains_summary_and_board(self) -> None:
        result = solve_one(
            {
                "counts": [6, 4, 6, 0, 4],
                "tokens": KNOWN_INCUMBENT_TOKENS,
                "solver_workers": 1,
                "relaxation_time_limit": 1.0,
                "connected_time_limit": 1.0,
                "seed": 20260722,
            }
        )
        markdown = render_markdown(
            {
                "proof_complete": False,
                "completed_count": 1,
                "allocation_count": 826,
                "results": [result],
            }
        )
        self.assertIn("| 6 | 4 | 6 | 0 | 4 | 68 | 19/13/20/0/16 |", markdown)
        self.assertIn("### 001. B/E/S/H/F = 6/4/6/0/4", markdown)


if __name__ == "__main__":
    unittest.main()
