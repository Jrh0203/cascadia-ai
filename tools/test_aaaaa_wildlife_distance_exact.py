from __future__ import annotations

import copy
import unittest

from ortools.sat.python import cp_model

from tools.aaaaa_wildlife_distance_exact import build_model
from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS


class AaaaaWildlifeDistanceExactTests(unittest.TestCase):
    def solve_fixed(self, tokens: list[dict[str, int | str]], connected: bool) -> int:
        model, _ = build_model(
            (6, 4, 6, 0, 4),
            0,
            enforce_connectivity=connected,
            initial_tokens=tokens,
            fix_initial_tokens=True,
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_search_workers = 1
        return solver.solve(model)

    def test_connected_production_witness_is_representable(self) -> None:
        self.assertEqual(
            self.solve_fixed(KNOWN_INCUMBENT_TOKENS, connected=True),
            cp_model.OPTIMAL,
        )

    def test_disconnected_fixed_board_is_rejected_only_when_required(self) -> None:
        disconnected = copy.deepcopy(KNOWN_INCUMBENT_TOKENS)
        disconnected[-1]["q"] = 10
        disconnected[-1]["r"] = 0
        self.assertEqual(self.solve_fixed(disconnected, connected=False), cp_model.OPTIMAL)
        self.assertEqual(self.solve_fixed(disconnected, connected=True), cp_model.INFEASIBLE)


if __name__ == "__main__":
    unittest.main()
