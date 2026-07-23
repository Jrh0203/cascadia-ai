from __future__ import annotations

import unittest

from ortools.sat.python import cp_model

from tools.aaaaa_wildlife_center_exact import CENTER_RADIUS, build_model
from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS


class AaaaaWildlifeCenterExactTests(unittest.TestCase):
    def solve_fixed(
        self,
        counts: tuple[int, int, int, int, int],
        tokens: list[dict[str, int | str]],
    ) -> int:
        model, _ = build_model(
            counts,
            0,
            enforce_connectivity=True,
            initial_tokens=tokens,
            fix_initial_tokens=True,
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_search_workers = 1
        return solver.solve(model)

    def test_known_optimum_is_representable(self) -> None:
        self.assertEqual(
            cp_model.OPTIMAL,
            self.solve_fixed((6, 4, 6, 0, 4), KNOWN_INCUMBENT_TOKENS),
        )

    def test_twenty_token_path_at_maximum_graph_radius_is_representable(self) -> None:
        tokens = []
        species = ("bear", "elk", "salmon", "hawk", "fox")
        for q in range(20):
            tokens.append({"q": q, "r": 0, "wildlife": species[q // 4]})
        self.assertEqual(CENTER_RADIUS, 10)
        self.assertEqual(cp_model.OPTIMAL, self.solve_fixed((4, 4, 4, 4, 4), tokens))


if __name__ == "__main__":
    unittest.main()
