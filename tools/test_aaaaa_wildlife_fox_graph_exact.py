from __future__ import annotations

import unittest

from ortools.sat.python import cp_model

from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS
from tools.aaaaa_wildlife_fox_graph_exact import (
    _connected_graph_codes,
    build_model,
    fox_graph_rows,
    minimum_nonisolated_foxes,
)


class AaaaaWildlifeFoxGraphExactTests(unittest.TestCase):
    def test_connected_polyhex_graph_catalog_is_pinned(self) -> None:
        self.assertEqual([1, 2, 4, 8, 22], [len(_connected_graph_codes(n)) for n in range(2, 7)])
        self.assertLess(len(fox_graph_rows(6, 0)), 1 << 15)
        self.assertTrue(fox_graph_rows(6, 6))

    def test_minimum_nonisolated_bound(self) -> None:
        self.assertEqual(minimum_nonisolated_foxes((3, 6, 6, 0, 5), 62), 5)
        self.assertEqual(minimum_nonisolated_foxes((6, 1, 5, 2, 6), 68), 2)

    def test_known_optimal_board_remains_feasible(self) -> None:
        model, variables = build_model(
            (6, 4, 6, 0, 4),
            68,
            initial_tokens=KNOWN_INCUMBENT_TOKENS,
            fix_initial_tokens=True,
        )
        self.assertEqual(model.validate(), "")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_search_workers = 1
        self.assertEqual(solver.solve(model), cp_model.OPTIMAL)
        self.assertEqual(solver.value(variables.total_score), 68)


if __name__ == "__main__":
    unittest.main()
