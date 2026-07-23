from __future__ import annotations

import unittest

from ortools.sat.python import cp_model

from tools.aaaaa_wildlife_eager_exact import build_model
from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS


class AaaaaWildlifeEagerExactTests(unittest.TestCase):
    def test_eager_channeling_accepts_known_optimal_board(self) -> None:
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
