from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_fox_witness_exact import (
    build_model,
    conditional_ring_triple_rows,
    ring_triple_rows,
)


def test_ring_triple_catalog_is_exact_and_conditional() -> None:
    assert ring_triple_rows()
    assert all(0 not in row for row in ring_triple_rows())
    rows = conditional_ring_triple_rows()
    assert (1, 1, 1, *ring_triple_rows()[0]) in rows
    assert (1, 1, 1, 1, 1, 1) not in rows
    assert (1, 1, 1, 0, 0, 0) not in rows
    assert (1, 1, 0, 0, 0, 0) in rows


def test_known_optimal_board_remains_feasible() -> None:
    model, variables = build_model(
        (6, 4, 6, 0, 4),
        68,
        initial_tokens=base.KNOWN_INCUMBENT_TOKENS,
        fix_initial_tokens=True,
    )
    assert model.validate() == ""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(variables.total_score) == 68
