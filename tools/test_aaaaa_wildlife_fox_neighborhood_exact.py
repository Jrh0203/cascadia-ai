import itertools

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_fox_neighborhood_exact import (
    _connected_state_codes,
    _near_connected_shapes,
    _state_code_for_order,
    build_model,
    neighborhood_rows,
)


def _row_for(coords: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    row = []
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            dq = coords[right][0] - coords[left][0]
            dr = coords[right][1] - coords[left][1]
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            row.append(distance if distance <= 2 else 0)
    return tuple(row)


def test_radius_two_shape_and_graph_catalogs_are_nonempty() -> None:
    assert [len(_near_connected_shapes(size)) for size in range(1, 6)] == [
        1,
        3,
        15,
        127,
        1338,
    ]
    assert [len(_connected_state_codes(size)) for size in range(1, 5)] == [
        1,
        2,
        6,
        28,
    ]


def test_tables_reject_an_impossible_four_fox_clique() -> None:
    impossible = (1,) * 6
    assert impossible not in neighborhood_rows(4)
    actual = ((0, 0), (1, 0), (0, 1), (-1, 1))
    assert _row_for(actual) in neighborhood_rows(4)


def test_connected_state_code_is_label_invariant() -> None:
    coords = ((0, 0), (1, 0), (0, 1), (2, -1))
    codes = {
        _state_code_for_order(coords, order)
        for order in itertools.permutations(range(4))
    }
    assert min(codes) in _connected_state_codes(4)


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
