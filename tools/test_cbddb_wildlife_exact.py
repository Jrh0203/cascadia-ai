from __future__ import annotations

import random
import unittest

from ortools.sat.python import cp_model

from tools.cbddb_wildlife_exact import (
    BEAR_C_STANDALONE,
    ELK_B_STANDALONE,
    SALMON_D_STANDALONE,
    SPECIES,
    build_model,
    count_relaxation,
    count_vectors,
    score_tokens,
)


def tokens(entries: list[tuple[int, int, str]]) -> list[dict[str, int | str]]:
    return [{"q": q, "r": r, "wildlife": wildlife} for q, r, wildlife in entries]


class CBDDBWildlifeExactTests(unittest.TestCase):
    def test_count_space_and_standalone_tables(self) -> None:
        self.assertEqual(len(count_vectors()), 826)
        self.assertEqual(BEAR_C_STANDALONE, (0, 2, 5, 8, 10, 13, 18))
        self.assertEqual(ELK_B_STANDALONE, (0, 2, 5, 9, 13, 15, 18))
        self.assertEqual(SALMON_D_STANDALONE, (0, 0, 0, 13, 16, 19, 26))
        for counts, bound in count_vectors():
            self.assertEqual(bound, count_relaxation(counts))

    def test_bear_c_components_and_set_bonus(self) -> None:
        entries = [
            (0, 0, "bear"),
            (3, 0, "bear"),
            (4, 0, "bear"),
            (0, 3, "bear"),
            (1, 3, "bear"),
            (0, 4, "bear"),
        ]
        self.assertEqual(score_tokens(tokens(entries))[0], 18)
        oversize = [(0, 0, "bear"), (1, 0, "bear"), (0, 1, "bear"), (1, 1, "bear")]
        self.assertEqual(score_tokens(tokens(oversize))[0], 0)

    def test_elk_b_shape_packing(self) -> None:
        rhombus = [(0, 0, "elk"), (1, 0, "elk"), (0, 1, "elk"), (1, 1, "elk")]
        self.assertEqual(score_tokens(tokens(rhombus))[1], 13)
        line = [(0, 0, "elk"), (1, 0, "elk"), (2, 0, "elk")]
        self.assertEqual(score_tokens(tokens(line))[1], 7)
        two_triangles = [*rhombus[:3], (5, 0, "elk"), (6, 0, "elk"), (5, 1, "elk")]
        self.assertEqual(score_tokens(tokens(two_triangles))[1], 18)

    def test_salmon_d_scores_only_valid_runs_and_unique_neighbors(self) -> None:
        entries = [
            (0, 0, "salmon"),
            (1, 0, "salmon"),
            (2, 0, "salmon"),
            (0, 1, "bear"),
            (1, 1, "bear"),
            (2, 1, "elk"),
        ]
        self.assertEqual(score_tokens(tokens(entries))[2], 6)
        branched = [
            (0, 0, "salmon"),
            (1, 0, "salmon"),
            (-1, 0, "salmon"),
            (0, 1, "salmon"),
        ]
        self.assertEqual(score_tokens(tokens(branched))[2], 0)

    def test_hawk_d_line_of_sight_uses_matching_and_distinct_between_types(self) -> None:
        entries = [
            (0, 0, "hawk"),
            (4, 0, "hawk"),
            (1, 0, "bear"),
            (2, 0, "bear"),
            (3, 0, "elk"),
            (0, 2, "hawk"),
            (4, 2, "hawk"),
            (1, 2, "salmon"),
            (2, 2, "fox"),
            (3, 2, "elk"),
        ]
        self.assertEqual(score_tokens(tokens(entries))[3], 16)
        blocked = [entry for entry in entries if entry[:2] != (2, 0)]
        blocked.append((2, 0, "hawk"))
        self.assertEqual(score_tokens(tokens(blocked))[3], 13)

    def test_fox_b_counts_species_with_at_least_two_neighbors(self) -> None:
        entries = [
            (0, 0, "fox"),
            (1, 0, "bear"),
            (0, 1, "bear"),
            (-1, 1, "elk"),
            (-1, 0, "elk"),
            (0, -1, "salmon"),
            (1, -1, "salmon"),
        ]
        self.assertEqual(score_tokens(tokens(entries))[4], 7)

    def test_fixed_board_cp_model_matches_independent_scorer(self) -> None:
        cells = [
            (q, r)
            for q in range(-2, 3)
            for r in range(-2, 3)
            if max(abs(q), abs(r), abs(q + r)) <= 2
        ]
        cells.append((3, 0))
        board = tokens(
            [(q, r, SPECIES[index % len(SPECIES)]) for index, (q, r) in enumerate(cells)]
        )
        expected = sum(score_tokens(board))
        model, variables = build_model(
            (4, 4, 4, 4, 4),
            0,
            initial_tokens=board,
            fix_initial_tokens=True,
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_search_workers = 2
        status = solver.solve(model)
        self.assertEqual(status, cp_model.OPTIMAL)
        self.assertEqual(solver.value(variables.total_score), expected)

    def test_fixed_board_cp_model_matches_varied_random_layouts(self) -> None:
        rng = random.Random(20260723)
        count_cases = [
            (0, 2, 6, 6, 6),
            (6, 0, 6, 6, 2),
            (6, 6, 2, 0, 6),
            (4, 4, 4, 4, 4),
            (6, 4, 3, 2, 5),
        ]
        for case_index, counts in enumerate(count_cases):
            occupied = {(0, 0)}
            while len(occupied) < 20:
                q, r = rng.choice(sorted(occupied))
                dq, dr = rng.choice(((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)))
                occupied.add((q + dq, r + dr))
            wildlife = [
                species
                for species, count in zip(SPECIES, counts, strict=True)
                for _ in range(count)
            ]
            rng.shuffle(wildlife)
            board = tokens(
                [
                    (q, r, species)
                    for (q, r), species in zip(sorted(occupied), wildlife, strict=True)
                ]
            )
            expected = sum(score_tokens(board))
            with self.subTest(case_index=case_index, counts=counts, expected=expected):
                model, variables = build_model(
                    counts,
                    0,
                    initial_tokens=board,
                    fix_initial_tokens=True,
                )
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 30
                solver.parameters.num_search_workers = 2
                status = solver.solve(model)
                self.assertEqual(status, cp_model.OPTIMAL)
                self.assertEqual(solver.value(variables.total_score), expected)


if __name__ == "__main__":
    unittest.main()
