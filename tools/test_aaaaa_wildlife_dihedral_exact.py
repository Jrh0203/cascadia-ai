from __future__ import annotations

import unittest

from ortools.sat.python import cp_model

from tools.aaaaa_wildlife_dihedral_exact import DIHEDRAL_TRANSFORMS, build_model
from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS


def transform_tokens(
    tokens: list[dict[str, int | str]],
    transform: tuple[tuple[int, int], tuple[int, int]],
) -> list[dict[str, int | str]]:
    (qq, qr), (rq, rr) = transform
    return [
        {
            "q": qq * int(row["q"]) + qr * int(row["r"]),
            "r": rq * int(row["q"]) + rr * int(row["r"]),
            "wildlife": str(row["wildlife"]),
        }
        for row in tokens
    ]


class AaaaaWildlifeDihedralExactTests(unittest.TestCase):
    def fixed_status(self, tokens: list[dict[str, int | str]]) -> int:
        model, _ = build_model(
            (6, 4, 6, 0, 4),
            0,
            enforce_connectivity=True,
            initial_tokens=tokens,
            fix_initial_tokens=True,
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_search_workers = 1
        return solver.solve(model)

    def test_some_dihedral_copy_of_known_optimum_is_representable(self) -> None:
        statuses = [
            self.fixed_status(transform_tokens(KNOWN_INCUMBENT_TOKENS, transform))
            for transform in DIHEDRAL_TRANSFORMS
        ]
        self.assertIn(cp_model.OPTIMAL, statuses)

    def test_transform_set_has_twelve_unique_images(self) -> None:
        images = {
            tuple(
                (int(row["q"]), int(row["r"]))
                for row in transform_tokens(
                    [{"q": 2, "r": 1, "wildlife": "fox"}], transform
                )
            )
            for transform in DIHEDRAL_TRANSFORMS
        }
        self.assertEqual(12, len(images))


if __name__ == "__main__":
    unittest.main()
