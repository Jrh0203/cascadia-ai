from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS
from tools.aaaaa_wildlife_motif_relaxation import (
    elk_partitions,
    salmon_component_offsets,
    salmon_partitions,
    solve_relaxation,
)
from tools.aaaaa_wildlife_motif_relaxation_batch import parse_case


class MotifRelaxationTest(unittest.TestCase):
    def test_partitions_cover_every_relevant_score_structure(self) -> None:
        self.assertEqual(9, len(elk_partitions(6)))
        self.assertIn((4, 2), elk_partitions(6))
        self.assertIn((3, 3), elk_partitions(6))
        self.assertEqual(30, len(salmon_partitions(6)))
        self.assertIn((), salmon_partitions(6))
        self.assertIn((4, 2), salmon_partitions(6))
        self.assertIn((3, 2, 1), salmon_partitions(6))

    def test_salmon_shapes_include_all_dihedral_line_orientations(self) -> None:
        self.assertEqual(3, len(salmon_component_offsets(2)))
        self.assertGreater(len(salmon_component_offsets(6)), 200)

    def test_fixed_holistic_witness_is_contained_at_score_68(self) -> None:
        result = solve_relaxation(
            (6, 4, 6, 0, 4),
            68,
            workers=1,
            time_limit=10.0,
            seed=20260723,
            fixed_tokens=KNOWN_INCUMBENT_TOKENS,
        )
        self.assertEqual("OPTIMAL", result["status"])
        self.assertGreaterEqual(
            result["relaxation_non_fox_score"] + result["relaxation_fox_score"], 68
        )

    def test_rejects_target_above_standalone_relaxation(self) -> None:
        with self.assertRaisesRegex(ValueError, "target exceeds"):
            solve_relaxation(
                (6, 4, 6, 0, 4),
                69,
                workers=1,
                time_limit=1.0,
                seed=1,
            )

    def test_batch_case_parser(self) -> None:
        self.assertEqual(((3, 6, 6, 0, 5), 62), parse_case("3,6,6,0,5:62"))


if __name__ == "__main__":
    unittest.main()
