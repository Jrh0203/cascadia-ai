from __future__ import annotations

import itertools
import unittest

from tools.aaaaa_wildlife_gap_one_salmon_bound import INTERACTION_DISTANCE, hex_distance
from tools.aaaaa_wildlife_motif_certificate import adjacent, boundary
from tools.aaaaa_wildlife_two_missing_fox_bound import (
    SCREEN_CASES,
    interacting_layouts,
    remote_pair_branch_bound,
)


class TwoMissingFoxBoundTest(unittest.TestCase):
    def test_interacting_layouts_are_complete_canonical_shapes(self) -> None:
        layouts = interacting_layouts()
        self.assertEqual(145, len(layouts))
        self.assertEqual(144, sum(not row.factorized_far_case for row in layouts))
        self.assertEqual(1, sum(row.factorized_far_case for row in layouts))
        for layout in layouts:
            self.assertEqual(2, len(layout.salmon))
            self.assertEqual(2, len(layout.remote_fox_pair))
            self.assertTrue(
                any(adjacent(*pair) for pair in itertools.combinations(layout.salmon, 2))
            )
            self.assertTrue(
                any(
                    adjacent(*pair)
                    for pair in itertools.combinations(layout.remote_fox_pair, 2)
                )
            )
            self.assertTrue(
                layout.remote_fox_pair.isdisjoint(layout.salmon | boundary(layout.salmon))
            )
            distance = min(
                hex_distance(fox, fish)
                for fox in layout.remote_fox_pair
                for fish in layout.salmon
            )
            if layout.factorized_far_case:
                self.assertEqual(INTERACTION_DISTANCE + 1, distance)
            else:
                self.assertIn(distance, range(2, INTERACTION_DISTANCE + 1))

    def test_screen_targets_force_28_fox_points_in_maximum_salmon_branch(self) -> None:
        expected_non_fox = (39, 36, 35, 35)
        for (_counts, target), non_fox in zip(SCREEN_CASES, expected_non_fox, strict=True):
            self.assertEqual(28, target - non_fox)

    def test_rejects_target_that_does_not_force_full_remaining_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "force every non-salmon"):
            remote_pair_branch_bound(SCREEN_CASES[0][0], SCREEN_CASES[0][1] - 1)


if __name__ == "__main__":
    unittest.main()
