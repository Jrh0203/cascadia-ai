from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_gap_one_salmon_bound import (
    COUNTS,
    INTERACTION_DISTANCE,
    TARGET,
    hex_distance,
    joint_split_salmon_shapes,
)


class AaaaaWildlifeGapOneSalmonBoundTests(unittest.TestCase):
    def test_fixed_case(self) -> None:
        self.assertEqual(COUNTS, (3, 6, 3, 3, 5))
        self.assertEqual(TARGET, 62)

    def test_joint_shapes_have_one_pair_and_one_separate_singleton(self) -> None:
        shapes = joint_split_salmon_shapes()
        self.assertGreater(len(shapes), 1)
        for shape in shapes:
            degrees = sorted(
                sum(hex_distance(cell, other) == 1 for other in shape) for cell in shape
            )
            self.assertEqual(degrees, [0, 1, 1])
        self.assertEqual(INTERACTION_DISTANCE, 7)


if __name__ == "__main__":
    unittest.main()
