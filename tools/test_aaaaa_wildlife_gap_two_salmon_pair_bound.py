from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_gap_two_salmon_pair_bound import (
    SCREEN_CASES,
    split_singleton_shapes,
)


class AaaaaWildlifeGapTwoSalmonPairBoundTests(unittest.TestCase):
    def test_screen_cases_have_two_salmon(self) -> None:
        self.assertEqual(len(SCREEN_CASES), 4)
        self.assertTrue(all(counts[2] == 2 for counts, _ in SCREEN_CASES))

    def test_split_shapes_are_nonadjacent_singletons(self) -> None:
        shapes = split_singleton_shapes()
        self.assertGreater(len(shapes), 1)
        self.assertTrue(all(len(shape) == 2 for shape in shapes))


if __name__ == "__main__":
    unittest.main()
