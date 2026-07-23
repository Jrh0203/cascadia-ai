from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_zero_hawk_bound import (
    elk_partitions_by_score,
    maximum_bear_structure,
    second_salmon_score,
    unbranched_shapes,
)


class AaaaaWildlifeZeroHawkBoundTests(unittest.TestCase):
    def test_unbranched_shape_counts(self) -> None:
        self.assertEqual([len(unbranched_shapes(n)) for n in range(1, 7)], [1, 1, 3, 4, 10, 25])

    def test_maximum_bear_structures(self) -> None:
        self.assertEqual(maximum_bear_structure(3), (4, 1, 1))
        self.assertEqual(maximum_bear_structure(4), (11, 2, 0))
        self.assertEqual(maximum_bear_structure(6), (19, 3, 0))

    def test_elk_score_partitions(self) -> None:
        self.assertEqual(set(elk_partitions_by_score(6)[18]), {(4, 2), (3, 3)})
        self.assertEqual(set(elk_partitions_by_score(6)[17]), {(4, 1, 1)})

    def test_second_salmon_scores(self) -> None:
        self.assertEqual([second_salmon_score(n) for n in range(2, 7)], [4, 7, 10, 14, 18])


if __name__ == "__main__":
    unittest.main()
