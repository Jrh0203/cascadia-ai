from __future__ import annotations

import itertools
import unittest

from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_gap_two_salmon_pair_bound import SCREEN_CASES, split_singleton_shapes
from tools.aaaaa_wildlife_motif_certificate import adjacent


class SplitSalmonFeasibilityTest(unittest.TestCase):
    def test_split_shapes_cover_all_interacting_separations_and_far_case(self) -> None:
        shapes = split_singleton_shapes()
        self.assertEqual(19, len(shapes))
        for shape in shapes:
            self.assertEqual(2, len(shape))
            self.assertFalse(any(adjacent(*pair) for pair in itertools.combinations(shape, 2)))

    def test_only_top_two_elk_scores_can_reach_registered_targets(self) -> None:
        expected = {
            5: {15: ((4, 1),), 14: ((3, 2),)},
            6: {18: ((4, 2), (3, 3)), 17: ((4, 1, 1),)},
        }
        for counts, _target in SCREEN_CASES:
            elk = counts[1]
            threshold = 14 if elk == 5 else 17
            observed = {
                score: rows
                for score, rows in zero.elk_partitions_by_score(elk).items()
                if score >= threshold
            }
            self.assertEqual(expected[elk], observed)


if __name__ == "__main__":
    unittest.main()
