from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_hawk_packing_bound import SCREEN_CASES, second_hawk_score


class AaaaaWildlifeHawkPackingBoundTests(unittest.TestCase):
    def test_second_hawk_scores(self) -> None:
        self.assertEqual([second_hawk_score(count) for count in range(1, 7)], [0, 0, 2, 5, 8, 11])

    def test_screen_cases_are_distinct_hawk_vectors(self) -> None:
        counts = [row for row, _ in SCREEN_CASES]
        self.assertEqual(len(set(counts)), len(counts))
        self.assertTrue(all(row[3] > 0 for row in counts))

    def test_single_salmon_cases_are_retained(self) -> None:
        self.assertEqual(sum(counts[2] == 1 for counts, _ in SCREEN_CASES), 4)

    def test_one_loss_cases_are_the_four_salmon_tail(self) -> None:
        one_loss = [(counts, target) for counts, target in SCREEN_CASES if counts[2] == 4]
        self.assertEqual(
            one_loss,
            [((4, 6, 4, 2, 4), 65), ((3, 5, 4, 3, 5), 63)],
        )


if __name__ == "__main__":
    unittest.main()
