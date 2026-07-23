from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_gap_one_salmon_bound import COUNTS, TARGET


class AaaaaWildlifeGapOneSalmonBoundTests(unittest.TestCase):
    def test_fixed_case(self) -> None:
        self.assertEqual(COUNTS, (3, 6, 3, 3, 5))
        self.assertEqual(TARGET, 62)


if __name__ == "__main__":
    unittest.main()
