from __future__ import annotations

import unittest

from tools.aaaaa_wildlife_motif_certificate import (
    enumerate_relaxed_superset,
    free_polyhexes,
    optimal_elk_partitions,
    valid_six_salmon_shapes,
)


class AaaaaWildlifeMotifCertificateTests(unittest.TestCase):
    def test_free_polyhex_counts_through_six(self) -> None:
        self.assertEqual(
            [len(free_polyhexes(size)) for size in range(1, 7)],
            [1, 1, 3, 7, 22, 82],
        )

    def test_six_salmon_valid_shape_count(self) -> None:
        self.assertEqual(len(valid_six_salmon_shapes()), 25)

    def test_optimal_six_elk_partitions(self) -> None:
        self.assertEqual(set(optimal_elk_partitions(6)), {(4, 2), (3, 3)})

    def test_relaxed_superset_is_exhausted_without_a_realisation(self) -> None:
        self.assertEqual(
            enumerate_relaxed_superset(),
            {
                "free_polyhexes_size_6": 82,
                "valid_salmon_shapes": 25,
                "fox_boundary_sets_no_isolates": 4623,
                "fox_sets_with_relaxed_bear_coverage": 2355,
                "fox_sets_with_relaxed_bear_and_elk_coverage": 2342,
                "nonoverlapping_relaxed_realisations": 0,
                "infeasible": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
