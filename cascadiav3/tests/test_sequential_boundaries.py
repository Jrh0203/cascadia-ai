"""Validation tests for the group-sequential boundary solver.

Two anchor classes: (a) analytic identities — a single look must give
the fixed-N critical value, and the O'Brien-Fleming-like first-look
boundary is exactly z_{alpha/2}/sqrt(t_1); (b) an end-to-end fixed-seed
Monte Carlo check that the whole boundary schedule realizes the planned
overall type-I error under H0.
"""

import math
import random
import unittest

from cascadiav3.sequential_boundaries import (
    NO_EXIT_BOUNDARY,
    boundary_nominal_alpha,
    normal_cdf,
    normal_quantile,
    obrien_fleming_spending,
    pocock_spending,
    sequential_boundaries,
)

CAMPAIGN_FRACTIONS = [0.4, 0.6, 0.8, 1.0]


def monte_carlo_type_one_error(
    fractions: list[float], boundaries: list[float], paths: int, seed: int = 0
) -> float:
    rng = random.Random(seed)
    rejections = 0
    for _ in range(paths):
        score = 0.0
        previous = 0.0
        for fraction, boundary in zip(fractions, boundaries, strict=True):
            score += rng.gauss(0.0, math.sqrt(fraction - previous))
            previous = fraction
            if abs(score / math.sqrt(fraction)) >= boundary:
                rejections += 1
                break
    return rejections / paths


class NormalHelpersTest(unittest.TestCase):
    def test_quantile_inverts_cdf(self) -> None:
        for p in (0.025, 0.5, 0.975, 0.999999):
            self.assertAlmostEqual(normal_cdf(normal_quantile(p)), p, places=9)

    def test_nominal_alpha_of_fixed_n_boundary(self) -> None:
        self.assertAlmostEqual(boundary_nominal_alpha(1.959964), 0.05, places=6)


class SpendingFunctionTest(unittest.TestCase):
    def test_both_functions_exhaust_alpha_at_full_information(self) -> None:
        self.assertAlmostEqual(obrien_fleming_spending(1.0, 0.05), 0.05, places=12)
        self.assertAlmostEqual(pocock_spending(1.0, 0.05), 0.05, places=12)

    def test_obrien_fleming_is_stingy_early(self) -> None:
        self.assertLess(obrien_fleming_spending(0.4, 0.05), 0.005)
        self.assertLess(
            obrien_fleming_spending(0.4, 0.05), pocock_spending(0.4, 0.05)
        )


class SequentialBoundariesTest(unittest.TestCase):
    def test_single_look_recovers_fixed_n_critical_value(self) -> None:
        (boundary,) = sequential_boundaries([1.0], alpha=0.05)
        self.assertAlmostEqual(boundary, 1.959964, places=4)

    def test_first_look_matches_analytic_obf_identity(self) -> None:
        # For O'Brien-Fleming-like spending the first boundary is exactly
        # z_{alpha/2} / sqrt(t_1) — no recursion involved.
        boundaries = sequential_boundaries(CAMPAIGN_FRACTIONS, alpha=0.05)
        self.assertAlmostEqual(boundaries[0], 1.959964 / math.sqrt(0.4), places=3)
        two_look = sequential_boundaries([0.5, 1.0], alpha=0.05)
        self.assertAlmostEqual(two_look[0], 1.959964 / math.sqrt(0.5), places=3)

    def test_campaign_schedule_shape(self) -> None:
        boundaries = sequential_boundaries(CAMPAIGN_FRACTIONS, alpha=0.05)
        self.assertEqual(len(boundaries), 4)
        for earlier, later in zip(boundaries, boundaries[1:], strict=False):
            self.assertGreater(earlier, later)
        # The whole point of OBF spending: the final look stays close to
        # the fixed-N 1.96 (the price of four looks is a few percent).
        self.assertGreater(boundaries[-1], 1.96)
        self.assertLess(boundaries[-1], 2.10)

    def test_monte_carlo_overall_type_one_error(self) -> None:
        boundaries = sequential_boundaries(CAMPAIGN_FRACTIONS, alpha=0.05)
        realized = monte_carlo_type_one_error(
            CAMPAIGN_FRACTIONS, boundaries, paths=400_000
        )
        self.assertAlmostEqual(realized, 0.05, delta=0.0035)

    def test_pocock_boundaries_are_roughly_flat(self) -> None:
        boundaries = sequential_boundaries(
            [0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="pocock"
        )
        self.assertLess(max(boundaries) - min(boundaries), 0.2)
        realized = monte_carlo_type_one_error(
            [0.25, 0.5, 0.75, 1.0], boundaries, paths=400_000, seed=1
        )
        self.assertAlmostEqual(realized, 0.05, delta=0.0035)

    def test_no_exit_boundary_when_increment_spend_is_zero(self) -> None:
        # A spending function that is flat between two looks allocates a
        # zero increment to the middle look: it must be marked
        # unreachable rather than produce a bogus finite boundary.
        from cascadiav3.sequential_boundaries import SPENDING_FUNCTIONS

        SPENDING_FUNCTIONS["flat_test"] = lambda fraction, alpha: (
            alpha if fraction >= 1.0 else alpha / 10.0
        )
        try:
            boundaries = sequential_boundaries(
                [0.4, 0.6, 1.0], alpha=0.05, spending="flat_test"
            )
        finally:
            del SPENDING_FUNCTIONS["flat_test"]
        self.assertNotEqual(boundaries[0], NO_EXIT_BOUNDARY)
        self.assertEqual(boundaries[1], NO_EXIT_BOUNDARY)
        self.assertLess(boundaries[2], NO_EXIT_BOUNDARY)

    def test_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sequential_boundaries([0.5, 0.4, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sequential_boundaries([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "at least one"):
            sequential_boundaries([])
        with self.assertRaisesRegex(ValueError, "alpha"):
            sequential_boundaries([1.0], alpha=1.5)
        with self.assertRaisesRegex(ValueError, "unknown spending"):
            sequential_boundaries([1.0], spending="hockey_stick")


if __name__ == "__main__":
    unittest.main()
