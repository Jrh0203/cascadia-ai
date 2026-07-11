"""Contract tests for the CUDA concurrency comparator's divergence frontier."""

import unittest

from cascadiav3.compare_cuda_concurrency import _compare_arm


def decision(action="a", free="not_available", action_count=10, sims=64, root_value=90.0):
    return {
        "chosen_action_id": action,
        "free_three_of_a_kind_choice": free,
        "action_count": action_count,
        "simulations_run": sims,
        "market_branches_searched": 1,
        "market_chance_samples": 0,
        "total_simulations_run": sims,
        "exact_endgame": False,
        "root_value": root_value,
    }


def game(totals, decision_count):
    return {
        "scores": [{"total": total} for total in totals],
        "decision_count": decision_count,
    }


class CompareArmDivergenceTest(unittest.TestCase):
    def test_identical_arms_have_full_parity(self) -> None:
        decisions = {(1, ply): decision() for ply in range(4)}
        games = {1: game([90, 91, 92, 93], 4)}
        result = _compare_arm(decisions, dict(decisions), games, dict(games), 0.5)
        self.assertEqual(result["divergent_seed_count"], 0)
        self.assertTrue(result["policy_parity"])
        self.assertTrue(result["eligible_for_knee_selection"])
        self.assertEqual(result["compared_decision_count"], 4)

    def test_divergence_tolerates_downstream_invariant_changes(self) -> None:
        # Candidate flips the chosen action at ply 1; downstream plies belong
        # to a different game (different menus, different lengths).
        baseline = {(1, ply): decision(action=f"a{ply}") for ply in range(4)}
        candidate = {
            (1, 0): decision(action="a0"),
            (1, 1): decision(action="DIFFERENT"),
            (1, 2): decision(action="x", action_count=7, sims=48),
        }
        baseline_games = {1: game([90, 91, 92, 93], 4)}
        candidate_games = {1: game([88, 91, 92, 95], 3)}
        result = _compare_arm(baseline, candidate, baseline_games, candidate_games, 0.5)
        self.assertEqual(result["divergent_seed_count"], 1)
        self.assertEqual(
            result["divergent_seeds"], [{"seed": 1, "first_divergence_ply": 1}]
        )
        # Plies 0 and 1 were compared (the flip ply still shares its root
        # state); ply 2+ was not.
        self.assertEqual(result["compared_decision_count"], 2)
        self.assertFalse(result["policy_parity"])
        self.assertTrue(result["eligible_for_knee_selection"])
        self.assertAlmostEqual(result["paired_score_delta_stats"]["mean"], 0.0)

    def test_pre_divergence_invariant_mismatch_is_a_real_bug(self) -> None:
        baseline = {(1, 0): decision(action="a0"), (1, 1): decision(action="a1")}
        candidate = {
            (1, 0): decision(action="a0", action_count=9),
            (1, 1): decision(action="a1"),
        }
        games = {1: game([90, 91, 92, 93], 2)}
        with self.assertRaisesRegex(ValueError, "pre-divergence"):
            _compare_arm(baseline, candidate, games, dict(games), 0.5)

    def test_length_mismatch_without_divergence_is_refused(self) -> None:
        baseline = {(1, ply): decision(action=f"a{ply}") for ply in range(3)}
        candidate = {(1, ply): decision(action=f"a{ply}") for ply in range(2)}
        games = {1: game([90, 91, 92, 93], 3)}
        with self.assertRaisesRegex(ValueError, "without an action divergence"):
            _compare_arm(baseline, candidate, games, dict(games), 0.5)

    def test_numeric_drift_beyond_tolerance_blocks_selection(self) -> None:
        baseline = {(1, 0): decision(root_value=90.0)}
        candidate = {(1, 0): decision(root_value=91.0)}
        games = {1: game([90, 91, 92, 93], 1)}
        result = _compare_arm(baseline, candidate, games, dict(games), 0.5)
        self.assertFalse(result["numeric_parity_within_tolerance"])
        self.assertFalse(result["eligible_for_knee_selection"])


if __name__ == "__main__":
    unittest.main()
