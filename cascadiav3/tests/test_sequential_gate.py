"""Contract tests for the group-sequential gate verdict layer."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.sequential_gate import (
    build_sequential_verdict,
    parse_looks,
    sequential_decision,
)

RULES = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
LOOKS = [40, 60, 80, 100]


def make_report(worlds: int, seed_scores: dict[int, float], revision="rev1") -> dict:
    return {
        "status": "pass",
        "ruleset_id": RULES,
        "control": {"kind": "none"},
        "source_revision": revision,
        "seeds": list(seed_scores),
        "search": {
            "n_simulations": 256,
            "determinizations": worlds,
            "exact_endgame_turns": 1,
            "market_decision_samples": 8,
        },
        "manifest": "checkpoints/x/best_locked_val.manifest.json",
        "candidate_per_seed": [
            {"seed": seed, "mean_score_per_seat": score}
            for seed, score in seed_scores.items()
        ],
        "strategies": {
            "gumbel-search": {
                "mean_seat_score": sum(seed_scores.values()) / len(seed_scores),
                "mean_total_decision_seconds": 12.0,
            }
        },
        "candidate_wall_seconds": 100.0,
    }


def write(tmp: str, name: str, payload: dict) -> Path:
    path = Path(tmp) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def paired_reports(tmp: str, pairs: int, mean_delta: float, wiggle: float = 0.5):
    """Baseline at 97.0; candidate at 97.0 + mean_delta +/- wiggle."""
    seeds = list(range(1000, 1000 + pairs))
    baseline = write(tmp, "b.json", make_report(4, {s: 97.0 for s in seeds}))
    candidate_scores = {
        seed: 97.0 + mean_delta + (wiggle if index % 2 == 0 else -wiggle)
        for index, seed in enumerate(seeds)
    }
    candidate = write(tmp, "c.json", make_report(8, candidate_scores))
    return baseline, candidate


class ParseLooksTest(unittest.TestCase):
    def test_accepts_commas_and_spaces(self) -> None:
        self.assertEqual(parse_looks("40,60,80,100"), LOOKS)
        self.assertEqual(parse_looks("40 60 80 100"), LOOKS)

    def test_rejects_non_increasing_and_garbage(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_looks("40,40,100")
        with self.assertRaisesRegex(ValueError, "unparseable"):
            parse_looks("forty")
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_looks("  ")


class SequentialDecisionTest(unittest.TestCase):
    def test_superiority_labels(self) -> None:
        kwargs = {"rule": "superiority", "margin": -0.25}
        self.assertEqual(
            sequential_decision(0.2, 0.9, is_final_look=False, **kwargs), "stop_positive"
        )
        self.assertEqual(
            sequential_decision(-0.9, -0.2, is_final_look=False, **kwargs),
            "stop_negative",
        )
        self.assertEqual(
            sequential_decision(-0.1, 0.4, is_final_look=False, **kwargs), "continue"
        )
        self.assertEqual(
            sequential_decision(0.2, 0.9, is_final_look=True, **kwargs), "final_positive"
        )
        self.assertEqual(
            sequential_decision(-0.1, 0.4, is_final_look=True, **kwargs),
            "final_inconclusive",
        )

    def test_noninferiority_labels(self) -> None:
        kwargs = {"rule": "noninferiority", "margin": -0.25}
        self.assertEqual(
            sequential_decision(-0.2, 0.1, is_final_look=False, **kwargs),
            "stop_noninferior",
        )
        self.assertEqual(
            sequential_decision(-0.9, -0.3, is_final_look=False, **kwargs),
            "stop_inferior",
        )
        self.assertEqual(
            sequential_decision(-0.4, 0.1, is_final_look=False, **kwargs), "continue"
        )


class BuildSequentialVerdictTest(unittest.TestCase):
    def test_strong_effect_stops_at_first_look(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(tmp, pairs=40, mean_delta=1.0)
            report = build_sequential_verdict(baseline, candidate, LOOKS)
            seq = report["sequential"]
            self.assertEqual(seq["current_look"], 1)
            self.assertEqual(seq["decision"], "stop_positive")
            self.assertGreater(seq["rci_low"], 0.0)
            self.assertTrue(report["proceed_to_high_budget"])
            self.assertEqual(
                report["scientific_eligibility"], "promotion_scale_sequential_gate"
            )
            # The interim RCI must be wider than the naive 95% CI.
            self.assertLess(seq["rci_low"], seq["naive_ci_low_non_inferential"])

    def test_null_effect_continues_at_interim_looks(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(tmp, pairs=60, mean_delta=0.0)
            report = build_sequential_verdict(baseline, candidate, LOOKS)
            self.assertEqual(report["sequential"]["current_look"], 2)
            self.assertEqual(report["sequential"]["decision"], "continue")
            self.assertFalse(report["proceed_to_high_budget"])

    def test_null_effect_at_final_look_is_inconclusive(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(tmp, pairs=100, mean_delta=0.0)
            report = build_sequential_verdict(baseline, candidate, LOOKS)
            self.assertEqual(report["sequential"]["decision"], "final_inconclusive")

    def test_strong_regression_stops_negative(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(tmp, pairs=80, mean_delta=-1.0)
            report = build_sequential_verdict(baseline, candidate, LOOKS)
            self.assertEqual(report["sequential"]["decision"], "stop_negative")
            self.assertFalse(report["proceed_to_high_budget"])

    def test_noninferiority_rule_stops_on_tight_null(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(
                tmp, pairs=40, mean_delta=0.0, wiggle=0.3
            )
            report = build_sequential_verdict(
                baseline, candidate, LOOKS, rule="noninferiority", margin=-0.25
            )
            seq = report["sequential"]
            self.assertEqual(seq["decision"], "stop_noninferior")
            self.assertGreater(seq["rci_low"], -0.25)
            # Noninferiority is never scaling evidence.
            self.assertFalse(report["proceed_to_high_budget"])

    def test_off_schedule_pair_count_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(tmp, pairs=50, mean_delta=1.0)
            with self.assertRaisesRegex(ValueError, "does not equal any planned look"):
                build_sequential_verdict(baseline, candidate, LOOKS)

    def test_small_planned_design_is_not_promotion_scale(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = paired_reports(tmp, pairs=20, mean_delta=1.0)
            report = build_sequential_verdict(baseline, candidate, [20, 50])
            self.assertEqual(
                report["scientific_eligibility"], "engineering_smoke_only"
            )
            self.assertFalse(report["proceed_to_high_budget"])


if __name__ == "__main__":
    unittest.main()
