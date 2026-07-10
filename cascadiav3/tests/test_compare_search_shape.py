"""Contract tests for the worlds-allocation (search-shape) comparator."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.compare_search_shape import build_comparison

RULES = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
SEEDS = [31, 32, 33]


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


class CompareSearchShapeTest(unittest.TestCase):
    def test_happy_path_reports_paired_verdict(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline = write(tmp, "b.json", make_report(4, {s: 97.0 for s in SEEDS}))
            candidate = write(
                tmp, "c.json", make_report(8, {31: 97.5, 32: 97.25, 33: 97.75})
            )
            report = build_comparison(baseline, candidate)
            self.assertEqual(report["paired_delta_stats"]["mean"], 0.5)
            self.assertEqual(report["search"]["baseline_determinizations"], 4)
            self.assertEqual(report["search"]["candidate_determinizations"], 8)
            # 3 seeds is not promotion scale: never proceed automatically.
            self.assertFalse(report["proceed_to_high_budget"])

    def test_identical_world_counts_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline = write(tmp, "b.json", make_report(4, {s: 97.0 for s in SEEDS}))
            candidate = write(tmp, "c.json", make_report(4, {s: 97.5 for s in SEEDS}))
            with self.assertRaisesRegex(ValueError, "distinct positive world counts"):
                build_comparison(baseline, candidate)

    def test_other_search_deltas_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline = write(tmp, "b.json", make_report(4, {s: 97.0 for s in SEEDS}))
            payload = make_report(8, {s: 97.5 for s in SEEDS})
            payload["search"]["market_decision_samples"] = 4
            candidate = write(tmp, "c.json", payload)
            with self.assertRaisesRegex(ValueError, "differ beyond determinizations"):
                build_comparison(baseline, candidate)

    def test_revision_mismatch_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline = write(tmp, "b.json", make_report(4, {s: 97.0 for s in SEEDS}))
            candidate = write(
                tmp, "c.json", make_report(8, {s: 97.5 for s in SEEDS}, revision="rev2")
            )
            with self.assertRaisesRegex(ValueError, "one non-empty source revision"):
                build_comparison(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
