"""Contract tests for the market-sample comparator's trace classification."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.compare_market_samples import build_comparison

RULES = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
SEEDS = [21, 22, 23]
EXPOSURE_PLY = 10  # every seed sees its first refresh opportunity here


def make_report(samples: int, seed_scores: dict[int, float]) -> dict:
    return {
        "status": "pass",
        "ruleset_id": RULES,
        "control": {"kind": "none"},
        "source_revision": "rev1",
        "seeds": list(seed_scores),
        "search": {"n_simulations": 256, "market_decision_samples": samples},
        "manifest": "checkpoints/x/best_locked_val.manifest.json",
        "market_decisions": {
            "mean_chance_samples_when_available": float(samples),
            "total_simulations_including_market_decision": 1000 * samples,
            "market_decision_simulation_overhead": 100 * samples,
        },
        "candidate_per_seed": [
            {"seed": seed, "mean_score_per_seat": score}
            for seed, score in seed_scores.items()
        ],
        "strategies": {
            "gumbel-search": {
                "mean_seat_score": sum(seed_scores.values()) / len(seed_scores),
                "mean_total_decision_seconds": 12.0 if samples == 8 else 6.0,
            }
        },
        "candidate_wall_seconds": 100.0,
        "candidate_decision_seconds_p95": 3.0,
    }


def make_decisions(arm: str, divergent: dict[int, int]) -> list[dict]:
    """divergent maps seed -> first ply where this arm's actions differ."""
    rows = []
    for seed in SEEDS:
        flip_from = divergent.get(seed, 80)
        for ply in range(80):
            action = 1000 + ply if (arm == "baseline" or ply < flip_from) else 7000 + ply
            rows.append(
                {
                    "type": "gumbel_decision",
                    "ruleset_id": RULES,
                    "seed": seed,
                    "ply": ply,
                    "chosen_action_id": f"sha256:{action:064x}",
                    "free_three_of_a_kind_choice": (
                        "accept" if ply == EXPOSURE_PLY else None
                    ),
                    "decision_seconds": 2.0,
                }
            )
    return rows


class CompareMarketSamplesTest(unittest.TestCase):
    def _build(self, tmp: str, divergent: dict[int, int]) -> dict:
        base = Path(tmp)
        (base / "baseline.json").write_text(
            json.dumps(make_report(8, {s: 97.0 for s in SEEDS})), encoding="utf-8"
        )
        (base / "candidate.json").write_text(
            json.dumps(make_report(4, {21: 97.0, 22: 96.5, 23: 97.5})),
            encoding="utf-8",
        )
        (base / "baseline_decisions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in make_decisions("baseline", {})) + "\n",
            encoding="utf-8",
        )
        (base / "candidate_decisions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in make_decisions("candidate", divergent))
            + "\n",
            encoding="utf-8",
        )
        return build_comparison(
            base / "baseline.json",
            base / "candidate.json",
            base / "baseline_decisions.jsonl",
            base / "candidate_decisions.jsonl",
        )

    def test_pre_exposure_divergence_is_classified_not_fatal(self) -> None:
        with TemporaryDirectory() as tmp:
            report = self._build(tmp, {21: 3, 22: EXPOSURE_PLY + 2})
            self.assertEqual(report["trace"]["pre_exposure_divergent_seeds"], 1)
            self.assertEqual(report["trace"]["causally_changed_seeds"], 1)
            self.assertEqual(len(report["paired_score_deltas"]), 3)

    def test_all_pre_exposure_divergence_still_yields_verdict(self) -> None:
        with TemporaryDirectory() as tmp:
            report = self._build(tmp, {21: 0, 22: 1, 23: 2})
            self.assertEqual(report["trace"]["pre_exposure_divergent_seeds"], 3)
            self.assertEqual(report["trace"]["causally_changed_seeds"], 0)
            self.assertIn("paired_delta_stats", report)

    def test_preregistered_gate_fields_present(self) -> None:
        with TemporaryDirectory() as tmp:
            report = self._build(tmp, {})
            self.assertEqual(report["noninferiority_margin"], -0.25)
            self.assertEqual(report["minimum_speedup"], 1.15)
            # 3-seed fixture is not promotion scale, so the gate cannot pass.
            self.assertFalse(report["performance_gate_pass"])
            self.assertEqual(
                report["scientific_eligibility"], "engineering_smoke_only"
            )


if __name__ == "__main__":
    unittest.main()
