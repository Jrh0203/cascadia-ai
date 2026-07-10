"""Contract tests for the exact-K1 comparator's declared-exclusion mechanism."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.compare_exact_endgame import build_comparison

RULES = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
SEEDS = [11, 12, 13]
RULING = "user ruling 2026-07-10: concurrency-divergent seed excluded by declaration"


def make_report(exact_turns: int, seed_scores: dict[int, float]) -> dict:
    return {
        "status": "pass",
        "ruleset_id": RULES,
        "control": {"kind": "none"},
        "source_revision": "rev1",
        "seeds": list(seed_scores),
        "search": {"n_simulations": 256, "exact_endgame_turns": exact_turns},
        "manifest": "checkpoints/x/best_locked_val.manifest.json",
        "market_decisions": {
            "exact_endgame_decisions": 4 * len(seed_scores) if exact_turns else 0
        },
        "candidate_per_seed": [
            {
                "seed": seed,
                "mean_score_per_seat": score,
                "seat_scores": [score, score, score, score],
            }
            for seed, score in seed_scores.items()
        ],
        "strategies": {
            "gumbel-search": {
                "mean_seat_score": sum(seed_scores.values()) / len(seed_scores),
                "mean_total_decision_seconds": 2.0 if not exact_turns else 1.5,
            }
        },
        "candidate_wall_seconds": 100.0 if not exact_turns else 80.0,
        "candidate_decision_seconds_p50": 1.0,
        "candidate_decision_seconds_p95": 3.0,
    }


def make_decisions(exact_arm: bool, divergent: dict[int, int]) -> list[dict]:
    """divergent maps seed -> first ply at which this arm's actions differ."""
    rows = []
    for seed in SEEDS:
        flip_from = divergent.get(seed, 80)
        for ply in range(80):
            action = 1000 + ply if (not exact_arm or ply < flip_from) else 5000 + ply
            row = {
                "type": "gumbel_decision",
                "ruleset_id": RULES,
                "seed": seed,
                "ply": ply,
                "chosen_action_id": f"sha256:{action:064x}",
                "free_three_of_a_kind_choice": None,
                "decision_seconds": 0.5 if (exact_arm and ply >= 76) else 2.0,
            }
            if exact_arm and ply >= 76:
                row["exact_endgame"] = True
                row["total_simulations_run"] = 0
            rows.append(row)
    return rows


class CompareExactEndgameTest(unittest.TestCase):
    def _write(self, tmp: str, divergent: dict[int, int]) -> dict[str, Path]:
        base = Path(tmp)
        paths = {
            "baseline": base / "baseline.json",
            "exact": base / "exact.json",
            "baseline_decisions": base / "baseline_decisions.jsonl",
            "exact_decisions": base / "exact_decisions.jsonl",
        }
        paths["baseline"].write_text(
            json.dumps(make_report(0, {11: 97.0, 12: 96.5, 13: 97.0})),
            encoding="utf-8",
        )
        paths["exact"].write_text(
            json.dumps(make_report(1, {11: 97.5, 12: 96.5, 13: 97.0})),
            encoding="utf-8",
        )
        paths["baseline_decisions"].write_text(
            "\n".join(json.dumps(r) for r in make_decisions(False, {})) + "\n",
            encoding="utf-8",
        )
        paths["exact_decisions"].write_text(
            "\n".join(json.dumps(r) for r in make_decisions(True, divergent)) + "\n",
            encoding="utf-8",
        )
        return paths

    def _build(self, paths: dict[str, Path], **kwargs) -> dict:
        return build_comparison(
            paths["baseline"],
            paths["exact"],
            paths["baseline_decisions"],
            paths["exact_decisions"],
            **kwargs,
        )

    def test_clean_traces_pass_without_exclusions(self) -> None:
        with TemporaryDirectory() as tmp:
            report = self._build(self._write(tmp, {}))
            self.assertEqual(report["retained_seed_count"], 3)
            self.assertEqual(report["declared_exclusions"]["seeds"], [])
            self.assertEqual(len(report["paired_score_deltas"]), 3)

    def test_undeclared_pre_k1_divergence_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = self._write(tmp, {12: 18})
            with self.assertRaisesRegex(ValueError, "diverges at seed 12 ply 18"):
                self._build(paths)

    def test_declared_exclusion_retains_other_seeds(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = self._write(tmp, {12: 18})
            report = self._build(
                paths, declared_divergent_seeds=[12], exclusion_ruling=RULING
            )
            self.assertEqual(report["retained_seed_count"], 2)
            self.assertEqual(report["declared_exclusions"]["seeds"], [12])
            self.assertEqual(
                report["declared_exclusions"]["first_divergent_ply"], {"12": 18}
            )
            self.assertEqual(report["declared_exclusions"]["ruling"], RULING)
            self.assertEqual(
                report["scientific_eligibility"], "engineering_smoke_only"
            )
            retained_seeds = [row["seed"] for row in report["paired_score_deltas"]]
            self.assertEqual(retained_seeds, [11, 13])
            self.assertEqual(
                [row["delta"] for row in report["paired_score_deltas"]], [0.5, 0.0]
            )

    def test_declared_exclusion_requires_ruling(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = self._write(tmp, {12: 18})
            with self.assertRaisesRegex(ValueError, "explicit exclusion ruling"):
                self._build(paths, declared_divergent_seeds=[12])

    def test_ruling_without_seeds_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = self._write(tmp, {})
            with self.assertRaisesRegex(ValueError, "no seeds were declared"):
                self._build(paths, exclusion_ruling=RULING)

    def test_causally_clean_seed_cannot_be_excluded(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = self._write(tmp, {})
            with self.assertRaisesRegex(ValueError, "refusing the exclusion"):
                self._build(
                    paths, declared_divergent_seeds=[12], exclusion_ruling=RULING
                )

    def test_declared_seed_missing_from_reports_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = self._write(tmp, {})
            with self.assertRaisesRegex(ValueError, "not all present"):
                self._build(
                    paths, declared_divergent_seeds=[99], exclusion_ruling=RULING
                )


if __name__ == "__main__":
    unittest.main()
