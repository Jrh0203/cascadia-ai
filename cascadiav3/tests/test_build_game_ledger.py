"""Contract tests for rebuilding game ledgers from raw per-seed files."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.build_game_ledger import LedgerBuildError, build_ledger
from cascadiav3.compare_game_categories import RULESET_ID

SEEDS = [2027070900, 2027070901, 2027070902]

REPORT_SEARCH = {
    "n_simulations": 1024,
    "top_m": 16,
    "depth_rounds": 1,
    "determinizations": 16,
    "market_decision_samples": 8,
    "exact_endgame_turns": 0,
    "blend_weight": 0.5,
    "k_interior": 16,
}

LEDGER_SEARCH = {
    "n_simulations": 1024,
    "top_m": 16,
    "depth_rounds": 1,
    "determinization_samples": 16,
    "market_decision_samples": 8,
    "exact_endgame_turns": 0,
    "rollout_blend_weight": 0.5,
    "k_interior": 16,
    "exploration": False,
}


def make_score(total: int) -> dict:
    return {
        "wildlife": [total - 12, 4, 4, 0, 0],
        "habitat": [1, 1, 1, 0, 0],
        "nature_tokens": 1,
        "total": total,
    }


def make_done_row(seed: int, totals) -> dict:
    return {
        "type": "gumbel_game_done",
        "seed": seed,
        "ruleset_id": RULESET_ID,
        "decision_count": 80,
        "search": dict(LEDGER_SEARCH),
        "scores": [make_score(t) for t in totals],
    }


def make_report(totals_by_seed) -> dict:
    return {
        "status": "pass",
        "ruleset_id": RULESET_ID,
        "experiment_id": "test_arm",
        "source_revision": "d20daf44",
        "seeds": SEEDS,
        "search": dict(REPORT_SEARCH),
        "candidate_per_seed": [
            {
                "seed": seed,
                "mean_score_per_seat": sum(totals) / len(totals),
                "seat_scores": [float(t) for t in totals],
            }
            for seed, totals in totals_by_seed.items()
        ],
    }


def write_raw_dir(tmp: str, totals_by_seed, skip_seed=None, decisions=80) -> Path:
    raw = Path(tmp) / "raw_games"
    raw.mkdir()
    for seed, totals in totals_by_seed.items():
        if seed == skip_seed:
            continue
        rows = [
            {"type": "gumbel_decision", "seed": seed, "ply": ply,
             "chosen_action_id": ply, "ruleset_id": RULESET_ID}
            for ply in range(decisions)
        ]
        rows.append(make_done_row(seed, totals))
        (raw / f"gumbel_game_seed_{seed}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    return raw


TOTALS = {SEEDS[0]: (97, 98, 97, 100), SEEDS[1]: (96, 96, 98, 98), SEEDS[2]: (99, 97, 97, 99)}


class BuildGameLedgerTest(unittest.TestCase):
    def test_happy_path_builds_ledger_and_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = write_raw_dir(tmp, TOTALS)
            games_out = Path(tmp) / "arm_games.jsonl"
            summary_out = Path(tmp) / "arm_category_summary.json"
            summary = build_ledger(
                raw_games_dir=raw,
                report=make_report(TOTALS),
                games_out=games_out,
                category_summary_out=summary_out,
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["summary"]["games"], 3)
            rows = [json.loads(l) for l in games_out.read_text().splitlines()]
            self.assertEqual([r["seed"] for r in rows], sorted(SEEDS))
            written = json.loads(summary_out.read_text())
            self.assertEqual(written["status"], "complete")

    def test_missing_seed_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = write_raw_dir(tmp, TOTALS, skip_seed=SEEDS[1])
            with self.assertRaisesRegex(LedgerBuildError, "missing seeds"):
                build_ledger(
                    raw_games_dir=raw,
                    report=make_report(TOTALS),
                    games_out=Path(tmp) / "g.jsonl",
                    category_summary_out=Path(tmp) / "c.json",
                )

    def test_wrong_decision_count_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = write_raw_dir(tmp, TOTALS, decisions=79)
            with self.assertRaisesRegex(LedgerBuildError, "expected 81/80/1"):
                build_ledger(
                    raw_games_dir=raw,
                    report=make_report(TOTALS),
                    games_out=Path(tmp) / "g.jsonl",
                    category_summary_out=Path(tmp) / "c.json",
                )

    def test_report_total_mismatch_fails_via_consumer_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = write_raw_dir(tmp, TOTALS)
            bad_totals = dict(TOTALS)
            bad_totals[SEEDS[0]] = (97, 98, 97, 101)  # report disagrees with raw
            games_out = Path(tmp) / "g.jsonl"
            with self.assertRaisesRegex(LedgerBuildError, "consumer validation"):
                build_ledger(
                    raw_games_dir=raw,
                    report=make_report(bad_totals),
                    games_out=games_out,
                    category_summary_out=Path(tmp) / "c.json",
                )
            self.assertFalse(games_out.exists())  # nothing published on failure


if __name__ == "__main__":
    unittest.main()
