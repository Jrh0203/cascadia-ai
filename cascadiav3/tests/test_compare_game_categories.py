"""Paired score-category attribution tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cascadiav3.compare_game_categories import RULESET_ID, build_category_comparison


class CompareGameCategoriesTest(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        label: str,
        *,
        wildlife_offset: int = 0,
        habitat_offset: int = 0,
        token_offset: int = 0,
        search_n: int = 64,
    ) -> tuple[Path, Path]:
        seeds = [101, 102]
        report_path = root / f"{label}.json"
        games_path = root / f"{label}_games.jsonl"
        game_rows = []
        per_seed = []
        for seed in seeds:
            scores = []
            for seat in range(4):
                wildlife = [10 + wildlife_offset + (seed - 101), seat]
                habitat = [20 + habitat_offset, seat]
                nature = 5 + token_offset
                total = sum(wildlife) + sum(habitat) + nature
                scores.append(
                    {
                        "wildlife": wildlife,
                        "habitat": habitat,
                        "nature_tokens": nature,
                        "total": total,
                    }
                )
            per_seed.append(
                {
                    "seed": seed,
                    "mean_score_per_seat": sum(score["total"] for score in scores) / 4,
                }
            )
            game_rows.append(
                {
                    "type": "gumbel_game_done",
                    "seed": seed,
                    "ruleset_id": RULESET_ID,
                    "decision_count": 80,
                    "search": {
                        "n_simulations": search_n,
                        "top_m": 16,
                        "depth_rounds": 1,
                        "determinization_samples": 4,
                        "market_decision_samples": 8,
                        "exact_endgame_turns": 0,
                        "rollout_blend_weight": 0.5,
                        "k_interior": 16,
                    },
                    "scores": scores,
                }
            )
        report_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "ruleset_id": RULESET_ID,
                    "source_revision": "revision",
                    "experiment_id": label,
                    "seeds": seeds,
                    "candidate_per_seed": per_seed,
                    "control": {"kind": "none"},
                    "search": {
                        "n_simulations": search_n,
                        "top_m": 16,
                        "depth_rounds": 1,
                        "determinizations": 4,
                        "market_decision_samples": 8,
                        "blend_weight": 0.5,
                        "k_interior": 16,
                    },
                }
            ),
            encoding="utf-8",
        )
        games_path.write_text(
            "".join(json.dumps(row) + "\n" for row in game_rows),
            encoding="utf-8",
        )
        return report_path, games_path

    def test_category_deltas_sum_to_paired_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_report, left_games = self._write_arm(
                root,
                "left",
                wildlife_offset=2,
                habitat_offset=-1,
                token_offset=1,
            )
            right_report, right_games = self._write_arm(root, "right")
            result = build_category_comparison(
                left_report_path=left_report,
                left_games_path=left_games,
                right_report_path=right_report,
                right_games_path=right_games,
                source_revision="revision",
                label="left - right",
            )
            self.assertEqual(result["status"], "pass")
            stats = result["paired_left_minus_right"]
            self.assertEqual(stats["wildlife"]["mean"], 2.0)
            self.assertEqual(stats["habitat"]["mean"], -1.0)
            self.assertEqual(stats["nature_tokens"]["mean"], 1.0)
            self.assertEqual(stats["total"]["mean"], 2.0)

    def test_search_mismatch_and_category_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_report, left_games = self._write_arm(root, "left", search_n=64)
            right_report, right_games = self._write_arm(root, "right", search_n=32)
            with self.assertRaisesRegex(ValueError, "search contract"):
                build_category_comparison(
                    left_report_path=left_report,
                    left_games_path=left_games,
                    right_report_path=right_report,
                    right_games_path=right_games,
                    source_revision="revision",
                    label="bad",
                )
            rows = [json.loads(line) for line in left_games.read_text().splitlines()]
            rows[0]["scores"][0]["total"] += 1
            left_games.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "category sum"):
                build_category_comparison(
                    left_report_path=left_report,
                    left_games_path=left_games,
                    right_report_path=left_report,
                    right_games_path=left_games,
                    source_revision="revision",
                    label="tampered",
                )

    def test_missing_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, games = self._write_arm(root, "arm")
            games.write_text(games.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "row count"):
                build_category_comparison(
                    left_report_path=report,
                    left_games_path=games,
                    right_report_path=report,
                    right_games_path=games,
                    source_revision="revision",
                    label="missing",
                )


if __name__ == "__main__":
    unittest.main()
