"""Contract tests for the puzzle-bank screen scorer."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.analyze_puzzle_screen import analyze


def puzzle_root(seed, ply, qs, chosen, action_ids=None, repeats=2):
    return {
        "type": "puzzle_root",
        "seed": seed,
        "ply": ply,
        "active_seat": 0,
        "action_count": len(qs),
        "ledger_chosen_action_id": "x",
        "action_ids": action_ids or [f"a{i}" for i in range(len(qs))],
        "mean_completed_q": qs,
        "total_visits": [4] * len(qs),
        "repeat_chosen_indexes": [chosen] * repeats,
        "repeat_agreement": 1.0,
        "repeats": repeats,
        "search": {"n_simulations": 4096},
    }


def write_shards(tmp, name, rows):
    directory = Path(tmp) / name
    directory.mkdir()
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    for seed, seed_rows in by_seed.items():
        (directory / f"puzzle_seed_{seed}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in seed_rows) + "\n", encoding="utf-8"
        )
    return directory


class AnalyzePuzzleScreenTest(unittest.TestCase):
    def test_regret_scored_against_bank_values(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = write_shards(
                tmp,
                "bank",
                [
                    puzzle_root(1, 0, [90.0, 92.0, 91.0], chosen=1),
                    puzzle_root(1, 3, [88.0, 87.5], chosen=0),
                ],
            )
            # Screen agrees at root (1,0) but picks the 0.5-worse action at
            # (1,3) per the bank's values.
            screen = write_shards(
                tmp,
                "screen",
                [
                    puzzle_root(1, 0, [89.0, 91.0, 90.0], chosen=1, repeats=1),
                    puzzle_root(1, 3, [87.0, 88.5], chosen=1, repeats=1),
                ],
            )
            report = analyze(bank, screen)
            self.assertEqual(report["roots"], 2)
            self.assertAlmostEqual(report["mean_regret"], 0.25)
            self.assertAlmostEqual(report["zero_regret_rate"], 0.5)

    def test_menu_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = write_shards(tmp, "bank", [puzzle_root(1, 0, [90.0, 92.0], chosen=1)])
            screen = write_shards(
                tmp,
                "screen",
                [puzzle_root(1, 0, [90.0, 92.0], chosen=1, action_ids=["zz", "a1"])],
            )
            with self.assertRaisesRegex(ValueError, "menu mismatch"):
                analyze(bank, screen)

    def test_disjoint_roots_fail_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = write_shards(tmp, "bank", [puzzle_root(1, 0, [90.0, 92.0], chosen=1)])
            screen = write_shards(tmp, "screen", [puzzle_root(2, 0, [90.0, 92.0], chosen=1)])
            with self.assertRaisesRegex(ValueError, "share no roots"):
                analyze(bank, screen)


if __name__ == "__main__":
    unittest.main()
