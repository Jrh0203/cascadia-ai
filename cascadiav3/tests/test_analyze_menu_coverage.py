"""Contract tests for the R1.3a menu-coverage analyzer."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.analyze_menu_coverage import analyze


def menu_root(seed, ply, action_ids, qs, chosen=0):
    return {
        "type": "puzzle_root",
        "seed": seed,
        "ply": ply,
        "active_seat": 0,
        "action_count": len(qs),
        "ledger_chosen_action_id": action_ids[chosen],
        "action_ids": list(action_ids),
        "mean_completed_q": list(qs),
        "total_visits": [4] * len(qs),
        "repeat_chosen_indexes": [chosen],
        "repeat_agreement": 1.0,
        "repeats": 1,
        "search": {"n_simulations": 1024},
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


class AnalyzeMenuCoverageTest(unittest.TestCase):
    def test_drop_detected_and_regret_scored_on_full_q(self) -> None:
        with TemporaryDirectory() as tmp:
            # Root (1,0): the cap keeps a0/a2 but drops a1, the full-run best
            # (Q 95). Regret is 95 - 92 = 3 against the best kept action a2.
            # Root (1,3): the full-run best b0 survives the cap — regret 0.
            capped = write_shards(
                tmp,
                "capped",
                [
                    menu_root(1, 0, ["a0", "a2"], [89.0, 91.0]),
                    menu_root(1, 3, ["b0", "b1"], [88.0, 87.5]),
                ],
            )
            full = write_shards(
                tmp,
                "full",
                [
                    menu_root(1, 0, ["a0", "a1", "a2"], [90.0, 95.0, 92.0]),
                    menu_root(1, 3, ["b0", "b1"], [88.0, 87.0]),
                ],
            )
            report = analyze(capped, full)
            self.assertEqual(report["roots"], 2)
            self.assertEqual(report["drop_count"], 1)
            self.assertAlmostEqual(report["drop_rate"], 0.5)
            self.assertAlmostEqual(report["mean_regret_when_dropped"], 3.0)
            self.assertAlmostEqual(report["mean_regret_overall"], 1.5)
            self.assertEqual(report["capped_menu_median_size"], 2)
            self.assertAlmostEqual(report["full_menu_median_size"], 2.5)
            dropped_row = next(r for r in report["per_root"] if r["ply"] == 0)
            self.assertTrue(dropped_row["dropped"])
            self.assertEqual(dropped_row["full_best_action_id"], "a1")
            self.assertAlmostEqual(dropped_row["regret"], 3.0)
            kept_row = next(r for r in report["per_root"] if r["ply"] == 3)
            self.assertFalse(kept_row["dropped"])
            self.assertEqual(kept_row["regret"], 0.0)

    def test_capped_not_subset_roots_skipped_and_counted(self) -> None:
        with TemporaryDirectory() as tmp:
            # Ply 0 violates the subset invariant ("zz" is not in the full
            # menu); plies 1..10 are clean, keeping the skip rate at 1/11
            # (below the 10% fail-closed tolerance).
            capped_rows = [menu_root(1, 0, ["zz", "a1"], [90.0, 91.0])]
            full_rows = [menu_root(1, 0, ["a0", "a1"], [90.0, 91.0])]
            for ply in range(1, 11):
                capped_rows.append(menu_root(1, ply, ["c0", "c1"], [80.0, 81.0]))
                full_rows.append(menu_root(1, ply, ["c0", "c1"], [80.0, 81.0]))
            report = analyze(
                write_shards(tmp, "capped", capped_rows),
                write_shards(tmp, "full", full_rows),
            )
            self.assertEqual(report["roots"], 10)
            self.assertEqual(report["subset_mismatch_roots"], 1)
            self.assertEqual(report["skipped_roots"], 1)
            self.assertEqual(report["drop_count"], 0)
            self.assertNotIn(0, [row["ply"] for row in report["per_root"]])

    def test_capped_missing_roots_skipped_and_counted(self) -> None:
        with TemporaryDirectory() as tmp:
            capped_rows = [
                menu_root(1, ply, ["c0", "c1"], [80.0, 81.0]) for ply in range(10)
            ]
            full_rows = capped_rows + [menu_root(1, 10, ["d0", "d1"], [70.0, 71.0])]
            report = analyze(
                write_shards(tmp, "capped", capped_rows),
                write_shards(tmp, "full", full_rows),
            )
            self.assertEqual(report["roots"], 10)
            self.assertEqual(report["capped_missing_roots"], 1)
            self.assertEqual(report["skipped_roots"], 1)

    def test_excess_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            # 1 mismatch out of 2 full roots (50%) exceeds the 10% tolerance.
            capped = write_shards(
                tmp,
                "capped",
                [
                    menu_root(1, 0, ["zz", "a1"], [90.0, 91.0]),
                    menu_root(1, 3, ["b0", "b1"], [88.0, 87.5]),
                ],
            )
            full = write_shards(
                tmp,
                "full",
                [
                    menu_root(1, 0, ["a0", "a1"], [90.0, 95.0]),
                    menu_root(1, 3, ["b0", "b1"], [88.0, 87.0]),
                ],
            )
            with self.assertRaisesRegex(ValueError, "tolerance"):
                analyze(capped, full)

    def test_empty_join_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            capped = write_shards(
                tmp, "capped", [menu_root(1, 0, ["a0", "a1"], [90.0, 92.0])]
            )
            full = write_shards(
                tmp, "full", [menu_root(2, 0, ["a0", "a1"], [90.0, 92.0])]
            )
            with self.assertRaisesRegex(ValueError, "share no roots"):
                analyze(capped, full)


if __name__ == "__main__":
    unittest.main()
