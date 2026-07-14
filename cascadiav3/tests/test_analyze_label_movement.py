import json
import tempfile
import unittest
from pathlib import Path

from cascadiav3.analyze_label_movement import (
    MOVEMENT_BAR,
    NEAR_TIE_GUARD,
    analyze,
    write_markdown,
)


def _root(
    seed,
    ply,
    qs,
    serving_index,
    agreement=1.0,
    repeats=2,
    ledger_id="__from_index__",
):
    action_ids = [f"sha256:{seed:04d}{ply:03d}{i:03d}" for i in range(len(qs))]
    best = max(range(len(qs)), key=qs.__getitem__)
    return {
        "type": "puzzle_root",
        "seed": seed,
        "ply": ply,
        "repeats": repeats,
        "repeat_agreement": agreement,
        "repeat_chosen_indexes": [best] * repeats,
        "action_ids": action_ids,
        "mean_completed_q": qs,
        "ledger_chosen_action_id": (
            action_ids[serving_index] if ledger_id == "__from_index__" else ledger_id
        ),
        "search": {"n_simulations": 2048, "determinizations": 16},
    }


class AnalyzeLabelMovementTest(unittest.TestCase):
    def _write_bank(self, records):
        directory = Path(tempfile.mkdtemp())
        by_seed = {}
        for record in records:
            by_seed.setdefault(record["seed"], []).append(record)
        for seed, rows in by_seed.items():
            path = directory / f"puzzle_seed_{seed}.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
        return directory

    def test_movement_and_regret_per_root(self):
        # serving picked index 1 while the mega argmax is index 0 -> moved,
        # regret = 5.0 - 3.0 = 2.0; second root agrees with serving.
        bank = self._write_bank(
            [
                _root(1, 0, [5.0, 3.0, 1.0], serving_index=1),
                _root(1, 4, [4.0, 2.0, 1.0], serving_index=0),
            ]
        )
        report = analyze(bank)
        self.assertEqual(report["roots"], 2)
        moved, kept = report["per_root"]
        self.assertTrue(moved["moved"])
        self.assertAlmostEqual(moved["mega_regret"], 2.0)
        self.assertAlmostEqual(moved["top2_gap"], 2.0)
        self.assertFalse(kept["moved"])
        self.assertAlmostEqual(kept["mega_regret"], 0.0)
        self.assertAlmostEqual(report["overall"]["movement_rate"], 0.5)

    def test_bar_uses_stable_stratum_only(self):
        # Three unstable roots all moved; two stable roots did not move.
        # Overall movement is 3/5 but the stable-stratum rate is 0/2, so the
        # preregistered bar must FAIL.
        records = [
            _root(1, p, [2.0, 1.9], serving_index=1, agreement=0.5)
            for p in (0, 4, 8)
        ] + [
            _root(2, p, [3.0, 1.0], serving_index=0)
            for p in (0, 4)
        ]
        report = analyze(self._write_bank(records))
        self.assertAlmostEqual(report["unstable_fraction"], 0.6)
        self.assertAlmostEqual(report["overall"]["movement_rate"], 0.6)
        self.assertAlmostEqual(report["stable"]["movement_rate"], 0.0)
        self.assertFalse(report["preregistered_bar"]["bar_pass"])

    def test_bar_pass_and_near_tie_guard(self):
        # 2 of 4 stable roots moved (rate 0.5 >= bar) but both moves are
        # worth < NEAR_TIE_GUARD points -> pass with the churn flag up.
        records = [
            _root(1, 0, [2.0, 2.0 - NEAR_TIE_GUARD / 2], serving_index=1),
            _root(1, 4, [2.0, 2.0 - NEAR_TIE_GUARD / 2], serving_index=1),
            _root(2, 0, [3.0, 1.0], serving_index=0),
            _root(2, 4, [3.0, 1.0], serving_index=0),
        ]
        report = analyze(self._write_bank(records))
        bar = report["preregistered_bar"]
        self.assertGreaterEqual(bar["stable_movement_rate"], MOVEMENT_BAR)
        self.assertTrue(bar["bar_pass"])
        self.assertTrue(bar["near_tie_churn_flag"])

    def test_bar_pass_with_material_regret_has_no_flag(self):
        records = [
            _root(1, 0, [5.0, 3.0], serving_index=1),
            _root(2, 0, [3.0, 1.0], serving_index=0),
        ]
        report = analyze(self._write_bank(records))
        bar = report["preregistered_bar"]
        self.assertTrue(bar["bar_pass"])
        self.assertFalse(bar["near_tie_churn_flag"])

    def test_phase_split_follows_tile_count_proxy(self):
        # ply 0 -> 3 tiles (opening), ply 20 -> 8 tiles (mid),
        # ply 40 -> 13 tiles (late; the V1b gate boundary).
        records = [
            _root(1, 0, [3.0, 1.0], serving_index=0),
            _root(1, 20, [3.0, 1.0], serving_index=0),
            _root(1, 40, [3.0, 1.0], serving_index=0),
        ]
        report = analyze(self._write_bank(records))
        phases = [row["phase"] for row in report["per_root"]]
        self.assertEqual(phases, ["opening", "mid", "late"])
        for phase in ("opening", "mid", "late"):
            self.assertEqual(report["by_phase"][phase]["roots"], 1)

    def test_requires_repeats_and_ledger_linkage(self):
        with self.assertRaisesRegex(ValueError, "repeats >= 2"):
            analyze(
                self._write_bank([_root(1, 0, [1.0, 2.0], 0, repeats=1)])
            )
        with self.assertRaisesRegex(ValueError, "ledger_chosen_action_id"):
            analyze(self._write_bank([_root(1, 0, [1.0, 2.0], 0, ledger_id="")]))
        with self.assertRaisesRegex(ValueError, "menu drift"):
            analyze(
                self._write_bank(
                    [_root(1, 0, [1.0, 2.0], 0, ledger_id="sha256:not-in-menu")]
                )
            )

    def test_markdown_summary_renders(self):
        records = [
            _root(1, 0, [5.0, 3.0], serving_index=1),
            _root(2, 40, [3.0, 1.0], serving_index=0),
        ]
        report = analyze(self._write_bank(records))
        out = Path(tempfile.mkdtemp()) / "summary.md"
        write_markdown(report, out)
        text = out.read_text(encoding="utf-8")
        self.assertIn("D1 Label-Movement Pilot", text)
        self.assertIn("PASS", text)
        self.assertIn("| late | 1 |", text)


if __name__ == "__main__":
    unittest.main()
