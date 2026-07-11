"""Contract tests for the R1.1a contention-audit analyzer."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.analyze_table_contention import analyze


def decision(seed, ply, delta, sacrifice, chosen_index=0, model_best_index=0):
    return {
        "type": "contention_decision",
        "ruleset_id": "rules-test",
        "seed": seed,
        "ply": ply,
        "active_seat": ply % 4,
        "action_count": 10,
        "free_three_of_a_kind_choice": "not_available",
        "full_menu_fallback": False,
        "chosen": {"index": chosen_index, "action_id": "a", "model_q": 90.0},
        "model_best": {"index": model_best_index, "action_id": "b", "model_q": 90.5},
        "runner": {"index": 1, "action_id": "c", "model_q": 90.0 - sacrifice},
        "chosen_table": 320.0,
        "chosen_table_exact": False,
        "runner_table": 320.0 + delta,
        "runner_table_exact": False,
        "table_delta_runner_minus_chosen": delta,
        "own_q_sacrifice_chosen_minus_runner": sacrifice,
    }


def write_audit(tmp, rows):
    path = Path(tmp) / "audit.jsonl"
    records = rows + [
        {
            "type": "contention_summary",
            "seeds": len({row["seed"] for row in rows}),
            "decisions_audited": len(rows),
            "single_action_skipped": 0,
        }
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


class AnalyzeTableContentionTest(unittest.TestCase):
    def test_bound_counts_only_positive_deltas_within_epsilon(self) -> None:
        rows = [
            # Seed 1: +2.0 table points available at 0.05 sacrifice (cheap flip),
            # -1.0 (runner worse for the table), +4.0 at a 3.0 sacrifice
            # (expensive flip - excluded at small epsilon).
            decision(1, 0, +2.0, 0.05),
            decision(1, 1, -1.0, 0.10),
            decision(1, 2, +4.0, 3.00),
            # Seed 2: nothing cheap.
            decision(2, 0, -0.5, 0.20),
        ]
        with TemporaryDirectory() as tmp:
            report = analyze(write_audit(tmp, rows), epsilons=(0.25,))
            self.assertEqual(report["games"], 2)
            self.assertEqual(report["decisions"], 4)
            # Unconditional: positive deltas 2.0 + 4.0 over 2 games.
            unconditional = report["unconditional"]
            self.assertAlmostEqual(
                unconditional["recoverable_table_points_per_game"], 3.0
            )
            self.assertAlmostEqual(unconditional["recoverable_gate_points_per_game"], 0.75)
            self.assertAlmostEqual(unconditional["flip_rate"], 0.5)
            # Epsilon 0.25 admits three decisions; only the +2.0 is positive.
            cheap = report["by_epsilon"][0]
            self.assertEqual(cheap["decisions"], 3)
            self.assertAlmostEqual(cheap["recoverable_table_points_per_game"], 1.0)
            self.assertAlmostEqual(cheap["recoverable_gate_points_per_game"], 0.25)

    def test_missing_summary_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text(json.dumps(decision(1, 0, 1.0, 0.1)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete audit"):
                analyze(path)


if __name__ == "__main__":
    unittest.main()
