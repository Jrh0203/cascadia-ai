"""Contract tests for the R0.2 search-stability analyzer."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.analyze_search_stability import analyze


def search_row(seed, ply, paired, repeat, chosen_index, gap):
    return {
        "type": "stability_search",
        "seed": seed,
        "ply": ply,
        "action_count": 10,
        "paired_rollouts": paired,
        "repeat": repeat,
        "search_seed": 1000 + repeat,
        "chosen_index": chosen_index,
        "chosen_action_id": f"a{chosen_index}",
        "simulations_run": 8,
        "top_overall": [],
        "top_visited": [
            {"index": chosen_index, "action_id": "x", "completed_q": 90.0 + gap, "visits": 4},
            {"index": 9, "action_id": "y", "completed_q": 90.0, "visits": 4},
        ],
    }


def write_probe(tmp, rows):
    path = Path(tmp) / "probe.jsonl"
    records = rows + [
        {
            "type": "stability_summary",
            "sampled_roots": len({(row["seed"], row["ply"]) for row in rows}),
            "repeats_per_variant": 3,
            "stride": 1,
            "search": {"n_simulations": 8},
        }
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


class AnalyzeSearchStabilityTest(unittest.TestCase):
    def test_variance_reduction_and_flip_rates(self) -> None:
        rows = []
        # Root (1, 0): unpaired gaps vary {0.0, 0.6, 0.0} and flip once;
        # paired gaps are constant {0.3} and never flip.
        for repeat, (gap, chosen) in enumerate([(0.0, 0), (0.6, 1), (0.0, 0)]):
            rows.append(search_row(1, 0, False, repeat, chosen, gap))
        for repeat in range(3):
            rows.append(search_row(1, 0, True, repeat, 0, 0.3))
        with TemporaryDirectory() as tmp:
            report = analyze(write_probe(tmp, rows), variance_reduction_floor=0.20)
            self.assertEqual(report["roots"], 1)
            pooled = report["pooled_gap_variance"]
            self.assertGreater(pooled["unpaired"], 0.0)
            self.assertEqual(pooled["paired"], 0.0)
            self.assertAlmostEqual(pooled["reduction"], 1.0)
            self.assertTrue(report["proceed_to_gate"])
            flips = report["flip_rate"]
            self.assertAlmostEqual(flips["unpaired_mean"], 1.0 / 3.0)
            self.assertAlmostEqual(flips["paired_mean"], 0.0)

    def test_no_reduction_does_not_proceed(self) -> None:
        rows = []
        for repeat, gap in enumerate([0.0, 0.6, 0.0]):
            rows.append(search_row(1, 0, False, repeat, 0, gap))
            rows.append(search_row(1, 0, True, repeat, 0, gap))
        with TemporaryDirectory() as tmp:
            report = analyze(write_probe(tmp, rows), variance_reduction_floor=0.20)
            self.assertAlmostEqual(report["pooled_gap_variance"]["reduction"], 0.0)
            self.assertFalse(report["proceed_to_gate"])

    def test_missing_summary_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.jsonl"
            path.write_text(
                json.dumps(search_row(1, 0, False, 0, 0, 0.1)) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incomplete probe"):
                analyze(path)


if __name__ == "__main__":
    unittest.main()
