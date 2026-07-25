"""Tests for the direct strength comparator."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.compare_search_shape import build_comparison


def make_report(seed_scores: dict[int, float], **extra: object) -> dict:
    report = {
        "status": "pass",
        "seeds": list(seed_scores),
        "search": {"determinizations": 4},
        "manifest": "checkpoints/model.manifest.json",
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
    report.update(extra)
    return report


def write(root: str, name: str, payload: dict) -> Path:
    path = Path(root) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class CompareSearchShapeTest(unittest.TestCase):
    def test_compares_common_seeds_without_identity_gates(self) -> None:
        with TemporaryDirectory() as root:
            baseline = write(
                root,
                "baseline.json",
                make_report({1: 90.0, 2: 92.0}, source_revision="old"),
            )
            candidate = write(
                root,
                "candidate.json",
                make_report(
                    {2: 93.0, 3: 99.0},
                    source_revision="new",
                    ruleset_id="anything",
                    manifest="another-model.json",
                ),
            )
            result = build_comparison(baseline, candidate)
        self.assertEqual(result["matched_seeds"], [2])
        self.assertEqual(result["paired_delta_stats"]["mean"], 1.0)
        self.assertEqual(result["assessment"], "uncertain")

    def test_reports_candidate_ahead_when_interval_is_positive(self) -> None:
        with TemporaryDirectory() as root:
            baseline = write(
                root, "baseline.json", make_report({seed: 90.0 for seed in range(8)})
            )
            candidate = write(
                root, "candidate.json", make_report({seed: 91.0 for seed in range(8)})
            )
            result = build_comparison(baseline, candidate)
        self.assertEqual(result["assessment"], "candidate_ahead")
        self.assertEqual(result["paired_delta_stats"]["mean"], 1.0)

    def test_requires_at_least_one_common_score(self) -> None:
        with TemporaryDirectory() as root:
            baseline = write(root, "baseline.json", make_report({1: 90.0}))
            candidate = write(root, "candidate.json", make_report({2: 91.0}))
            with self.assertRaisesRegex(ValueError, "no scored seeds in common"):
                build_comparison(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
