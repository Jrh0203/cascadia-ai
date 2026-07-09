"""Selection-block routing for structured-Q learning-rate arms."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cascadiav3.select_structured_q_candidate import select_candidate


class SelectStructuredQCandidateTest(unittest.TestCase):
    def _arm(self, root: Path, label: str, metric: float, lr: float, dataset: str = "same") -> Path:
        checkpoint = root / label / "checkpoints"
        checkpoint.mkdir(parents=True)
        weights = checkpoint / "best.weights.pt"
        weights.write_bytes(f"weights-{label}".encode())
        manifest = {
            "step": 50,
            "weights": weights.name,
        }
        (checkpoint / "best_locked_val.manifest.json").write_text(json.dumps(manifest))
        report = {
            "status": "pass",
            "objective": "gumbel-selfplay-structured-q",
            "selection_metric": "locked_val_q_decomposition",
            "selection_mode": "min",
            "q_decomposition_head_only": True,
            "q_component_initialization": "equal_split_of_loaded_legacy_q",
            "config": {"q_decomposition": True},
            "schema_ids": ["cascadiav3.expert_tensor_shard.v4"],
            "best_selection_metric_value": metric,
            "dataset_manifests": {"train": [{"sha256": dataset}], "val": []},
            "source_hashes": {"trainer": "trainer", "model": "model"},
            "checkpoint_dir": str(checkpoint),
            "optimizer": {"lr": lr},
        }
        path = root / label / "report.json"
        path.write_text(json.dumps(report))
        return path

    def test_lowest_selection_loss_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = {
                "slow": self._arm(root, "slow", 2.0, 3e-4),
                "best": self._arm(root, "best", 1.0, 1e-3),
                "fast": self._arm(root, "fast", 1.5, 3e-3),
            }
            report = select_candidate(arms)
            self.assertEqual(report["chosen"]["label"], "best")
            self.assertEqual(len(report["arms"]), 3)
            self.assertTrue(Path(report["chosen"]["manifest"]).exists())

    def test_dataset_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = {
                "one": self._arm(root, "one", 1.0, 1e-3, dataset="one"),
                "two": self._arm(root, "two", 0.5, 3e-3, dataset="two"),
            }
            with self.assertRaisesRegex(ValueError, "exact data"):
                select_candidate(arms)

    def test_runner_pins_three_way_split_and_learning_rate_grid(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_structured_q_head_pilot.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(': "${FIT_SHA256:?', script)
        self.assertIn(': "${SELECTION_SHA256:?', script)
        self.assertIn(': "${VERDICT_SHA256:?', script)
        self.assertIn("learning_rates=(0.0003 0.001 0.003)", script)
        self.assertIn('--train "$FIT_TAIL"', script)
        self.assertIn('--val "$SELECTION_TAIL"', script)
        self.assertIn("--q-decomposition-head-only", script)
        self.assertIn("--shards \"$VERDICT_TAIL\"", script)
        self.assertIn("no gameplay was launched", script)


if __name__ == "__main__":
    unittest.main()
