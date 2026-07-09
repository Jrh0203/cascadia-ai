"""Verdict math for the exact-grounded structured-Q held-out gate."""

from __future__ import annotations

import unittest

import numpy as np

from cascadiav3.torch_structured_q_probe import summarize_structured_q_observations


class StructuredQProbeTest(unittest.TestCase):
    def _summary(self, candidate_offset: float):  # type: ignore[no-untyped-def]
        real = np.asarray([10.0, 20.0, 30.0, 40.0])
        target_components = np.asarray(
            [[2.0, 7.0, 1.0], [4.0, 14.0, 2.0], [6.0, 21.0, 3.0], [8.0, 28.0, 4.0]]
        )
        return summarize_structured_q_observations(
            target_components=target_components,
            candidate_components=target_components + candidate_offset / 3.0,
            real_final=real,
            candidate_selected=real + candidate_offset,
            incumbent_selected=real + 2.0,
            teacher_selected=real + 3.0,
            candidate_all_q_errors=np.asarray([1.0, -1.0, 1.0, -1.0]),
            incumbent_all_q_errors=np.asarray([1.0, -1.0, 1.0, -1.0]),
            candidate_q_regret=np.zeros(4),
            incumbent_q_regret=np.zeros(4),
            candidate_q_top1=np.ones(4),
            incumbent_q_top1=np.ones(4),
            seed=7,
        )

    def test_pass_requires_primary_improvement_and_retention(self) -> None:
        report = self._summary(0.0)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["selected_real_outcome"]["best_baseline"], "incumbent_model_q")
        self.assertEqual(report["selected_real_outcome"]["candidate"]["rmse"], 0.0)
        self.assertTrue(all(report["gates"].values()))

    def test_worse_selected_prediction_fails_gate(self) -> None:
        report = self._summary(4.0)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["gates"]["selected_rmse_improvement_at_least_10pct"])
        self.assertFalse(report["gates"]["paired_absolute_error_ci_excludes_zero"])

    def test_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "share"):
            summarize_structured_q_observations(
                target_components=np.zeros((2, 3)),
                candidate_components=np.zeros((2, 2)),
                real_final=np.zeros(2),
                candidate_selected=np.zeros(2),
                incumbent_selected=np.ones(2),
                teacher_selected=np.ones(2),
                candidate_all_q_errors=np.zeros(2),
                incumbent_all_q_errors=np.ones(2),
                candidate_q_regret=np.zeros(2),
                incumbent_q_regret=np.zeros(2),
                candidate_q_top1=np.ones(2),
                incumbent_q_top1=np.ones(2),
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
