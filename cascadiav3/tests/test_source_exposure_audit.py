"""Fail-closed per-source draw exposure audit (pure python, no torch)."""

import unittest

from cascadiav3.torch_train_cascadiaformer import audit_source_exposure


class SourceExposureAuditTest(unittest.TestCase):
    def test_d1_recipe_exposure_matches_preregistration(self):
        # Stage A base : cycle4 : cycle3 : D1 at 4:2:1:1 over 2,500 steps
        # of batch 192 -> 12.5% D1 share, four expected passes per root.
        report = audit_source_exposure(
            source_lengths=[100_000, 100_000, 100_000, 15_000],
            source_weights=[4.0, 2.0, 1.0, 1.0],
            seed=20260630,
            batch_size=192,
            total_batches=2_500,
            tolerance=0.02,
        )
        self.assertEqual(report["total_draws"], 480_000)
        self.assertAlmostEqual(report["source_shares"][3], 0.125, delta=0.01)
        self.assertAlmostEqual(
            report["expected_passes_per_record"][3], 4.0, delta=0.35
        )
        self.assertEqual(report["failures"], [])

    def test_zero_draw_positive_weight_source_fails(self):
        # One batch of one draw cannot cover four positive-weight sources.
        with self.assertRaisesRegex(ValueError, "exposure audit FAILED"):
            audit_source_exposure(
                source_lengths=[10, 10, 10, 10],
                source_weights=[1.0, 1.0, 1.0, 1.0],
                seed=1,
                batch_size=1,
                total_batches=1,
            )

    def test_share_deviation_beyond_tolerance_fails(self):
        # Three draws over two equal-weight sources can never split 50/50,
        # so the minimum deviation is 1/6 — far beyond the strict tolerance.
        with self.assertRaises(ValueError):
            audit_source_exposure(
                source_lengths=[10, 10],
                source_weights=[1.0, 1.0],
                seed=3,
                batch_size=3,
                total_batches=1,
                tolerance=0.001,
            )

    def test_audit_is_deterministic(self):
        kwargs = dict(
            source_lengths=[1_000, 500],
            source_weights=[3.0, 1.0],
            seed=99,
            batch_size=64,
            total_batches=50,
        )
        self.assertEqual(
            audit_source_exposure(**kwargs)["source_draws"],
            audit_source_exposure(**kwargs)["source_draws"],
        )


if __name__ == "__main__":
    unittest.main()
