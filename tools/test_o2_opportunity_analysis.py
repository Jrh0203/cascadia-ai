import unittest

import numpy as np

from o2_opportunity_analysis import _fit_ridge, _predict_ridge, select_index


class O2OpportunityAnalysisTests(unittest.TestCase):
    def test_ridge_recovers_linear_residual(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        target = 2.0 + 3.0 * features[:, 0]
        predicted = _predict_ridge(features, _fit_ridge(features, target, 0.01))
        self.assertLess(float(np.mean((predicted - target) ** 2)), 1e-3)

    def test_selection_uses_hash_for_exact_ties(self) -> None:
        self.assertEqual(select_index(np.asarray([1.0, 1.0]), ["b", "a"]), 1)

    def test_selection_rejects_nonfinite_scores(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "selection inputs"):
            select_index(np.asarray([1.0, np.nan]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
