from __future__ import annotations

import unittest


class PairwiseLabelAuditTest(unittest.TestCase):
    def test_pairwise_summary_separates_margin_and_confidence(self) -> None:
        import numpy as np

        from cascadiav3.torch_pairwise_label_audit import summarize_pairwise_examples

        examples = [
            {
                "target_q": np.asarray([10.0, 9.0, 8.0]),
                "q_valid": np.asarray([True, True, True]),
                "q_variance": np.asarray([1.0, 1.0, 1.0]),
                "q_count": np.asarray([4.0, 4.0, 4.0]),
                "visits": np.asarray([4.0, 4.0, 4.0]),
                "selected_action_index": 0,
            },
            {
                "target_q": np.asarray([5.0, 5.0, 3.0]),
                "q_valid": np.asarray([True, True, False]),
                "q_variance": np.asarray([0.0, 0.0, 1.0]),
                "q_count": np.asarray([2.0, 2.0, 1.0]),
                "visits": np.asarray([2.0, 2.0, 0.0]),
                "selected_action_index": 0,
            },
        ]
        summary = summarize_pairwise_examples(examples, reservoir_size=100)

        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["valid_action_count"], 5)
        self.assertEqual(summary["pair_count"], 4)
        self.assertEqual(summary["tied_pair_count"], 1)
        self.assertEqual(summary["roots_with_at_least_two_valid_actions"], 2)
        self.assertEqual(summary["selected_is_q_best_count"], 2)
        self.assertEqual(summary["absolute_margin"]["fraction_at_least"]["1.0"], 0.75)
        self.assertEqual(
            summary["variance_aware_confidence"]["evaluable_pair_count"], 4
        )
        self.assertEqual(
            summary["variance_aware_confidence"]["zero_standard_error_non_tie_count"],
            0,
        )
        self.assertAlmostEqual(summary["top_two_margin"]["mean"], 0.5)

    def test_one_sample_zero_variance_is_not_treated_as_confident(self) -> None:
        import numpy as np

        from cascadiav3.torch_pairwise_label_audit import summarize_pairwise_examples

        summary = summarize_pairwise_examples(
            [
                {
                    "target_q": np.asarray([2.0, 0.0]),
                    "q_valid": np.asarray([True, True]),
                    "q_variance": np.asarray([0.0, 0.0]),
                    "q_count": np.asarray([1.0, 1.0]),
                    "visits": np.asarray([1.0, 1.0]),
                    "selected_action_index": 0,
                }
            ]
        )
        self.assertEqual(
            summary["variance_aware_confidence"]["evaluable_pair_count"], 0
        )
        self.assertEqual(
            summary["variance_aware_confidence"]["evaluable_pair_fraction"], 0.0
        )

    def test_shape_mismatch_fails_closed(self) -> None:
        import numpy as np

        from cascadiav3.torch_pairwise_label_audit import summarize_pairwise_examples

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            summarize_pairwise_examples(
                [
                    {
                        "target_q": np.asarray([1.0, 0.0]),
                        "q_valid": np.asarray([True]),
                        "q_variance": np.asarray([0.0, 0.0]),
                        "q_count": np.asarray([1.0, 1.0]),
                        "visits": np.asarray([1.0, 1.0]),
                        "selected_action_index": 0,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
