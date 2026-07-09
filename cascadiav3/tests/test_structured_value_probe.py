from __future__ import annotations

import unittest

import numpy as np


class StructuredValueProbeTest(unittest.TestCase):
    def test_action_query_encoder_is_the_forward_representation(self) -> None:
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        torch.manual_seed(7)
        model = build_cascadiaformer(config_for_size("tiny"))
        model.eval()
        tokens = torch.randn(2, 5, model.config.token_feature_dim)
        token_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        actions = torch.randn(2, 4, model.config.action_feature_dim)
        action_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        relation_tail = torch.randint(0, 8, (2, 4, 9), dtype=torch.uint8)
        captured = []
        handle = model.q_head.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            with torch.inference_mode():
                outputs = model(
                    tokens,
                    token_mask,
                    actions,
                    action_mask,
                    relation_tail=relation_tail,
                )
                root_h, decoded, cgab_bias = model.encode_action_queries(
                    tokens,
                    token_mask,
                    actions,
                    action_mask,
                    relation_tail=relation_tail,
                )
                selected_root_h, selected_decoded, _ = model.encode_action_queries(
                    tokens,
                    token_mask,
                    actions[:, 2:3],
                    action_mask[:, 2:3],
                    relation_tail=relation_tail[:, 2:3],
                )
        finally:
            handle.remove()
        torch.testing.assert_close(decoded, captured[0], rtol=0.0, atol=0.0)
        with torch.inference_mode():
            encoded_value = model.value_head(root_h)
        torch.testing.assert_close(encoded_value, outputs["value_vector"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(cgab_bias, outputs["cgab_bias"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(selected_root_h, root_h, rtol=0.0, atol=0.0)
        torch.testing.assert_close(selected_decoded[:, 0], decoded[:, 2], rtol=1.0e-5, atol=1.0e-6)

    def test_selected_record_preserves_complete_relation_row(self) -> None:
        from cascadiav3.torch_structured_value_probe import _selected_record

        actions = np.zeros((3, 61), dtype=np.float32)
        actions[:, 0] = np.float32(1.0 / 3.0)
        example = {
            "tokens": np.zeros((2, 41), dtype=np.float32),
            "actions": actions,
            "selected_action_index": 1,
            "relation_edges": np.asarray(
                [
                    [3, 0, 5],
                    [3, 4, 7],
                    [2, 0, 9],
                ],
                dtype=np.int32,
            ),
            "score_decomposition": np.asarray(
                [
                    [10, 20, 30, 40],
                    [11, 21, 31, 41],
                    [1, 2, 3, 4],
                ],
                dtype=np.float32,
            ),
            "final_score_vector": np.asarray([22, 43, 64, 85], dtype=np.float32),
            "exact_afterstate_score_active": np.asarray([15, 16, 17], dtype=np.float32),
            "target_q": np.asarray([50, 51, 52], dtype=np.float32),
        }
        record = _selected_record(example)
        self.assertEqual(record["active_seat"], 1)
        self.assertEqual(record["action_count"], 3)
        np.testing.assert_array_equal(
            record["relation_ids"], np.asarray([5, 0, 0, 0, 7], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            record["categories"], np.asarray([20, 21, 2], dtype=np.float32)
        )
        self.assertEqual(record["final_score"], 43.0)
        self.assertEqual(record["exact_afterstate"], 16.0)
        self.assertEqual(record["teacher_q"], 51.0)

    def test_ridge_and_error_helpers_fail_closed(self) -> None:
        from cascadiav3.torch_structured_value_probe import _error_stats, _fit_ridge

        design = np.asarray(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64
        )
        targets = np.stack((2.0 * design[:, 0] + 1.0, -design[:, 0] + 4.0), axis=1)
        coefficients = _fit_ridge(design, targets, 0.01)
        prediction = design @ coefficients
        self.assertLess(float(np.sqrt(np.mean((prediction - targets) ** 2))), 0.01)
        stats = _error_stats(prediction[:, 0], targets[:, 0])
        self.assertEqual(stats["n"], 4)
        self.assertLess(stats["rmse"], 0.01)
        with self.assertRaisesRegex(ValueError, "positive"):
            _fit_ridge(design, targets, 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            _error_stats(np.asarray([np.nan]), np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()
