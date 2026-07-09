"""Exact-grounded, action-conditioned score-to-go decomposition."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest


class StructuredQTest(unittest.TestCase):
    def _require_torch(self):  # type: ignore[no-untyped-def]
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")

    def _forward(self, *, quantiles: int = 1):  # type: ignore[no-untyped-def]
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        config = replace(
            config_for_size("tiny"),
            q_decomposition=True,
            q_quantiles=quantiles,
        )
        torch.manual_seed(19)
        model = build_cascadiaformer(config).eval()
        with torch.inference_mode():
            outputs = model(
                torch.randn(2, 6, config.token_feature_dim),
                torch.ones(2, 6, dtype=torch.bool),
                torch.randn(2, 5, config.action_feature_dim),
                torch.ones(2, 5, dtype=torch.bool),
            )
        return config, model, outputs

    def test_disabled_mode_preserves_legacy_parameter_contract(self) -> None:
        self._require_torch()
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        model = build_cascadiaformer(config_for_size("tiny"))
        self.assertIsNone(model.q_component_head)
        self.assertFalse(any(name.startswith("q_component_head.") for name in model.state_dict()))

    def test_component_means_sum_exactly_to_served_score_to_go(self) -> None:
        self._require_torch()
        import torch

        _, _, outputs = self._forward()
        self.assertEqual(outputs["q_score_to_go_components"].shape, (2, 5, 3))
        torch.testing.assert_close(
            outputs["q"],
            outputs["q_score_to_go_components"].sum(dim=-1),
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotIn("q_quantile_values", outputs)

    def test_distributional_components_preserve_total_quantile_contract(self) -> None:
        self._require_torch()
        import torch

        _, _, outputs = self._forward(quantiles=8)
        self.assertEqual(outputs["q_component_quantile_values"].shape, (2, 5, 3, 8))
        torch.testing.assert_close(
            outputs["q_quantile_values"],
            outputs["q_component_quantile_values"].sum(dim=-2),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            outputs["q"],
            outputs["q_quantile_values"].mean(dim=-1),
        )

    def test_selected_action_component_loss_uses_terminal_minus_afterstate(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import LossWeights, _loss_components

        target_score = torch.zeros((1, 3, 4), dtype=torch.float32)
        target_score[0, :, 2] = torch.tensor([5.0, 7.0, 9.0])
        exact_components = torch.zeros((1, 2, 3), dtype=torch.float32)
        exact_components[0, 1] = torch.tensor([1.0, 2.0, 3.0])
        predictions = torch.zeros((1, 2, 3), dtype=torch.float32)
        predictions[0, 1] = torch.tensor([4.0, 5.0, 6.0])
        outputs = {
            "logits": torch.zeros((1, 2), dtype=torch.float32),
            "q": predictions.sum(dim=-1),
            "q_score_to_go_components": predictions,
            "uncertainty": torch.ones((1, 2), dtype=torch.float32),
            "value_vector": torch.zeros((1, 4), dtype=torch.float32),
            "rank_logits": torch.zeros((1, 4, 4), dtype=torch.float32),
            "score_decomposition": torch.zeros((1, 3, 4), dtype=torch.float32),
        }
        batch = {
            "action_mask": torch.ones((1, 2), dtype=torch.bool),
            "q_valid": torch.zeros((1, 2), dtype=torch.bool),
            "selected_action_index": torch.tensor([1]),
            "greedy_action_index": torch.tensor([0]),
            "target_q": torch.tensor([[0.0, 21.0]]),
            "target_score_to_go": torch.tensor([[0.0, 15.0]]),
            "exact_afterstate_score_active": torch.tensor([[0.0, 6.0]]),
            "exact_afterstate_score_decomposition_active": exact_components,
            "active_seat": torch.tensor([2]),
            "target_value": torch.tensor([[0.0, 0.0, 21.0, 0.0]]),
            "target_rank": torch.tensor([[0, 1, 2, 3]]),
            "target_score": target_score,
            "has_improved_policy": False,
        }
        weights = LossWeights(
            policy=0.0,
            q=0.0,
            value=0.0,
            score=0.0,
            rank=0.0,
            uncertainty=0.0,
            q_decomposition=1.0,
        )
        losses = _loss_components(outputs, batch, weights)
        self.assertEqual(float(losses["q_decomposition"]), 0.0)
        self.assertEqual(float(losses["total"]), 0.0)

        outputs["q_score_to_go_components"] = predictions + torch.tensor([[[0.0], [1.0]]])
        losses = _loss_components(outputs, batch, weights)
        self.assertAlmostEqual(float(losses["q_decomposition"]), 0.5, places=6)
        self.assertAlmostEqual(float(losses["total"]), 0.5, places=6)

    def test_head_only_exposes_only_component_projection(self) -> None:
        self._require_torch()
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_train_cascadiaformer import _configure_q_decomposition_head_only

        config = replace(config_for_size("tiny"), q_decomposition=True, q_quantiles=8)
        model = build_cascadiaformer(config)
        trainable = _configure_q_decomposition_head_only(model)
        self.assertEqual(trainable, (config.d_model + 1) * 3 * config.q_quantiles)
        for name, parameter in model.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                name.startswith("q_component_head."),
                name,
            )

    def test_inference_bridge_loads_structured_checkpoint_contract(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _load_model

        config = replace(config_for_size("tiny"), q_decomposition=True, q_quantiles=4)
        model = build_cascadiaformer(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            manifest = root / "manifest.json"
            torch.save(model.state_dict(), weights)
            payload = {
                "config": config.to_dict(),
                "weights": weights.name,
                "weights_format": "torch_state_dict",
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            loaded = _load_model(
                manifest,
                manifest_path=manifest,
                manifest_payload=payload,
                device_name="cpu",
            )
        self.assertTrue(loaded.config.q_decomposition)
        self.assertEqual(loaded.config.q_quantiles, 4)
        self.assertIsNotNone(loaded.q_component_head)

    def test_schema_gate_rejects_jsonl_and_pre_v4_shards(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import _validate_q_decomposition_corpora

        with self.assertRaisesRegex(ValueError, "packed NPZ"):
            _validate_q_decomposition_corpora(
                enabled=True,
                train_format="jsonl",
                val_format="npz",
                schema_ids=["cascadiav3.expert_tensor_shard.v4"],
            )
        with self.assertRaisesRegex(ValueError, "v4 exact-grounded"):
            _validate_q_decomposition_corpora(
                enabled=True,
                train_format="npz",
                val_format="npz",
                schema_ids=["cascadiav3.expert_tensor_shard.v3"],
            )
        _validate_q_decomposition_corpora(
            enabled=True,
            train_format="npz",
            val_format="npz",
            schema_ids=["cascadiav3.expert_tensor_shard.v4"],
        )


if __name__ == "__main__":
    unittest.main()
