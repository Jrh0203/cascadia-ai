from __future__ import annotations

import unittest
from dataclasses import replace
import json
from pathlib import Path
import tempfile


class PairwiseComparatorTest(unittest.TestCase):
    def _batch(self):  # type: ignore[no-untyped-def]
        import torch

        from cascadiav3.torch_cascadiaformer import config_for_size

        config = replace(
            config_for_size("tiny"),
            pairwise_comparator=True,
            pairwise_rank=8,
            pairwise_max_pairs_per_root=4,
            pairwise_min_margin=0.25,
            pairwise_min_snr=1.0,
        )
        batch = {
            "tokens": torch.randn(1, 3, config.token_feature_dim),
            "token_mask": torch.ones(1, 3, dtype=torch.bool),
            "actions": torch.randn(1, 3, config.action_feature_dim),
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "target_q": torch.tensor([[3.0, 2.0, 1.0]]),
            "target_score_to_go": torch.tensor([[2.0, 1.0, 0.0]]),
            "q_valid": torch.ones(1, 3, dtype=torch.bool),
            "target_q_variance": torch.tensor([[1.0, 1.0, 0.0]]),
            "target_q_count": torch.tensor([[4.0, 4.0, 1.0]]),
            "target_truncated_count": torch.zeros(1, 3),
            "visits": torch.tensor([[4.0, 4.0, 1.0]]),
            "exact_afterstate_score_active": torch.ones(1, 3),
            "selected_action_index": torch.tensor([0]),
            "greedy_action_index": torch.tensor([0]),
            "target_value": torch.zeros(1, 4),
            "target_rank": torch.tensor([[0, 1, 2, 3]]),
            "target_score": torch.zeros(1, 3, 4),
            "has_improved_policy": True,
            "improved_policy": torch.tensor([[0.6, 0.3, 0.1]]),
            "search_root_value": torch.tensor([3.0]),
        }
        return config, batch

    def test_comparator_is_antisymmetric_and_self_comparison_is_zero(self) -> None:
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer

        config, batch = self._batch()
        model = build_cascadiaformer(config).eval()
        with torch.no_grad():
            outputs = model(
                batch["tokens"],
                batch["token_mask"],
                batch["actions"],
                batch["action_mask"],
                pairwise_root_indices=torch.tensor([0, 0, 0]),
                pairwise_left_indices=torch.tensor([0, 1, 2]),
                pairwise_right_indices=torch.tensor([1, 0, 2]),
                return_pairwise_borda=True,
            )
        logits = outputs["pairwise_logits"]
        self.assertAlmostEqual(float(logits[0] + logits[1]), 0.0, places=6)
        self.assertEqual(float(logits[2]), 0.0)
        borda = outputs["pairwise_borda_logits"]
        self.assertAlmostEqual(float(borda.sum()), 0.0, places=6)

        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            permuted = model(
                batch["tokens"],
                batch["token_mask"],
                batch["actions"][:, permutation],
                batch["action_mask"][:, permutation],
                return_pairwise_borda=True,
            )["pairwise_borda_logits"]
        self.assertTrue(torch.allclose(permuted, borda[:, permutation], atol=1.0e-6))

    def test_supervision_excludes_one_sample_pairs_and_emits_both_directions(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import _add_pairwise_supervision

        config, batch = self._batch()
        _add_pairwise_supervision(batch, config)

        self.assertEqual(batch["pairwise_root_indices"].tolist(), [0, 0])
        self.assertEqual(batch["pairwise_left_indices"].tolist(), [0, 1])
        self.assertEqual(batch["pairwise_right_indices"].tolist(), [1, 0])
        self.assertEqual(batch["pairwise_targets"].tolist(), [1.0, 0.0])
        self.assertAlmostEqual(float(batch["pairwise_snr"][0]), 2**0.5, places=6)
        self.assertEqual(batch["pairwise_snr"][0], batch["pairwise_snr"][1])

    def test_pairwise_objective_backpropagates_into_skew_head(self) -> None:
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer
        from cascadiav3.torch_train_cascadiaformer import (
            _add_pairwise_supervision,
            _loss_components,
            _model_forward,
            loss_weights_for_objective,
        )

        config, batch = self._batch()
        model = build_cascadiaformer(config).train()
        _add_pairwise_supervision(batch, config)
        outputs = _model_forward(model, batch)
        losses = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay-pairwise"),
        )
        self.assertGreater(float(losses["pairwise"]), 0.0)
        self.assertEqual(float(losses["pairwise_examples"]), 2.0)
        losses["total"].backward()
        self.assertIsNotNone(model.pairwise_left.weight.grad)
        self.assertGreater(float(model.pairwise_left.weight.grad.abs().sum()), 0.0)

    def test_disabled_comparator_preserves_legacy_parameter_contract(self) -> None:
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        model = build_cascadiaformer(config_for_size("tiny"))
        self.assertIsNone(model.pairwise_merit)
        self.assertIsNone(model.pairwise_left)
        self.assertIsNone(model.pairwise_right)

    def test_head_only_mode_freezes_every_legacy_parameter(self) -> None:
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer
        from cascadiav3.torch_train_cascadiaformer import _configure_pairwise_head_only

        config, _ = self._batch()
        model = build_cascadiaformer(config)
        trainable = _configure_pairwise_head_only(model)
        expected = config.d_model + 2 * config.d_model * config.pairwise_rank
        self.assertEqual(trainable, expected)
        for name, parameter in model.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                name.startswith(("pairwise_merit.", "pairwise_left.", "pairwise_right.")),
                name,
            )

    def test_bridge_policy_mode_is_explicit_and_changes_only_priors(self) -> None:
        import base64

        import numpy as np

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer
        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
            default_model_service_command,
        )
        from cascadiav3.torch_inference_bridge import (
            _model_eval,
            validate_policy_mode_manifest,
        )

        config, _ = self._batch()
        model = build_cascadiaformer(config).eval()
        token_count = 2
        action_count = 3
        tokens = np.zeros((token_count, config.token_feature_dim), dtype="<f4")
        actions = np.arange(
            action_count * config.action_feature_dim,
            dtype="<f4",
        ).reshape(action_count, config.action_feature_dim)
        relation_tail = np.zeros(
            (action_count, token_count + action_count),
            dtype=np.uint8,
        )
        root = {
            "schema_id": "pairwise-bridge-test",
            "state_hash": "pairwise-bridge-test-root",
            "active_seat": 0,
            "action_ids": ["a", "b", "c"],
            "exact_afterstate_score_active": [10.0, 11.0, 12.0],
            "packed_features": {
                "token_count": token_count,
                "action_count": action_count,
                "token_feature_dim": config.token_feature_dim,
                "action_feature_dim": config.action_feature_dim,
                "tokens_f32_b64": base64.b64encode(tokens.tobytes()).decode("ascii"),
                "actions_f32_b64": base64.b64encode(actions.tobytes()).decode("ascii"),
                "relation_tail_u8_b64": base64.b64encode(relation_tail.tobytes()).decode(
                    "ascii"
                ),
            },
        }
        logits_response = _model_eval(model, root, policy_mode="logits")
        pairwise_response = _model_eval(model, root, policy_mode="pairwise-borda")

        self.assertEqual(logits_response["q"], pairwise_response["q"])
        self.assertNotEqual(logits_response["priors"], pairwise_response["priors"])
        self.assertAlmostEqual(sum(pairwise_response["priors"]), 1.0, places=6)
        validate_policy_mode_manifest(
            {"config": {"pairwise_comparator": True}},
            "pairwise-borda",
        )
        with self.assertRaisesRegex(ValueError, "requires a pairwise-comparator"):
            validate_policy_mode_manifest(
                {"config": {"pairwise_comparator": False}},
                "pairwise-borda",
            )
        command = default_model_service_command(
            Path("model.json"),
            "cpu",
            policy_mode="pairwise-borda",
        )
        self.assertIn("--policy-mode pairwise-borda", command)

    def test_offline_probe_runs_end_to_end_on_v3_shard(self) -> None:
        import torch

        from test_expert_tensor_v3 import ExpertTensorV3Test

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_pairwise_policy_probe import run_probe

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = root / "val.npz"
            ExpertTensorV3Test()._write_v3(tensor)
            config = replace(
                config_for_size("tiny"),
                pairwise_comparator=True,
                pairwise_rank=8,
                pairwise_max_pairs_per_root=4,
                pairwise_min_margin=0.25,
                pairwise_min_snr=1.0,
            )
            model = build_cascadiaformer(config)
            weights = root / "weights.pt"
            torch.save(model.state_dict(), weights)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoint_tag": "test",
                        "step": 0,
                        "config": config.to_dict(),
                        "weights": weights.name,
                        "weights_format": "torch_state_dict",
                    }
                ),
                encoding="utf-8",
            )
            report = run_probe(
                manifest=manifest,
                tensors=[tensor],
                device_name="cpu",
                batch_size=1,
                max_records=0,
            )
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["eligible_policy_root_count"], 1)
            self.assertEqual(report["pairwise"]["directed_pair_count"], 2)
            self.assertEqual(set(report["policy_modes"]), {
                "logits",
                "pairwise-borda",
                "logits-plus-pairwise",
            })


if __name__ == "__main__":
    unittest.main()
