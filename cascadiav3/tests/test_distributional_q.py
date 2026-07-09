"""Distributional (quantile) score-to-go head: shapes, mean identity, loss."""

from __future__ import annotations

import unittest


class DistributionalQTest(unittest.TestCase):
    def _require_torch(self):  # type: ignore[no-untyped-def]
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")

    def _forward(self, cfg):  # type: ignore[no-untyped-def]
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer

        torch.manual_seed(7)
        model = build_cascadiaformer(cfg).eval()
        batch, seq, actions = 2, 6, 5
        tokens = torch.randn(batch, seq, cfg.token_feature_dim)
        token_mask = torch.ones(batch, seq, dtype=torch.bool)
        action_feats = torch.randn(batch, actions, cfg.action_feature_dim)
        action_mask = torch.ones(batch, actions, dtype=torch.bool)
        with torch.inference_mode():
            return model(tokens, token_mask, action_feats, action_mask)

    def test_scalar_head_has_no_quantile_output(self) -> None:
        self._require_torch()
        from cascadiav3.torch_cascadiaformer import config_for_size

        out = self._forward(config_for_size("tiny"))
        self.assertNotIn("q_quantile_values", out)
        self.assertEqual(out["q"].dim(), 2)

    def test_quantile_head_mean_is_served_q(self) -> None:
        self._require_torch()
        import dataclasses

        import torch

        from cascadiav3.torch_cascadiaformer import config_for_size

        cfg = dataclasses.replace(config_for_size("tiny"), q_quantiles=8)
        out = self._forward(cfg)
        self.assertEqual(out["q_quantile_values"].shape[-1], 8)
        torch.testing.assert_close(out["q"], out["q_quantile_values"].mean(dim=-1))

    def test_risk_modes_interpolate_centered_quantiles_and_fail_on_scalar(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_inference_bridge import (
            select_score_to_go_for_risk,
            serve,
            validate_q_risk_manifest,
        )

        quantiles = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        outputs = {"q": quantiles.mean(dim=-1), "q_quantile_values": quantiles}
        self.assertEqual(float(select_score_to_go_for_risk(outputs, "mean")), 3.5)
        self.assertEqual(float(select_score_to_go_for_risk(outputs, "q25")), 1.5)
        self.assertEqual(float(select_score_to_go_for_risk(outputs, "q50")), 3.5)
        self.assertEqual(float(select_score_to_go_for_risk(outputs, "q75")), 5.5)
        with self.assertRaisesRegex(ValueError, "distributional-Q checkpoint"):
            select_score_to_go_for_risk({"q": torch.ones(1, 1)}, "q25")
        with self.assertRaisesRegex(ValueError, "unsupported q risk mode"):
            select_score_to_go_for_risk(outputs, "optimistic-ish")
        validate_q_risk_manifest({"config": {"q_quantiles": 8}}, "q25")
        validate_q_risk_manifest({"config": {"q_quantiles": 1}}, "mean")
        with self.assertRaisesRegex(ValueError, "distributional-Q checkpoint"):
            validate_q_risk_manifest({"config": {"q_quantiles": 1}}, "q25")

        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "scalar.manifest.json"
            manifest.write_text(
                json.dumps({"config": {"q_quantiles": 1}}), encoding="utf-8"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return_code = serve(
                    checkpoint=None,
                    manifest=manifest,
                    allow_dry_run_fallback=False,
                    q_risk_mode="q25",
                )
            self.assertEqual(return_code, 2)
            error = json.loads(output.getvalue())
            self.assertEqual(error["type"], "error")
            self.assertIn("distributional-Q checkpoint", error["error"])

        crossed = torch.tensor([[[7.0, 0.0, 6.0, 1.0, 5.0, 2.0, 4.0, 3.0]]])
        crossed_outputs = {
            "q": crossed.mean(dim=-1),
            "q_quantile_values": crossed,
        }
        self.assertEqual(float(select_score_to_go_for_risk(crossed_outputs, "q25")), 1.5)
        self.assertEqual(float(select_score_to_go_for_risk(crossed_outputs, "q50")), 3.5)
        self.assertEqual(float(select_score_to_go_for_risk(crossed_outputs, "q75")), 5.5)
        self.assertEqual(
            float(select_score_to_go_for_risk(crossed_outputs, "mean")), 3.5
        )

    def test_gumbel_default_service_records_risk_mode(self) -> None:
        from pathlib import Path

        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
            default_model_service_command,
        )

        command = default_model_service_command(Path("checkpoint.json"), "mps", "q25")
        self.assertIn("--q-risk-mode q25", command)
        with self.assertRaisesRegex(ValueError, "unsupported q risk mode"):
            default_model_service_command(Path("checkpoint.json"), "cpu", "tail")

    def test_q_risk_probe_reports_crossing_and_policy_flips(self) -> None:
        from cascadiav3.torch_q_risk_probe import summarize_q_risk_rows

        quantiles = [
            [0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0],
            [3.0] * 8,
            [7.0, 0.0, 6.0, 1.0, 5.0, 2.0, 4.0, 3.0],
        ]
        summary = summarize_q_risk_rows(
            [
                {
                    "state_hash": "fixed-root",
                    "action_ids": ["wide", "safe", "crossed"],
                    "exact_afterstate_score_active": [0.0, 0.0, 0.0],
                    "served_mean": [4.0, 3.0, 3.5],
                    "quantiles": quantiles,
                }
            ]
        )
        crossing = summary["raw_quantile_crossing"]
        self.assertEqual(crossing["crossing_pair_count"], 4)
        self.assertAlmostEqual(crossing["crossing_pair_rate"], 4 / 21)
        self.assertAlmostEqual(crossing["action_crossing_rate"], 1 / 3)
        self.assertEqual(
            summary["serving_modes"]["q25"]["direct_argmax_flip_count_vs_mean"],
            1,
        )
        self.assertEqual(
            summary["serving_modes"]["q25"]["flip_examples"][0]["risk_action_id"],
            "safe",
        )
        self.assertEqual(
            summary["serving_modes"]["q50"]["direct_argmax_flip_count_vs_mean"],
            0,
        )
        self.assertEqual(
            summary["serving_modes"]["q75"]["direct_argmax_flip_count_vs_mean"],
            0,
        )

    def test_pinball_loss_path_runs_and_is_minimized_at_target(self) -> None:
        self._require_torch()
        import torch

        # Standalone pinball identity check: for a degenerate (constant)
        # target the loss is zero exactly when every quantile equals it.
        levels = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
        target = torch.full((3,), 2.0)
        exact = torch.full((3, 4), 2.0)
        residual = target.unsqueeze(-1) - exact
        loss = torch.maximum(levels * residual, (levels - 1.0) * residual).mean()
        self.assertEqual(float(loss), 0.0)
        off = exact + 1.0
        residual_off = target.unsqueeze(-1) - off
        loss_off = torch.maximum(levels * residual_off, (levels - 1.0) * residual_off).mean()
        self.assertGreater(float(loss_off), 0.0)

    def test_loss_components_accepts_quantile_outputs(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        batch_size, actions, quantiles = 2, 3, 4
        torch.manual_seed(11)
        outputs = {
            "logits": torch.randn(batch_size, actions),
            "q": torch.randn(batch_size, actions),
            "q_quantile_values": torch.randn(batch_size, actions, quantiles),
            "uncertainty": torch.rand(batch_size, actions),
            "value_vector": torch.randn(batch_size, 4),
            "rank_logits": torch.randn(batch_size, 4, 4),
            "score_decomposition": torch.randn(batch_size, 3, 4),
        }
        batch = {
            "action_mask": torch.ones(batch_size, actions, dtype=torch.bool),
            "q_valid": torch.ones(batch_size, actions, dtype=torch.bool),
            "selected_action_index": torch.zeros(batch_size, dtype=torch.long),
            "greedy_action_index": torch.zeros(batch_size, dtype=torch.long),
            "target_q": torch.randn(batch_size, actions),
            "target_score_to_go": torch.randn(batch_size, actions),
            "exact_afterstate_score_active": torch.zeros(batch_size, actions),
            "target_value": torch.randn(batch_size, 4),
            "target_rank": torch.zeros(batch_size, 4, dtype=torch.long),
            "target_score": torch.randn(batch_size, 3, 4),
        }
        losses = _loss_components(outputs, batch, loss_weights_for_objective("gumbel-selfplay"))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(losses["q"]))


if __name__ == "__main__":
    unittest.main()
