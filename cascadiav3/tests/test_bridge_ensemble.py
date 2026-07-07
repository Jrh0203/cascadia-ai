"""Ensemble serving must equal the mean of its members' head outputs."""

from __future__ import annotations

import unittest


class BridgeEnsembleTest(unittest.TestCase):
    def _require_torch(self):  # type: ignore[no-untyped-def]
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")

    def test_ensemble_forward_is_member_mean(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _EnsembleModel

        cfg = config_for_size("tiny")
        torch.manual_seed(1)
        model_a = build_cascadiaformer(cfg).eval()
        torch.manual_seed(2)
        model_b = build_cascadiaformer(cfg).eval()

        batch, seq, actions = 2, 7, 5
        tokens = torch.randn(batch, seq, cfg.token_feature_dim)
        token_mask = torch.ones(batch, seq, dtype=torch.bool)
        action_feats = torch.randn(batch, actions, cfg.action_feature_dim)
        action_mask = torch.ones(batch, actions, dtype=torch.bool)

        ensemble = _EnsembleModel([model_a, model_b])
        with torch.inference_mode():
            out_a = model_a(tokens, token_mask, action_feats, action_mask)
            out_b = model_b(tokens, token_mask, action_feats, action_mask)
            out_e = ensemble(tokens, token_mask, action_feats, action_mask)

        for key in out_a:
            expected = torch.stack([out_a[key], out_b[key]], dim=0).mean(dim=0)
            torch.testing.assert_close(out_e[key], expected, rtol=1e-6, atol=1e-6)

    def test_singleton_ensemble_is_identity(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _EnsembleModel

        cfg = config_for_size("tiny")
        torch.manual_seed(3)
        model = build_cascadiaformer(cfg).eval()
        batch, seq, actions = 1, 4, 3
        tokens = torch.randn(batch, seq, cfg.token_feature_dim)
        token_mask = torch.ones(batch, seq, dtype=torch.bool)
        action_feats = torch.randn(batch, actions, cfg.action_feature_dim)
        action_mask = torch.ones(batch, actions, dtype=torch.bool)
        ensemble = _EnsembleModel([model])
        with torch.inference_mode():
            direct = model(tokens, token_mask, action_feats, action_mask)
            wrapped = ensemble(tokens, token_mask, action_feats, action_mask)
        for key in direct:
            torch.testing.assert_close(wrapped[key], direct[key])


if __name__ == "__main__":
    unittest.main()
