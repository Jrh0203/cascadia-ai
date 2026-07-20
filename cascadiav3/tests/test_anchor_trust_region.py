"""Trust-region anchor regularizers for warm-start CascadiaFormer fine-tunes.

Motivation: warm-start fine-tuning a strong incumbent was observed to improve
value-prediction loss while degrading actual play -- the value head drifted in a
way that hurt search-time blending. Two optional, default-off regularizers pin a
fine-tune toward a FROZEN anchor model:

  * ``--anchor-policy-kl-weight``  -> forward KL(anchor_policy || current_policy)
    over each root's candidate-action distribution (anchor as target).
  * ``--anchor-value-l2-weight``   -> L2 between the current model's value/score
    head outputs and the frozen anchor's outputs on the same batch.

The cardinal constraint under test: with both weights 0.0 (or no anchor at all)
``_loss_components`` is BYTE-IDENTICAL to the no-anchor trainer -- same total,
same component values, and NO extra metric keys. When active, the two terms are
computed on exactly the same masked candidate set / valid-row structure the
existing policy and value losses use, and are recorded as ``anchor_policy_kl`` /
``anchor_value_l2`` (prefixed ``train_`` / ``locked_val_`` in the metrics jsonl).

Torch-dependent tests skip cleanly when torch is unavailable (the repo's standard
``_require_torch`` pattern). The argparse and helper tests run without torch.

Run on the GPU box (torch present):

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \\
        python3 -m unittest cascadiav3.tests.test_anchor_trust_region -v
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

# Reuse the deterministic, arithmetic-only outputs/batch fixture that the Stage 1
# bit-identity tests already vet, so the anchor tests inherit the same stable
# floats and the same masked/valid-action structure the trainer sees.
from test_train_stage1_flags import build_loss_fixture


def build_anchor_outputs(outputs, *, scale: float, shift: float):  # type: ignore[no-untyped-def]
    """A frozen-anchor forward with the SAME shapes as ``outputs``.

    Only the three tensors the anchor regularizers read are populated
    (``logits``, ``value_vector``, ``score_decomposition``); values are a
    deterministic affine remap of the live outputs so the anchor genuinely
    differs from the live model without introducing RNG.
    """
    import torch

    def remap(tensor):  # type: ignore[no-untyped-def]
        flat = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
        return flat * scale + shift

    return {
        "logits": remap(outputs["logits"]),
        "value_vector": remap(outputs["value_vector"]),
        "score_decomposition": remap(outputs["score_decomposition"]),
    }


class _TorchCase(unittest.TestCase):
    def _require_torch(self):  # type: ignore[no-untyped-def]
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")

    def _components(self, **kwargs):  # type: ignore[no-untyped-def]
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        weights = loss_weights_for_objective("gumbel-selfplay")
        return outputs, batch, _loss_components(outputs, batch, weights, **kwargs)


class DefaultOffBitIdentityTest(_TorchCase):
    """Flag-off behavior must match the no-anchor trainer exactly."""

    def test_no_anchor_args_is_the_baseline(self) -> None:
        self._require_torch()
        _, _, baseline = self._components()
        for key in ("anchor_policy_kl", "anchor_value_l2"):
            self.assertNotIn(key, baseline)

    def test_zero_weights_with_anchor_outputs_are_byte_identical(self) -> None:
        self._require_torch()
        outputs, _batch, baseline = self._components()
        anchor_outputs = build_anchor_outputs(outputs, scale=0.07, shift=1.5)
        _, _, gated = self._components(
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=0.0,
            anchor_value_l2_weight=0.0,
        )
        # Same keys, exact same values -- no extra graph node, no anchor keys.
        self.assertEqual(set(gated), set(baseline))
        for key, value in baseline.items():
            self.assertEqual(float(gated[key]), float(value), msg=f"{key!r} drifted")
        self.assertNotIn("anchor_policy_kl", gated)
        self.assertNotIn("anchor_value_l2", gated)

    def test_positive_weight_but_no_anchor_outputs_stays_off(self) -> None:
        self._require_torch()
        # Guard also requires anchor_outputs is not None; weights alone can't
        # switch the term on (the trainer never passes weights without outputs).
        _, _, gated = self._components(
            anchor_outputs=None,
            anchor_policy_kl_weight=1.0,
            anchor_value_l2_weight=1.0,
        )
        self.assertNotIn("anchor_policy_kl", gated)
        self.assertNotIn("anchor_value_l2", gated)


class AnchorTermsTest(_TorchCase):
    """Numerics: forward KL and value/score L2 on the candidate/valid structure."""

    def test_keys_present_and_total_reflects_weighted_terms(self) -> None:
        self._require_torch()
        outputs, _batch, baseline = self._components()
        anchor_outputs = build_anchor_outputs(outputs, scale=0.09, shift=-2.0)
        kl_w, l2_w = 0.3, 0.7
        _, _, active = self._components(
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=kl_w,
            anchor_value_l2_weight=l2_w,
        )
        self.assertIn("anchor_policy_kl", active)
        self.assertIn("anchor_value_l2", active)
        expected_total = (
            float(baseline["total"])
            + kl_w * float(active["anchor_policy_kl"])
            + l2_w * float(active["anchor_value_l2"])
        )
        self.assertAlmostEqual(float(active["total"]), expected_total, places=4)

    def test_forward_kl_matches_independent_recomputation(self) -> None:
        self._require_torch()
        import torch

        outputs, batch = build_loss_fixture(improved_policy=True)
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        anchor_outputs = build_anchor_outputs(outputs, scale=0.11, shift=0.3)
        active = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay"),
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=1.0,
            anchor_value_l2_weight=0.0,
        )
        # Independent forward KL(anchor || current) over the SAME candidate mask.
        mask = batch["action_mask"]
        live_logp = torch.log_softmax(outputs["logits"].masked_fill(~mask, -1.0e9), dim=1)
        anchor_logp = torch.log_softmax(
            anchor_outputs["logits"].masked_fill(~mask, -1.0e9), dim=1
        )
        anchor_p = anchor_logp.exp().masked_fill(~mask, 0.0)
        expected_kl = (anchor_p * (anchor_logp - live_logp)).sum(dim=1).mean()
        self.assertTrue(torch.isfinite(expected_kl))
        self.assertAlmostEqual(float(active["anchor_policy_kl"]), float(expected_kl), places=5)
        # Forward KL is nonnegative.
        self.assertGreaterEqual(float(active["anchor_policy_kl"]), -1e-6)

    def test_kl_is_finite_even_with_huge_anchor_logit_on_masked_action(self) -> None:
        self._require_torch()
        import torch

        outputs, batch = build_loss_fixture(improved_policy=True)
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        anchor_outputs = build_anchor_outputs(outputs, scale=0.05, shift=0.0)
        # Records 1 and 2 have masked (invalid) actions; put a huge anchor logit
        # exactly there. The masked_fill(-1e9) + softmax + masked_fill(0) path
        # must drive those to anchor prob 0 so the KL summand is 0*finite = 0.
        mask = batch["action_mask"]
        blown = anchor_outputs["logits"].clone()
        blown = blown.masked_fill(~mask, 1.0e6)
        anchor_outputs["logits"] = blown
        losses = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay"),
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=1.0,
            anchor_value_l2_weight=0.0,
        )
        self.assertTrue(torch.isfinite(torch.as_tensor(float(losses["anchor_policy_kl"]))))

    def test_value_l2_matches_independent_recomputation(self) -> None:
        self._require_torch()
        import torch

        outputs, batch = build_loss_fixture(improved_policy=True)
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        anchor_outputs = build_anchor_outputs(outputs, scale=0.13, shift=4.0)
        losses = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay"),
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=0.0,
            anchor_value_l2_weight=1.0,
        )
        # No outcome_valid in the fixture -> plain means, value + score heads.
        value_res = outputs["value_vector"] - anchor_outputs["value_vector"]
        value_l2 = (value_res * value_res).mean(dim=1).mean()
        score_res = outputs["score_decomposition"] - anchor_outputs["score_decomposition"]
        score_l2 = (score_res * score_res).reshape(score_res.shape[0], -1).mean(dim=1).mean()
        expected = value_l2 + score_l2
        self.assertAlmostEqual(float(losses["anchor_value_l2"]), float(expected), places=4)

    def test_value_l2_respects_outcome_valid_rows(self) -> None:
        self._require_torch()
        import torch

        outputs, batch = build_loss_fixture(improved_policy=True)
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        anchor_outputs = build_anchor_outputs(outputs, scale=0.17, shift=-1.0)
        # Mask out the middle record; only rows 0 and 2 should contribute.
        outcome_valid = torch.tensor([True, False, True])
        batch["outcome_valid"] = outcome_valid
        losses = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay"),
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=0.0,
            anchor_value_l2_weight=1.0,
        )

        def masked_mean(per_record):  # type: ignore[no-untyped-def]
            w = outcome_valid.to(per_record.dtype)
            return (per_record * w).sum() / w.sum().clamp_min(1.0)

        value_res = outputs["value_vector"] - anchor_outputs["value_vector"]
        value_l2 = masked_mean((value_res * value_res).mean(dim=1))
        score_res = outputs["score_decomposition"] - anchor_outputs["score_decomposition"]
        score_l2 = masked_mean(
            (score_res * score_res).reshape(score_res.shape[0], -1).mean(dim=1)
        )
        expected = value_l2 + score_l2
        self.assertAlmostEqual(float(losses["anchor_value_l2"]), float(expected), places=4)

    def test_identical_anchor_gives_zero_regularizers(self) -> None:
        self._require_torch()
        import torch

        outputs, batch = build_loss_fixture(improved_policy=True)
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        # Anchor == current model on this batch: KL and value/score L2 are 0.
        anchor_outputs = {
            "logits": outputs["logits"].clone(),
            "value_vector": outputs["value_vector"].clone(),
            "score_decomposition": outputs["score_decomposition"].clone(),
        }
        losses = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay"),
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=1.0,
            anchor_value_l2_weight=1.0,
        )
        self.assertAlmostEqual(float(losses["anchor_policy_kl"]), 0.0, places=6)
        self.assertEqual(float(losses["anchor_value_l2"]), 0.0)

    def test_anchor_terms_pull_gradient_only_into_live_heads(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        # Make the live heads leaves so we can inspect their gradients.
        logits_leaf = outputs["logits"].clone().requires_grad_(True)
        value_leaf = outputs["value_vector"].clone().requires_grad_(True)
        outputs["logits"] = logits_leaf
        outputs["value_vector"] = value_leaf
        # Anchor tensors are constants -- a detached leaf must never get grad.
        anchor_logits = torch.zeros_like(logits_leaf, requires_grad=True)
        anchor_value = torch.zeros_like(value_leaf, requires_grad=True)
        anchor_outputs = {
            "logits": anchor_logits,
            "value_vector": anchor_value,
            "score_decomposition": outputs["score_decomposition"].detach(),
        }
        weights = LossWeights(
            policy=0.0, q=0.0, value=0.0, score=0.0, rank=0.0, uncertainty=0.0
        )
        losses = _loss_components(
            outputs,
            batch,
            weights,
            anchor_outputs=anchor_outputs,
            anchor_policy_kl_weight=1.0,
            anchor_value_l2_weight=1.0,
        )
        losses["total"].backward()
        # The frozen anchor side receives no gradient in the training graph...
        self.assertIsNone(anchor_logits.grad)
        self.assertIsNone(anchor_value.grad)
        # ...while the live heads are pulled toward the anchor.
        self.assertIsNotNone(logits_leaf.grad)
        self.assertIsNotNone(value_leaf.grad)


class AggregateKeyHelperTest(unittest.TestCase):
    """The itemization key set gains anchor keys only when active (no torch)."""

    def test_keys_extend_only_when_anchor_enabled(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import (
            AGGREGATE_KEYS,
            ANCHOR_METRIC_KEYS,
            _aggregate_keys_with_anchor,
            _anchor_active,
        )

        self.assertEqual(_aggregate_keys_with_anchor(False), AGGREGATE_KEYS)
        self.assertEqual(
            _aggregate_keys_with_anchor(True), AGGREGATE_KEYS + ANCHOR_METRIC_KEYS
        )
        self.assertEqual(
            ANCHOR_METRIC_KEYS, ("anchor_policy_kl", "anchor_value_l2")
        )
        self.assertFalse(_anchor_active(0.0, 0.0))
        self.assertTrue(_anchor_active(0.1, 0.0))
        self.assertTrue(_anchor_active(0.0, 0.1))


class AnchorArgparseTest(unittest.TestCase):
    """CLI flags default off and thread into run_training (no torch)."""

    def _captured_kwargs(self, argv: list[str]):  # type: ignore[no-untyped-def]
        import cascadiav3.torch_train_cascadiaformer as trainer

        captured: dict[str, object] = {}

        def fake_run_training(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return {"status": "pass"}

        with mock.patch.object(trainer, "run_training", fake_run_training):
            with mock.patch(
                "sys.argv",
                ["torch_train_cascadiaformer.py", "--train", "t.jsonl", "--val", "v.jsonl"]
                + argv,
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(trainer.main(), 0)
        return captured

    def test_defaults_are_off(self) -> None:
        captured = self._captured_kwargs([])
        self.assertIsNone(captured["anchor_manifest"])
        self.assertEqual(captured["anchor_policy_kl_weight"], 0.0)
        self.assertEqual(captured["anchor_value_l2_weight"], 0.0)

    def test_flags_thread_through(self) -> None:
        from pathlib import Path

        captured = self._captured_kwargs(
            [
                "--anchor-manifest",
                "checkpoints/incumbent/best.manifest.json",
                "--anchor-policy-kl-weight",
                "0.25",
                "--anchor-value-l2-weight",
                "0.5",
            ]
        )
        self.assertEqual(
            captured["anchor_manifest"],
            Path("checkpoints/incumbent/best.manifest.json"),
        )
        self.assertEqual(captured["anchor_policy_kl_weight"], 0.25)
        self.assertEqual(captured["anchor_value_l2_weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
