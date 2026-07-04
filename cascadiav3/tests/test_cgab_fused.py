"""Equivalence and contract tests for the fused CGAB relation tail.

The CGAB relation tail's contribution is a masked mean over embedding-table
rows selected by relation ids (mask = id != 0, matching padding_idx=0). That
mean equals ``(counts_per_relation_id / valid_positions) @ embedding_table``,
so the fused path (CASCADIA_CGAB_FUSED=1 / --cgab-fused /
model.set_cgab_fused) never materializes the [B, A, seq, d_model] tensor.
Fused output is mathematically equivalent but NOT bit-identical (floating
point reassociation); these tests pin the agreement tolerance and the
padding contract, and pin that the default stays the untouched materialized
path.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock


def _require_torch(test: unittest.TestCase):
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - torch present in v3 env
        test.skipTest("torch unavailable")
    return torch


def _build_model(size: str = "tiny"):
    from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

    return build_cascadiaformer(config_for_size(size))


def _synthetic_batch(torch, model, *, batch, max_tokens, max_actions, seed, tail_only):
    """Random inputs with realistic padding: per-row token/action counts vary,
    padded relation columns/rows stay id 0, and one row's actions are fully
    padded. Relation ids are drawn from the full vocab including id 0."""
    cfg = model.config
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, max_tokens, cfg.token_feature_dim, generator=generator)
    actions = torch.randn(batch, max_actions, cfg.action_feature_dim, generator=generator)
    token_mask = torch.zeros(batch, max_tokens, dtype=torch.bool)
    action_mask = torch.zeros(batch, max_actions, dtype=torch.bool)
    seq_len = max_tokens + max_actions
    token_counts = [max(1, ((seed + row * 3) % max_tokens) or max_tokens) for row in range(batch)]
    action_counts = [max(1, ((seed + row * 5) % max_actions) or max_actions) for row in range(batch)]
    if batch > 1:
        action_counts[-1] = 0  # fully padded action row
    if tail_only:
        relation = torch.zeros(batch, max_actions, seq_len, dtype=torch.uint8)
    else:
        relation = torch.zeros(batch, seq_len, seq_len, dtype=torch.long)
    for row in range(batch):
        token_mask[row, : token_counts[row]] = True
        action_mask[row, : action_counts[row]] = True
        ids = torch.randint(
            0,
            cfg.relation_vocab_size,
            (action_counts[row], token_counts[row]),
            generator=generator,
        )
        if tail_only:
            relation[row, : action_counts[row], : token_counts[row]] = ids.to(torch.uint8)
        else:
            relation[row, max_tokens : max_tokens + action_counts[row], : token_counts[row]] = ids
            # action-to-action relations also occur in the combined matrix
            relation[row, max_tokens, max_tokens : max_tokens + action_counts[row]] = 1
    inputs = {
        "tokens": tokens,
        "token_mask": token_mask,
        "actions": actions,
        "action_mask": action_mask,
    }
    if tail_only:
        inputs["relation_tail"] = relation
    else:
        inputs["relation_ids"] = relation
    return inputs


def _forward(torch, model, inputs, *, fused, inference):
    model.set_cgab_fused(fused)
    try:
        if inference:
            with torch.inference_mode():
                return {key: value.clone() for key, value in model(**inputs).items()}
        return model(**inputs)
    finally:
        model.set_cgab_fused(False)


class CgabFusedEquivalenceTest(unittest.TestCase):
    OUTPUT_KEYS = (
        "logits",
        "q",
        "uncertainty",
        "value_vector",
        "rank_logits",
        "differential",
        "score_decomposition",
        "opponent_aux",
        "market_aux",
        "cgab_bias",
    )

    def _assert_outputs_close(self, torch, reference, candidate, *, rtol=1e-5, atol=1e-6):
        worst = 0.0
        for key in self.OUTPUT_KEYS:
            torch.testing.assert_close(
                candidate[key], reference[key], rtol=rtol, atol=atol, msg=lambda m, key=key: f"{key}: {m}"
            )
            diff = (candidate[key] - reference[key]).abs().max().item()
            worst = max(worst, diff)
        return worst

    def test_full_model_equivalence_relation_ids_path(self) -> None:
        torch = _require_torch(self)
        model = _build_model("tiny").eval()
        inputs = _synthetic_batch(
            torch, model, batch=5, max_tokens=9, max_actions=7, seed=20260704, tail_only=False
        )
        reference = _forward(torch, model, inputs, fused=False, inference=True)
        fused = _forward(torch, model, inputs, fused=True, inference=True)
        worst = self._assert_outputs_close(torch, reference, fused)
        self.assertLess(worst, 1e-5)

    def test_full_model_equivalence_relation_tail_path(self) -> None:
        torch = _require_torch(self)
        model = _build_model("tiny").eval()
        inputs = _synthetic_batch(
            torch, model, batch=4, max_tokens=11, max_actions=6, seed=20260705, tail_only=True
        )
        reference = _forward(torch, model, inputs, fused=False, inference=True)
        fused = _forward(torch, model, inputs, fused=True, inference=True)
        worst = self._assert_outputs_close(torch, reference, fused)
        self.assertLess(worst, 1e-5)

    def test_cgab_module_equivalence_at_production_dims(self) -> None:
        """Standalone CGAB comparison at S-model width and a full-menu tail
        shape, including negative ids (exercising clamp_min) and all-zero
        rows."""
        torch = _require_torch(self)
        model = _build_model("S")
        cgab = model.cgab
        generator = torch.Generator().manual_seed(20260706)
        batch, action_count, seq_len = 3, 64, 320
        action_h = torch.randn(batch, action_count, model.config.d_model, generator=generator)
        tail = torch.randint(
            -2, model.config.relation_vocab_size, (batch, action_count, seq_len), generator=generator
        )
        tail[0, 0] = 0  # action with no relations at all
        with torch.inference_mode():
            cgab.fused = False
            ref_h, ref_bias = cgab(action_h, relation_tail=tail)
            cgab.fused = True
            fused_h, fused_bias = cgab(action_h, relation_tail=tail)
            cgab.fused = False
        torch.testing.assert_close(fused_bias, ref_bias, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fused_h, ref_h, rtol=1e-5, atol=1e-6)
        # The no-relation action must contribute exactly zero bias on both paths.
        self.assertEqual(ref_bias[0, 0].abs().max().item(), 0.0)
        self.assertEqual(fused_bias[0, 0].abs().max().item(), 0.0)

    def test_gradients_match(self) -> None:
        """Fused backward must agree with the materialized backward on every
        parameter, and the padding row of the relation table must receive zero
        gradient on both paths (padding_idx contract)."""
        torch = _require_torch(self)
        model = _build_model("tiny").train()
        inputs = _synthetic_batch(
            torch, model, batch=4, max_tokens=8, max_actions=5, seed=20260707, tail_only=False
        )

        def grads(fused: bool):
            model.zero_grad(set_to_none=True)
            outputs = _forward(torch, model, inputs, fused=fused, inference=False)
            loss = sum(outputs[key].square().mean() for key in ("logits", "q", "cgab_bias"))
            loss.backward()
            return {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

        reference = grads(False)
        fused = grads(True)
        model.zero_grad(set_to_none=True)
        self.assertEqual(set(reference), set(fused))
        for name in reference:
            torch.testing.assert_close(
                fused[name], reference[name], rtol=1e-5, atol=1e-6, msg=lambda m, name=name: f"{name}: {m}"
            )
        embed_grad = reference["cgab.relation_embed.weight"]
        self.assertEqual(embed_grad[0].abs().max().item(), 0.0)
        self.assertEqual(fused["cgab.relation_embed.weight"][0].abs().max().item(), 0.0)

    def test_fully_padded_and_zero_relation_actions(self) -> None:
        """All-zero relation tails (padding id 0 everywhere) must yield exactly
        zero cgab_bias on both paths, and fully action-masked rows must produce
        identical (zeroed) decoded outputs."""
        torch = _require_torch(self)
        model = _build_model("tiny").eval()
        inputs = _synthetic_batch(
            torch, model, batch=3, max_tokens=6, max_actions=4, seed=20260708, tail_only=True
        )
        inputs["relation_tail"] = torch.zeros_like(inputs["relation_tail"])
        inputs["action_mask"][1] = False  # fully padded action row
        reference = _forward(torch, model, inputs, fused=False, inference=True)
        fused = _forward(torch, model, inputs, fused=True, inference=True)
        self.assertEqual(reference["cgab_bias"].abs().max().item(), 0.0)
        self.assertEqual(fused["cgab_bias"].abs().max().item(), 0.0)
        for key in self.OUTPUT_KEYS:
            torch.testing.assert_close(fused[key], reference[key], rtol=0.0, atol=0.0)


class CgabFusedSwitchTest(unittest.TestCase):
    def test_default_is_materialized(self) -> None:
        _require_torch(self)
        with mock.patch.dict(os.environ):
            os.environ.pop("CASCADIA_CGAB_FUSED", None)
            model = _build_model("tiny")
        self.assertFalse(model.cgab.fused)

    def test_env_flag_enables_fused_at_construction(self) -> None:
        _require_torch(self)
        with mock.patch.dict(os.environ, {"CASCADIA_CGAB_FUSED": "1"}):
            model = _build_model("tiny")
        self.assertTrue(model.cgab.fused)

    def test_setter_toggles(self) -> None:
        _require_torch(self)
        model = _build_model("tiny")
        model.set_cgab_fused(True)
        self.assertTrue(model.cgab.fused)
        model.set_cgab_fused(False)
        self.assertFalse(model.cgab.fused)

    def test_trainer_exposes_cgab_fused_flag(self) -> None:
        import inspect

        from cascadiav3.torch_train_cascadiaformer import run_training

        self.assertIn("cgab_fused", inspect.signature(run_training).parameters)


class EvalCellBudgetOverrideTest(unittest.TestCase):
    @staticmethod
    def _roots(count: int) -> list[dict]:
        return [
            {"packed_features": {"action_count": 256, "token_count": 64}}
            for _ in range(count)
        ]

    def test_default_budget_unchanged(self) -> None:
        from cascadiav3.torch_inference_bridge import (
            EVAL_BATCH_CELL_BUDGET,
            _eval_cell_budget,
        )

        self.assertEqual(EVAL_BATCH_CELL_BUDGET, 2_097_152)
        with mock.patch.dict(os.environ):
            os.environ.pop("CASCADIA_EVAL_CELL_BUDGET", None)
            self.assertEqual(_eval_cell_budget(), EVAL_BATCH_CELL_BUDGET)

    def test_env_override_raises_chunk_capacity(self) -> None:
        from cascadiav3.torch_inference_bridge import _eval_batch_chunks

        roots = self._roots(64)
        with mock.patch.dict(os.environ):
            os.environ.pop("CASCADIA_BRIDGE_BUCKET", None)
            os.environ.pop("CASCADIA_EVAL_CELL_BUDGET", None)
            default_chunks = _eval_batch_chunks(roots, chunk_size=32)
            os.environ["CASCADIA_EVAL_CELL_BUDGET"] = str(64 * 256 * 320)
            raised_chunks = _eval_batch_chunks(roots, chunk_size=32)
        # Full-menu rows at 256 actions x 320 seq: the default 2^21 budget
        # caps chunks at 25 rows; the raised budget lets chunk_size rule.
        self.assertEqual(max(len(chunk) for chunk in default_chunks), 25)
        self.assertEqual([len(chunk) for chunk in raised_chunks], [32, 32])

    def test_invalid_override_falls_back(self) -> None:
        from cascadiav3.torch_inference_bridge import (
            EVAL_BATCH_CELL_BUDGET,
            _eval_cell_budget,
        )

        for bad in ("garbage", "0", "-5"):
            with mock.patch.dict(os.environ, {"CASCADIA_EVAL_CELL_BUDGET": bad}):
                self.assertEqual(_eval_cell_budget(), EVAL_BATCH_CELL_BUDGET)


if __name__ == "__main__":
    unittest.main()
