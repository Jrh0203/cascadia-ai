"""Parity and smoke tests for the opt-in CascadiaFormer trainer perf knobs.

Hard contract pinned here: with no knobs set, training is bit-identical to the
legacy trainer (same batches in the same order, same parameter updates, same
eval numbers), and the multi-worker data path reproduces the exact batches the
in-process path builds.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

FIXTURE = Path("cascadiav3/fixtures/gumbel_tiny_tensor.npz")


def _require_torch_and_fixture(test: unittest.TestCase):  # type: ignore[no-untyped-def]
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        test.skipTest("torch unavailable")
    if not FIXTURE.exists():
        test.skipTest("gumbel tiny tensor fixture has not been generated")


def _run_tiny_training(tmp: Path, **overrides):  # type: ignore[no-untyped-def]
    import torch

    from cascadiav3.torch_train_cascadiaformer import run_training

    kwargs = dict(
        train_format="npz",
        val_format="npz",
        model_size="tiny",
        steps=3,
        batch_size=4,
        lr=1.0e-3,
        weight_decay=0.01,
        device_name="cpu",
        seed=1234,
        grad_accum=2,
        warmup_fraction=0.1,
        checkpoint_dir=tmp / "checkpoints",
        metrics_jsonl=tmp / "metrics.jsonl",
        out=tmp / "train.json",
        overfit_one_batch=False,
        val_max_batches=1,
        swa_fraction=0.5,
        objective="gumbel-selfplay",
        eval_every_steps=2,
    )
    train_paths = overrides.pop("train_paths", [FIXTURE])
    val_paths = overrides.pop("val_paths", [FIXTURE])
    kwargs.update(overrides)
    report = run_training(train_paths, val_paths, **kwargs)
    weights_path = tmp / "checkpoints" / f"step_{kwargs['steps']:07d}.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file

        state = load_file(weights_path)
    else:
        state = torch.load(
            tmp / "checkpoints" / f"step_{kwargs['steps']:07d}.weights.pt",
            map_location="cpu",
            weights_only=False,
        )
    metrics_text = (tmp / "metrics.jsonl").read_text(encoding="utf-8")
    return state, metrics_text, report


def _assert_state_dicts_equal(test: unittest.TestCase, left, right) -> None:  # type: ignore[no-untyped-def]
    import torch

    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertTrue(torch.equal(left[key], right[key]), f"tensor {key} differs")


class NestedTensorWarningTest(unittest.TestCase):
    def test_encoder_construction_emits_no_nested_tensor_warning(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_cascadiaformer(config_for_size("tiny"))
        messages = [str(item.message) for item in caught]
        self.assertFalse(
            any("enable_nested_tensor" in message for message in messages),
            messages,
        )


class DataWorkerParityTest(unittest.TestCase):
    """Multi-worker loading must reproduce the in-process batches exactly."""

    def test_loader_batches_match_in_process_collate(self) -> None:
        _require_torch_and_fixture(self)
        import torch

        from cascadiav3.expert_tensor_shards import ExpertTensorCorpus
        from cascadiav3.torch_train_cascadiaformer import (
            _GlobalBatchIndexSampler,
            _build_train_loader,
            _collate_examples,
            _corpus_examples,
        )

        corpus = ExpertTensorCorpus([FIXTURE, FIXTURE])
        source_lengths = corpus.source_lengths()
        for source_weights in (None, [0.75, 0.25]):
            sampler = _GlobalBatchIndexSampler(
                first_global_batch=1,
                last_global_batch=3,
                batch_size=4,
                record_count=len(corpus),
                seed=99,
                shuffle=True,
                source_lengths=source_lengths if source_weights else None,
                source_weights=source_weights,
            )
            expected = [
                _collate_examples(
                    _corpus_examples(corpus, indices, corpus_format="npz"),
                    corpus_format="npz",
                )
                for indices in sampler
            ]
            loader = _build_train_loader(
                train_paths=[FIXTURE, FIXTURE],
                train_format="npz",
                sampler=sampler,
                data_workers=2,
                prefetch_factor=2,
                pin_memory=False,
            )
            actual = list(loader)
            del loader
            self.assertEqual(len(actual), len(expected))
            for expected_batch, actual_batch in zip(expected, actual):
                self.assertEqual(set(expected_batch), set(actual_batch))
                for key, expected_value in expected_batch.items():
                    actual_value = actual_batch[key]
                    if torch.is_tensor(expected_value):
                        self.assertTrue(
                            torch.equal(expected_value, actual_value),
                            f"{key} differs (weights={source_weights})",
                        )
                    else:
                        self.assertEqual(expected_value, actual_value, key)

    def test_training_with_workers_is_bit_identical(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            state_default, metrics_default, _ = _run_tiny_training(
                Path(tmp_a),
                train_paths=[FIXTURE, FIXTURE],
                train_source_weights=[0.7, 0.3],
            )
            state_workers, metrics_workers, report = _run_tiny_training(
                Path(tmp_b),
                train_paths=[FIXTURE, FIXTURE],
                train_source_weights=[0.7, 0.3],
                data_workers=2,
            )
        _assert_state_dicts_equal(self, state_default, state_workers)
        self.assertEqual(metrics_default, metrics_workers)
        self.assertEqual(report["perf_knobs"]["data_workers"], 2)


class DefaultPathBitIdentityTest(unittest.TestCase):
    def test_default_run_is_deterministic(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            state_a, metrics_a, report = _run_tiny_training(Path(tmp_a))
            state_b, metrics_b, _ = _run_tiny_training(Path(tmp_b))
        _assert_state_dicts_equal(self, state_a, state_b)
        self.assertEqual(metrics_a, metrics_b)
        knobs = report["perf_knobs"]
        self.assertEqual(knobs["data_workers"], 0)
        self.assertFalse(knobs["tf32"])
        self.assertFalse(knobs["compile"])
        self.assertFalse(knobs["autocast_enabled"])  # CPU + auto = legacy fp32
        self.assertFalse(knobs["grad_checkpoint_applied"])

    def test_timing_mode_does_not_change_numerics(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            state_default, metrics_default, _ = _run_tiny_training(Path(tmp_a))
            with mock.patch.dict(os.environ, {"CASCADIA_TRAIN_TIMING": "1", "CASCADIA_TRAIN_TIMING_EVERY": "2"}):
                state_timed, metrics_timed, report = _run_tiny_training(Path(tmp_b))
        _assert_state_dicts_equal(self, state_default, state_timed)
        self.assertEqual(metrics_default, metrics_timed)
        self.assertIsNotNone(report["phase_timing"])
        self.assertIn("forward", report["phase_timing"]["totals_s"])
        self.assertIn("data", report["phase_timing"]["totals_s"])

    def test_autocast_off_matches_default_on_cpu(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            state_default, metrics_default, _ = _run_tiny_training(Path(tmp_a))
            state_off, metrics_off, _ = _run_tiny_training(Path(tmp_b), autocast_mode="off")
        _assert_state_dicts_equal(self, state_default, state_off)
        self.assertEqual(metrics_default, metrics_off)

    def test_epoch_order_cache_matches_legacy_shuffle(self) -> None:
        import random

        from cascadiav3.torch_train_cascadiaformer import _batch_indices_for_global_batch

        record_count, batch_size, seed = 37, 8, 11

        def legacy_order(epoch: int) -> list[int]:
            order = list(range(record_count))
            rng = random.Random(seed + epoch * 1_000_003)
            rng.shuffle(order)
            return order

        for global_batch in range(1, 16):
            indices, cursor = _batch_indices_for_global_batch(
                global_batch=global_batch,
                batch_size=batch_size,
                record_count=record_count,
                seed=seed,
                shuffle=True,
            )
            start = (global_batch - 1) * batch_size
            expected: list[int] = []
            epoch, position = start // record_count, start % record_count
            while len(expected) < batch_size:
                order = legacy_order(epoch)
                take = min(batch_size - len(expected), record_count - position)
                expected.extend(order[position : position + take])
                position += take
                if position == record_count and len(expected) < batch_size:
                    epoch += 1
                    position = 0
            self.assertEqual(indices, expected)
            self.assertEqual(cursor["record_count"], record_count)

    def test_loss_scalars_match_per_key_floats(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_train_cascadiaformer import _loss_scalars

        losses = {
            "a": torch.tensor(1.2345678, dtype=torch.float32),
            "b": torch.tensor(-7.0, dtype=torch.bfloat16),
            "c": torch.tensor(0.0, dtype=torch.float32, requires_grad=True) * 3.0,
        }
        values = _loss_scalars(losses, ("a", "b", "c"))
        for key in losses:
            self.assertEqual(values[key], float(losses[key].detach().cpu()))

    def test_loss_scalars_support_mps_without_device_float64(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS unavailable")
        from cascadiav3.torch_train_cascadiaformer import _loss_scalars

        losses = {
            "a": torch.tensor(1.25, dtype=torch.float32, device="mps"),
            "b": torch.tensor(-3.5, dtype=torch.float32, device="mps", requires_grad=True),
        }
        values = _loss_scalars(losses, ("a", "b"))
        self.assertEqual(values, {"a": 1.25, "b": -3.5})


class OptInKnobSmokeTest(unittest.TestCase):
    def test_grad_checkpoint_on_matches_off_forward_backward(self) -> None:
        _require_torch_and_fixture(self)
        import torch

        from cascadiav3.expert_tensor_shards import ExpertTensorCorpus
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _collate_examples,
            _corpus_examples,
            _loss_components,
            _model_forward,
        )

        corpus = ExpertTensorCorpus([FIXTURE])
        batch = _collate_examples(
            _corpus_examples(corpus, list(range(min(4, len(corpus)))), corpus_format="npz"),
            corpus_format="npz",
        )
        torch.manual_seed(5)
        model = build_cascadiaformer(config_for_size("tiny"))
        results = {}
        for mode in ("off", "on"):
            model.zero_grad(set_to_none=True)
            model.set_gradient_checkpointing(mode == "on")
            model.train()
            losses = _loss_components(_model_forward(model, batch), batch, LossWeights())
            losses["total"].backward()
            results[mode] = (
                losses["total"].detach().clone(),
                {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None},
            )
        loss_off, grads_off = results["off"]
        loss_on, grads_on = results["on"]
        self.assertTrue(torch.equal(loss_off, loss_on))
        self.assertEqual(set(grads_off), set(grads_on))
        for name in grads_off:
            self.assertTrue(torch.equal(grads_off[name], grads_on[name]), f"grad {name} differs")

    def test_grad_checkpoint_knob_via_run_training(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            state_default, metrics_default, _ = _run_tiny_training(Path(tmp_a))
            state_ckpt, metrics_ckpt, report = _run_tiny_training(Path(tmp_b), grad_checkpoint="on")
        self.assertTrue(report["perf_knobs"]["grad_checkpoint_applied"])
        # CPU recompute is deterministic, so checkpointing is bit-identical here.
        _assert_state_dicts_equal(self, state_default, state_ckpt)
        self.assertEqual(metrics_default, metrics_ckpt)

    def test_fused_optimizer_ignored_on_cpu(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, report = _run_tiny_training(Path(tmp), steps=1, fused_optimizer=True)
        self.assertFalse(report["perf_knobs"]["fused_optimizer"])

    def test_sdpa_env_math_runs_and_invalid_value_raises(self) -> None:
        _require_torch_and_fixture(self)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"CASCADIA_TRAIN_SDPA": "math"}):
                _, _, report = _run_tiny_training(Path(tmp), steps=1)
        self.assertEqual(report["perf_knobs"]["sdpa"], "math")
        from cascadiav3.torch_train_cascadiaformer import _sdpa_context_factory

        with self.assertRaises(ValueError):
            _sdpa_context_factory("not_a_backend")

    def test_autocast_bf16_smoke_on_cpu(self) -> None:
        _require_torch_and_fixture(self)
        import math

        with tempfile.TemporaryDirectory() as tmp:
            _, _, report = _run_tiny_training(Path(tmp), steps=1, autocast_mode="bf16")
        self.assertTrue(report["perf_knobs"]["autocast_enabled"])
        self.assertTrue(math.isfinite(report["latest_metrics"]["train_total"]))

    def test_compile_smoke_on_cpu(self) -> None:
        _require_torch_and_fixture(self)
        import math

        with tempfile.TemporaryDirectory() as tmp:
            _, _, report = _run_tiny_training(Path(tmp), steps=2, compile_model=True)
        self.assertTrue(report["perf_knobs"]["compile"])
        self.assertTrue(math.isfinite(report["latest_metrics"]["train_total"]))
        self.assertTrue(math.isfinite(report["latest_metrics"]["locked_val_total"]))


if __name__ == "__main__":
    unittest.main()
