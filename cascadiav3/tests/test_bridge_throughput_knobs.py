"""Contract tests for the R2.4 bridge serving knobs.

Pins: every throughput knob defaults OFF (eager fp32 path), torch.compile
failures always fall back to eager (loudly, never fatally), the hello payload
reports the effective knob configuration for run attribution, and the chunk-row
override only changes behavior when explicitly set.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


def _require_torch(test: unittest.TestCase):
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - torch present in v3 env
        test.skipTest("torch unavailable")
    return torch


def _require_numpy(test: unittest.TestCase):
    try:
        import numpy
    except ModuleNotFoundError:  # pragma: no cover - numpy present in v3 env
        test.skipTest("numpy unavailable")
    return numpy


class BridgeEnvProvenanceTest(unittest.TestCase):
    def test_all_knobs_default_off(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        with mock.patch.dict(os.environ, {}, clear=True):
            provenance = bridge.bridge_env_provenance()
        self.assertFalse(provenance["compile"])
        self.assertEqual(provenance["compile_mode"], bridge.DEFAULT_COMPILE_MODE)
        self.assertEqual(bridge.DEFAULT_COMPILE_MODE, "reduce-overhead")
        self.assertFalse(provenance["tf32"])
        self.assertEqual(provenance["autocast"], "")
        self.assertFalse(provenance["bucket"])
        self.assertFalse(provenance["cgab_fused"])
        self.assertEqual(provenance["eval_cell_budget"], bridge.EVAL_BATCH_CELL_BUDGET)
        self.assertEqual(provenance["eval_chunk_rows"], bridge.EVAL_BATCH_CHUNK_SIZE)
        self.assertTrue(provenance["pinned_h2d"])
        self.assertFalse(provenance["timing"])
        self.assertEqual(provenance["ensemble_size"], 1)

    def test_set_knobs_are_reported(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        env = {
            "CASCADIA_BRIDGE_COMPILE": "1",
            "CASCADIA_BRIDGE_COMPILE_MODE": "max-autotune",
            "CASCADIA_BRIDGE_BUCKET": "1",
            "CASCADIA_BRIDGE_TF32": "1",
            "CASCADIA_BRIDGE_AUTOCAST": "bf16",
            "CASCADIA_CGAB_FUSED": "1",
            "CASCADIA_EVAL_CELL_BUDGET": "16777216",
            "CASCADIA_EVAL_CHUNK_ROWS": "128",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            provenance = bridge.bridge_env_provenance()
        self.assertTrue(provenance["compile"])
        self.assertEqual(provenance["compile_mode"], "max-autotune")
        self.assertTrue(provenance["bucket"])
        self.assertTrue(provenance["tf32"])
        self.assertEqual(provenance["autocast"], "bf16")
        self.assertTrue(provenance["cgab_fused"])
        self.assertEqual(provenance["eval_cell_budget"], 16777216)
        self.assertEqual(provenance["eval_chunk_rows"], 128)

    def test_hello_payload_reports_bridge_env(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        stdout = io.StringIO()
        stdin = io.StringIO('{"type": "shutdown"}\n')
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(sys, "stdin", stdin), contextlib.redirect_stdout(stdout):
                exit_code = bridge.serve(
                    checkpoint=None,
                    manifest=None,
                    allow_dry_run_fallback=True,
                )
        self.assertEqual(exit_code, 0)
        hello = json.loads(stdout.getvalue().splitlines()[0])
        self.assertEqual(hello["type"], "hello")
        self.assertIn("bridge_env", hello)
        self.assertFalse(hello["bridge_env"]["compile"])
        self.assertEqual(hello["bridge_env"]["compile_mode"], "reduce-overhead")
        self.assertFalse(hello["bridge_env"]["tf32"])
        self.assertFalse(hello["bridge_env"]["bucket"])


class CompileKnobTest(unittest.TestCase):
    def test_default_mode_is_reduce_overhead(self) -> None:
        torch = _require_torch(self)

        from cascadiav3 import torch_inference_bridge as bridge

        calls: list[dict] = []
        sentinel = SimpleNamespace(name="compiled")

        def fake_compile(model, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return sentinel

        model = SimpleNamespace(name="eager")
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(torch, "compile", fake_compile):
                compiled = bridge._maybe_compile_model(model, torch.device("cpu"))
        self.assertIs(compiled, sentinel)
        self.assertEqual(calls, [{"mode": "reduce-overhead"}])

    def test_mode_env_overrides_and_default_keyword(self) -> None:
        torch = _require_torch(self)

        from cascadiav3 import torch_inference_bridge as bridge

        calls: list[dict] = []

        def fake_compile(model, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return model

        model = SimpleNamespace()
        with mock.patch.object(torch, "compile", fake_compile):
            with mock.patch.dict(
                os.environ, {"CASCADIA_BRIDGE_COMPILE_MODE": "max-autotune"}, clear=True
            ):
                bridge._maybe_compile_model(model, torch.device("cpu"))
            with mock.patch.dict(
                os.environ, {"CASCADIA_BRIDGE_COMPILE_MODE": "default"}, clear=True
            ):
                bridge._maybe_compile_model(model, torch.device("cpu"))
        self.assertEqual(calls, [{"mode": "max-autotune"}, {}])

    def test_compile_failure_falls_back_to_eager_without_crashing(self) -> None:
        torch = _require_torch(self)

        from cascadiav3 import torch_inference_bridge as bridge

        model = SimpleNamespace(name="eager")
        stderr = io.StringIO()
        with mock.patch.object(torch, "compile", side_effect=RuntimeError("boom")):
            with contextlib.redirect_stderr(stderr):
                compiled = bridge._maybe_compile_model(model, torch.device("cpu"))
        self.assertIs(compiled, model)
        message = stderr.getvalue()
        self.assertIn("WARNING", message)
        self.assertIn("serving eager", message)
        self.assertIn("boom", message)

    def test_cuda_warmup_failure_falls_back_to_eager(self) -> None:
        torch = _require_torch(self)

        from cascadiav3 import torch_inference_bridge as bridge

        model = SimpleNamespace(name="eager")
        # The wrapper carries a config so the warmup actually runs; it is not
        # callable, so the warmup forward raises on any host (and on
        # CUDA-less hosts the device allocation raises first). Either way
        # the bridge must revert to the eager model, loudly.
        broken_compiled = SimpleNamespace(
            config=SimpleNamespace(token_feature_dim=4, action_feature_dim=6)
        )
        stderr = io.StringIO()
        with mock.patch.object(torch, "compile", return_value=broken_compiled):
            with contextlib.redirect_stderr(stderr):
                compiled = bridge._maybe_compile_model(model, torch.device("cuda"))
        self.assertIs(compiled, model)
        message = stderr.getvalue()
        self.assertIn("WARNING", message)
        self.assertIn("warmup failed", message)
        self.assertIn("serving eager", message)

    def test_load_model_compiles_only_when_flag_set(self) -> None:
        import tempfile
        from pathlib import Path

        torch = _require_torch(self)

        from cascadiav3 import torch_inference_bridge as bridge
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        cfg = config_for_size("tiny")
        model = build_cascadiaformer(cfg)
        with tempfile.TemporaryDirectory() as tempdir:
            weights_path = Path(tempdir) / "tiny.weights.pt"
            torch.save(model.state_dict(), weights_path)
            manifest_path = Path(tempdir) / "tiny.manifest.json"
            manifest_payload = {"config": cfg.to_dict(), "weights": str(weights_path)}
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

            with mock.patch.object(
                bridge, "_maybe_compile_model", side_effect=lambda model, device: model
            ) as compile_gate:
                with mock.patch.dict(os.environ, {}, clear=True):
                    bridge._load_model(
                        manifest_path,
                        manifest_path=manifest_path,
                        manifest_payload=manifest_payload,
                        device_name="cpu",
                    )
                compile_gate.assert_not_called()
                with mock.patch.dict(
                    os.environ, {"CASCADIA_BRIDGE_COMPILE": "1"}, clear=True
                ):
                    bridge._load_model(
                        manifest_path,
                        manifest_path=manifest_path,
                        manifest_payload=manifest_payload,
                        device_name="cpu",
                    )
                compile_gate.assert_called_once()


class EvalChunkRowsTest(unittest.TestCase):
    def test_default_and_override_and_invalid(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(bridge._eval_chunk_rows(), bridge.EVAL_BATCH_CHUNK_SIZE)
        with mock.patch.dict(os.environ, {"CASCADIA_EVAL_CHUNK_ROWS": "128"}, clear=True):
            self.assertEqual(bridge._eval_chunk_rows(), 128)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with mock.patch.dict(os.environ, {"CASCADIA_EVAL_CHUNK_ROWS": "junk"}, clear=True):
                self.assertEqual(bridge._eval_chunk_rows(), bridge.EVAL_BATCH_CHUNK_SIZE)
            with mock.patch.dict(os.environ, {"CASCADIA_EVAL_CHUNK_ROWS": "0"}, clear=True):
                self.assertEqual(bridge._eval_chunk_rows(), bridge.EVAL_BATCH_CHUNK_SIZE)
        self.assertIn("CASCADIA_EVAL_CHUNK_ROWS", stderr.getvalue())

    def test_model_eval_batch_resolves_chunk_rows_from_env(self) -> None:
        _require_torch(self)
        from cascadiav3 import torch_inference_bridge as bridge

        captured: dict[str, int] = {}

        def fake_chunks(roots, *, chunk_size):  # type: ignore[no-untyped-def]
            captured["chunk_size"] = chunk_size
            return []

        root = [{"state_hash": "x"}]
        with mock.patch.object(bridge, "_eval_batch_chunks", fake_chunks):
            with mock.patch.dict(os.environ, {"CASCADIA_EVAL_CHUNK_ROWS": "96"}, clear=True):
                self.assertEqual(bridge._model_eval_batch(object(), root), [])
            self.assertEqual(captured["chunk_size"], 96)
            with mock.patch.dict(os.environ, {}, clear=True):
                bridge._model_eval_batch(object(), root)
            self.assertEqual(captured["chunk_size"], bridge.EVAL_BATCH_CHUNK_SIZE)
            # An explicit argument always wins over the environment.
            with mock.patch.dict(os.environ, {"CASCADIA_EVAL_CHUNK_ROWS": "96"}, clear=True):
                bridge._model_eval_batch(object(), root, chunk_size=7)
            self.assertEqual(captured["chunk_size"], 7)


class ThroughputProbeHelpersTest(unittest.TestCase):
    def test_arm_env_matrix(self) -> None:
        from cascadiav3.torch_bridge_throughput_probe import ARMS, _arm_env

        expectations = {
            "eager": (None, None),
            "bucket": (None, "1"),
            "compile": ("1", None),
            "compile_bucket": ("1", "1"),
        }
        self.assertEqual(set(ARMS), set(expectations))
        for arm, (compile_value, bucket_value) in expectations.items():
            env = _arm_env(arm)
            self.assertEqual(env["CASCADIA_BRIDGE_COMPILE"], compile_value, arm)
            self.assertEqual(env["CASCADIA_BRIDGE_BUCKET"], bucket_value, arm)
        with self.assertRaises(ValueError):
            _arm_env("warp-speed")

    def test_patched_env_sets_and_restores(self) -> None:
        from cascadiav3.torch_bridge_throughput_probe import _patched_env

        with mock.patch.dict(
            os.environ, {"CASCADIA_BRIDGE_COMPILE": "1"}, clear=True
        ):
            with _patched_env({"CASCADIA_BRIDGE_COMPILE": None, "CASCADIA_BRIDGE_BUCKET": "1"}):
                self.assertNotIn("CASCADIA_BRIDGE_COMPILE", os.environ)
                self.assertEqual(os.environ.get("CASCADIA_BRIDGE_BUCKET"), "1")
            self.assertEqual(os.environ.get("CASCADIA_BRIDGE_COMPILE"), "1")
            self.assertNotIn("CASCADIA_BRIDGE_BUCKET", os.environ)

    def test_max_abs_diff_reports_per_key_and_rejects_mismatches(self) -> None:
        _require_numpy(self)
        from cascadiav3.torch_bridge_throughput_probe import _max_abs_diff

        base = [
            {
                "action_ids": ["a", "b"],
                "priors": [0.5, 0.5],
                "q": [1.0, 2.0],
                "score_to_go": [0.0, 0.0],
                "uncertainty": [1.0, 1.0],
                "value": [0.0, 0.0, 0.0, 0.0],
            }
        ]
        candidate = json.loads(json.dumps(base))
        candidate[0]["q"] = [1.0, 2.5]
        diffs = _max_abs_diff(base, candidate)
        self.assertEqual(diffs["q"], 0.5)
        self.assertEqual(diffs["priors"], 0.0)
        mismatched = json.loads(json.dumps(base))
        mismatched[0]["action_ids"] = ["a", "c"]
        with self.assertRaises(ValueError):
            _max_abs_diff(base, mismatched)


if __name__ == "__main__":
    unittest.main()
