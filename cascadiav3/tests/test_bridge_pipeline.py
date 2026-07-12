"""R2.4 lever #1 (request pipelining) contract tests for the Python bridge.

Pins for CASCADIA_BRIDGE_PIPELINE=1:
- default OFF (provenance + serial serve loop untouched when unset),
- strict FIFO response order across bursts written without interleaved reads
  (the CASCADIA_SHARED_INFLIGHT=2+ Rust client pattern),
- byte-identical responses versus the serial loop for identical input bursts,
  including interleaved hello, malformed JSON, protocol errors, single
  eval_request, and shutdown/EOF mid-burst drain,
- the phase-split prepare/launch/finalize path produces EXACTLY (==) the
  outputs of the untouched _model_eval_batch path (torch-only tests),
- the stdin reader thread always delivers its EOF sentinel.

Every subprocess read goes through Popen.communicate(timeout=...) so a
pipelining deadlock fails the test instead of hanging the suite.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
BURST_TIMEOUT_S = 60.0
MODEL_BURST_TIMEOUT_S = 300.0


def _require_torch(test: unittest.TestCase):  # type: ignore[no-untyped-def]
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - torch present in v3 env
        test.skipTest("torch unavailable")
    return torch


def _bridge_env(pipeline: bool) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CASCADIA_")
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + existing if existing else "")
    if pipeline:
        env["CASCADIA_BRIDGE_PIPELINE"] = "1"
    return env


def _run_bridge_burst(
    lines: list[str],
    *,
    pipeline: bool,
    args: tuple[str, ...] = (),
    timeout: float = BURST_TIMEOUT_S,
) -> tuple[list[dict], int, str]:
    """Writes the whole burst without reading between writes (the inflight>=2
    client pattern), then collects every response line. Returns (parsed
    response payloads, returncode, stderr text)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "cascadiav3.torch_inference_bridge", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_bridge_env(pipeline),
        cwd=str(SRC_ROOT),
        text=True,
    )
    try:
        stdout, stderr = proc.communicate("".join(lines), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AssertionError(
            f"bridge (pipeline={pipeline}) deadlocked: no full response within {timeout}s"
        )
    payloads = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return payloads, proc.returncode, stderr


def _fallback_root(tag: str, action_count: int) -> dict:
    """Model-less uniform-fallback root: q echoes exact_afterstate exactly and
    action_ids are unique per request, so responses are position-taggable."""
    return {
        "schema_id": "pipeline-test.v1",
        "state_hash": f"hash-{tag}",
        "action_ids": [f"{tag}-a{index}" for index in range(action_count)],
        "exact_afterstate_score_active": [
            float(100 * (index + 1)) + float(sum(map(ord, tag))) / 128.0
            for index in range(action_count)
        ],
    }


def _batch_request(roots: list[dict], *, allow_model_fallback: bool = True, **extra) -> str:
    message = {"type": "eval_batch_request", "roots": roots, **extra}
    if allow_model_fallback:
        message["allow_model_fallback"] = True
    return json.dumps(message) + "\n"


def _assert_hello_equal_modulo_pipeline(
    test: unittest.TestCase, serial_hello: dict, pipeline_hello: dict
) -> None:
    expected = json.loads(json.dumps(serial_hello))
    test.assertIn("bridge_env", expected)
    test.assertFalse(expected["bridge_env"]["pipeline"])
    expected["bridge_env"]["pipeline"] = True
    test.assertEqual(pipeline_hello, expected)


class PipelineProtocolFallbackTest(unittest.TestCase):
    """Subprocess protocol tests against the REAL bridge, model-less (no
    manifest, per-request allow_model_fallback) — no torch/numpy required."""

    def test_burst_fifo_order_and_activation_banner(self) -> None:
        request_count = 12
        lines = [
            _batch_request(
                [
                    _fallback_root(f"r{index}-root{root_index}", 2 + (index + root_index) % 5)
                    for root_index in range(1 + index % 3)
                ]
            )
            for index in range(request_count)
        ]
        lines.append(json.dumps({"type": "shutdown"}) + "\n")
        payloads, returncode, stderr = _run_bridge_burst(lines, pipeline=True)

        self.assertEqual(returncode, 0)
        self.assertIn("bridge: pipeline mode ON", stderr)
        self.assertEqual(payloads[0]["type"], "hello")
        self.assertTrue(payloads[0]["bridge_env"]["pipeline"])
        self.assertEqual(payloads[-1], {"type": "shutdown", "status": "ok"})
        responses = payloads[1:-1]
        self.assertEqual(len(responses), request_count)
        for index, (request_line, payload) in enumerate(zip(lines, responses)):
            request = json.loads(request_line)
            self.assertEqual(payload["type"], "eval_batch_response", f"request {index}")
            results = payload["results"]
            self.assertEqual(len(results), len(request["roots"]))
            for root, result in zip(request["roots"], results):
                self.assertEqual(result["state_hash"], root["state_hash"])
                self.assertEqual(result["action_ids"], root["action_ids"])
                self.assertTrue(result["model_fallback"])
                # score_to_go is zero in fallback, so q is exactly the exact
                # afterstate vector — a per-request fingerprint.
                self.assertEqual(result["q"], root["exact_afterstate_score_active"])

    def test_pipeline_matches_serial_exactly_with_interleaved_traffic(self) -> None:
        lines: list[str] = []
        for index in range(10):
            lines.append(
                _batch_request(
                    [_fallback_root(f"m{index}-root{j}", 1 + (index + j) % 6) for j in range(2)]
                )
            )
        # Interleaved protocol traffic and errors, mid-burst, each of which
        # must land in its exact FIFO slot in both modes:
        lines.insert(3, json.dumps({"type": "hello"}) + "\n")
        lines.insert(5, "this is not json {\n")
        lines.insert(7, _batch_request([]))  # non-empty-roots ValueError
        lines.insert(9, json.dumps({"type": "eval_batch_request"}) + "\n")  # KeyError 'roots'
        lines.insert(
            11,
            _batch_request([_fallback_root("nofallback", 3)], allow_model_fallback=False),
        )  # RuntimeError: fallback disabled
        lines.insert(
            13,
            json.dumps(
                {
                    "type": "eval_request",
                    "allow_model_fallback": True,
                    "root": _fallback_root("single", 4),
                }
            )
            + "\n",
        )
        lines.insert(15, json.dumps({"type": "warp-speed"}) + "\n")  # unknown type
        lines.append(json.dumps({"type": "shutdown"}) + "\n")

        serial_payloads, serial_rc, serial_stderr = _run_bridge_burst(lines, pipeline=False)
        pipeline_payloads, pipeline_rc, pipeline_stderr = _run_bridge_burst(lines, pipeline=True)

        self.assertEqual(serial_rc, 0)
        self.assertEqual(pipeline_rc, 0)
        self.assertNotIn("pipeline mode ON", serial_stderr)
        self.assertIn("pipeline mode ON", pipeline_stderr)
        self.assertEqual(len(serial_payloads), len(lines) + 1)  # + initial hello
        _assert_hello_equal_modulo_pipeline(self, serial_payloads[0], pipeline_payloads[0])
        self.assertEqual(pipeline_payloads[1:], serial_payloads[1:])
        # Spot-check the interleaved slots carry what serial mode promises.
        self.assertEqual(serial_payloads[4]["type"], "hello")
        self.assertEqual(serial_payloads[6]["type"], "error")
        self.assertIn("non-empty roots", serial_payloads[8]["error"])
        self.assertEqual(serial_payloads[10], {"type": "error", "error": "'roots'"})
        self.assertIn("fallback is disabled", serial_payloads[12]["error"])
        self.assertEqual(serial_payloads[14]["type"], "eval_response")
        self.assertIn("unknown message type", serial_payloads[16]["error"])

    def test_shutdown_mid_burst_drains_cleanly(self) -> None:
        lines = [
            _batch_request([_fallback_root(f"pre{index}", 3 + index)]) for index in range(5)
        ]
        lines.append(json.dumps({"type": "shutdown"}) + "\n")
        # Lines after shutdown must be ignored in both modes.
        lines.extend(
            _batch_request([_fallback_root(f"post{index}", 2)]) for index in range(3)
        )

        serial_payloads, serial_rc, _ = _run_bridge_burst(lines, pipeline=False)
        pipeline_payloads, pipeline_rc, _ = _run_bridge_burst(lines, pipeline=True)

        self.assertEqual(serial_rc, 0)
        self.assertEqual(pipeline_rc, 0)
        # hello + 5 responses + shutdown ack, nothing for the post lines.
        self.assertEqual(len(serial_payloads), 7)
        self.assertEqual(serial_payloads[-1], {"type": "shutdown", "status": "ok"})
        _assert_hello_equal_modulo_pipeline(self, serial_payloads[0], pipeline_payloads[0])
        self.assertEqual(pipeline_payloads[1:], serial_payloads[1:])

    def test_eof_without_shutdown_drains_and_exits_zero(self) -> None:
        lines = [
            _batch_request([_fallback_root(f"eof{index}", 1 + index % 4)])
            for index in range(11)
        ]
        serial_payloads, serial_rc, _ = _run_bridge_burst(lines, pipeline=False)
        pipeline_payloads, pipeline_rc, _ = _run_bridge_burst(lines, pipeline=True)
        self.assertEqual(serial_rc, 0)
        self.assertEqual(pipeline_rc, 0)
        self.assertEqual(len(serial_payloads), len(lines) + 1)
        _assert_hello_equal_modulo_pipeline(self, serial_payloads[0], pipeline_payloads[0])
        self.assertEqual(pipeline_payloads[1:], serial_payloads[1:])


class PipelineUnitTest(unittest.TestCase):
    def test_provenance_pipeline_flag_defaults_off(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(bridge._pipeline_enabled())
            self.assertFalse(bridge.bridge_env_provenance()["pipeline"])
        with mock.patch.dict(os.environ, {"CASCADIA_BRIDGE_PIPELINE": "1"}, clear=True):
            self.assertTrue(bridge._pipeline_enabled())
            self.assertTrue(bridge.bridge_env_provenance()["pipeline"])
        # Anything but the literal "1" stays off.
        with mock.patch.dict(os.environ, {"CASCADIA_BRIDGE_PIPELINE": "true"}, clear=True):
            self.assertFalse(bridge._pipeline_enabled())

    def test_stdin_reader_delivers_lines_then_sentinel(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        line_queue: queue.Queue = queue.Queue()
        with mock.patch.object(sys, "stdin", io.StringIO("first\nsecond\n")):
            bridge._stdin_reader_loop(line_queue)
        self.assertEqual(line_queue.get_nowait(), "first\n")
        self.assertEqual(line_queue.get_nowait(), "second\n")
        self.assertIs(line_queue.get_nowait(), bridge._PIPELINE_EOF)
        self.assertTrue(line_queue.empty())

    def test_stdin_reader_delivers_sentinel_when_stdin_raises(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge

        class ExplodingStdin:
            def __iter__(self):  # type: ignore[no-untyped-def]
                raise OSError("stdin torn down")

        line_queue: queue.Queue = queue.Queue()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdin", ExplodingStdin()):
            with contextlib.redirect_stderr(stderr):
                bridge._stdin_reader_loop(line_queue)
        self.assertIs(line_queue.get_nowait(), bridge._PIPELINE_EOF)
        self.assertIn("stdin reader thread failed", stderr.getvalue())

    def test_execution_provenance_records_pipelining_knobs(self) -> None:
        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import execution_provenance

        with mock.patch.dict(os.environ, {}, clear=True):
            execution = execution_provenance(
                batch_runner=True, jobs=4, seed_count=3, device_name="cuda"
            )
        self.assertEqual(execution["shared_inflight"], 1)
        self.assertFalse(execution["bridge_pipeline"])
        with mock.patch.dict(
            os.environ,
            {"CASCADIA_SHARED_INFLIGHT": "3", "CASCADIA_BRIDGE_PIPELINE": "1"},
            clear=True,
        ):
            execution = execution_provenance(
                batch_runner=True, jobs=4, seed_count=3, device_name="cuda"
            )
        self.assertEqual(execution["shared_inflight"], 3)
        self.assertTrue(execution["bridge_pipeline"])

    def test_shared_inflight_from_env_mirrors_rust_resolution(self) -> None:
        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import shared_inflight_from_env

        cases = {
            None: 1,  # unset -> default serial
            "": 1,
            " 2 ": 2,  # Rust trims before parsing
            "8": 8,
            "9": 8,  # capped at 8, like shared_inflight() in model_bridge.rs
            "0": 1,  # non-positive -> default
            "-3": 1,
            "junk": 1,  # unparsable -> default
        }
        for raw, expected in cases.items():
            env = {} if raw is None else {"CASCADIA_SHARED_INFLIGHT": raw}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(shared_inflight_from_env(), expected, f"raw={raw!r}")


class PipelinePhaseSplitTorchTest(unittest.TestCase):
    """Torch-backed exactness pins: the phase-split path must equal the
    untouched _model_eval_batch path bit for bit on CPU."""

    @staticmethod
    def _packed_root(rng, tag: str, token_count: int, action_count: int, cfg) -> dict:  # type: ignore[no-untyped-def]
        import base64

        import numpy as np

        tokens = rng.standard_normal((token_count, cfg.token_feature_dim)).astype("<f4")
        actions = rng.standard_normal((action_count, cfg.action_feature_dim)).astype("<f4")
        tail = rng.integers(
            0, 8, size=(action_count, token_count + action_count), dtype=np.uint8
        )
        return {
            "schema_id": "pipeline-test.v1",
            "state_hash": f"hash-{tag}",
            "action_ids": [f"{tag}-a{index}" for index in range(action_count)],
            "exact_afterstate_score_active": [
                float(index + 1) * 2.5 for index in range(action_count)
            ],
            "packed_features": {
                "token_count": token_count,
                "action_count": action_count,
                "token_feature_dim": int(tokens.shape[1]),
                "action_feature_dim": int(actions.shape[1]),
                "tokens_f32_b64": base64.b64encode(tokens.tobytes()).decode("ascii"),
                "actions_f32_b64": base64.b64encode(actions.tobytes()).decode("ascii"),
                "relation_tail_u8_b64": base64.b64encode(tail.tobytes()).decode("ascii"),
            },
        }

    def _roots(self, cfg, count: int = 7, seed: int = 20260712) -> list[dict]:  # type: ignore[no-untyped-def]
        import numpy as np

        rng = np.random.default_rng(seed)
        return [
            self._packed_root(rng, f"root{index}", 5 + index % 3, 2 + index % 4, cfg)
            for index in range(count)
        ]

    def _run_phase_split(self, model, roots, **kwargs):  # type: ignore[no-untyped-def]
        from cascadiav3.torch_inference_bridge import _PipelinedEvalBatch

        state = _PipelinedEvalBatch(model, roots, **kwargs)
        state.prepare()
        state.launch()
        return state.finalize()

    def test_phase_split_equals_model_eval_batch_exactly(self) -> None:
        torch = _require_torch(self)
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _model_eval_batch

        cfg = config_for_size("tiny")
        torch.manual_seed(20260712)
        model = build_cascadiaformer(cfg).eval()
        roots = self._roots(cfg)

        for packed_response in (False, True):
            with self.subTest(packed_response=packed_response):
                # chunk_size=2 forces multiple chunks through both paths.
                serial = _model_eval_batch(
                    model, roots, chunk_size=2, packed_response=packed_response
                )
                pipelined = self._run_phase_split(
                    model, roots, chunk_size=2, packed_response=packed_response
                )
                self.assertEqual(len(serial), len(roots))
                self.assertEqual(pipelined, serial)

    def test_phase_split_exact_for_pairwise_and_quantile_modes(self) -> None:
        torch = _require_torch(self)
        from dataclasses import replace

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _model_eval_batch

        base_cfg = config_for_size("tiny")
        scenarios = [
            (
                replace(base_cfg, pairwise_comparator=True, pairwise_rank=8),
                {"policy_mode": "pairwise-borda", "pairwise_policy_top_k": 4},
            ),
            (
                replace(base_cfg, pairwise_comparator=True, pairwise_rank=8),
                {"policy_mode": "logits-plus-pairwise", "pairwise_policy_top_k": 4},
            ),
            (replace(base_cfg, q_quantiles=4), {"q_risk_mode": "q50"}),
        ]
        for cfg, kwargs in scenarios:
            with self.subTest(**kwargs):
                torch.manual_seed(20260713)
                model = build_cascadiaformer(cfg).eval()
                roots = self._roots(cfg, count=5, seed=20260714)
                serial = _model_eval_batch(model, roots, chunk_size=2, **kwargs)
                pipelined = self._run_phase_split(model, roots, chunk_size=2, **kwargs)
                self.assertEqual(pipelined, serial)

    def test_phase_split_exact_for_ensemble_model(self) -> None:
        torch = _require_torch(self)
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _EnsembleModel, _model_eval_batch

        cfg = config_for_size("tiny")
        torch.manual_seed(1)
        model_a = build_cascadiaformer(cfg).eval()
        torch.manual_seed(2)
        model_b = build_cascadiaformer(cfg).eval()
        ensemble = _EnsembleModel([model_a, model_b])
        roots = self._roots(cfg, count=4, seed=20260715)
        serial = _model_eval_batch(ensemble, roots, chunk_size=2)
        pipelined = self._run_phase_split(ensemble, roots, chunk_size=2)
        self.assertEqual(pipelined, serial)

    def test_phase_split_timing_records_same_chunk_accounting(self) -> None:
        _require_torch(self)
        import torch

        from cascadiav3 import torch_inference_bridge as bridge
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        cfg = config_for_size("tiny")
        torch.manual_seed(20260716)
        model = build_cascadiaformer(cfg).eval()
        roots = self._roots(cfg, count=6, seed=20260717)

        serial_timing = bridge._BridgeTiming()
        with mock.patch.object(bridge, "_BRIDGE_TIMING", serial_timing):
            bridge._model_eval_batch(model, roots, chunk_size=2)
        pipeline_timing = bridge._BridgeTiming()
        with mock.patch.object(bridge, "_BRIDGE_TIMING", pipeline_timing):
            self._run_phase_split(model, roots, chunk_size=2)

        self.assertGreater(serial_timing.chunks, 1)
        self.assertEqual(pipeline_timing.chunks, serial_timing.chunks)
        self.assertEqual(pipeline_timing.rows, serial_timing.rows)
        self.assertEqual(pipeline_timing.actions, serial_timing.actions)
        for field in ("collate_s", "h2d_s", "forward_s", "d2h_s", "encode_s"):
            self.assertGreaterEqual(getattr(pipeline_timing, field), 0.0)

    def test_pipeline_serve_subprocess_matches_serial_with_model(self) -> None:
        torch = _require_torch(self)
        import tempfile

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        cfg = config_for_size("tiny")
        torch.manual_seed(20260718)
        model = build_cascadiaformer(cfg)
        roots = self._roots(cfg, count=14, seed=20260719)

        with tempfile.TemporaryDirectory() as tempdir:
            weights_path = Path(tempdir) / "tiny.weights.pt"
            torch.save(model.state_dict(), weights_path)
            manifest_path = Path(tempdir) / "tiny.manifest.json"
            manifest_path.write_text(
                json.dumps({"config": cfg.to_dict(), "weights": str(weights_path)}),
                encoding="utf-8",
            )

            lines: list[str] = []
            for index in range(6):
                lines.append(
                    _batch_request(
                        roots[2 * index : 2 * index + 2],
                        allow_model_fallback=False,
                        packed_response=index % 2 == 1,
                    )
                )
            # Interleaved traffic mid-burst forces drains around outstanding
            # model-backed requests.
            lines.insert(2, json.dumps({"type": "hello"}) + "\n")
            lines.insert(4, "malformed json mid burst\n")
            lines.insert(
                6,
                json.dumps(
                    {"type": "eval_request", "root": roots[12], "packed_response": False}
                )
                + "\n",
            )
            lines.append(json.dumps({"type": "shutdown"}) + "\n")
            args = ("--manifest", str(manifest_path), "--device", "cpu")

            serial_payloads, serial_rc, _ = _run_bridge_burst(
                lines, pipeline=False, args=args, timeout=MODEL_BURST_TIMEOUT_S
            )
            pipeline_payloads, pipeline_rc, pipeline_stderr = _run_bridge_burst(
                lines, pipeline=True, args=args, timeout=MODEL_BURST_TIMEOUT_S
            )

        self.assertEqual(serial_rc, 0)
        self.assertEqual(pipeline_rc, 0)
        self.assertIn("bridge: pipeline mode ON", pipeline_stderr)
        self.assertEqual(len(serial_payloads), len(lines) + 1)
        self.assertTrue(serial_payloads[0]["model_loaded"])
        _assert_hello_equal_modulo_pipeline(self, serial_payloads[0], pipeline_payloads[0])
        self.assertEqual(pipeline_payloads[1:], serial_payloads[1:])
        # Sanity: the burst exercised real model batches (packed + JSON), the
        # malformed-line error slot, and the interleaved hello.
        types = [payload["type"] for payload in serial_payloads[1:]]
        self.assertEqual(types.count("eval_batch_response"), 6)
        self.assertEqual(types.count("error"), 1)
        self.assertEqual(types.count("hello"), 1)
        self.assertEqual(types.count("eval_response"), 1)
        self.assertTrue(
            any(
                "packed" in result
                for payload in serial_payloads
                if payload.get("type") == "eval_batch_response"
                for result in payload["results"]
            )
        )


if __name__ == "__main__":
    unittest.main()
