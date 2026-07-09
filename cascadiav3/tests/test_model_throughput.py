from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cascadiav3.torch_cascadiaformer import (
    build_cascadiaformer,
    config_for_size,
    parameter_count,
)
from cascadiav3.torch_public_token_merit import PUBLIC_TOKEN_FEATURE_DIM
from cascadiav3.torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
)
from cascadiav3.torch_model_throughput_benchmark import (
    _p95,
    load_roots,
    parse_positive_ints,
    prepare_roots,
    production_packed_root,
    run_benchmark,
)


class ModelThroughputBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _packed_root() -> dict[str, object]:
        token_count = 1
        action_count = 2
        sequence_length = token_count + action_count
        tokens = np.zeros((token_count, PUBLIC_TOKEN_FEATURE_DIM), dtype="<f4")
        actions = np.zeros(
            (action_count, SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM), dtype="<f4"
        )
        relation_tail = np.zeros((action_count, sequence_length), dtype=np.uint8)
        return {
            "schema_id": "throughput-test",
            "state_hash": "throughput-test-root",
            "active_seat": 0,
            "action_ids": ["a", "b"],
            "exact_afterstate_score_active": [10.0, 11.0],
            "packed_features": {
                "token_count": token_count,
                "action_count": action_count,
                "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
                "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
                "tokens_f32_b64": base64.b64encode(tokens.tobytes()).decode("ascii"),
                "actions_f32_b64": base64.b64encode(actions.tobytes()).decode("ascii"),
                "relation_tail_u8_b64": base64.b64encode(relation_tail.tobytes()).decode(
                    "ascii"
                ),
            },
        }

    def test_xs_configuration_is_a_real_intermediate_student(self) -> None:
        tiny = parameter_count(build_cascadiaformer(config_for_size("tiny")))
        xs = parameter_count(build_cascadiaformer(config_for_size("XS")))
        small = parameter_count(build_cascadiaformer(config_for_size("S")))

        self.assertEqual(config_for_size("XS").model_size, "XS")
        self.assertEqual(config_for_size("XS").d_model, 256)
        self.assertEqual(config_for_size("XS").layers, 6)
        self.assertLess(tiny, xs)
        self.assertLess(xs, small)

    def test_parsers_and_percentile_fail_closed(self) -> None:
        self.assertEqual(parse_positive_ints("1,2,2, 8"), [1, 2, 8])
        self.assertEqual(_p95([1.0, 2.0, 3.0, 4.0]), 4.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_positive_ints("1,0")
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_positive_ints("")
        with self.assertRaisesRegex(ValueError, "empty"):
            _p95([])

    def test_raw_audit_root_is_packed_before_timing(self) -> None:
        from unittest import mock

        raw = {
            "schema_id": "audit-root",
            "ruleset_id": "corrected-rules",
            "state_hash": "state",
            "active_seat": 2,
            "action_ids": ["a", "b"],
            "exact_afterstate_score_active": [10.0, 11.0],
            "legal_actions": [{"action_id": "a"}, {"action_id": "b"}],
            "public_tokens": {"tokens": [{}], "token_count": 1},
        }
        with (
            mock.patch(
                "cascadiav3.torch_public_token_merit.public_token_features",
                return_value=[[0.0] * PUBLIC_TOKEN_FEATURE_DIM],
            ),
            mock.patch(
                "cascadiav3.torch_semantic_relation_bias_merit.semantic_public_token_action_features",
                return_value=[
                    [0.0] * SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
                    [0.0] * SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
                ],
            ),
            mock.patch(
                "cascadiav3.torch_relation_bias_merit.combined_relation_ids_array",
                return_value=np.zeros((3, 3), dtype=np.uint8),
            ),
        ):
            packed = production_packed_root(raw)

        self.assertNotIn("legal_actions", packed)
        self.assertNotIn("public_tokens", packed)
        self.assertEqual(packed["action_ids"], ["a", "b"])
        self.assertEqual(packed["ruleset_id"], "corrected-rules")
        self.assertEqual(packed["packed_features"]["token_count"], 1)
        self.assertEqual(packed["packed_features"]["action_count"], 2)
        self.assertEqual(prepare_roots([packed], "production-packed"), [packed])
        self.assertIs(prepare_roots([raw], "as-is")[0], raw)
        with self.assertRaisesRegex(ValueError, "unsupported root format"):
            prepare_roots([raw], "raw-ish")

    def test_cpu_shape_probe_is_deterministic_and_reports_end_to_end_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots_path = Path(tmp) / "roots.jsonl"
            roots_path.write_text(
                json.dumps(self._packed_root()) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_roots(roots_path)), 1)

            report = run_benchmark(
                roots_path=roots_path,
                manifests=[],
                synthetic_model_sizes=["tiny"],
                batch_sizes=[1, 2],
                warmup_iterations=0,
                measured_iterations=2,
                device_name="cpu",
                baseline_label="synthetic_tiny",
                source_revision="tested-revision",
            )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["scientific_eligibility"], "engineering_throughput_only")
        self.assertEqual(report["source_revision"], "tested-revision")
        self.assertEqual(report["roots"]["benchmark_format"], "production-packed")
        self.assertEqual(len(report["roots"]["benchmark_payload_sha256"]), 64)
        self.assertEqual(report["baseline_label"], "synthetic_tiny")
        self.assertEqual([row["batch_size"] for row in report["models"][0]["batches"]], [1, 2])
        self.assertTrue(
            all(row["rows_per_second"] > 0.0 for row in report["models"][0]["batches"])
        )
        self.assertTrue(
            all(
                row["throughput_speedup_vs_baseline"] == 1.0
                for row in report["comparisons"][0]["batches"]
            )
        )


if __name__ == "__main__":
    unittest.main()
