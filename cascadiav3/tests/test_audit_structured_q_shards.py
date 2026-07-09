"""Cross-shard admission tests for raw structured-Q data."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cascadiav3.audit_structured_q_shards import audit_shards, parse_seed_domain
from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard


class AuditStructuredQShardsTest(unittest.TestCase):
    def _write(self, path: Path, *, first_seed: int, source_revision: str = "source") -> None:
        records = 4
        metadata = {
            "schema_id": "cascadiav3.expert_tensor_shard.v4",
            "ruleset_id": "rules",
            "source_revision": source_revision,
            "mode": "gumbel_selfplay_tensor_corpus",
            "scientific_eligibility": "gumbel_selfplay_expert_iteration",
            "seed_domain": (
                f"first_seed={first_seed},seed_count=1,plies_per_seed=4,"
                "max_actions=1,mode=gumbel_selfplay_tensor_corpus"
            ),
            "search": {
                "n_simulations": 1,
                "top_m": 1,
                "depth_rounds": 1,
                "determinization_samples": 1,
                "market_decision_samples": 1,
                "exact_endgame_turns": 1,
                "rollout_blend_weight": 0.5,
                "exploration": True,
                "peek": False,
                "table_total": False,
                "table_native_q": False,
                "leaf_softmix": None,
                "tta": 1,
                "k_interior": 1,
                "max_root_actions": None,
                "root_menu": 16,
            },
            "execution": {
                "rayon_threads_requested": 1,
                "rayon_current_num_threads": 1,
                "model_sessions_requested": 1,
                "shared_model_session": True,
                "seed_scheduler": "dynamic_atomic_queue",
                "model_session_topology": "one_shared_bridge_with_worker_clients",
            },
            "teacher_model": {
                "manifest": {"sha256": "1" * 64, "bytes": 10},
                "weights": {"sha256": "2" * 64, "bytes": 20},
                "checkpoint_tag": "best",
                "step": 7,
                "model_name": "M",
                "model_size": "M",
            },
            "generator": {"sha256": "3" * 64, "bytes": 30},
            "created_unix_seconds": 1_700_000_000,
            "canonical_targets": [
                "improved_policy",
                "search_root_value",
                "exact_endgame",
                "active_seat",
                "exact_afterstate_score_decomposition_active",
            ],
        }
        exact_components = np.tile(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), (records, 1))
        score_components = np.tile(
            np.asarray([[[1.0] * 4, [2.0] * 4, [3.0] * 4]], dtype=np.float32),
            (records, 1, 1),
        )
        _save_expert_tensor_shard(
            out_path=path,
            metadata=metadata,
            tokens=np.zeros((records, 41), dtype=np.float16),
            actions=np.zeros((records, 61), dtype=np.float16),
            token_offsets=np.arange(records + 1, dtype=np.int64),
            action_offsets=np.arange(records + 1, dtype=np.int64),
            relation_edges=np.zeros((0, 3), dtype=np.int32),
            relation_offsets=np.zeros((records + 1,), dtype=np.int64),
            selected_action_index=np.zeros((records,), dtype=np.int16),
            target_q=np.full((records,), 6.0, dtype=np.float32),
            target_score_to_go=np.zeros((records,), dtype=np.float32),
            q_valid=np.ones((records,), dtype=np.uint8),
            priors=np.ones((records,), dtype=np.float32),
            visits=np.ones((records,), dtype=np.float32),
            q_variance=np.zeros((records,), dtype=np.float32),
            q_count=np.ones((records,), dtype=np.float32),
            truncated_count=np.zeros((records,), dtype=np.float32),
            exact_afterstate_score_active=np.full((records,), 6.0, dtype=np.float32),
            exact_afterstate_score_decomposition_active=exact_components,
            active_seat=np.zeros((records,), dtype=np.uint8),
            final_score_vector=np.full((records, 4), 6.0, dtype=np.float32),
            rank_vector=np.tile(np.arange(1, 5, dtype=np.int16), (records, 1)),
            score_decomposition=score_components,
            improved_policy=np.ones((records,), dtype=np.float32),
            search_root_value=np.full((records,), 6.0, dtype=np.float32),
            exact_endgame=np.ones((records,), dtype=np.uint8),
        )
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "checksum": checksum,
                    "metadata": metadata,
                    "record_count": records,
                    "schema_id": metadata["schema_id"],
                    "scientific_eligibility": metadata["scientific_eligibility"],
                    "seed_domain": metadata["seed_domain"],
                    "total_action_count": records,
                    "version": metadata["schema_id"],
                }
            ),
            encoding="utf-8",
        )

    def test_seed_domain_parser_is_strict(self) -> None:
        parsed = parse_seed_domain(
            "first_seed=10,seed_count=2,plies_per_seed=80,mode=gumbel_selfplay_tensor_corpus"
        )
        self.assertEqual(parsed.last_seed, 11)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_seed_domain(
                "first_seed=10,first_seed=11,seed_count=1,plies_per_seed=80,mode=x"
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_seed_domain("first_seed=10,seed_count=1,mode=x")

    def test_audit_accepts_one_contract_and_disjoint_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npz"
            second = root / "second.npz"
            excluded = root / "excluded.npz"
            self._write(first, first_seed=10)
            self._write(second, first_seed=11)
            self._write(excluded, first_seed=9)
            report = audit_shards(
                {"first": first, "second": second},
                excluded_shards={"locked": excluded},
                expected_source_revision="source",
                expected_teacher_manifest_sha256="1" * 64,
                expected_teacher_weights_sha256="2" * 64,
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["totals"]["seeds"], 2)
            self.assertEqual(report["totals"]["records"], 8)
            self.assertEqual(report["totals"]["exact_rows"], 8)

    def test_audit_rejects_overlap_contract_and_sidecar_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npz"
            overlap = root / "overlap.npz"
            mismatch = root / "mismatch.npz"
            self._write(first, first_seed=10)
            self._write(overlap, first_seed=10)
            self._write(mismatch, first_seed=11, source_revision="other")
            with self.assertRaisesRegex(ValueError, "seed overlap"):
                audit_shards({"first": first, "overlap": overlap})
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                audit_shards({"first": first, "mismatch": mismatch})
            manifest = json.loads(first.with_suffix(".manifest.json").read_text())
            manifest["checksum"] = "0" * 64
            first.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit_shards({"first": first})


if __name__ == "__main__":
    unittest.main()
