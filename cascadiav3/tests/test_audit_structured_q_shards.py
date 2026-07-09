"""Cross-shard admission tests for raw structured-Q data."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cascadiav3.audit_structured_q_shards import SeedDomain, audit_shards, parse_seed_domain
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
                expected_seed_domains={
                    "first": SeedDomain(10, 1, 4, "gumbel_selfplay_tensor_corpus"),
                    "second": SeedDomain(11, 1, 4, "gumbel_selfplay_tensor_corpus"),
                },
                expected_source_revision="source",
                expected_teacher_manifest_sha256="1" * 64,
                expected_teacher_weights_sha256="2" * 64,
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["totals"]["seeds"], 2)
            self.assertEqual(report["totals"]["records"], 8)
            self.assertEqual(report["totals"]["exact_rows"], 8)
            distribution = report["primary"][0]["target_distribution"]
            self.assertIsNone(distribution["final_score_non_exact"])
            self.assertEqual(distribution["q_valid_per_root"]["mean"], 1.0)

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
            with self.assertRaisesRegex(ValueError, "does not match expectation"):
                audit_shards(
                    {"first": first},
                    expected_seed_domains={
                        "first": SeedDomain(12, 1, 4, "gumbel_selfplay_tensor_corpus")
                    },
                )
            with self.assertRaisesRegex(ValueError, "labels do not match"):
                audit_shards(
                    {"first": first},
                    expected_seed_domains={
                        "other": SeedDomain(10, 1, 4, "gumbel_selfplay_tensor_corpus")
                    },
                )
            manifest = json.loads(first.with_suffix(".manifest.json").read_text())
            manifest["checksum"] = "0" * 64
            first.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit_shards({"first": first})

    def test_fetch_script_pins_hosts_hashes_exclusions_and_quarantine(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "fetch_structured_q_expansion.sh"
        ).read_text(encoding="utf-8")
        for host, label in (
            ("john2", "expansion_a"),
            ("john3", "expansion_b"),
            ("john4", "expansion_c"),
        ):
            self.assertIn(f"fetch_one {host} {label}", script)
        self.assertIn("6e89d9555f6126bdc29f65657d8431cab3d2c024", script)
        self.assertIn("b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d", script)
        self.assertIn("33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354", script)
        self.assertEqual(script.count("--exclude-shard"), 3)
        self.assertEqual(script.count("--expected-seed-domain"), 3)
        self.assertNotIn("john0:", script)
        self.assertIn("data remains quarantined", script)

    def test_reserve_holdouts_are_preregistered_and_data_only(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "arm_structured_q_reserve_holdouts.sh"
        ).read_text(encoding="utf-8")
        for host, current, reserve, seed in (
            ("john2", "expansion_a", "reserve_selection", 2027073750),
            ("john3", "expansion_b", "reserve_verdict", 2027073770),
            ("john4", "expansion_c", "reserve_replication", 2027073790),
        ):
            self.assertIn(f"arm_one {host} {current} {reserve} {seed}", script)
        self.assertIn("SEED_COUNT=20", script)
        self.assertIn("current expansion validation failed", script)
        self.assertIn('--manifest "${out%.npz}.manifest.json"', script)
        self.assertIn("no fetch or training action", script)
        self.assertNotIn("ssh john0", script)

    def test_reserve_fetch_pins_roles_domains_exclusions_and_quarantine(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "fetch_structured_q_reserve_holdouts.sh"
        ).read_text(encoding="utf-8")
        for host, label, seed in (
            ("john2", "reserve_selection", 2027073750),
            ("john3", "reserve_verdict", 2027073770),
            ("john4", "reserve_replication", 2027073790),
        ):
            self.assertIn(f"fetch_one {host} {label} {seed}", script)
        self.assertEqual(script.count("--exclude-shard"), 6)
        self.assertEqual(script.count("--expected-seed-domain"), 3)
        self.assertIn("structured_q_v4_expansion_20260709", script)
        self.assertIn("holdouts remain quarantined", script)
        self.assertNotIn("john0:", script)


if __name__ == "__main__":
    unittest.main()
