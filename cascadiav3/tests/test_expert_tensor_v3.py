from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class ExpertTensorV3Test(unittest.TestCase):
    def _write_v3(
        self,
        path: Path,
        *,
        eligibility: str = "gumbel_selfplay_expert_iteration",
    ) -> None:
        import numpy as np

        from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard

        _save_expert_tensor_shard(
            out_path=path,
            metadata={
                "schema_id": "cascadiav3.expert_tensor_shard.v3",
                "ruleset_id": "test-ruleset",
                "source_revision": "test-source-revision",
                "mode": "gumbel_selfplay_tensor_corpus",
                "scientific_eligibility": eligibility,
                "search": {
                    "n_simulations": 4,
                    "top_m": 2,
                    "depth_rounds": 1,
                    "determinization_samples": 2,
                    "market_decision_samples": 2,
                    "exact_endgame_turns": 1,
                    "rollout_blend_weight": 0.5,
                    "exploration": True,
                    "peek": False,
                    "table_total": False,
                    "table_native_q": False,
                    "leaf_softmix": None,
                    "tta": 1,
                    "k_interior": 2,
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
                },
                "generator": {"sha256": "3" * 64, "bytes": 30},
                "created_unix_seconds": 1_700_000_000,
                "canonical_targets": [
                    "improved_policy",
                    "search_root_value",
                    "exact_endgame",
                ],
            },
            tokens=np.zeros((2, 41), dtype=np.float16),
            actions=np.zeros((3, 61), dtype=np.float16),
            token_offsets=np.asarray([0, 2], dtype=np.int64),
            action_offsets=np.asarray([0, 3], dtype=np.int64),
            relation_edges=np.zeros((0, 3), dtype=np.int32),
            relation_offsets=np.asarray([0, 0], dtype=np.int64),
            selected_action_index=np.asarray([0], dtype=np.int16),
            target_q=np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            target_score_to_go=np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            q_valid=np.asarray([1, 1, 1], dtype=np.uint8),
            priors=np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
            visits=np.asarray([2.0, 2.0, 1.0], dtype=np.float32),
            q_variance=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            q_count=np.asarray([2.0, 2.0, 1.0], dtype=np.float32),
            truncated_count=np.zeros((3,), dtype=np.float32),
            exact_afterstate_score_active=np.zeros((3,), dtype=np.float32),
            final_score_vector=np.asarray([[100.0, 90.0, 80.0, 70.0]], dtype=np.float32),
            rank_vector=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
            score_decomposition=np.zeros((1, 3, 4), dtype=np.float32),
            improved_policy=np.asarray([0.6, 0.3, 0.1], dtype=np.float32),
            search_root_value=np.asarray([3.0], dtype=np.float32),
            exact_endgame=np.asarray([1], dtype=np.uint8),
        )

    def test_v3_exact_endgame_survives_filter_tail_and_collation(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.expert_tensor_shards import (
            SHARD_VERSION_V3,
            ExpertTensorCorpus,
            ExpertTensorShard,
            collate_expert_tensor_examples,
            filter_expert_tensor_shard,
            materialize_relation_tail_shard,
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            filtered = Path(directory) / "filtered.npz"
            tailed = Path(directory) / "tailed.npz"
            self._write_v3(source)
            filter_expert_tensor_shard(source, filtered, top_k=2)
            materialize_relation_tail_shard(filtered, tailed)

            shard = ExpertTensorShard(tailed)
            try:
                self.assertEqual(shard.version, SHARD_VERSION_V3)
                self.assertEqual(shard.exact_endgame.tolist(), [1])
                example = shard.example(0)
                self.assertTrue(example["exact_endgame"])
                self.assertEqual(example["schema_id"], SHARD_VERSION_V3)
                batch = collate_expert_tensor_examples([example])
                self.assertEqual(batch["schema_ids"], [SHARD_VERSION_V3])
                self.assertEqual(batch["exact_endgame"].tolist(), [True])
            finally:
                shard.close()
            corpus = ExpertTensorCorpus([tailed])
            try:
                self.assertEqual(corpus.schema_ids(), [SHARD_VERSION_V3])
            finally:
                corpus.close()

    def test_v3_without_exact_endgame_fails_closed(self) -> None:
        import numpy as np

        from cascadiav3.expert_tensor_shards import ExpertTensorShard

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.npz"
            self._write_v3(path)
            with np.load(path, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files if name != "exact_endgame"}
            with path.open("wb") as handle:
                np.savez(handle, **arrays)
            with self.assertRaisesRegex(ValueError, "requires exact_endgame"):
                ExpertTensorShard(path)

    def test_audit_only_v3_is_readable_but_rejected_for_training(self) -> None:
        from cascadiav3.expert_tensor_shards import ExpertTensorCorpus, ExpertTensorShard

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit-only.npz"
            self._write_v3(
                path,
                eligibility="audit_only_unverified_or_uniform_model_fallback",
            )
            shard = ExpertTensorShard(path)
            shard.close()
            with self.assertRaisesRegex(ValueError, "not training eligible"):
                ExpertTensorCorpus([path])


if __name__ == "__main__":
    unittest.main()
