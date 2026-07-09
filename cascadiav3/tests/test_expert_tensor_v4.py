"""Schema-v4 exact component grounding survives every packed-data transform."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class ExpertTensorV4Test(unittest.TestCase):
    def _write_v4(
        self,
        path: Path,
        *,
        exact_scalar_offset: float = 0.0,
        terminal_component_offset: float = 0.0,
    ) -> None:
        import numpy as np

        from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard

        exact_components = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 4.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        exact_scalar = exact_components.sum(axis=1)
        exact_scalar[0] += exact_scalar_offset
        score_decomposition = np.asarray(
            [
                [
                    [5.0 + terminal_component_offset, 3.0, 3.0, 2.0],
                    [7.0, 4.0, 3.0, 3.0],
                    [9.0, 3.0, 3.0, 3.0],
                ]
            ],
            dtype=np.float32,
        )
        _save_expert_tensor_shard(
            out_path=path,
            metadata={
                "schema_id": "cascadiav3.expert_tensor_shard.v4",
                "ruleset_id": "test-ruleset",
                "source_revision": "test-source-revision",
                "mode": "gumbel_selfplay_tensor_corpus",
                "scientific_eligibility": "gumbel_selfplay_expert_iteration",
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
                    "active_seat",
                    "exact_afterstate_score_decomposition_active",
                ],
            },
            tokens=np.zeros((2, 41), dtype=np.float16),
            actions=np.zeros((3, 61), dtype=np.float16),
            token_offsets=np.asarray([0, 2], dtype=np.int64),
            action_offsets=np.asarray([0, 3], dtype=np.int64),
            relation_edges=np.zeros((0, 3), dtype=np.int32),
            relation_offsets=np.asarray([0, 0], dtype=np.int64),
            selected_action_index=np.asarray([1], dtype=np.int16),
            target_q=np.asarray([20.0, 21.0, 19.0], dtype=np.float32),
            target_score_to_go=np.asarray([14.0, 12.0, 16.0], dtype=np.float32),
            q_valid=np.asarray([1, 1, 1], dtype=np.uint8),
            priors=np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
            visits=np.asarray([2.0, 2.0, 1.0], dtype=np.float32),
            q_variance=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            q_count=np.asarray([2.0, 2.0, 1.0], dtype=np.float32),
            truncated_count=np.zeros((3,), dtype=np.float32),
            exact_afterstate_score_active=exact_scalar,
            exact_afterstate_score_decomposition_active=exact_components,
            active_seat=np.asarray([0], dtype=np.uint8),
            final_score_vector=np.asarray([[21.0, 10.0, 9.0, 8.0]], dtype=np.float32),
            rank_vector=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
            score_decomposition=score_decomposition,
            improved_policy=np.asarray([0.6, 0.3, 0.1], dtype=np.float32),
            search_root_value=np.asarray([3.0], dtype=np.float32),
            exact_endgame=np.asarray([1], dtype=np.uint8),
        )

    def test_v4_grounding_survives_filter_tail_and_collation(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.expert_tensor_shards import (
            SHARD_VERSION_V4,
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
            self._write_v4(source)
            filter_expert_tensor_shard(source, filtered, top_k=2)
            materialize_relation_tail_shard(filtered, tailed)

            shard = ExpertTensorShard(tailed)
            try:
                self.assertEqual(shard.version, SHARD_VERSION_V4)
                self.assertEqual(shard.active_seat.tolist(), [0])
                self.assertEqual(
                    shard.exact_afterstate_score_decomposition_active.shape,
                    (2, 3),
                )
                example = shard.example(0)
                batch = collate_expert_tensor_examples([example])
                self.assertTrue(batch["has_structured_grounding"])
                self.assertEqual(batch["active_seat"].tolist(), [0])
                self.assertEqual(
                    tuple(batch["exact_afterstate_score_decomposition_active"].shape),
                    (1, 2, 3),
                )
                torch.testing.assert_close(
                    batch["exact_afterstate_score_decomposition_active"].sum(dim=-1),
                    batch["exact_afterstate_score_active"],
                )
            finally:
                shard.close()
            corpus = ExpertTensorCorpus([tailed])
            try:
                self.assertEqual(corpus.schema_ids(), [SHARD_VERSION_V4])
            finally:
                corpus.close()

    def test_v4_rejects_inconsistent_afterstate_and_terminal_components(self) -> None:
        from cascadiav3.expert_tensor_shards import ExpertTensorShard

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-afterstate.npz"
            self._write_v4(path, exact_scalar_offset=1.0)
            with self.assertRaisesRegex(ValueError, "afterstate score decomposition total mismatch"):
                ExpertTensorShard(path)

            path = Path(directory) / "bad-terminal.npz"
            self._write_v4(path, terminal_component_offset=1.0)
            with self.assertRaisesRegex(ValueError, "terminal score decomposition"):
                ExpertTensorShard(path)

    def test_structured_head_only_training_runs_end_to_end_on_v4(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_train_cascadiaformer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            self._write_v4(source)
            report = run_training(
                [source],
                [source],
                train_format="npz",
                val_format="npz",
                model_size="tiny",
                q_decomposition=True,
                q_decomposition_head_only=True,
                steps=2,
                batch_size=1,
                lr=1.0e-3,
                weight_decay=0.0,
                device_name="cpu",
                seed=17,
                grad_accum=1,
                warmup_fraction=0.1,
                checkpoint_dir=root / "checkpoints",
                metrics_jsonl=root / "metrics.jsonl",
                out=root / "report.json",
                overfit_one_batch=True,
                val_max_batches=1,
                swa_fraction=0.5,
                objective="gumbel-selfplay-structured-q",
                eval_every_steps=1,
            )
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["config"]["q_decomposition"])
            self.assertTrue(report["q_decomposition_head_only"])
            self.assertEqual(report["trainable_parameter_count"], (64 + 1) * 3)
            self.assertEqual(report["schema_ids"], ["cascadiav3.expert_tensor_shard.v4"])
            self.assertIn("locked_val_q_decomposition", report["latest_metrics"])


if __name__ == "__main__":
    unittest.main()
