from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest


class PolicyCandidateProbeTest(unittest.TestCase):
    def test_prior_filter_retains_every_q_valid_action_before_filling(self) -> None:
        import numpy as np

        from cascadiav3.expert_tensor_shards import _retained_action_indices

        keep = _retained_action_indices(
            np.arange(6, dtype=np.float32),
            np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8),
            selected_action_index=3,
            top_k=4,
            filter_mode="top-prior-with-q-valid",
            priors=np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.04], dtype=np.float32),
        )
        self.assertEqual(keep.tolist(), [0, 3, 4, 5])

    def test_policy_recall_hinge_rewards_teacher_best_inside_top16(self) -> None:
        import torch

        from cascadiav3.torch_train_cascadiaformer import _policy_recall_terms

        logits = torch.arange(20, dtype=torch.float32).unsqueeze(0)
        action_mask = torch.ones_like(logits, dtype=torch.bool)
        target_q = torch.zeros_like(logits)
        target_q[0, 0] = 2.0
        target_q[0, 1] = 1.0
        q_valid = torch.zeros_like(logits, dtype=torch.bool)
        q_valid[0, :2] = True
        q_count = torch.full_like(logits, 4.0)
        q_variance = torch.full_like(logits, 0.01)
        bad_loss, bad_recall, bad_confident_recall, examples = _policy_recall_terms(
            logits,
            action_mask,
            target_q,
            q_valid,
            q_count,
            q_variance,
        )
        improved = logits.clone()
        improved[0, 0] = 30.0
        good_loss, good_recall, good_confident_recall, _ = _policy_recall_terms(
            improved,
            action_mask,
            target_q,
            q_valid,
            q_count,
            q_variance,
        )
        self.assertEqual(float(examples), 1.0)
        self.assertGreater(float(bad_loss), float(good_loss))
        self.assertEqual(float(bad_recall), 0.0)
        self.assertEqual(float(good_recall), 1.0)
        self.assertEqual(float(bad_confident_recall), 0.0)
        self.assertEqual(float(good_confident_recall), 1.0)

    def test_identical_checkpoints_have_zero_paired_deltas(self) -> None:
        import torch

        from test_expert_tensor_v3 import ExpertTensorV3Test

        from cascadiav3.expert_tensor_shards import filter_expert_tensor_shard
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_policy_candidate_probe import run_probe

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = root / "val.npz"
            ExpertTensorV3Test()._write_v3(tensor)
            config = replace(config_for_size("tiny"), dropout=0.0)
            model = build_cascadiaformer(config)
            weights = root / "weights.pt"
            torch.save(model.state_dict(), weights)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoint_tag": "test",
                        "step": 0,
                        "config": config.to_dict(),
                        "weights": weights.name,
                        "weights_format": "torch_state_dict",
                    }
                ),
                encoding="utf-8",
            )
            report = run_probe(
                baseline_manifest=manifest,
                candidate_manifest=manifest,
                tensors=[tensor],
                device_name="cpu",
                action_chunk_size=2,
                top_k=2,
                max_records=0,
                min_margin=0.25,
                min_snr=1.0,
            )
            self.assertEqual(report["schema_id"], "cascadiav3.policy_candidate_probe.v2")
            self.assertEqual(report["action_surface"], "unfiltered_full_legal_menu")
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["valid_q_root_count"], 1)
            self.assertEqual(report["baseline"], report["candidate"])
            self.assertEqual(
                report["comparison"]["candidate_minus_baseline_best_coverage"]["mean"],
                0.0,
            )
            self.assertEqual(
                report["comparison"]["candidate_minus_baseline_candidate_oracle_regret"][
                    "mean"
                ],
                0.0,
            )
            filtered = root / "filtered.npz"
            filter_expert_tensor_shard(tensor, filtered, top_k=2)
            with self.assertRaisesRegex(ValueError, "unfiltered full-menu"):
                run_probe(
                    baseline_manifest=manifest,
                    candidate_manifest=manifest,
                    tensors=[filtered],
                    device_name="cpu",
                    action_chunk_size=2,
                    top_k=2,
                    max_records=0,
                    min_margin=0.25,
                    min_snr=1.0,
                )


if __name__ == "__main__":
    unittest.main()
