from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest


class PolicyCandidateProbeTest(unittest.TestCase):
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
