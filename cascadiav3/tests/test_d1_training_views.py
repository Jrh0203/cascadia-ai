"""D1 training views: mask plumbing, view builder, and masked losses.

Requires numpy + torch (runs on john0; skipped where they are absent),
matching the convention of the other tensor-path tests.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401

    _DEPS = True
except ModuleNotFoundError:
    _DEPS = False

from test_analyze_label_density import _v4_metadata, write_v4_fixture


def _fixture_arrays():
    """The four-record hand-computed v4 fixture as writer kwargs."""
    import numpy as np

    from test_analyze_label_density import _other_token_row, _player_token_row

    seats = [0, 1, 2, 3]
    tile_counts = [3, 4, 22, 22]
    tokens = []
    for seat, tile_count in zip(seats, tile_counts, strict=True):
        tokens.extend([_player_token_row(seat, tile_count), _other_token_row()])
    final_scores = np.asarray(
        [
            [20.0, 10.0, 55.0, 65.0],
            [20.0, 10.0, 55.0, 65.0],
            [12.0, 11.0, 30.0, 40.0],
            [12.0, 11.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )
    score_decomposition = np.zeros((4, 3, 4), dtype=np.float32)
    score_decomposition[:, 0, :] = final_scores
    exact_components = np.zeros((12, 3), dtype=np.float32)
    exact_components[:, 0] = 1.0
    return {
        "tokens": np.stack(tokens, axis=0),
        "actions": np.zeros((12, 61), dtype=np.float16),
        "token_offsets": np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        "action_offsets": np.asarray([0, 3, 6, 9, 12], dtype=np.int64),
        "relation_edges": np.zeros((0, 3), dtype=np.int32),
        "relation_offsets": np.asarray([0, 0, 0, 0, 0], dtype=np.int64),
        "selected_action_index": np.asarray([0, 0, 0, 1], dtype=np.int16),
        "target_q": np.asarray(
            [10.0, 9.6, 7.0, 20.0, 10.0, 5.0, 15.0, 14.0, 13.0, 8.0, 8.5, 3.0],
            dtype=np.float32,
        ),
        "target_score_to_go": np.asarray(
            [9.0, 8.6, 6.0, 19.0, 9.0, 4.0, 14.0, 13.0, 12.0, 7.0, 7.5, 2.0],
            dtype=np.float32,
        ),
        "q_valid": np.asarray([1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8),
        "priors": np.full((12,), 1.0 / 3.0, dtype=np.float32),
        "visits": np.asarray(
            [4.0, 2.0, 0.0, 4.0, 4.0, 2.0, 5.0, 0.0, 0.0, 2.0, 2.0, 0.0],
            dtype=np.float32,
        ),
        "q_variance": np.zeros((12,), dtype=np.float32),
        "q_count": np.ones((12,), dtype=np.float32),
        "truncated_count": np.zeros((12,), dtype=np.float32),
        "exact_afterstate_score_active": np.ones((12,), dtype=np.float32),
        "exact_afterstate_score_decomposition_active": exact_components,
        "active_seat": np.asarray([0, 1, 2, 3], dtype=np.uint8),
        "final_score_vector": final_scores,
        "rank_vector": np.asarray(
            [[3, 4, 2, 1], [3, 4, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]], dtype=np.int16
        ),
        "score_decomposition": score_decomposition,
        "improved_policy": np.asarray(
            [0.5, 0.3, 0.2, 0.7, 0.2, 0.1, 0.9, 0.06, 0.04, 0.4, 0.5, 0.1],
            dtype=np.float32,
        ),
        "search_root_value": np.asarray([22.0, 9.0, 30.0, 41.0], dtype=np.float32),
        "exact_endgame": np.zeros((4,), dtype=np.uint8),
    }


def write_fixture(path: Path, **overrides):
    from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard

    kwargs = _fixture_arrays()
    kwargs.update(overrides)
    metadata = kwargs.pop("metadata", _v4_metadata())
    _save_expert_tensor_shard(out_path=path, metadata=metadata, **kwargs)


@unittest.skipUnless(_DEPS, "requires numpy and torch")
class ValidityMaskPlumbingTest(unittest.TestCase):
    def test_shard_and_collate_roundtrip_masks(self):
        import numpy as np

        from cascadiav3.expert_tensor_shards import (
            ExpertTensorShard,
            collate_expert_tensor_examples,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "shard.npz"
            write_fixture(
                path,
                policy_valid=np.asarray([1, 0, 1, 1], dtype=np.uint8),
                outcome_valid=np.asarray([1, 1, 0, 1], dtype=np.uint8),
            )
            shard = ExpertTensorShard(path)
            examples = [shard.example(index) for index in range(4)]
            self.assertEqual(
                [example["policy_valid"] for example in examples],
                [True, False, True, True],
            )
            batch = collate_expert_tensor_examples(examples)
            self.assertEqual(batch["policy_valid"].tolist(), [True, False, True, True])
            self.assertEqual(batch["outcome_valid"].tolist(), [True, True, False, True])
            shard.close()

    def test_legacy_shard_has_no_mask_keys(self):
        from cascadiav3.expert_tensor_shards import (
            ExpertTensorShard,
            collate_expert_tensor_examples,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "legacy.npz"
            write_v4_fixture(path)
            shard = ExpertTensorShard(path)
            examples = [shard.example(index) for index in range(4)]
            self.assertNotIn("policy_valid", examples[0])
            batch = collate_expert_tensor_examples(examples)
            self.assertNotIn("policy_valid", batch)
            self.assertNotIn("outcome_valid", batch)
            shard.close()

    def test_d1_relabel_mode_metadata_is_training_eligible(self):
        from cascadiav3.expert_tensor_shards import ExpertTensorCorpus

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "d1.npz"
            metadata = _v4_metadata()
            metadata["mode"] = "puzzle_bank_d1_relabel"
            write_fixture(path, metadata=metadata)
            corpus = ExpertTensorCorpus([path])
            self.assertEqual(len(corpus), 4)
            corpus.close()


@unittest.skipUnless(_DEPS, "requires numpy and torch")
class MaskCarryThroughTest(unittest.TestCase):
    def test_filter_and_relation_tail_preserve_validity_masks(self):
        import numpy as np

        from cascadiav3.expert_tensor_shards import (
            ExpertTensorShard,
            filter_expert_tensor_shard,
            materialize_relation_tail_shard,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            raw = Path(tempdir) / "raw.npz"
            write_fixture(
                raw,
                policy_valid=np.asarray([1, 0, 1, 1], dtype=np.uint8),
                outcome_valid=np.asarray([0, 1, 1, 0], dtype=np.uint8),
            )
            filtered = Path(tempdir) / "filtered.npz"
            filter_expert_tensor_shard(raw, filtered, top_k=2)
            tail = Path(tempdir) / "tail.npz"
            materialize_relation_tail_shard(filtered, tail)
            shard = ExpertTensorShard(tail)
            self.assertEqual(
                [shard.example(index)["policy_valid"] for index in range(4)],
                [True, False, True, True],
            )
            self.assertEqual(
                [shard.example(index)["outcome_valid"] for index in range(4)],
                [False, True, True, False],
            )
            shard.close()


@unittest.skipUnless(_DEPS, "requires numpy and torch")
class MaskedLossTest(unittest.TestCase):
    def _batch(self, **mask_arrays):
        import numpy as np

        from cascadiav3.expert_tensor_shards import (
            ExpertTensorShard,
            collate_expert_tensor_examples,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "shard.npz"
            write_fixture(
                path,
                **{
                    key: np.asarray(value, dtype=np.uint8)
                    for key, value in mask_arrays.items()
                },
            )
            shard = ExpertTensorShard(path)
            batch = collate_expert_tensor_examples(
                [shard.example(index) for index in range(4)]
            )
            shard.close()
            return batch

    def _outputs(self, batch):
        import torch

        generator = torch.Generator().manual_seed(7)
        batch_size, max_actions = batch["target_q"].shape
        return {
            "logits": torch.randn(batch_size, max_actions, generator=generator),
            "q": torch.randn(batch_size, max_actions, generator=generator),
            "value_vector": torch.randn(batch_size, 4, generator=generator),
            "score_decomposition": torch.randn(batch_size, 3, 4, generator=generator),
            "rank_logits": torch.randn(batch_size, 4, 4, generator=generator),
            "uncertainty": torch.rand(batch_size, max_actions, generator=generator),
        }

    def _losses(self, batch):
        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        outputs = self._outputs(batch)
        return _loss_components(outputs, batch, LossWeights())

    @staticmethod
    def _scalar(losses, key):
        return float(losses[key].detach())

    def test_all_true_masks_match_maskless_losses(self):
        maskless = self._losses(self._batch())
        masked = self._losses(
            self._batch(policy_valid=[1, 1, 1, 1], outcome_valid=[1, 1, 1, 1])
        )
        for key in ("policy", "q", "value", "score", "rank"):
            self.assertEqual(self._scalar(maskless, key), self._scalar(masked, key))

    def test_policy_valid_false_removes_record_from_policy_loss(self):
        import torch

        full = self._batch()
        masked = self._batch(policy_valid=[1, 0, 1, 1])
        outputs = self._outputs(full)

        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        losses_masked = _loss_components(outputs, masked, LossWeights())
        # Reference: soft improved-policy CE recomputed by hand over the
        # three valid records.
        mask = full["action_mask"]
        logits = outputs["logits"].masked_fill(~mask, -1.0e9)
        log_policy = torch.log_softmax(logits, dim=1)
        target = full["improved_policy"].masked_fill(~mask, 0.0)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        per_record = -(target * log_policy).sum(dim=1)
        expected = per_record[[0, 2, 3]].mean()
        self.assertAlmostEqual(
            self._scalar(losses_masked, "policy"), float(expected), places=5
        )
        # Q/value/score/rank are untouched by policy_valid.
        losses_full = _loss_components(outputs, full, LossWeights())
        for key in ("q", "value", "score", "rank"):
            self.assertEqual(
                self._scalar(losses_full, key), self._scalar(losses_masked, key)
            )

    def test_outcome_valid_false_removes_record_from_outcome_losses(self):
        import torch
        import torch.nn.functional as F

        full = self._batch()
        masked = self._batch(outcome_valid=[1, 1, 0, 1])
        outputs = self._outputs(full)

        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        losses_masked = _loss_components(outputs, masked, LossWeights())
        keep = [0, 1, 3]
        expected_value = F.mse_loss(
            outputs["value_vector"][keep], full["target_value"][keep]
        )
        self.assertAlmostEqual(
            self._scalar(losses_masked, "value"), float(expected_value), places=5
        )
        expected_rank = F.cross_entropy(
            outputs["rank_logits"][keep].reshape(-1, 4),
            full["target_rank"][keep].reshape(-1),
        )
        self.assertAlmostEqual(
            self._scalar(losses_masked, "rank"), float(expected_rank), places=5
        )
        losses_full = _loss_components(outputs, full, LossWeights())
        for key in ("policy", "q"):
            self.assertEqual(
                self._scalar(losses_full, key), self._scalar(losses_masked, key)
            )


@unittest.skipUnless(_DEPS, "requires numpy and torch")
class ViewBuilderTest(unittest.TestCase):
    def _write_ledger(self, path: Path, roots):
        with path.open("w") as handle:
            for seed, ply in roots:
                handle.write(
                    json.dumps(
                        {
                            "type": "gumbel_decision",
                            "seed": seed,
                            "ply": ply,
                            "chosen_action_id": "a0",
                            "action_count": 3,
                        }
                    )
                    + "\n"
                )

    def _write_mask(self, path: Path, roots):
        with path.open("w") as handle:
            for seed, ply in roots:
                handle.write(json.dumps({"seed": seed, "ply": ply}) + "\n")

    def test_base_view_masks_exactly_the_tranche_roots(self):
        from cascadiav3 import build_d1_training_views as views
        from cascadiav3.expert_tensor_shards import ExpertTensorShard

        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir) / "base.npz"
            write_fixture(base)
            ledger = Path(tempdir) / "decisions.jsonl"
            # Record order: seed 5 plies 0,1 then seed 9 plies 0,1.
            self._write_ledger(ledger, [(5, 0), (5, 1), (9, 0), (9, 1)])
            mask = Path(tempdir) / "tranche.jsonl"
            self._write_mask(mask, [(5, 1), (9, 0)])
            out = Path(tempdir) / "base_view.npz"
            report = views.build_base_view(base, ledger, mask, out)
            self.assertEqual(report["masked_roots"], 2)
            shard = ExpertTensorShard(out)
            self.assertEqual(
                [shard.example(index)["policy_valid"] for index in range(4)],
                [True, False, False, True],
            )
            self.assertFalse(shard.example(1)["q_valid"].any())
            self.assertTrue(shard.example(0)["q_valid"].any())
            self.assertEqual(
                shard.metadata["view"]["type"], "d1_base_view_masked_stale_search"
            )
            # Outcomes stay valid in the base view.
            self.assertNotIn("outcome_valid", shard.example(1))
            shard.close()

    def test_base_view_rejects_root_missing_from_ledger(self):
        from cascadiav3 import build_d1_training_views as views

        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir) / "base.npz"
            write_fixture(base)
            ledger = Path(tempdir) / "decisions.jsonl"
            self._write_ledger(ledger, [(5, 0), (5, 1), (9, 0), (9, 1)])
            mask = Path(tempdir) / "tranche.jsonl"
            self._write_mask(mask, [(7, 3)])
            with self.assertRaisesRegex(ValueError, "absent from the ledger"):
                views.build_base_view(
                    base, ledger, mask, Path(tempdir) / "out.npz"
                )

    def test_base_view_rejects_record_count_mismatch(self):
        from cascadiav3 import build_d1_training_views as views

        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir) / "base.npz"
            write_fixture(base)
            ledger = Path(tempdir) / "decisions.jsonl"
            self._write_ledger(ledger, [(5, 0), (5, 1), (9, 0)])
            mask = Path(tempdir) / "tranche.jsonl"
            self._write_mask(mask, [(5, 1)])
            with self.assertRaisesRegex(ValueError, "!= shard records"):
                views.build_base_view(base, ledger, mask, Path(tempdir) / "out.npz")

    def test_d1_view_masks_outcomes_and_verifies_audit(self):
        from cascadiav3 import build_d1_training_views as views
        from cascadiav3.expert_tensor_shards import ExpertTensorShard

        with tempfile.TemporaryDirectory() as tempdir:
            d1 = Path(tempdir) / "d1.npz"
            metadata = _v4_metadata()
            metadata["mode"] = "puzzle_bank_d1_relabel"
            write_fixture(d1, metadata=metadata)
            audit = Path(tempdir) / "audit.jsonl"
            with audit.open("w") as handle:
                for seed, ply in [(5, 1), (5, 3), (9, 0), (9, 2)]:
                    handle.write(
                        json.dumps(
                            {"type": "d1_repeat_audit", "seed": seed, "ply": ply}
                        )
                        + "\n"
                    )
            mask = Path(tempdir) / "tranche.jsonl"
            self._write_mask(mask, [(5, 1), (5, 3), (9, 0), (9, 2), (11, 4)])
            out = Path(tempdir) / "d1_view.npz"
            report = views.build_d1_view(d1, audit, mask, out)
            self.assertEqual(report["records"], 4)
            self.assertAlmostEqual(report["mask_coverage"], 0.8)
            shard = ExpertTensorShard(out)
            self.assertEqual(
                [shard.example(index)["outcome_valid"] for index in range(4)],
                [False, False, False, False],
            )
            self.assertEqual(
                [shard.example(index)["policy_valid"] for index in range(4)],
                [True, True, True, True],
            )
            self.assertEqual(
                shard.metadata["view"]["type"], "d1_relabel_view_masked_outcomes"
            )
            shard.close()

    def test_dose_subsets_are_nested_and_deterministic(self):
        from cascadiav3 import build_d1_training_views as views
        from cascadiav3.expert_tensor_shards import ExpertTensorShard

        with tempfile.TemporaryDirectory() as tempdir:
            d1 = Path(tempdir) / "d1.npz"
            metadata = _v4_metadata()
            metadata["mode"] = "puzzle_bank_d1_relabel"
            write_fixture(d1, metadata=metadata)
            audit = Path(tempdir) / "audit.jsonl"
            roots = [(5, 1), (5, 3), (9, 0), (9, 2)]
            with audit.open("w") as handle:
                for seed, ply in roots:
                    handle.write(
                        json.dumps(
                            {"type": "d1_repeat_audit", "seed": seed, "ply": ply}
                        )
                        + "\n"
                    )
            two = Path(tempdir) / "d1_2.npz"
            three = Path(tempdir) / "d1_3.npz"
            views.subset_view_by_hash_order(d1, audit, 2, two)
            views.subset_view_by_hash_order(d1, audit, 3, three)
            shard_two = ExpertTensorShard(two)
            shard_three = ExpertTensorShard(three)
            self.assertEqual(len(shard_two), 2)
            self.assertEqual(len(shard_three), 3)
            # Nesting: the 2-subset's records appear in the 3-subset. Compare
            # per-record fingerprints (final_score_vector rows + active seat).
            def fingerprints(shard):
                return {
                    (
                        int(shard.example(index)["active_seat"]),
                        tuple(float(x) for x in shard.example(index)["final_score_vector"]),
                        tuple(float(x) for x in shard.example(index)["target_q"]),
                    )
                    for index in range(len(shard))
                }

            self.assertTrue(fingerprints(shard_two) <= fingerprints(shard_three))
            shard_two.close()
            shard_three.close()

    def test_d1_view_rejects_roots_outside_the_mask(self):
        from cascadiav3 import build_d1_training_views as views

        with tempfile.TemporaryDirectory() as tempdir:
            d1 = Path(tempdir) / "d1.npz"
            metadata = _v4_metadata()
            metadata["mode"] = "puzzle_bank_d1_relabel"
            write_fixture(d1, metadata=metadata)
            audit = Path(tempdir) / "audit.jsonl"
            with audit.open("w") as handle:
                handle.write(
                    json.dumps({"type": "d1_repeat_audit", "seed": 99, "ply": 0}) + "\n"
                )
            mask = Path(tempdir) / "tranche.jsonl"
            self._write_mask(mask, [(5, 1)])
            with self.assertRaisesRegex(ValueError, "outside the tranche mask"):
                views.build_d1_view(d1, audit, mask, Path(tempdir) / "out.npz")


if __name__ == "__main__":
    unittest.main()
