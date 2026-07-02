from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from cascadiav3.fixtures import tiny_replay_manifest, tiny_replay_records, tiny_search_root_record
from cascadiav3.hex import RADIUS6_CELL_COUNT, RADIUS6_COORDS, cell_index, coord_for_index, coord_ref
from cascadiav3.model_smoke import mock_forward, validate_mock_output
from cascadiav3.replay import (
    read_replay_jsonl,
    replay_manifest_for_records,
    write_replay_jsonl,
)
from cascadiav3.schema import (
    EXPERT_ROOT_SCHEMA_ID,
    EXPERT_TENSOR_SHARD_SCHEMA_ID,
    GREEDY_TENSOR_SHARD_SCHEMA_ID,
    PRE_GPU_SCHEMA_ID,
    SchemaError,
    registry_report,
    validate_replay_manifest,
    validate_search_root_record,
)
from cascadiav3.validate import run_validation


class HexContractTest(unittest.TestCase):
    def test_radius6_has_127_stable_cells(self) -> None:
        self.assertEqual(RADIUS6_CELL_COUNT, 127)
        self.assertEqual(len(RADIUS6_COORDS), 127)
        self.assertEqual(
            sorted(cell_index(coord.q, coord.r) for coord in RADIUS6_COORDS),
            list(range(127)),
        )
        for index in range(127):
            coord = coord_for_index(index)
            self.assertEqual(cell_index(coord.q, coord.r), index)

    def test_overflow_is_exact_and_requires_identity(self) -> None:
        overflow = coord_ref(7, 0, owner_seat=2, placement_id=42)
        self.assertEqual(overflow["kind"], "overflow")
        self.assertEqual(overflow["q"], 7)
        self.assertEqual(overflow["r"], 0)
        self.assertEqual(overflow["s"], -7)
        self.assertFalse(overflow["radius6_member"])
        with self.assertRaises(ValueError):
            coord_ref(7, 0)


class SchemaContractTest(unittest.TestCase):
    def test_schema_registry_keeps_legacy_and_expert_contracts(self) -> None:
        report = registry_report(include_legacy=True, include_expert=True)
        self.assertEqual(report["status"], "pass")
        schema_ids = {schema["schema_id"] for schema in report["schemas"]}
        self.assertIn(PRE_GPU_SCHEMA_ID, schema_ids)
        self.assertIn(GREEDY_TENSOR_SHARD_SCHEMA_ID, schema_ids)
        self.assertIn(EXPERT_ROOT_SCHEMA_ID, schema_ids)
        self.assertIn(EXPERT_TENSOR_SHARD_SCHEMA_ID, schema_ids)

    def test_tiny_search_root_validates(self) -> None:
        root = tiny_search_root_record()
        validate_search_root_record(root)
        validate_replay_manifest(tiny_replay_manifest(root))

    def test_action_arrays_must_align(self) -> None:
        root = tiny_search_root_record()
        broken = copy.deepcopy(root)
        broken["visits"] = [1]
        with self.assertRaises(SchemaError):
            validate_search_root_record(broken)


class BridgeContractTest(unittest.TestCase):
    def test_manifest_resolution_accepts_trainer_project_relative_paths(self) -> None:
        from cascadiav3.torch_inference_bridge import resolve_checkpoint_path

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            manifest_dir = project_root / "cascadiav3" / "checkpoints" / "cascadiaformer"
            manifest_dir.mkdir(parents=True)
            project_weight = manifest_dir / "step_0000001.weights.pt"
            project_weight.write_text("weights", encoding="utf-8")
            manifest_path = manifest_dir / "step_0000001.manifest.json"
            resolved = resolve_checkpoint_path(
                "cascadiav3/checkpoints/cascadiaformer/step_0000001.weights.pt",
                manifest_path=manifest_path,
                cwd=project_root,
            )
            self.assertEqual(resolved, project_weight)

            relative_weight = manifest_dir / "relative.weights.pt"
            relative_weight.write_text("weights", encoding="utf-8")
            resolved_relative = resolve_checkpoint_path(
                "relative.weights.pt",
                manifest_path=manifest_path,
                cwd=project_root / "different-cwd",
            )
            self.assertEqual(resolved_relative, relative_weight)

    def test_inference_request_view_accepts_public_eval_shape_without_labels(self) -> None:
        from cascadiav3.torch_inference_bridge import TRAINING_LABEL_KEYS, collate_inference_roots, inference_request_view

        path = Path("cascadiav3/fixtures/expert_tiny.jsonl")
        if not path.exists():
            self.skipTest("expert tiny roots have not been generated")
        root = read_replay_jsonl(path)[0]
        public_root = {key: value for key, value in root.items() if key not in TRAINING_LABEL_KEYS}
        view = inference_request_view(public_root)
        self.assertEqual(view["training_labels_present"], [])
        self.assertEqual(len(view["action_ids"]), len(public_root["legal_actions"]))
        self.assertNotIn("per_action_Q", public_root)
        try:
            batch = collate_inference_roots([public_root])
        except ModuleNotFoundError as exc:
            self.assertIn("torch", str(exc))
            return
        self.assertEqual(batch["action_mask"].shape[1], len(public_root["legal_actions"]))
        self.assertEqual(batch["action_ids"][0], view["action_ids"])


class TrainerCursorContractTest(unittest.TestCase):
    def test_loader_cursor_points_to_next_unconsumed_microbatch(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import (
            _batch_indices_for_global_batch,
            _loader_cursor_for_next_batch,
        )

        indices, consumed_cursor = _batch_indices_for_global_batch(
            global_batch=1,
            batch_size=2,
            record_count=10,
            seed=7,
            shuffle=False,
        )
        self.assertEqual(indices, [0, 1])
        self.assertEqual(consumed_cursor["next_global_batch"], 2)
        self.assertEqual(consumed_cursor["position"], 2)

        cursor = _loader_cursor_for_next_batch(
            next_global_batch=2,
            batch_size=2,
            record_count=10,
            seed=7,
            shuffle=False,
            overfit_one_batch=False,
        )
        self.assertEqual(cursor["next_global_batch"], 2)
        self.assertEqual(cursor["last_consumed_global_batch"], 1)
        self.assertEqual(cursor["position"], 2)
        self.assertNotEqual(cursor["next_global_batch"], 3)
        self.assertNotEqual(cursor["position"], 4)

    def test_weighted_source_sampler_is_deterministic_and_records_contract(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import (
            _loader_cursor_for_next_weighted_batch,
            _weighted_batch_indices_for_global_batch,
        )

        source_lengths = [10, 20, 30]
        source_weights = [0.5, 0.3, 0.2]
        indices_a, cursor_a = _weighted_batch_indices_for_global_batch(
            global_batch=7,
            batch_size=64,
            source_lengths=source_lengths,
            source_weights=source_weights,
            seed=20260701,
        )
        indices_b, cursor_b = _weighted_batch_indices_for_global_batch(
            global_batch=7,
            batch_size=64,
            source_lengths=source_lengths,
            source_weights=source_weights,
            seed=20260701,
        )

        self.assertEqual(indices_a, indices_b)
        self.assertEqual(cursor_a, cursor_b)
        self.assertTrue(all(0 <= index < sum(source_lengths) for index in indices_a))
        self.assertEqual(sum(cursor_a["source_counts"]), 64)
        self.assertEqual(cursor_a["source_weights"], source_weights)
        self.assertEqual(cursor_a["resume_semantics"], "deterministic_weighted_source_sampling_with_replacement")

        cursor = _loader_cursor_for_next_weighted_batch(
            next_global_batch=8,
            batch_size=64,
            source_lengths=source_lengths,
            source_weights=source_weights,
            seed=20260701,
            overfit_one_batch=False,
        )
        self.assertEqual(cursor["last_consumed_global_batch"], 7)
        self.assertEqual(cursor["source_lengths"], source_lengths)
        self.assertEqual(cursor["source_weights"], source_weights)


class CascadiaFormerBenchmarkContractTest(unittest.TestCase):
    def test_game_benchmark_contract_helpers(self) -> None:
        import inspect

        from cascadiav3.torch_cascadiaformer_game_benchmark import completed_game_result_row
        from cascadiav3.torch_cascadiaformer_game_benchmark import run_benchmark
        from cascadiav3.torch_cascadiaformer_game_benchmark import parse_seeds, summarize_game_results

        self.assertIn("treatment_workers", inspect.signature(run_benchmark).parameters)
        self.assertIn("game_results_path", inspect.signature(run_benchmark).parameters)
        self.assertEqual(parse_seeds(seeds="7, 9", first_seed=1, games=3), [7, 9])
        self.assertEqual(parse_seeds(seeds="", first_seed=10, games=3), [10, 11, 12])
        completed = completed_game_result_row(
            {
                "seed": 10,
                "strategy": "cascadiaformer",
                "selection_head": "q",
                "done": {
                    "scores": [{"total": 80}, {"total": 90}, {"total": 100}, {"total": 110}],
                    "turns": 80,
                    "elapsed_seconds": 12.5,
                    "final_state_hash": "state",
                },
                "decisions": [{}, {}],
            }
        )
        self.assertEqual(completed["mean_score_per_seat"], 95)
        self.assertEqual(completed["decision_count"], 2)
        self.assertEqual(completed["seat_scores"], [80.0, 90.0, 100.0, 110.0])
        summary = summarize_game_results(
            [
                {
                    "done": {
                        "scores": [{"total": 80}, {"total": 90}, {"total": 100}, {"total": 110}],
                    },
                    "decisions": [
                        {
                            "model_score_seconds": 0.25,
                            "model_matches_greedy_top": False,
                            "greedy_rank_in_model": 3,
                        }
                    ],
                }
            ]
        )
        self.assertEqual(summary["games"], 1)
        self.assertEqual(summary["decisions"], 1)
        self.assertEqual(summary["mean_seat_score"], 95)
        self.assertEqual(summary["action_match_rate_vs_greedy_top"], 0.0)
        self.assertEqual(summary["mean_greedy_rank_in_model"], 3)

    def test_search_benchmark_reports_gate_timing_fields(self) -> None:
        import inspect

        from cascadiav3.torch_cascadiaformer_search_benchmark import (
            paired_score_deltas,
            run_search_benchmark,
            summarize_game_results,
        )
        from cascadiav3.validate_runbook_performance import validate_time_ratio

        self.assertIn("candidate_workers", inspect.signature(run_search_benchmark).parameters)
        self.assertIn("game_results_path", inspect.signature(run_search_benchmark).parameters)
        candidate = [
            {
                "seed": 1,
                "selection_head": "q",
                "done": {"scores": [{"total": 100}, {"total": 96}]},
                "decisions": [
                    {
                        "model_score_seconds": 0.1,
                        "decision_seconds": 0.9,
                        "candidate_count": 64,
                        "retained_count": 32,
                        "full_best_retained": True,
                        "search_regret": 0.0,
                    }
                ],
            }
        ]
        control = [
            {
                "seed": 1,
                "selection_head": "full-search",
                "done": {"scores": [{"total": 98}, {"total": 94}]},
                "decisions": [
                    {
                        "model_score_seconds": 0.0,
                        "decision_seconds": 1.0,
                        "candidate_count": 64,
                        "retained_count": 64,
                    }
                ],
            }
        ]
        candidate_summary = summarize_game_results(candidate)
        control_summary = summarize_game_results(control)
        report = {
            "treatment_mean_decision_seconds": candidate_summary["mean_total_decision_seconds"],
            "control_mean_decision_seconds": control_summary["mean_total_decision_seconds"],
        }
        self.assertEqual(candidate_summary["shadow_full_best_retained_rate"], 1.0)
        self.assertEqual(candidate_summary["estimated_non_shadow_rollout_fraction"], 0.5)
        self.assertEqual(validate_time_ratio(report, 1.20), 1.0)
        self.assertEqual(paired_score_deltas(candidate, control)[0]["delta_candidate_minus_full_search"], 2.0)

    def test_search_decision_trace_analyzer_reports_retention_by_k(self) -> None:
        from cascadiav3.analyze_search_decision_trace import build_report

        rows = [
            {
                "strategy": "cascadiaformer-search",
                "selection_head": "q",
                "seed_u64": 1,
                "ply_index": 0,
                "active_seat": 0,
                "candidate_count": 4,
                "retained_count": 2,
                "model_ranked_action_ids": ["a", "b", "c", "d"],
                "full_best_action_id": "b",
                "search_regret": 0.0,
                "selected_active_score": 90.0,
                "full_best_active_score": 90.0,
            },
            {
                "strategy": "cascadiaformer-search",
                "selection_head": "q",
                "seed_u64": 1,
                "ply_index": 40,
                "active_seat": 1,
                "candidate_count": 4,
                "retained_count": 2,
                "model_ranked_action_ids": ["a", "b", "c", "d"],
                "full_best_action_id": "d",
                "search_regret": 2.5,
                "selected_active_score": 88.0,
                "full_best_active_score": 90.5,
            },
            {
                "strategy": "full-search",
                "selection_head": "full-search",
                "seed_u64": 1,
                "ply_index": 0,
                "candidate_count": 4,
                "retained_count": 4,
            },
        ]
        report = build_report(
            rows,
            source_path="synthetic.jsonl",
            k_values=[1, 2, 4],
            target_recall=1.0,
            miss_example_k=2,
            miss_example_limit=10,
        )
        self.assertEqual(report["candidate_rows"], 2)
        self.assertEqual(report["retention_by_k"]["1"]["full_best_retained_rate"], 0.0)
        self.assertEqual(report["retention_by_k"]["2"]["full_best_retained_rate"], 0.5)
        self.assertEqual(report["retention_by_k"]["4"]["full_best_retained_rate"], 1.0)
        self.assertEqual(report["recommended_min_k_for_target_recall"], 4)
        self.assertEqual(report["phase_summary"]["opening"]["retention_by_k"]["2"]["full_best_retained_rate"], 1.0)
        self.assertEqual(report["phase_summary"]["late_mid"]["retention_by_k"]["2"]["full_best_retained_rate"], 0.0)
        self.assertEqual(report["largest_k_misses"][0]["full_best_model_rank"], 4)


class ReplayContractTest(unittest.TestCase):
    def test_tiny_replay_records_have_variable_action_counts(self) -> None:
        records = tiny_replay_records()
        self.assertEqual([len(record["legal_actions"]) for record in records], [2, 3])
        for record in records:
            validate_search_root_record(record)

    def test_replay_jsonl_roundtrip_and_manifest(self) -> None:
        records = tiny_replay_records()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny_replay.jsonl"
            write_replay_jsonl(path, records)
            roundtrip = read_replay_jsonl(path)
        self.assertEqual([record["state_hash"] for record in roundtrip], ["tiny-state-0001", "tiny-state-0002"])
        manifest = replay_manifest_for_records(
            roundtrip,
            source_generator="test",
            seed_domain="fixed-test-seed",
        )
        validate_replay_manifest(manifest)
        self.assertEqual(manifest["record_count"], 2)

    def test_torch_collate_pads_actions_and_emits_mask(self) -> None:
        from cascadiav3.torch_replay import collate_search_roots

        try:
            batch = collate_search_roots(tiny_replay_records())
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(list(batch["actions"].shape), [2, 3, 16])
        self.assertEqual(list(batch["action_mask"].shape), [2, 3])
        self.assertEqual(batch["action_mask"].tolist(), [[True, True, False], [True, True, True]])
        self.assertEqual(list(batch["target_q"].shape), [2, 3])

    def test_expert_tensor_collate_pads_mixed_relation_tail_capacities(self) -> None:
        try:
            import numpy as np

            from cascadiav3.expert_tensor_shards import collate_expert_tensor_examples
        except ModuleNotFoundError as exc:
            self.skipTest(f"numeric stack not installed: {exc}")

        def example(token_count: int, action_count: int, token_capacity: int, token_value: int, action_value: int):
            relation_tail = np.zeros((action_count, token_capacity + action_count), dtype=np.uint8)
            relation_tail[:, :token_capacity] = token_value
            relation_tail[:, token_capacity : token_capacity + action_count] = action_value
            return {
                "tokens": np.zeros((token_count, 41), dtype=np.float32),
                "actions": np.zeros((action_count, 61), dtype=np.float32),
                "relation_edges": np.zeros((0, 3), dtype=np.int64),
                "selected_action_index": 0,
                "target_q": np.zeros((action_count,), dtype=np.float32),
                "target_score_to_go": np.zeros((action_count,), dtype=np.float32),
                "q_valid": np.ones((action_count,), dtype=np.bool_),
                "priors": np.zeros((action_count,), dtype=np.float32),
                "visits": np.ones((action_count,), dtype=np.float32),
                "q_variance": np.zeros((action_count,), dtype=np.float32),
                "q_count": np.ones((action_count,), dtype=np.float32),
                "truncated_count": np.zeros((action_count,), dtype=np.float32),
                "exact_afterstate_score_active": np.zeros((action_count,), dtype=np.float32),
                "final_score_vector": np.zeros((4,), dtype=np.float32),
                "rank_vector": np.ones((4,), dtype=np.int64),
                "score_decomposition": np.zeros((3, 4), dtype=np.float32),
                "relation_tail": relation_tail,
            }

        batch = collate_expert_tensor_examples(
            [
                example(token_count=2, action_count=2, token_capacity=3, token_value=1, action_value=2),
                example(token_count=3, action_count=3, token_capacity=4, token_value=3, action_value=4),
            ]
        )
        tail = batch["relation_tail"]
        self.assertEqual(list(tail.shape), [2, 3, 7])
        self.assertTrue((tail[0, :2, :3] == 1).all())
        self.assertTrue((tail[0, :2, 3] == 0).all())
        self.assertTrue((tail[0, :2, 4:6] == 2).all())
        self.assertTrue((tail[1, :3, :4] == 3).all())
        self.assertTrue((tail[1, :3, 4:7] == 4).all())

    def test_real_roots_artifact_validates_when_present(self) -> None:
        path = Path("cascadiav3/fixtures/real_roots.jsonl")
        manifest_path = Path("cascadiav3/fixtures/real_roots_manifest.json")
        if not path.exists() or not manifest_path.exists():
            self.skipTest("real simulator roots have not been generated")
        records = read_replay_jsonl(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_replay_manifest(manifest)
        self.assertEqual(manifest["record_count"], len(records))
        self.assertTrue(all(record["metadata"]["source"].startswith("canonical_simulator") for record in records))

    def test_expert_tiny_artifact_validates_when_present(self) -> None:
        path = Path("cascadiav3/fixtures/expert_tiny.jsonl")
        manifest_path = Path("cascadiav3/fixtures/expert_tiny_manifest.json")
        if not path.exists() or not manifest_path.exists():
            self.skipTest("expert tiny roots have not been generated")
        records = read_replay_jsonl(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_replay_manifest(manifest)
        self.assertEqual(manifest["schema_id"], EXPERT_ROOT_SCHEMA_ID)
        self.assertEqual(manifest["record_count"], len(records))
        self.assertTrue(all(record["schema_id"] == EXPERT_ROOT_SCHEMA_ID for record in records))
        self.assertTrue(all(record["metadata"]["legal_action_coverage"] == 1.0 for record in records))

    def test_expert_tensor_shard_when_present(self) -> None:
        path = Path("cascadiav3/fixtures/expert_tiny_tensor.npz")
        manifest_path = Path("cascadiav3/fixtures/expert_tiny_tensor_manifest.json")
        if not path.exists() or not manifest_path.exists():
            self.skipTest("expert tensor shard has not been generated")
        try:
            from cascadiav3.expert_tensor_shards import (
                ExpertTensorCorpus,
                collate_expert_tensor_examples,
                summarize_expert_tensor_shard,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"numeric stack not installed: {exc}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = summarize_expert_tensor_shard(path).to_dict()
        self.assertEqual(manifest["schema_id"], EXPERT_TENSOR_SHARD_SCHEMA_ID)
        self.assertEqual(summary["version"], EXPERT_TENSOR_SHARD_SCHEMA_ID)
        self.assertEqual(summary["record_count"], manifest["record_count"])
        corpus = ExpertTensorCorpus([path])
        try:
            examples = corpus.examples([0, min(1, len(corpus) - 1)])
            batch = collate_expert_tensor_examples(examples)
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        finally:
            corpus.close()
        self.assertEqual(batch["tokens"].shape[0], len(examples))
        self.assertEqual(batch["actions"].shape[0], len(examples))
        self.assertEqual(batch["target_q"].shape, batch["action_mask"].shape)
        self.assertEqual(batch["q_valid"].shape, batch["action_mask"].shape)
        self.assertEqual(batch["relation_ids"].shape[0], len(examples))

    def test_expert_tensor_topk_filter_when_present(self) -> None:
        path = Path("cascadiav3/fixtures/expert_tiny_tensor.npz")
        if not path.exists():
            self.skipTest("expert tensor shard has not been generated")
        try:
            from cascadiav3.expert_tensor_shards import (
                ExpertTensorCorpus,
                collate_expert_tensor_examples,
                filter_expert_tensor_shard,
                summarize_expert_tensor_shard,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"numeric stack not installed: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "expert_tiny_tensor_top16.npz"
            report = filter_expert_tensor_shard(path, out, top_k=16)
            summary = summarize_expert_tensor_shard(out).to_dict()
            self.assertEqual(report["version"], EXPERT_TENSOR_SHARD_SCHEMA_ID)
            self.assertLessEqual(summary["max_action_count"], 16)
            corpus = ExpertTensorCorpus([out])
            try:
                examples = corpus.examples([0, min(1, len(corpus) - 1)])
                self.assertTrue(all(0 <= example["selected_action_index"] < example["actions"].shape[0] for example in examples))
                batch = collate_expert_tensor_examples(examples)
            except ModuleNotFoundError as exc:
                self.skipTest(f"torch not installed: {exc}")
            finally:
                corpus.close()
            self.assertLessEqual(batch["actions"].shape[1], 16)
            self.assertEqual(batch["target_q"].shape, batch["action_mask"].shape)

    def test_expert_tensor_relation_tail_materialization_when_present(self) -> None:
        path = Path("cascadiav3/fixtures/expert_tiny_tensor.npz")
        if not path.exists():
            self.skipTest("expert tensor shard has not been generated")
        try:
            from cascadiav3.expert_tensor_shards import (
                ExpertTensorCorpus,
                collate_expert_tensor_examples,
                filter_expert_tensor_shard,
                materialize_relation_tail_shard,
                summarize_expert_tensor_shard,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"numeric stack not installed: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            filtered = Path(tmp) / "expert_tiny_tensor_top16.npz"
            materialized = Path(tmp) / "expert_tiny_tensor_top16_tail.npz"
            filter_expert_tensor_shard(path, filtered, top_k=16)
            report = materialize_relation_tail_shard(filtered, materialized)
            summary = summarize_expert_tensor_shard(materialized).to_dict()
            self.assertEqual(report["version"], EXPERT_TENSOR_SHARD_SCHEMA_ID)
            self.assertTrue(summary["relation_tail_present"])
            self.assertEqual(summary["relation_tail_dtype"], "uint8")

            sparse_corpus = ExpertTensorCorpus([filtered])
            tail_corpus = ExpertTensorCorpus([materialized])
            try:
                indices = [0, min(1, len(tail_corpus) - 1)]
                sparse_batch = collate_expert_tensor_examples(sparse_corpus.examples(indices))
                tail_batch = collate_expert_tensor_examples(tail_corpus.examples(indices))
            except ModuleNotFoundError as exc:
                self.skipTest(f"torch not installed: {exc}")
            finally:
                sparse_corpus.close()
                tail_corpus.close()
            self.assertIn("relation_ids", sparse_batch)
            self.assertIn("relation_tail", tail_batch)
            self.assertNotIn("relation_ids", tail_batch)
            token_capacity = tail_batch["tokens"].shape[1]
            action_capacity = tail_batch["actions"].shape[1]
            sparse_token_capacity = sparse_batch["tokens"].shape[1]
            sparse_action_capacity = sparse_batch["actions"].shape[1]
            sparse_tail = sparse_batch["relation_ids"][
                :,
                sparse_token_capacity : sparse_token_capacity + sparse_action_capacity,
                :,
            ]
            expected = sparse_tail.new_zeros(tail_batch["relation_tail"].shape)
            expected[:, :sparse_action_capacity, :sparse_token_capacity] = sparse_tail[
                :,
                :,
                :sparse_token_capacity,
            ]
            expected[
                :,
                :sparse_action_capacity,
                token_capacity : token_capacity + sparse_action_capacity,
            ] = sparse_tail[
                :,
                :,
                sparse_token_capacity : sparse_token_capacity + sparse_action_capacity,
            ]
            self.assertEqual(tail_batch["relation_tail"].shape[1], action_capacity)
            self.assertEqual(tail_batch["relation_tail"].shape[2], token_capacity + action_capacity)
            self.assertTrue((tail_batch["relation_tail"].to(expected.dtype) == expected).all())

    def test_merit_feature_contract_when_real_roots_present(self) -> None:
        from cascadiav3.torch_action_query_merit import (
            MERIT_ACTION_FEATURE_DIM,
            MERIT_STATE_FEATURE_DIM,
            baseline_metrics_for_records,
            merit_action_features,
            merit_state_features,
        )

        path = Path("cascadiav3/fixtures/real_roots.jsonl")
        if not path.exists():
            self.skipTest("real simulator roots have not been generated")
        records = read_replay_jsonl(path)
        self.assertEqual(len(merit_state_features(records[0])), MERIT_STATE_FEATURE_DIM)
        action_rows = merit_action_features(records[0])
        self.assertEqual(len(action_rows), len(records[0]["legal_actions"]))
        self.assertTrue(all(len(row) == MERIT_ACTION_FEATURE_DIM for row in action_rows))
        without_target = dict(records[0])
        del without_target["final_score_vector"]
        self.assertEqual(len(merit_state_features(without_target)), MERIT_STATE_FEATURE_DIM)
        try:
            metrics = baseline_metrics_for_records(records[:2])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertIn("immediate_base", metrics)
        self.assertEqual(metrics["immediate_base"]["roots"], 2)

    def test_public_token_feature_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_public_token_merit import (
            PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            PUBLIC_TOKEN_FEATURE_DIM,
            public_token_action_features,
            public_token_features,
        )

        path = Path("cascadiav3/fixtures/crt_token_merit_train.jsonl")
        fallback = Path("cascadiav3/fixtures/real_roots.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("enriched simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        token_rows = public_token_features(records[0])
        action_rows = public_token_action_features(records[0])
        self.assertEqual(len(token_rows), records[0]["public_tokens"]["token_count"])
        self.assertTrue(all(len(row) == PUBLIC_TOKEN_FEATURE_DIM for row in token_rows))
        self.assertEqual(len(action_rows), len(records[0]["legal_actions"]))
        self.assertTrue(all(len(row) == PUBLIC_TOKEN_ACTION_FEATURE_DIM for row in action_rows))
        relation_kinds = {rel["relation_kind"] for rel in records[0]["public_tokens"]["relations"]}
        self.assertIn("same_market_slot", relation_kinds)

    def test_relation_bias_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_relation_bias_merit import (
            RELATION_KINDS,
            RELATION_TO_ID,
            collate_relation_bias_roots,
            combined_relation_ids,
            relation_counts,
        )

        path = Path("cascadiav3/fixtures/crt_token_merit_train.jsonl")
        if not path.exists():
            self.skipTest("enriched simulator roots have not been generated")
        records = read_replay_jsonl(path)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        matrix = combined_relation_ids(records[0])
        seq_len = records[0]["public_tokens"]["token_count"] + len(records[0]["legal_actions"])
        self.assertEqual(len(matrix), seq_len)
        self.assertTrue(all(len(row) == seq_len for row in matrix))
        counts = relation_counts(matrix)
        self.assertGreater(counts.get("same_owner_board", 0), 0)
        self.assertGreater(counts.get("action_uses_tile_slot", 0), 0)
        self.assertGreater(counts.get("action_uses_wildlife_slot", 0), 0)
        self.assertGreater(counts.get("action_targets_tile_frontier", 0), 0)
        self.assertIn("action_targets_wildlife_cell", RELATION_KINDS)
        self.assertEqual(RELATION_TO_ID["none"], 0)
        try:
            batch = collate_relation_bias_roots(records[:2])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(batch["relation_ids"].shape[0], 2)
        self.assertEqual(
            batch["relation_ids"].shape[1],
            batch["tokens"].shape[1] + batch["actions"].shape[1],
        )
        self.assertEqual(
            batch["action_ids"][0],
            [action["action_id"] for action in records[0]["legal_actions"]],
        )

    def test_semantic_action_feature_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_semantic_relation_bias_merit import (
            SEMANTIC_ACTION_FEATURE_DIM,
            SEMANTIC_ACTION_FEATURE_NAMES,
            SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
            collate_semantic_relation_bias_roots,
            semantic_action_features,
            semantic_public_token_action_features,
        )

        path = Path("cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
        fallback = Path("cascadiav3/fixtures/crt_token_merit_train.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("enriched simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        semantic_rows = semantic_action_features(records[0])
        combined_rows = semantic_public_token_action_features(records[0])
        self.assertEqual(len(SEMANTIC_ACTION_FEATURE_NAMES), SEMANTIC_ACTION_FEATURE_DIM)
        self.assertEqual(len(semantic_rows), len(records[0]["legal_actions"]))
        self.assertTrue(all(len(row) == SEMANTIC_ACTION_FEATURE_DIM for row in semantic_rows))
        self.assertTrue(all(len(row) == SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM for row in combined_rows))
        self.assertTrue(any(any(abs(value) > 0.0 for value in row) for row in semantic_rows))
        try:
            batch = collate_semantic_relation_bias_roots(records[:2])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(batch["actions"].shape[2], SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM)
        self.assertEqual(
            batch["relation_ids"].shape[1],
            batch["tokens"].shape[1] + batch["actions"].shape[1],
        )

    def test_greedy_tensor_shard_roundtrip_when_corpus_present(self) -> None:
        path = Path("cascadiav3/fixtures/greedy_policy_corpus_tiny.jsonl")
        if not path.exists():
            self.skipTest("greedy policy corpus has not been generated")
        try:
            from cascadiav3.greedy_tensor_shards import summarize_tensor_shard, write_tensor_shard_from_jsonl
            from cascadiav3.torch_greedy_policy_pretrain import (
                GreedyTensorShardIterableDataset,
                collate_greedy_tensor_examples,
                count_records,
            )
            from cascadiav3.torch_semantic_relation_bias_merit import SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
        except ModuleNotFoundError as exc:
            self.skipTest(f"numeric stack not installed: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            shard_path = Path(tmp) / "greedy_policy_tiny.npz"
            try:
                report = write_tensor_shard_from_jsonl([path], shard_path, dtype_name="float16")
            except ModuleNotFoundError as exc:
                self.skipTest(f"numeric stack not installed: {exc}")
            summary = summarize_tensor_shard(shard_path).to_dict()
            self.assertEqual(report["record_count"], summary["record_count"])
            self.assertEqual(count_records([shard_path], corpus_format="npz"), summary["record_count"])
            self.assertLess(summary["output_bytes"], path.stat().st_size)
            examples = []
            for example in GreedyTensorShardIterableDataset([shard_path], shuffle_buffer=1, seed=7):
                examples.append(example)
                if len(examples) == 2:
                    break
            batch = collate_greedy_tensor_examples(examples)
            self.assertEqual(batch["tokens"].shape[0], 2)
            self.assertEqual(batch["actions"].shape[2], SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM)
            self.assertEqual(batch["selected_action_index"].shape[0], 2)

    def test_semantic_cross_attention_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_semantic_cross_attention_merit import (
            SemanticCrossAttentionConfig,
            build_semantic_cross_attention_transformer,
        )
        from cascadiav3.torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots

        path = Path("cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
        fallback = Path("cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("semantic simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        try:
            batch = collate_semantic_relation_bias_roots(records[:2])
            model = build_semantic_cross_attention_transformer(
                SemanticCrossAttentionConfig(hidden_dim=32, layers=1, heads=4, mlp_dim=64)
            )
            output = model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(output["q"].shape, batch["action_mask"].shape)
        self.assertEqual(output["logits"].shape, batch["action_mask"].shape)

    def test_semantic_residual_attention_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots
        from cascadiav3.torch_semantic_residual_attention_merit import (
            SemanticResidualAttentionConfig,
            build_semantic_residual_attention_transformer,
        )

        path = Path("cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
        fallback = Path("cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("semantic simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        try:
            batch = collate_semantic_relation_bias_roots(records[:2])
            model = build_semantic_residual_attention_transformer(
                SemanticResidualAttentionConfig(hidden_dim=32, layers=1, heads=4, mlp_dim=64, residual_scale=0.25)
            )
            output = model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(output["q"].shape, batch["action_mask"].shape)
        self.assertEqual(output["logits"].shape, batch["action_mask"].shape)

    def test_semantic_vanilla_public_token_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_public_token_merit import build_public_token_transformer
        from cascadiav3.torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots
        from cascadiav3.torch_semantic_vanilla_public_token_merit import SemanticVanillaPublicTokenConfig

        path = Path("cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
        fallback = Path("cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("semantic simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        try:
            batch = collate_semantic_relation_bias_roots(records[:2])
            model = build_public_token_transformer(
                SemanticVanillaPublicTokenConfig(hidden_dim=32, layers=1, heads=4, mlp_dim=64)
            )
            output = model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(output["q"].shape, batch["action_mask"].shape)
        self.assertEqual(output["logits"].shape, batch["action_mask"].shape)

    def test_semantic_action_set_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_semantic_action_set_merit import (
            SemanticActionSetConfig,
            build_semantic_action_set_transformer,
        )
        from cascadiav3.torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots

        path = Path("cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
        fallback = Path("cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("semantic simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        try:
            batch = collate_semantic_relation_bias_roots(records[:2])
            model = build_semantic_action_set_transformer(
                SemanticActionSetConfig(hidden_dim=32, layers=1, heads=4, mlp_dim=64)
            )
            output = model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(output["q"].shape, batch["action_mask"].shape)
        self.assertEqual(output["logits"].shape, batch["action_mask"].shape)

    def test_semantic_species_moe_contract_when_enriched_roots_present(self) -> None:
        from cascadiav3.torch_semantic_species_moe_merit import (
            SemanticSpeciesMoEConfig,
            build_semantic_species_moe_transformer,
            collate_semantic_species_moe_roots,
        )

        path = Path("cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
        fallback = Path("cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
        source = path if path.exists() else fallback
        if not source.exists():
            self.skipTest("semantic simulator roots have not been generated")
        records = read_replay_jsonl(source)
        if "public_tokens" not in records[0]:
            self.skipTest("simulator roots predate public_tokens export")
        try:
            batch = collate_semantic_species_moe_roots(records[:2])
            model = build_semantic_species_moe_transformer(
                SemanticSpeciesMoEConfig(hidden_dim=32, layers=1, heads=4, mlp_dim=64)
            )
            output = model(
                batch["tokens"],
                batch["token_mask"],
                batch["actions"],
                batch["action_mask"],
                batch["relation_ids"],
                batch["action_species"],
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertEqual(output["q"].shape, batch["action_mask"].shape)
        self.assertEqual(output["logits"].shape, batch["action_mask"].shape)
        self.assertEqual(batch["action_species"].shape, batch["action_mask"].shape)
        self.assertGreaterEqual(int(batch["action_species"].min().item()), 0)
        self.assertLessEqual(int(batch["action_species"].max().item()), 5)

    def test_prefilter_metrics_when_sampled_roots_present(self) -> None:
        from cascadiav3.torch_relation_bias_merit import (
            _evaluate_relation_scores,
            collate_relation_bias_roots,
        )

        path = Path("cascadiav3/fixtures/crt_sampled_teacher_val.jsonl")
        if not path.exists():
            self.skipTest("sampled-teacher roots have not been generated")
        records = read_replay_jsonl(path)
        if "public_tokens" not in records[0]:
            self.skipTest("sampled roots predate public_tokens export")
        try:
            batch = collate_relation_bias_roots(records[:2])
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        metrics = _evaluate_relation_scores(
            [batch],
            lambda eval_batch: eval_batch["immediate"],
        )
        self.assertIn("prefilter", metrics)
        self.assertIn("4", metrics["prefilter"])
        self.assertIn("mean_oracle_regret", metrics["prefilter"]["4"])
        self.assertIn("top8_recall", metrics)
        self.assertIn("32", metrics["prefilter"])
        self.assertIn("top32_recall", metrics)
        self.assertIn("24", metrics["prefilter"])
        self.assertIn("top24_recall", metrics)

    def test_prefilter_eval_gate_contract(self) -> None:
        from cascadiav3.torch_prefilter_eval import (
            _config_from_report,
            _serving_decision,
            _uses_vanilla_public_token,
            parse_k_values,
        )
        from cascadiav3.torch_prefilter_blend_eval import simplex_weight_grid
        from cascadiav3.torch_prefilter_forensics import _source_metrics
        from cascadiav3.torch_prefilter_gate_eval import source_gate_feature_names
        from cascadiav3.torch_prefilter_seed_ensemble_eval import (
            _align_sources,
            evaluate_aligned_groups,
            parse_weights,
        )
        from cascadiav3.torch_prefilter_union_eval import quota_grid

        self.assertEqual(parse_k_values("24, 8, 16, 16"), [8, 16, 24])
        vanilla_config = _config_from_report(
            {
                "config": {
                    "model_name": "CRT-semantic-vanilla-public-token-query-v1",
                    "action_feature_dim": 61,
                    "hidden_dim": 256,
                    "layers": 4,
                    "heads": 8,
                    "mlp_dim": 512,
                }
            }
        )
        self.assertTrue(_uses_vanilla_public_token(vanilla_config))
        self.assertEqual(vanilla_config.action_feature_dim, 61)
        self.assertEqual(simplex_weight_grid(2, 0.5), [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
        self.assertEqual(parse_weights("2,1", 2), [2 / 3, 1 / 3])
        from cascadiav3.torch_greedy_policy_game_benchmark import parse_seeds as parse_game_benchmark_seeds

        self.assertEqual(
            parse_game_benchmark_seeds(seeds="", first_seed=2026990000, games=3),
            [2026990000, 2026990001, 2026990002],
        )
        self.assertEqual(
            parse_game_benchmark_seeds(seeds="7, 11", first_seed=0, games=99),
            [7, 11],
        )
        metrics = {
            "prefilter": {
                "8": {"recall": 0.60, "mean_oracle_regret": 0.50},
                "16": {"recall": 0.70, "mean_oracle_regret": 0.24},
                "24": {"recall": 0.82, "mean_oracle_regret": 0.12},
            }
        }
        decision = _serving_decision(
            metrics,
            k_values=[8, 16, 24],
            min_recall=0.75,
            max_oracle_regret=0.25,
        )
        self.assertTrue(decision["passes"])
        self.assertEqual(decision["recommended_k"], 24)
        self.assertFalse(decision["gates"]["16"]["passes"])
        rows = [
            {
                "state_hash": f"s{index}",
                "features": {
                    "active_tile_count": float(index),
                    "active_turns_remaining_est": 20.0 - float(index),
                    "active_current_base_score": 0.0,
                    "active_current_wildlife_total": 0.0,
                    "active_current_habitat_total": 0.0,
                    "active_nature_tokens": 0.0,
                    "public_token_count": 0.0,
                    "public_relation_count": 0.0,
                    "teacher_q_spread": 1.0,
                    "teacher_best_to_16th_margin": 0.5,
                    "teacher_best_variance": 1.0,
                    "teacher_best_immediate_delta_vs_root": 0.0,
                    "best_bear_pair_signal": 0.0,
                    "best_elk_best_line_length": 0.0,
                    "best_salmon_component_size": 0.0,
                    "best_hawk_isolated_signal": 0.0,
                    "best_fox_unique_adjacent_species_count": 0.0,
                    "best_public_market_species_count": 0.0,
                    "best_opponent_species_count_gap": 0.0,
                    "best_wildlife_bag_species_count": 0.0,
                    "best_unseen_tile_species_capacity": 0.0,
                },
                "categories": {
                    "wildlife_species": "bear",
                    "tile_slot": "0",
                    "wildlife_slot": "0",
                    "nature_spend": "0",
                    "cleanup_choice": "none",
                    "wildlife_present": "True",
                },
                "teacher_best_action": {"action_id": f"a{index}"},
                "sources": {
                    "mlp": {
                        "top16_hit": index < 3,
                        "top16_oracle_regret": 0.0 if index < 3 else 1.0,
                        "teacher_best_pred_rank": 1 if index < 3 else 17,
                        "selected_regret": 0.0,
                    }
                },
            }
            for index in range(4)
        ]
        metrics = _source_metrics(rows, "mlp", k=16)
        self.assertEqual(metrics["hits"], 3)
        self.assertEqual(metrics["misses"], 1)
        self.assertEqual(metrics["hits_needed_for_0_750"], 0)
        self.assertEqual(
            quota_grid(2, 3),
            [(0, 3), (1, 2), (2, 1), (3, 0)],
        )
        gate_features = source_gate_feature_names(("mlp", "immediate"))
        self.assertIn("mlp_zscore", gate_features)
        self.assertIn("immediate_top16", gate_features)
        self.assertIn("source_top16_votes", gate_features)
        seed_a = [
            {
                "state_hash": "s0",
                "ranked_action_ids": ["a0", "a1", "a2", "a3"],
                "ranked_predicted_q": [4.0, 3.0, 2.0, 1.0],
                "ranked_teacher_q": [0.0, 2.0, 3.0, 1.0],
                "teacher_best": {"action_id": "a2", "q": 3.0},
            }
        ]
        seed_b = [
            {
                "state_hash": "s0",
                "ranked_action_ids": ["a2", "a1", "a0", "a3"],
                "ranked_predicted_q": [8.0, 2.0, 1.0, 0.0],
                "ranked_teacher_q": [3.0, 2.0, 0.0, 1.0],
                "teacher_best": {"action_id": "a2", "q": 3.0},
            }
        ]
        ensemble_metrics = evaluate_aligned_groups(
            _align_sources([seed_a, seed_b]),
            weights=[0.5, 0.5],
            k_values=[1, 2],
        )
        self.assertEqual(ensemble_metrics["prefilter"]["1"]["recall"], 1.0)
        self.assertEqual(ensemble_metrics["prefilter"]["1"]["mean_oracle_regret"], 0.0)
        from cascadiav3.torch_prefilter_game_pilot import summarize_game_results

        pilot_summary = summarize_game_results(
            [
                {
                    "done": {
                        "scores": [{"total": 100}, {"total": 96}, {"total": 90}, {"total": 86}],
                    },
                    "decisions": [
                        {
                            "retained_count": 16,
                            "candidate_count": 32,
                            "model_score_seconds": 0.02,
                            "decision_seconds": 0.50,
                            "full_best_retained": True,
                            "search_regret": 0.0,
                        },
                        {
                            "retained_count": 16,
                            "candidate_count": 32,
                            "model_score_seconds": 0.03,
                            "decision_seconds": 0.60,
                            "full_best_retained": False,
                            "search_regret": 1.5,
                        },
                    ],
                }
            ]
        )
        self.assertEqual(pilot_summary["games"], 1)
        self.assertEqual(pilot_summary["decisions"], 2)
        self.assertEqual(pilot_summary["mean_seat_score"], 93.0)
        self.assertEqual(pilot_summary["shadow_full_best_retained_rate"], 0.5)
        self.assertEqual(pilot_summary["estimated_non_shadow_rollout_savings"], 0.5)
        from cascadiav3.torch_prefilter_game_compare import compare_reports

        comparison = compare_reports(
            candidate_report={
                "experiment_id": "candidate",
                "strategies": {
                    "prefilter-search": {"mean_total_decision_seconds": 2.0},
                },
                "games": [
                    {
                        "seed": 1,
                        "strategy": "prefilter-search",
                        "scores": [{"total": 100}, {"total": 96}],
                    },
                    {
                        "seed": 2,
                        "strategy": "prefilter-search",
                        "scores": [{"total": 90}, {"total": 94}],
                    },
                ],
            },
            baseline_report={
                "experiment_id": "baseline",
                "strategies": {
                    "full-search": {"mean_total_decision_seconds": 4.0},
                },
                "games": [
                    {
                        "seed": 1,
                        "strategy": "full-search",
                        "scores": [{"total": 98}, {"total": 96}],
                    },
                    {
                        "seed": 2,
                        "strategy": "full-search",
                        "scores": [{"total": 94}, {"total": 96}],
                    },
                ],
            },
            candidate_strategy="prefilter-search",
            baseline_strategy="full-search",
        )
        self.assertEqual(comparison["paired_seed_count"], 2)
        self.assertEqual(comparison["mean_delta_candidate_minus_baseline"], -1.0)
        self.assertEqual(comparison["speedup_factor"], 2.0)

    def test_top16_prefilter_loss_when_sampled_roots_present(self) -> None:
        from argparse import Namespace

        from cascadiav3.torch_relation_bias_merit import (
            RelationBiasConfig,
            _relation_loss_with_mode,
            build_relation_bias_transformer,
            collate_relation_bias_roots,
        )

        path = Path("cascadiav3/fixtures/crt_wide32_sampled_teacher_val.jsonl")
        if not path.exists():
            self.skipTest("wide32 sampled-teacher roots have not been generated")
        records = read_replay_jsonl(path)
        try:
            batch = collate_relation_bias_roots(records[:2])
            model = build_relation_bias_transformer(RelationBiasConfig(hidden_dim=32, layers=1, heads=4, mlp_dim=64))
            loss = _relation_loss_with_mode(
                model,
                batch,
                Namespace(
                    loss_mode="top16-prefilter",
                    q_loss_weight=0.25,
                    policy_loss_weight=0.5,
                    best_margin_loss_weight=1.0,
                    retention_loss_weight=1.0,
                    retention_k=16,
                    pairwise_margin=0.25,
                    policy_temperature=0.5,
                ),
            )
            retention_loss = _relation_loss_with_mode(
                model,
                batch,
                Namespace(
                    loss_mode="topk-retention",
                    q_loss_weight=0.15,
                    policy_loss_weight=0.25,
                    best_margin_loss_weight=1.0,
                    retention_loss_weight=1.5,
                    retention_k=16,
                    pairwise_margin=0.15,
                    policy_temperature=0.75,
                ),
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        self.assertGreater(float(loss.detach().cpu()), 0.0)
        self.assertGreater(float(retention_loss.detach().cpu()), 0.0)
        self.assertIn("target_q_count", batch)
        self.assertIn("target_q_variance", batch)


class ModelSmokeTest(unittest.TestCase):
    def test_greedy_prefix_filter_preserves_menu_order_and_teacher(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            self.skipTest(f"numpy not installed: {exc}")
        from cascadiav3.expert_tensor_shards import _retained_action_indices

        keep = _retained_action_indices(
            np.asarray([0.0, 9.0, 8.0, 7.0, 6.0], dtype=np.float32),
            np.asarray([True, True, True, True, True]),
            selected_action_index=4,
            top_k=3,
            filter_mode="greedy-prefix-with-selected",
        )
        self.assertEqual(keep.tolist(), [0, 1, 2, 4])
        strict_keep = _retained_action_indices(
            np.asarray([0.0, 9.0, 8.0, 7.0, 6.0], dtype=np.float32),
            np.asarray([True, True, True, True, True]),
            selected_action_index=4,
            top_k=3,
            filter_mode="greedy-prefix-strict",
        )
        self.assertEqual(strict_keep.tolist(), [0, 1, 2])
        union_keep = _retained_action_indices(
            np.asarray([0.0, 9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float32),
            np.asarray([True, True, True, True, True, True]),
            selected_action_index=5,
            top_k=4,
            filter_mode="greedy-prefix-plus-prior-with-selected",
            priors=np.asarray([0.01, 0.02, 0.03, 0.99, 0.98, 0.04], dtype=np.float32),
            greedy_prefix_k=2,
        )
        self.assertEqual(union_keep.tolist(), [0, 1, 3, 5])

    def test_k32_greedy_retention_loss_tracks_greedy_target_separately(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        from cascadiav3.torch_train_cascadiaformer import _loss_components, loss_weights_for_objective

        outputs = {
            "logits": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
            "q": torch.zeros((1, 3), dtype=torch.float32),
            "value_vector": torch.zeros((1, 4), dtype=torch.float32),
            "score_decomposition": torch.zeros((1, 3, 4), dtype=torch.float32),
            "rank_logits": torch.zeros((1, 4, 4), dtype=torch.float32),
            "uncertainty": torch.zeros((1, 3), dtype=torch.float32),
        }
        batch = {
            "action_mask": torch.tensor([[True, True, True]]),
            "q_valid": torch.tensor([[True, True, True]]),
            "selected_action_index": torch.tensor([2], dtype=torch.long),
            "greedy_action_index": torch.tensor([0], dtype=torch.long),
            "target_q": torch.zeros((1, 3), dtype=torch.float32),
            "target_value": torch.zeros((1, 4), dtype=torch.float32),
            "target_score": torch.zeros((1, 3, 4), dtype=torch.float32),
            "target_rank": torch.zeros((1, 4), dtype=torch.long),
        }
        weights = loss_weights_for_objective("k32-greedy-retention")
        losses = _loss_components(outputs, batch, weights)
        self.assertGreater(weights.greedy_policy, weights.policy)
        self.assertEqual(float(losses["teacher_top1"]), 1.0)
        self.assertEqual(float(losses["greedy_top1"]), 0.0)
        self.assertEqual(float(losses["mean_greedy_rank"]), 3.0)

    def test_search_improved_objective_uses_score_to_go_q(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch not installed: {exc}")
        from cascadiav3.torch_train_cascadiaformer import _loss_components, loss_weights_for_objective

        outputs = {
            "logits": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            "q": torch.tensor([[1.0, 8.0]], dtype=torch.float32),
            "value_vector": torch.zeros((1, 4), dtype=torch.float32),
            "score_decomposition": torch.zeros((1, 3, 4), dtype=torch.float32),
            "rank_logits": torch.zeros((1, 4, 4), dtype=torch.float32),
            "uncertainty": torch.zeros((1, 2), dtype=torch.float32),
        }
        batch = {
            "action_mask": torch.tensor([[True, True]]),
            "q_valid": torch.tensor([[True, True]]),
            "selected_action_index": torch.tensor([0], dtype=torch.long),
            "greedy_action_index": torch.tensor([0], dtype=torch.long),
            "target_q": torch.tensor([[101.0, 90.0]], dtype=torch.float32),
            "target_score_to_go": torch.tensor([[1.0, 90.0]], dtype=torch.float32),
            "exact_afterstate_score_active": torch.tensor([[100.0, 0.0]], dtype=torch.float32),
            "target_q_count": torch.ones((1, 2), dtype=torch.float32),
            "target_q_variance": torch.zeros((1, 2), dtype=torch.float32),
            "target_value": torch.zeros((1, 4), dtype=torch.float32),
            "target_score": torch.zeros((1, 3, 4), dtype=torch.float32),
            "target_rank": torch.zeros((1, 4), dtype=torch.long),
        }
        weights = loss_weights_for_objective("search-improved-greedy-retention")
        losses = _loss_components(outputs, batch, weights)
        self.assertAlmostEqual(weights.q, 0.20)
        self.assertIn("score_to_go_q", losses)
        self.assertIn("final_q_regret", losses)
        self.assertLess(float(losses["teacher_advantage_over_greedy"].detach().cpu()), 0.01)

    def test_q_serving_semantics_rank_by_afterstate_plus_score_to_go(self) -> None:
        from cascadiav3.torch_inference_bridge import derived_final_q_values, q_selection_index

        root = {
            "state_hash": "synthetic:q-serving",
            "active_seat": 0,
            "legal_actions": [{"action_id": "current"}, {"action_id": "remaining"}],
            "public_tokens": {"tokens": [], "token_count": 0},
            "exact_afterstate_score_active": [100.0, 0.0],
        }
        score_to_go = [-1.0, 10.0]
        self.assertEqual(max(range(2), key=lambda index: score_to_go[index]), 1)
        self.assertEqual(derived_final_q_values(root, score_to_go), [99.0, 10.0])
        self.assertEqual(q_selection_index(root, score_to_go), 0)

    def test_trainer_eval_cadence_is_configurable(self) -> None:
        import inspect

        from cascadiav3.torch_train_cascadiaformer import _passes_selection_guards, run_training

        parameters = inspect.signature(run_training).parameters
        self.assertIn("eval_every_steps", parameters)
        self.assertEqual(parameters["eval_every_steps"].default, 250)
        self.assertIn("min_selection_greedy_top1", parameters)
        self.assertIn("train_source_weights", parameters)
        self.assertTrue(_passes_selection_guards({"locked_val_greedy_top1": 0.19}, min_greedy_top1=0.0))
        self.assertFalse(_passes_selection_guards({"locked_val_greedy_top1": 0.19}, min_greedy_top1=0.20))
        self.assertTrue(_passes_selection_guards({"locked_val_greedy_top1": 0.20}, min_greedy_top1=0.20))

    def test_full_v3_runner_supports_weighted_extra_train_tensors(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_full_v3_training_pipeline.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("EXTRA_TRAIN_TAIL_TENSORS", text)
        self.assertIn('TRAIN_INPUT="\\$TRAIN_INPUT,$EXTRA_TRAIN_TAIL_TENSORS"', text)
        self.assertIn('--train "\\$TRAIN_INPUT"', text)
        self.assertIn('"extra_train_tail_tensors": "$EXTRA_TRAIN_TAIL_TENSORS"', text)

    def test_ei0_benchmark_runner_can_disable_shadow_full_search(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_cascadiaformer_ei0_benchmark_suite.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn('SEARCH_SHADOW_FULL_SEARCH="${SEARCH_SHADOW_FULL_SEARCH:-1}"', text)
        self.assertIn('SEARCH_INCLUDE_FULL_SEARCH_BASELINE="${SEARCH_INCLUDE_FULL_SEARCH_BASELINE:-1}"', text)
        self.assertIn('SEARCH_CPU_WORKERS="${SEARCH_CPU_WORKERS:-16}"', text)
        self.assertIn('SEARCH_CANDIDATE_WORKERS="${SEARCH_CANDIDATE_WORKERS:-$SEARCH_CPU_WORKERS}"', text)
        self.assertIn('SEARCH_BASELINE_WORKERS="${SEARCH_BASELINE_WORKERS:-$SEARCH_CPU_WORKERS}"', text)
        self.assertIn("NO_SEARCH_GAME_RESULTS", text)
        self.assertIn("SEARCH_GAME_RESULTS", text)
        self.assertIn("--game-results-out '$NO_SEARCH_GAME_RESULTS'", text)
        self.assertIn("--game-results-out '$SEARCH_GAME_RESULTS'", text)
        self.assertIn("PYTHONUNBUFFERED=1", text)
        self.assertIn("[ei0-bench] failed exit_code=", text)
        self.assertIn("search_extra_flags+=(--shadow-full-search)", text)
        self.assertIn("search_extra_flags+=(--include-full-search-baseline)", text)
        self.assertIn('"\\${search_extra_flags[@]}"', text)
        self.assertIn("skipping treatment/control ratio validation because full baseline is disabled", text)

    def test_mock_model_shapes_match_legal_actions(self) -> None:
        root = tiny_search_root_record()
        output = mock_forward(
            state_tokens=[{"token_kind": "GameToken"}],
            action_tokens=root["legal_actions"],
            cgab_edges=[],
        )
        validate_mock_output(output, action_count=len(root["legal_actions"]))


class ValidationCliTest(unittest.TestCase):
    def test_run_validation_passes(self) -> None:
        result = run_validation()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["radius6_cell_count"], 127)
        self.assertEqual(result["root_action_count"], 2)
        self.assertEqual(result["replay_action_counts"], [2, 3])


if __name__ == "__main__":
    unittest.main()
