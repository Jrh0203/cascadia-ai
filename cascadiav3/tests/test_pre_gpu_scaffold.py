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

    @staticmethod
    def _public_fixture_roots(limit: int) -> list[dict]:
        from cascadiav3.torch_inference_bridge import TRAINING_LABEL_KEYS

        path = Path("cascadiav3/fixtures/expert_tiny.jsonl")
        if not path.exists():
            return []
        roots = read_replay_jsonl(path)[:limit]
        return [
            {key: value for key, value in root.items() if key not in TRAINING_LABEL_KEYS}
            for root in roots
        ]

    def test_combined_relation_ids_array_matches_legacy_reference(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy unavailable")
        from cascadiav3.torch_relation_bias_merit import (
            RELATION_TO_ID,
            _coord_key,
            _set_relation,
            _token_indexes,
            combined_relation_ids,
            combined_relation_ids_array,
            relation_counts,
        )

        def legacy_combined_relation_ids(root, *, action_offset=None, seq_len=None):
            token_count = int(root["public_tokens"]["token_count"])
            action_count = len(root["legal_actions"])
            action_offset = token_count if action_offset is None else action_offset
            seq_len = action_offset + action_count if seq_len is None else seq_len
            matrix = [[0 for _ in range(seq_len)] for _ in range(seq_len)]
            same_board_id = RELATION_TO_ID["same_owner_board"]
            tokens_by_owner: dict[int, list[int]] = {}
            for token in root["public_tokens"]["tokens"]:
                kind = token.get("token_kind")
                owner = token.get("owner_seat")
                if owner is None or kind not in {"player", "placed_tile", "frontier"}:
                    continue
                tokens_by_owner.setdefault(int(owner), []).append(int(token["token_index"]))
            for indexes in tokens_by_owner.values():
                for source in indexes:
                    for target in indexes:
                        _set_relation(matrix, source, target, same_board_id)
            for relation in root["public_tokens"].get("relations", []):
                source = int(relation["source"])
                target = int(relation["target"])
                kind = relation.get("relation_kind")
                if kind == "adjacent_hex":
                    relation_id = (
                        RELATION_TO_ID["terrain_match_adjacent"]
                        if relation.get("terrain_matches")
                        else RELATION_TO_ID["adjacent_hex"]
                    )
                    _set_relation(matrix, source, target, relation_id, overwrite=True)
                elif kind == "same_market_slot":
                    _set_relation(matrix, source, target, RELATION_TO_ID["same_market_slot"], overwrite=True)
            indexes = _token_indexes(root)
            for action_index, action in enumerate(root["legal_actions"]):
                action_pos = action_offset + action_index
                tile_slot = int(action.get("tile_slot", action.get("draft_slot", -1)))
                wildlife_slot = int(action.get("wildlife_slot", action.get("draft_slot", -1)))
                tile_token = indexes["market_tile"].get(tile_slot)
                wildlife_token = indexes["market_wildlife"].get(wildlife_slot)
                target_frontier = indexes["active_frontier"].get(_coord_key(action.get("target_coord_ref")))
                wildlife_key = _coord_key(action.get("wildlife_coord_ref"))
                wildlife_target = indexes["active_tile"].get(wildlife_key)
                if wildlife_target is None:
                    wildlife_target = indexes["active_frontier"].get(wildlife_key)
                for target, relation_name in (
                    (tile_token, "action_uses_tile_slot"),
                    (wildlife_token, "action_uses_wildlife_slot"),
                    (target_frontier, "action_targets_tile_frontier"),
                    (wildlife_target, "action_targets_wildlife_cell"),
                ):
                    if target is None:
                        continue
                    relation_id = RELATION_TO_ID[relation_name]
                    _set_relation(matrix, action_pos, target, relation_id, overwrite=True)
                    _set_relation(matrix, target, action_pos, relation_id, overwrite=True)
            return matrix

        roots = self._public_fixture_roots(limit=4)
        if not roots:
            self.skipTest("expert tiny roots have not been generated")
        for root in roots:
            legacy = legacy_combined_relation_ids(root)
            vectorized = combined_relation_ids_array(root)
            self.assertTrue(np.array_equal(np.asarray(legacy), vectorized))
            self.assertEqual(combined_relation_ids(root), legacy)
            self.assertEqual(relation_counts(vectorized), relation_counts(legacy))

    def test_model_eval_batch_matches_single_evals_and_reports_value(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _model_eval, _model_eval_batch

        roots = self._public_fixture_roots(limit=3)
        if len(roots) < 2:
            self.skipTest("expert tiny roots have not been generated")
        torch.manual_seed(20260702)
        model = build_cascadiaformer(config_for_size("tiny"))
        model.eval()

        batch_responses = _model_eval_batch(model, roots)
        single_responses = [_model_eval(model, root) for root in roots]
        self.assertEqual(len(batch_responses), len(roots))
        for batch_response, single_response, root in zip(batch_responses, single_responses, roots):
            self.assertEqual(batch_response["action_ids"], single_response["action_ids"])
            self.assertEqual(len(batch_response["value"]), 4)
            for key in ("priors", "q", "score_to_go", "uncertainty"):
                for batched, single in zip(batch_response[key], single_response[key]):
                    self.assertAlmostEqual(batched, single, places=4)
            self.assertEqual(
                len(batch_response["priors"]), len(root["legal_actions"])
            )

    @staticmethod
    def _packed_variant(root: dict) -> dict:
        import base64

        import numpy as np

        from cascadiav3.torch_public_token_merit import public_token_features
        from cascadiav3.torch_relation_bias_merit import combined_relation_ids_array
        from cascadiav3.torch_semantic_relation_bias_merit import (
            semantic_public_token_action_features,
        )

        token_count = int(root["public_tokens"]["token_count"])
        action_count = len(root["legal_actions"])
        tokens = np.asarray(public_token_features(root), dtype="<f4")
        actions = np.asarray(semantic_public_token_action_features(root), dtype="<f4")
        matrix = combined_relation_ids_array(root)
        tail = matrix[token_count:, :].astype(np.uint8)
        return {
            "schema_id": root.get("schema_id"),
            "state_hash": root.get("state_hash"),
            "active_seat": root.get("active_seat"),
            "action_ids": [action["action_id"] for action in root["legal_actions"]],
            "exact_afterstate_score_active": root["exact_afterstate_score_active"],
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

    def test_packed_request_matches_raw_request_outputs(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _model_eval, _model_eval_batch

        roots = self._public_fixture_roots(limit=3)
        if len(roots) < 2:
            self.skipTest("expert tiny roots have not been generated")
        torch.manual_seed(20260702)
        model = build_cascadiaformer(config_for_size("tiny"))
        model.eval()

        packed_roots = [self._packed_variant(root) for root in roots]
        for root, packed_root in zip(roots, packed_roots):
            raw_response = _model_eval(model, root)
            packed_response = _model_eval(model, packed_root)
            self.assertEqual(raw_response["action_ids"], packed_response["action_ids"])
            for key in ("priors", "q", "score_to_go", "uncertainty", "value"):
                for raw_value, packed_value in zip(raw_response[key], packed_response[key]):
                    self.assertAlmostEqual(
                        raw_value,
                        packed_value,
                        places=4,
                        msg=f"{key} diverged between raw and packed paths",
                    )
        # Batched packed requests collate through the relation_tail path.
        batch_responses = _model_eval_batch(model, packed_roots)
        self.assertEqual(len(batch_responses), len(packed_roots))

    def test_pack_f64_b64_is_bit_exact_with_json_float_path(self) -> None:
        import base64
        import json as json_module
        import struct

        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy unavailable")
        from cascadiav3.torch_inference_bridge import pack_f64_b64

        # f32 model outputs widened to f64: exact, so the packed bytes must
        # equal what the JSON float-list wire path delivers after round-trip.
        values = np.array([0.1, 1.0 / 3.0, -2.5e-7, 80.0, 1e30], dtype=np.float32)
        encoded = pack_f64_b64(values)
        decoded = list(struct.unpack("<5d", base64.b64decode(encoded)))
        json_wire = json_module.loads(json_module.dumps(values.tolist()))
        self.assertEqual(decoded, json_wire)
        self.assertEqual(
            np.frombuffer(base64.b64decode(encoded), dtype="<f8").tolist(), json_wire
        )
        # Plain Python floats (f64) also round-trip bit-exactly.
        floats = [0.1, -1.0 / 7.0, 3.141592653589793]
        self.assertEqual(
            list(struct.unpack("<3d", base64.b64decode(pack_f64_b64(floats)))),
            json_module.loads(json_module.dumps(floats)),
        )

    def test_packed_response_decodes_to_exact_json_response_values(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        import base64
        import json as json_module

        import numpy as np

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_inference_bridge import _model_eval_batch

        roots = self._public_fixture_roots(limit=3)
        if len(roots) < 2:
            self.skipTest("expert tiny roots have not been generated")
        torch.manual_seed(20260702)
        model = build_cascadiaformer(config_for_size("tiny"))
        model.eval()

        json_rows = _model_eval_batch(model, roots)
        packed_rows = _model_eval_batch(model, roots, packed_response=True)
        self.assertEqual(len(json_rows), len(packed_rows))
        field_map = {
            "priors": "priors_f64_b64",
            "q": "q_f64_b64",
            "score_to_go": "score_to_go_f64_b64",
            "uncertainty": "uncertainty_f64_b64",
            "value": "value_f64_b64",
        }
        for json_row, packed_row in zip(json_rows, packed_rows):
            self.assertEqual(json_row["action_ids"], packed_row["action_ids"])
            self.assertIn("packed", packed_row)
            for key in field_map:
                self.assertNotIn(key, packed_row)
            packed = packed_row["packed"]
            for key, field in field_map.items():
                decoded = np.frombuffer(base64.b64decode(packed[field]), dtype="<f8").tolist()
                wire = json_module.loads(json_module.dumps(json_row[key]))
                self.assertEqual(
                    decoded, wire, msg=f"packed {key} diverged from the JSON path"
                )

    def test_serve_answers_eval_batch_request_with_fallback(self) -> None:
        import json as json_module
        import subprocess
        import sys

        roots = self._public_fixture_roots(limit=2)
        if len(roots) < 2:
            self.skipTest("expert tiny roots have not been generated")
        process = subprocess.Popen(
            [sys.executable, "-m", "cascadiav3.torch_inference_bridge", "--allow-dry-run-fallback"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={"PYTHONPATH": "cascadiav3/src", "PATH": "/usr/bin:/bin"},
        )
        try:
            hello = json_module.loads(process.stdout.readline())
            self.assertEqual(hello["type"], "hello")
            self.assertIn("eval_batch", hello.get("protocol_features", []))
            request = {"type": "eval_batch_request", "roots": roots, "allow_model_fallback": True}
            process.stdin.write(json_module.dumps(request) + "\n")
            process.stdin.flush()
            response = json_module.loads(process.stdout.readline())
            self.assertEqual(response["type"], "eval_batch_response")
            self.assertEqual(len(response["results"]), len(roots))
            for result, root in zip(response["results"], roots):
                self.assertEqual(result["type"], "eval_response")
                self.assertTrue(result["model_fallback"])
                self.assertEqual(len(result["priors"]), len(root["legal_actions"]))
                self.assertEqual(len(result["value"]), 4)
            process.stdin.write(json_module.dumps({"type": "shutdown"}) + "\n")
            process.stdin.flush()
        finally:
            process.stdin.close()
            process.stdout.close()
            process.wait(timeout=30)


class BridgeForwardOptimizationTest(unittest.TestCase):
    """CASCADIA_BRIDGE_BUCKET / _COMPILE / _TIMING forward-path knobs.

    All knobs default off; the default path must stay byte-identical, which the
    chunker-parity test pins and the pre-existing bridge tests cover.
    """

    @staticmethod
    def _public_fixture_roots(limit: int) -> list[dict]:
        return BridgeContractTest._public_fixture_roots(limit)

    @staticmethod
    def _truncated_root(root: dict, action_count: int) -> dict:
        trimmed = copy.deepcopy(root)
        for key in ("legal_actions", "exact_afterstate_score_active", "action_ids"):
            if key in trimmed:
                trimmed[key] = trimmed[key][:action_count]
        return trimmed

    def _tiny_model(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        torch.manual_seed(20260703)
        model = build_cascadiaformer(config_for_size("tiny"))
        model.eval()
        return model

    def _varied_shape_roots(self) -> list[dict]:
        roots = self._public_fixture_roots(limit=3)
        if len(roots) < 2:
            self.skipTest("expert tiny roots have not been generated")
        return [
            self._truncated_root(roots[0], 5),
            self._truncated_root(roots[1], 17),
            self._truncated_root(roots[0], 33),
            self._truncated_root(roots[1], 130),
            roots[2],
        ]

    def test_bucket_dim_contract(self) -> None:
        from cascadiav3.torch_inference_bridge import (
            EVAL_BUCKET_CAP,
            EVAL_BUCKET_MIN,
            EVAL_BUCKET_STEP_ABOVE_CAP,
            _bucket_dim,
        )

        self.assertEqual(_bucket_dim(1), EVAL_BUCKET_MIN)
        self.assertEqual(_bucket_dim(EVAL_BUCKET_MIN), EVAL_BUCKET_MIN)
        self.assertEqual(_bucket_dim(9), 16)
        self.assertEqual(_bucket_dim(61), 64)
        self.assertEqual(_bucket_dim(64), 64)
        self.assertEqual(_bucket_dim(65), 128)
        self.assertEqual(_bucket_dim(EVAL_BUCKET_CAP), EVAL_BUCKET_CAP)
        for size in (EVAL_BUCKET_CAP + 1, 648, 1000):
            padded = _bucket_dim(size)
            self.assertGreaterEqual(padded, size)
            self.assertEqual(padded % EVAL_BUCKET_STEP_ABOVE_CAP, 0)
            self.assertLess(padded - size, EVAL_BUCKET_STEP_ABOVE_CAP)

    def test_default_chunker_matches_legacy_behavior(self) -> None:
        import os
        import random
        from unittest import mock

        from cascadiav3.torch_inference_bridge import (
            EVAL_BATCH_CELL_BUDGET,
            _eval_batch_chunks,
        )

        def legacy_chunks(roots: list[dict], chunk_size: int) -> list[list[dict]]:
            chunks: list[list[dict]] = []
            current: list[dict] = []
            max_actions = 0
            max_seq = 0
            for root in roots:
                packed = root["packed_features"]
                action_count = int(packed.get("action_count", 0)) or 1
                token_count = int(packed.get("token_count", 0)) or 1
                candidate_actions = max(max_actions, action_count)
                candidate_seq = max(max_seq, token_count + action_count)
                cells = (len(current) + 1) * candidate_actions * candidate_seq
                if current and (len(current) >= chunk_size or cells > EVAL_BATCH_CELL_BUDGET):
                    chunks.append(current)
                    current = []
                    candidate_actions = action_count
                    candidate_seq = token_count + action_count
                current.append(root)
                max_actions = candidate_actions
                max_seq = candidate_seq
            if current:
                chunks.append(current)
            return chunks

        rng = random.Random(20260703)
        roots = [
            {
                "packed_features": {
                    "action_count": rng.choice([1, 5, 33, 256, 648]),
                    "token_count": rng.choice([9, 61, 64]),
                }
            }
            for _ in range(200)
        ]
        with mock.patch.dict(os.environ):
            os.environ.pop("CASCADIA_BRIDGE_BUCKET", None)
            self.assertEqual(_eval_batch_chunks(roots, chunk_size=32), legacy_chunks(roots, 32))

    def test_bucketed_chunker_bounds_padded_cells(self) -> None:
        import os
        from unittest import mock

        from cascadiav3.torch_inference_bridge import (
            EVAL_BATCH_CELL_BUDGET,
            _bucket_dim,
            _eval_batch_chunks,
        )

        roots = [
            {"packed_features": {"action_count": actions, "token_count": tokens}}
            for actions, tokens in [(648, 64), (405, 61), (256, 64), (33, 61)] * 12
        ]
        with mock.patch.dict(os.environ, {"CASCADIA_BRIDGE_BUCKET": "1"}):
            chunks = _eval_batch_chunks(roots, chunk_size=32)
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(roots))
        for chunk in chunks:
            max_actions = _bucket_dim(max(r["packed_features"]["action_count"] for r in chunk))
            max_tokens = _bucket_dim(max(r["packed_features"]["token_count"] for r in chunk))
            padded_cells = len(chunk) * max_actions * (max_tokens + max_actions)
            if len(chunk) > 1:
                self.assertLessEqual(padded_cells, EVAL_BATCH_CELL_BUDGET)

    def test_bucketed_collate_pads_capacities_to_buckets(self) -> None:
        import os
        from unittest import mock

        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_inference_bridge import _bucket_dim, collate_inference_roots

        roots = self._varied_shape_roots()[:3]
        packed_roots = [BridgeContractTest._packed_variant(root) for root in roots]
        token_bucket = _bucket_dim(max(int(root["public_tokens"]["token_count"]) for root in roots))
        action_bucket = _bucket_dim(max(len(root["legal_actions"]) for root in roots))
        with mock.patch.dict(os.environ, {"CASCADIA_BRIDGE_BUCKET": "1"}):
            raw_batch = collate_inference_roots(roots)
            packed_batch = collate_inference_roots(packed_roots)
        for batch in (raw_batch, packed_batch):
            self.assertEqual(batch["tokens"].shape[1], token_bucket)
            self.assertEqual(batch["actions"].shape[1], action_bucket)
            self.assertEqual(batch["combined_seq_len"], token_bucket + action_bucket)
            for row, root in enumerate(roots):
                self.assertEqual(
                    int(batch["token_mask"][row].sum()), int(root["public_tokens"]["token_count"])
                )
                self.assertEqual(int(batch["action_mask"][row].sum()), len(root["legal_actions"]))
        self.assertEqual(raw_batch["relation_ids"].shape[1], token_bucket + action_bucket)
        self.assertEqual(packed_batch["relation_tail"].shape[1], action_bucket)
        self.assertEqual(packed_batch["relation_tail"].shape[2], token_bucket + action_bucket)

    def test_bucketed_padding_is_garbage_invariant(self) -> None:
        """Exactness of mask handling: padded token/action feature rows must not
        influence real rows at a fixed bucketed shape. Filling the padded region
        with garbage instead of zeros must leave every real output bit-identical
        (relation-id padding stays 0 by contract: id 0 = "no relation" via
        padding_idx and the CGAB ne(0) mask)."""
        import os
        from unittest import mock

        model = self._tiny_model()
        import torch

        from cascadiav3.torch_inference_bridge import collate_inference_roots

        roots = self._varied_shape_roots()[:4]
        packed_roots = [BridgeContractTest._packed_variant(root) for root in roots]
        with mock.patch.dict(os.environ, {"CASCADIA_BRIDGE_BUCKET": "1"}):
            batches = [collate_inference_roots(roots), collate_inference_roots(packed_roots)]
        generator = torch.Generator().manual_seed(20260703)
        for batch in batches:
            tokens_garbage = batch["tokens"].clone()
            pad_tokens = ~batch["token_mask"]
            tokens_garbage[pad_tokens] = torch.randn(
                (int(pad_tokens.sum()), tokens_garbage.shape[-1]), generator=generator
            )
            actions_garbage = batch["actions"].clone()
            pad_actions = ~batch["action_mask"]
            actions_garbage[pad_actions] = torch.randn(
                (int(pad_actions.sum()), actions_garbage.shape[-1]), generator=generator
            )
            with torch.inference_mode():
                reference = model(
                    batch["tokens"],
                    batch["token_mask"],
                    batch["actions"],
                    batch["action_mask"],
                    relation_ids=batch.get("relation_ids"),
                    relation_tail=batch.get("relation_tail"),
                )
                garbage = model(
                    tokens_garbage,
                    batch["token_mask"],
                    actions_garbage,
                    batch["action_mask"],
                    relation_ids=batch.get("relation_ids"),
                    relation_tail=batch.get("relation_tail"),
                )
            self.assertTrue(torch.equal(reference["value_vector"], garbage["value_vector"]))
            for key in ("logits", "q", "uncertainty"):
                for row, action_count in enumerate(batch["action_counts"]):
                    self.assertTrue(
                        torch.equal(
                            reference[key][row, :action_count],
                            garbage[key][row, :action_count],
                        ),
                        msg=f"{key} row {row} leaked padding",
                    )

    def test_bucketed_eval_matches_unbucketed_within_reduction_tolerance(self) -> None:
        """Bucketed vs unbucketed responses. Bit-exact equality is impossible on
        this stack: CPU attention/sum kernels block reductions over the padded
        length, so appending exact zeros regroups the floating-point reduction of
        the real prefix (measured ~2e-7; the SDPA MATH backend shows the same).
        The default chunk-max padding already admits the same drift class, so
        the gate here is a tight tolerance, not torch.equal."""
        import os
        from unittest import mock

        model = self._tiny_model()
        import numpy as np

        from cascadiav3.torch_inference_bridge import _model_eval_batch

        roots = self._varied_shape_roots()
        packed_roots = [BridgeContractTest._packed_variant(root) for root in roots]
        for request_roots in (roots, packed_roots):
            baseline = _model_eval_batch(model, request_roots)
            with mock.patch.dict(os.environ, {"CASCADIA_BRIDGE_BUCKET": "1"}):
                bucketed = _model_eval_batch(model, request_roots)
            self.assertEqual(len(baseline), len(bucketed))
            for base_row, bucket_row in zip(baseline, bucketed):
                self.assertEqual(base_row["action_ids"], bucket_row["action_ids"])
                for key in ("priors", "q", "score_to_go", "uncertainty", "value"):
                    self.assertEqual(len(base_row[key]), len(bucket_row[key]))
                    self.assertTrue(
                        np.allclose(base_row[key], bucket_row[key], rtol=1e-4, atol=1e-5),
                        msg=(
                            f"{key} drifted beyond reduction tolerance: "
                            f"{np.max(np.abs(np.asarray(base_row[key]) - np.asarray(bucket_row[key])))}"
                        ),
                    )

    def test_compile_knob_smoke_on_cpu(self) -> None:
        model = self._tiny_model()
        import torch

        from cascadiav3.torch_inference_bridge import _maybe_compile_model, _model_eval_batch

        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        roots = [self._truncated_root(root, 5) for root in self._public_fixture_roots(limit=2)]
        baseline = _model_eval_batch(model, roots)
        try:
            compiled = _maybe_compile_model(model, torch.device("cpu"))
            responses = _model_eval_batch(compiled, roots)
        except Exception as exc:  # pragma: no cover - depends on local toolchain
            self.skipTest(f"torch.compile unusable in this environment: {exc}")
        import numpy as np

        self.assertEqual(len(responses), len(baseline))
        for base_row, compiled_row in zip(baseline, responses):
            self.assertEqual(base_row["action_ids"], compiled_row["action_ids"])
            for key in ("priors", "q", "score_to_go", "uncertainty", "value"):
                self.assertTrue(
                    np.allclose(base_row[key], compiled_row[key], rtol=1e-4, atol=1e-5),
                    msg=f"{key} diverged under torch.compile",
                )

    def test_timing_knob_defaults_off_and_accumulates_when_patched(self) -> None:
        import contextlib
        import io
        from unittest import mock

        model = self._tiny_model()
        from cascadiav3 import torch_inference_bridge as bridge

        self.assertIsNone(bridge._BRIDGE_TIMING)
        fixture_roots = self._public_fixture_roots(limit=2)
        if not fixture_roots:
            self.skipTest("expert tiny roots have not been generated")
        roots = [self._truncated_root(root, 9) for root in fixture_roots]
        timing = bridge._BridgeTiming()
        with mock.patch.object(bridge, "_BRIDGE_TIMING", timing):
            bridge._model_eval_batch(model, roots)
        self.assertGreaterEqual(timing.chunks, 1)
        self.assertEqual(timing.rows, len(roots))
        self.assertEqual(timing.actions, 9 * len(roots))
        self.assertGreater(timing.collate_s + timing.forward_s, 0.0)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            timing.emit("test")
        summary = stderr.getvalue()
        self.assertIn("[bridge-timing test]", summary)
        self.assertIn(f"rows={len(roots)}", summary)
        for phase in ("collate=", "h2d=", "forward=", "d2h=", "encode="):
            self.assertIn(phase, summary)

    def test_timing_sync_dispatches_to_both_accelerators(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from cascadiav3.torch_inference_bridge import (
            _synchronize_device_for_timing,
        )

        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=mock.Mock()),
            mps=SimpleNamespace(synchronize=mock.Mock()),
        )
        _synchronize_device_for_timing(torch_module, SimpleNamespace(type="cuda"))
        torch_module.cuda.synchronize.assert_called_once_with()
        torch_module.mps.synchronize.assert_not_called()
        _synchronize_device_for_timing(torch_module, SimpleNamespace(type="mps"))
        torch_module.mps.synchronize.assert_called_once_with()
        _synchronize_device_for_timing(torch_module, SimpleNamespace(type="cpu"))
        torch_module.cuda.synchronize.assert_called_once_with()
        torch_module.mps.synchronize.assert_called_once_with()


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
        from cascadiav3.torch_cascadiaformer_game_benchmark import (
            parse_seeds,
            summarize_game_results,
            summarize_market_decisions,
        )

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
        market = summarize_market_decisions(
            [
                {
                    "decisions": [
                        {"free_three_of_a_kind_choice": "accept"},
                        {"free_three_of_a_kind_choice": "decline"},
                        {"free_three_of_a_kind_choice": "not_available"},
                    ]
                }
            ]
        )
        self.assertEqual(market["available_decisions"], 2)
        self.assertEqual(market["acceptance_rate_when_available"], 0.5)

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
        self.assertIn('Q_DECOMPOSITION="${Q_DECOMPOSITION:-0}"', text)
        self.assertIn('Q_DECOMPOSITION_HEAD_ONLY="${Q_DECOMPOSITION_HEAD_ONLY:-0}"', text)
        self.assertIn("TRAINER_Q_ARGS+=(--q-decomposition)", text)
        self.assertIn("TRAINER_Q_ARGS+=(--q-decomposition-head-only)", text)
        self.assertIn('"q_decomposition_head_only": "$Q_DECOMPOSITION_HEAD_ONLY" == "1"', text)

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


class GumbelSelfplayContractTest(unittest.TestCase):
    FIXTURE = Path("cascadiav3/fixtures/gumbel_tiny_tensor.npz")

    def _require_numpy_and_fixture(self):  # type: ignore[no-untyped-def]
        try:
            import numpy as np  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("numpy unavailable")
        if not self.FIXTURE.exists():
            self.skipTest("gumbel tiny tensor fixture has not been generated")

    def test_v2_shard_loads_with_improved_policy(self) -> None:
        self._require_numpy_and_fixture()
        import numpy as np

        from cascadiav3.expert_tensor_shards import SHARD_VERSION_V2, ExpertTensorShard

        shard = ExpertTensorShard(self.FIXTURE)
        try:
            self.assertEqual(shard.version, SHARD_VERSION_V2)
            self.assertIsNotNone(shard.improved_policy)
            self.assertIsNotNone(shard.search_root_value)
            for index in range(len(shard)):
                example = shard.example(index)
                policy = np.asarray(example["improved_policy"], dtype=np.float64)
                self.assertEqual(policy.shape[0], example["actions"].shape[0])
                self.assertAlmostEqual(float(policy.sum()), 1.0, places=4)
                visits = np.asarray(example["visits"], dtype=np.float64)
                q_valid = np.asarray(example["q_valid"], dtype=bool)
                self.assertTrue(np.array_equal(visits > 0, q_valid))
        finally:
            shard.close()

    def test_v2_fields_survive_filter_and_relation_tail(self) -> None:
        self._require_numpy_and_fixture()
        import numpy as np

        from cascadiav3.expert_tensor_shards import (
            ExpertTensorShard,
            filter_expert_tensor_shard,
            materialize_relation_tail_shard,
        )

        with tempfile.TemporaryDirectory() as tmp:
            filtered = Path(tmp) / "filtered.npz"
            filter_expert_tensor_shard(self.FIXTURE, filtered, top_k=8)
            shard = ExpertTensorShard(filtered)
            try:
                self.assertIsNotNone(shard.improved_policy)
                for index in range(len(shard)):
                    example = shard.example(index)
                    policy = np.asarray(example["improved_policy"], dtype=np.float64)
                    self.assertAlmostEqual(float(policy.sum()), 1.0, places=4)
            finally:
                shard.close()

            tailed = Path(tmp) / "tailed.npz"
            materialize_relation_tail_shard(filtered, tailed)
            shard = ExpertTensorShard(tailed)
            try:
                self.assertIsNotNone(shard.improved_policy)
                self.assertIsNotNone(shard.relation_tail)
            finally:
                shard.close()

    def test_gumbel_selfplay_objective_weights(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import loss_weights_for_objective

        weights = loss_weights_for_objective("gumbel-selfplay")
        self.assertEqual(weights.policy, 1.0)
        self.assertEqual(weights.q, 0.5)
        self.assertEqual(weights.value, 0.5)
        self.assertEqual(weights.greedy_policy, 0.0)
        self.assertEqual(weights.greedy_margin, 0.0)

    def test_improved_policy_soft_target_loss(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        from cascadiav3.torch_train_cascadiaformer import LossWeights, _loss_components

        batch_size, actions = 2, 3
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 0.5]])
        improved = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.25, 0.25]])
        outputs = {
            "logits": logits.clone(),
            "q": torch.zeros((batch_size, actions)),
            "uncertainty": torch.ones((batch_size, actions)),
            "value_vector": torch.zeros((batch_size, 4)),
            "rank_logits": torch.zeros((batch_size, 4, 4)),
            "score_decomposition": torch.zeros((batch_size, 3, 4)),
        }
        batch = {
            "action_mask": torch.ones((batch_size, actions), dtype=torch.bool),
            "q_valid": torch.ones((batch_size, actions), dtype=torch.bool),
            "target_q": torch.zeros((batch_size, actions)),
            "target_score_to_go": torch.zeros((batch_size, actions)),
            "exact_afterstate_score_active": torch.zeros((batch_size, actions)),
            "selected_action_index": torch.zeros((batch_size,), dtype=torch.long),
            "target_value": torch.zeros((batch_size, 4)),
            "target_rank": torch.zeros((batch_size, 4), dtype=torch.long),
            "target_score": torch.zeros((batch_size, 3, 4)),
            "improved_policy": improved,
            "has_improved_policy": True,
        }
        components = _loss_components(outputs, batch, LossWeights())
        expected = -(improved * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        self.assertAlmostEqual(float(components["policy"]), float(expected), places=5)

        batch_without = dict(batch)
        batch_without["has_improved_policy"] = False
        components_without = _loss_components(outputs, batch_without, LossWeights())
        self.assertNotAlmostEqual(
            float(components_without["policy"]), float(expected), places=5
        )

    def test_training_smoke_with_max_example_passes_clamp(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")
        self._require_numpy_and_fixture()
        from cascadiav3.torch_train_cascadiaformer import run_training

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = run_training(
                [self.FIXTURE],
                [self.FIXTURE],
                train_format="npz",
                val_format="npz",
                model_size="tiny",
                steps=50,
                batch_size=4,
                lr=1.0e-3,
                weight_decay=0.0,
                device_name="cpu",
                seed=7,
                grad_accum=1,
                warmup_fraction=0.1,
                checkpoint_dir=tmp_path / "checkpoints",
                metrics_jsonl=tmp_path / "metrics.jsonl",
                out=tmp_path / "train.json",
                overfit_one_batch=False,
                val_max_batches=1,
                swa_fraction=0.5,
                objective="gumbel-selfplay",
                max_example_passes=4.0,
            )
            # 12 records * 4 passes / batch 4 = 12 steps, clamped from 50.
            self.assertEqual(report["steps"], 12)
            self.assertEqual(report["objective"], "gumbel-selfplay")


class BenchmarkStatsTest(unittest.TestCase):
    def test_t_quantile_matches_reference_values(self) -> None:
        from cascadiav3.torch_benchmark_stats import t_quantile

        self.assertAlmostEqual(t_quantile(0.975, 10), 2.2281, places=3)
        self.assertAlmostEqual(t_quantile(0.975, 1), 12.7062, places=2)
        self.assertAlmostEqual(t_quantile(0.975, 100), 1.9840, places=3)
        self.assertAlmostEqual(t_quantile(0.025, 10), -2.2281, places=3)

    def test_paired_delta_stats_known_values(self) -> None:
        from cascadiav3.torch_benchmark_stats import paired_delta_stats

        stats = paired_delta_stats([1.0, 2.0, 3.0, 4.0, 5.0], seed=1)
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["se"], 0.70710678, places=6)
        self.assertAlmostEqual(stats["t_ci_low"], 3.0 - 2.7764 * 0.70710678, places=3)
        self.assertAlmostEqual(stats["t_ci_high"], 3.0 + 2.7764 * 0.70710678, places=3)
        self.assertTrue(stats["ci_excludes_zero"])
        self.assertLess(stats["bootstrap_ci_low"], stats["mean"])
        self.assertGreater(stats["bootstrap_ci_high"], stats["mean"])

        # Deterministic given the seed.
        again = paired_delta_stats([1.0, 2.0, 3.0, 4.0, 5.0], seed=1)
        self.assertEqual(stats, again)

        # A noisy near-zero delta set must not claim significance.
        noisy = paired_delta_stats([0.5, -0.75, 1.25, -1.0, 0.25, -0.25])
        self.assertFalse(noisy["ci_excludes_zero"])

        empty = paired_delta_stats([])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["mean"])

    def test_gumbel_benchmark_collects_canned_results(self) -> None:
        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
            _contiguous_runs,
            collect_gumbel_results,
            summarize_market_decisions,
            summarize_score_categories,
            write_completed_game_rows,
        )

        self.assertEqual(_contiguous_runs([5, 6, 7, 10, 12, 13]), [(5, 3), (10, 1), (12, 2)])

        lines = [
            {
                "type": "gumbel_decision",
                "seed": 5,
                "ply": 0,
                "decision_seconds": 0.5,
                "free_three_of_a_kind_choice": "accept",
                "market_chance_samples": 8,
                "simulations_run": 16,
                "total_simulations_run": 160,
            },
            {
                "type": "gumbel_decision",
                "seed": 5,
                "ply": 1,
                "decision_seconds": 0.7,
                "free_three_of_a_kind_choice": "decline",
                "market_chance_samples": 8,
                "simulations_run": 16,
                "total_simulations_run": 144,
            },
            {
                "type": "gumbel_game_done",
                "seed": 5,
                "scores": [
                    {"wildlife": [25, 25], "habitat": [35], "nature_tokens": 5, "total": 90},
                    {"wildlife": [30, 25], "habitat": [34], "nature_tokens": 6, "total": 95},
                    {"wildlife": [24, 24], "habitat": [35], "nature_tokens": 5, "total": 88},
                    {"wildlife": [26, 26], "habitat": [35], "nature_tokens": 5, "total": 92},
                ],
                "decision_count": 2,
                "elapsed_seconds": 3.5,
            },
        ]
        results = collect_gumbel_results(lines)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["seed"], 5)
        self.assertEqual(len(results[0]["decisions"]), 2)
        self.assertEqual(results[0]["done"]["scores"][1]["total"], 95)
        market = summarize_market_decisions(lines)
        self.assertEqual(market["available_decisions"], 2)
        self.assertEqual(market["acceptance_rate_when_available"], 0.5)
        self.assertEqual(market["mean_chance_samples_when_available"], 8.0)
        self.assertEqual(market["market_decision_simulation_overhead"], 272)
        categories = summarize_score_categories(results)
        assert categories is not None
        self.assertEqual(categories["overall_mean"]["total"], (90 + 95 + 88 + 92) / 4)
        self.assertEqual(categories["overall_mean"]["wildlife"], (50 + 55 + 48 + 52) / 4)
        self.assertEqual(categories["games_mean_at_least_100"], 0)
        self.assertEqual(categories["seat_scores_at_least_100"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            games_path = Path(tmp) / "nested" / "games.jsonl"
            write_completed_game_rows(lines, [5], games_path)
            self.assertEqual(
                [json.loads(line)["seed"] for line in games_path.read_text().splitlines()],
                [5],
            )
            incomplete_path = Path(tmp) / "incomplete.jsonl"
            with self.assertRaisesRegex(RuntimeError, "do not match the battery"):
                write_completed_game_rows(lines, [5, 6], incomplete_path)
            self.assertFalse(incomplete_path.exists())

        from cascadiav3.torch_cascadiaformer_search_benchmark import summarize_game_results

        summary = summarize_game_results(results)
        self.assertEqual(summary["games"], 1)
        self.assertAlmostEqual(summary["mean_seat_score"], (90 + 95 + 88 + 92) / 4)

    def test_corrected_rules_comparator_validates_and_pairs_reports(self) -> None:
        from cascadiav3.compare_rules_rebaseline import RULESET_ID, build_comparison

        revision = "tested-revision"
        specs = {
            "rules_20260709_cycle4_n256_d4.json": (256, 4, [90.0, 92.0]),
            "rules_20260709_distq_k8_n256_d4.json": (256, 4, [91.0, 94.0]),
            "rules_20260709_cycle4_n1024_d16.json": (1024, 16, [93.0, 95.0]),
            "rules_20260709_distq_k8_n1024_d16.json": (1024, 16, [94.0, 97.0]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, (simulations, determinizations, scores) in specs.items():
                report = {
                    "status": "pass",
                    "ruleset_id": RULESET_ID,
                    "source_revision": revision,
                    "experiment_id": name.removesuffix(".json"),
                    "seeds": [1, 2],
                    "search": {
                        "n_simulations": simulations,
                        "determinizations": determinizations,
                        "market_decision_samples": 8,
                    },
                    "strategies": {"gumbel-search": {"mean_seat_score": sum(scores) / 2}},
                    "candidate_per_seed": [
                        {"seed": seed, "mean_score_per_seat": score}
                        for seed, score in zip((1, 2), scores)
                    ],
                    "market_decisions": {
                        "accepted": 3,
                        "declined": 1,
                        "acceptance_rate_when_available": 0.75,
                    },
                }
                (root / name).write_text(json.dumps(report), encoding="utf-8")

            result = build_comparison(root, revision)

        low = result["comparisons"]["distq_minus_cycle4_n256_d4"]
        self.assertEqual(low["paired_delta_stats"]["mean"], 1.5)
        self.assertEqual(result["reports"]["cycle4_n256_d4"]["market_decisions"]["declined"], 1)

    def test_exact_endgame_comparator_validates_contract_and_pairs_reports(self) -> None:
        from cascadiav3.compare_exact_endgame import RULESET_ID, build_comparison

        def report(exact_turns: int, scores: list[float]) -> dict[str, object]:
            return {
                "status": "pass",
                "ruleset_id": RULESET_ID,
                "source_revision": "tested-revision",
                "experiment_id": f"exact-{exact_turns}",
                "manifest": f"/host-{exact_turns}/best_locked_val.manifest.json",
                "seeds": [1, 2],
                "search": {
                    "n_simulations": 16,
                    "determinizations": 2,
                    "market_decision_samples": 4,
                    "exact_endgame_turns": exact_turns,
                },
                "control": {"kind": "none"},
                "strategies": {
                    "gumbel-search": {
                        "mean_seat_score": sum(scores) / len(scores),
                        "mean_total_decision_seconds": 2.0 - 0.5 * exact_turns,
                    }
                },
                "candidate_per_seed": [
                    {"seed": seed, "mean_score_per_seat": score, "seat_scores": [score] * 4}
                    for seed, score in zip((1, 2), scores)
                ],
                "market_decisions": {"exact_endgame_decisions": 8 * exact_turns},
                "candidate_decision_seconds_p50": 1.0,
                "candidate_decision_seconds_p95": 3.0 - exact_turns,
                "candidate_wall_seconds": 20.0 - 5.0 * exact_turns,
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            exact_path = root / "exact.json"
            baseline_decisions_path = root / "baseline_decisions.jsonl"
            exact_decisions_path = root / "exact_decisions.jsonl"
            baseline_path.write_text(json.dumps(report(0, [90.0, 92.0])), encoding="utf-8")
            exact_path.write_text(json.dumps(report(1, [91.0, 94.0])), encoding="utf-8")
            baseline_rows = []
            exact_rows = []
            for seed in (1, 2):
                for ply in range(80):
                    base = {
                        "type": "gumbel_decision",
                        "ruleset_id": RULESET_ID,
                        "seed": seed,
                        "ply": ply,
                        "chosen_action_id": f"action-{seed}-{ply}",
                        "free_three_of_a_kind_choice": "not_available",
                        "exact_endgame": False,
                        "total_simulations_run": 16,
                        "decision_seconds": 1.0,
                    }
                    baseline_rows.append(base)
                    exact_rows.append(
                        base
                        | {
                            "chosen_action_id": (
                                base["chosen_action_id"] if ply < 76 else f"exact-{seed}-{ply}"
                            ),
                            "exact_endgame": ply >= 76,
                            "total_simulations_run": 0 if ply >= 76 else 16,
                        }
                    )
            baseline_decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_rows), encoding="utf-8"
            )
            exact_decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in exact_rows), encoding="utf-8"
            )
            result = build_comparison(
                baseline_path,
                exact_path,
                baseline_decisions_path,
                exact_decisions_path,
                "tested-revision",
            )

            self.assertEqual(result["paired_delta_stats"]["mean"], 1.5)
            self.assertEqual(result["exact_decisions"], 8)
            self.assertEqual(result["exact_frontier"]["action_changes"], 8)
            self.assertEqual(result["exact_frontier"]["speedup"], 1.0)
            self.assertEqual(result["seat0_exact_score_deltas"], [1.0, 2.0])
            self.assertAlmostEqual(result["timing"]["mean_decision_speedup"], 4.0 / 3.0)

            broken = report(1, [91.0, 94.0])
            broken["market_decisions"] = {"exact_endgame_decisions": 7}
            exact_path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 8"):
                build_comparison(
                    baseline_path,
                    exact_path,
                    baseline_decisions_path,
                    exact_decisions_path,
                )

            exact_path.write_text(json.dumps(report(1, [91.0, 94.0])), encoding="utf-8")
            exact_rows[5]["chosen_action_id"] = "early-divergence"
            exact_decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in exact_rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "pre-K1 action trace diverges"):
                build_comparison(
                    baseline_path,
                    exact_path,
                    baseline_decisions_path,
                    exact_decisions_path,
                )

    def test_gumbel_execution_comparator_requires_exact_policy_parity(self) -> None:
        from cascadiav3.compare_gumbel_execution import RULESET_ID, build_comparison

        def report(parallel: bool, wall: float, mean_decision: float) -> dict[str, object]:
            return {
                "status": "pass",
                "scientific_eligibility": "candidate_only_search_arm",
                "ruleset_id": RULESET_ID,
                "source_revision": "tested-revision",
                "seeds": [1, 2],
                "execution": {"runner": "gumbel-benchmark-batch", "requested_jobs": 1},
                "search": {
                    "n_simulations": 16,
                    "determinizations": 2,
                    "blend_weight": 0.5,
                    "parallel_leaf_rollouts": parallel,
                },
                "artifacts": {
                    "binary_sha256": "binary",
                    "manifest_sha256": "manifest",
                    "weights_sha256": "weights",
                    "checkpoint_tag": "best",
                    "checkpoint_step": 7,
                    "q_quantiles": 8,
                },
                "control": {"kind": "none"},
                "strategies": {
                    "gumbel-search": {
                        "mean_seat_score": 92.0,
                        "mean_total_decision_seconds": mean_decision,
                    }
                },
                "candidate_wall_seconds": wall,
                "candidate_decision_seconds_p50": mean_decision / 2,
                "candidate_decision_seconds_p95": mean_decision * 2,
            }

        score = {
            "wildlife": [10, 10, 10, 10, 10],
            "habitat": [5, 5, 5, 5, 5],
            "nature_tokens": 5,
            "total": 80,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                name: root / name
                for name in (
                    "baseline.json",
                    "candidate.json",
                    "baseline_decisions.jsonl",
                    "candidate_decisions.jsonl",
                    "baseline_games.jsonl",
                    "candidate_games.jsonl",
                )
            }
            paths["baseline.json"].write_text(
                json.dumps(report(False, 100.0, 2.0)), encoding="utf-8"
            )
            paths["candidate.json"].write_text(
                json.dumps(report(True, 80.0, 1.6)), encoding="utf-8"
            )
            decisions = []
            games = []
            for seed in (1, 2):
                for ply in range(80):
                    decisions.append(
                        {
                            "type": "gumbel_decision",
                            "ruleset_id": RULESET_ID,
                            "seed": seed,
                            "ply": ply,
                            "action_count": 16,
                            "chosen_action_id": f"action-{seed}-{ply}",
                            "free_three_of_a_kind_choice": "not_available",
                            "root_value": float(seed + ply),
                            "simulations_run": 16,
                            "market_branches_searched": 1,
                            "market_chance_samples": 0,
                            "total_simulations_run": 16,
                            "exact_endgame": False,
                            "decision_seconds": 1.0,
                        }
                    )
                games.append(
                    {
                        "type": "gumbel_game_done",
                        "ruleset_id": RULESET_ID,
                        "seed": seed,
                        "scores": [score] * 4,
                        "decision_count": 80,
                    }
                )
            payload = "".join(json.dumps(row) + "\n" for row in decisions)
            game_payload = "".join(json.dumps(row) + "\n" for row in games)
            for name in ("baseline_decisions.jsonl", "candidate_decisions.jsonl"):
                paths[name].write_text(payload, encoding="utf-8")
            for name in ("baseline_games.jsonl", "candidate_games.jsonl"):
                paths[name].write_text(game_payload, encoding="utf-8")

            result = build_comparison(
                paths["baseline.json"],
                paths["candidate.json"],
                paths["baseline_decisions.jsonl"],
                paths["candidate_decisions.jsonl"],
                paths["baseline_games.jsonl"],
                paths["candidate_games.jsonl"],
                "tested-revision",
            )
            self.assertTrue(result["comparison"]["policy_parity"])
            self.assertTrue(result["comparison"]["exact_numeric_parity"])
            self.assertEqual(result["timing"]["wall_speedup"], 1.25)
            self.assertTrue(result["performance_gate_pass"])

            candidate_rows = [dict(row) for row in decisions]
            candidate_rows[5]["chosen_action_id"] = "different"
            paths["candidate_decisions.jsonl"].write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
            )
            failed = build_comparison(
                paths["baseline.json"],
                paths["candidate.json"],
                paths["baseline_decisions.jsonl"],
                paths["candidate_decisions.jsonl"],
                paths["baseline_games.jsonl"],
                paths["candidate_games.jsonl"],
            )
            self.assertEqual(failed["comparison"]["action_difference_count"], 1)
            self.assertFalse(failed["performance_gate_pass"])

    def test_cuda_concurrency_comparator_selects_smallest_near_fastest_knee(self) -> None:
        from cascadiav3.compare_cuda_concurrency import RULESET_ID, build_comparison

        seeds = list(range(100, 124))
        score = {
            "wildlife": [10, 10, 10, 10, 10],
            "habitat": [5, 5, 5, 5, 5],
            "nature_tokens": 5,
            "total": 80,
        }

        def report(jobs: int, wall: float, mean_decision: float) -> dict[str, object]:
            return {
                "status": "pass",
                "scientific_eligibility": "candidate_only_search_arm",
                "ruleset_id": RULESET_ID,
                "source_revision": "tested-revision",
                "seeds": seeds,
                "execution": {
                    "runner": "gumbel-benchmark-batch",
                    "batch_runner": True,
                    "requested_jobs": jobs,
                    "seed_count": len(seeds),
                    "parallel_game_cap": jobs,
                    "seed_scheduler": "dynamic_seed_queue",
                    "shared_model_session": True,
                    "bridge_process_topology": "one_shared_bridge",
                    "maximum_concurrent_bridge_processes": 1,
                    "device": "cuda",
                },
                "search": {
                    "n_simulations": 64,
                    "top_m": 16,
                    "determinizations": 4,
                    "blend_weight": 0.5,
                    "parallel_leaf_rollouts": False,
                },
                "artifacts": {
                    "binary_sha256": "binary",
                    "manifest_sha256": "manifest",
                    "weights_sha256": "weights",
                    "checkpoint_tag": "best",
                    "checkpoint_step": 7,
                    "q_quantiles": 8,
                },
                "control": {"kind": "none"},
                "strategies": {
                    "gumbel-search": {
                        "mean_seat_score": 92.0,
                        "mean_total_decision_seconds": mean_decision,
                    }
                },
                "candidate_wall_seconds": wall,
                "candidate_decision_seconds_p50": mean_decision / 2,
                "candidate_decision_seconds_p95": mean_decision * 2,
            }

        decisions = [
            {
                "type": "gumbel_decision",
                "ruleset_id": RULESET_ID,
                "seed": seed,
                "ply": ply,
                "action_count": 16,
                "chosen_action_id": f"action-{seed}-{ply}",
                "free_three_of_a_kind_choice": "not_available",
                "root_value": float(seed + ply),
                "simulations_run": 64,
                "market_branches_searched": 1,
                "market_chance_samples": 0,
                "total_simulations_run": 64,
                "exact_endgame": False,
                "decision_seconds": 1.0,
            }
            for seed in seeds
            for ply in range(80)
        ]
        games = [
            {
                "type": "gumbel_game_done",
                "ruleset_id": RULESET_ID,
                "seed": seed,
                "scores": [score] * 4,
                "decision_count": 80,
            }
            for seed in seeds
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = {12: (120.0, 2.0), 16: (100.0, 2.5), 24: (99.0, 3.5)}
            arms = {}
            profiles = {}
            for jobs, (wall, mean_decision) in specs.items():
                report_path = root / f"jobs{jobs}.json"
                decisions_path = root / f"jobs{jobs}_decisions.jsonl"
                games_path = root / f"jobs{jobs}_games.jsonl"
                report_path.write_text(
                    json.dumps(report(jobs, wall, mean_decision)), encoding="utf-8"
                )
                decisions_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8"
                )
                games_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in games), encoding="utf-8"
                )
                arms[jobs] = (report_path, decisions_path, games_path)
                profile_path = root / f"jobs{jobs}_gpu.csv"
                profile_path.write_text(
                    "".join(f"{jobs}, {300 + jobs}, 2500, 60\n" for _ in range(30)),
                    encoding="utf-8",
                )
                profiles[jobs] = profile_path

            result = build_comparison(
                arms, "tested-revision", gpu_profiles=profiles
            )
            self.assertEqual(result["selection"]["fastest_jobs"], 24)
            self.assertEqual(result["selection"]["recommended_jobs"], 16)
            self.assertTrue(result["selection"]["change_from_jobs12"])
            self.assertEqual(
                result["arms"]["16"]["gpu_profile"]["gpu_utilization_percent"]["mean"],
                16.0,
            )
            self.assertEqual(
                result["arms"]["24"]["comparison_vs_jobs12"]["action_difference_count"],
                0,
            )

            jobs16_rows = [dict(row) for row in decisions]
            jobs16_rows[5]["chosen_action_id"] = "different"
            arms[16][1].write_text(
                "".join(json.dumps(row) + "\n" for row in jobs16_rows), encoding="utf-8"
            )
            without_jobs16 = build_comparison(arms, gpu_profiles=profiles)
            self.assertFalse(
                without_jobs16["arms"]["16"]["comparison_vs_jobs12"][
                    "eligible_for_knee_selection"
                ]
            )
            self.assertEqual(without_jobs16["selection"]["recommended_jobs"], 24)

            arms[16][1].write_text(
                "".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8"
            )
            jobs16_games = [dict(row) for row in games]
            jobs16_games[0]["decision_count"] = 79
            arms[16][2].write_text(
                "".join(json.dumps(row) + "\n" for row in jobs16_games), encoding="utf-8"
            )
            mismatched_count = build_comparison(arms, gpu_profiles=profiles)
            self.assertEqual(
                mismatched_count["arms"]["16"]["comparison_vs_jobs12"][
                    "decision_count_difference_seeds"
                ],
                [seeds[0]],
            )
            self.assertFalse(
                mismatched_count["arms"]["16"]["comparison_vs_jobs12"][
                    "eligible_for_knee_selection"
                ]
            )

            arms[16][2].write_text(
                "".join(json.dumps(row) + "\n" for row in games), encoding="utf-8"
            )
            profiles[24].write_text("nan, 324, 2500, 60\n" * 30, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                build_comparison(arms, gpu_profiles=profiles)

    def test_market_sample_comparator_validates_causal_divergence_and_cost(self) -> None:
        from cascadiav3.compare_market_samples import RULESET_ID, build_comparison

        def report(samples: int, scores: list[float]) -> dict[str, object]:
            return {
                "status": "pass",
                "ruleset_id": RULESET_ID,
                "source_revision": "tested-revision",
                "experiment_id": f"market-samples-{samples}",
                "manifest": "/same-host/best_locked_val.manifest.json",
                "seeds": [1, 2],
                "search": {
                    "n_simulations": 16,
                    "determinizations": 2,
                    "market_decision_samples": samples,
                    "exact_endgame_turns": 0,
                },
                "control": {"kind": "none"},
                "strategies": {
                    "gumbel-search": {
                        "mean_seat_score": sum(scores) / len(scores),
                        "mean_total_decision_seconds": 1.0 + samples / 8.0,
                    }
                },
                "candidate_per_seed": [
                    {"seed": seed, "mean_score_per_seat": score, "seat_scores": [score] * 4}
                    for seed, score in zip((1, 2), scores)
                ],
                "market_decisions": {
                    "mean_chance_samples_when_available": float(samples),
                    "total_simulations_including_market_decision": 2560 + samples * 100,
                    "market_decision_simulation_overhead": samples * 100,
                },
                "candidate_decision_seconds_p50": 1.0,
                "candidate_decision_seconds_p95": 2.0 + samples / 8.0,
                "candidate_wall_seconds": 100.0 + samples,
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_decisions_path = root / "baseline_decisions.jsonl"
            candidate_decisions_path = root / "candidate_decisions.jsonl"
            baseline_path.write_text(json.dumps(report(8, [90.0, 92.0])), encoding="utf-8")
            candidate_path.write_text(json.dumps(report(4, [90.5, 93.0])), encoding="utf-8")
            baseline_rows = []
            candidate_rows = []
            for seed in (1, 2):
                for ply in range(80):
                    choice = "decline" if ply == 10 else "not_available"
                    base = {
                        "type": "gumbel_decision",
                        "ruleset_id": RULESET_ID,
                        "seed": seed,
                        "ply": ply,
                        "chosen_action_id": f"action-{seed}-{ply}",
                        "free_three_of_a_kind_choice": choice,
                        "decision_seconds": 2.0 if ply == 10 else 1.0,
                    }
                    baseline_rows.append(base)
                    candidate_rows.append(
                        base
                        | (
                            {
                                "chosen_action_id": f"sample4-{seed}-{ply}",
                                "free_three_of_a_kind_choice": (
                                    "accept" if ply == 10 else choice
                                ),
                                "decision_seconds": 1.0,
                            }
                            if ply >= 10
                            else {}
                        )
                    )
            baseline_decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_rows), encoding="utf-8"
            )
            candidate_decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
            )
            result = build_comparison(
                baseline_path,
                candidate_path,
                baseline_decisions_path,
                candidate_decisions_path,
                "tested-revision",
            )

            self.assertEqual(result["paired_delta_stats"]["mean"], 0.75)
            self.assertEqual(result["trace"]["causally_changed_seeds"], 2)
            self.assertEqual(result["trace"]["first_exposure_by_seed"], {"1": 10, "2": 10})
            self.assertEqual(result["trace"]["first_divergence_by_seed"], {"1": 10, "2": 10})
            self.assertEqual(result["trace"]["available_decision_speedup"], 2.0)
            self.assertEqual(result["trace"]["available_total_seconds_ratio"], 2.0)
            self.assertEqual(
                result["simulations"]["baseline_market_overhead_per_opportunity"], 400.0
            )
            self.assertEqual(
                result["simulations"]["candidate_market_overhead_per_opportunity"], 200.0
            )
            self.assertFalse(result["performance_gate_pass"])

            # Pre-exposure divergence is expected for this knob (the sample
            # count reaches every ply through simulated refresh nodes) and is
            # classified descriptively rather than failing the comparison.
            candidate_rows[5]["chosen_action_id"] = "unrelated-early-divergence"
            candidate_decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
            )
            reclassified = build_comparison(
                baseline_path,
                candidate_path,
                baseline_decisions_path,
                candidate_decisions_path,
            )
            self.assertEqual(reclassified["trace"]["pre_exposure_divergent_seeds"], 1)
            self.assertEqual(reclassified["trace"]["causally_changed_seeds"], 1)

class GumbelBatchRunnerTest(unittest.TestCase):
    """Contract tests for the --gumbel-benchmark-batch per-seed JSONL path."""

    @staticmethod
    def _canned_seed_lines(seed: int) -> list[dict]:
        return [
            {"type": "gumbel_decision", "seed": seed, "ply": 0, "decision_seconds": 0.5},
            {"type": "gumbel_decision", "seed": seed, "ply": 1, "decision_seconds": 0.7},
            {
                "type": "gumbel_game_done",
                "seed": seed,
                "scores": [
                    {"total": 90 + seed % 7},
                    {"total": 95},
                    {"total": 88},
                    {"total": 92},
                ],
                "decision_count": 2,
                "elapsed_seconds": 3.5,
            },
        ]

    def test_batch_seed_files_produce_identical_results(self) -> None:
        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
            collect_gumbel_results,
            read_batch_seed_lines,
        )

        seeds = [5, 6]
        flat_lines = [line for seed in seeds for line in self._canned_seed_lines(seed)]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for seed in seeds:
                path = output_dir / f"gumbel_game_seed_{seed}.jsonl"
                path.write_text(
                    "".join(json.dumps(line) + "\n" for line in self._canned_seed_lines(seed)),
                    encoding="utf-8",
                )
            batch_lines = read_batch_seed_lines(output_dir, seeds)

        self.assertEqual(batch_lines, flat_lines)
        # Downstream report structures are identical to the per-seed
        # subprocess path.
        self.assertEqual(
            collect_gumbel_results(batch_lines), collect_gumbel_results(flat_lines)
        )

    def test_batch_seed_files_fail_loudly_when_missing_or_incomplete(self) -> None:
        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import read_batch_seed_lines

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            path = output_dir / "gumbel_game_seed_5.jsonl"
            path.write_text(
                "".join(json.dumps(line) + "\n" for line in self._canned_seed_lines(5)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "seed 6"):
                read_batch_seed_lines(output_dir, [5, 6])
            # A decisions-only file (crashed game) must not pass silently.
            truncated = output_dir / "gumbel_game_seed_7.jsonl"
            truncated.write_text(
                json.dumps({"type": "gumbel_decision", "seed": 7, "ply": 0}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "seed 7.*no done record"):
                read_batch_seed_lines(output_dir, [7])

    def test_batch_runner_mock_bridge_integration(self) -> None:
        import sys

        from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
            collect_gumbel_results,
            run_gumbel_games_batch,
            summarize_score_categories,
        )

        binary = Path(
            "cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter"
        )
        if not binary.exists():
            self.skipTest("real-root-exporter release binary has not been built")
        mock_bridge = Path("cascadiav3/tests/mock_model_bridge.py").resolve()
        seeds = [2026070600, 2026070601]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "mock_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            lines = run_gumbel_games_batch(
                binary,
                seeds=seeds,
                model_service=f"{sys.executable} {mock_bridge}",
                model_manifest=manifest,
                output_dir=tmp_path / "batch",
                jobs=2,
                n_simulations=4,
                top_m=2,
                depth_rounds=1,
                determinizations=2,
                market_decision_samples=2,
                exact_endgame_turns=1,
                blend_weight=1.0,
                k_interior=3,
                max_root_actions=None,
                rollout_max_actions=4,
                rollout_top_k=2,
                model_timeout_ms=120_000,
                exploration=False,
                parallel_leaf_rollouts=True,
            )
        results = collect_gumbel_results(lines)
        categories = summarize_score_categories(results)
        self.assertIsNotNone(categories)
        assert categories is not None
        self.assertEqual(categories["overall_mean"]["total"], sum(
            float(score["total"])
            for result in results
            for score in result["done"]["scores"]
        ) / 8)
        self.assertEqual([result["seed"] for result in results], seeds)
        for result in results:
            self.assertEqual(len(result["done"]["scores"]), 4)
            self.assertGreater(len(result["decisions"]), 0)
            self.assertEqual(result["done"]["turns"], len(result["decisions"]))
            exact_decisions = [
                decision for decision in result["decisions"] if decision["exact_endgame"]
            ]
            self.assertEqual(len(exact_decisions), 4)
            self.assertTrue(
                all(decision["total_simulations_run"] == 0 for decision in exact_decisions)
            )
            self.assertEqual(result["done"]["search"]["exact_endgame_turns"], 1)
            self.assertTrue(result["done"]["search"]["parallel_leaf_rollouts"])


class ValidationCliTest(unittest.TestCase):
    def test_run_validation_passes(self) -> None:
        result = run_validation()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["radius6_cell_count"], 127)
        self.assertEqual(result["root_action_count"], 2)
        self.assertEqual(result["replay_action_counts"], [2, 3])


if __name__ == "__main__":
    unittest.main()
