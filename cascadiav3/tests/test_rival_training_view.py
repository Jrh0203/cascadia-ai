"""CPU contract tests for Rival preference sidecars and derived views."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metadata(version: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_id": version,
        "ruleset_id": "test-ruleset",
        "source_revision": "test-revision",
        "mode": "gumbel_selfplay_tensor_corpus",
        "scientific_eligibility": "gumbel_selfplay_expert_iteration",
    }
    if version.endswith((".v3", ".v4")):
        targets = ["improved_policy", "search_root_value", "exact_endgame"]
        if version.endswith(".v4"):
            targets += ["active_seat", "exact_afterstate_score_decomposition_active"]
        metadata.update(
            {
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
                    "manifest": {"sha256": _digest("manifest"), "bytes": 10},
                    "weights": {"sha256": _digest("weights"), "bytes": 20},
                },
                "generator": {"sha256": _digest("generator"), "bytes": 30},
                "created_unix_seconds": 1_700_000_000,
                "canonical_targets": targets,
            }
        )
    return metadata


def _write_expert(path: Path, version: str, *, record_count: int = 1) -> None:
    import numpy as np
    from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard

    v2_plus = version.endswith((".v2", ".v3", ".v4"))
    v3_plus = version.endswith((".v3", ".v4"))
    v4 = version.endswith(".v4")
    action_count = record_count * 2
    _save_expert_tensor_shard(
        out_path=path,
        metadata=_metadata(version),
        tokens=np.zeros((record_count, 41), dtype=np.float16),
        actions=np.zeros((action_count, 61), dtype=np.float16),
        token_offsets=np.arange(record_count + 1, dtype=np.int64),
        action_offsets=np.arange(0, action_count + 1, 2, dtype=np.int64),
        relation_edges=np.zeros((0, 3), dtype=np.int32),
        relation_offsets=np.zeros((record_count + 1,), dtype=np.int64),
        selected_action_index=np.zeros((record_count,), dtype=np.int16),
        target_q=np.tile(np.asarray([2.0, 1.0], dtype=np.float32), record_count),
        target_score_to_go=np.tile(np.asarray([2.0, 1.0], dtype=np.float32), record_count),
        q_valid=np.ones((action_count,), dtype=np.uint8),
        priors=np.tile(np.asarray([0.75, 0.25], dtype=np.float32), record_count),
        visits=np.tile(np.asarray([3.0, 1.0], dtype=np.float32), record_count),
        q_variance=np.full((action_count,), 0.5, dtype=np.float32),
        q_count=np.full((action_count,), 2.0, dtype=np.float32),
        truncated_count=np.zeros((action_count,), dtype=np.float32),
        exact_afterstate_score_active=np.zeros((action_count,), dtype=np.float32),
        exact_afterstate_score_decomposition_active=(
            np.zeros((action_count, 3), dtype=np.float32) if v4 else None
        ),
        active_seat=np.zeros((record_count,), dtype=np.uint8) if v4 else None,
        final_score_vector=np.zeros((record_count, 4), dtype=np.float32),
        rank_vector=np.tile(np.asarray([[1, 2, 3, 4]], dtype=np.int16), (record_count, 1)),
        score_decomposition=np.zeros((record_count, 3, 4), dtype=np.float32),
        improved_policy=(
            np.tile(np.asarray([0.6, 0.4], dtype=np.float32), record_count) if v2_plus else None
        ),
        search_root_value=(np.full((record_count,), 2.0, dtype=np.float32) if v2_plus else None),
        exact_endgame=(np.zeros((record_count,), dtype=np.uint8) if v3_plus else None),
    )


def _rival_digest(label: str) -> str:
    return "sha256:" + _digest(label)


def _namespaced(prefix: str, label: str) -> str:
    return prefix + _digest(label)


def _binding(expert_path: Path, record_index: int = 0) -> dict[str, Any]:
    from cascadiav3.expert_tensor_shards import ExpertTensorShard
    from cascadiav3.rival.training_view import (
        ACTION_CONTENT_PREFIX,
        CANDIDATE_OCCURRENCE_PREFIX,
        INCUMBENT_MENU_PREFIX,
        PUBLIC_ROOT_PREFIX,
        RULES_MENU_PREFIX,
        _tensor_row_sha256,
    )

    expert = ExpertTensorShard(expert_path)
    try:
        action_start = int(expert.action_offsets[record_index])
        action_end = int(expert.action_offsets[record_index + 1])
        tensor_hashes = [
            _tensor_row_sha256(expert.actions[index]) for index in range(action_start, action_end)
        ]
    finally:
        expert.close()
    return {
        "record_index": record_index,
        "public_root_id": _namespaced(PUBLIC_ROOT_PREFIX, f"public-root-{record_index}"),
        "rules_legal_menu_hash": _namespaced(RULES_MENU_PREFIX, f"rules-menu-{record_index}"),
        "incumbent_candidate_menu_hash": _namespaced(
            INCUMBENT_MENU_PREFIX, f"incumbent-menu-{record_index}"
        ),
        "ordered_action_content_ids": [
            _namespaced(ACTION_CONTENT_PREFIX, f"action-{record_index}-0"),
            _namespaced(ACTION_CONTENT_PREFIX, f"action-{record_index}-1"),
        ],
        "candidate_action_occurrence_ids": [
            _namespaced(CANDIDATE_OCCURRENCE_PREFIX, f"occurrence-{record_index}-0"),
            _namespaced(CANDIDATE_OCCURRENCE_PREFIX, f"occurrence-{record_index}-1"),
        ],
        "action_tensor_row_sha256": tensor_hashes,
        "selected_action_index": 0,
    }


def _identity_index(root: Path, expert: Path):  # type: ignore[no-untyped-def]
    from cascadiav3.rival.training_view import (
        ROOT_IDENTITY_INDEX_SCHEMA_ID,
        attach_content_hash,
        file_sha256,
        write_expert_root_identity_index,
    )

    raw_root_ledger = root / "raw-roots.json"
    raw_root_ledger.write_text('{"fixture":"raw-root-ledger"}\n', encoding="utf-8")
    from cascadiav3.expert_tensor_shards import ExpertTensorShard

    shard = ExpertTensorShard(expert)
    try:
        bindings = [_binding(expert, index) for index in range(len(shard))]
    finally:
        shard.close()
    binding = bindings[0]
    payload = attach_content_hash(
        {
            "schema_id": ROOT_IDENTITY_INDEX_SCHEMA_ID,
            "source_revision": "test-revision",
            "expert_shard_sha256": file_sha256(expert),
            "raw_root_ledger_sha256": file_sha256(raw_root_ledger),
            "bindings": bindings,
        }
    )
    index = write_expert_root_identity_index(
        root / "root-identity-index.json",
        payload,
        expert_shard_path=expert,
        raw_root_ledger_path=raw_root_ledger,
    )
    return raw_root_ledger, binding, index


def _write_raw_world_ledger(root: Path) -> Path:
    raw_world_ledger = root / "raw-worlds.json"
    raw_world_ledger.write_text('{"fixture":"raw-world-ledger"}\n', encoding="utf-8")
    return raw_world_ledger


def _payload(
    expert: Path,
    binding: dict[str, Any],
    identity_index: Any,
    raw_world_ledger: Path,
    *,
    valid: bool = True,
    inference_mode: str = "multifidelity",
) -> dict[str, Any]:
    from cascadiav3.expert_tensor_shards import SHARD_VERSION_V4
    from cascadiav3.rival.schema import RIVAL_PREFERENCE_SHARD_SCHEMA_ID
    from cascadiav3.rival.training_view import attach_content_hash, file_sha256

    preference_weight = 2.5
    return attach_content_hash(
        {
            "schema_id": RIVAL_PREFERENCE_SHARD_SCHEMA_ID,
            "source_revision": "test-revision",
            "ruleset_identity_sha256": _rival_digest("ruleset"),
            "expert_shard": {
                "sha256": file_sha256(expert),
                "schema_id": SHARD_VERSION_V4,
                "record_count": len(identity_index.bindings),
            },
            "root_identity_index_sha256": identity_index.content_sha256,
            "incumbent_policy_identity_sha256": _rival_digest("incumbent-policy"),
            "challenger_policy_identity_sha256": _rival_digest("challenger-policy"),
            "coefficient_identity_sha256": (
                _rival_digest("coefficient") if inference_mode == "multifidelity" else None
            ),
            "allocation_identity_sha256": _rival_digest("allocation"),
            "bound_identity_sha256": _rival_digest("bound"),
            "error_ledger_identity_sha256": _rival_digest("error-ledger"),
            "parent_manifest_sha256": _rival_digest("parent-manifest"),
            "raw_root_ledger_sha256": identity_index.raw_root_ledger_sha256,
            "raw_world_ledger_sha256": file_sha256(raw_world_ledger),
            "inference_mode": inference_mode,
            "a_panel_enabled": False,
            "preference_weight": preference_weight,
            "records": [
                {
                    **binding,
                    "incumbent_action_index": 0,
                    "challenger_action_index": 1,
                    "categorical_preference": (
                        "challenger_over_incumbent" if valid else "unlabeled"
                    ),
                    "preference_valid": valid,
                    "preference_weight": preference_weight,
                    "activation_stratum": "final_four_personal_turns",
                    "natural_frequency_weight": 7.0,
                    "sampling_probability": 0.25,
                    "root_cohort_role": "relabel_selection",
                    "panel_identities": {
                        "S": _rival_digest("panel-S"),
                        "H": _rival_digest("panel-H"),
                        "L": (
                            _rival_digest("panel-L") if inference_mode == "multifidelity" else None
                        ),
                        "A": None,
                    },
                    "advantage_target": None,
                    "advantage_valid": False,
                },
            ],
        }
    )


def _write_preference(
    root: Path,
    expert: Path,
    binding: dict[str, Any],
    identity_index: Any,
    raw_world_ledger: Path,
    *,
    valid: bool = True,
    inference_mode: str = "multifidelity",
):  # type: ignore[no-untyped-def]
    from cascadiav3.rival.training_view import write_rival_preference_shard

    return write_rival_preference_shard(
        root / "preferences.json",
        _payload(
            expert,
            binding,
            identity_index,
            raw_world_ledger,
            valid=valid,
            inference_mode=inference_mode,
        ),
        expert_shard_path=expert,
        root_identity_index=identity_index,
        raw_world_ledger_path=raw_world_ledger,
    )


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    from cascadiav3.rival.training_view import attach_content_hash

    content = dict(payload)
    content.pop("content_sha256", None)
    return attach_content_hash(content)


class RivalTrainingViewTest(unittest.TestCase):
    def test_preference_opt_in_requires_an_actual_boolean(self) -> None:
        from cascadiav3.rival import training_view

        legacy_collator = mock.Mock(side_effect=AssertionError("must not delegate"))
        with mock.patch.object(
            training_view,
            "collate_expert_tensor_examples",
            legacy_collator,
        ):
            for invalid in (1, "true", "false", None):
                with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                    training_view.collate_rival_training_examples(
                        [],
                        enable_preferences=invalid,  # type: ignore[arg-type]
                    )
        legacy_collator.assert_not_called()

        for invalid in (1, "false", None):
            with self.subTest(view_invalid=invalid), self.assertRaises(TypeError):
                training_view.RivalTrainingView(
                    Path("must-not-open.npz"),
                    enable_preferences=invalid,  # type: ignore[arg-type]
                )

    def test_training_view_closes_expert_when_post_open_validation_raises(self) -> None:
        from cascadiav3.rival import training_view

        for failure_stage in ("path_resolution", "source_recheck"):
            with self.subTest(failure_stage=failure_stage):
                expert = mock.Mock(version="cascadiav3.expert_tensor_shard.v4")
                if failure_stage == "path_resolution":
                    preference_path = mock.Mock()
                    preference_path.resolve.side_effect = OSError("path resolution failed")
                    source_recheck = mock.Mock()
                    expected_error = "path resolution failed"
                else:
                    preference_path = Path("expert.npz")
                    source_recheck = mock.Mock(side_effect=ValueError("source recheck failed"))
                    expected_error = "source recheck failed"
                preference = types.SimpleNamespace(
                    is_pinned=True,
                    expert_shard_path=preference_path,
                    assert_source_unchanged=source_recheck,
                    expert_schema_id="cascadiav3.expert_tensor_shard.v4",
                )
                with (
                    mock.patch.object(
                        training_view,
                        "ExpertTensorShard",
                        return_value=expert,
                    ),
                    self.assertRaisesRegex((OSError, ValueError), expected_error),
                ):
                    training_view.RivalTrainingView(
                        Path("expert.npz"),
                        preference_shard=preference,
                        enable_preferences=True,
                    )
                expert.close.assert_called_once_with()
                if failure_stage == "path_resolution":
                    source_recheck.assert_not_called()

    def test_default_off_collator_is_exact_legacy_delegation(self) -> None:
        from cascadiav3.rival import training_view

        examples = [{"legacy": "unchanged"}]
        legacy_batch = {"identity": object()}
        with mock.patch.object(
            training_view,
            "collate_expert_tensor_examples",
            return_value=legacy_batch,
        ) as legacy_collator:
            actual = training_view.collate_rival_training_examples(examples)
        self.assertIs(actual, legacy_batch)
        legacy_collator.assert_called_once_with(examples)

    def test_sidecar_round_trip_and_hash_bound_join(self) -> None:
        from cascadiav3.rival.training_view import (
            PREFERENCE_CHALLENGER,
            RivalTrainingView,
            write_rival_preference_shard,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            sidecar = root / "preferences.json"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            preference = write_rival_preference_shard(
                sidecar,
                _payload(expert, binding, identity_index, raw_world_ledger),
                expert_shard_path=expert,
                root_identity_index=identity_index,
                raw_world_ledger_path=raw_world_ledger,
            )
            record = preference.record(0)
            assert record is not None
            self.assertEqual(record.challenger_action_index, 1)
            self.assertEqual(PREFERENCE_CHALLENGER, "challenger_over_incumbent")
            self.assertEqual(record.categorical_preference, PREFERENCE_CHALLENGER)
            self.assertTrue(sidecar.read_bytes().endswith(b"\n"))
            self.assertEqual(sidecar.stat().st_mode & 0o222, 0)
            assert identity_index.path is not None
            self.assertEqual(identity_index.path.stat().st_mode & 0o222, 0)
            canonical_bytes = sidecar.read_bytes()
            with self.assertRaises(FileExistsError):
                write_rival_preference_shard(
                    sidecar,
                    _payload(expert, binding, identity_index, raw_world_ledger),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )
            self.assertEqual(sidecar.read_bytes(), canonical_bytes)

            with RivalTrainingView(
                expert,
                preference_shard=preference,
                enable_preferences=True,
            ) as view:
                example = view.example(0)
                self.assertEqual(example["preference_incumbent_index"], 0)
                self.assertEqual(example["preference_challenger_index"], 1)
                self.assertTrue(example["preference_valid"])
                self.assertEqual(example["policy_weight"], 2.5)
                self.assertEqual(example["natural_frequency_weight"], 7.0)
                self.assertEqual(example["sampling_probability"], 0.25)
                self.assertFalse(example["advantage_valid"])
                self.assertEqual(example["advantage_target"], 0.0)

    def test_collated_policy_weight_is_not_sampling_or_frequency_weight(self) -> None:
        import numpy as np
        from cascadiav3.rival import training_view

        fake_torch = types.SimpleNamespace(
            bool=np.bool_,
            float32=np.float32,
            long=np.int64,
            tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
        )

        with tempfile.TemporaryDirectory() as directory:
            expert = Path(directory) / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            root = Path(directory)
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            preference = _write_preference(
                root,
                expert,
                binding,
                identity_index,
                raw_world_ledger,
            )
            with (
                mock.patch.object(
                    training_view,
                    "collate_expert_tensor_examples",
                    return_value={},
                ),
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                training_view.RivalTrainingView(
                    expert,
                    preference_shard=preference,
                    enable_preferences=True,
                ) as view,
            ):
                batch = view.collate([0])
            self.assertEqual(batch["preference_incumbent_index"].tolist(), [0])
            self.assertEqual(batch["preference_challenger_index"].tolist(), [1])
            self.assertEqual(batch["policy_weight"].tolist(), [2.5])
            self.assertEqual(batch["natural_frequency_weight"].tolist(), [7.0])
            self.assertEqual(batch["sampling_probability"].tolist(), [0.25])
            self.assertEqual(batch["advantage_target"].tolist(), [0.0])
            self.assertEqual(batch["advantage_valid"].tolist(), [False])

    def test_public_collator_rejects_fabricated_or_incoherent_targets(self) -> None:
        from cascadiav3.rival import training_view

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            preference = _write_preference(
                root,
                expert,
                binding,
                identity_index,
                raw_world_ledger,
            )
            with training_view.RivalTrainingView(
                expert,
                preference_shard=preference,
                enable_preferences=True,
            ) as view:
                verified_example = view.example(0)

            corruptions = {
                "non-boolean validity": ({"preference_valid": 1}, "must be boolean"),
                "NaN policy weight": ({"policy_weight": float("nan")}, "must be finite"),
                "negative policy weight": ({"policy_weight": -1.0}, "must be >= 0"),
                "same valid action": (
                    {"preference_challenger_index": 0},
                    "indices must differ",
                ),
                "nonzero invalid target": (
                    {"preference_valid": False},
                    "zero indices and weight",
                ),
                "zero natural weight": (
                    {"natural_frequency_weight": 0.0},
                    "must be positive",
                ),
                "impossible sampling probability": (
                    {"sampling_probability": 1.1},
                    "must be in",
                ),
                "v1 advantage leakage": (
                    {"advantage_valid": True, "advantage_target": 1.0},
                    "requires disabled advantage",
                ),
            }
            for label, (changes, expected_message) in corruptions.items():
                with self.subTest(label=label):
                    corrupted = {**verified_example, **changes}
                    legacy_collator = mock.Mock(
                        side_effect=AssertionError("invalid target reached legacy collator")
                    )
                    with (
                        mock.patch.object(
                            training_view,
                            "collate_expert_tensor_examples",
                            legacy_collator,
                        ),
                        self.assertRaisesRegex(ValueError, expected_message),
                    ):
                        training_view.collate_rival_training_examples(
                            [corrupted],
                            enable_preferences=True,
                        )
                    legacy_collator.assert_not_called()

    def test_unlabeled_record_has_zero_loss_weight(self) -> None:
        from cascadiav3.rival.training_view import RivalTrainingView

        with tempfile.TemporaryDirectory() as directory:
            expert = Path(directory) / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            root = Path(directory)
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            preference = _write_preference(
                root,
                expert,
                binding,
                identity_index,
                raw_world_ledger,
                valid=False,
            )
            with RivalTrainingView(
                expert,
                preference_shard=preference,
                enable_preferences=True,
            ) as view:
                example = view.example(0)
            self.assertFalse(example["preference_valid"])
            self.assertEqual(example["preference_incumbent_index"], 0)
            self.assertEqual(example["preference_challenger_index"], 0)
            self.assertEqual(example["policy_weight"], 0.0)

    def test_rejects_stale_expert_menu_occurrence_and_a_panel_leakage(self) -> None:
        from cascadiav3.rival.training_view import RivalPreferenceShard

        with tempfile.TemporaryDirectory() as directory:
            expert = Path(directory) / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            root = Path(directory)
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            base = _payload(expert, binding, identity_index, raw_world_ledger)

            stale = copy.deepcopy(base)
            stale["expert_shard"]["sha256"] = _rival_digest("stale")
            with self.assertRaisesRegex(ValueError, "expert shard SHA-256 mismatch"):
                RivalPreferenceShard(
                    _reseal(stale),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            wrong_menu = copy.deepcopy(base)
            wrong_menu["records"][0]["incumbent_candidate_menu_hash"] = _namespaced(
                "cascadiav3.rival_incumbent_menu.v1:sha256:", "other-menu"
            )
            with self.assertRaisesRegex(ValueError, "preference identity mismatch"):
                RivalPreferenceShard(
                    _reseal(wrong_menu),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            wrong_occurrence = copy.deepcopy(base)
            wrong_occurrence["records"][0]["candidate_action_occurrence_ids"][1] = _namespaced(
                "cascadiav3.rival_candidate_action_occurrence.v1:sha256:",
                "other-occurrence",
            )
            with self.assertRaisesRegex(ValueError, "preference identity mismatch"):
                RivalPreferenceShard(
                    _reseal(wrong_occurrence),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            leaked_advantage = copy.deepcopy(base)
            leaked_advantage["records"][0]["advantage_valid"] = True
            leaked_advantage["records"][0]["advantage_target"] = 3.0
            with self.assertRaisesRegex(ValueError, "quantitative advantage must be absent"):
                RivalPreferenceShard(
                    _reseal(leaked_advantage),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            enabled_a = copy.deepcopy(base)
            enabled_a["a_panel_enabled"] = True
            enabled_a["records"][0]["panel_identities"]["A"] = _rival_digest("panel-A")
            enabled_a["records"][0]["advantage_valid"] = True
            enabled_a["records"][0]["advantage_target"] = 3.0
            with self.assertRaisesRegex(ValueError, "v1 has A disabled"):
                RivalPreferenceShard(
                    _reseal(enabled_a),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            reused_panel = copy.deepcopy(base)
            reused_panel["records"][0]["panel_identities"]["L"] = reused_panel["records"][0][
                "panel_identities"
            ]["H"]
            with self.assertRaisesRegex(ValueError, "pairwise disjoint"):
                RivalPreferenceShard(
                    _reseal(reused_panel),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            wrong_revision = copy.deepcopy(base)
            wrong_revision["source_revision"] = "different-revision"
            with self.assertRaisesRegex(ValueError, "does not match expert shard"):
                RivalPreferenceShard(
                    _reseal(wrong_revision),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

    def test_panel_identities_cannot_be_reused_across_roots(self) -> None:
        from cascadiav3.rival.training_view import RivalPreferenceShard

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(
                expert,
                "cascadiav3.expert_tensor_shard.v4",
                record_count=2,
            )
            _, first_binding, identity_index = _identity_index(root, expert)
            second_binding = _binding(expert, 1)
            raw_world_ledger = _write_raw_world_ledger(root)
            base = _payload(
                expert,
                first_binding,
                identity_index,
                raw_world_ledger,
            )
            second_record = {**copy.deepcopy(base["records"][0]), **second_binding}

            valid_two_root_payload = copy.deepcopy(base)
            unique_second = copy.deepcopy(second_record)
            unique_second["panel_identities"] = {
                "S": _rival_digest("panel-S-root-1"),
                "H": _rival_digest("panel-H-root-1"),
                "L": _rival_digest("panel-L-root-1"),
                "A": None,
            }
            valid_two_root_payload["records"].append(unique_second)
            valid_shard = RivalPreferenceShard(
                _reseal(valid_two_root_payload),
                expert_shard_path=expert,
                root_identity_index=identity_index,
                raw_world_ledger_path=raw_world_ledger,
            )
            self.assertIsNotNone(valid_shard.record(1))

            reused_panel_payload = copy.deepcopy(base)
            reused_panel_payload["records"].append(second_record)
            with self.assertRaisesRegex(ValueError, "reused across preference records"):
                RivalPreferenceShard(
                    _reseal(reused_panel_payload),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

    def test_source_mutation_after_validation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expert = Path(directory) / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            root = Path(directory)
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            preference = _write_preference(
                root,
                expert,
                binding,
                identity_index,
                raw_world_ledger,
            )
            with expert.open("ab") as handle:
                handle.write(b"source-mutation")
            with self.assertRaisesRegex(ValueError, "changed after preference join"):
                preference.assert_source_unchanged()

    def test_root_index_is_immutable_pinned_and_detects_source_mutation(self) -> None:
        from cascadiav3.rival.training_view import (
            ExpertRootIdentityIndex,
            write_expert_root_identity_index,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            raw_root_ledger, _, identity_index = _identity_index(root, expert)
            assert identity_index.path is not None
            serialized = identity_index.path.read_bytes()
            payload = json.loads(serialized)

            with self.assertRaises(FileExistsError):
                write_expert_root_identity_index(
                    identity_index.path,
                    payload,
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                )
            self.assertEqual(identity_index.path.read_bytes(), serialized)

            with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
                ExpertRootIdentityIndex.load(
                    identity_index.path,
                    expected_file_sha256=_rival_digest("wrong-file"),
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                )

            raw_root_ledger.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw root ledger changed"):
                identity_index.assert_sources_unchanged()

    def test_persisted_index_and_sidecar_require_pinned_canonical_single_link_files(self) -> None:
        from cascadiav3.rival.training_view import (
            ExpertRootIdentityIndex,
            RivalPreferenceShard,
            file_sha256,
        )

        for artifact_kind in ("index", "sidecar"):
            with (
                self.subTest(artifact_kind=artifact_kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                expert = root / "expert-v4.npz"
                _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
                raw_root_ledger, binding, identity_index = _identity_index(root, expert)
                raw_world_ledger = _write_raw_world_ledger(root)
                preference = _write_preference(
                    root,
                    expert,
                    binding,
                    identity_index,
                    raw_world_ledger,
                )

                if artifact_kind == "index":
                    target = identity_index.path
                    expected_file_sha256 = identity_index.index_file_sha256
                else:
                    target = preference.path
                    expected_file_sha256 = preference.preference_file_sha256
                assert target is not None and expected_file_sha256 is not None

                symlink = root / f"{artifact_kind}-symlink.json"
                symlink.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "could not safely open"):
                    if artifact_kind == "index":
                        ExpertRootIdentityIndex.load(
                            symlink,
                            expected_file_sha256=expected_file_sha256,
                            expert_shard_path=expert,
                            raw_root_ledger_path=raw_root_ledger,
                        )
                    else:
                        RivalPreferenceShard.load(
                            symlink,
                            expected_file_sha256=expected_file_sha256,
                            expert_shard_path=expert,
                            root_identity_index=identity_index,
                            raw_world_ledger_path=raw_world_ledger,
                        )

                hardlink = root / f"{artifact_kind}-hardlink.json"
                os.link(target, hardlink)
                try:
                    with self.assertRaisesRegex(ValueError, "single-link regular file"):
                        if artifact_kind == "index":
                            ExpertRootIdentityIndex.load(
                                hardlink,
                                expected_file_sha256=expected_file_sha256,
                                expert_shard_path=expert,
                                raw_root_ledger_path=raw_root_ledger,
                            )
                        else:
                            RivalPreferenceShard.load(
                                hardlink,
                                expected_file_sha256=expected_file_sha256,
                                expert_shard_path=expert,
                                root_identity_index=identity_index,
                                raw_world_ledger_path=raw_world_ledger,
                            )
                finally:
                    hardlink.unlink()

                decoded = json.loads(target.read_bytes())
                noncanonical = root / f"{artifact_kind}-noncanonical.json"
                noncanonical.write_text(
                    json.dumps(decoded, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "must use canonical JSON"):
                    if artifact_kind == "index":
                        ExpertRootIdentityIndex.load(
                            noncanonical,
                            expected_file_sha256=file_sha256(noncanonical),
                            expert_shard_path=expert,
                            raw_root_ledger_path=raw_root_ledger,
                        )
                    else:
                        RivalPreferenceShard.load(
                            noncanonical,
                            expected_file_sha256=file_sha256(noncanonical),
                            expert_shard_path=expert,
                            root_identity_index=identity_index,
                            raw_world_ledger_path=raw_world_ledger,
                        )

                target.chmod(stat.S_IRUSR | stat.S_IWUSR)
                target.write_bytes(target.read_bytes() + b" ")
                expected_change = (
                    "root identity index changed"
                    if artifact_kind == "index"
                    else "preference shard changed"
                )
                with self.assertRaisesRegex(ValueError, expected_change):
                    if artifact_kind == "index":
                        identity_index.assert_sources_unchanged()
                    else:
                        preference.assert_source_unchanged()

    def test_writers_pin_expected_bytes_before_publication(self) -> None:
        from cascadiav3.rival import training_view

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            raw_root_ledger, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            assert identity_index.path is not None
            index_payload = json.loads(identity_index.path.read_text(encoding="utf-8"))

            def publish_substituted_bytes(destination: Path, data: bytes) -> None:
                destination.write_bytes(data + b" ")

            with (
                mock.patch.object(
                    training_view,
                    "_write_immutable_bytes",
                    side_effect=publish_substituted_bytes,
                ),
                self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"),
            ):
                training_view.write_expert_root_identity_index(
                    root / "substituted-index.json",
                    index_payload,
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                )

            with (
                mock.patch.object(
                    training_view,
                    "_write_immutable_bytes",
                    side_effect=publish_substituted_bytes,
                ),
                self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"),
            ):
                training_view.write_rival_preference_shard(
                    root / "substituted-preferences.json",
                    _payload(expert, binding, identity_index, raw_world_ledger),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

    def test_pinned_path_cannot_authenticate_a_substituted_payload(self) -> None:
        from cascadiav3.rival.training_view import (
            PUBLIC_ROOT_PREFIX,
            ExpertRootIdentityIndex,
            RivalPreferenceShard,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            raw_root_ledger, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            preference = _write_preference(
                root,
                expert,
                binding,
                identity_index,
                raw_world_ledger,
            )
            assert identity_index.path is not None
            assert identity_index.index_file_sha256 is not None
            index_substitution = json.loads(identity_index.path.read_text(encoding="utf-8"))
            index_substitution["bindings"][0]["public_root_id"] = _namespaced(
                PUBLIC_ROOT_PREFIX,
                "different-valid-root",
            )
            with self.assertRaisesRegex(ValueError, "payload does not match"):
                ExpertRootIdentityIndex(
                    _reseal(index_substitution),
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                    path=identity_index.path,
                    expected_file_sha256=identity_index.index_file_sha256,
                )

            assert preference.path is not None
            assert preference.preference_file_sha256 is not None
            preference_substitution = json.loads(preference.path.read_text(encoding="utf-8"))
            preference_substitution["challenger_policy_identity_sha256"] = _rival_digest(
                "different-valid-challenger"
            )
            with self.assertRaisesRegex(ValueError, "payload does not match"):
                RivalPreferenceShard(
                    _reseal(preference_substitution),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                    path=preference.path,
                    expected_file_sha256=preference.preference_file_sha256,
                )

    def test_pinned_json_rejects_duplicate_keys(self) -> None:
        from cascadiav3.rival.training_view import ExpertRootIdentityIndex

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            raw_root_ledger, _, _ = _identity_index(root, expert)
            duplicate = root / "duplicate-index.json"
            duplicate_bytes = b'{"schema_id":"first","schema_id":"second"}\n'
            duplicate.write_bytes(duplicate_bytes)
            expected_hash = "sha256:" + hashlib.sha256(duplicate_bytes).hexdigest()
            with self.assertRaisesRegex(ValueError, "duplicate JSON key 'schema_id'"):
                ExpertRootIdentityIndex.load(
                    duplicate,
                    expected_file_sha256=expected_hash,
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                )

    def test_root_index_rejects_modified_tensor_row_and_selected_action(self) -> None:
        from cascadiav3.rival.training_view import ExpertRootIdentityIndex

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            raw_root_ledger, _, identity_index = _identity_index(root, expert)
            assert identity_index.path is not None
            base = json.loads(identity_index.path.read_text(encoding="utf-8"))

            wrong_row = copy.deepcopy(base)
            wrong_row["bindings"][0]["action_tensor_row_sha256"][0] = _rival_digest(
                "substituted-row"
            )
            with self.assertRaisesRegex(ValueError, "action tensor mismatch"):
                ExpertRootIdentityIndex(
                    _reseal(wrong_row),
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                )

            wrong_selected = copy.deepcopy(base)
            wrong_selected["bindings"][0]["selected_action_index"] = 1
            with self.assertRaisesRegex(ValueError, "selected action mismatch"):
                ExpertRootIdentityIndex(
                    _reseal(wrong_selected),
                    expert_shard_path=expert,
                    raw_root_ledger_path=raw_root_ledger,
                )

    def test_sidecar_and_world_ledger_mutations_are_detected(self) -> None:
        for mutated_source in ("sidecar", "world"):
            with (
                self.subTest(mutated_source=mutated_source),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                expert = root / "expert-v4.npz"
                _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
                _, binding, identity_index = _identity_index(root, expert)
                raw_world_ledger = _write_raw_world_ledger(root)
                preference = _write_preference(
                    root,
                    expert,
                    binding,
                    identity_index,
                    raw_world_ledger,
                )
                if mutated_source == "sidecar":
                    assert preference.path is not None
                    preference.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                    with preference.path.open("ab") as handle:
                        handle.write(b" ")
                    expected_message = "preference shard changed"
                else:
                    raw_world_ledger.write_text("mutated\n", encoding="utf-8")
                    expected_message = "raw world ledger changed"
                with self.assertRaisesRegex(ValueError, expected_message):
                    preference.assert_source_unchanged()

    def test_selected_incumbent_mismatch_is_rejected(self) -> None:
        from cascadiav3.rival.training_view import RivalPreferenceShard

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            payload = _payload(expert, binding, identity_index, raw_world_ledger)
            payload["records"][0]["incumbent_action_index"] = 1
            payload["records"][0]["challenger_action_index"] = 0
            with self.assertRaisesRegex(ValueError, "not the expert-selected action"):
                RivalPreferenceShard(
                    _reseal(payload),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

    def test_rust_identity_namespaces_are_not_substitutable(self) -> None:
        from cascadiav3.rival.training_view import ExpertRootBinding

        with tempfile.TemporaryDirectory() as directory:
            expert = Path(directory) / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            base = _binding(expert)
            substitutions = {
                "public_root_id": base["ordered_action_content_ids"][0],
                "rules_legal_menu_hash": base["incumbent_candidate_menu_hash"],
                "incumbent_candidate_menu_hash": base["rules_legal_menu_hash"],
                "ordered_action_content_ids": [
                    base["candidate_action_occurrence_ids"][0],
                    base["candidate_action_occurrence_ids"][1],
                ],
                "candidate_action_occurrence_ids": [
                    base["ordered_action_content_ids"][0],
                    base["ordered_action_content_ids"][1],
                ],
            }
            for field, substitution in substitutions.items():
                with self.subTest(field=field):
                    mutated = copy.deepcopy(base)
                    mutated[field] = substitution
                    with self.assertRaisesRegex(ValueError, "exact Rust namespace"):
                        ExpertRootBinding.from_mapping(mutated)

    def test_high_fidelity_only_mode_has_no_coefficient_or_l_panel(self) -> None:
        from cascadiav3.rival.training_view import RivalPreferenceShard

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            _, binding, identity_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            payload = _payload(
                expert,
                binding,
                identity_index,
                raw_world_ledger,
                inference_mode="high_fidelity_only",
            )
            preference = _write_preference(
                root,
                expert,
                binding,
                identity_index,
                raw_world_ledger,
                inference_mode="high_fidelity_only",
            )
            self.assertIsNone(preference.coefficient_identity_sha256)
            record = preference.record(0)
            assert record is not None
            self.assertIsNone(record.panel_identities["L"])

            bad_coefficient = copy.deepcopy(payload)
            bad_coefficient["coefficient_identity_sha256"] = _rival_digest("unexpected-coefficient")
            with self.assertRaisesRegex(ValueError, "coefficient identity null"):
                RivalPreferenceShard(
                    _reseal(bad_coefficient),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            bad_l_panel = copy.deepcopy(payload)
            bad_l_panel["records"][0]["panel_identities"]["L"] = _rival_digest("unexpected-L")
            with self.assertRaisesRegex(ValueError, "require L=null"):
                RivalPreferenceShard(
                    _reseal(bad_l_panel),
                    expert_shard_path=expert,
                    root_identity_index=identity_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

    def test_ephemeral_identity_capabilities_cannot_reach_training(self) -> None:
        from cascadiav3.rival.training_view import (
            ExpertRootIdentityIndex,
            RivalPreferenceShard,
            RivalTrainingView,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / "expert-v4.npz"
            _write_expert(expert, "cascadiav3.expert_tensor_shard.v4")
            raw_root_ledger, binding, pinned_index = _identity_index(root, expert)
            raw_world_ledger = _write_raw_world_ledger(root)
            assert pinned_index.path is not None
            index_payload = json.loads(pinned_index.path.read_text(encoding="utf-8"))
            ephemeral_index = ExpertRootIdentityIndex(
                index_payload,
                expert_shard_path=expert,
                raw_root_ledger_path=raw_root_ledger,
            )
            payload = _payload(expert, binding, pinned_index, raw_world_ledger)
            with self.assertRaisesRegex(ValueError, "loaded from a pinned file"):
                RivalPreferenceShard(
                    payload,
                    expert_shard_path=expert,
                    root_identity_index=ephemeral_index,
                    raw_world_ledger_path=raw_world_ledger,
                )

            ephemeral_preference = RivalPreferenceShard(
                payload,
                expert_shard_path=expert,
                root_identity_index=pinned_index,
                raw_world_ledger_path=raw_world_ledger,
            )
            with self.assertRaisesRegex(ValueError, "loaded from a pinned file"):
                RivalTrainingView(
                    expert,
                    preference_shard=ephemeral_preference,
                    enable_preferences=True,
                )

    def test_default_off_preserves_v1_through_v4_examples(self) -> None:
        import numpy as np
        from cascadiav3.expert_tensor_shards import (
            SHARD_VERSION,
            SHARD_VERSION_V2,
            SHARD_VERSION_V3,
            SHARD_VERSION_V4,
            ExpertTensorShard,
        )
        from cascadiav3.rival.training_view import RivalTrainingView

        versions = (SHARD_VERSION, SHARD_VERSION_V2, SHARD_VERSION_V3, SHARD_VERSION_V4)
        with tempfile.TemporaryDirectory() as directory:
            for index, version in enumerate(versions):
                with self.subTest(version=version):
                    expert_path = Path(directory) / f"expert-{index}.npz"
                    _write_expert(expert_path, version)
                    source = ExpertTensorShard(expert_path)
                    expected = source.example(0)
                    source.close()
                    with RivalTrainingView(expert_path) as view:
                        self.assertEqual(view.expert.version, version)
                        actual = view.example(0)
                    self.assertEqual(set(actual), set(expected))
                    self.assertNotIn("preference_target", actual)
                    for key in expected:
                        if hasattr(expected[key], "shape"):
                            self.assertTrue(np.array_equal(actual[key], expected[key]), key)
                        else:
                            self.assertEqual(actual[key], expected[key], key)


if __name__ == "__main__":
    unittest.main()
