"""Fail-closed tests for additive Rival schemas and frozen manifests."""

import hashlib
import json
import math
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.rival.manifest import (
    ACTION_CONTENT_ID_PREFIX,
    CANDIDATE_OCCURRENCE_ID_PREFIX,
    PUBLIC_ROOT_ID_PREFIX,
    CandidateSelectionEntry,
    candidate_set_identity,
    deployment_design_identity,
    load_root_manifest,
    require_externally_pinned_root_manifest,
    require_validated_root_manifest,
    validate_policy_identity,
    validate_root_manifest,
)
from cascadiav3.rival.schema import (
    MAX_RIVAL_JSON_BYTES,
    RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
    RIVAL_BOUND_CERTIFICATE_SCHEMA_ID,
    RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
    RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
    RIVAL_GPU_PERMIT_SCHEMA_ID,
    RIVAL_POLICY_IDENTITY_SCHEMA_ID,
    RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
    RIVAL_POWER_ENVELOPE_SCHEMA_ID,
    RIVAL_PREFERENCE_SHARD_SCHEMA_ID,
    RIVAL_ROOT_MANIFEST_SCHEMA_ID,
    RIVAL_SCHEMA_DEFINITIONS,
    RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID,
    RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID,
    RIVAL_TRAINING_VIEW_SCHEMA_ID,
    RivalSchemaError,
    attach_content_hash,
    canonical_json_bytes,
    read_pinned_canonical_json_object,
    read_strict_json_object,
)
from cascadiav3.schema import RIVAL_SCHEMA_IDS, SCHEMA_REGISTRY

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rival"


def policy_record(kind: str = "B_k") -> dict[str, object]:
    """Return the exact nested wire emitted by the Rust policy identity types."""
    return {
        "schema_id": RIVAL_POLICY_IDENTITY_SCHEMA_ID,
        "policy_kind": kind,
        "fields": {
            "ruleset": {
                "schema_id": "cascadiav3.research_ruleset_identity.v1",
                "legacy_ruleset_id": (
                    "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
                ),
                "rules_semantics_id": "cascadia-base-official-2026-07-16",
                "game_config_sha256": (
                    "sha256:f5b2c782a483db870c50366b33cccde6d9a82a92a571cf9f29c752b750a5c07c"
                ),
            },
            "source_revision": "0123456789abcdef",
            "source_digest": SHA_A,
            "executable_sha256": SHA_A,
            "model_manifest_sha256": SHA_A,
            "checkpoint_sha256": SHA_A,
            "weights_sha256": SHA_A,
            "bridge_protocol": "bridge.v1",
            "tensor_schema": "tensor.v4",
            "numerical_mode": "deterministic",
            "precision": "fp32",
            "gumbel_config_sha256": SHA_A,
            "search_config_sha256": SHA_A,
            "refresh_config_sha256": SHA_A,
            "exact_endgame_config_sha256": SHA_A,
            "action_content_id_version": "cascadiav3.rival_action_content.v1",
            "rules_action_occurrence_id_version": ("cascadiav3.rival_root_action_occurrence.v1"),
            "candidate_action_occurrence_id_version": (
                "cascadiav3.rival_candidate_action_occurrence.v1"
            ),
            "rules_menu_hash_version": "cascadiav3.rival_rules_menu.v1",
            "incumbent_menu_hash_version": "cascadiav3.rival_incumbent_menu.v1",
            "rng_contracts": {
                "physical": "cascadiav3.rival_rng_domains.v1",
                "policy": "cascadiav3.rival_rng_domains.v1",
                "redetermination": "cascadiav3.rival_rng_domains.v1",
                "search": "cascadiav3.rival_rng_domains.v1",
                "tie_break": "cascadiav3.rival_rng_domains.v1",
            },
            "public_observation_schema": ("cascadiav3.rival_public_policy_observation.v1"),
            "policy_memory_schema": "cascadiav3.rival_seat_local_memory.v1",
            "failure_behavior": {
                "timeout": "record_incomplete_no_label",
                "incomplete_unit": "record_incomplete_no_label",
                "oom": "record_incomplete_no_label",
                "fallback": "forbidden",
            },
            "compiler_identity": "compiler.v1",
            "simulator_identity": "simulator.v1",
            "sampler_identity": "sampler.v1",
            "candidate_generator_identity": "candidate.v1",
            "forbidden_capabilities": {
                "table_total_utility": False,
                "table_native_q": False,
                "true_hidden_peeking": False,
                "model_fallback": False,
            },
        },
    }


def leaf_paths(value: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if not isinstance(value, dict):
        return [prefix]
    return [path for key, child in value.items() for path in leaf_paths(child, (*prefix, key))]


def remove_path(value: dict[str, object], path: tuple[str, ...]) -> None:
    cursor: dict[str, object] = value
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    del cursor[path[-1]]


def root_manifest() -> dict[str, object]:
    candidates = (
        CandidateSelectionEntry(
            CANDIDATE_OCCURRENCE_ID_PREFIX + "a" * 64,
            ACTION_CONTENT_ID_PREFIX + "a" * 64,
            0,
        ),
        CandidateSelectionEntry(
            CANDIDATE_OCCURRENCE_ID_PREFIX + "b" * 64,
            ACTION_CONTENT_ID_PREFIX + "b" * 64,
            4,
        ),
    )
    record = {
        "schema_id": RIVAL_ROOT_MANIFEST_SCHEMA_ID,
        "manifest_id": "manifest:1",
        "ruleset_identity": SHA_A,
        "source_revision": "abc123",
        "root_id": PUBLIC_ROOT_ID_PREFIX + "a" * 64,
        "source_game_id": "game:1",
        "source_game_identity_sha256": SHA_A,
        "root_kind": "draft_policy_root",
        "root_cohort_role": "untouched_coverage",
        "complete_game_seed_role": None,
        "inference_mode": "multifidelity",
        "required_panels": ["S", "H", "L"],
        "forbidden_panels": ["A"],
        "panel_identities": {
            "S": "sha256:" + "1" * 64,
            "H": "sha256:" + "2" * 64,
            "L": "sha256:" + "3" * 64,
            "A": None,
        },
        "beta_cv": 0.5,
        "multifidelity_claim": True,
        "incumbent_policy_identity": SHA_A,
        "incumbent_action_id": ("cascadiav3.rival_action_content.v1:sha256:" + "a" * 64),
        "incumbent_candidate_occurrence_id": (CANDIDATE_OCCURRENCE_ID_PREFIX + "a" * 64),
        "rules_menu_hash": "cascadiav3.rival_rules_menu.v1:sha256:" + "a" * 64,
        "incumbent_menu_hash": ("cascadiav3.rival_incumbent_menu.v1:sha256:" + "a" * 64),
        "low_policy_identity": SHA_A,
        "candidate_set_identity": candidate_set_identity(candidates),
        "candidate_selection_entries": [
            {
                "candidate_action_occurrence_id": row.candidate_action_occurrence_id,
                "action_content_id": row.action_content_id,
                "expected_s": row.expected_s,
            }
            for row in candidates
        ],
        "sampler_identity": SHA_A,
        "policy_rng_factory_identity": SHA_A,
        "terminal_verifier_executable_sha256": SHA_A,
        "terminal_verifier_contract_id": "cascadia-rival.verify-terminal-pair.v1",
        "coefficient_identity": SHA_A,
        "allocation_identity": SHA_A,
        "bound_certificate_identity": SHA_A,
        "error_ledger_identity": SHA_A,
        "expected_s": 4,
        "expected_h": 8,
        "expected_l": 16,
        "practical_margin": 0.25,
        "preference_weight": 2.0,
        "selection_rule": "highest_mean_then_lexicographic_action_id",
        "low_expectation_id": "expectation:1",
        "low_law_h_id": "law:1",
        "low_law_l_id": "law:1",
        "max_abs_beta": 2.0,
        "a_panel_enabled": False,
        "quantitative_target_enabled": False,
    }
    record["deployment_design_sha256"] = deployment_design_identity(record)
    return attach_content_hash(record)


class RivalSchemaTest(unittest.TestCase):
    def test_strict_json_reader_rejects_duplicate_and_nonfinite_tokens(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            for payload, reason in (
                ('{"a":1,"a":2}\n', "duplicate JSON key"),
                ('{"value":NaN}\n', "non-finite JSON constant"),
                ("[1,2,3]\n", "must contain one JSON object"),
            ):
                path.write_text(payload, encoding="utf-8")
                with (
                    self.subTest(payload=payload),
                    self.assertRaisesRegex(RivalSchemaError, reason),
                ):
                    read_strict_json_object(path)

    def test_generic_json_readers_reject_files_over_64_mib_before_reading(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pinned in (False, True):
                path = root / f"oversized-{pinned}.json"
                with path.open("wb") as handle:
                    handle.truncate(MAX_RIVAL_JSON_BYTES + 1)
                with (
                    self.subTest(pinned=pinned),
                    self.assertRaisesRegex(RivalSchemaError, "exceeds maximum JSON artifact size"),
                ):
                    if pinned:
                        read_pinned_canonical_json_object(
                            path,
                            expected_file_sha256="0" * 64,
                        )
                    else:
                        read_strict_json_object(path)

    def test_all_namespaced_schema_ids_are_registered(self) -> None:
        expected = {
            RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
            RIVAL_POLICY_IDENTITY_SCHEMA_ID,
            RIVAL_ROOT_MANIFEST_SCHEMA_ID,
            RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID,
            RIVAL_BOUND_CERTIFICATE_SCHEMA_ID,
            RIVAL_POWER_ENVELOPE_SCHEMA_ID,
            RIVAL_GPU_PERMIT_SCHEMA_ID,
            RIVAL_PREFERENCE_SHARD_SCHEMA_ID,
            RIVAL_TRAINING_VIEW_SCHEMA_ID,
            RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID,
            RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
            RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
            RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
        }
        self.assertEqual(set(RIVAL_SCHEMA_DEFINITIONS), expected)
        self.assertEqual(set(RIVAL_SCHEMA_IDS), expected)
        self.assertTrue(expected <= set(SCHEMA_REGISTRY))
        self.assertTrue(all(schema.startswith("cascadiav3.rival_") for schema in expected))

    def test_canonical_json_refuses_nan(self) -> None:
        with self.assertRaisesRegex(RivalSchemaError, "canonical-JSON"):
            canonical_json_bytes({"bad": math.nan})

    def test_policy_kinds_are_machine_incompatible(self) -> None:
        for kind in ("B_k", "pi_L", "W_k", "M_(k+1)"):
            self.assertEqual(validate_policy_identity(policy_record(kind)).policy_kind, kind)
        invalid = policy_record("generic_policy")
        with self.assertRaisesRegex(RivalSchemaError, "policy_kind"):
            validate_policy_identity(invalid)
        with self.assertRaisesRegex(RivalSchemaError, "substitution"):
            validate_policy_identity(policy_record("pi_L"), expected_policy_kind="B_k")

    def test_python_consumes_rust_emitted_policy_identity_and_digest_golden(self) -> None:
        emitted = json.loads(
            (FIXTURE_DIR / "policy_identity_bk_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(emitted, policy_record())
        parsed = validate_policy_identity(emitted, expected_policy_kind="B_k")
        self.assertEqual(
            parsed.identity_sha256,
            "sha256:323838bfcbd94446f958f90cc268cef6cfaa9806d58ba429492937727b39fbf1",
        )

    def test_every_policy_identity_field_is_hash_bound_and_required(self) -> None:
        original = policy_record()
        paths = leaf_paths(original)
        self.assertGreaterEqual(len(paths), 40)
        for path in paths:
            missing = deepcopy(original)
            remove_path(missing, path)
            with self.subTest(path=".".join(path)), self.assertRaises(RivalSchemaError):
                validate_policy_identity(missing)

        original_hash = validate_policy_identity(original).identity_sha256
        mutated = deepcopy(original)
        fields = mutated["fields"]
        assert isinstance(fields, dict)
        fields["sampler_identity"] = "sampler.v2"
        self.assertNotEqual(
            validate_policy_identity(mutated).identity_sha256,
            original_hash,
        )
        extra = deepcopy(original)
        extra["table_total"] = 0
        with self.assertRaisesRegex(RivalSchemaError, "unknown"):
            validate_policy_identity(extra)

    def test_policy_identity_rejects_non_rust_rules_and_unqualified_hashes(self) -> None:
        changed_rules = policy_record()
        fields = changed_rules["fields"]
        assert isinstance(fields, dict)
        ruleset = fields["ruleset"]
        assert isinstance(ruleset, dict)
        ruleset["game_config_sha256"] = SHA_B
        with self.assertRaisesRegex(RivalSchemaError, "canonical Rust-authored"):
            validate_policy_identity(changed_rules)

        unqualified = policy_record()
        fields = unqualified["fields"]
        assert isinstance(fields, dict)
        fields["weights_sha256"] = "a" * 64
        with self.assertRaisesRegex(RivalSchemaError, "Rust 'sha256:' wire"):
            validate_policy_identity(unqualified)

        forbidden = policy_record()
        fields = forbidden["fields"]
        assert isinstance(fields, dict)
        capabilities = fields["forbidden_capabilities"]
        assert isinstance(capabilities, dict)
        capabilities["true_hidden_peeking"] = True
        with self.assertRaisesRegex(RivalSchemaError, "forbidden capability"):
            validate_policy_identity(forbidden)

    def test_root_manifest_separates_root_and_complete_game_axes(self) -> None:
        parsed = validate_root_manifest(root_manifest())
        self.assertEqual(parsed.root_cohort_role, "untouched_coverage")
        bad = root_manifest()
        bad.pop("content_sha256")
        bad["complete_game_seed_role"] = "promotion"
        bad = attach_content_hash(bad)
        with self.assertRaisesRegex(RivalSchemaError, "not root cohorts"):
            validate_root_manifest(bad)

    def test_root_manifest_capability_rejects_direct_and_mutated_dataclasses(self) -> None:
        validated = validate_root_manifest(root_manifest())
        self.assertTrue(validated.validated)
        self.assertFalse(validated.externally_pinned)
        with self.assertRaisesRegex(RivalSchemaError, "externally byte-pinned"):
            require_externally_pinned_root_manifest(validated)

        direct = replace(validated, _validation_capability=None)
        self.assertFalse(direct.validated)
        with self.assertRaisesRegex(RivalSchemaError, "artifact validator"):
            require_validated_root_manifest(direct)

        with self.assertRaisesRegex(RivalSchemaError, "does not match"):
            replace(validated, practical_margin=validated.practical_margin + 1.0)

    def test_root_manifest_loader_requires_external_byte_and_content_pins(self) -> None:
        record = root_manifest()
        canonical = canonical_json_bytes(record) + b"\n"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_bytes(canonical)
            loaded = load_root_manifest(
                path,
                expected_file_sha256=hashlib.sha256(canonical).hexdigest(),
                expected_content_sha256=record["content_sha256"],
            )
            self.assertTrue(loaded.validated)
            self.assertTrue(loaded.externally_pinned)
            require_externally_pinned_root_manifest(loaded)

            with self.assertRaisesRegex(RivalSchemaError, "does not match"):
                replace(loaded, manifest_id="manifest:mutated-after-load")

            stripped = replace(
                loaded,
                _artifact_file_sha256=None,
                _external_pin_capability=None,
            )
            self.assertTrue(stripped.validated)
            self.assertFalse(stripped.externally_pinned)

            with self.assertRaisesRegex(RivalSchemaError, "file SHA-256"):
                load_root_manifest(
                    path,
                    expected_file_sha256="f" * 64,
                    expected_content_sha256=record["content_sha256"],
                )

            rewritten = deepcopy(record)
            rewritten.pop("content_sha256")
            rewritten["manifest_id"] = "manifest:rewritten"
            rewritten = attach_content_hash(rewritten)
            rewritten_bytes = canonical_json_bytes(rewritten) + b"\n"
            rewritten_path = Path(temporary) / "rewritten.json"
            rewritten_path.write_bytes(rewritten_bytes)
            with self.assertRaisesRegex(RivalSchemaError, "externally preregistered"):
                load_root_manifest(
                    rewritten_path,
                    expected_file_sha256=hashlib.sha256(rewritten_bytes).hexdigest(),
                    expected_content_sha256=record["content_sha256"],
                )

    def test_candidate_allocation_and_deployment_digest_are_not_opaque(self) -> None:
        original = root_manifest()
        changed = deepcopy(original)
        changed.pop("content_sha256")
        entries = changed["candidate_selection_entries"]
        assert isinstance(entries, list)
        entry = entries[1]
        assert isinstance(entry, dict)
        entry["expected_s"] = 3
        with self.assertRaisesRegex(RivalSchemaError, "expected_s"):
            validate_root_manifest(attach_content_hash(changed))

        changed = deepcopy(original)
        changed.pop("content_sha256")
        changed["expected_h"] = 9
        with self.assertRaisesRegex(RivalSchemaError, "deployment_design_sha256"):
            validate_root_manifest(attach_content_hash(changed))

        changed = deepcopy(original)
        changed.pop("content_sha256")
        entries = changed["candidate_selection_entries"]
        assert isinstance(entries, list)
        entry = entries[1]
        assert isinstance(entry, dict)
        entry["action_content_id"] = ACTION_CONTENT_ID_PREFIX + "c" * 64
        changed["deployment_design_sha256"] = deployment_design_identity(changed)
        with self.assertRaisesRegex(RivalSchemaError, "candidate_set_identity"):
            validate_root_manifest(attach_content_hash(changed))

    def test_locked_panel_manifest_is_valid(self) -> None:
        locked = json.loads((FIXTURE_DIR / "panel_manifest.json").read_text(encoding="utf-8"))
        manifest = validate_root_manifest(locked)
        self.assertEqual(manifest.required_panels, ("S", "H", "L"))

    def test_v1_root_manifest_cannot_enable_a_or_quantitative_targets(self) -> None:
        for field in ("a_panel_enabled", "quantitative_target_enabled"):
            bad = root_manifest()
            bad.pop("content_sha256")
            bad[field] = True
            bad = attach_content_hash(bad)
            with self.subTest(field=field), self.assertRaises(RivalSchemaError):
                validate_root_manifest(bad)

    def test_high_fidelity_control_forbids_l_and_multifidelity_claim(self) -> None:
        control = root_manifest()
        control.pop("content_sha256")
        control.update(
            {
                "inference_mode": "high_fidelity_only",
                "required_panels": ["S", "H"],
                "forbidden_panels": ["L", "A"],
                "panel_identities": {
                    "S": "sha256:" + "1" * 64,
                    "H": "sha256:" + "2" * 64,
                    "L": None,
                    "A": None,
                },
                "beta_cv": 0.0,
                "multifidelity_claim": False,
                "low_policy_identity": None,
                "coefficient_identity": None,
                "expected_l": 0,
                "low_expectation_id": None,
                "low_law_h_id": None,
                "low_law_l_id": None,
                "max_abs_beta": None,
            }
        )
        control["deployment_design_sha256"] = deployment_design_identity(control)
        parsed = validate_root_manifest(attach_content_hash(control))
        self.assertEqual(parsed.inference_mode, "high_fidelity_only")
        self.assertEqual(parsed.expected_l, 0)
        self.assertIsNone(parsed.low_policy_identity)

        for field, value in (("beta_cv", 0.1), ("multifidelity_claim", True)):
            invalid = deepcopy(control)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(RivalSchemaError):
                validate_root_manifest(attach_content_hash(invalid))


if __name__ == "__main__":
    unittest.main()
