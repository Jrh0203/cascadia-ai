"""Cohort-axis, panel-state, timeout, and categorical-label tests."""

import hashlib
import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.rival import appeals as appeals_module
from cascadiav3.rival.analysis import summarize_appeals
from cascadiav3.rival.appeals import (
    AppealError,
    AppealStateMachine,
    EvidenceDomain,
    HighFidelityAppealStateMachine,
    HighOnlyHRow,
    HRow,
    LRow,
    SelectionRow,
    UnitStatus,
    bind_high_fidelity_design,
    bind_multifidelity_design,
)
from cascadiav3.rival.bounds import verify_bound_certificate
from cascadiav3.rival.cohorts import (
    SEED_COMMITMENT_PREFIX,
    AllocationRegistry,
    CohortError,
    CompleteGameAssignment,
    CompleteGameSeedOpening,
    RootAssignment,
    RootSeedOpening,
    load_allocation_registry,
    root_source_set_identity,
    seed_commitment_for_value,
    validate_allocation_registry,
    validate_allocations,
    validate_manifest_collection,
    validate_seed_realizations,
)
from cascadiav3.rival.cohorts import (
    main as cohort_main,
)
from cascadiav3.rival.coverage import (
    ErrorFamilyLedger,
    HighOnlyRootErrorEntry,
    PotentialRootCensus,
    RootErrorEntry,
    validate_error_family_ledger,
    validate_potential_root_census,
)
from cascadiav3.rival.manifest import (
    ACTION_CONTENT_ID_PREFIX,
    CANDIDATE_OCCURRENCE_ID_PREFIX,
    PUBLIC_ROOT_ID_PREFIX,
    CandidateSelectionEntry,
    candidate_set_identity,
    deployment_design_identity,
    validate_root_manifest,
)
from cascadiav3.rival.multifidelity import validate_coefficient_calibration
from cascadiav3.rival.schema import (
    RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
    RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
    RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
    RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
    RIVAL_ROOT_MANIFEST_SCHEMA_ID,
    RivalSchemaError,
    attach_content_hash,
    canonical_json_bytes,
    sha256_hex,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rival"
SHA_A = "sha256:" + "a" * 64
INCUMBENT_ACTION_ID = "cascadiav3.rival_action_content.v1:sha256:" + "a" * 64
CHALLENGER_ACTION_ID = ACTION_CONTENT_ID_PREFIX + "b" * 64
BETTER_ACTION_ID = ACTION_CONTENT_ID_PREFIX + "c" * 64
WORSE_ACTION_ID = ACTION_CONTENT_ID_PREFIX + "d" * 64
INCUMBENT_OCCURRENCE_ID = CANDIDATE_OCCURRENCE_ID_PREFIX + "a" * 64
CHALLENGER_OCCURRENCE_ID = CANDIDATE_OCCURRENCE_ID_PREFIX + "b" * 64
BETTER_OCCURRENCE_ID = CANDIDATE_OCCURRENCE_ID_PREFIX + "c" * 64
WORSE_OCCURRENCE_ID = CANDIDATE_OCCURRENCE_ID_PREFIX + "d" * 64
ROOT_ID = PUBLIC_ROOT_ID_PREFIX + "1" * 64
HIGH_ROOT_ID = PUBLIC_ROOT_ID_PREFIX + "2" * 64
SELECTION_ROOT_ID = PUBLIC_ROOT_ID_PREFIX + "3" * 64


def seed_commitment(digit: str) -> str:
    return SEED_COMMITMENT_PREFIX + digit * 64


def allocation_registry_record(
    roots: list[dict[str, object]],
    games: list[dict[str, object]] | None = None,
    *,
    root_source_set_sha256: str | None = None,
) -> dict[str, object]:
    if root_source_set_sha256 is None:
        root_source_set_sha256 = root_source_set_identity(
            tuple(
                RootAssignment(
                    root_id=row["root_id"],
                    source_game_id=row["source_game_id"],
                    cohort_role=row["cohort_role"],
                    root_seed_commitment=row["root_seed_commitment"],
                )
                for row in roots
            )
        )
    return attach_content_hash(
        {
            "schema_id": RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
            "registry_id": "synthetic:test-registry",
            "root_source_set_sha256": root_source_set_sha256,
            "root_assignments": roots,
            "complete_game_assignments": games or [],
        }
    )


def externally_pinned_registry(record: dict[str, object]) -> AllocationRegistry:
    payload = canonical_json_bytes(record) + b"\n"
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "allocation-registry.json"
        path.write_bytes(payload)
        return load_allocation_registry(
            path,
            expected_file_sha256=hashlib.sha256(payload).hexdigest(),
            expected_content_sha256=record["content_sha256"],
        )


def panel_manifest_record(
    *,
    root_id: str = ROOT_ID,
    source_game_id: str = "synthetic:game:1",
    cohort_role: str = "untouched_coverage",
) -> dict[str, object]:
    record = json.loads((FIXTURE_DIR / "panel_manifest.json").read_text(encoding="utf-8"))
    record.pop("content_sha256")
    record.update(
        {
            "manifest_id": f"synthetic:manifest:{root_id}",
            "root_id": root_id,
            "source_game_id": source_game_id,
            "root_cohort_role": cohort_role,
        }
    )
    record["deployment_design_sha256"] = deployment_design_identity(record)
    return attach_content_hash(record)


def selection_entries(
    expected_s: int,
    challengers: tuple[tuple[str, str, int], ...] | None = None,
) -> tuple[CandidateSelectionEntry, ...]:
    challenger_rows = challengers or ((CHALLENGER_OCCURRENCE_ID, CHALLENGER_ACTION_ID, expected_s),)
    return (
        CandidateSelectionEntry(
            INCUMBENT_OCCURRENCE_ID,
            INCUMBENT_ACTION_ID,
            0,
        ),
        *(CandidateSelectionEntry(*row) for row in challenger_rows),
    )


def validated_error_family(
    root_id: str,
    *,
    inference_mode: str,
) -> tuple[PotentialRootCensus, ErrorFamilyLedger]:
    """Build validator-issued, census-complete test artifacts for one root."""

    registry_record = allocation_registry_record(
        [
            {
                "root_id": root_id,
                "source_game_id": f"synthetic:census-game:{root_id}",
                "cohort_role": "shadow_one_seat",
                "root_seed_commitment": seed_commitment_for_value(101),
            }
        ]
    )
    registry = externally_pinned_registry(registry_record)
    census_record = attach_content_hash(
        {
            "schema_id": RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
            "census_id": f"census:{root_id}",
            "family_kind": "one_seat_instrument",
            "source_root_set_sha256": registry.root_source_set_sha256,
            "allocation_registry_identity": registry.identity,
            "eligible_root_ids": [root_id],
        }
    )
    census = validate_potential_root_census(
        census_record,
        allocation_registry=registry,
        expected_allocation_registry_identity=registry.identity,
        expected_content_sha256=census_record["content_sha256"],
    )
    if inference_mode == "multifidelity":
        root_record: dict[str, object] = {
            "inference_mode": inference_mode,
            "root_id": root_id,
            "delta_root": 0.02,
            "delta_h": 0.01,
            "delta_l": 0.01,
        }
    elif inference_mode == "high_fidelity_only":
        root_record = {
            "inference_mode": inference_mode,
            "root_id": root_id,
            "delta_root": 0.02,
            "delta_h": 0.02,
        }
    else:  # pragma: no cover - helper contract
        raise AssertionError(inference_mode)
    ledger_record = attach_content_hash(
        {
            "schema_id": RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
            "family_id": f"errors:{inference_mode}:{root_id}",
            "family_kind": "one_seat_instrument",
            "source_census_sha256": census.identity,
            "delta_family": 0.02,
            "roots": [root_record],
        }
    )
    ledger = validate_error_family_ledger(
        ledger_record,
        census=census,
        expected_content_sha256=ledger_record["content_sha256"],
    )
    return census, ledger


def verified_design(
    *,
    root_id: str = ROOT_ID,
    expected_s: int = 1,
    expected_h: int = 1,
    expected_l: int = 1,
    beta_cv: float = 0.5,
    challengers: tuple[tuple[str, str, int], ...] | None = None,
    return_inputs: bool = False,
):
    record = json.loads(
        (FIXTURE_DIR / "global_bound_certificate_v1.json").read_text(encoding="utf-8")
    )
    certificate = verify_bound_certificate(
        record,
        expected_certificate_sha256=record["certificate_sha256"],
        expected_ruleset=record["ruleset"],
    )
    manifest_id = f"manifest:{root_id}"
    candidates = selection_entries(expected_s, challengers)
    _, errors = validated_error_family(root_id, inference_mode="multifidelity")
    error_entry = errors.roots[0]
    assert isinstance(error_entry, RootErrorEntry)
    manifest_record = {
        "schema_id": RIVAL_ROOT_MANIFEST_SCHEMA_ID,
        "manifest_id": manifest_id,
        "ruleset_identity": "sha256:" + sha256_hex(certificate.ruleset),
        "source_revision": "abc123",
        "root_id": root_id,
        "source_game_id": f"game:{root_id}",
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
        "beta_cv": beta_cv,
        "multifidelity_claim": True,
        "incumbent_policy_identity": SHA_A,
        "incumbent_action_id": INCUMBENT_ACTION_ID,
        "incumbent_candidate_occurrence_id": INCUMBENT_OCCURRENCE_ID,
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
        "allocation_identity": error_entry.allocation_identity,
        "bound_certificate_identity": "sha256:" + certificate.content_sha256,
        "error_ledger_identity": errors.identity,
        "expected_s": expected_s,
        "expected_h": expected_h,
        "expected_l": expected_l,
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
    deployment_identity = deployment_design_identity(manifest_record)
    coefficient_record = attach_content_hash(
        {
            "schema_id": RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
            "coefficient_id": "coefficient:1",
            "beta_cv": beta_cv,
            "calibration_cohort_id": "calibration:1",
            "calibration_source_corpus_sha256": SHA_A,
            "calibration_root_index_sha256": SHA_A,
            "calibration_data_sha256": SHA_A,
            "deployment_design_id": manifest_id,
            "deployment_design_sha256": deployment_identity,
            "incumbent_policy_id": SHA_A,
            "low_policy_id": SHA_A,
            "sampler_id": SHA_A,
            "allocation_id": error_entry.allocation_identity,
            "low_expectation_h_id": "expectation:1",
            "low_expectation_l_id": "expectation:1",
            "low_law_h_id": "law:1",
            "low_law_l_id": "law:1",
            "max_abs_beta": 2.0,
            "estimator_identity": SHA_A,
        }
    )
    coefficient = validate_coefficient_calibration(
        coefficient_record,
        expected_content_sha256=coefficient_record["content_sha256"],
    )
    manifest_record["coefficient_identity"] = coefficient.identity
    manifest_record["deployment_design_sha256"] = deployment_identity
    manifest = validate_root_manifest(attach_content_hash(manifest_record))
    if return_inputs:
        return manifest, coefficient, certificate, errors
    return bind_multifidelity_design(
        manifest=manifest,
        coefficient=coefficient,
        bound_certificate=certificate,
        error_family=errors,
    )


def verified_high_design(*, expected_s: int = 1, expected_h: int = 1):
    record = json.loads(
        (FIXTURE_DIR / "global_bound_certificate_v1.json").read_text(encoding="utf-8")
    )
    certificate = verify_bound_certificate(
        record,
        expected_certificate_sha256=record["certificate_sha256"],
        expected_ruleset=record["ruleset"],
    )
    root_id = HIGH_ROOT_ID
    candidates = selection_entries(expected_s)
    _, errors = validated_error_family(root_id, inference_mode="high_fidelity_only")
    error_entry = errors.roots[0]
    assert isinstance(error_entry, HighOnlyRootErrorEntry)
    manifest_record = {
        "schema_id": RIVAL_ROOT_MANIFEST_SCHEMA_ID,
        "manifest_id": "manifest:high-only",
        "ruleset_identity": "sha256:" + sha256_hex(certificate.ruleset),
        "source_revision": "abc123",
        "root_id": root_id,
        "source_game_id": "game:high-only",
        "source_game_identity_sha256": SHA_A,
        "root_kind": "draft_policy_root",
        "root_cohort_role": "untouched_coverage",
        "complete_game_seed_role": None,
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
        "incumbent_policy_identity": SHA_A,
        "incumbent_action_id": INCUMBENT_ACTION_ID,
        "incumbent_candidate_occurrence_id": INCUMBENT_OCCURRENCE_ID,
        "rules_menu_hash": ("cascadiav3.rival_rules_menu.v1:sha256:" + "a" * 64),
        "incumbent_menu_hash": ("cascadiav3.rival_incumbent_menu.v1:sha256:" + "a" * 64),
        "low_policy_identity": None,
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
        "coefficient_identity": None,
        "allocation_identity": error_entry.allocation_identity,
        "bound_certificate_identity": "sha256:" + certificate.content_sha256,
        "error_ledger_identity": errors.identity,
        "expected_s": expected_s,
        "expected_h": expected_h,
        "expected_l": 0,
        "practical_margin": 0.25,
        "preference_weight": 2.0,
        "selection_rule": "highest_mean_then_lexicographic_action_id",
        "low_expectation_id": None,
        "low_law_h_id": None,
        "low_law_l_id": None,
        "max_abs_beta": None,
        "a_panel_enabled": False,
        "quantitative_target_enabled": False,
    }
    manifest_record["deployment_design_sha256"] = deployment_design_identity(manifest_record)
    manifest = validate_root_manifest(attach_content_hash(manifest_record))
    return bind_high_fidelity_design(
        manifest=manifest,
        bound_certificate=certificate,
        error_family=errors,
    )


def machine(*, expected_h: int = 1, expected_l: int = 1) -> AppealStateMachine:
    return AppealStateMachine(design=verified_design(expected_h=expected_h, expected_l=expected_l))


def freeze(value: AppealStateMachine, score: float = 999.0) -> None:
    value.add_selection(
        SelectionRow.contract_test(
            "s0", CHALLENGER_OCCURRENCE_ID, score, UnitStatus.COMPLETE, "rng:s0"
        )
    )
    value.freeze_challenger(CHALLENGER_OCCURRENCE_ID)


class RivalAppealStateMachineTest(unittest.TestCase):
    def test_unverified_python_rows_cannot_be_constructed_as_evidence(self) -> None:
        with self.assertRaises(TypeError):
            SelectionRow(  # type: ignore[call-arg]
                "s0",
                CHALLENGER_OCCURRENCE_ID,
                1.0,
                UnitStatus.COMPLETE,
                "rng:s0",
            )

        valid = SelectionRow.contract_test(
            "s0",
            CHALLENGER_OCCURRENCE_ID,
            1.0,
            UnitStatus.COMPLETE,
            "rng:s0",
        )
        unsealed = SelectionRow(
            valid.unit_id,
            valid.challenger_id,
            valid.selection_score,
            valid.status,
            valid.rng_key,
            valid.world_redetermination_seed_sha256s,
            valid.evidence_domain,
            valid.evidence_receipt_sha256,
            None,
        )
        with self.assertRaisesRegex(AppealError, "must be produced"):
            machine().add_selection(unsealed)
        with self.assertRaisesRegex(AppealError, "does not match"):
            SelectionRow(
                valid.unit_id,
                valid.challenger_id,
                2.0,
                valid.status,
                valid.rng_key,
                valid.world_redetermination_seed_sha256s,
                valid.evidence_domain,
                valid.evidence_receipt_sha256,
                valid._validation_capability,
            )
        self.assertFalse(hasattr(appeals_module, "_VERIFIED_TERMINAL_EVIDENCE_PROOF"))
        self.assertFalse(hasattr(appeals_module, "_VERIFIED_DESIGN_PROOF"))

    def test_production_terminal_domain_is_structurally_unavailable_pre_gpu(self) -> None:
        with self.assertRaisesRegex(AppealError, "structurally unavailable"):
            appeals_module._verified_selection_row(
                unit_id="s-production",
                challenger_id=CHALLENGER_OCCURRENCE_ID,
                selection_score=10.0,
                rng_key="rng:production",
                world_redetermination_seed_sha256s=(
                    "sha256:" + "1" * 64,
                    "sha256:" + "2" * 64,
                ),
                evidence_domain=EvidenceDomain.PRODUCTION_TERMINAL,
                receipt_sha256="sha256:" + "3" * 64,
            )
        forged_state = machine()
        forged_state._evidence_domain = EvidenceDomain.PRODUCTION_TERMINAL
        with self.assertRaisesRegex(AppealError, "structurally unavailable"):
            forged_state.finalize()

    def test_row_and_design_capabilities_bind_every_runtime_field(self) -> None:
        row = SelectionRow.contract_test(
            "s0",
            CHALLENGER_OCCURRENCE_ID,
            1.0,
            UnitStatus.COMPLETE,
            "rng:s0",
        )
        with self.assertRaisesRegex(AppealError, "does not match"):
            replace(row, selection_score=2.0)

        design = verified_design()
        with self.assertRaisesRegex(AppealError, "does not match"):
            replace(design, practical_margin=design.practical_margin + 1.0)
        unsealed = replace(design, _validation_capability=None)
        with self.assertRaisesRegex(AppealError, "must be produced"):
            AppealStateMachine(design=unsealed)

    def test_failed_s_panel_finalizes_once_as_an_accounted_no_label(self) -> None:
        value = machine()
        value.add_selection(
            SelectionRow.contract_test(
                "s0",
                CHALLENGER_OCCURRENCE_ID,
                None,
                UnitStatus.TIMEOUT,
                "rng:s0",
            )
        )
        decision = value.finalize()
        self.assertEqual(decision.status, "no_label")
        self.assertEqual(decision.operational.attempted_s, 1)
        self.assertEqual(decision.operational.completed_s, 0)
        self.assertEqual(decision.operational.timeouts, 1)
        self.assertEqual(decision.operational.attempted_h, 0)
        with self.assertRaisesRegex(AppealError, "already been consumed"):
            value.finalize()

    def test_selection_rejects_unregistered_candidates_and_allocation_overrun(self) -> None:
        value = machine()
        with self.assertRaisesRegex(AppealError, "registered challenger"):
            value.add_selection(
                SelectionRow.contract_test(
                    "unknown",
                    BETTER_OCCURRENCE_ID,
                    1.0,
                    UnitStatus.COMPLETE,
                    "rng:unknown",
                )
            )
        value = AppealStateMachine(
            design=verified_design(
                root_id=SELECTION_ROOT_ID,
                expected_s=2,
                challengers=(
                    (BETTER_OCCURRENCE_ID, BETTER_ACTION_ID, 1),
                    (WORSE_OCCURRENCE_ID, WORSE_ACTION_ID, 1),
                ),
            )
        )
        value.add_selection(
            SelectionRow.contract_test(
                "s0", BETTER_OCCURRENCE_ID, 1.0, UnitStatus.COMPLETE, "rng:s0"
            )
        )
        with self.assertRaisesRegex(AppealError, "challenger-specific"):
            value.add_selection(
                SelectionRow.contract_test(
                    "s1", BETTER_OCCURRENCE_ID, 2.0, UnitStatus.COMPLETE, "rng:s1"
                )
            )

    def test_s_selects_one_then_is_discarded_from_fixed_categorical_label(self) -> None:
        sample_count = 1_000
        value = AppealStateMachine(
            design=verified_design(
                expected_h=sample_count,
                expected_l=sample_count,
                beta_cv=0.0,
            )
        )
        freeze(value, score=-1_000_000.0)
        for index in range(sample_count):
            value.add_h(
                HRow.contract_test(
                    f"h{index}",
                    CHALLENGER_OCCURRENCE_ID,
                    100.0,
                    0.0,
                    UnitStatus.COMPLETE,
                    f"physical:h{index}",
                    (f"inner:h:{index}:inc", f"inner:h:{index}:challenger"),
                )
            )
        for index in range(sample_count):
            value.add_l(
                LRow.contract_test(
                    f"l{index}",
                    CHALLENGER_OCCURRENCE_ID,
                    0.0,
                    UnitStatus.COMPLETE,
                    f"physical:l{index}",
                    (f"inner:l:{index}:inc", f"inner:l:{index}:challenger"),
                )
            )
        decision = value.finalize()
        self.assertEqual(decision.status, "no_label")
        self.assertIsNone(decision.preference)
        self.assertEqual(decision.evidence_domain, EvidenceDomain.CONTRACT_TEST)
        self.assertFalse(decision.scientific_evidence)
        self.assertEqual(decision.estimate_audit.estimate, 100.0)
        with self.assertRaisesRegex(AppealError, "already been consumed"):
            value.finalize()

    def test_timeout_stays_in_denominator_and_emits_no_label_or_mean(self) -> None:
        value = machine()
        freeze(value)
        value.add_h(
            HRow.contract_test(
                "h0",
                CHALLENGER_OCCURRENCE_ID,
                None,
                None,
                UnitStatus.TIMEOUT,
                "physical:h0",
                ("inner:h:inc", "inner:h:challenger"),
            )
        )
        value.add_l(
            LRow.contract_test(
                "l0",
                CHALLENGER_OCCURRENCE_ID,
                10.0,
                UnitStatus.COMPLETE,
                "physical:l0",
                ("inner:l:inc", "inner:l:challenger"),
            )
        )
        decision = value.finalize()
        self.assertEqual(decision.status, "no_label")
        self.assertIsNone(decision.estimate_audit)
        self.assertEqual(decision.operational.timeouts, 1)
        summary = summarize_appeals([decision])
        # Operational cost includes the completed S attempt as well as both
        # confirmation attempts; selection is discarded inferentially, not
        # erased from throughput/accounting.
        self.assertEqual(summary.attempted_terminal_units, 3)
        self.assertEqual(summary.completed_terminal_units, 2)
        self.assertEqual(summary.completion_rate, 2 / 3)

    def test_h_l_access_before_selection_and_incomplete_peek_reject(self) -> None:
        value = machine()
        with self.assertRaisesRegex(AppealError, "before S"):
            value.add_h(
                HRow.contract_test(
                    "h0",
                    CHALLENGER_OCCURRENCE_ID,
                    1.0,
                    1.0,
                    UnitStatus.COMPLETE,
                    "physical:h0",
                    ("inner:1", "inner:2"),
                )
            )
        freeze(value)
        with self.assertRaisesRegex(AppealError, "complete fixed H/L"):
            value.finalize()

    def test_s_winner_rule_is_frozen_and_deterministic(self) -> None:
        value = AppealStateMachine(
            design=verified_design(
                root_id=SELECTION_ROOT_ID,
                expected_s=2,
                beta_cv=0.0,
                challengers=(
                    (BETTER_OCCURRENCE_ID, BETTER_ACTION_ID, 1),
                    (WORSE_OCCURRENCE_ID, WORSE_ACTION_ID, 1),
                ),
            )
        )
        value.add_selection(
            SelectionRow.contract_test(
                "s0", BETTER_OCCURRENCE_ID, 2.0, UnitStatus.COMPLETE, "rng:s0"
            )
        )
        value.add_selection(
            SelectionRow.contract_test(
                "s1", WORSE_OCCURRENCE_ID, 1.0, UnitStatus.COMPLETE, "rng:s1"
            )
        )
        with self.assertRaisesRegex(AppealError, "winner rule"):
            value.freeze_challenger(WORSE_OCCURRENCE_ID)
        value.freeze_challenger(BETTER_OCCURRENCE_ID)

    def test_s_tie_breaks_by_action_content_not_occurrence_identity(self) -> None:
        lower_action_occurrence = CANDIDATE_OCCURRENCE_ID_PREFIX + "e" * 64
        higher_action_occurrence = CANDIDATE_OCCURRENCE_ID_PREFIX + "2" * 64
        lower_action_id = ACTION_CONTENT_ID_PREFIX + "2" * 64
        higher_action_id = ACTION_CONTENT_ID_PREFIX + "e" * 64
        value = AppealStateMachine(
            design=verified_design(
                root_id=SELECTION_ROOT_ID,
                expected_s=2,
                beta_cv=0.0,
                challengers=(
                    (lower_action_occurrence, lower_action_id, 1),
                    (higher_action_occurrence, higher_action_id, 1),
                ),
            )
        )
        value.add_selection(
            SelectionRow.contract_test(
                "s0", lower_action_occurrence, 7.0, UnitStatus.COMPLETE, "rng:s0"
            )
        )
        value.add_selection(
            SelectionRow.contract_test(
                "s1", higher_action_occurrence, 7.0, UnitStatus.COMPLETE, "rng:s1"
            )
        )

        # Occurrence IDs deliberately sort in the opposite order from action
        # content IDs.  The frozen manifest contract names the latter.
        with self.assertRaisesRegex(AppealError, "winner rule"):
            value.freeze_challenger(higher_action_occurrence)
        value.freeze_challenger(lower_action_occurrence)

    def test_second_challenger_key_reuse_and_a_panel_reject(self) -> None:
        manifest_record = json.loads(
            (FIXTURE_DIR / "panel_manifest.json").read_text(encoding="utf-8")
        )
        manifest_record.pop("content_sha256")
        manifest_record["a_panel_enabled"] = True
        with self.assertRaisesRegex(ValueError, "structurally disabled"):
            validate_root_manifest(attach_content_hash(manifest_record))
        value = machine()
        freeze(value)
        with self.assertRaisesRegex(AppealError, "frozen challenger"):
            value.add_h(
                HRow.contract_test(
                    "h0",
                    "other",
                    1.0,
                    1.0,
                    UnitStatus.COMPLETE,
                    "physical:h0",
                    ("inner:1", "inner:2"),
                )
            )

    def test_h_and_l_redetermination_seeds_are_independent(self) -> None:
        value = machine()
        freeze(value)
        value.add_h(
            HRow.contract_test(
                "h0",
                CHALLENGER_OCCURRENCE_ID,
                1.0,
                1.0,
                UnitStatus.COMPLETE,
                "physical:shared",
                ("inner:1", "inner:2"),
            )
        )
        with self.assertRaisesRegex(AppealError, "world-redetermination seed commitment reused"):
            value.add_l(
                LRow.contract_test(
                    "l0",
                    CHALLENGER_OCCURRENCE_ID,
                    1.0,
                    UnitStatus.COMPLETE,
                    "physical:shared",
                    ("inner:3", "inner:4"),
                )
            )

    def test_redetermination_seed_commitments_are_global_across_all_panels(self) -> None:
        value = machine()
        shared_physical_key = "physical:shared-across-s-h"
        value.add_selection(
            SelectionRow.contract_test(
                "s0",
                CHALLENGER_OCCURRENCE_ID,
                1.0,
                UnitStatus.COMPLETE,
                shared_physical_key,
            )
        )
        value.freeze_challenger(CHALLENGER_OCCURRENCE_ID)
        with self.assertRaisesRegex(AppealError, "world-redetermination seed commitment reused"):
            value.add_h(
                HRow.contract_test(
                    "h0",
                    CHALLENGER_OCCURRENCE_ID,
                    1.0,
                    1.0,
                    UnitStatus.COMPLETE,
                    shared_physical_key,
                    ("inner:h:inc", "inner:h:challenger"),
                )
            )

        forged = HRow.contract_test(
            "h-forged",
            CHALLENGER_OCCURRENCE_ID,
            1.0,
            1.0,
            UnitStatus.COMPLETE,
            "physical:forged",
            ("inner:forged:inc", "inner:forged:challenger"),
        )
        repeated = forged.world_redetermination_seed_sha256s[0]
        object.__setattr__(
            forged,
            "world_redetermination_seed_sha256s",
            (repeated, repeated),
        )
        with self.assertRaisesRegex(AppealError, "reused within one row"):
            value.add_h(forged)

        distinct = machine()
        freeze(distinct)
        distinct.add_h(
            HRow.contract_test(
                "h0",
                CHALLENGER_OCCURRENCE_ID,
                1.0,
                1.0,
                UnitStatus.COMPLETE,
                "physical:distinct-h",
                ("inner:distinct-h:inc", "inner:distinct-h:challenger"),
            )
        )
        distinct.add_l(
            LRow.contract_test(
                "l0",
                CHALLENGER_OCCURRENCE_ID,
                1.0,
                UnitStatus.COMPLETE,
                "physical:distinct-l",
                ("inner:distinct-l:inc", "inner:distinct-l:challenger"),
            )
        )
        self.assertEqual(distinct.operational_accounting().attempted_total, 3)

    def test_certified_ranges_reject_impossible_terminal_differences(self) -> None:
        value = machine()
        freeze(value)
        with self.assertRaisesRegex(AppealError, "outside certified range"):
            value.add_h(
                HRow.contract_test(
                    "h0",
                    CHALLENGER_OCCURRENCE_ID,
                    167.0,
                    0.0,
                    UnitStatus.COMPLETE,
                    "physical:h0",
                    ("inner:1", "inner:2"),
                )
            )

    def test_high_fidelity_control_is_distinct_and_forbids_l(self) -> None:
        sample_count = 1_000
        value = HighFidelityAppealStateMachine(design=verified_high_design(expected_h=sample_count))
        freeze(value)
        with self.assertRaisesRegex(AppealError, "structurally forbidden"):
            value.add_l(
                LRow.contract_test(
                    "l0",
                    CHALLENGER_OCCURRENCE_ID,
                    0.0,
                    UnitStatus.COMPLETE,
                    "physical:l0",
                    ("inner:l:1", "inner:l:2"),
                )
            )
        for index in range(sample_count):
            value.add_h(
                HighOnlyHRow.contract_test(
                    f"h{index}",
                    CHALLENGER_OCCURRENCE_ID,
                    100.0,
                    UnitStatus.COMPLETE,
                    f"physical:h{index}",
                    (f"inner:h:{index}:1", f"inner:h:{index}:2"),
                )
            )
        decision = value.finalize()
        self.assertEqual(decision.status, "no_label")
        self.assertIsNone(decision.preference)
        self.assertFalse(decision.scientific_evidence)
        self.assertEqual(decision.estimate_audit.estimate, 100.0)
        self.assertEqual(decision.operational.attempted_l, 0)


class RivalCohortAllocationTest(unittest.TestCase):
    def test_root_and_complete_game_roles_are_separate(self) -> None:
        summary = validate_allocations(
            [
                RootAssignment(
                    ROOT_ID,
                    "synthetic:g1",
                    "coefficient_calibration",
                    seed_commitment("1"),
                )
            ],
            [CompleteGameAssignment(seed_commitment("2"), "promotion")],
        )
        self.assertEqual(summary.root_count, 1)
        self.assertEqual(summary.complete_game_count, 1)

    def test_calibration_and_coverage_cannot_share_source_game(self) -> None:
        with self.assertRaisesRegex(CohortError, "share source game"):
            validate_allocations(
                [
                    RootAssignment(
                        ROOT_ID,
                        "synthetic:game",
                        "coefficient_calibration",
                        seed_commitment("1"),
                    ),
                    RootAssignment(
                        HIGH_ROOT_ID,
                        "synthetic:game",
                        "untouched_coverage",
                        seed_commitment("2"),
                    ),
                ],
                [],
            )

    def test_commitments_are_namespaced_and_globally_unique(self) -> None:
        with self.assertRaisesRegex(CohortError, "must use"):
            RootAssignment(
                ROOT_ID,
                "synthetic:g1",
                "design_tomography",
                "sha256:" + "1" * 64,
            )
        with self.assertRaisesRegex(CohortError, "reused across"):
            validate_allocations(
                [
                    RootAssignment(
                        ROOT_ID,
                        "synthetic:g1",
                        "design_tomography",
                        seed_commitment("1"),
                    )
                ],
                [CompleteGameAssignment(seed_commitment("1"), "target")],
            )
        with self.assertRaisesRegex(CohortError, "reused across"):
            validate_allocations(
                [],
                [
                    CompleteGameAssignment(seed_commitment("2"), "promotion"),
                    CompleteGameAssignment(seed_commitment("2"), "target"),
                ],
            )

    def test_registry_validates_sorted_exact_assignments(self) -> None:
        record = allocation_registry_record(
            [
                {
                    "root_id": ROOT_ID,
                    "source_game_id": "synthetic:game:1",
                    "cohort_role": "relabel_selection",
                    "root_seed_commitment": seed_commitment("1"),
                },
                {
                    "root_id": HIGH_ROOT_ID,
                    "source_game_id": "synthetic:game:2",
                    "cohort_role": "shadow_one_seat",
                    "root_seed_commitment": seed_commitment("2"),
                },
            ],
            [
                {
                    "seed_commitment": seed_commitment("3"),
                    "seed_role": "promotion",
                },
                {
                    "seed_commitment": seed_commitment("4"),
                    "seed_role": "target",
                },
            ],
        )
        registry = externally_pinned_registry(record)
        self.assertEqual(registry.identity, "sha256:" + record["content_sha256"])
        self.assertEqual(len(registry.root_assignments), 2)
        self.assertEqual(len(registry.complete_game_assignments), 2)
        registry.require_validated_artifact()
        semantic_only = validate_allocation_registry(
            record,
            expected_content_sha256=record["content_sha256"],
        )
        with self.assertRaisesRegex(CohortError, "artifact validator"):
            semantic_only.eligible_root_ids_for_family(
                "finite_training_corpus",
                expected_allocation_registry_identity=semantic_only.identity,
            )
        self.assertEqual(
            registry.eligible_root_ids_for_family(
                "finite_training_corpus",
                expected_allocation_registry_identity=registry.identity,
            ),
            (ROOT_ID,),
        )
        self.assertEqual(
            registry.eligible_root_ids_for_family(
                "one_seat_instrument",
                expected_allocation_registry_identity=registry.identity,
            ),
            (HIGH_ROOT_ID,),
        )
        with self.assertRaisesRegex(CohortError, "explicit cross-artifact"):
            registry.eligible_root_ids_for_family(
                "finite_training_corpus",
                expected_allocation_registry_identity="sha256:" + "f" * 64,
            )
        with self.assertRaisesRegex(CohortError, "unknown error family"):
            registry.eligible_root_ids_for_family(
                "substituted",
                expected_allocation_registry_identity=registry.identity,
            )

        tampered = deepcopy(record)
        tampered["registry_id"] = "synthetic:substituted"
        with self.assertRaisesRegex(RivalSchemaError, "content_sha256 mismatch"):
            validate_allocation_registry(
                tampered,
                expected_content_sha256=record["content_sha256"],
            )

        unqualified_parent = deepcopy(record)
        unqualified_parent.pop("content_sha256")
        unqualified_parent["root_source_set_sha256"] = "a" * 64
        unqualified_parent = attach_content_hash(unqualified_parent)
        with self.assertRaisesRegex(RivalSchemaError, "'sha256:' wire"):
            validate_allocation_registry(
                unqualified_parent,
                expected_content_sha256=unqualified_parent["content_sha256"],
            )

    def test_registry_rejects_duplicates_reordering_roles_and_cross_axis_reuse(self) -> None:
        root_one = {
            "root_id": ROOT_ID,
            "source_game_id": "synthetic:game:1",
            "cohort_role": "design_tomography",
            "root_seed_commitment": seed_commitment("1"),
        }
        root_two = {
            "root_id": HIGH_ROOT_ID,
            "source_game_id": "synthetic:game:2",
            "cohort_role": "untouched_coverage",
            "root_seed_commitment": seed_commitment("2"),
        }
        game_one = {
            "seed_commitment": seed_commitment("3"),
            "seed_role": "promotion",
        }
        game_two = {
            "seed_commitment": seed_commitment("4"),
            "seed_role": "target",
        }
        cases: list[tuple[str, dict[str, object], str]] = []

        duplicate_root = deepcopy(root_two)
        duplicate_root["root_id"] = ROOT_ID
        cases.append(
            (
                "duplicate root",
                allocation_registry_record(
                    [root_one, duplicate_root],
                    root_source_set_sha256=SHA_A,
                ),
                "root_id allocated more than once",
            )
        )
        duplicate_commitment = deepcopy(root_two)
        duplicate_commitment["root_seed_commitment"] = seed_commitment("1")
        cases.append(
            (
                "duplicate commitment",
                allocation_registry_record([root_one, duplicate_commitment]),
                "seed commitment reused",
            )
        )
        cases.append(
            (
                "root reordering",
                allocation_registry_record([root_two, root_one]),
                "root_assignments must be in canonical",
            )
        )
        cases.append(
            (
                "game reordering",
                allocation_registry_record([root_one], [game_two, game_one]),
                "complete_game_assignments must be in canonical",
            )
        )
        bad_root_role = deepcopy(root_one)
        bad_root_role["cohort_role"] = "promotion"
        cases.append(
            (
                "wrong root role",
                allocation_registry_record(
                    [bad_root_role],
                    root_source_set_sha256=SHA_A,
                ),
                "invalid root cohort role",
            )
        )
        bad_game_role = deepcopy(game_one)
        bad_game_role["seed_role"] = "untouched_coverage"
        cases.append(
            (
                "wrong game role",
                allocation_registry_record([root_one], [bad_game_role]),
                "invalid complete-game seed role",
            )
        )
        cross_axis = deepcopy(game_one)
        cross_axis["seed_commitment"] = seed_commitment("1")
        cases.append(
            (
                "cross-axis reuse",
                allocation_registry_record([root_one], [cross_axis]),
                "seed commitment reused",
            )
        )

        for name, invalid, reason in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(RivalSchemaError, reason),
            ):
                validate_allocation_registry(
                    invalid,
                    expected_content_sha256=invalid["content_sha256"],
                )

    def test_pinned_registry_loader_rejects_hash_noncanonical_and_mutable_paths(self) -> None:
        record = allocation_registry_record(
            [
                {
                    "root_id": ROOT_ID,
                    "source_game_id": "synthetic:game:1",
                    "cohort_role": "design_tomography",
                    "root_seed_commitment": seed_commitment("1"),
                }
            ]
        )
        canonical = canonical_json_bytes(record) + b"\n"
        expected_file_sha256 = hashlib.sha256(canonical).hexdigest()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "registry.json"
            path.write_bytes(canonical)
            loaded = load_allocation_registry(
                path,
                expected_file_sha256=expected_file_sha256,
                expected_content_sha256=record["content_sha256"],
            )
            self.assertEqual(loaded.content_sha256, record["content_sha256"])

            with self.assertRaisesRegex(RivalSchemaError, "preregistered pin"):
                load_allocation_registry(
                    path,
                    expected_file_sha256=expected_file_sha256,
                    expected_content_sha256="f" * 64,
                )

            path.write_bytes(canonical + b" ")
            with self.assertRaisesRegex(RivalSchemaError, "file SHA-256 mismatch"):
                load_allocation_registry(
                    path,
                    expected_file_sha256=expected_file_sha256,
                    expected_content_sha256=record["content_sha256"],
                )
            mutated_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RivalSchemaError, "canonical JSON"):
                load_allocation_registry(
                    path,
                    expected_file_sha256=mutated_hash,
                    expected_content_sha256=record["content_sha256"],
                )

            path.write_bytes(canonical)
            link = root / "registry-link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(RivalSchemaError, "safely open"):
                load_allocation_registry(
                    link,
                    expected_file_sha256=expected_file_sha256,
                    expected_content_sha256=record["content_sha256"],
                )

            hardlink = root / "registry-hardlink.json"
            hardlink.hardlink_to(path)
            with self.assertRaisesRegex(RivalSchemaError, "single-link regular file"):
                load_allocation_registry(
                    path,
                    expected_file_sha256=expected_file_sha256,
                    expected_content_sha256=record["content_sha256"],
                )

    def test_collection_requires_exact_registry_join_and_validated_capability(self) -> None:
        manifests = [
            panel_manifest_record(
                root_id=ROOT_ID,
                source_game_id="synthetic:game:1",
                cohort_role="design_tomography",
            ),
            panel_manifest_record(
                root_id=HIGH_ROOT_ID,
                source_game_id="synthetic:game:2",
                cohort_role="untouched_coverage",
            ),
        ]
        source_set_identity = root_source_set_identity(
            tuple(validate_root_manifest(record) for record in manifests)
        )
        record = allocation_registry_record(
            [
                {
                    "root_id": ROOT_ID,
                    "source_game_id": "synthetic:game:1",
                    "cohort_role": "design_tomography",
                    "root_seed_commitment": seed_commitment("1"),
                },
                {
                    "root_id": HIGH_ROOT_ID,
                    "source_game_id": "synthetic:game:2",
                    "cohort_role": "untouched_coverage",
                    "root_seed_commitment": seed_commitment("2"),
                },
            ],
            root_source_set_sha256=source_set_identity,
        )
        registry = externally_pinned_registry(record)
        structural = validate_manifest_collection(manifests)
        self.assertFalse(structural.commitment_uniqueness_validated)
        validated = validate_manifest_collection(
            manifests,
            allocation_registry=registry,
            expected_allocation_registry_identity=registry.identity,
        )
        self.assertTrue(validated.commitment_uniqueness_validated)
        self.assertEqual(validated.allocation_registry_identity, registry.identity)
        semantic_only_registry = validate_allocation_registry(
            record,
            expected_content_sha256=record["content_sha256"],
        )
        with self.assertRaisesRegex(CohortError, "artifact validator"):
            validate_manifest_collection(
                manifests,
                allocation_registry=semantic_only_registry,
                expected_allocation_registry_identity=semantic_only_registry.identity,
            )
        with self.assertRaisesRegex(CohortError, "explicit cross-artifact identity pin"):
            validate_manifest_collection(manifests, allocation_registry=registry)
        with self.assertRaisesRegex(CohortError, "differs from the explicit"):
            validate_manifest_collection(
                manifests,
                allocation_registry=registry,
                expected_allocation_registry_identity="sha256:" + "f" * 64,
            )

        omitted_record = allocation_registry_record(
            record["root_assignments"][:1],
            root_source_set_sha256=root_source_set_identity(
                (RootAssignment(**record["root_assignments"][0]),)
            ),
        )
        omitted_registry = externally_pinned_registry(omitted_record)
        for name, candidate_manifests, candidate_registry, reason in (
            (
                "registry omission",
                manifests,
                omitted_registry,
                "exact root-source set",
            ),
            (
                "manifest omission",
                manifests[:1],
                registry,
                "exact root-source set",
            ),
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(CohortError, reason),
            ):
                validate_manifest_collection(
                    candidate_manifests,
                    allocation_registry=candidate_registry,
                    expected_allocation_registry_identity=candidate_registry.identity,
                )

        extra_record = deepcopy(record)
        extra_record.pop("content_sha256")
        extra_record["root_assignments"].append(
            {
                "root_id": SELECTION_ROOT_ID,
                "source_game_id": "synthetic:game:3",
                "cohort_role": "shadow_one_seat",
                "root_seed_commitment": seed_commitment("3"),
            }
        )
        extra_record["root_source_set_sha256"] = root_source_set_identity(
            tuple(RootAssignment(**row) for row in extra_record["root_assignments"])
        )
        extra_record = attach_content_hash(extra_record)
        extra_registry = externally_pinned_registry(extra_record)
        with self.assertRaisesRegex(CohortError, "exact root-source set"):
            validate_manifest_collection(
                manifests,
                allocation_registry=extra_registry,
                expected_allocation_registry_identity=extra_registry.identity,
            )

        for field, value in (
            ("source_game_id", "synthetic:substituted"),
            ("cohort_role", "shadow_one_seat"),
        ):
            changed = deepcopy(record)
            changed.pop("content_sha256")
            changed["root_assignments"][0][field] = value
            changed["root_source_set_sha256"] = root_source_set_identity(
                tuple(RootAssignment(**row) for row in changed["root_assignments"])
            )
            changed = attach_content_hash(changed)
            changed_registry = externally_pinned_registry(changed)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(CohortError, "exact root-source set"),
            ):
                validate_manifest_collection(
                    manifests,
                    allocation_registry=changed_registry,
                    expected_allocation_registry_identity=changed_registry.identity,
                )

        direct = AllocationRegistry(
            registry_id=registry.registry_id,
            root_source_set_sha256=registry.root_source_set_sha256,
            root_assignments=registry.root_assignments,
            complete_game_assignments=registry.complete_game_assignments,
            content_sha256=registry.content_sha256,
        )
        with self.assertRaisesRegex(CohortError, "artifact validator"):
            validate_manifest_collection(
                manifests,
                allocation_registry=direct,
                expected_allocation_registry_identity=registry.identity,
            )
        stripped = replace(registry, _validation_capability=None)
        with self.assertRaisesRegex(CohortError, "artifact validator"):
            validate_manifest_collection(
                manifests,
                allocation_registry=stripped,
                expected_allocation_registry_identity=registry.identity,
            )
        with self.assertRaisesRegex(CohortError, "does not match"):
            replace(registry, registry_id="synthetic:substituted")

    def test_realized_openings_are_exact_rehashed_and_globally_disjoint(self) -> None:
        manifests = [
            panel_manifest_record(
                root_id=ROOT_ID,
                source_game_id="synthetic:game:1",
                cohort_role="design_tomography",
            ),
            panel_manifest_record(
                root_id=HIGH_ROOT_ID,
                source_game_id="synthetic:game:2",
                cohort_role="untouched_coverage",
            ),
        ]
        source_set_identity = root_source_set_identity(
            tuple(validate_root_manifest(record) for record in manifests)
        )
        root_rows = [
            {
                "root_id": ROOT_ID,
                "source_game_id": "synthetic:game:1",
                "cohort_role": "design_tomography",
                "root_seed_commitment": seed_commitment_for_value(101),
            },
            {
                "root_id": HIGH_ROOT_ID,
                "source_game_id": "synthetic:game:2",
                "cohort_role": "untouched_coverage",
                "root_seed_commitment": seed_commitment_for_value(102),
            },
        ]
        game_rows = sorted(
            [
                {
                    "seed_commitment": seed_commitment_for_value(201),
                    "seed_role": "promotion",
                },
                {
                    "seed_commitment": seed_commitment_for_value(202),
                    "seed_role": "target",
                },
            ],
            key=lambda row: (row["seed_commitment"], row["seed_role"]),
        )
        record = allocation_registry_record(
            root_rows,
            game_rows,
            root_source_set_sha256=source_set_identity,
        )
        registry = externally_pinned_registry(record)
        manifest_collection = validate_manifest_collection(
            manifests,
            allocation_registry=registry,
            expected_allocation_registry_identity=registry.identity,
        )
        valid_roots = (
            RootSeedOpening(ROOT_ID, 101),
            RootSeedOpening(HIGH_ROOT_ID, 102),
        )
        valid_games = (
            CompleteGameSeedOpening("promotion", 201),
            CompleteGameSeedOpening("target", 202),
        )
        summary = validate_seed_realizations(
            registry,
            expected_allocation_registry_identity=registry.identity,
            manifest_collection=manifest_collection,
            root_openings=valid_roots,
            complete_game_openings=valid_games,
        )
        self.assertTrue(summary.realized_seed_disjointness_validated)
        self.assertEqual(summary.opened_root_count, 2)
        self.assertEqual(summary.opened_complete_game_count, 2)

        stripped_collection = replace(
            manifest_collection,
            _validation_capability=None,
        )
        with self.assertRaisesRegex(CohortError, "artifact validator"):
            validate_seed_realizations(
                registry,
                expected_allocation_registry_identity=registry.identity,
                manifest_collection=stripped_collection,
                root_openings=valid_roots,
                complete_game_openings=valid_games,
            )
        with self.assertRaisesRegex(CohortError, "does not match"):
            replace(
                manifest_collection,
                root_count=manifest_collection.root_count + 1,
            )

        with self.assertRaisesRegex(CohortError, "realized seed reused"):
            validate_seed_realizations(
                registry,
                expected_allocation_registry_identity=registry.identity,
                manifest_collection=manifest_collection,
                root_openings=(
                    RootSeedOpening(ROOT_ID, 101),
                    RootSeedOpening(HIGH_ROOT_ID, 101),
                ),
                complete_game_openings=valid_games,
            )
        with self.assertRaisesRegex(CohortError, "does not open"):
            validate_seed_realizations(
                registry,
                expected_allocation_registry_identity=registry.identity,
                manifest_collection=manifest_collection,
                root_openings=(
                    RootSeedOpening(ROOT_ID, 102),
                    RootSeedOpening(HIGH_ROOT_ID, 101),
                ),
                complete_game_openings=valid_games,
            )
        with self.assertRaisesRegex(CohortError, "root seed openings must exactly"):
            validate_seed_realizations(
                registry,
                expected_allocation_registry_identity=registry.identity,
                manifest_collection=manifest_collection,
                root_openings=valid_roots[:1],
                complete_game_openings=valid_games,
            )
        with self.assertRaisesRegex(CohortError, "complete-game seed openings must exactly"):
            validate_seed_realizations(
                registry,
                expected_allocation_registry_identity=registry.identity,
                manifest_collection=manifest_collection,
                root_openings=valid_roots,
                complete_game_openings=(
                    CompleteGameSeedOpening("target", 201),
                    CompleteGameSeedOpening("promotion", 202),
                ),
            )

        fake_record = deepcopy(record)
        fake_record.pop("content_sha256")
        fake_record["root_assignments"][0]["root_seed_commitment"] = seed_commitment("f")
        fake_record = attach_content_hash(fake_record)
        fake_registry = externally_pinned_registry(fake_record)
        fake_manifest_collection = validate_manifest_collection(
            manifests,
            allocation_registry=fake_registry,
            expected_allocation_registry_identity=fake_registry.identity,
        )
        with self.assertRaisesRegex(CohortError, "does not open"):
            validate_seed_realizations(
                fake_registry,
                expected_allocation_registry_identity=fake_registry.identity,
                manifest_collection=fake_manifest_collection,
                root_openings=valid_roots,
                complete_game_openings=valid_games,
            )

        for invalid in (True, -1, 1 << 64):
            with (
                self.subTest(seed=invalid),
                self.assertRaisesRegex(CohortError, "unsigned 64-bit"),
            ):
                RootSeedOpening(ROOT_ID, invalid)

    def test_cli_never_claims_seed_validation_without_pinned_registry(self) -> None:
        manifest = panel_manifest_record(
            root_id=ROOT_ID,
            source_game_id="synthetic:game:1",
            cohort_role="untouched_coverage",
        )
        registry_record = allocation_registry_record(
            [
                {
                    "root_id": ROOT_ID,
                    "source_game_id": "synthetic:game:1",
                    "cohort_role": "untouched_coverage",
                    "root_seed_commitment": seed_commitment_for_value(101),
                }
            ],
            root_source_set_sha256=root_source_set_identity((validate_root_manifest(manifest),)),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
            registry_path = root / "registry.json"
            registry_bytes = canonical_json_bytes(registry_record) + b"\n"
            registry_path.write_bytes(registry_bytes)
            common = [
                "validate",
                "--manifest",
                str(manifest_path),
                "--require-panels",
                "S,H,L",
                "--require-disjoint",
                "calibration,coverage",
                "--require-a-disabled",
            ]

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cohort_main(common), 0)
            structural = json.loads(output.getvalue())
            self.assertEqual(
                structural["seed_disjointness"],
                "NOT_VALIDATED_UNTIL_OPENINGS",
            )
            self.assertEqual(structural["commitment_uniqueness"], "NOT_VALIDATED")
            self.assertNotIn("allocation_registry_identity", structural)

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cohort_main([*common, "--claim-seed-disjointness"]), 2)
            denied = json.loads(output.getvalue())
            self.assertIn("byte-pinned allocation registry", denied["reason"])

            pinned = [
                *common,
                "--allocation-registry",
                str(registry_path),
                "--allocation-registry-file-sha256",
                hashlib.sha256(registry_bytes).hexdigest(),
                "--allocation-registry-content-sha256",
                registry_record["content_sha256"],
            ]
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cohort_main(pinned), 0)
            committed = json.loads(output.getvalue())
            self.assertEqual(
                committed["commitment_uniqueness"],
                "VALIDATED_FROM_PINNED_REGISTRY",
            )
            self.assertEqual(
                committed["seed_disjointness"],
                "NOT_VALIDATED_UNTIL_OPENINGS",
            )

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cohort_main([*pinned, "--claim-seed-disjointness"]),
                    2,
                )
            unopened = json.loads(output.getvalue())
            self.assertIn("root seed openings must exactly", unopened["reason"])

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cohort_main(
                        [
                            *pinned,
                            "--claim-seed-disjointness",
                            "--root-seed-opening",
                            f"{ROOT_ID}=101",
                        ]
                    ),
                    0,
                )
            validated = json.loads(output.getvalue())
            self.assertEqual(
                validated["seed_disjointness"],
                "VALIDATED_FROM_EXACT_OPENINGS",
            )
            self.assertEqual(
                validated["allocation_registry_identity"],
                "sha256:" + registry_record["content_sha256"],
            )

    def test_binder_rejects_every_directly_constructed_scientific_artifact(self) -> None:
        manifest, coefficient, certificate, errors = verified_design(return_inputs=True)
        common = {
            "manifest": manifest,
            "coefficient": coefficient,
            "bound_certificate": certificate,
            "error_family": errors,
        }
        for name, direct in (
            ("manifest", replace(manifest, _validation_capability=None)),
            ("coefficient", replace(coefficient, _validation_capability=None)),
            ("bound_certificate", replace(certificate, _validation_capability=None)),
            ("error_family", replace(errors, _validation_capability=None)),
        ):
            inputs = {**common, name: direct}
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(AppealError, "unvalidated scientific"),
            ):
                bind_multifidelity_design(**inputs)

    def test_manifest_collection_requires_one_census_complete_ledger(self) -> None:
        root_ids = (ROOT_ID, HIGH_ROOT_ID)
        registry_record = allocation_registry_record(
            [
                {
                    "root_id": root_id,
                    "source_game_id": f"game:collection:{root_id}",
                    "cohort_role": "shadow_one_seat",
                    "root_seed_commitment": seed_commitment_for_value(301 + index),
                }
                for index, root_id in enumerate(root_ids)
            ]
        )
        registry = externally_pinned_registry(registry_record)
        census_record = attach_content_hash(
            {
                "schema_id": RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
                "census_id": "census:collection",
                "family_kind": "one_seat_instrument",
                "source_root_set_sha256": registry.root_source_set_sha256,
                "allocation_registry_identity": registry.identity,
                "eligible_root_ids": list(root_ids),
            }
        )
        census = validate_potential_root_census(
            census_record,
            allocation_registry=registry,
            expected_allocation_registry_identity=registry.identity,
            expected_content_sha256=census_record["content_sha256"],
        )
        ledger_record = attach_content_hash(
            {
                "schema_id": RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
                "family_id": "errors:collection",
                "family_kind": "one_seat_instrument",
                "source_census_sha256": census.identity,
                "delta_family": 0.04,
                "roots": [
                    {
                        "inference_mode": "multifidelity",
                        "root_id": root_id,
                        "delta_root": 0.02,
                        "delta_h": 0.01,
                        "delta_l": 0.01,
                    }
                    for root_id in root_ids
                ],
            }
        )
        ledger = validate_error_family_ledger(
            ledger_record,
            census=census,
            expected_content_sha256=ledger_record["content_sha256"],
        )

        records: list[dict[str, object]] = []
        for root_id, allocation in zip(root_ids, ledger.roots, strict=True):
            record = json.loads((FIXTURE_DIR / "panel_manifest.json").read_text(encoding="utf-8"))
            record.pop("content_sha256")
            record.update(
                {
                    "manifest_id": f"manifest:collection:{root_id}",
                    "root_id": root_id,
                    "source_game_id": f"game:collection:{root_id}",
                    "root_cohort_role": "shadow_one_seat",
                    "allocation_identity": allocation.allocation_identity,
                    "error_ledger_identity": ledger.identity,
                }
            )
            record["deployment_design_sha256"] = deployment_design_identity(record)
            records.append(attach_content_hash(record))

        validate_manifest_collection(
            records,
            allocation_registry=registry,
            expected_allocation_registry_identity=registry.identity,
            potential_root_census=census,
            error_family=ledger,
        )
        with self.assertRaisesRegex(CohortError, "exact root-source set"):
            validate_manifest_collection(
                records[:1],
                allocation_registry=registry,
                expected_allocation_registry_identity=registry.identity,
                potential_root_census=census,
                error_family=ledger,
            )

        divergent_records = deepcopy(records)
        divergent_records[1].pop("content_sha256")
        divergent_records[1]["error_ledger_identity"] = SHA_A
        divergent_records[1]["deployment_design_sha256"] = deployment_design_identity(
            divergent_records[1]
        )
        divergent_records[1] = attach_content_hash(divergent_records[1])
        with self.assertRaisesRegex(CohortError, "same error ledger"):
            validate_manifest_collection(
                divergent_records,
                allocation_registry=registry,
                expected_allocation_registry_identity=registry.identity,
                potential_root_census=census,
                error_family=ledger,
            )

        different_ledger_record = deepcopy(ledger_record)
        different_ledger_record.pop("content_sha256")
        different_ledger_record["family_id"] = "errors:collection:substituted"
        different_ledger_record = attach_content_hash(different_ledger_record)
        different_ledger = validate_error_family_ledger(
            different_ledger_record,
            census=census,
            expected_content_sha256=different_ledger_record["content_sha256"],
        )
        with self.assertRaisesRegex(CohortError, "same error ledger"):
            validate_manifest_collection(
                records,
                allocation_registry=registry,
                expected_allocation_registry_identity=registry.identity,
                potential_root_census=census,
                error_family=different_ledger,
            )


if __name__ == "__main__":
    unittest.main()
