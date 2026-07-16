"""Multiplicity, exact coverage, clustering, and symbolic power tests."""

import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.rival.analysis import (
    AnalysisError,
    RootMeasurement,
    iid_root_standard_error_for_diagnostic,
    source_game_clustered_summary,
)
from cascadiav3.rival.bounds import RootErrorAllocation
from cascadiav3.rival.cohorts import (
    ERROR_FAMILY_ROOT_COHORT_ROLE,
    AllocationRegistry,
    RootAssignment,
    load_allocation_registry,
    root_source_set_identity,
    seed_commitment_for_value,
)
from cascadiav3.rival.coverage import (
    CoverageError,
    DiscreteJointOutcome,
    DiscreteLowOutcome,
    ErrorFamilyLedger,
    HighOnlyRootErrorEntry,
    PotentialRootCensus,
    RootErrorEntry,
    binomial_upper_confidence_bound,
    enumerate_exact_coverage,
    load_error_family_ledger,
    load_potential_root_census,
    run_coverage_design,
    validate_error_family_ledger,
    validate_potential_root_census,
    validate_separate_error_families,
    zero_failure_replications,
)
from cascadiav3.rival.coverage import (
    main as coverage_main,
)
from cascadiav3.rival.manifest import PUBLIC_ROOT_ID_PREFIX
from cascadiav3.rival.power import (
    NO_FINITE_HOURS,
    NON_FUNDING_STATUS,
    UNRESOLVED,
    CertifiedStratumRange,
    HypotheticalThroughput,
    MemoryAssumption,
    PowerEnvelopeSpec,
    build_power_envelope,
    validate_power_envelope,
)
from cascadiav3.rival.schema import (
    RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
    RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
    RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
    RivalSchemaError,
    attach_content_hash,
    canonical_json_bytes,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rival"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
PUBLIC_ROOT_1 = PUBLIC_ROOT_ID_PREFIX + "1" * 64
PUBLIC_ROOT_2 = PUBLIC_ROOT_ID_PREFIX + "2" * 64
PUBLIC_ROOT_3 = PUBLIC_ROOT_ID_PREFIX + "3" * 64
PUBLIC_ROOT_4 = PUBLIC_ROOT_ID_PREFIX + "4" * 64


def pinned_allocation_registry(
    assignments: tuple[tuple[str, str], ...],
    *,
    registry_id: str = "synthetic:coverage-registry",
    root_source_set_sha256: str | None = None,
    source_game_ids: tuple[str, ...] | None = None,
) -> AllocationRegistry:
    if source_game_ids is not None and len(source_game_ids) != len(assignments):
        raise ValueError("source_game_ids must align exactly with assignments")
    rows = [
        {
            "root_id": root_id,
            "source_game_id": (
                source_game_ids[index] if source_game_ids is not None else f"synthetic:game:{index}"
            ),
            "cohort_role": cohort_role,
            "root_seed_commitment": seed_commitment_for_value(index + 1),
        }
        for index, (root_id, cohort_role) in enumerate(assignments)
    ]
    rows.sort(key=lambda row: row["root_id"])
    typed_rows = tuple(RootAssignment(**row) for row in rows)
    record = attach_content_hash(
        {
            "schema_id": RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
            "registry_id": registry_id,
            "root_source_set_sha256": (
                root_source_set_sha256
                if root_source_set_sha256 is not None
                else root_source_set_identity(typed_rows)
            ),
            "root_assignments": rows,
            "complete_game_assignments": [],
        }
    )
    payload = canonical_json_bytes(record) + b"\n"
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "allocation-registry.json"
        path.write_bytes(payload)
        return load_allocation_registry(
            path,
            expected_file_sha256=hashlib.sha256(payload).hexdigest(),
            expected_content_sha256=record["content_sha256"],
        )


def potential_root_census_record(
    allocation_registry: AllocationRegistry,
    *,
    family_kind: str = "one_seat_instrument",
    root_ids: tuple[str, ...] = (PUBLIC_ROOT_1, PUBLIC_ROOT_2),
) -> dict[str, object]:
    return attach_content_hash(
        {
            "schema_id": RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
            "census_id": "census:instrument:locked",
            "family_kind": family_kind,
            "source_root_set_sha256": allocation_registry.root_source_set_sha256,
            "allocation_registry_identity": allocation_registry.identity,
            "eligible_root_ids": list(root_ids),
        }
    )


def validated_potential_root_census() -> PotentialRootCensus:
    registry = pinned_allocation_registry(
        (
            (PUBLIC_ROOT_1, "shadow_one_seat"),
            (PUBLIC_ROOT_2, "shadow_one_seat"),
        )
    )
    record = potential_root_census_record(registry)
    return validate_potential_root_census(
        record,
        allocation_registry=registry,
        expected_allocation_registry_identity=registry.identity,
        expected_content_sha256=record["content_sha256"],
    )


def error_family_record(
    census: PotentialRootCensus,
    *,
    root_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    selected_roots = root_ids or census.eligible_root_ids
    return attach_content_hash(
        {
            "schema_id": RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
            "family_id": f"errors:{census.family_kind}:locked",
            "family_kind": census.family_kind,
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
                for root_id in selected_roots
            ],
        }
    )


def validated_single_root_family(
    *,
    family_kind: str,
    root_id: str,
) -> ErrorFamilyLedger:
    registry = pinned_allocation_registry(((root_id, ERROR_FAMILY_ROOT_COHORT_ROLE[family_kind]),))
    census_record = attach_content_hash(
        {
            "schema_id": RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
            "census_id": f"census:{family_kind}:{root_id}",
            "family_kind": family_kind,
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
    record = error_family_record(census)
    return validate_error_family_ledger(
        record,
        census=census,
        expected_content_sha256=record["content_sha256"],
    )


class RivalErrorLedgerTest(unittest.TestCase):
    def test_sum_of_every_potential_root_budget_is_bounded(self) -> None:
        roots = tuple(RootErrorEntry(f"root:{index}", 0.01, 0.004, 0.006) for index in range(4))
        ledger = ErrorFamilyLedger("instrument:1", "one_seat_instrument", 0.05, 4, roots)
        self.assertAlmostEqual(ledger.validate(), 0.04)

    def test_unseen_eligible_roots_and_overspend_reject(self) -> None:
        with self.assertRaisesRegex(CoverageError, "enumerate every"):
            ErrorFamilyLedger(
                "instrument:1",
                "one_seat_instrument",
                0.05,
                2,
                (RootErrorEntry("root:1", 0.01, 0.005, 0.005),),
            ).validate()
        with self.assertRaisesRegex(CoverageError, "exceeds"):
            ErrorFamilyLedger(
                "instrument:1",
                "one_seat_instrument",
                0.01,
                1,
                (RootErrorEntry("root:1", 0.02, 0.01, 0.01),),
            ).validate()

    def test_training_and_instrument_families_are_separate(self) -> None:
        training = validated_single_root_family(
            family_kind="finite_training_corpus",
            root_id=PUBLIC_ROOT_1,
        )
        instrument = validated_single_root_family(
            family_kind="one_seat_instrument",
            root_id=PUBLIC_ROOT_2,
        )
        validate_separate_error_families((training, instrument))
        with self.assertRaisesRegex(CoverageError, "artifact validator"):
            validate_separate_error_families(
                (replace(training, _validation_capability=None), instrument)
            )
        overlapping = validated_single_root_family(
            family_kind="one_seat_instrument",
            root_id=PUBLIC_ROOT_1,
        )
        with self.assertRaisesRegex(CoverageError, "reuse root"):
            validate_separate_error_families((training, overlapping))
        with self.assertRaisesRegex(CoverageError, "requires separate"):
            validate_separate_error_families((training,))

    def test_high_only_error_family_spends_no_synthetic_l_term(self) -> None:
        ledger = ErrorFamilyLedger(
            "high-only:1",
            "one_seat_instrument",
            0.05,
            1,
            (HighOnlyRootErrorEntry("high-root", 0.01, 0.01),),
        )
        self.assertEqual(ledger.validate(), 0.01)
        self.assertFalse(hasattr(ledger.roots[0], "delta_l"))

    def test_census_and_ledger_capabilities_reject_forgery_and_mutation(self) -> None:
        census = validated_potential_root_census()
        census.require_validated_artifact()
        constructed_census = PotentialRootCensus(
            census_id=census.census_id,
            family_kind=census.family_kind,
            source_root_set_sha256=census.source_root_set_sha256,
            allocation_registry_identity=census.allocation_registry_identity,
            eligible_root_ids=census.eligible_root_ids,
            content_sha256=census.content_sha256,
        )
        with self.assertRaisesRegex(CoverageError, "artifact validator"):
            constructed_census.require_validated_artifact()
        direct_census = replace(census, _validation_capability=None)
        with self.assertRaisesRegex(CoverageError, "artifact validator"):
            direct_census.require_validated_artifact()
        with self.assertRaisesRegex(CoverageError, "does not match"):
            replace(census, source_root_set_sha256=SHA_B)

        record = error_family_record(census)
        with self.assertRaisesRegex(RivalSchemaError, "artifact validator"):
            validate_error_family_ledger(
                record,
                census=constructed_census,
                expected_content_sha256=record["content_sha256"],
            )
        ledger = validate_error_family_ledger(
            record,
            census=census,
            expected_content_sha256=record["content_sha256"],
        )
        ledger.require_validated_artifact(census=census)
        self.assertEqual(tuple(row.root_id for row in ledger.roots), census.eligible_root_ids)
        direct_ledger = replace(ledger, _validation_capability=None)
        self.assertAlmostEqual(direct_ledger.validate(), 0.04)
        with self.assertRaisesRegex(CoverageError, "artifact validator"):
            direct_ledger.require_validated_artifact(census=census)
        with self.assertRaisesRegex(CoverageError, "does not match"):
            replace(ledger, family_id="errors:substituted")

    def test_census_capability_requires_the_exact_externally_pinned_registry(self) -> None:
        assignments = (
            (PUBLIC_ROOT_1, "shadow_one_seat"),
            (PUBLIC_ROOT_2, "shadow_one_seat"),
        )
        registry = pinned_allocation_registry(assignments)
        record = potential_root_census_record(registry)

        semantic_only = replace(registry, _external_pin_capability=None)
        with self.assertRaisesRegex(RivalSchemaError, "artifact validator"):
            validate_potential_root_census(
                record,
                allocation_registry=semantic_only,
                expected_allocation_registry_identity=semantic_only.identity,
                expected_content_sha256=record["content_sha256"],
            )

        with self.assertRaisesRegex(RivalSchemaError, "explicit cross-artifact"):
            validate_potential_root_census(
                record,
                allocation_registry=registry,
                expected_allocation_registry_identity="sha256:" + "f" * 64,
                expected_content_sha256=record["content_sha256"],
            )

        wrong_registry = pinned_allocation_registry(
            assignments,
            registry_id="synthetic:wrong-coverage-registry",
        )
        with self.assertRaisesRegex(RivalSchemaError, "allocation_registry_identity"):
            validate_potential_root_census(
                record,
                allocation_registry=wrong_registry,
                expected_allocation_registry_identity=wrong_registry.identity,
                expected_content_sha256=record["content_sha256"],
            )

        wrong_family = deepcopy(record)
        wrong_family.pop("content_sha256")
        wrong_family["family_kind"] = "finite_training_corpus"
        wrong_family = attach_content_hash(wrong_family)
        with self.assertRaisesRegex(RivalSchemaError, "contains no roots"):
            validate_potential_root_census(
                wrong_family,
                allocation_registry=registry,
                expected_allocation_registry_identity=registry.identity,
                expected_content_sha256=wrong_family["content_sha256"],
            )

        wrong_manifest_source = deepcopy(record)
        wrong_manifest_source.pop("content_sha256")
        wrong_manifest_source["source_root_set_sha256"] = SHA_B
        wrong_manifest_source = attach_content_hash(wrong_manifest_source)
        with self.assertRaisesRegex(RivalSchemaError, "root-source set"):
            validate_potential_root_census(
                wrong_manifest_source,
                allocation_registry=registry,
                expected_allocation_registry_identity=registry.identity,
                expected_content_sha256=wrong_manifest_source["content_sha256"],
            )

    def test_census_and_ledger_reject_omission_order_duplicates_and_substitution(self) -> None:
        registry = pinned_allocation_registry(
            (
                (PUBLIC_ROOT_1, "shadow_one_seat"),
                (PUBLIC_ROOT_2, "shadow_one_seat"),
            )
        )
        for root_ids in (
            (PUBLIC_ROOT_1,),
            (PUBLIC_ROOT_1, PUBLIC_ROOT_2, PUBLIC_ROOT_ID_PREFIX + "3" * 64),
            (PUBLIC_ROOT_2, PUBLIC_ROOT_1),
            (PUBLIC_ROOT_1, PUBLIC_ROOT_1),
        ):
            record = potential_root_census_record(registry, root_ids=root_ids)
            reason = (
                "externally pinned allocation registry family universe"
                if root_ids
                in {
                    (PUBLIC_ROOT_1,),
                    (PUBLIC_ROOT_1, PUBLIC_ROOT_2, PUBLIC_ROOT_ID_PREFIX + "3" * 64),
                }
                else "unique and sorted"
            )
            with self.subTest(root_ids=root_ids), self.assertRaisesRegex(RivalSchemaError, reason):
                validate_potential_root_census(
                    record,
                    allocation_registry=registry,
                    expected_allocation_registry_identity=registry.identity,
                    expected_content_sha256=record["content_sha256"],
                )

        census = validated_potential_root_census()
        omitted = error_family_record(census, root_ids=(PUBLIC_ROOT_1,))
        with self.assertRaisesRegex(RivalSchemaError, "exactly equal"):
            validate_error_family_ledger(
                omitted,
                census=census,
                expected_content_sha256=omitted["content_sha256"],
            )
        reordered = error_family_record(
            census,
            root_ids=(PUBLIC_ROOT_2, PUBLIC_ROOT_1),
        )
        with self.assertRaisesRegex(RivalSchemaError, "canonical order"):
            validate_error_family_ledger(
                reordered,
                census=census,
                expected_content_sha256=reordered["content_sha256"],
            )
        substituted = error_family_record(census)
        substituted["delta_family"] = 0.05
        with self.assertRaisesRegex(RivalSchemaError, "content_sha256 mismatch"):
            validate_error_family_ledger(
                substituted,
                census=census,
                expected_content_sha256=substituted["content_sha256"],
            )

    def test_census_and_ledger_loaders_pin_canonical_file_and_content_bytes(self) -> None:
        registry = pinned_allocation_registry(
            (
                (PUBLIC_ROOT_1, "shadow_one_seat"),
                (PUBLIC_ROOT_2, "shadow_one_seat"),
            )
        )
        census_record = potential_root_census_record(registry)
        census_bytes = canonical_json_bytes(census_record) + b"\n"
        with TemporaryDirectory() as temporary:
            census_path = Path(temporary) / "census.json"
            census_path.write_bytes(census_bytes)
            census = load_potential_root_census(
                census_path,
                allocation_registry=registry,
                expected_allocation_registry_identity=registry.identity,
                expected_file_sha256=hashlib.sha256(census_bytes).hexdigest(),
                expected_content_sha256=census_record["content_sha256"],
            )
            ledger_record = error_family_record(census)
            ledger_bytes = canonical_json_bytes(ledger_record) + b"\n"
            ledger_path = Path(temporary) / "ledger.json"
            ledger_path.write_bytes(ledger_bytes)
            ledger = load_error_family_ledger(
                ledger_path,
                census=census,
                expected_file_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
                expected_content_sha256=ledger_record["content_sha256"],
            )
            ledger.require_validated_artifact(census=census)

            noncanonical = (json.dumps(ledger_record, indent=2) + "\n").encode()
            noncanonical_path = Path(temporary) / "ledger-pretty.json"
            noncanonical_path.write_bytes(noncanonical)
            with self.assertRaisesRegex(RivalSchemaError, "canonical JSON"):
                load_error_family_ledger(
                    noncanonical_path,
                    census=census,
                    expected_file_sha256=hashlib.sha256(noncanonical).hexdigest(),
                    expected_content_sha256=ledger_record["content_sha256"],
                )
            with self.assertRaisesRegex(RivalSchemaError, "file SHA-256"):
                load_error_family_ledger(
                    ledger_path,
                    census=census,
                    expected_file_sha256="f" * 64,
                    expected_content_sha256=ledger_record["content_sha256"],
                )


class RivalExactCoverageTest(unittest.TestCase):
    def test_cli_binds_fixture_tree_and_never_replaces_an_existing_report(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "coverage.json"
            arguments = [
                "--fixtures",
                str(FIXTURE_DIR),
                "--coverage-design",
                str(FIXTURE_DIR / "coverage_design.json"),
                "--device",
                "cpu",
                "--out",
                str(output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(coverage_main(arguments), 0)
            original = output.read_bytes()
            report = json.loads(original)
            self.assertGreater(report["fixture_file_count"], 0)
            self.assertTrue(report["fixture_tree_sha256"].startswith("sha256:"))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(coverage_main(arguments), 2)
            self.assertEqual(output.read_bytes(), original)

            duplicate = Path(temporary) / "duplicate.json"
            duplicate.write_text('{"schema_id":"x","schema_id":"y"}\n', encoding="utf-8")
            denied_output = Path(temporary) / "denied.json"
            denied = [
                *arguments[:2],
                "--coverage-design",
                str(duplicate),
                "--device",
                "cpu",
                "--out",
                str(denied_output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(coverage_main(denied), 2)
            self.assertFalse(denied_output.exists())

    def test_locked_exact_coverage_design_runs(self) -> None:
        design = json.loads((FIXTURE_DIR / "coverage_design.json").read_text(encoding="utf-8"))
        report = run_coverage_design(design)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["strength_evidence"])

    def test_coverage_design_rejects_coercion_unknown_fields_and_hash_drift(self) -> None:
        original = json.loads((FIXTURE_DIR / "coverage_design.json").read_text(encoding="utf-8"))
        for field, bad_value in (("n_h", True), ("beta_cv", "0.5")):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed.pop("content_sha256")
                changed["exact_cases"][0][field] = bad_value
                with self.assertRaises((CoverageError, RivalSchemaError)):
                    run_coverage_design(attach_content_hash(changed))
        changed = deepcopy(original)
        changed.pop("content_sha256")
        changed["exact_cases"][0]["unknown"] = 1
        with self.assertRaises(RivalSchemaError):
            run_coverage_design(attach_content_hash(changed))
        changed = deepcopy(original)
        changed["exact_cases"][0]["n_h"] = 3
        with self.assertRaisesRegex(RivalSchemaError, "content_sha256 mismatch"):
            run_coverage_design(changed)

    def test_enumerable_distribution_is_unbiased_and_within_declared_error(self) -> None:
        result = enumerate_exact_coverage(
            h_distribution=(
                DiscreteJointOutcome(-1.0, -1.0, 0.5),
                DiscreteJointOutcome(1.0, 1.0, 0.5),
            ),
            l_distribution=(
                DiscreteLowOutcome(-1.0, 0.5),
                DiscreteLowOutcome(1.0, 0.5),
            ),
            n_h=2,
            n_l=2,
            beta_cv=0.5,
            high_difference_width=2.0,
            low_difference_width=2.0,
            allocation=RootErrorAllocation(0.01, 0.01, 0.02),
        )
        self.assertAlmostEqual(result.expected_estimate, result.true_high_mean)
        self.assertLessEqual(result.undercoverage_probability, 0.02)
        self.assertEqual(result.enumerated_sample_panels, 16)

    def test_shifted_low_panel_law_rejects_coverage_claim(self) -> None:
        with self.assertRaisesRegex(CoverageError, "laws to match"):
            enumerate_exact_coverage(
                h_distribution=(DiscreteJointOutcome(0.0, 0.0, 1.0),),
                l_distribution=(DiscreteLowOutcome(1.0, 1.0),),
                n_h=1,
                n_l=1,
                beta_cv=1.0,
                high_difference_width=0.0,
                low_difference_width=1.0,
                allocation=RootErrorAllocation(0.01, 0.01, 0.02),
            )

    def test_declared_ranges_must_cover_the_enumerated_support(self) -> None:
        with self.assertRaisesRegex(CoverageError, "high difference width"):
            enumerate_exact_coverage(
                h_distribution=(
                    DiscreteJointOutcome(-1.0, -1.0, 0.5),
                    DiscreteJointOutcome(1.0, 1.0, 0.5),
                ),
                l_distribution=(
                    DiscreteLowOutcome(-1.0, 0.5),
                    DiscreteLowOutcome(1.0, 0.5),
                ),
                n_h=1,
                n_l=1,
                beta_cv=0.0,
                high_difference_width=1.0,
                low_difference_width=2.0,
                allocation=RootErrorAllocation(0.01, 0.01, 0.02),
            )

    def test_replication_requirement_and_clopper_pearson_bound(self) -> None:
        replications = zero_failure_replications(tolerance=0.01, confidence=0.95)
        self.assertEqual(replications, 299)
        self.assertLessEqual(binomial_upper_confidence_bound(0, replications, alpha=0.05), 0.01)
        general = binomial_upper_confidence_bound(5, 100, alpha=0.05)
        self.assertGreater(general, 0.05)
        self.assertLess(general, 0.12)


class RivalClusteredAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = pinned_allocation_registry(
            (
                (PUBLIC_ROOT_1, "shadow_one_seat"),
                (PUBLIC_ROOT_2, "shadow_one_seat"),
                (PUBLIC_ROOT_3, "shadow_one_seat"),
            ),
            source_game_ids=("game:1", "game:1", "game:2"),
        )

    def summarize(self, rows: tuple[RootMeasurement, ...]):
        return source_game_clustered_summary(
            rows,
            allocation_registry=self.registry,
            expected_allocation_registry_identity=self.registry.identity,
            family_kind="one_seat_instrument",
        )

    def test_registry_complete_rows_are_clustered_by_registered_source_game(self) -> None:
        rows = (
            RootMeasurement(PUBLIC_ROOT_1, "game:1", 1.0),
            RootMeasurement(PUBLIC_ROOT_2, "game:1", 3.0),
            RootMeasurement(PUBLIC_ROOT_3, "game:2", 6.0),
        )
        clustered = self.summarize(rows)
        self.assertEqual(clustered.mean, 4.0)
        self.assertEqual(clustered.standard_error, 2.0)
        self.assertEqual(clustered.source_game_count, 2)
        self.assertEqual(clustered.root_count, 3)

        duplicate_forbidden_iid_unit = (*rows, RootMeasurement(PUBLIC_ROOT_4, "game:1", 2.0))
        self.assertNotEqual(
            iid_root_standard_error_for_diagnostic(rows),
            iid_root_standard_error_for_diagnostic(duplicate_forbidden_iid_unit),
        )
        with self.assertRaisesRegex(AnalysisError, "extra"):
            self.summarize(duplicate_forbidden_iid_unit)

    def test_clustered_summary_rejects_omitted_extra_and_source_mismatched_rows(self) -> None:
        complete = (
            RootMeasurement(PUBLIC_ROOT_1, "game:1", 1.0),
            RootMeasurement(PUBLIC_ROOT_2, "game:1", 3.0),
            RootMeasurement(PUBLIC_ROOT_3, "game:2", 6.0),
        )
        cases = (
            (complete[:-1], "missing"),
            ((*complete, RootMeasurement(PUBLIC_ROOT_4, "game:3", 8.0)), "extra"),
            (
                (
                    RootMeasurement(PUBLIC_ROOT_1, "game:1", 1.0),
                    RootMeasurement(PUBLIC_ROOT_2, "wrong-game", 3.0),
                    RootMeasurement(PUBLIC_ROOT_3, "game:2", 6.0),
                ),
                "source_game_id differs",
            ),
        )
        for rows, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(AnalysisError, reason):
                self.summarize(rows)

    def test_clustered_summary_requires_explicit_matching_registry_identity_and_family(
        self,
    ) -> None:
        rows = (
            RootMeasurement(PUBLIC_ROOT_1, "game:1", 1.0),
            RootMeasurement(PUBLIC_ROOT_2, "game:1", 3.0),
            RootMeasurement(PUBLIC_ROOT_3, "game:2", 6.0),
        )
        with self.assertRaisesRegex(AnalysisError, "identity"):
            source_game_clustered_summary(
                rows,
                allocation_registry=self.registry,
                expected_allocation_registry_identity=SHA_A,
                family_kind="one_seat_instrument",
            )
        with self.assertRaisesRegex(AnalysisError, "unknown error family"):
            source_game_clustered_summary(
                rows,
                allocation_registry=self.registry,
                expected_allocation_registry_identity=self.registry.identity,
                family_kind="unknown-family",
            )
        with self.assertRaisesRegex(AnalysisError, "ExternallyPinnedAllocationRegistry"):
            source_game_clustered_summary(
                rows,
                allocation_registry=replace(self.registry, _external_pin_capability=None),
                expected_allocation_registry_identity=self.registry.identity,
                family_kind="one_seat_instrument",
            )


def power_spec() -> PowerEnvelopeSpec:
    return PowerEnvelopeSpec(
        envelope_id="symbolic:test",
        source_revision="revision:test",
        certified_ranges=(CertifiedStratumRange("late", "sha256:" + "a" * 64, 10.0, 8.0),),
        candidate_count=16,
        finite_training_family_count=1,
        one_seat_family_count=1,
        certified_potential_appeals=80,
        selection_units_per_candidate=2,
        delta_game=0.05,
        n_h_grid=(16,),
        n_l_grid=(64,),
        covariance_grid=(-1.0, 0.0, 1.0),
        variance_high_assumption=4.0,
        variance_low_h_assumption=4.0,
        variance_low_l_assumption=4.0,
        target_gap_grid=(0.0, 2.0),
        activation_frequency_grid=(0.1,),
        timeout_rate_grid=(0.0, 0.01),
        practical_margin=0.25,
        target_confirmed_roots=100,
        calibration_root_requirement=50,
        throughput_assumptions=(
            HypotheticalThroughput(
                "optimistic", 0.05, 0.5, 0.1, 0.1, 16, "symbolic upper scenario"
            ),
            HypotheticalThroughput("central", 0.1, 1.0, 0.2, 0.2, 8, "symbolic middle scenario"),
            HypotheticalThroughput("pessimistic", 0.2, 2.0, 0.4, 0.4, 4, "symbolic lower scenario"),
        ),
        memory_assumptions=(
            MemoryAssumption("central", 16.0, 0.25, 2.0, "unmeasured planning placeholder"),
        ),
    )


class RivalPowerEnvelopeTest(unittest.TestCase):
    def test_grid_is_reproducible_explicitly_symbolic_and_non_funding(self) -> None:
        first = build_power_envelope(power_spec())
        second = build_power_envelope(power_spec())
        self.assertEqual(first, second)
        validate_power_envelope(first)
        self.assertEqual(first["status"], NON_FUNDING_STATUS)
        self.assertFalse(first["can_fund_program"])
        self.assertFalse(first["can_close_program"])
        self.assertTrue(
            all(value == UNRESOLVED for value in first["measured_cost_fields"].values())
        )
        self.assertEqual({row["covariance_assumption"] for row in first["rows"]}, {-1.0, 0.0, 1.0})
        by_covariance = {
            row["covariance_assumption"]: row
            for row in first["rows"]
            if row["target_gap_assumption"] == 0.0
            and row["timeout_rate_assumption"] == 0.0
            and row["throughput_scenario"] == "central"
        }
        self.assertLess(
            by_covariance[1.0]["hypothetical_estimator_variance"],
            by_covariance[0.0]["hypothetical_estimator_variance"],
        )
        self.assertGreater(by_covariance[1.0]["beta_cv_derived_from_covariance"], 0.0)
        self.assertLess(by_covariance[-1.0]["beta_cv_derived_from_covariance"], 0.0)
        unresolved = [row for row in first["rows"] if not row["symbolically_resolves_margin"]]
        self.assertTrue(unresolved)
        self.assertTrue(
            all(
                row["hypothetical_hours_not_decision_grade"] == NO_FINITE_HOURS
                for row in unresolved
            )
        )
        self.assertTrue(
            all(row["selection_units_per_attempted_root"] == 30 for row in first["rows"])
        )

    def test_resolving_a_measured_field_before_p2_rejects(self) -> None:
        envelope = build_power_envelope(power_spec())
        envelope["measured_cost_fields"]["post_d1_john0_gpu_hours"] = 12.0
        with self.assertRaisesRegex(RivalSchemaError, "UNRESOLVED"):
            validate_power_envelope(envelope)
        forged = deepcopy(build_power_envelope(power_spec()))
        forged["can_fund_program"] = True
        with self.assertRaisesRegex(RivalSchemaError, "cannot fund"):
            validate_power_envelope(forged)


if __name__ == "__main__":
    unittest.main()
