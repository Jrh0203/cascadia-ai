"""Algebraic and perturbation tests for the fixed two-panel estimator."""

import hashlib
import json
import math
import unittest
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.rival.multifidelity import (
    CoefficientBinding,
    HighDifference,
    LowDifference,
    MultifidelityError,
    PairedDifference,
    estimate_fixed_panels,
    estimate_high_fidelity_only,
    estimator_variance_general,
    load_coefficient_calibration,
    negate_low,
    negate_paired,
    optimal_beta_equal_law,
    optimal_beta_for_registered_design,
    optimal_beta_general,
    validate_coefficient_calibration,
)
from cascadiav3.rival.schema import (
    RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
    RivalSchemaError,
    attach_content_hash,
    canonical_json_bytes,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def coefficient_artifact_record() -> dict[str, object]:
    return attach_content_hash(
        {
            "schema_id": RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
            "coefficient_id": "beta:locked",
            "beta_cv": -0.25,
            "calibration_cohort_id": "calibration:locked",
            "calibration_source_corpus_sha256": SHA_A,
            "calibration_root_index_sha256": SHA_B,
            "calibration_data_sha256": SHA_A,
            "deployment_design_id": "manifest:coverage",
            "deployment_design_sha256": SHA_B,
            "incumbent_policy_id": SHA_A,
            "low_policy_id": SHA_B,
            "sampler_id": SHA_A,
            "allocation_id": SHA_B,
            "low_expectation_h_id": "expectation:locked",
            "low_expectation_l_id": "expectation:locked",
            "low_law_h_id": "law:locked",
            "low_law_l_id": "law:locked",
            "max_abs_beta": 2.0,
            "estimator_identity": SHA_B,
        }
    )


class RivalMultifidelityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = (
            PairedDifference("h0", 4.0, 2.0),
            PairedDifference("h1", 2.0, -2.0),
        )
        self.l = (LowDifference("l0", 1.0), LowDifference("l1", -1.0))

    def estimate(self, beta: float = 0.5):
        return estimate_fixed_panels(self.h, self.l, beta_cv=beta, expected_n_h=2, expected_n_l=2)

    def test_beta_zero_is_exact_high_fidelity_mean(self) -> None:
        result = self.estimate(0.0)
        self.assertEqual(result.estimate, 3.0)
        self.assertEqual(result.high_corrected_mean, 3.0)
        self.assertEqual(result.low_correction_mean, 0.0)

    def test_high_only_control_is_separate_and_never_requires_l(self) -> None:
        result = estimate_high_fidelity_only(
            (HighDifference("h0", 4.0), HighDifference("h1", 2.0)),
            expected_n_h=2,
        )
        self.assertEqual(result.estimate, 3.0)
        self.assertFalse(hasattr(result, "beta_cv"))
        self.assertFalse(hasattr(result, "n_l"))

    def test_action_orientation_swap_negates_every_estimate(self) -> None:
        forward = self.estimate().estimate
        reverse = estimate_fixed_panels(
            negate_paired(self.h),
            negate_low(self.l),
            beta_cv=0.5,
            expected_n_h=2,
            expected_n_l=2,
        ).estimate
        self.assertEqual(reverse, -forward)

    def test_ordering_does_not_change_result(self) -> None:
        forward = self.estimate().estimate
        reordered = estimate_fixed_panels(
            tuple(reversed(self.h)),
            tuple(reversed(self.l)),
            beta_cv=0.5,
            expected_n_h=2,
            expected_n_l=2,
        ).estimate
        self.assertEqual(reordered, forward)

    def test_general_optimizer_reduces_to_special_equal_law_formula(self) -> None:
        special = optimal_beta_equal_law(n_h=4, n_l=12, covariance_high_low=3.0, variance_low=2.0)
        general = optimal_beta_general(
            n_h=4,
            n_l=12,
            covariance_high_low_h=3.0,
            variance_low_h=2.0,
            variance_low_l=2.0,
        )
        self.assertAlmostEqual(special, 12 / 16 * 3 / 2)
        self.assertAlmostEqual(general, special)
        selected = optimal_beta_for_registered_design(
            n_h=4,
            n_l=12,
            covariance_high_low_h=3.0,
            variance_high=5.0,
            variance_low_h=2.0,
            variance_low_l=2.0,
            low_expectation_h_id="expectation:1",
            low_expectation_l_id="expectation:1",
            low_law_h_id="law:1",
            low_law_l_id="law:1",
            equal_law_assumptions_certified=True,
        )
        self.assertEqual(selected.method, "certified_equal_low_law")
        self.assertAlmostEqual(selected.beta_cv, special)

    def test_general_optimizer_handles_shifted_panel_variance_and_negative_covariance(self) -> None:
        positive = optimal_beta_general(
            n_h=5,
            n_l=20,
            covariance_high_low_h=2.0,
            variance_low_h=4.0,
            variance_low_l=9.0,
        )
        negative = optimal_beta_general(
            n_h=5,
            n_l=20,
            covariance_high_low_h=-2.0,
            variance_low_h=4.0,
            variance_low_l=9.0,
        )
        self.assertAlmostEqual(positive, 2.0 / (4.0 + 5 / 20 * 9.0))
        self.assertAlmostEqual(negative, -positive)
        selected = optimal_beta_for_registered_design(
            n_h=5,
            n_l=20,
            covariance_high_low_h=2.0,
            variance_high=4.0,
            variance_low_h=4.0,
            variance_low_l=9.0,
            low_expectation_h_id="expectation:1",
            low_expectation_l_id="expectation:1",
            low_law_h_id="law:h",
            low_law_l_id="law:l",
            equal_law_assumptions_certified=False,
        )
        self.assertEqual(selected.method, "general_independent_panel_variance")
        self.assertAlmostEqual(selected.beta_cv, positive)
        with self.assertRaisesRegex(MultifidelityError, "biased"):
            optimal_beta_for_registered_design(
                n_h=5,
                n_l=20,
                covariance_high_low_h=2.0,
                variance_low_h=4.0,
                variance_low_l=9.0,
                variance_high=1.0,
                low_expectation_h_id="expectation:h",
                low_expectation_l_id="expectation:l",
                low_law_h_id="law:h",
                low_law_l_id="law:l",
                equal_law_assumptions_certified=False,
            )
        with self.assertRaisesRegex(MultifidelityError, "Cauchy-Schwarz"):
            optimal_beta_for_registered_design(
                n_h=5,
                n_l=20,
                covariance_high_low_h=3.0,
                variance_low_h=1.0,
                variance_low_l=1.0,
                variance_high=1.0,
                low_expectation_h_id="expectation",
                low_expectation_l_id="expectation",
                low_law_h_id="law",
                low_law_l_id="law",
                equal_law_assumptions_certified=True,
            )

    def test_general_variance_agrees_with_exact_enumeration(self) -> None:
        # H is (+/-1,+/-1) with perfect covariance; independent L is +/-1.
        beta = 0.5
        values: list[tuple[float, float]] = []
        for high_low, extra_low in product(((1.0, 1.0), (-1.0, -1.0)), (1.0, -1.0)):
            estimate = high_low[0] - beta * high_low[1] + beta * extra_low
            values.append((estimate, 0.25))
        mean = sum(value * probability for value, probability in values)
        enumerated = sum((value - mean) ** 2 * probability for value, probability in values)
        formula = estimator_variance_general(
            n_h=1,
            n_l=1,
            beta_cv=beta,
            variance_high=1.0,
            variance_low_h=1.0,
            variance_low_l=1.0,
            covariance_high_low_h=1.0,
        )
        self.assertAlmostEqual(enumerated, 0.5)
        self.assertAlmostEqual(formula, enumerated)

    def test_impossible_covariance_rejects_even_when_beta_is_zero(self) -> None:
        with self.assertRaisesRegex(MultifidelityError, "Cauchy-Schwarz"):
            estimator_variance_general(
                n_h=1,
                n_l=1,
                beta_cv=0.0,
                variance_high=1.0,
                variance_low_h=1.0,
                variance_low_l=1.0,
                covariance_high_low_h=2.0,
            )

    def test_incomplete_duplicate_overlap_nan_and_zero_variance_reject(self) -> None:
        with self.assertRaisesRegex(MultifidelityError, "incomplete H"):
            estimate_fixed_panels(self.h[:1], self.l, beta_cv=0.0, expected_n_h=2, expected_n_l=2)
        with self.assertRaisesRegex(MultifidelityError, "overlap"):
            estimate_fixed_panels(
                self.h,
                (LowDifference("h0", 1.0), LowDifference("l1", -1.0)),
                beta_cv=0.0,
                expected_n_h=2,
                expected_n_l=2,
            )
        with self.assertRaises(MultifidelityError):
            PairedDifference("bad", math.nan, 0.0)
        with self.assertRaisesRegex(MultifidelityError, "positive"):
            optimal_beta_equal_law(n_h=2, n_l=2, covariance_high_low=1.0, variance_low=0.0)

    def test_coefficient_is_bound_to_disjoint_calibration_and_exact_design(self) -> None:
        binding = CoefficientBinding(
            coefficient_id="beta:1",
            beta_cv=-0.25,
            calibration_cohort_id="calibration:1",
            deployment_design_id="coverage:1",
            deployment_design_sha256="sha256:" + "a" * 64,
            incumbent_policy_id="b:1",
            low_policy_id="low:1",
            sampler_id="sampler:1",
            allocation_id="allocation:1",
            low_expectation_h_id="low-expectation:1",
            low_expectation_l_id="low-expectation:1",
            low_law_h_id="low-law:1",
            low_law_l_id="low-law:1",
            max_abs_beta=2.0,
        )
        binding.require_design(
            deployment_design_id="coverage:1",
            deployment_design_sha256="sha256:" + "a" * 64,
            incumbent_policy_id="b:1",
            low_policy_id="low:1",
            sampler_id="sampler:1",
            allocation_id="allocation:1",
        )
        with self.assertRaisesRegex(MultifidelityError, "sampler_id"):
            binding.require_design(
                deployment_design_id="coverage:1",
                deployment_design_sha256="sha256:" + "a" * 64,
                incumbent_policy_id="b:1",
                low_policy_id="low:1",
                sampler_id="changed",
                allocation_id="allocation:1",
            )
        with self.assertRaisesRegex(MultifidelityError, "disjoint"):
            CoefficientBinding(
                coefficient_id="beta:bad",
                beta_cv=0.0,
                calibration_cohort_id="same",
                deployment_design_id="same",
                deployment_design_sha256="sha256:" + "a" * 64,
                incumbent_policy_id="b",
                low_policy_id="l",
                sampler_id="s",
                allocation_id="a",
                low_expectation_h_id="expectation",
                low_expectation_l_id="expectation",
                low_law_h_id="law",
                low_law_l_id="law",
                max_abs_beta=1.0,
            )

    def test_calibration_artifact_is_strict_pinned_and_validator_capable(self) -> None:
        record = coefficient_artifact_record()
        binding = validate_coefficient_calibration(
            record,
            expected_content_sha256=record["content_sha256"],
        )
        self.assertTrue(binding.is_validated_artifact)
        self.assertEqual(binding.identity, "sha256:" + record["content_sha256"])
        binding.require_validated_artifact()

        direct = replace(binding, _validation_capability=None)
        self.assertFalse(direct.is_validated_artifact)
        with self.assertRaisesRegex(MultifidelityError, "artifact validator"):
            direct.require_validated_artifact()
        with self.assertRaisesRegex(MultifidelityError, "does not match"):
            replace(binding, beta_cv=0.0)

        substituted = deepcopy(record)
        substituted["beta_cv"] = 0.0
        with self.assertRaisesRegex(RivalSchemaError, "content_sha256 mismatch"):
            validate_coefficient_calibration(
                substituted,
                expected_content_sha256=record["content_sha256"],
            )
        unknown = deepcopy(record)
        unknown["unregistered_estimator_option"] = True
        with self.assertRaisesRegex(RivalSchemaError, "unknown"):
            validate_coefficient_calibration(
                unknown,
                expected_content_sha256=record["content_sha256"],
            )

    def test_calibration_loader_requires_exact_canonical_bytes_and_file_pin(self) -> None:
        record = coefficient_artifact_record()
        canonical = canonical_json_bytes(record) + b"\n"
        with TemporaryDirectory() as temporary:
            canonical_path = Path(temporary) / "coefficient.json"
            canonical_path.write_bytes(canonical)
            file_sha256 = hashlib.sha256(canonical).hexdigest()
            loaded = load_coefficient_calibration(
                canonical_path,
                expected_file_sha256=file_sha256,
                expected_content_sha256=record["content_sha256"],
            )
            self.assertTrue(loaded.is_validated_artifact)
            with self.assertRaisesRegex(RivalSchemaError, "file SHA-256"):
                load_coefficient_calibration(
                    canonical_path,
                    expected_file_sha256="f" * 64,
                    expected_content_sha256=record["content_sha256"],
                )

            noncanonical_path = Path(temporary) / "noncanonical.json"
            noncanonical = (json.dumps(record, indent=2, sort_keys=False) + "\n").encode()
            noncanonical_path.write_bytes(noncanonical)
            with self.assertRaisesRegex(RivalSchemaError, "canonical JSON"):
                load_coefficient_calibration(
                    noncanonical_path,
                    expected_file_sha256=hashlib.sha256(noncanonical).hexdigest(),
                    expected_content_sha256=record["content_sha256"],
                )

            duplicate_path = Path(temporary) / "duplicate.json"
            duplicate = b'{"schema_id":"x","schema_id":"y"}\n'
            duplicate_path.write_bytes(duplicate)
            with self.assertRaisesRegex(RivalSchemaError, "duplicate JSON key"):
                load_coefficient_calibration(
                    duplicate_path,
                    expected_file_sha256=hashlib.sha256(duplicate).hexdigest(),
                    expected_content_sha256=record["content_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
