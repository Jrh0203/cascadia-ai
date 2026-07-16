"""Cross-language wire and hand-computed bound tests."""

import json
import math
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from cascadiav3.rival.bounds import (
    BoundError,
    HighOnlyErrorAllocation,
    RootErrorAllocation,
    TransformedWidths,
    fixed_high_only_hoeffding_lower_bound,
    fixed_hoeffding_lower_bound,
    transformed_widths,
    transformed_widths_from_certificate,
    verify_bound_certificate,
)
from cascadiav3.rival.schema import RivalSchemaError, sha256_hex

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rival" / "global_bound_certificate_v1.json"


def rust_global_certificate() -> dict[str, object]:
    """Load the checked-in bytes emitted by the Rust contract utility."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class RivalBoundCertificateTest(unittest.TestCase):
    def test_locked_fixture_is_the_rust_certificate_not_a_python_rederivation(self) -> None:
        record = rust_global_certificate()
        self.assertEqual(
            record["certificate_sha256"],
            "sha256:67fa21e1f4e887f73a1f0f4e22397ca23f79ca67972b44e34f94f385734eec64",
        )
        certificate = verify_bound_certificate(
            record,
            expected_certificate_sha256=record["certificate_sha256"],
            expected_ruleset=record["ruleset"],
        )
        self.assertEqual(certificate.high.width, 332)

    def test_python_consumes_exact_rust_wire_and_maps_common_range(self) -> None:
        record = rust_global_certificate()
        certificate = verify_bound_certificate(
            record,
            expected_certificate_sha256=record["certificate_sha256"],
            expected_ruleset=record["ruleset"],
        )
        self.assertEqual(certificate.terminal_score.minimum, 0)
        self.assertEqual(certificate.terminal_score.maximum, 166)
        self.assertEqual(certificate.high.width, 332)
        self.assertEqual(certificate.low.width, 332)
        widths = transformed_widths_from_certificate(certificate, beta_cv=-0.5)
        self.assertEqual(widths.high_corrected, 498)
        self.assertEqual(widths.low_correction, 166)

    def test_bound_capability_rejects_direct_and_mutated_dataclasses(self) -> None:
        record = rust_global_certificate()
        certificate = verify_bound_certificate(
            record,
            expected_certificate_sha256=record["certificate_sha256"],
            expected_ruleset=record["ruleset"],
        )
        direct = replace(certificate, _validation_capability=None)
        with self.assertRaisesRegex(BoundError, "artifact validator"):
            transformed_widths_from_certificate(direct, beta_cv=0.5)
        with self.assertRaisesRegex(RivalSchemaError, "does not match"):
            replace(certificate, content_sha256="b" * 64)

    def test_certificate_is_caller_pinned_and_every_mutation_fails(self) -> None:
        record = rust_global_certificate()
        pinned = record["certificate_sha256"]
        for field, replacement in (
            ("score_difference_width", 331),
            ("terminal_score_max", 165),
            ("authority_id", "python-derived"),
            ("scope", "changed"),
        ):
            mutated = deepcopy(record)
            mutated[field] = replacement
            with self.subTest(field=field), self.assertRaises(RivalSchemaError):
                verify_bound_certificate(
                    mutated,
                    expected_certificate_sha256=pinned,
                    expected_ruleset=record["ruleset"],
                )
        unknown = deepcopy(record)
        unknown["observed_sample_max"] = 100
        with self.assertRaisesRegex(RivalSchemaError, "unknown"):
            verify_bound_certificate(
                unknown,
                expected_certificate_sha256=pinned,
                expected_ruleset=record["ruleset"],
            )

    def test_unqualified_rust_hashes_reject(self) -> None:
        record = rust_global_certificate()
        pinned = record["certificate_sha256"]
        record["certificate_sha256"] = record["certificate_sha256"].removeprefix("sha256:")
        with self.assertRaisesRegex(RivalSchemaError, "Rust 'sha256:' wire"):
            verify_bound_certificate(
                record,
                expected_certificate_sha256=pinned,
                expected_ruleset=record["ruleset"],
            )

    def test_rehashed_caller_substitution_cannot_impersonate_rust_authority(self) -> None:
        record = rust_global_certificate()
        record.pop("certificate_sha256")
        record["terminal_score_max"] = 165
        record["certificate_sha256"] = "sha256:" + sha256_hex(record)
        with self.assertRaisesRegex(RivalSchemaError, "substitute a different"):
            verify_bound_certificate(
                record,
                expected_certificate_sha256=record["certificate_sha256"],
                expected_ruleset=record["ruleset"],
            )


class RivalHoeffdingTest(unittest.TestCase):
    def test_general_widths_and_hand_calculated_penalties(self) -> None:
        widths = transformed_widths(
            beta_cv=-0.5, high_difference_width=10.0, low_difference_width=4.0
        )
        self.assertEqual(widths.high_corrected, 12.0)
        self.assertEqual(widths.low_correction, 2.0)
        allocation = RootErrorAllocation(delta_h=0.01, delta_l=0.02, delta_root=0.03)
        result = fixed_hoeffding_lower_bound(
            high_corrected_mean=3.0,
            low_correction_mean=1.0,
            widths=widths,
            allocation=allocation,
            n_h=100,
            n_l=200,
        )
        expected_h = 12.0 * math.sqrt(math.log(100.0) / 200.0)
        expected_l = 2.0 * math.sqrt(math.log(50.0) / 400.0)
        self.assertAlmostEqual(result.high_penalty, expected_h)
        self.assertAlmostEqual(result.low_penalty, expected_l)
        self.assertAlmostEqual(result.lower_bound, 4.0 - expected_h - expected_l)

    def test_common_range_is_one_plus_abs_beta_special_case(self) -> None:
        for beta in (-2.0, 0.0, 0.25):
            widths = transformed_widths(
                beta_cv=beta, high_difference_width=7.0, low_difference_width=7.0
            )
            self.assertEqual(widths.high_corrected, (1.0 + abs(beta)) * 7.0)
            self.assertEqual(widths.low_correction, abs(beta) * 7.0)

    def test_error_allocation_and_invalid_widths_fail_closed(self) -> None:
        with self.assertRaises(BoundError):
            RootErrorAllocation(delta_h=0.02, delta_l=0.02, delta_root=0.03)
        with self.assertRaises(BoundError):
            transformed_widths(beta_cv=1.0, high_difference_width=-1.0, low_difference_width=2.0)
        with self.assertRaises(RivalSchemaError):
            fixed_hoeffding_lower_bound(
                high_corrected_mean=0.0,
                low_correction_mean=0.0,
                widths=TransformedWidths(math.nan, 0.0),
                allocation=RootErrorAllocation(0.01, 0.01, 0.02),
                n_h=1,
                n_l=1,
            )

    def test_high_only_control_has_one_error_term_and_no_l_panel(self) -> None:
        result = fixed_high_only_hoeffding_lower_bound(
            high_mean=2.0,
            certified_width=10.0,
            allocation=HighOnlyErrorAllocation(delta_h=0.01, delta_root=0.01),
            n_h=100,
        )
        expected = 10.0 * math.sqrt(math.log(100.0) / 200.0)
        self.assertAlmostEqual(result.penalty, expected)
        self.assertAlmostEqual(result.lower_bound, 2.0 - expected)
        self.assertFalse(hasattr(result, "n_l"))


if __name__ == "__main__":
    unittest.main()
