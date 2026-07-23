from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.aaaaa_wildlife_merge_certificates import (
    validate_hawk_one_loss_certificates,
    validate_motif_certificate,
    validate_zero_hawk_certificates,
)

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT / "docs" / "v3" / "evidence" / "aaaaa_motif_certificate_3_6_6_0_5_2026-07-23.json"
)
ZERO_HAWK_CERTIFICATE = (
    ROOT / "docs" / "v3" / "evidence" / "aaaaa_zero_hawk_certificates_2026-07-23.json"
)
HAWK_CERTIFICATE = (
    ROOT / "docs" / "v3" / "evidence" / "aaaaa_hawk_one_loss_certificates_2026-07-23.json"
)


class AaaaaWildlifeMergeCertificatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
        cls.row = {
            "counts": cls.certificate["counts"],
            "optimum": cls.certificate["incumbent"]["score"],
            "score_breakdown": cls.certificate["incumbent"]["score_breakdown"],
            "tokens": cls.certificate["incumbent"]["tokens"],
            "proof_method": "incomplete_timeout",
            "proof_complete": False,
            "attempts": [],
        }

    def test_valid_certificate_promotes_row(self) -> None:
        result = validate_motif_certificate(CERTIFICATE, self.row)
        self.assertTrue(result["proof_complete"])
        self.assertEqual(result["optimum"], 61)
        self.assertEqual(result["proof_method"], "standalone_maximum_motif_incompatibility")

    def test_wrong_catalog_counts_fail_closed(self) -> None:
        row = copy.deepcopy(self.row)
        row["counts"] = [4, 6, 6, 0, 4]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_motif_certificate(CERTIFICATE, row, reproduce=False)

    def test_zero_hawk_certificate_promotes_all_three_rows(self) -> None:
        certificate = json.loads(ZERO_HAWK_CERTIFICATE.read_text(encoding="utf-8"))
        rows = {
            tuple(result["counts"]): {
                "counts": result["counts"],
                "optimum": result["incumbent"]["score"],
                "score_breakdown": result["incumbent"]["score_breakdown"],
                "tokens": result["incumbent"]["tokens"],
                "proof_method": "incomplete_timeout",
                "proof_complete": False,
                "attempts": [],
            }
            for result in certificate["results"]
        }
        promoted = validate_zero_hawk_certificates(
            ZERO_HAWK_CERTIFICATE, rows, reproduce=False
        )
        self.assertEqual(len(promoted), 3)
        self.assertTrue(all(row["proof_complete"] for _, row in promoted))

    def test_hawk_certificate_promotes_both_rows(self) -> None:
        certificate = json.loads(HAWK_CERTIFICATE.read_text(encoding="utf-8"))
        rows = {
            tuple(result["counts"]): {
                "counts": result["counts"],
                "optimum": result["incumbent"]["score"],
                "score_breakdown": result["incumbent"]["score_breakdown"],
                "tokens": result["incumbent"]["tokens"],
                "proof_method": "incomplete_timeout",
                "proof_complete": False,
                "attempts": [],
            }
            for result in certificate["results"]
        }
        promoted = validate_hawk_one_loss_certificates(
            HAWK_CERTIFICATE, rows, reproduce=False
        )
        self.assertEqual(len(promoted), 2)
        self.assertTrue(all(row["proof_complete"] for _, row in promoted))


if __name__ == "__main__":
    unittest.main()
