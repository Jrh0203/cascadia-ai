from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.aaaaa_wildlife_merge_certificates import validate_motif_certificate

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT / "docs" / "v3" / "evidence" / "aaaaa_motif_certificate_3_6_6_0_5_2026-07-23.json"
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


if __name__ == "__main__":
    unittest.main()
