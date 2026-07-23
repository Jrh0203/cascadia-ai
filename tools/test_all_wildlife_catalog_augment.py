from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_catalog_augment import (
    _validate_aaaaa_certificate,
    augment,
)
from tools.test_all_wildlife_proof_catalog import _candidate_catalog

AAAAA_CERTIFICATE = Path(
    "docs/v3/evidence/aaaaa_wildlife_optimum_2026-07-22.json"
)


def _base_catalog(candidates: dict[str, object]) -> dict[str, object]:
    rows = []
    for candidate in candidates["candidates"]:  # type: ignore[union-attr]
        ruleset = candidate["ruleset"]
        rows.append(
            {
                "index": candidate["index"],
                "ruleset": ruleset,
                "proof_complete": True,
                "optimum": candidate["score"],
                "score_breakdown": candidate["score_breakdown"],
                "counts": candidate["counts"],
                "tokens": candidate["tokens"],
                "unresolved_counts": [],
                "proof_paths": [],
            }
        )
    return {
        "schema": "all-wildlife-optimal-catalog-v1",
        "proof_complete": True,
        "completed_rulesets": len(rows),
        "ruleset_count": len(rows),
        "token_count": rules.TOKEN_COUNT,
        "count_cap": rules.COUNT_CAP,
        "candidate_sha256": "old-candidate",
        "results": rows,
    }


def test_real_aaaaa_certificate_has_complete_exact_coverage() -> None:
    row, identity = _validate_aaaaa_certificate(AAAAA_CERTIFICATE)

    assert row["ruleset"] == "AAAAA"
    assert row["proof_complete"]
    assert row["optimum"] == 68
    assert row["unresolved_counts"] == []
    assert identity["excluded_allocations"] == 128


def test_augment_imports_aaaaa_and_validates_every_row(tmp_path) -> None:
    candidates_payload = _candidate_catalog()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(candidates_payload))
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_base_catalog(candidates_payload)))

    result = augment(base, candidates, AAAAA_CERTIFICATE)

    assert result["results"][0]["proof_complete"]
    assert result["results"][0]["optimum"] == 68
    assert result["completed_rulesets"] >= 1
    assert result["candidate_sha256"]
    assert result["base_catalog_sha256"]
    assert result["production_response_sha256"] is None


def test_augment_rejects_incomplete_external_certificate(tmp_path) -> None:
    payload = json.loads(AAAAA_CERTIFICATE.read_text())
    payload["proof_complete"] = False
    certificate = tmp_path / "bad.json"
    certificate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="certificate contract mismatch"):
        _validate_aaaaa_certificate(certificate)
