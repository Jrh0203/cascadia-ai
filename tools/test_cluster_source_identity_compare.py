from __future__ import annotations

import json
from pathlib import Path

import cluster_source_identity_compare as compare
import pytest


def _write_identity(path: Path, *, host: str, bundle: str) -> None:
    path.write_text(
        json.dumps(
            {
                "identity_kind": compare.IDENTITY_KIND,
                "host": host,
                "bundle_sha256": bundle,
            }
        )
    )


def test_compare_accepts_matching_preregistered_bundles(
    tmp_path: Path,
) -> None:
    bundle = "a" * 64
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_identity(left, host="john1", bundle=bundle)
    _write_identity(right, host="john4", bundle=bundle)
    report = compare.compare_identities(
        [left, right],
        expected_bundle_sha256=bundle,
    )
    assert report["all_identities_match"]
    assert report["hosts"] == ["john1", "john4"]


def test_compare_rejects_source_drift(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_identity(left, host="john1", bundle="a" * 64)
    _write_identity(right, host="john4", bundle="b" * 64)
    with pytest.raises(compare.SourceIdentityError, match="differ"):
        compare.compare_identities([left, right])
