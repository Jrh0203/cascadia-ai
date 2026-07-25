from __future__ import annotations

import pytest

import cluster_build_push as subject


def test_manifest_push_digest_requires_exactly_one_digest_line() -> None:
    digest = "sha256:" + "a" * 64
    assert subject._manifest_push_digest(f"pushing\n{digest}\n") == digest
    with pytest.raises(SystemExit, match="uniquely"):
        subject._manifest_push_digest("no digest")
