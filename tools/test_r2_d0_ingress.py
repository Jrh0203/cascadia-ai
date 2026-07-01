from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest
from r2_d0 import ingress as subject
from r2_d0.canonical import D0_RUN_ID, D0Error


def verification(*, report_sha256: str = "a" * 64) -> dict[str, Any]:
    return {
        "manifest": {
            "run_id": D0_RUN_ID,
            "host": "john3",
            "cycle_id": "qualification",
            "manifest_sha256": "b" * 64,
        },
        "packet": {"packet_sha256": "c" * 64},
        "report": {"phase": "verify", "operation": "verify-runtime"},
        "report_sha256": report_sha256,
    }


def storage(root: Path) -> dict[str, Any]:
    return {
        "status": "pass",
        "root": str(root),
        "host_identity_sha256": "d" * 64,
    }


def test_result_ingress_is_john1_owned_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(subject, "CANONICAL_ROOT", root)
    commits: list[Path] = []
    archive = b"signed-canonical-bundle"
    kwargs = {
        "archive": archive,
        "public_key": b"fixture",
        "campaign_root": root,
        "storage_verifier": lambda **_kwargs: storage(root),
        "bundle_verifier": lambda *_args, **_kwargs: verification(),
        "commit_verifier": lambda path: commits.append(path),
    }

    first = subject.install_result_ingress(**kwargs)
    second = subject.install_result_ingress(**kwargs)

    assert first["disposition"] == "installed"
    assert second["disposition"] == "present"
    destination = root / first["receipt"]["destination_relative"]
    assert commits == [destination, destination]
    assert (destination / "bundle.tar").read_bytes() == archive
    assert stat.S_IMODE((destination / "bundle.tar").stat().st_mode) == 0o400
    assert stat.S_IMODE((destination / "ingress-receipt.json").stat().st_mode) == 0o400
    assert not list(destination.parent.glob(".*.partial-*"))


def test_result_ingress_collision_and_wrong_root_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(subject, "CANONICAL_ROOT", root)
    common = {
        "public_key": b"fixture",
        "campaign_root": root,
        "storage_verifier": lambda **_kwargs: storage(root),
        "bundle_verifier": lambda *_args, **_kwargs: verification(),
        "commit_verifier": lambda _path: None,
    }
    result = subject.install_result_ingress(b"first", **common)
    with pytest.raises(D0Error, match="different bytes"):
        subject.install_result_ingress(b"second", **common)
    destination = root / result["receipt"]["destination_relative"]
    assert (destination / "bundle.tar").read_bytes() == b"first"

    with pytest.raises(D0Error, match="not John1"):
        subject.install_result_ingress(
            b"first",
            **{**common, "campaign_root": tmp_path / "other"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("host", "john4"), ("run_id", "other"), ("cycle_id", "unknown")],
)
def test_result_ingress_rejects_noncampaign_identity(field: str, value: str) -> None:
    value_document = verification()
    value_document["manifest"][field] = value
    with pytest.raises(D0Error, match="identity differs"):
        subject.result_ingress_relative(value_document)
