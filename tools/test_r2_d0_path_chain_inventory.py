from __future__ import annotations

import json
import os
from pathlib import Path

import r2_d0_path_chain_inventory as inventory


def test_path_chain_reports_symlink_ancestor_and_resolved_content(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "payload.bin"
    target.write_bytes(b"immutable")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    report = inventory.inspect_root(link / "payload.bin", 1024)
    assert [item["path"] for item in report["symlink_ancestors"]] == [str(link)]
    assert report["resolved_path"] == str(target)
    assert report["resolved_content"] == {
        "kind": "full-file",
        "sha256": inventory.sha256_bytes(b"immutable"),
    }


def test_snapshot_is_stable_and_detects_content_change(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"before")
    before = inventory.snapshot([payload], 1024)
    assert before == inventory.snapshot([payload], 1024)
    payload.write_bytes(b"after")
    after = inventory.snapshot([payload], 1024)
    assert before["snapshot_sha256"] != after["snapshot_sha256"]


def test_homebrew_owner_binds_install_receipt(tmp_path: Path) -> None:
    keg = tmp_path / "Cellar" / "docker-buildx" / "0.28.0"
    keg.mkdir(parents=True)
    receipt = keg / "INSTALL_RECEIPT.json"
    receipt.write_text(json.dumps({"source": "bottle"}))
    binary = keg / "bin" / "docker-buildx"
    binary.parent.mkdir()
    binary.write_bytes(b"binary")
    owner = inventory._homebrew_owner(binary, 1024)
    assert owner is not None
    assert owner["formula"] == "docker-buildx"
    assert owner["version"] == "0.28.0"
    assert owner["install_receipt"]["content_sha256"] == inventory.sha256_bytes(
        receipt.read_bytes()
    )


def test_path_record_never_follows_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"value")
    link = tmp_path / "link"
    os.symlink(target, link)
    record = inventory.path_record(link)
    assert record["type"] == "symlink"
    assert record["symlink_target"] == str(target)
