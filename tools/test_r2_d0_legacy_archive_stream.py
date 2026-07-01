from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path

import pytest
from r2_d0.legacy_archive_stream import (
    LegacyArchiveError,
    emit_archive,
    load_manifest,
    verify_archive,
)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "legacy"
    (root / "nested").mkdir(parents=True)
    first = root / "nested/first.bin"
    first.write_bytes(b"hardlinked-payload")
    os.link(first, root / "nested/second.bin")
    (root / "plain.txt").write_bytes(b"plain-payload")
    paths = [".", "nested", "nested/first.bin", "nested/second.bin", "plain.txt"]
    entries = []
    unique: dict[tuple[int, int], int] = {}
    for relative in paths:
        path = root if relative == "." else root / relative
        details = os.lstat(path)
        is_dir = stat.S_ISDIR(details.st_mode)
        digest = (
            hashlib.sha256(("tree:" + relative).encode()).hexdigest()
            if is_dir
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        entries.append(
            {
                "allocated_bytes": details.st_blocks * 512,
                "device": details.st_dev,
                "flags_after": getattr(details, "st_flags", 0),
                "flags_before": 0,
                "inode": details.st_ino,
                "mode": f"{stat.S_IMODE(details.st_mode):04o}",
                "mtime_ns": details.st_mtime_ns,
                "nlink": details.st_nlink,
                "path": relative,
                "sha256": digest,
                "size": details.st_size,
                "type": "directory" if is_dir else "file",
            }
        )
        if not is_dir:
            unique.setdefault((details.st_dev, details.st_ino), details.st_size)
    hardlink = next(item for item in entries if item["path"] == "nested/first.bin")
    manifest = {
        "schema_id": "cascadia.r2-map.john3-legacy-native-workspace-freeze.v1",
        "schema_version": 1,
        "host": "john3",
        "root": "/Users/john3/cascadia-bench/r2-map-v1",
        "entries": entries,
        "hardlink_groups": [
            {
                "allocated_bytes": hardlink["allocated_bytes"],
                "device": hardlink["device"],
                "group_id": "g" * 64,
                "inode": hardlink["inode"],
                "logical_bytes": hardlink["size"] * 2,
                "members": ["nested/first.bin", "nested/second.bin"],
                "nlink": 2,
                "sha256": hardlink["sha256"],
                "size": hardlink["size"],
            }
        ],
        "totals": {
            "entry_count": len(entries),
            "unique_regular_bytes": sum(unique.values()),
            "root_tree_sha256": entries[0]["sha256"],
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    return root, _canonical(manifest)


def test_archive_is_deterministic_verified_and_preserves_hardlinks(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    first = io.BytesIO()
    second = io.BytesIO()
    emit_archive(root, manifest, first)
    emit_archive(root, manifest, second)
    assert first.getvalue() == second.getvalue()

    proof = verify_archive(io.BytesIO(first.getvalue()), manifest)
    assert proof["status"] == "pass"
    assert proof["entry_count"] == 5
    assert proof["hardlink_group_count"] == 1
    assert proof["archive_sha256"] == hashlib.sha256(first.getvalue()).hexdigest()


def test_manifest_source_and_archive_tamper_fail_closed(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    archive = io.BytesIO()
    emit_archive(root, manifest, archive)
    (root / "plain.txt").write_bytes(b"changed")
    with pytest.raises(LegacyArchiveError, match=r"metadata differs|content differs"):
        emit_archive(root, manifest, io.BytesIO())

    value = bytearray(archive.getvalue())
    index = value.find(b"plain-payload")
    assert index >= 0
    value[index] ^= 1
    with pytest.raises(LegacyArchiveError, match="content differs"):
        verify_archive(io.BytesIO(value), manifest)

    decoded = load_manifest(manifest)
    decoded["totals"]["entry_count"] += 1
    with pytest.raises(LegacyArchiveError, match=r"identity differs|entry count"):
        load_manifest(_canonical(decoded))
