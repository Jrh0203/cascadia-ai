from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from r2_d0 import legacy_cleanup as subject


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _manifest(root: Path) -> bytes:
    (root / "nested").mkdir(parents=True)
    (root / "nested/data.bin").write_bytes(b"payload")
    entries = []
    for relative in (".", "nested", "nested/data.bin"):
        path = root if relative == "." else root / relative
        details = os.lstat(path)
        is_dir = stat.S_ISDIR(details.st_mode)
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
                "sha256": (
                    hashlib.sha256(("tree:" + relative).encode()).hexdigest()
                    if is_dir
                    else hashlib.sha256(path.read_bytes()).hexdigest()
                ),
                "size": details.st_size,
                "type": "directory" if is_dir else "file",
            }
        )
    manifest = {
        "schema_id": "cascadia.r2-map.john3-legacy-native-workspace-freeze.v1",
        "schema_version": 1,
        "host": "john3",
        "root": "/Users/john3/cascadia-bench/r2-map-v1",
        "entries": entries,
        "hardlink_groups": [],
        "totals": {
            "entry_count": 3,
            "unique_regular_bytes": 7,
            "root_tree_sha256": entries[0]["sha256"],
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    return _canonical(manifest)


def _packet(root: Path, manifest: bytes) -> dict:
    subject.SOURCE_ROOT = root
    value = json.loads(manifest)
    return subject.build_cleanup_packet(
        manifest_file_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_sha256=value["manifest_sha256"],
        root_tree_sha256=value["totals"]["root_tree_sha256"],
        archive_plan_sha256="a" * 64,
        john2_commit_receipt_sha256="b" * 64,
        john1_reopen_receipt_sha256="c" * 64,
        archive_sha256="d" * 64,
        archive_size=100,
        goal_sha256="e" * 64,
        cleanup_helper_sha256="f" * 64,
        archive_helper_sha256=hashlib.sha256(
            Path(subject.__file__).with_name("legacy_archive_stream.py").read_bytes()
        ).hexdigest(),
    )


def test_cleanup_is_exact_crash_resumable_and_replay_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    manifest = _manifest(root)
    monkeypatch.setattr(subject, "SOURCE_ROOT", root)
    packet = _packet(root, manifest)
    receipt_root = tmp_path / "receipts"
    flags: list[Path] = []

    def fault(point: str) -> None:
        if point == "after-rename":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        subject.execute_cleanup(
            packet,
            manifest,
            source_root=root,
            receipt_root=receipt_root,
            current_host="john3",
            flag_setter=lambda path, _value: flags.append(path),
            fault_injector=fault,
        )
    assert not root.exists()
    receipt = subject.execute_cleanup(
        packet,
        manifest,
        source_root=root,
        receipt_root=receipt_root,
        current_host="john3",
        flag_setter=lambda path, _value: flags.append(path),
    )
    replay = subject.execute_cleanup(
        packet,
        manifest,
        source_root=root,
        receipt_root=receipt_root,
        current_host="john3",
        flag_setter=lambda path, _value: flags.append(path),
    )
    assert receipt == replay
    assert receipt["source_root_absent"] is True
    assert receipt["deleted_entry_count"] == 3
    assert not list(tmp_path.glob(".r2-map-v1.cleanup-*"))


def test_cleanup_rejects_wrong_host_and_tampered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    manifest = _manifest(root)
    monkeypatch.setattr(subject, "SOURCE_ROOT", root)
    packet = _packet(root, manifest)
    with pytest.raises(subject.LegacyCleanupError, match="only as John3"):
        subject.execute_cleanup(
            packet,
            manifest,
            source_root=root,
            receipt_root=tmp_path / "receipts",
            current_host="john2",
            flag_setter=lambda _path, _value: None,
        )
    with pytest.raises(subject.LegacyCleanupError, match="manifest file hash"):
        subject.execute_cleanup(
            packet,
            manifest + b"x",
            source_root=root,
            receipt_root=tmp_path / "receipts",
            current_host="john3",
            flag_setter=lambda _path, _value: None,
        )


def test_exact_isolated_launcher_loads_hash_bound_sibling_without_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    manifest = _manifest(root)
    production_root = Path("/Users/john3/cascadia-bench/r2-map-v1")
    monkeypatch.setattr(subject, "SOURCE_ROOT", production_root)
    packet = _packet(production_root, manifest)
    staged = tmp_path / "staged"
    staged.mkdir()
    cleanup = staged / "legacy_cleanup.py"
    archive = staged / "legacy_archive_stream.py"
    cleanup.write_bytes(Path(subject.__file__).read_bytes())
    archive.write_bytes(Path(subject.__file__).with_name("legacy_archive_stream.py").read_bytes())
    packet_path = staged / "packet.json"
    manifest_path = staged / "manifest.json"
    packet_path.write_bytes(subject.encode_cleanup_packet(packet))
    manifest_path.write_bytes(manifest)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(cleanup),
            "--packet",
            str(packet_path),
            "--manifest",
            str(manifest_path),
            "--execute",
            "--confirm-packet-sha256",
            packet["packet_sha256"],
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "only as John3" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
