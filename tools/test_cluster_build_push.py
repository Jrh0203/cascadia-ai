from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import cluster_build_push as subject


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_workspace_source_identity_covers_tracked_and_untracked_source(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".gitignore").write_text("ignored.bin\n")
    (tmp_path / "tracked.txt").write_text("tracked\n")
    _git(tmp_path, "add", ".gitignore", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "fixture")
    (tmp_path / "untracked.txt").write_text("first\n")
    (tmp_path / "ignored.bin").write_bytes(b"ignored")

    first = subject._workspace_source_identity(tmp_path)
    second = subject._workspace_source_identity(tmp_path)
    assert first == second
    assert first["schema_id"] == "cascadia.cluster.build-workspace-identity.v1"
    assert first["git_dirty"] is True
    assert first["files"] == 3

    (tmp_path / "untracked.txt").write_text("second\n")
    changed = subject._workspace_source_identity(tmp_path)
    assert changed["workspace_blake3"] != first["workspace_blake3"]

    (tmp_path / "ignored.bin").write_bytes(b"changed but excluded")
    ignored = subject._workspace_source_identity(tmp_path)
    assert ignored["workspace_blake3"] == changed["workspace_blake3"]


def test_workspace_source_identity_excludes_mutable_operational_state(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.txt").write_text("tracked\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "fixture")
    before = subject._workspace_source_identity(tmp_path)
    (tmp_path / "STATE.md").write_text("phase one\n")
    first = subject._workspace_source_identity(tmp_path)
    (tmp_path / "STATE.md").write_text("phase two\n")
    second = subject._workspace_source_identity(tmp_path)
    assert first == before
    assert second == before


def test_manifest_push_digest_requires_exactly_one_digest_line() -> None:
    digest = "sha256:" + "a" * 64
    assert subject._manifest_push_digest(f"pushing\n{digest}\n") == digest
    with pytest.raises(SystemExit, match="uniquely"):
        subject._manifest_push_digest("no digest")
