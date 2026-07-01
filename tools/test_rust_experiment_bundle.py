from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("rust_experiment_bundle.py")
_SPEC = importlib.util.spec_from_file_location("rust_experiment_bundle", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bundle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bundle
_SPEC.loader.exec_module(bundle)


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    (repository / "crates" / "demo" / "src").mkdir(parents=True)
    (repository / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
    (repository / "Cargo.lock").write_text("# lock\n")
    (repository / "crates" / "demo" / "Cargo.toml").write_text(
        '[package]\nname = "demo"\nversion = "0.1.0"\n'
    )
    (repository / "crates" / "demo" / "src" / "lib.rs").write_text("pub fn value() -> u8 { 7 }\n")
    binary = repository / "target" / "release" / "demo"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary-v1")
    binary.chmod(0o755)
    return repository, binary


def test_build_is_content_addressed_validated_and_idempotent(tmp_path: Path) -> None:
    repository, binary = _repository(tmp_path)
    output_root = tmp_path / "bundles"
    first_path, first_manifest, first_reused = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("Cargo.lock"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )
    second_path, second_manifest, second_reused = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("Cargo.lock"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )
    assert first_reused is False
    assert second_reused is True
    assert first_path == second_path
    assert first_manifest["bundle_id"] == second_manifest["bundle_id"]
    assert bundle.validate_bundle(first_path) == first_manifest
    assert (first_path / "bin" / "demo").stat().st_mode & 0o111
    assert first_path.stat().st_mode & 0o222 == 0
    assert (first_path / "source" / "Cargo.toml").stat().st_mode & 0o222 == 0
    assert (first_path / "bin" / "demo").stat().st_mode & 0o222 == 0


def test_source_only_bundle_is_supported(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    path, manifest, reused = bundle.build_bundle(
        repository=repository,
        experiment_id="source-only-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[],
        output_root=tmp_path / "bundles",
    )
    assert reused is False
    assert manifest["identity"]["binaries"] == []
    assert bundle.validate_bundle(path) == manifest
    assert not any((path / "bin").iterdir())


def test_source_or_binary_drift_changes_bundle_identity(tmp_path: Path) -> None:
    repository, binary = _repository(tmp_path)
    output_root = tmp_path / "bundles"
    original, _, _ = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )
    (repository / "crates" / "demo" / "src" / "lib.rs").write_text("pub fn value() -> u8 { 8 }\n")
    changed_source, _, _ = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )
    binary.write_bytes(b"binary-v2")
    changed_binary, _, _ = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )
    assert len({original.name, changed_source.name, changed_binary.name}) == 3


def test_existing_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    repository, binary = _repository(tmp_path)
    path, _, _ = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=tmp_path / "bundles",
    )
    source = path / "source" / "Cargo.toml"
    source.chmod(0o644)
    source.write_text("tampered\n")
    with pytest.raises(bundle.BundleError, match="mismatch"):
        bundle.validate_bundle(path)


def test_reuse_repairs_writable_permissions(tmp_path: Path) -> None:
    repository, binary = _repository(tmp_path)
    output_root = tmp_path / "bundles"
    path, _, _ = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )
    source = path / "source" / "Cargo.toml"
    path.chmod(0o755)
    (path / "source").chmod(0o755)
    source.chmod(0o644)

    reused_path, _, reused = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=output_root,
    )

    assert reused is True
    assert reused_path == path
    assert path.stat().st_mode & 0o222 == 0
    assert (path / "source").stat().st_mode & 0o222 == 0
    assert source.stat().st_mode & 0o222 == 0


def test_symlink_and_repository_escape_are_rejected(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    with pytest.raises(bundle.BundleError, match="escapes the repository"):
        bundle.collect_source_files(repository, [outside])

    link = repository / "crates" / "demo" / "linked.rs"
    os.symlink(outside, link)
    with pytest.raises(bundle.BundleError, match="symlinks"):
        bundle.collect_source_files(repository, [Path("crates")])


def test_duplicate_binary_names_are_rejected(tmp_path: Path) -> None:
    repository, binary = _repository(tmp_path)
    second = tmp_path / "other" / binary.name
    second.parent.mkdir()
    second.write_bytes(b"second")
    with pytest.raises(bundle.BundleError, match="duplicate binary"):
        bundle.collect_binaries(repository, [binary, second])


def test_manifest_identity_rejects_forged_bundle_id(tmp_path: Path) -> None:
    repository, binary = _repository(tmp_path)
    path, _, _ = bundle.build_bundle(
        repository=repository,
        experiment_id="demo-v1",
        includes=[Path("Cargo.toml"), Path("crates")],
        binaries=[binary],
        output_root=tmp_path / "bundles",
    )
    manifest_path = path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_id"] = "0" * 64
    path.chmod(0o755)
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(bundle.BundleError, match="scientific identity"):
        bundle.validate_bundle(path)
