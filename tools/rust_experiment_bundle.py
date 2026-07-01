#!/usr/bin/env python3
"""Build and validate a content-addressed immutable experiment bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

SCHEMA_VERSION = 1
IGNORED_DIRECTORY_NAMES = {".git", "__pycache__", "target"}


class BundleError(RuntimeError):
    """Raised when an immutable bundle cannot be built or validated."""


@dataclass(frozen=True)
class SourceFile:
    source: Path
    relative: Path
    size: int
    blake3: str


@dataclass(frozen=True)
class BinaryFile:
    source: Path
    name: str
    size: int
    blake3: str


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _beneath(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BundleError(f"{label} escapes the repository: {path}") from error
    return resolved


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise BundleError(f"bundle inputs may not contain symlinks: {path}")


def _walk_regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        retained_directories = []
        for name in sorted(directory_names):
            child = current / name
            _reject_symlink(child)
            if name not in IGNORED_DIRECTORY_NAMES:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            child = current / name
            _reject_symlink(child)
            if not child.is_file():
                raise BundleError(f"bundle input is not a regular file: {child}")
            files.append(child)
    return files


def collect_source_files(
    repository: Path,
    includes: list[Path],
) -> list[SourceFile]:
    repository = repository.resolve()
    if not repository.is_dir():
        raise BundleError(f"repository is not a directory: {repository}")
    if not includes:
        raise BundleError("at least one source include is required")

    paths: dict[Path, Path] = {}
    for include in includes:
        requested = include if include.is_absolute() else repository / include
        resolved = _beneath(requested, repository, "source include")
        if not resolved.exists():
            raise BundleError(f"source include does not exist: {include}")
        _reject_symlink(resolved)
        candidates = [resolved] if resolved.is_file() else _walk_regular_files(resolved)
        for candidate in candidates:
            relative = candidate.relative_to(repository)
            paths.setdefault(relative, candidate)

    if not paths:
        raise BundleError("source includes contain no regular files")
    return [
        SourceFile(
            source=source,
            relative=relative,
            size=source.stat().st_size,
            blake3=file_blake3(source),
        )
        for relative, source in sorted(paths.items())
    ]


def collect_binaries(repository: Path, binaries: list[Path]) -> list[BinaryFile]:
    repository = repository.resolve()
    by_name: dict[str, Path] = {}
    for binary in binaries:
        requested = binary if binary.is_absolute() else repository / binary
        resolved = requested.resolve()
        if not resolved.is_file():
            raise BundleError(f"binary does not exist or is not a file: {binary}")
        _reject_symlink(resolved)
        if resolved.name in by_name:
            raise BundleError(f"duplicate binary destination name: {resolved.name}")
        by_name[resolved.name] = resolved
    return [
        BinaryFile(
            source=source,
            name=name,
            size=source.stat().st_size,
            blake3=file_blake3(source),
        )
        for name, source in sorted(by_name.items())
    ]


def _source_payload(files: list[SourceFile]) -> list[dict[str, Any]]:
    return [
        {
            "path": file.relative.as_posix(),
            "bytes": file.size,
            "blake3": file.blake3,
        }
        for file in files
    ]


def _binary_payload(files: list[BinaryFile]) -> list[dict[str, Any]]:
    return [
        {
            "name": file.name,
            "bytes": file.size,
            "blake3": file.blake3,
        }
        for file in files
    ]


def _git_provenance(repository: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "-z"],
        check=False,
        capture_output=True,
    )
    if revision.returncode != 0 or status.returncode != 0:
        return {
            "git_revision": None,
            "git_dirty": None,
            "git_status_blake3": None,
        }
    status_bytes = status.stdout
    return {
        "git_revision": revision.stdout.strip(),
        "git_dirty": bool(status_bytes),
        "git_status_blake3": blake3.blake3(status_bytes).hexdigest(),
    }


def bundle_identity(
    experiment_id: str,
    source_files: list[SourceFile],
    binaries: list[BinaryFile],
) -> tuple[str, dict[str, Any]]:
    if not experiment_id.strip():
        raise BundleError("experiment ID must not be empty")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "source_files": _source_payload(source_files),
        "binaries": _binary_payload(binaries),
    }
    return blake3.blake3(canonical_json(identity)).hexdigest(), identity


def validate_bundle(path: Path, expected_bundle_id: str | None = None) -> dict[str, Any]:
    manifest_path = path / "bundle.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read bundle manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise BundleError("bundle manifest root must be an object")
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or len(bundle_id) != 64:
        raise BundleError("bundle manifest has an invalid bundle ID")
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise BundleError(f"bundle ID mismatch: expected {expected_bundle_id}, found {bundle_id}")

    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise BundleError("bundle manifest lacks its scientific identity")
    if blake3.blake3(canonical_json(identity)).hexdigest() != bundle_id:
        raise BundleError("bundle scientific identity does not match its bundle ID")

    expected_paths = {"bundle.json"}
    for entry in identity.get("source_files", []):
        if not isinstance(entry, dict):
            raise BundleError("invalid source entry in bundle manifest")
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise BundleError(f"invalid source path in bundle manifest: {relative}")
        candidate = path / "source" / relative
        _reject_symlink(candidate)
        if not candidate.is_file():
            raise BundleError(f"bundle source file is missing: {candidate}")
        if candidate.stat().st_size != entry.get("bytes"):
            raise BundleError(f"bundle source size mismatch: {candidate}")
        if file_blake3(candidate) != entry.get("blake3"):
            raise BundleError(f"bundle source checksum mismatch: {candidate}")
        expected_paths.add(candidate.relative_to(path).as_posix())

    for entry in identity.get("binaries", []):
        if not isinstance(entry, dict):
            raise BundleError("invalid binary entry in bundle manifest")
        name = str(entry.get("name", ""))
        if not name or Path(name).name != name:
            raise BundleError(f"invalid binary name in bundle manifest: {name}")
        candidate = path / "bin" / name
        _reject_symlink(candidate)
        if not candidate.is_file():
            raise BundleError(f"bundle binary is missing: {candidate}")
        if candidate.stat().st_size != entry.get("bytes"):
            raise BundleError(f"bundle binary size mismatch: {candidate}")
        if file_blake3(candidate) != entry.get("blake3"):
            raise BundleError(f"bundle binary checksum mismatch: {candidate}")
        expected_paths.add(candidate.relative_to(path).as_posix())

    actual_paths = set()
    for candidate in path.rglob("*"):
        _reject_symlink(candidate)
        if candidate.is_file():
            actual_paths.add(candidate.relative_to(path).as_posix())
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise BundleError(f"bundle file set drifted: missing={missing}, extra={extra}")
    return manifest


def seal_bundle(path: Path) -> None:
    """Remove write permissions from a validated bundle tree."""
    if not path.is_dir():
        raise BundleError(f"bundle path is not a directory: {path}")
    directories = [path]
    for candidate in path.rglob("*"):
        _reject_symlink(candidate)
        if candidate.is_dir():
            directories.append(candidate)
        elif candidate.is_file():
            executable = bool(candidate.stat().st_mode & 0o111)
            candidate.chmod(0o555 if executable else 0o444)
        else:
            raise BundleError(f"bundle contains a non-regular entry: {candidate}")
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        directory.chmod(0o555)


def build_bundle(
    *,
    repository: Path,
    experiment_id: str,
    includes: list[Path],
    binaries: list[Path],
    output_root: Path,
) -> tuple[Path, dict[str, Any], bool]:
    repository = repository.resolve()
    source_files = collect_source_files(repository, includes)
    binary_files = collect_binaries(repository, binaries)
    bundle_id, identity = bundle_identity(
        experiment_id,
        source_files,
        binary_files,
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / bundle_id
    if destination.exists():
        validate_bundle(destination, bundle_id)
        seal_bundle(destination)
        return destination, validate_bundle(destination, bundle_id), True

    temporary = output_root / f".tmp-{bundle_id}-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        (temporary / "source").mkdir(parents=True)
        (temporary / "bin").mkdir()
        for source in source_files:
            target = temporary / "source" / source.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.source, target, follow_symlinks=False)
        for binary in binary_files:
            target = temporary / "bin" / binary.name
            shutil.copy2(binary.source, target, follow_symlinks=False)
            target.chmod(target.stat().st_mode | 0o111)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "identity": identity,
            "source_tree_blake3": blake3.blake3(
                canonical_json(identity["source_files"])
            ).hexdigest(),
            "binary_tree_blake3": blake3.blake3(canonical_json(identity["binaries"])).hexdigest(),
            "provenance": _git_provenance(repository),
            "created_unix_seconds": int(time.time()),
        }
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        validate_bundle(temporary, bundle_id)
        try:
            temporary.rename(destination)
        except FileExistsError:
            shutil.rmtree(temporary)
            validate_bundle(destination, bundle_id)
            seal_bundle(destination)
            return destination, validate_bundle(destination, bundle_id), True
        seal_bundle(destination)
        return destination, validate_bundle(destination, bundle_id), False
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--include", type=Path, action="append", required=True)
    parser.add_argument("--binary", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path, manifest, reused = build_bundle(
            repository=args.repository,
            experiment_id=args.experiment_id,
            includes=args.include,
            binaries=args.binary,
            output_root=args.output_root,
        )
    except BundleError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "bundle_id": manifest["bundle_id"],
                "bundle_path": str(path),
                "reused": reused,
                "source_files": len(manifest["identity"]["source_files"]),
                "binaries": len(manifest["identity"]["binaries"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
