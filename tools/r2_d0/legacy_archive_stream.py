"""Deterministic no-follow archive stream for a frozen legacy workspace."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 500_000
ARCHIVE_PREFIX = "r2-map-v1"
MANIFEST_MEMBER = ".cascadia/freeze-manifest.json"


class LegacyArchiveError(RuntimeError):
    """The frozen source or archive stream differed from its manifest."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def load_manifest(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise LegacyArchiveError("legacy freeze manifest size differs")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacyArchiveError("legacy freeze manifest is invalid JSON") from error
    if not isinstance(value, dict):
        raise LegacyArchiveError("legacy freeze manifest is not an object")
    embedded = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if (
        value.get("schema_id")
        != "cascadia.r2-map.john3-legacy-native-workspace-freeze.v1"
        or value.get("schema_version") != 1
        or value.get("host") != "john3"
        or value.get("root") != "/Users/john3/cascadia-bench/r2-map-v1"
        or not isinstance(embedded, str)
        or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != embedded
        or not isinstance(value.get("entries"), list)
        or not 1 <= len(value["entries"]) <= MAX_ENTRIES
        or not isinstance(value.get("hardlink_groups"), list)
    ):
        raise LegacyArchiveError("legacy freeze manifest identity differs")
    paths = [entry.get("path") for entry in value["entries"] if isinstance(entry, Mapping)]
    if (
        len(paths) != len(value["entries"])
        or paths != sorted(paths)
        or len(set(paths)) != len(paths)
    ):
        raise LegacyArchiveError("legacy freeze manifest paths are not sorted and unique")
    if paths[0] != "." or value.get("totals", {}).get("entry_count") != len(paths):
        raise LegacyArchiveError("legacy freeze manifest root or entry count differs")
    return value


def _actual_paths(root: Path) -> list[str]:
    pending = [(root, ".")]
    paths: list[str] = []
    while pending:
        directory, relative = pending.pop()
        paths.append(relative)
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
        except OSError as error:
            raise LegacyArchiveError(
                f"legacy source directory cannot be read: {relative}"
            ) from error
        for child in children:
            child_relative = child.name if relative == "." else f"{relative}/{child.name}"
            details = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(details.st_mode):
                raise LegacyArchiveError(f"legacy source contains symlink: {child_relative}")
            if stat.S_ISDIR(details.st_mode):
                pending.append((Path(child.path), child_relative))
            elif stat.S_ISREG(details.st_mode):
                paths.append(child_relative)
            else:
                raise LegacyArchiveError(f"legacy source contains special file: {child_relative}")
            if len(paths) + len(pending) > MAX_ENTRIES:
                raise LegacyArchiveError("legacy source exceeds its entry limit")
    return sorted(paths)


def verify_frozen_source(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    root_details = os.lstat(root)
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        raise LegacyArchiveError("legacy source root is not a real directory")
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    if _actual_paths(root) != list(entries):
        raise LegacyArchiveError("legacy source path set differs from the frozen manifest")
    hash_cache: dict[tuple[int, int], str] = {}
    unique_bytes = 0
    for relative, expected in entries.items():
        path = root if relative == "." else root / relative
        details = os.lstat(path)
        expected_type = "directory" if stat.S_ISDIR(details.st_mode) else "file"
        if (
            expected.get("type") != expected_type
            or details.st_dev != expected.get("device")
            or details.st_ino != expected.get("inode")
            or details.st_nlink != expected.get("nlink")
            or stat.S_IMODE(details.st_mode) != int(expected.get("mode"), 8)
            or details.st_size != expected.get("size")
            or details.st_mtime_ns != expected.get("mtime_ns")
            or getattr(details, "st_flags", expected.get("flags_after"))
            != expected.get("flags_after")
        ):
            raise LegacyArchiveError(f"legacy source metadata differs: {relative}")
        if expected_type == "file":
            key = (details.st_dev, details.st_ino)
            digest = hash_cache.get(key)
            if digest is None:
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                try:
                    before = os.fstat(descriptor)
                    hasher = hashlib.sha256()
                    observed_size = 0
                    while True:
                        chunk = os.read(descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        observed_size += len(chunk)
                    after = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
                if (
                    observed_size != expected["size"]
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                ):
                    raise LegacyArchiveError(f"legacy source changed while hashing: {relative}")
                digest = hasher.hexdigest()
                hash_cache[key] = digest
                unique_bytes += observed_size
            if digest != expected.get("sha256"):
                raise LegacyArchiveError(f"legacy source content differs: {relative}")
    totals = manifest["totals"]
    if unique_bytes != totals.get("unique_regular_bytes"):
        raise LegacyArchiveError("legacy source unique-byte total differs")
    for group in manifest["hardlink_groups"]:
        members = group.get("members") if isinstance(group, Mapping) else None
        if not isinstance(members, list) or len(members) < 2 or members != sorted(members):
            raise LegacyArchiveError("legacy hardlink group differs")
        identities = {(entries[item]["device"], entries[item]["inode"]) for item in members}
        if len(identities) != 1 or group.get("nlink") != len(members):
            raise LegacyArchiveError("legacy hardlink topology differs")
    return {
        "entry_count": len(entries),
        "unique_regular_bytes": unique_bytes,
        "root_tree_sha256": totals["root_tree_sha256"],
        "status": "pass",
    }


def _hardlink_targets(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in manifest["hardlink_groups"]:
        first, *rest = group["members"]
        for relative in rest:
            result[relative] = first
    return result


def _tar_name(relative: str) -> str:
    return ARCHIVE_PREFIX if relative == "." else f"{ARCHIVE_PREFIX}/{relative}"


def emit_archive(root: Path, manifest_payload: bytes, output: BinaryIO) -> dict[str, Any]:
    manifest = load_manifest(manifest_payload)
    proof = verify_frozen_source(root, manifest)
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    hardlinks = _hardlink_targets(manifest)
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        manifest_info = tarfile.TarInfo(MANIFEST_MEMBER)
        manifest_info.size = len(manifest_payload)
        manifest_info.mode = 0o400
        manifest_info.uid = manifest_info.gid = 0
        manifest_info.mtime = 0
        archive.addfile(manifest_info, io.BytesIO(manifest_payload))
        for relative, expected in entries.items():
            info = tarfile.TarInfo(_tar_name(relative))
            info.mode = int(expected["mode"], 8)
            info.uid = info.gid = 0
            info.mtime = expected["mtime_ns"] // 1_000_000_000
            info.pax_headers = {
                "cascadia.mtime_ns": str(expected["mtime_ns"]),
                "cascadia.sha256": expected["sha256"],
            }
            if expected["type"] == "directory":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
                continue
            if relative in hardlinks:
                info.type = tarfile.LNKTYPE
                info.linkname = _tar_name(hardlinks[relative])
                archive.addfile(info)
                continue
            info.size = expected["size"]
            path = root / relative
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                archive.addfile(info, stream)
    return proof


class _HashingReader:
    def __init__(self, source: BinaryIO):
        self.source = source
        self.hasher = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        value = self.source.read(size)
        self.hasher.update(value)
        self.size += len(value)
        return value


def verify_archive(
    source: BinaryIO,
    manifest_payload: bytes,
    *,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_payload)
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    hardlinks = _hardlink_targets(manifest)
    reader = _HashingReader(source)
    observed: list[str] = []
    with tarfile.open(fileobj=reader, mode="r|*") as archive:
        for member in archive:
            if member.name == MANIFEST_MEMBER:
                if observed:
                    raise LegacyArchiveError("archive manifest is not the first member")
                stream = archive.extractfile(member)
                if stream is None or stream.read() != manifest_payload:
                    raise LegacyArchiveError("archive manifest bytes differ")
                observed.append(member.name)
                continue
            if not member.name.startswith(f"{ARCHIVE_PREFIX}/") and member.name != ARCHIVE_PREFIX:
                raise LegacyArchiveError("archive member escapes its fixed prefix")
            relative = (
                "."
                if member.name == ARCHIVE_PREFIX
                else member.name[len(ARCHIVE_PREFIX) + 1 :]
            )
            expected = entries.get(relative)
            if expected is None or relative in observed:
                raise LegacyArchiveError("archive contains an unexpected or duplicate member")
            if (
                member.uid != 0
                or member.gid != 0
                or member.mode != int(expected["mode"], 8)
                or member.pax_headers.get("cascadia.mtime_ns") != str(expected["mtime_ns"])
                or member.pax_headers.get("cascadia.sha256") != expected["sha256"]
            ):
                raise LegacyArchiveError(f"archive member metadata differs: {relative}")
            if expected["type"] == "directory":
                if not member.isdir():
                    raise LegacyArchiveError(f"archive directory type differs: {relative}")
            elif relative in hardlinks:
                if not member.islnk() or member.linkname != _tar_name(hardlinks[relative]):
                    raise LegacyArchiveError(f"archive hardlink differs: {relative}")
            else:
                stream = archive.extractfile(member)
                if stream is None:
                    raise LegacyArchiveError(f"archive file is unreadable: {relative}")
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                    raise LegacyArchiveError(f"archive file content differs: {relative}")
            observed.append(relative)
    while reader.read(1024 * 1024):
        pass
    if observed != [MANIFEST_MEMBER, *entries]:
        raise LegacyArchiveError("archive member order or completeness differs")
    archive_sha256 = reader.hasher.hexdigest()
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise LegacyArchiveError("archive byte hash differs")
    return {
        "archive_size": reader.size,
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "entry_count": len(entries),
        "root_tree_sha256": manifest["totals"]["root_tree_sha256"],
        "hardlink_group_count": len(manifest["hardlink_groups"]),
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    emit.add_argument("--root", type=Path, required=True)
    emit.add_argument("--manifest", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-archive-sha256")
    arguments = parser.parse_args()
    try:
        payload = arguments.manifest.read_bytes()
        if arguments.command == "emit":
            proof = emit_archive(arguments.root, payload, sys.stdout.buffer)
            print(json.dumps(proof, sort_keys=True), file=sys.stderr)
        else:
            proof = verify_archive(
                sys.stdin.buffer,
                payload,
                expected_archive_sha256=arguments.expected_archive_sha256,
            )
            print(json.dumps(proof, sort_keys=True))
    except (OSError, LegacyArchiveError) as error:
        print(f"legacy archive stream failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
