#!/usr/bin/env python3
"""Build and verify the manifest-exact deterministic R2-MAP source archive."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

MANIFEST_SCHEMA = "cascadia.r2-map.w0-w5-source-manifest.v1"
ARCHIVE_SCHEMA = "cascadia.r2-map.source-archive-verification.v1"
CLEANUP_SCHEMA = "cascadia.r2-map.pytest-basetemp-cleanup.v1"
PYTEST_BASETEMP_NAME = "r2-map-python-boundary-pytest"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
BLOCK_SIZE = 512
RECORD_SIZE = 20 * BLOCK_SIZE
ZERO_BLOCK = bytes(BLOCK_SIZE)
ALLOWED_MODES = frozenset({"0400", "0500"})
ALLOWED_EXTRACTED_XATTRS = frozenset({b"com.apple.provenance"})
_DARWIN_FLISTXATTR: Any = None
_DARWIN_LIBC: Any = None


class SourceArchiveError(ValueError):
    """The source manifest, archive, or extracted tree is not exact."""


def _darwin_flistxattr() -> Any:
    global _DARWIN_FLISTXATTR, _DARWIN_LIBC
    if _DARWIN_FLISTXATTR is None:
        _DARWIN_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        function = _DARWIN_LIBC.flistxattr
        function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        function.restype = ctypes.c_ssize_t
        _DARWIN_FLISTXATTR = function
    return _DARWIN_FLISTXATTR


def _xattr_names_fd(descriptor: int, label: str) -> tuple[bytes, ...]:
    if sys.platform == "darwin":
        function = _darwin_flistxattr()
        for _attempt in range(3):
            ctypes.set_errno(0)
            size = function(descriptor, None, 0, 0)
            if size < 0:
                error_number = ctypes.get_errno()
                raise SourceArchiveError(
                    f"cannot enumerate extended attributes for {label}: "
                    f"[errno {error_number}] {os.strerror(error_number)}"
                )
            if size == 0:
                return ()
            buffer = ctypes.create_string_buffer(size)
            ctypes.set_errno(0)
            actual = function(descriptor, buffer, size, 0)
            if actual >= 0:
                payload = bytes(buffer.raw[:actual])
                if payload and not payload.endswith(b"\0"):
                    raise SourceArchiveError("extended-attribute name vector is malformed")
                return tuple(sorted(name for name in payload.split(b"\0") if name))
            error_number = ctypes.get_errno()
            # ERANGE means the xattr set changed between the size and data calls.
            if error_number != errno.ERANGE:
                raise SourceArchiveError(
                    f"cannot enumerate extended attributes for {label}: "
                    f"[errno {error_number}] {os.strerror(error_number)}"
                )
        raise SourceArchiveError("extended attributes changed repeatedly during verification")
    listxattr = getattr(os, "listxattr", None)
    if listxattr is None:
        raise SourceArchiveError("this platform has no audited no-follow xattr enumerator")
    try:
        names = listxattr(descriptor)
    except OSError as error:
        raise SourceArchiveError(f"cannot enumerate extended attributes for {label}") from error
    return tuple(sorted(os.fsencode(name) for name in names))


def _xattr_names_no_follow(path: Path) -> tuple[bytes, ...]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceArchiveError(f"cannot open path for xattr enumeration: {path}") from error
    try:
        return _xattr_names_fd(descriptor, str(path))
    finally:
        os.close(descriptor)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SourceArchiveError("source path is not a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} or part.startswith("._") for part in path.parts)
    ):
        raise SourceArchiveError(f"unsafe or AppleDouble source path: {value!r}")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise SourceArchiveError("source paths must fit the frozen ASCII USTAR contract") from error
    if len(value.encode("ascii")) > 100:
        # The current closure is below 80 bytes. Keeping the name field below
        # 100 avoids PAX/GNU control records and prefix-splitting ambiguity.
        raise SourceArchiveError(f"source path exceeds the strict USTAR name field: {value}")
    return value


def _read_bounded(stream: BinaryIO, maximum: int, label: str) -> bytes:
    value = stream.read(maximum + 1)
    if len(value) > maximum or stream.read(1):
        raise SourceArchiveError(f"{label} exceeds its byte bound")
    return value


def _input_bytes(path: str, maximum: int, label: str) -> bytes:
    if path == "-":
        return _read_bounded(sys.stdin.buffer, maximum, label)
    with Path(path).open("rb") as stream:
        return _read_bounded(stream, maximum, label)


def load_manifest_bytes(value: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceArchiveError("source manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_id") != MANIFEST_SCHEMA:
        raise SourceArchiveError("source manifest schema is invalid")
    expected_document = dict(manifest)
    observed_document_sha256 = expected_document.pop("document_sha256", None)
    if (
        not isinstance(observed_document_sha256, str)
        or _sha256(canonical_json(expected_document)) != observed_document_sha256
    ):
        raise SourceArchiveError("source manifest document SHA-256 is invalid")
    canonical = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    if value != canonical:
        raise SourceArchiveError("source manifest is not in its canonical rendered form")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SourceArchiveError("source manifest files are absent")
    normalized: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"relative", "size", "sha256", "mode"}:
            raise SourceArchiveError("source manifest file entry is invalid")
        relative = _safe_relative(entry["relative"])
        size = entry["size"]
        digest = entry["sha256"]
        mode = entry["mode"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or mode not in ALLOWED_MODES
        ):
            raise SourceArchiveError(f"source manifest metadata is invalid: {relative}")
        normalized.append({"relative": relative, "size": size, "sha256": digest, "mode": mode})
    relatives = [entry["relative"] for entry in normalized]
    if relatives != sorted(relatives) or len(relatives) != len(set(relatives)):
        raise SourceArchiveError("source manifest paths are not unique and sorted")
    file_count = manifest.get("file_count")
    total_bytes = manifest.get("total_bytes")
    if (
        not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count != len(normalized)
    ):
        raise SourceArchiveError("source manifest file count is invalid")
    if (
        not isinstance(total_bytes, int)
        or isinstance(total_bytes, bool)
        or total_bytes != sum(entry["size"] for entry in normalized)
    ):
        raise SourceArchiveError("source manifest byte count is invalid")
    relative_set = set(relatives)
    for relative in relatives:
        parent = PurePosixPath(relative).parent
        while str(parent) not in {"", "."}:
            if str(parent) in relative_set:
                raise SourceArchiveError(
                    f"source manifest has a file/directory path collision: {relative}"
                )
            parent = parent.parent
    raw_member_bytes = sum(
        BLOCK_SIZE + ((entry["size"] + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        for entry in normalized
    )
    canonical_archive_bytes = (
        (raw_member_bytes + 2 * BLOCK_SIZE + RECORD_SIZE - 1) // RECORD_SIZE
    ) * RECORD_SIZE
    if total_bytes > MAX_ARCHIVE_BYTES or canonical_archive_bytes > MAX_ARCHIVE_BYTES:
        raise SourceArchiveError("source manifest exceeds the archive byte bound")
    if manifest.get("protected_seed_values_opened") is not False:
        raise SourceArchiveError("source manifest opens a protected seed domain")
    return manifest, normalized


def load_manifest(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return load_manifest_bytes(_input_bytes(path, MAX_MANIFEST_BYTES, "source manifest"))


def _open_root(repository: Path) -> int:
    try:
        before = repository.lstat()
    except OSError as error:
        raise SourceArchiveError("source root cannot be inspected") from error
    if repository.is_symlink() or not stat.S_ISDIR(before.st_mode):
        raise SourceArchiveError("source root is not a direct regular directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(repository, flags)
    except OSError as error:
        raise SourceArchiveError("source root cannot be opened without following links") from error
    after = os.fstat(descriptor)
    if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        os.close(descriptor)
        raise SourceArchiveError("source root changed while it was opened")
    return descriptor


def _read_file_descriptor(descriptor: int, entry: dict[str, Any], label: str) -> bytes:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise SourceArchiveError(f"source is not a single-link regular file: {label}")
    if details.st_size != entry["size"]:
        raise SourceArchiveError(f"source size differs from manifest: {label}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = entry["size"] + 1
    while remaining:
        chunk = os.read(descriptor, min(1 << 20, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) != entry["size"] or _sha256(value) != entry["sha256"]:
        raise SourceArchiveError(f"source bytes differ from manifest: {label}")
    return value


def _source_bytes_at(root_descriptor: int, entry: dict[str, Any]) -> bytes:
    relative = entry["relative"]
    current = os.dup(root_descriptor)
    try:
        parts = PurePosixPath(relative).parts
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        for part in parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=current)
            except OSError as error:
                raise SourceArchiveError(
                    f"source parent cannot be opened without following links: {relative}"
                ) from error
            details = os.fstat(child)
            if not stat.S_ISDIR(details.st_mode):
                os.close(child)
                raise SourceArchiveError(f"source parent is not a directory: {relative}")
            os.close(current)
            current = child
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
        except OSError as error:
            raise SourceArchiveError(
                f"source cannot be opened without following links: {relative}"
            ) from error
        try:
            return _read_file_descriptor(descriptor, entry, relative)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _tar_info(entry: dict[str, Any]) -> tarfile.TarInfo:
    info = tarfile.TarInfo(entry["relative"])
    info.size = entry["size"]
    info.mode = int(entry["mode"], 8)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.linkname = ""
    info.devmajor = 0
    info.devminor = 0
    info.pax_headers = {}
    return info


def _write_archive(output: BinaryIO, entries: list[dict[str, Any]], payloads: list[bytes]) -> None:
    if len(entries) != len(payloads):
        raise SourceArchiveError("source archive entry/payload accounting drifted")
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for index, entry in enumerate(entries):
            payload = payloads[index]
            archive.addfile(_tar_info(entry), io.BytesIO(payload))


def create_archive(repository: Path, manifest_bytes: bytes, output: BinaryIO) -> dict[str, Any]:
    manifest, entries = load_manifest_bytes(manifest_bytes)
    root_descriptor = _open_root(repository)
    try:
        payloads = [_source_bytes_at(root_descriptor, entry) for entry in entries]
    finally:
        os.close(root_descriptor)
    _write_archive(output, entries, payloads)
    return {
        "schema_id": ARCHIVE_SCHEMA,
        "status": "created",
        "document_sha256": manifest["document_sha256"],
        "member_count": len(entries),
        "content_bytes": sum(len(payload) for payload in payloads),
    }


def _octal(field: bytes, label: str) -> int:
    stripped = field.rstrip(b"\0 ").lstrip(b" ")
    if not stripped:
        return 0
    if any(value not in b"01234567" for value in stripped):
        raise SourceArchiveError(f"USTAR {label} is not canonical octal")
    return int(stripped, 8)


def _raw_members(value: bytes) -> tuple[list[dict[str, Any]], int]:
    if not value or len(value) % RECORD_SIZE != 0:
        raise SourceArchiveError("source archive is not aligned to the 10 KiB USTAR record size")
    offset = 0
    members: list[dict[str, Any]] = []
    while offset + BLOCK_SIZE <= len(value):
        header = value[offset : offset + BLOCK_SIZE]
        if header == ZERO_BLOCK:
            if value[offset + BLOCK_SIZE : offset + 2 * BLOCK_SIZE] != ZERO_BLOCK:
                raise SourceArchiveError("source archive has only one terminal zero block")
            if any(value[offset:]):
                raise SourceArchiveError("source archive has nonzero trailing records")
            return members, offset
        stored_checksum = _octal(header[148:156], "checksum")
        checksum_header = header[:148] + b" " * 8 + header[156:]
        if sum(checksum_header) != stored_checksum:
            raise SourceArchiveError("source archive header checksum is invalid")
        if header[257:263] != b"ustar\0" or header[263:265] != b"00":
            raise SourceArchiveError("source archive contains a non-USTAR header")
        if header[156:157] != tarfile.REGTYPE:
            raise SourceArchiveError("source archive contains a PAX/GNU/link/control member")
        if any(header[157:257]):
            raise SourceArchiveError("source archive regular member has a link target")
        if any(header[345:500]):
            raise SourceArchiveError("source archive uses a USTAR prefix extension")
        try:
            name = header[:100].split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise SourceArchiveError("source archive member name is not ASCII") from error
        name = _safe_relative(name)
        size = _octal(header[124:136], "size")
        payload_start = offset + BLOCK_SIZE
        payload_end = payload_start + size
        padded_end = payload_start + ((size + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        if padded_end > len(value):
            raise SourceArchiveError("source archive member exceeds the archive")
        if any(value[payload_end:padded_end]):
            raise SourceArchiveError("source archive member has nonzero block padding")
        members.append(
            {
                "relative": name,
                "mode": _octal(header[100:108], "mode"),
                "uid": _octal(header[108:116], "uid"),
                "gid": _octal(header[116:124], "gid"),
                "size": size,
                "mtime": _octal(header[136:148], "mtime"),
                "uname": header[265:297].split(b"\0", 1)[0],
                "gname": header[297:329].split(b"\0", 1)[0],
                "devmajor": _octal(header[329:337], "device major"),
                "devminor": _octal(header[337:345], "device minor"),
                "payload": value[payload_start:payload_end],
            }
        )
        offset = padded_end
    raise SourceArchiveError("source archive has no terminal zero records")


def verify_archive_bytes(manifest_bytes: bytes, archive_bytes: bytes) -> dict[str, Any]:
    manifest, entries = load_manifest_bytes(manifest_bytes)
    members, terminal_offset = _raw_members(archive_bytes)
    if len(members) != len(entries):
        raise SourceArchiveError(
            f"archive member count {len(members)} differs from manifest {len(entries)}"
        )
    payloads: list[bytes] = []
    for index, entry in enumerate(entries):
        member = members[index]
        if member["relative"] != entry["relative"]:
            raise SourceArchiveError(f"archive member {index} path/order differs from manifest")
        if member["mode"] != int(entry["mode"], 8):
            raise SourceArchiveError(f"archive mode differs from manifest: {entry['relative']}")
        if (
            member["uid"] != 0
            or member["gid"] != 0
            or member["mtime"] != 0
            or member["uname"]
            or member["gname"]
            or member["devmajor"] != 0
            or member["devminor"] != 0
        ):
            raise SourceArchiveError(f"archive metadata is not normalized: {entry['relative']}")
        payload = member["payload"]
        if (
            member["size"] != entry["size"]
            or len(payload) != entry["size"]
            or _sha256(payload) != entry["sha256"]
        ):
            raise SourceArchiveError(f"archive content differs from manifest: {entry['relative']}")
        payloads.append(payload)
    rebuilt = io.BytesIO()
    _write_archive(rebuilt, entries, payloads)
    if rebuilt.getvalue() != archive_bytes:
        raise SourceArchiveError("source archive bytes are not the canonical deterministic USTAR")
    return {
        "schema_id": ARCHIVE_SCHEMA,
        "status": "valid",
        "document_sha256": manifest["document_sha256"],
        "archive_sha256": _sha256(archive_bytes),
        "archive_bytes": len(archive_bytes),
        "member_count": len(members),
        "member_names_sha256": _sha256(
            ("\n".join(entry["relative"] for entry in entries) + "\n").encode("ascii")
        ),
        "content_bytes": manifest["total_bytes"],
        "terminal_zero_bytes": len(archive_bytes) - terminal_offset,
        "regular_only": True,
        "pax_or_extended_headers": False,
        "metadata_normalized": True,
    }


def _walk_tree(
    root_descriptor: int,
    visitor: Any,
) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def descend(directory_descriptor: int, parent: str) -> None:
        with os.scandir(directory_descriptor) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            relative = name if parent == "." else f"{parent}/{name}"
            _safe_relative(relative)
            details = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                try:
                    child = os.open(name, directory_flags, dir_fd=directory_descriptor)
                except OSError as error:
                    raise SourceArchiveError(
                        f"extracted directory cannot be opened without links: {relative}"
                    ) from error
                try:
                    opened = os.fstat(child)
                    if not stat.S_ISDIR(opened.st_mode) or (details.st_dev, details.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise SourceArchiveError(
                            f"extracted directory changed while opening: {relative}"
                        )
                    visitor(relative, "directory", child, opened)
                    descend(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(details.st_mode):
                try:
                    child = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                except OSError as error:
                    raise SourceArchiveError(
                        f"extracted file cannot be opened without links: {relative}"
                    ) from error
                try:
                    opened = os.fstat(child)
                    if not stat.S_ISREG(opened.st_mode) or (details.st_dev, details.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise SourceArchiveError(
                            f"extracted file changed while opening: {relative}"
                        )
                    visitor(relative, "file", child, opened)
                finally:
                    os.close(child)
            else:
                raise SourceArchiveError(
                    f"extracted tree contains a link or special file: {relative}"
                )

    root_details = os.fstat(root_descriptor)
    visitor(".", "directory", root_descriptor, root_details)
    descend(root_descriptor, ".")


def verify_tree(repository: Path, manifest_bytes: bytes) -> dict[str, Any]:
    manifest, entries = load_manifest_bytes(manifest_bytes)
    expected_files = {entry["relative"]: entry for entry in entries}
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while str(parent) not in {"", "."}:
            expected_directories.add(str(parent))
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories = {"."}
    provenance_paths: list[str] = []

    def visit(relative: str, kind: str, descriptor: int, details: os.stat_result) -> None:
        names = set(_xattr_names_fd(descriptor, relative))
        unexpected = names - ALLOWED_EXTRACTED_XATTRS
        if unexpected:
            rendered = ",".join(os.fsdecode(name) for name in sorted(unexpected))
            raise SourceArchiveError(
                f"extracted path has a non-allowlisted extended attribute: {relative}: {rendered}"
            )
        if b"com.apple.provenance" in names:
            provenance_paths.append(relative)
        if kind == "directory":
            if details.st_uid != os.getuid() or details.st_gid != os.getgid():
                raise SourceArchiveError(f"extracted directory ownership is unsafe: {relative}")
            mode = stat.S_IMODE(details.st_mode)
            if mode != 0o700:
                raise SourceArchiveError(f"extracted directory mode is not private: {relative}")
            observed_directories.add(relative)
            return
        entry = expected_files.get(relative)
        if entry is None:
            raise SourceArchiveError(f"extracted tree contains an extra file: {relative}")
        if (
            details.st_nlink != 1
            or details.st_uid != os.getuid()
            or details.st_gid != os.getgid()
            or stat.S_IMODE(details.st_mode) != int(entry["mode"], 8)
            or details.st_size != entry["size"]
        ):
            raise SourceArchiveError(f"extracted file metadata differs: {relative}")
        _read_file_descriptor(descriptor, entry, relative)
        observed_files.add(relative)

    root_descriptor = _open_root(repository)
    try:
        _walk_tree(root_descriptor, visit)
    finally:
        os.close(root_descriptor)
    if observed_files != set(expected_files):
        raise SourceArchiveError("extracted tree is missing manifest files")
    if observed_directories != expected_directories:
        raise SourceArchiveError("extracted tree directory set differs from manifest parents")
    return {
        "schema_id": ARCHIVE_SCHEMA,
        "status": "tree-valid",
        "document_sha256": manifest["document_sha256"],
        "member_count": len(entries),
        "content_bytes": manifest["total_bytes"],
        "file_modes_normalized": True,
        "extracted_xattr_policy": "allow-host-com.apple.provenance-only",
        "provenance_path_count": len(provenance_paths),
        "provenance_paths_sha256": _sha256(
            ("\n".join(sorted(provenance_paths)) + "\n").encode("ascii")
        ),
        "unexpected_extended_attributes_absent": True,
    }


def _validate_cleanup_entry(
    details: os.stat_result,
    *,
    root_device: int,
    label: str,
) -> None:
    if details.st_dev != root_device:
        raise SourceArchiveError(f"pytest cleanup refuses a device crossing: {label}")
    if details.st_uid != os.getuid() or details.st_gid != os.getgid():
        raise SourceArchiveError(f"pytest cleanup refuses foreign ownership: {label}")
    if not (
        stat.S_ISDIR(details.st_mode)
        or stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
    ):
        raise SourceArchiveError(f"pytest cleanup refuses a special file: {label}")


def _remove_owned_directory_contents(
    directory_descriptor: int,
    *,
    root_device: int,
    parent_label: str,
    counters: dict[str, int],
) -> None:
    details = os.fstat(directory_descriptor)
    _validate_cleanup_entry(
        details,
        root_device=root_device,
        label=parent_label,
    )
    if not stat.S_ISDIR(details.st_mode):
        raise SourceArchiveError(f"pytest cleanup root is not a directory: {parent_label}")
    os.fchmod(directory_descriptor, 0o700)
    with os.scandir(directory_descriptor) as iterator:
        names = sorted(entry.name for entry in iterator)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for name in names:
        label = f"{parent_label}/{name}"
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        _validate_cleanup_entry(before, root_device=root_device, label=label)
        if stat.S_ISDIR(before.st_mode):
            os.chmod(
                name,
                0o700,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            child = os.open(name, directory_flags, dir_fd=directory_descriptor)
            try:
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or (before.st_dev, before.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    raise SourceArchiveError(
                        f"pytest cleanup directory changed while opening: {label}"
                    )
                _remove_owned_directory_contents(
                    child,
                    root_device=root_device,
                    parent_label=label,
                    counters=counters,
                )
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
                    current.st_dev,
                    current.st_ino,
                ):
                    raise SourceArchiveError(
                        f"pytest cleanup directory changed before removal: {label}"
                    )
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=directory_descriptor)
            counters["directories"] += 1
        elif stat.S_ISREG(before.st_mode):
            # Unlink only needs a writable parent, but normalizing a private,
            # single-link pytest file first also handles read-only fixtures.
            # Never chmod a multiply-linked inode because that could mutate a
            # surviving link outside this bounded tree.
            if before.st_nlink == 1:
                os.chmod(
                    name,
                    0o600,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(current.st_mode) or (before.st_dev, before.st_ino) != (
                    current.st_dev,
                    current.st_ino,
                ):
                    raise SourceArchiveError(f"pytest cleanup file changed before removal: {label}")
            os.unlink(name, dir_fd=directory_descriptor)
            counters["regular_files"] += 1
            counters["regular_file_bytes"] += before.st_size
        else:
            os.unlink(name, dir_fd=directory_descriptor)
            counters["symlinks"] += 1


def cleanup_pytest_basetemp(parent: Path) -> dict[str, Any]:
    """Remove only the fixed pytest basetemp without following any links."""

    parent_descriptor = _open_root(parent)
    try:
        parent_details = os.fstat(parent_descriptor)
        if parent_details.st_uid != os.getuid() or parent_details.st_gid != os.getgid():
            raise SourceArchiveError("pytest cleanup parent has foreign ownership")
        try:
            before = os.stat(
                PYTEST_BASETEMP_NAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return {
                "schema_id": CLEANUP_SCHEMA,
                "status": "absent",
                "basetemp_name": PYTEST_BASETEMP_NAME,
                "removed": False,
                "directories_removed": 0,
                "regular_files_removed": 0,
                "regular_file_bytes_removed": 0,
                "symlinks_removed": 0,
                "no_links_followed": True,
            }
        _validate_cleanup_entry(
            before,
            root_device=parent_details.st_dev,
            label=PYTEST_BASETEMP_NAME,
        )
        if not stat.S_ISDIR(before.st_mode):
            raise SourceArchiveError("pytest basetemp is not a direct regular directory")
        os.chmod(
            PYTEST_BASETEMP_NAME,
            0o700,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_descriptor = os.open(
            PYTEST_BASETEMP_NAME,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        counters = {
            "directories": 0,
            "regular_files": 0,
            "regular_file_bytes": 0,
            "symlinks": 0,
        }
        try:
            opened = os.fstat(root_descriptor)
            if not stat.S_ISDIR(opened.st_mode) or (before.st_dev, before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise SourceArchiveError("pytest basetemp changed while opening")
            _remove_owned_directory_contents(
                root_descriptor,
                root_device=parent_details.st_dev,
                parent_label=PYTEST_BASETEMP_NAME,
                counters=counters,
            )
            current = os.stat(
                PYTEST_BASETEMP_NAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
                current.st_dev,
                current.st_ino,
            ):
                raise SourceArchiveError("pytest basetemp changed before removal")
        finally:
            os.close(root_descriptor)
        os.rmdir(PYTEST_BASETEMP_NAME, dir_fd=parent_descriptor)
        try:
            os.stat(
                PYTEST_BASETEMP_NAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise SourceArchiveError("pytest basetemp remains after cleanup")
        return {
            "schema_id": CLEANUP_SCHEMA,
            "status": "removed",
            "basetemp_name": PYTEST_BASETEMP_NAME,
            "removed": True,
            "directories_removed": counters["directories"] + 1,
            "regular_files_removed": counters["regular_files"],
            "regular_file_bytes_removed": counters["regular_file_bytes"],
            "symlinks_removed": counters["symlinks"],
            "no_links_followed": True,
        }
    finally:
        os.close(parent_descriptor)


def _gate_cleanup_parent() -> Path:
    environment_value = os.environ.get("TMPDIR")
    if not environment_value:
        raise SourceArchiveError("pytest cleanup requires the effective TMPDIR")
    if not os.path.isabs(environment_value):
        raise SourceArchiveError("pytest cleanup TMPDIR must be absolute")
    environment_normalized = os.path.normpath(environment_value)
    return Path(environment_normalized)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--repository", type=Path, required=True)
    create.add_argument("--manifest", default="-")
    create.add_argument("--output", default="-")
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--archive", required=True)
    tree = commands.add_parser("verify-tree")
    tree.add_argument("--manifest", required=True)
    tree.add_argument("--repository", type=Path, required=True)
    commands.add_parser("cleanup-pytest")
    return result


def _write_output(path: str, value: bytes) -> None:
    if path == "-":
        sys.stdout.buffer.write(value)
        sys.stdout.buffer.flush()
        return
    destination = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(value)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "create":
            manifest_bytes = _input_bytes(arguments.manifest, MAX_MANIFEST_BYTES, "source manifest")
            archive = io.BytesIO()
            create_archive(arguments.repository, manifest_bytes, archive)
            archive_bytes = archive.getvalue()
            verify_archive_bytes(manifest_bytes, archive_bytes)
            _write_output(arguments.output, archive_bytes)
        elif arguments.command == "verify":
            manifest_bytes = _input_bytes(arguments.manifest, MAX_MANIFEST_BYTES, "source manifest")
            archive_bytes = _input_bytes(arguments.archive, MAX_ARCHIVE_BYTES, "source archive")
            result = verify_archive_bytes(manifest_bytes, archive_bytes)
            sys.stdout.buffer.write(canonical_json(result) + b"\n")
        elif arguments.command == "verify-tree":
            manifest_bytes = _input_bytes(arguments.manifest, MAX_MANIFEST_BYTES, "source manifest")
            result = verify_tree(arguments.repository, manifest_bytes)
            sys.stdout.buffer.write(canonical_json(result) + b"\n")
        else:
            result = cleanup_pytest_basetemp(_gate_cleanup_parent())
            sys.stdout.buffer.write(canonical_json(result) + b"\n")
    except (OSError, SourceArchiveError, tarfile.TarError, ValueError) as error:
        print(f"R2-MAP source archive refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
