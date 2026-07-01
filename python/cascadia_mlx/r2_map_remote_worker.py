# ruff: noqa: UP006, UP031, UP035, UP045
"""Dependency-free john2 storage worker for the R2-MAP campaign.

The production copy of this file is content-addressed and installed below the
dedicated john2 campaign root.  Every invocation authenticates one hashed
command, revalidates the frozen host/root identity, and emits one framed,
hash-bound receipt.  The worker intentionally has no MLX or repository import.

Keep this module parseable by Apple's Python 3.9: john2's control plane must not
depend on a project virtual environment merely to protect project storage.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import plistlib
import pwd
import re
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
COMMAND_SCHEMA = "cascadia.r2-map.remote-command.v1"
RECEIPT_SCHEMA = "cascadia.r2-map.remote-receipt.v2"
LEGACY_RECEIPT_SCHEMA = "cascadia.r2-map.remote-receipt.v1"
OBJECT_TOKEN_SCHEMA = "cascadia.r2-map.remote-object-token.v1"
TRANSACTION_SCHEMA = "cascadia.r2-map.remote-transaction-manifest.v1"
TRANSACTION_OUTCOME_SCHEMA = "cascadia.r2-map.remote-transaction-outcome.v1"
RUN_STATE_SCHEMA = "cascadia.r2-map.remote-run-state.v1"
RUN_SUPERVISOR_SCHEMA = "cascadia.r2-map.remote-run-supervisor.v1"
RUN_CLEANUP_TOKEN_SCHEMA = "cascadia.r2-map.run-cleanup-token.v1"
FAILED_RUN_CLEANUP_TOKEN_SCHEMA = "cascadia.r2-map.failed-run-cleanup-token.v1"
PUT_JOURNAL_SCHEMA = "cascadia.r2-map.remote-put-journal.v1"
DATA_RESERVATION_SCHEMA = "cascadia.r2-map.remote-data-reservation.v1"
FRAME_MAGIC = b"R2RSv1\x00\x00"
FRAME_PREFIX = struct.Struct(">8sI")
FRAME_SUFFIX = struct.Struct(">I")

PRODUCTION_ROOT = Path("/Users/john2/cascadia-bench/r2-map-v1")
PRODUCTION_HOST = "john2"
PRODUCTION_USER = "john2"
PRODUCTION_UID = 501
PRODUCTION_GID = 20
PRODUCTION_ROOT_DEVICE = 16777233
PRODUCTION_ROOT_INODE = 861935
PRODUCTION_IDENTITY_SHA256 = "77b4a32efdaadaed8bab28d5d99f62e0192ce1dc67f978424691ffd6d805d1e1"
GIB = 1 << 30
MIN_FREE_BYTES = 100 * GIB
MAX_CAMPAIGN_BYTES = 80 * GIB
MAX_DATA_BYTES = 78 * GIB
RECEIPT_BUDGET_BYTES = 2 * GIB
MAX_RECEIPT_BYTES = 64 * (1 << 10)
MAX_RECEIPT_ENTRIES = 100_000
MAX_CAMPAIGN_ENTRIES = 500_000
MAX_PUT_JOURNAL_BYTES = 64 * (1 << 10)
MAX_DATA_RESERVATION_BYTES = 64 * (1 << 10)
MAX_RUN_BYTES = 40 * GIB
MAX_RANGE_BYTES = 64 * (1 << 20)
MAX_STATUS_BYTES = 64 * (1 << 10)
MAX_MANIFEST_BYTES = 2 * (1 << 20)
MAX_REQUEST_BYTES = 2 * (1 << 20)
MAX_UNKNOWN_STREAM_BYTES = 1 << 30
MAX_EPHEMERAL_RUNTIME_BYTES = 64 * (1 << 20)
MAX_EPHEMERAL_MANIFEST_BYTES = 64 * (1 << 10)
REQUEST_MAX_AGE_MS = 10 * 60 * 1000
REQUEST_MAX_FUTURE_MS = 60 * 1000
RUN_CLEANUP_TOKEN_LIFETIME_MS = 60 * 60 * 1000
MAX_RUN_CLEANUP_ENTRIES = 500_000

PROTOCOL_SCHEMA = "cascadia.r2-map.remote-protocol.v2"
PROTOCOL_VERSION = 2
CAPACITY_PROOF_SCHEMA = "cascadia.r2-map.remote-capacity-proof.v2"
PUT_RESULT_SCHEMA = "cascadia.r2-map.remote-put-result.v2"
PROTOCOL_STABLE_MUTATING_OPERATIONS = (
    "put-file",
    "put-stream",
    "publish-status",
    "lock-acquire",
    "lock-renew",
    "lock-release",
    "transaction-begin",
    "transaction-put",
    "transaction-import",
    "transaction-commit",
    "transaction-abort",
    "run-command",
    "run-controller",
    "run-cleanup-commit",
    "failed-run-cleanup-commit",
)
CAPACITY_PROOF_KEYS = (
    "schema_id",
    "schema_version",
    "protocol_sha256",
    "root",
    "root_mode",
    "root_uid",
    "root_gid",
    "root_device",
    "root_inode",
    "host_identity_sha256",
    "filesystem",
    "protocol",
    "internal",
    "removable",
    "solid_state",
    "free_bytes",
    "total_bytes",
    "min_free_bytes",
    "campaign_apparent_bytes",
    "max_campaign_bytes",
    "campaign_data_apparent_bytes",
    "max_data_bytes",
    "receipt_apparent_bytes",
    "receipt_entries",
    "receipt_reservation_bytes",
    "receipt_reservation_entries",
    "data_reservation_apparent_bytes",
    "data_reservation_reserved_bytes",
    "data_reservation_entries",
    "receipt_budget_bytes",
    "max_receipt_bytes",
    "max_receipt_entries",
)
PUT_RESULT_KEYS = (
    "schema_id",
    "schema_version",
    "protocol_sha256",
    "relative",
    "sha256",
    "size",
    "mode",
    "previous_sha256",
    "projected_campaign_bytes",
    "projected_data_bytes",
    "projected_free_bytes",
    "receipt_capacity_reserved_bytes",
    "receipt_reservation_apparent_bytes",
    "data_reservation_apparent_bytes",
    "journal_bytes",
    "backup_bytes",
    "transaction_overhead_bytes",
    "storage_precommit",
    "storage_staged",
    "storage_transaction",
    "payload_size",
    "payload_sha256",
)
PROTOCOL_DOCUMENT: Dict[str, Any] = {
    "schema_id": PROTOCOL_SCHEMA,
    "schema_version": PROTOCOL_VERSION,
    "command_schema": COMMAND_SCHEMA,
    "receipt_schema": RECEIPT_SCHEMA,
    "frame_schema": "cascadia.r2-map.remote-frame.v1",
    "capacity_proof_schema": CAPACITY_PROOF_SCHEMA,
    "capacity_proof_keys": list(CAPACITY_PROOF_KEYS),
    "put_result_schema": PUT_RESULT_SCHEMA,
    "put_result_keys": list(PUT_RESULT_KEYS),
    "stable_mutating_operations": list(PROTOCOL_STABLE_MUTATING_OPERATIONS),
    "limits": {
        "max_campaign_bytes": MAX_CAMPAIGN_BYTES,
        "max_data_bytes": MAX_DATA_BYTES,
        "receipt_budget_bytes": RECEIPT_BUDGET_BYTES,
        "max_receipt_bytes": MAX_RECEIPT_BYTES,
        "max_receipt_entries": MAX_RECEIPT_ENTRIES,
    },
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        PROTOCOL_DOCUMENT,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()

LAYOUT_DIRECTORIES = (
    "control",
    "control/bin",
    "control/locks",
    "control/receipt-reservations",
    "control/data-reservations",
    "control/transactions",
    "control/run-states",
    "control/run-supervisors",
    "control/receipts",
    "source",
    "build",
    "toolchains",
    "home",
    "cache",
    "cache/cargo-home",
    "cache/rustup",
    "cache/uv",
    "cache/pycache",
    "cache/runs",
    "tmp",
    "datasets",
    "checkpoints",
    "logs",
    "reports",
    "benchmarks",
    "bundles",
    "opponent-pool",
    "runs",
)

IMMUTABLE_TRANSACTION_TOP_LEVELS = frozenset(
    {
        "source",
        "datasets",
        "checkpoints",
        "reports",
        "benchmarks",
        "bundles",
        "opponent-pool",
        "runs",
    }
)
MUTABLE_FILE_TOP_LEVELS = frozenset(
    {"control", "logs", "reports", "benchmarks", "runs", "checkpoints", "bundles"}
)
RUN_OUTPUT_TOP_LEVELS = frozenset({"logs", "runs", "reports", "benchmarks"})
SAFE_SYSTEM_EXECUTABLES = frozenset(
    {
        "/usr/bin/clang",
        "/usr/bin/clang++",
        "/usr/bin/codesign",
        "/usr/bin/file",
        "/usr/bin/git",
        "/usr/bin/lipo",
        "/usr/bin/make",
        "/usr/bin/xcrun",
        "/opt/homebrew/bin/uv",
        "/opt/homebrew/bin/uvx",
    }
)
SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "CARGO_INCREMENTAL",
        "CARGO_PROFILE_RELEASE_DEBUG",
        "CARGO_TERM_COLOR",
        "RAYON_NUM_THREADS",
        "RUST_BACKTRACE",
        "RUSTFLAGS",
        "SOURCE_DATE_EPOCH",
    }
)
CONTROLLER_COMMANDS = frozenset(
    {
        "preflight",
        "show-state",
        "transition",
        "record-decision",
        "verify-decisions",
        "advance",
        "import-work-receipt",
        "import-benchmark-feed",
        "reconcile-controller",
        "recover-current-phase",
        "phase-barrier",
        "publish-dashboard-status",
    }
)
STABLE_MUTATING_OPERATIONS = frozenset(PROTOCOL_STABLE_MUTATING_OPERATIONS)
RESOURCE_LOCKED_OPERATIONS = STABLE_MUTATING_OPERATIONS | frozenset(
    {
        "put-stream",
        "lock-acquire",
        "lock-renew",
        "lock-release",
        "run-cleanup-prepare",
        "failed-run-cleanup-prepare",
    }
)
IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RemoteWorkerError(RuntimeError):
    """A command or storage invariant failed closed."""


@dataclass(frozen=True)
class WorkerContract:
    root: Path = PRODUCTION_ROOT
    expected_host: str = PRODUCTION_HOST
    expected_user: str = PRODUCTION_USER
    expected_uid: int = PRODUCTION_UID
    expected_gid: int = PRODUCTION_GID
    expected_root_device: int = PRODUCTION_ROOT_DEVICE
    expected_root_inode: int = PRODUCTION_ROOT_INODE
    expected_identity_sha256: str = PRODUCTION_IDENTITY_SHA256
    min_free_bytes: int = MIN_FREE_BYTES
    max_campaign_bytes: int = MAX_CAMPAIGN_BYTES
    max_data_bytes: int = MAX_DATA_BYTES
    receipt_budget_bytes: int = RECEIPT_BUDGET_BYTES
    max_receipt_bytes: int = MAX_RECEIPT_BYTES
    max_receipt_entries: int = MAX_RECEIPT_ENTRIES


PRODUCTION_CONTRACT = WorkerContract()
_OUTER_GLOBAL_LOCK_FD: Optional[int] = None


def protocol_info() -> Dict[str, Any]:
    return {**PROTOCOL_DOCUMENT, "protocol_sha256": PROTOCOL_SHA256}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def document_sha256(document: Mapping[str, Any], field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def request_semantic_sha256(operation: Any, arguments: Any) -> str:
    return hashlib.sha256(
        canonical_json({"operation": operation, "arguments": arguments})
    ).hexdigest()


def request_command_sha256(request: Mapping[str, Any]) -> str:
    payload = dict(request)
    payload.pop("command_sha256", None)
    payload.pop("issued_unix_ms", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise RemoteWorkerError("%s must be a lowercase SHA-256" % label)
    return value


def _validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise RemoteWorkerError("%s is not a safe identifier" % label)
    return value


def _relative_parts(relative: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(relative, str) or not relative or len(relative.encode("utf-8")) > 1024:
        raise RemoteWorkerError("%s must be a bounded relative path" % label)
    if "\x00" in relative or "\\" in relative:
        raise RemoteWorkerError("%s contains a forbidden character" % label)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or str(pure) != relative:
        raise RemoteWorkerError("%s is not canonical" % label)
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RemoteWorkerError("%s escapes the campaign root" % label)
    return tuple(parts)


def _lstat_no_symlink(path: Path, label: str) -> os.stat_result:
    details = os.lstat(str(path))
    if stat.S_ISLNK(details.st_mode):
        raise RemoteWorkerError("%s is a symlink" % label)
    return details


def _safe_path(
    contract: WorkerContract,
    relative: Any,
    label: str,
    *,
    create_parents: bool = False,
) -> Path:
    parts = _relative_parts(relative, label)
    root_stat = _lstat_no_symlink(contract.root, "campaign root")
    current = contract.root
    for part in parts[:-1]:
        current = current / part
        try:
            details = _lstat_no_symlink(current, label + " ancestor")
        except FileNotFoundError as error:
            if not create_parents:
                raise RemoteWorkerError("%s ancestor is missing" % label) from error
            os.mkdir(str(current), 0o700)
            _fsync_directory(current.parent)
            details = _lstat_no_symlink(current, label + " ancestor")
        if not stat.S_ISDIR(details.st_mode) or details.st_dev != root_stat.st_dev:
            raise RemoteWorkerError("%s ancestor is not a contained directory" % label)
    result = contract.root.joinpath(*parts)
    try:
        details = _lstat_no_symlink(result, label)
    except FileNotFoundError:
        return result
    if details.st_dev != root_stat.st_dev:
        raise RemoteWorkerError("%s crosses the campaign device" % label)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / (".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    descriptor = os.open(
        str(temporary),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RemoteWorkerError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(str(temporary))
        raise
    else:
        os.close(descriptor)
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def _atomic_write_recoverable(
    path: Path, payload: bytes, mode: int, label: str
) -> None:
    """Atomic write with one O(1), transaction-addressed crash residue path."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / (".%s.pending.tmp" % path.name)
    try:
        stale = os.lstat(str(temporary))
    except FileNotFoundError:
        pass
    else:
        parent = _lstat_no_symlink(path.parent, "%s parent" % label)
        if (
            not stat.S_ISREG(stale.st_mode)
            or stale.st_nlink != 1
            or stale.st_dev != parent.st_dev
            or (stale.st_uid, stale.st_gid) != (parent.st_uid, parent.st_gid)
            or stat.S_IMODE(stale.st_mode) != mode
            or stale.st_size > MAX_REQUEST_BYTES
        ):
            raise RemoteWorkerError("%s partial metadata is invalid" % label)
        os.unlink(str(temporary))
        _fsync_directory(path.parent)
    descriptor = os.open(
        str(temporary),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RemoteWorkerError("short %s write" % label)
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(str(temporary))
        raise
    else:
        os.close(descriptor)
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def _remove_recoverable_atomic_partial(
    contract: WorkerContract,
    entry: os.DirEntry[str],
    *,
    mode: int,
    maximum: int,
    label: str,
) -> bool:
    if not entry.name.startswith(".") or not entry.name.endswith(".pending.tmp"):
        return False
    details = entry.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid)
        != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != mode
        or details.st_size > maximum
    ):
        raise RemoteWorkerError("%s partial metadata is invalid" % label)
    return True


def _stage_stream(
    path: Path,
    source: BinaryIO,
    expected_size: int,
    expected_sha256: str,
    mode: int,
    *,
    temporary: Optional[Path] = None,
) -> Path:
    temporary = temporary or path.parent / (".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    descriptor = os.open(
        str(temporary),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    digest = hashlib.sha256()
    remaining = expected_size
    try:
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise RemoteWorkerError("upload ended before declared size")
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RemoteWorkerError("short upload write")
                view = view[written:]
            digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise RemoteWorkerError("upload exceeded declared size")
        if digest.hexdigest() != expected_sha256:
            raise RemoteWorkerError("upload SHA-256 mismatch")
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(str(temporary))
        raise
    else:
        os.close(descriptor)
    return temporary


def _atomic_stream(
    path: Path,
    source: BinaryIO,
    expected_size: int,
    expected_sha256: str,
    mode: int,
) -> None:
    temporary = _stage_stream(path, source, expected_size, expected_sha256, mode)
    try:
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(str(temporary))
        raise


def _write_unknown_stream_temporary(
    path: Path,
    source: BinaryIO,
    max_bytes: int,
    mode: int,
    *,
    temporary: Optional[Path] = None,
) -> Tuple[Path, str, int]:
    if temporary is None:
        temporary = path.parent / (".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    descriptor = os.open(
        str(temporary),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = source.read(min(1 << 20, max_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise RemoteWorkerError("unknown-size upload exceeded its declared bound")
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RemoteWorkerError("short unknown-size upload write")
                view = view[written:]
            digest.update(chunk)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(str(temporary))
        raise
    else:
        os.close(descriptor)
    return temporary, digest.hexdigest(), size


def _platform_uuid() -> str:
    output = subprocess.check_output(
        ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True
    )
    values = [
        line.split("=", 1)[1].strip().strip('"')
        for line in output.splitlines()
        if "IOPlatformUUID" in line
    ]
    if len(values) != 1:
        raise RemoteWorkerError("canonical platform UUID is unavailable")
    return values[0]


def _data_volume_info() -> Dict[str, Any]:
    data_mount = Path("/System/Volumes/Data")
    value = plistlib.loads(
        subprocess.check_output(["/usr/sbin/diskutil", "info", "-plist", str(data_mount)])
    )
    if not isinstance(value, dict):
        raise RemoteWorkerError("canonical Data-volume metadata is invalid")
    return value


def _host_identity(
    contract: WorkerContract, info: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    volume = dict(_data_volume_info() if info is None else info)
    root_stat = _lstat_no_symlink(contract.root, "campaign root")
    return {
        "device_identifier": volume.get("DeviceIdentifier"),
        "filesystem": volume.get("FilesystemType"),
        "gid": os.getgid(),
        "hostname": os.uname().nodename,
        "platform_uuid": _platform_uuid(),
        "protocol": volume.get("BusProtocol"),
        "root_device": root_stat.st_dev,
        "root_inode": root_stat.st_ino,
        "uid": os.getuid(),
        "user": pwd.getpwuid(os.getuid()).pw_name,
        "volume_uuid": volume.get("VolumeUUID"),
    }


def _symlink_boundary(path: Path, allowed_prefixes: Sequence[Path]) -> Optional[Path]:
    for prefix in allowed_prefixes:
        try:
            relative = path.relative_to(prefix)
        except ValueError:
            continue
        if prefix.name in {"runs", "build", "tmp"} and relative.parts:
            return prefix / relative.parts[0]
        return prefix
    return None


def _campaign_relative_diagnostic(path: Path, diagnostic_root: Path) -> str:
    try:
        relative = path.relative_to(diagnostic_root)
    except ValueError:
        return path.name
    return relative.as_posix() or "."


def _apparent_size(
    root: Path,
    *,
    allowed_symlink_prefixes: Sequence[Path] = (),
    diagnostic_root: Optional[Path] = None,
) -> int:
    total = 0
    entry_count = 0
    stack = [root]
    root_stat = _lstat_no_symlink(root, "campaign root")
    diagnostic_root = diagnostic_root or root
    while stack:
        current = stack.pop()
        with os.scandir(str(current)) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_CAMPAIGN_ENTRIES:
                    raise RemoteWorkerError("campaign tree exceeds its audited entry limit")
                details = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(details.st_mode):
                    path = Path(entry.path)
                    diagnostic = _campaign_relative_diagnostic(path, diagnostic_root)
                    boundary = _symlink_boundary(path, allowed_symlink_prefixes)
                    if boundary is None:
                        raise RemoteWorkerError(
                            f"campaign tree contains an unauthorized symlink: {diagnostic}"
                        )
                    try:
                        resolved = path.resolve(strict=True)
                    except FileNotFoundError as error:
                        raise RemoteWorkerError(
                            f"campaign symlink is dangling: {diagnostic}"
                        ) from error
                    except RuntimeError as error:
                        raise RemoteWorkerError(
                            f"campaign symlink has a resolution loop: {diagnostic}"
                        ) from error
                    try:
                        resolved.relative_to(boundary)
                    except ValueError as error:
                        raise RemoteWorkerError(
                            f"campaign symlink escapes its run boundary: {diagnostic}"
                        ) from error
                    try:
                        resolved_details = os.stat(str(resolved))
                    except FileNotFoundError as error:
                        raise RemoteWorkerError(
                            f"campaign symlink became dangling: {diagnostic}"
                        ) from error
                    if resolved_details.st_dev != root_stat.st_dev:
                        raise RemoteWorkerError(
                            f"campaign symlink target crosses a device boundary: {diagnostic}"
                        )
                    total += details.st_size
                    continue
                if details.st_dev != root_stat.st_dev:
                    raise RemoteWorkerError("campaign tree crosses a device boundary")
                if stat.S_ISDIR(details.st_mode):
                    stack.append(Path(entry.path))
                elif stat.S_ISREG(details.st_mode):
                    total += details.st_size
                else:
                    raise RemoteWorkerError("campaign tree contains a special file")
    return total


def _campaign_apparent_size(contract: WorkerContract) -> int:
    return _apparent_size(
        contract.root,
        allowed_symlink_prefixes=(
            contract.root / "cache/runs",
            contract.root / "build",
            contract.root / "toolchains",
        ),
        diagnostic_root=contract.root,
    )


def _receipt_budget_state(
    contract: WorkerContract, *, validate_documents: bool = False
) -> Dict[str, Any]:
    """Measure the small durable receipt namespace without rescanning campaign data."""

    if (
        contract.max_data_bytes <= 0
        or contract.receipt_budget_bytes <= 0
        or contract.max_data_bytes + contract.receipt_budget_bytes
        != contract.max_campaign_bytes
        or contract.max_receipt_bytes <= 0
        or contract.max_receipt_bytes > contract.receipt_budget_bytes
        or contract.max_receipt_entries <= 0
    ):
        raise RemoteWorkerError("worker data/receipt capacity partition is invalid")
    directory = _safe_path(contract, "control/receipts", "receipt directory")
    details = _lstat_no_symlink(directory, "receipt directory")
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
    ):
        raise RemoteWorkerError("receipt directory identity is invalid")
    apparent = 0
    count = 0
    with os.scandir(str(directory)) as scanner:
        for entry in scanner:
            if _remove_recoverable_atomic_partial(
                contract,
                entry,
                mode=0o400,
                maximum=contract.max_receipt_bytes,
                label="command receipt",
            ):
                apparent += entry.stat(follow_symlinks=False).st_size
                continue
            item = entry.stat(follow_symlinks=False)
            count += 1
            if count > contract.max_receipt_entries:
                raise RemoteWorkerError("receipt namespace exceeds its entry bound")
            if (
                not stat.S_ISREG(item.st_mode)
                or item.st_dev != contract.expected_root_device
                or item.st_nlink != 1
                or (item.st_uid, item.st_gid) != (contract.expected_uid, contract.expected_gid)
                or stat.S_IMODE(item.st_mode) != 0o400
                or item.st_size > contract.max_receipt_bytes
            ):
                raise RemoteWorkerError("receipt namespace contains an invalid object")
            if not entry.name.endswith(".json"):
                raise RemoteWorkerError("receipt namespace contains an invalid filename")
            request_id = entry.name[: -len(".json")]
            _validate_identifier(request_id, "receipt filename")
            if validate_documents:
                payload = _read_receipt_bytes(Path(entry.path), contract.max_receipt_bytes)
                try:
                    document = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RemoteWorkerError("stored command receipt is invalid JSON") from error
                base_fields = {
                    "schema_version",
                    "schema_id",
                    "request_id",
                    "command_sha256",
                    "operation",
                    "status",
                    "host",
                    "host_identity_sha256",
                    "root",
                    "completed_unix_ms",
                    "result",
                    "receipt_sha256",
                }
                schema_id = document.get("schema_id") if isinstance(document, dict) else None
                expected_fields = (
                    base_fields | {"semantic_sha256"}
                    if schema_id == RECEIPT_SCHEMA
                    else base_fields
                )
                semantic_valid = schema_id == LEGACY_RECEIPT_SCHEMA or (
                    isinstance(document.get("semantic_sha256"), str)
                    and SHA256.fullmatch(document["semantic_sha256"]) is not None
                )
                if (
                    not isinstance(document, dict)
                    or set(document) != expected_fields
                    or canonical_json(document) != payload
                    or document.get("schema_version") != SCHEMA_VERSION
                    or schema_id not in {LEGACY_RECEIPT_SCHEMA, RECEIPT_SCHEMA}
                    or document.get("request_id") != request_id
                    or not semantic_valid
                    or document.get("host") != contract.expected_host
                    or document.get("host_identity_sha256") != contract.expected_identity_sha256
                    or document.get("root") != str(contract.root)
                    or document.get("status") not in {"ok", "error"}
                    or not isinstance(document.get("result"), dict)
                    or not isinstance(document.get("completed_unix_ms"), int)
                    or document.get("receipt_sha256")
                    != document_sha256(document, "receipt_sha256")
                ):
                    raise RemoteWorkerError("stored command receipt identity is invalid")
            apparent += item.st_size
            if apparent > contract.receipt_budget_bytes:
                raise RemoteWorkerError("receipt namespace exceeds its dedicated budget")
    return {
        "receipt_apparent_bytes": apparent,
        "receipt_entries": count,
        "receipt_budget_bytes": contract.receipt_budget_bytes,
        "max_receipt_bytes": contract.max_receipt_bytes,
        "max_receipt_entries": contract.max_receipt_entries,
    }


def _storage_capacity_state(contract: WorkerContract) -> Dict[str, Any]:
    apparent = _campaign_apparent_size(contract)
    receipts = _receipt_budget_state(contract, validate_documents=True)
    reservations = _active_receipt_reservations(contract)
    data_reservations = _active_data_reservations(contract)
    data_apparent = (
        apparent
        - receipts["receipt_apparent_bytes"]
        - reservations["bytes"]
        - data_reservations["apparent_bytes"]
    )
    usage = shutil.disk_usage(str(contract.root))
    if (
        data_apparent < 0
        or data_apparent > contract.max_data_bytes
        or apparent > contract.max_campaign_bytes
        or usage.free < contract.min_free_bytes
    ):
        raise RemoteWorkerError("campaign capacity partition or free-space floor is exceeded")
    return {
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "min_free_bytes": contract.min_free_bytes,
        "campaign_apparent_bytes": apparent,
        "max_campaign_bytes": contract.max_campaign_bytes,
        "campaign_data_apparent_bytes": data_apparent,
        "max_data_bytes": contract.max_data_bytes,
        "receipt_reservation_bytes": reservations["bytes"],
        "receipt_reservation_entries": reservations["entries"],
        "data_reservation_apparent_bytes": data_reservations["apparent_bytes"],
        "data_reservation_reserved_bytes": data_reservations["reserved_bytes"],
        "data_reservation_entries": data_reservations["entries"],
        **receipts,
    }


def _require_receipt_capacity(
    contract: WorkerContract, state: Mapping[str, Any]
) -> None:
    if (
        state.get("receipt_entries", contract.max_receipt_entries) + 1
        > contract.max_receipt_entries
        or state.get("receipt_apparent_bytes", contract.receipt_budget_bytes)
        + contract.max_receipt_bytes
        > contract.receipt_budget_bytes
        or state.get("free_bytes", 0) - contract.max_receipt_bytes
        < contract.min_free_bytes
    ):
        raise RemoteWorkerError("no durable command-receipt capacity remains")


def _require_mutation_headroom(
    contract: WorkerContract,
    state: Mapping[str, Any],
    *,
    incoming_data_bytes: int,
    data_reservation_credit: int = 0,
    extra_physical_bytes: int = 0,
) -> None:
    if (
        not isinstance(incoming_data_bytes, int)
        or isinstance(incoming_data_bytes, bool)
        or incoming_data_bytes < 0
        or not isinstance(data_reservation_credit, int)
        or isinstance(data_reservation_credit, bool)
        or data_reservation_credit < 0
        or not isinstance(extra_physical_bytes, int)
        or isinstance(extra_physical_bytes, bool)
        or extra_physical_bytes < 0
    ):
        raise RemoteWorkerError("mutation capacity request is invalid")
    reserved_data = state.get("data_reservation_reserved_bytes")
    receipt_reservations = state.get("receipt_reservation_entries")
    if (
        not isinstance(reserved_data, int)
        or isinstance(reserved_data, bool)
        or not isinstance(receipt_reservations, int)
        or isinstance(receipt_reservations, bool)
        or data_reservation_credit > reserved_data
    ):
        raise RemoteWorkerError("mutation capacity state is invalid")
    outstanding_data = reserved_data - data_reservation_credit
    outstanding_receipts = receipt_reservations * contract.max_receipt_bytes
    if (
        state["campaign_data_apparent_bytes"]
        + outstanding_data
        + incoming_data_bytes
        > contract.max_data_bytes
        or state["campaign_apparent_bytes"]
        + outstanding_data
        + incoming_data_bytes
        + outstanding_receipts
        + extra_physical_bytes
        > contract.max_campaign_bytes
        or state["free_bytes"]
        - outstanding_data
        - incoming_data_bytes
        - outstanding_receipts
        - extra_physical_bytes
        < contract.min_free_bytes
    ):
        raise RemoteWorkerError("mutation would consume reserved campaign capacity")


def _remove_verified_tree(
    path: Path,
    *,
    allow_contained_symlinks: bool = False,
    diagnostic_root: Optional[Path] = None,
) -> None:
    prefixes: Sequence[Path] = (path,) if allow_contained_symlinks else ()
    _apparent_size(
        path,
        allowed_symlink_prefixes=prefixes,
        diagnostic_root=diagnostic_root,
    )
    for current_root, directories, _files in os.walk(str(path), topdown=False, followlinks=False):
        for name in directories:
            child = Path(current_root) / name
            details = os.lstat(str(child))
            if stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode):
                os.chmod(str(child), 0o700, follow_symlinks=False)
        os.chmod(current_root, 0o700, follow_symlinks=False)
    shutil.rmtree(str(path))
    _fsync_directory(path.parent)


def _run_cleanup_tree_inventory(contract: WorkerContract, path: Path) -> Dict[str, Any]:
    try:
        root_relative = path.relative_to(contract.root).as_posix()
    except ValueError as error:
        raise RemoteWorkerError("run cleanup tree escapes the campaign root") from error
    root_details = _lstat_no_symlink(path, "run cleanup tree")
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or root_details.st_dev != contract.expected_root_device
    ):
        raise RemoteWorkerError("run cleanup tree is not a contained directory")
    _apparent_size(
        path,
        allowed_symlink_prefixes=(path,),
        diagnostic_root=contract.root,
    )
    entries: List[Dict[str, Any]] = []
    stack = [path]
    apparent_bytes = 0
    while stack:
        current = stack.pop()
        with os.scandir(str(current)) as scanner:
            children = sorted(scanner, key=lambda entry: entry.name)
        for entry in children:
            child = Path(entry.path)
            details = os.lstat(str(child))
            relative = child.relative_to(path).as_posix()
            common = {
                "relative": relative,
                "device": details.st_dev,
                "inode": details.st_ino,
                "mode": "%04o" % stat.S_IMODE(details.st_mode),
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
            }
            if stat.S_ISLNK(details.st_mode):
                try:
                    resolved = child.resolve(strict=True)
                    resolved.relative_to(path)
                except (FileNotFoundError, RuntimeError, ValueError) as error:
                    raise RemoteWorkerError(
                        "run cleanup symlink escapes its exact run tree"
                    ) from error
                if os.stat(str(resolved)).st_dev != contract.expected_root_device:
                    raise RemoteWorkerError("run cleanup symlink crosses the campaign device")
                value = {**common, "kind": "symlink", "target": os.readlink(str(child))}
            elif stat.S_ISDIR(details.st_mode):
                value = {**common, "kind": "directory"}
                stack.append(child)
            elif stat.S_ISREG(details.st_mode):
                value = {**common, "kind": "file"}
                apparent_bytes += details.st_size
            else:
                raise RemoteWorkerError("run cleanup tree contains a special file")
            entries.append(value)
            if len(entries) > MAX_RUN_CLEANUP_ENTRIES:
                raise RemoteWorkerError("run cleanup inventory exceeds its entry bound")
    entries.sort(key=lambda value: value["relative"])
    return {
        "relative": root_relative,
        "root_device": root_details.st_dev,
        "root_inode": root_details.st_ino,
        "root_mode": "%04o" % stat.S_IMODE(root_details.st_mode),
        "entry_count": len(entries),
        "apparent_bytes": apparent_bytes,
        "inventory_sha256": hashlib.sha256(canonical_json(entries)).hexdigest(),
    }


def _failed_run_tree_inventory(contract: WorkerContract, path: Path) -> Dict[str, Any]:
    """Inventory one failed-run tree without resolving any symlink target.

    Failed tools can leave interpreter/cache links outside their run directory.
    Following those links is forbidden, but unlinking the directory entry itself
    is safe after a complete lstat-based CAS inventory.
    """
    try:
        root_relative = path.relative_to(contract.root).as_posix()
    except ValueError as error:
        raise RemoteWorkerError("failed-run cleanup tree escapes the campaign root") from error
    root_details = _lstat_no_symlink(path, "failed-run cleanup tree")
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or root_details.st_dev != contract.expected_root_device
        or (root_details.st_uid, root_details.st_gid)
        != (contract.expected_uid, contract.expected_gid)
    ):
        raise RemoteWorkerError("failed-run cleanup tree is not an owned contained directory")
    entries: List[Dict[str, Any]] = []
    stack = [path]
    apparent_bytes = 0
    while stack:
        current = stack.pop()
        with os.scandir(str(current)) as scanner:
            children = sorted(scanner, key=lambda entry: entry.name)
        for entry in children:
            child = Path(entry.path)
            details = os.lstat(str(child))
            if details.st_dev != contract.expected_root_device or (
                details.st_uid,
                details.st_gid,
            ) != (contract.expected_uid, contract.expected_gid):
                raise RemoteWorkerError("failed-run cleanup entry owner/device is invalid")
            relative = child.relative_to(path).as_posix()
            common = {
                "relative": relative,
                "device": details.st_dev,
                "inode": details.st_ino,
                "mode": "%04o" % stat.S_IMODE(details.st_mode),
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
            }
            if stat.S_ISLNK(details.st_mode):
                value = {**common, "kind": "symlink", "target": os.readlink(str(child))}
                apparent_bytes += details.st_size
            elif stat.S_ISDIR(details.st_mode):
                value = {**common, "kind": "directory"}
                stack.append(child)
            elif stat.S_ISREG(details.st_mode):
                value = {**common, "kind": "file", "nlink": details.st_nlink}
                apparent_bytes += details.st_size
            else:
                raise RemoteWorkerError("failed-run cleanup tree contains a special file")
            entries.append(value)
            if len(entries) > MAX_RUN_CLEANUP_ENTRIES:
                raise RemoteWorkerError("failed-run cleanup inventory exceeds its entry bound")
    entries.sort(key=lambda value: value["relative"])
    return {
        "relative": root_relative,
        "root_device": root_details.st_dev,
        "root_inode": root_details.st_ino,
        "root_mode": "%04o" % stat.S_IMODE(root_details.st_mode),
        "entry_count": len(entries),
        "apparent_bytes": apparent_bytes,
        "inventory_sha256": hashlib.sha256(canonical_json(entries)).hexdigest(),
    }


def _failed_run_paths(contract: WorkerContract, run_id: Any) -> Tuple[str, Dict[str, Path]]:
    safe = _validate_identifier(run_id, "failed-run cleanup run_id")
    paths = {
        "tmp": _safe_path(contract, "tmp/run-%s" % safe, "failed-run tmp"),
        "build": _safe_path(contract, "build/run-%s" % safe, "failed-run build"),
        "cache": _safe_path(contract, "cache/runs/run-%s" % safe, "failed-run cache"),
    }
    return safe, paths


def _remove_failed_run_tree(path: Path) -> None:
    """Remove a verified run tree without ever traversing a symlink."""
    for current_root, directories, files in os.walk(str(path), topdown=False, followlinks=False):
        os.chmod(current_root, 0o700, follow_symlinks=False)
        for name in files:
            child = Path(current_root) / name
            details = os.lstat(str(child))
            if not (stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode)):
                raise RemoteWorkerError("failed-run cleanup file changed type during removal")
            os.unlink(str(child))
        for name in directories:
            child = Path(current_root) / name
            details = os.lstat(str(child))
            if stat.S_ISLNK(details.st_mode):
                os.unlink(str(child))
            elif stat.S_ISDIR(details.st_mode):
                os.chmod(str(child), 0o700, follow_symlinks=False)
                os.rmdir(str(child))
            else:
                raise RemoteWorkerError("failed-run cleanup directory changed type during removal")
    os.rmdir(str(path))
    _fsync_directory(path.parent)


def _run_cleanup_tombstone(path: Path, cleanup_sha256: str) -> Path:
    _validate_sha256(cleanup_sha256, "cleanup token SHA-256")
    return path.parent / (
        ".%s.%s.cleanup-tombstone" % (path.name, cleanup_sha256[:32])
    )


def _validate_cleanup_tombstone(
    contract: WorkerContract, path: Path, label: str
) -> None:
    _failed_run_tree_inventory(contract, path)


def _prepare_failed_run_cleanup(
    contract: WorkerContract, arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    run_id, paths = _failed_run_paths(contract, arguments.get("run_id"))
    trees: Dict[str, Any] = {}
    for label, path in paths.items():
        trees[label] = _failed_run_tree_inventory(contract, path) if path.exists() else None
    if not any(value is not None for value in trees.values()):
        raise RemoteWorkerError("failed-run cleanup found no exact run trees")
    issued_unix_ms = time.time_ns() // 1_000_000
    token: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_id": FAILED_RUN_CLEANUP_TOKEN_SCHEMA,
        "run_id": run_id,
        "root": str(contract.root),
        "host_identity_sha256": contract.expected_identity_sha256,
        "issued_unix_ms": issued_unix_ms,
        "expires_unix_ms": issued_unix_ms + RUN_CLEANUP_TOKEN_LIFETIME_MS,
        "trees": trees,
    }
    token["cleanup_token_sha256"] = document_sha256(token, "cleanup_token_sha256")
    return {"cleanup_token": token}


def _commit_failed_run_cleanup(
    contract: WorkerContract, arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    token = arguments.get("cleanup_token")
    if (
        not isinstance(token, dict)
        or token.get("schema_version") != SCHEMA_VERSION
        or token.get("schema_id") != FAILED_RUN_CLEANUP_TOKEN_SCHEMA
        or token.get("root") != str(contract.root)
        or token.get("host_identity_sha256") != contract.expected_identity_sha256
        or token.get("cleanup_token_sha256") != document_sha256(token, "cleanup_token_sha256")
    ):
        raise RemoteWorkerError("failed-run cleanup token identity is invalid")
    now_ms = time.time_ns() // 1_000_000
    if (
        not isinstance(token.get("issued_unix_ms"), int)
        or not isinstance(token.get("expires_unix_ms"), int)
        or token["issued_unix_ms"] > now_ms + REQUEST_MAX_FUTURE_MS
        or token["expires_unix_ms"] < now_ms
    ):
        raise RemoteWorkerError("failed-run cleanup token is outside its validity window")
    trees = token.get("trees")
    if not isinstance(trees, dict) or set(trees) != {"tmp", "build", "cache"}:
        raise RemoteWorkerError("failed-run cleanup token trees are invalid")
    lock_fd, _ = _global_lock(contract)
    removed_bytes = 0
    removed: Dict[str, bool] = {}
    try:
        run_id, paths = _failed_run_paths(contract, token.get("run_id"))
        existing: Dict[str, bool] = {}
        tombstones = {
            label: _run_cleanup_tombstone(path, token["cleanup_token_sha256"])
            for label, path in paths.items()
        }
        # Validate every tree before deleting any tree so a CAS failure cannot
        # produce a partially recovered run.
        for label, path in paths.items():
            expected = trees[label]
            exists = path.exists()
            existing[label] = exists
            tombstone = tombstones[label]
            if exists and tombstone.exists():
                raise RemoteWorkerError(
                    "failed-run cleanup source and tombstone both exist"
                )
            if expected is None:
                if exists or tombstone.exists():
                    raise RemoteWorkerError("failed-run cleanup tree appeared after prepare")
                continue
            if exists:
                current = _failed_run_tree_inventory(contract, path)
                if current != expected:
                    raise RemoteWorkerError("failed-run cleanup CAS inventory changed")
            elif tombstone.exists():
                _validate_cleanup_tombstone(
                    contract, tombstone, "failed-run cleanup tombstone"
                )
        for label, path in paths.items():
            expected = trees[label]
            exists = existing[label]
            if expected is not None and exists:
                removed_bytes += expected["apparent_bytes"]
                os.rename(str(path), str(tombstones[label]))
                _fsync_directory(path.parent)
                removed[label] = True
            else:
                removed[label] = False
        for _label, tombstone in tombstones.items():
            if tombstone.exists():
                _validate_cleanup_tombstone(
                    contract, tombstone, "failed-run cleanup tombstone"
                )
                _remove_failed_run_tree(tombstone)
    finally:
        os.close(lock_fd)
    return {
        "run_id": run_id,
        "cleanup_token_sha256": token["cleanup_token_sha256"],
        "removed": removed,
        "removed_bytes": removed_bytes,
        "all_absent": all(not path.exists() for path in paths.values()),
    }


def _run_cleanup_paths(
    contract: WorkerContract, run_id: Any, *, require_exists: bool = True
) -> Tuple[str, Path, Path]:
    safe = _validate_identifier(run_id, "run cleanup run_id")
    build = _safe_path(contract, "build/run-%s" % safe, "run cleanup build")
    cache = _safe_path(contract, "cache/runs/run-%s" % safe, "run cleanup cache")
    for label, path in (("build", build), ("cache", cache)):
        try:
            details = _lstat_no_symlink(path, "run cleanup %s" % label)
        except FileNotFoundError as error:
            if require_exists:
                raise RemoteWorkerError("run cleanup %s tree is missing" % label) from error
            continue
        if not stat.S_ISDIR(details.st_mode) or details.st_dev != contract.expected_root_device:
            raise RemoteWorkerError("run cleanup %s tree is invalid" % label)
    return safe, build, cache


def _validate_run_output_token(
    contract: WorkerContract,
    run_id: str,
    value: Any,
    *,
    label: str,
    suffix: str,
    max_bytes: int,
) -> Dict[str, Any]:
    path, token = _validate_object_token(contract, value)
    required_prefix = "build/run-%s/" % run_id
    if not token["relative"].startswith(required_prefix) or not token["relative"].endswith(suffix):
        raise RemoteWorkerError("%s is outside the exact run or has the wrong suffix" % label)
    if token["size"] <= 0 or token["size"] > max_bytes:
        raise RemoteWorkerError("%s exceeds its bounded size" % label)
    digest, size = file_sha256(path)
    if digest != token["sha256"] or size != token["size"]:
        raise RemoteWorkerError("%s content changed after token issuance" % label)
    return token


def _prepare_run_cleanup(contract: WorkerContract, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    run_id, build, cache = _run_cleanup_paths(contract, arguments.get("run_id"))
    manifest_token = _validate_run_output_token(
        contract,
        run_id,
        arguments.get("manifest_object_token"),
        label="ephemeral window manifest",
        suffix=".json",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    dataset_token = _validate_run_output_token(
        contract,
        run_id,
        arguments.get("dataset_object_token"),
        label="ephemeral window dataset",
        suffix=".r2map",
        max_bytes=MAX_UNKNOWN_STREAM_BYTES,
    )
    if manifest_token["relative"] == dataset_token["relative"]:
        raise RemoteWorkerError("ephemeral window outputs must be distinct")
    issued_unix_ms = time.time_ns() // 1_000_000
    token: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_id": RUN_CLEANUP_TOKEN_SCHEMA,
        "run_id": run_id,
        "root": str(contract.root),
        "host_identity_sha256": contract.expected_identity_sha256,
        "issued_unix_ms": issued_unix_ms,
        "expires_unix_ms": issued_unix_ms + RUN_CLEANUP_TOKEN_LIFETIME_MS,
        "outputs": {
            "manifest": manifest_token,
            "dataset": dataset_token,
        },
        "trees": {
            "build": _run_cleanup_tree_inventory(contract, build),
            "cache": _run_cleanup_tree_inventory(contract, cache),
        },
    }
    token["cleanup_token_sha256"] = document_sha256(token, "cleanup_token_sha256")
    return {"cleanup_token": token}


def _commit_run_cleanup(contract: WorkerContract, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    token = arguments.get("cleanup_token")
    if (
        not isinstance(token, dict)
        or token.get("schema_version") != SCHEMA_VERSION
        or token.get("schema_id") != RUN_CLEANUP_TOKEN_SCHEMA
        or token.get("root") != str(contract.root)
        or token.get("host_identity_sha256") != contract.expected_identity_sha256
        or token.get("cleanup_token_sha256") != document_sha256(token, "cleanup_token_sha256")
    ):
        raise RemoteWorkerError("run cleanup token identity is invalid")
    now_ms = time.time_ns() // 1_000_000
    if (
        not isinstance(token.get("issued_unix_ms"), int)
        or not isinstance(token.get("expires_unix_ms"), int)
        or token["issued_unix_ms"] > now_ms + REQUEST_MAX_FUTURE_MS
        or token["expires_unix_ms"] < now_ms
    ):
        raise RemoteWorkerError("run cleanup token is outside its validity window")
    lock_fd, _ = _global_lock(contract)
    try:
        run_id, build, cache = _run_cleanup_paths(
            contract, token.get("run_id"), require_exists=False
        )
        outputs = token.get("outputs")
        trees = token.get("trees")
        if (
            not isinstance(outputs, dict)
            or set(outputs) != {"manifest", "dataset"}
            or not isinstance(trees, dict)
            or set(trees) != {"build", "cache"}
        ):
            raise RemoteWorkerError("run cleanup token sections are invalid")
        build_exists = build.exists()
        cache_exists = cache.exists()
        tombstones = {
            "build": _run_cleanup_tombstone(
                build, token["cleanup_token_sha256"]
            ),
            "cache": _run_cleanup_tombstone(
                cache, token["cleanup_token_sha256"]
            ),
        }
        if (build_exists and tombstones["build"].exists()) or (
            cache_exists and tombstones["cache"].exists()
        ):
            raise RemoteWorkerError("run cleanup source and tombstone both exist")
        current_trees: Dict[str, Any] = {}
        if build_exists:
            _validate_run_output_token(
                contract,
                run_id,
                outputs.get("manifest"),
                label="ephemeral window manifest",
                suffix=".json",
                max_bytes=MAX_MANIFEST_BYTES,
            )
            _validate_run_output_token(
                contract,
                run_id,
                outputs.get("dataset"),
                label="ephemeral window dataset",
                suffix=".r2map",
                max_bytes=MAX_UNKNOWN_STREAM_BYTES,
            )
            current_trees["build"] = _run_cleanup_tree_inventory(contract, build)
        if cache_exists:
            current_trees["cache"] = _run_cleanup_tree_inventory(contract, cache)
        for _label, tombstone in tombstones.items():
            if tombstone.exists():
                _validate_cleanup_tombstone(
                    contract, tombstone, "run cleanup tombstone"
                )
        if any(current_trees[label] != trees[label] for label in current_trees):
            raise RemoteWorkerError("run cleanup CAS inventory changed after token issuance")
        removed_bytes = trees["build"]["apparent_bytes"] + trees["cache"]["apparent_bytes"]
        if build_exists:
            os.rename(str(build), str(tombstones["build"]))
            _fsync_directory(build.parent)
        if cache_exists:
            os.rename(str(cache), str(tombstones["cache"]))
            _fsync_directory(cache.parent)
        for tombstone in tombstones.values():
            if tombstone.exists():
                _validate_cleanup_tombstone(
                    contract, tombstone, "run cleanup tombstone"
                )
                _remove_failed_run_tree(tombstone)
    finally:
        os.close(lock_fd)
    return {
        "run_id": run_id,
        "cleanup_token_sha256": token["cleanup_token_sha256"],
        "manifest_object_token_sha256": outputs["manifest"]["token_sha256"],
        "dataset_object_token_sha256": outputs["dataset"]["token_sha256"],
        "removed_bytes": removed_bytes,
        "build_already_removed": not build_exists,
        "cache_already_removed": not cache_exists,
        "build_removed": not build.exists(),
        "cache_removed": not cache.exists(),
    }


def verify_root(
    contract: WorkerContract,
    *,
    measure_size: bool = True,
    allow_low_free_recovery: bool = False,
) -> Dict[str, Any]:
    if not contract.root.is_absolute() or contract.root != PRODUCTION_ROOT:
        raise RemoteWorkerError("worker root differs from the frozen john2 root")
    root_stat = _lstat_no_symlink(contract.root, "campaign root")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RemoteWorkerError("campaign root is not a directory")
    if stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise RemoteWorkerError("campaign root mode must be 0700")
    if (root_stat.st_uid, root_stat.st_gid) != (contract.expected_uid, contract.expected_gid):
        raise RemoteWorkerError("campaign root owner differs from john2")
    if (root_stat.st_dev, root_stat.st_ino) != (
        contract.expected_root_device,
        contract.expected_root_inode,
    ):
        raise RemoteWorkerError("campaign root device/inode identity drifted")
    for ancestor in (
        Path("/Users"),
        Path("/Users/john2"),
        Path("/Users/john2/cascadia-bench"),
        contract.root,
    ):
        details = _lstat_no_symlink(ancestor, "campaign ancestor")
        if details.st_dev != root_stat.st_dev:
            raise RemoteWorkerError("campaign ancestor crosses the Data volume")
    volume_info = _data_volume_info()
    identity = _host_identity(contract, volume_info)
    identity_sha256 = hashlib.sha256(canonical_json(identity)).hexdigest()
    if identity_sha256 != contract.expected_identity_sha256:
        raise RemoteWorkerError("canonical john2 host identity drifted")
    if (
        identity["hostname"] != contract.expected_host
        or identity["user"] != contract.expected_user
        or identity["filesystem"] != "apfs"
        or identity["protocol"] != "Apple Fabric"
        or volume_info.get("Internal") is not True
        or volume_info.get("Removable") is not False
        or volume_info.get("SolidState") is not True
    ):
        raise RemoteWorkerError("john2 host/internal SSD APFS identity is invalid")
    usage = shutil.disk_usage(str(contract.root))
    if usage.free < contract.min_free_bytes and not allow_low_free_recovery:
        raise RemoteWorkerError("john2 free-space reserve is below the campaign floor")
    capacity = _storage_capacity_state(contract) if measure_size else None
    return {
        "schema_id": CAPACITY_PROOF_SCHEMA,
        "schema_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "root": str(contract.root),
        "root_mode": "0700",
        "root_uid": root_stat.st_uid,
        "root_gid": root_stat.st_gid,
        "root_device": root_stat.st_dev,
        "root_inode": root_stat.st_ino,
        "host_identity_sha256": identity_sha256,
        "filesystem": identity["filesystem"],
        "protocol": identity["protocol"],
        "internal": True,
        "removable": False,
        "solid_state": True,
        "free_bytes": capacity["free_bytes"] if capacity is not None else usage.free,
        "total_bytes": capacity["total_bytes"] if capacity is not None else usage.total,
        "min_free_bytes": contract.min_free_bytes,
        "campaign_apparent_bytes": (
            capacity["campaign_apparent_bytes"] if capacity is not None else None
        ),
        "max_campaign_bytes": contract.max_campaign_bytes,
        "campaign_data_apparent_bytes": (
            capacity["campaign_data_apparent_bytes"] if capacity is not None else None
        ),
        "max_data_bytes": contract.max_data_bytes,
        "receipt_apparent_bytes": (
            capacity["receipt_apparent_bytes"] if capacity is not None else None
        ),
        "receipt_entries": capacity["receipt_entries"] if capacity is not None else None,
        "receipt_reservation_bytes": (
            capacity["receipt_reservation_bytes"] if capacity is not None else None
        ),
        "receipt_reservation_entries": (
            capacity["receipt_reservation_entries"] if capacity is not None else None
        ),
        "data_reservation_apparent_bytes": (
            capacity["data_reservation_apparent_bytes"] if capacity is not None else None
        ),
        "data_reservation_reserved_bytes": (
            capacity["data_reservation_reserved_bytes"] if capacity is not None else None
        ),
        "data_reservation_entries": (
            capacity["data_reservation_entries"] if capacity is not None else None
        ),
        "receipt_budget_bytes": contract.receipt_budget_bytes,
        "max_receipt_bytes": contract.max_receipt_bytes,
        "max_receipt_entries": contract.max_receipt_entries,
    }


def provision_layout(contract: WorkerContract) -> Dict[str, Any]:
    verify_root(contract, measure_size=False)
    root_stat = os.lstat(str(contract.root))
    created: List[str] = []
    hardened: List[str] = []
    for relative in LAYOUT_DIRECTORIES:
        path = _safe_path(contract, relative, "layout directory", create_parents=True)
        try:
            details = _lstat_no_symlink(path, "layout directory")
        except FileNotFoundError:
            os.mkdir(str(path), 0o700)
            _fsync_directory(path.parent)
            details = _lstat_no_symlink(path, "layout directory")
            created.append(relative)
        if not stat.S_ISDIR(details.st_mode) or details.st_dev != root_stat.st_dev:
            raise RemoteWorkerError("layout entry is not a contained directory")
        if details.st_uid != contract.expected_uid or details.st_gid != contract.expected_gid:
            raise RemoteWorkerError("layout entry has the wrong owner")
        if stat.S_IMODE(details.st_mode) != 0o700:
            os.chmod(str(path), 0o700, follow_symlinks=False)
            _fsync_directory(path.parent)
            hardened.append(relative)
    return {"created": created, "hardened": hardened, "layout": list(LAYOUT_DIRECTORIES)}


def _atomic_preflight_probe(contract: WorkerContract) -> Dict[str, Any]:
    control = _safe_path(contract, "control", "control directory")
    token = uuid.uuid4().hex
    source = control / (".preflight-%s.source" % token)
    target = control / (".preflight-%s.target" % token)
    payload = os.urandom(64)
    try:
        descriptor = os.open(
            str(source),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RemoteWorkerError("atomic probe write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(str(source), str(target))
        _fsync_directory(control)
        descriptor = os.open(str(target), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            observed = os.read(descriptor, len(payload) + 1)
        finally:
            os.close(descriptor)
        if observed != payload:
            raise RemoteWorkerError("atomic rename/fsync probe changed bytes")
    finally:
        for path in (source, target):
            with suppress(FileNotFoundError):
                os.unlink(str(path))
        _fsync_directory(control)
    return {
        "atomic_rename": True,
        "file_fsync": True,
        "directory_fsync": True,
        "probe_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _request_from_base64(encoded: str, worker_sha256: str) -> Dict[str, Any]:
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
    except Exception as error:
        raise RemoteWorkerError("request base64 is invalid") from error
    if len(raw) > MAX_REQUEST_BYTES:
        raise RemoteWorkerError("request exceeds the control-plane limit")
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("request JSON is invalid") from error
    if not isinstance(request, dict) or request.get("schema_id") != COMMAND_SCHEMA:
        raise RemoteWorkerError("request schema is invalid")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise RemoteWorkerError("request schema version is invalid")
    if request.get("root") != str(PRODUCTION_ROOT):
        raise RemoteWorkerError("request root differs from the frozen john2 root")
    if request.get("worker_sha256") != worker_sha256:
        raise RemoteWorkerError("request worker identity is invalid")
    _validate_identifier(request.get("request_id"), "request_id")
    issued = request.get("issued_unix_ms")
    if not isinstance(issued, int):
        raise RemoteWorkerError("request timestamp is invalid")
    now = time.time_ns() // 1_000_000
    if issued < now - REQUEST_MAX_AGE_MS or issued > now + REQUEST_MAX_FUTURE_MS:
        raise RemoteWorkerError("request timestamp is outside the replay window")
    if not isinstance(request.get("arguments"), dict):
        raise RemoteWorkerError("request arguments must be an object")
    expected_semantic = request_semantic_sha256(
        request.get("operation"), request.get("arguments")
    )
    if request.get("semantic_sha256") != expected_semantic:
        raise RemoteWorkerError("request semantic identity is invalid")
    expected_hash = request_command_sha256(request)
    if request.get("command_sha256") != expected_hash:
        raise RemoteWorkerError("request hash is invalid")
    return request


def _worker_source_sha256(expected_worker_sha256: str) -> str:
    digest_text = _validate_sha256(expected_worker_sha256, "worker source SHA-256")
    expected_path = (
        PRODUCTION_ROOT
        / "control/bin"
        / ("r2-map-remote-worker-%s.py" % digest_text)
    )
    path = Path(__file__)
    if not path.is_absolute() or path != expected_path:
        raise RemoteWorkerError("worker source path is not its content-addressed production path")
    details = _lstat_no_symlink(path, "worker source")
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o500
        or (details.st_uid, details.st_gid) != (PRODUCTION_UID, PRODUCTION_GID)
        or details.st_nlink != 1
        or not 1 <= details.st_size <= MAX_MANIFEST_BYTES
    ):
        raise RemoteWorkerError("worker source metadata differs from its bootstrap contract")
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            observed_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        observed_size != details.st_size
        or any(getattr(before, field) != getattr(after, field) for field in stable)
        or any(getattr(details, field) != getattr(after, field) for field in stable)
    ):
        raise RemoteWorkerError("worker source changed while it was authenticated")
    return digest.hexdigest()


def _receipt(
    request: Mapping[str, Any],
    identity_sha256: str,
    status: str,
    result: Mapping[str, Any],
    *,
    contract: WorkerContract = PRODUCTION_CONTRACT,
) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_id": RECEIPT_SCHEMA,
        "request_id": request["request_id"],
        "semantic_sha256": request["semantic_sha256"],
        "command_sha256": request["command_sha256"],
        "operation": request["operation"],
        "status": status,
        "host": contract.expected_host,
        "host_identity_sha256": identity_sha256,
        "root": str(contract.root),
        "completed_unix_ms": time.time_ns() // 1_000_000,
        "result": dict(result),
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    return receipt


def _read_receipt_bytes(path: Path, maximum: int) -> bytes:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) > maximum:
        raise RemoteWorkerError("command receipt exceeds its per-object bound")
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or any(getattr(before, field) != getattr(after, field) for field in stable)
    ):
        raise RemoteWorkerError("command receipt changed while it was read")
    return bytes(payload)


def _persist_receipt_locked(
    contract: WorkerContract, receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Persist one receipt while the caller owns the global worker lock."""

    relative = "control/receipts/%s.json" % receipt["request_id"]
    path = _safe_path(contract, relative, "command receipt", create_parents=True)
    payload = canonical_json(receipt)
    if len(payload) > contract.max_receipt_bytes:
        raise RemoteWorkerError("command receipt exceeds its per-object bound")
    pending_path = path.parent / (".%s.pending.tmp" % path.name)
    pending_bytes = (
        os.lstat(str(pending_path)).st_size if pending_path.exists() else 0
    )
    before = _receipt_budget_state(contract)
    reservations = _active_receipt_reservations(contract)
    own_reservation = _receipt_reservation_path(contract, receipt["request_id"])
    if own_reservation.exists():
        _load_receipt_reservation(contract, own_reservation)
        reservations["entries"] -= 1
        reservations["bytes"] -= own_reservation.lstat().st_size
    if path.exists():
        existing = _read_receipt_bytes(path, contract.max_receipt_bytes)
        if existing != payload:
            raise RemoteWorkerError("request identifier collides with a different receipt")
        return {**before, "relative": relative, "disposition": "present"}
    if (
        before["receipt_entries"] + 1 > contract.max_receipt_entries
        or before["receipt_entries"] + reservations["entries"] + 1
        > contract.max_receipt_entries
        or before["receipt_apparent_bytes"]
        + len(payload)
        + reservations["entries"] * contract.max_receipt_bytes
        > contract.receipt_budget_bytes
    ):
        raise RemoteWorkerError("command receipt exceeds its dedicated capacity")
    free_before = shutil.disk_usage(str(contract.root)).free
    if (
        free_before
        - len(payload)
        - reservations["entries"] * contract.max_receipt_bytes
        < contract.min_free_bytes
    ):
        raise RemoteWorkerError("command receipt would violate the free-space floor")
    _atomic_write_recoverable(path, payload, 0o400, "command receipt")
    after = _receipt_budget_state(contract)
    free_after = shutil.disk_usage(str(contract.root)).free
    if (
        after["receipt_entries"] != before["receipt_entries"] + 1
        or after["receipt_apparent_bytes"]
        != before["receipt_apparent_bytes"] - pending_bytes + len(payload)
        or _read_receipt_bytes(path, contract.max_receipt_bytes) != payload
        or free_after < contract.min_free_bytes
    ):
        raise RemoteWorkerError("command receipt postcommit accounting differs")
    return {
        **after,
        "relative": relative,
        "disposition": "installed",
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
    }


def _load_replay_receipt_locked(
    contract: WorkerContract, request: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    relative = "control/receipts/%s.json" % request["request_id"]
    path = _safe_path(contract, relative, "command receipt", create_parents=True)
    try:
        payload = _read_receipt_bytes(path, contract.max_receipt_bytes)
    except FileNotFoundError:
        return None
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("replay receipt is invalid JSON") from error
    if (
        not isinstance(receipt, dict)
        or canonical_json(receipt) != payload
        or receipt.get("schema_id") != RECEIPT_SCHEMA
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("request_id") != request["request_id"]
        or receipt.get("semantic_sha256") != request["semantic_sha256"]
        or receipt.get("command_sha256") != request["command_sha256"]
        or receipt.get("operation") != request["operation"]
        or receipt.get("status") != "ok"
        or receipt.get("host") != contract.expected_host
        or receipt.get("host_identity_sha256") != contract.expected_identity_sha256
        or receipt.get("root") != str(contract.root)
        or receipt.get("receipt_sha256") != document_sha256(receipt, "receipt_sha256")
        or not isinstance(receipt.get("result"), dict)
        or receipt["result"].get("payload_size") != 0
        or receipt["result"].get("payload_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise RemoteWorkerError("replay receipt does not bind the exact command")
    return receipt


def _query_receipt(contract: WorkerContract, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {"request_id", "semantic_sha256", "command_sha256", "operation"}
    if set(arguments) != expected:
        raise RemoteWorkerError("receipt query fields differ")
    request = {
        "request_id": _validate_identifier(arguments.get("request_id"), "receipt query"),
        "semantic_sha256": _validate_sha256(
            arguments.get("semantic_sha256"), "receipt query semantic SHA-256"
        ),
        "command_sha256": _validate_sha256(
            arguments.get("command_sha256"), "receipt query command SHA-256"
        ),
        "operation": arguments.get("operation"),
    }
    if request["operation"] not in STABLE_MUTATING_OPERATIONS:
        raise RemoteWorkerError("receipt query operation is not replayable")
    request_lock_fd, _ = _request_identity_lock(contract, request["request_id"])
    lock_fd, _ = _global_lock(contract)
    try:
        receipt = _load_replay_receipt_locked(contract, request)
        journal = _put_journal_path(contract, request["request_id"])
        receipt_reservation = _receipt_reservation_path(contract, request["request_id"])
        data_reservation = _data_reservation_path(contract, request["request_id"])
        if receipt_reservation.exists():
            _load_receipt_reservation(contract, receipt_reservation, request)
        if data_reservation.exists():
            _load_data_reservation(contract, data_reservation, request)
        if journal.exists():
            if request["operation"] not in {"put-file", "put-stream", "publish-status"}:
                raise RemoteWorkerError("non-put receipt query has a put journal")
            pending = _load_put_commit_context(contract, request)
            if receipt is None:
                recovered = _recovered_put_result(contract, pending)
                recovered["payload_size"] = 0
                recovered["payload_sha256"] = hashlib.sha256(b"").hexdigest()
                receipt = _receipt(
                    request,
                    contract.expected_identity_sha256,
                    "ok",
                    recovered,
                    contract=contract,
                )
                _persist_receipt_locked(contract, receipt)
            else:
                result = receipt["result"]
                document = pending["journal_document"]
                if (
                    result.get("relative") != document["relative"]
                    or result.get("sha256") != document["sha256"]
                    or result.get("size") != document["size"]
                    or result.get("mode") != oct(pending["new_mode"])
                ):
                    raise RemoteWorkerError("put receipt and pending journal differ")
            _release_receipt_reservation(receipt_reservation)
            _finalize_put_commit(pending)
        elif receipt is not None:
            _release_receipt_reservation(receipt_reservation)
            _release_data_reservation(data_reservation)
        result = {
            "found": receipt is not None,
            "receipt": receipt,
            "journal_present": journal.exists(),
            "receipt_reservation_present": receipt_reservation.exists(),
            "data_reservation_present": data_reservation.exists(),
        }
    finally:
        os.close(lock_fd)
        os.close(request_lock_fd)
    return result


def _drain_replay_request_body(
    request: Mapping[str, Any],
    source: BinaryIO,
    expected_result: Optional[Mapping[str, Any]] = None,
) -> None:
    operation = request["operation"]
    arguments = request["arguments"]
    if operation in {"put-file", "publish-status", "transaction-begin", "transaction-put"}:
        size = arguments.get("size")
        expected_sha256 = arguments.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise RemoteWorkerError("replayed request body size is invalid")
        _validate_sha256(expected_sha256, "replayed request body SHA-256")
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise RemoteWorkerError("replayed request body ended early")
            digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1) or digest.hexdigest() != expected_sha256:
            raise RemoteWorkerError("replayed request body identity differs")
        return
    if operation == "put-stream":
        maximum = arguments.get("max_bytes")
        if not isinstance(maximum, int) or not 1 <= maximum <= MAX_UNKNOWN_STREAM_BYTES:
            raise RemoteWorkerError("replayed stream bound is invalid")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = source.read(min(1 << 20, maximum + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise RemoteWorkerError("replayed stream exceeded its bound")
            digest.update(chunk)
        if (
            expected_result is None
            or expected_result.get("size") != size
            or expected_result.get("sha256") != digest.hexdigest()
        ):
            raise RemoteWorkerError("replayed stream identity differs")
        return
    if source.read(1):
        raise RemoteWorkerError("replayed request unexpectedly included a body")


def _persist_receipt(contract: WorkerContract, receipt: Mapping[str, Any]) -> Dict[str, Any]:
    lock_fd, _ = _global_lock(contract)
    try:
        return _persist_receipt_locked(contract, receipt)
    finally:
        os.close(lock_fd)


def encode_frame(header: Mapping[str, Any], payload: bytes, receipt: Mapping[str, Any]) -> bytes:
    header_bytes = canonical_json(header)
    receipt_bytes = canonical_json(receipt)
    return b"".join(
        (
            FRAME_PREFIX.pack(FRAME_MAGIC, len(header_bytes)),
            header_bytes,
            payload,
            FRAME_SUFFIX.pack(len(receipt_bytes)),
            receipt_bytes,
        )
    )


def _object_token(contract: WorkerContract, relative: str) -> Dict[str, Any]:
    path = _safe_path(contract, relative, "object")
    before = _lstat_no_symlink(path, "object")
    if not stat.S_ISREG(before.st_mode):
        raise RemoteWorkerError("object is not a regular file")
    digest, size = file_sha256(path)
    after = _lstat_no_symlink(path, "object")
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RemoteWorkerError("object changed while it was hashed")
    token: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_id": OBJECT_TOKEN_SCHEMA,
        "relative": relative,
        "sha256": digest,
        "size": size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "mode": stat.S_IMODE(after.st_mode),
    }
    token["token_sha256"] = document_sha256(token, "token_sha256")
    return token


def _validate_object_token(contract: WorkerContract, value: Any) -> Tuple[Path, Dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_id") != OBJECT_TOKEN_SCHEMA:
        raise RemoteWorkerError("object token schema is invalid")
    if value.get("token_sha256") != document_sha256(value, "token_sha256"):
        raise RemoteWorkerError("object token hash is invalid")
    relative = value.get("relative")
    path = _safe_path(contract, relative, "object")
    details = _lstat_no_symlink(path, "object")
    expected = {
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
        "mode": stat.S_IMODE(details.st_mode),
    }
    for field, observed in expected.items():
        if value.get(field) != observed:
            raise RemoteWorkerError("object identity changed after token issuance")
    _validate_sha256(value.get("sha256"), "object token SHA-256")
    return path, dict(value)


def _read_range(
    contract: WorkerContract, arguments: Mapping[str, Any]
) -> Tuple[bytes, Dict[str, Any]]:
    path, token = _validate_object_token(contract, arguments.get("object_token"))
    offset = arguments.get("offset")
    length = arguments.get("length")
    max_bytes = arguments.get("max_bytes")
    if (
        not isinstance(offset, int)
        or not isinstance(length, int)
        or not isinstance(max_bytes, int)
        or offset < 0
        or length < 0
        or max_bytes < 0
        or max_bytes > MAX_RANGE_BYTES
        or length > max_bytes
        or offset + length > token["size"]
    ):
        raise RemoteWorkerError("range is outside the bounded object window")
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        os.lseek(descriptor, offset, os.SEEK_SET)
        chunks: List[bytes] = []
        remaining = length
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise RemoteWorkerError("object ended inside the requested range")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(before, field) != getattr(after, field):
            raise RemoteWorkerError("object changed during range read")
    payload = b"".join(chunks)
    return payload, {
        "object_token_sha256": token["token_sha256"],
        "object_sha256": token["sha256"],
        "offset": offset,
        "length": length,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _global_lock(contract: WorkerContract) -> Tuple[int, Path]:
    path = _safe_path(
        contract, "control/locks/.worker-global.lock", "global lock", create_parents=True
    )
    if _OUTER_GLOBAL_LOCK_FD is not None:
        return os.dup(_OUTER_GLOBAL_LOCK_FD), path
    descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor, path


def _request_lock(
    contract: WorkerContract, request: Mapping[str, Any]
) -> Tuple[int, Path]:
    request_id = _validate_identifier(request.get("request_id"), "request lock")
    operation = request.get("operation")
    arguments = request.get("arguments")
    if not isinstance(arguments, dict):
        raise RemoteWorkerError("request lock arguments differ")
    if operation in {"put-file", "publish-status", "put-stream"}:
        resource = {"kind": "path", "value": arguments.get("relative")}
    elif operation and str(operation).startswith("transaction-"):
        resource = {"kind": "transaction", "value": arguments.get("transaction_id")}
    elif operation in {
        "run-command",
        "run-controller",
        "run-cleanup-prepare",
        "failed-run-cleanup-prepare",
    }:
        resource = {"kind": "run", "value": arguments.get("run_id")}
    elif operation in {"run-cleanup-commit", "failed-run-cleanup-commit"}:
        token = arguments.get("cleanup_token")
        resource = {
            "kind": "run",
            "value": token.get("run_id") if isinstance(token, dict) else request_id,
        }
    elif operation in {"lock-acquire", "lock-renew", "lock-release"}:
        resource = {"kind": "lease", "value": arguments.get("name")}
    else:
        resource = {"kind": "request", "value": request_id}
    offset = int.from_bytes(
        hashlib.sha256(canonical_json(resource)).digest()[:8], "big"
    ) % ((1 << 63) - 1)
    path = _safe_path(
        contract,
        "control/locks/resource-ranges.lock",
        "request lock",
        create_parents=True,
    )
    descriptor = os.open(
        str(path),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    details = os.fstat(descriptor)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RemoteWorkerError("resource range-lock metadata is invalid")
    fcntl.lockf(descriptor, fcntl.LOCK_EX, 1, offset, os.SEEK_SET)
    return descriptor, path


def _request_identity_lock(
    contract: WorkerContract, request_id: Any
) -> Tuple[int, Path]:
    safe = _validate_identifier(request_id, "request identity lock")
    offset = int.from_bytes(
        hashlib.sha256(safe.encode("ascii")).digest()[:8], "big"
    ) % ((1 << 63) - 1)
    path = _safe_path(
        contract,
        "control/locks/request-ranges.lock",
        "request identity lock",
        create_parents=True,
    )
    descriptor = os.open(
        str(path),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    details = os.fstat(descriptor)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RemoteWorkerError("request range-lock metadata is invalid")
    fcntl.lockf(descriptor, fcntl.LOCK_EX, 1, offset, os.SEEK_SET)
    return descriptor, path


def _receipt_reservation_path(contract: WorkerContract, request_id: Any) -> Path:
    safe = _validate_identifier(request_id, "receipt reservation")
    return _safe_path(
        contract,
        "control/receipt-reservations/%s.json" % safe,
        "receipt reservation",
        create_parents=True,
    )


def _data_reservation_path(contract: WorkerContract, request_id: Any) -> Path:
    safe = _validate_identifier(request_id, "data reservation")
    return _safe_path(
        contract,
        "control/data-reservations/%s.json" % safe,
        "data reservation",
        create_parents=True,
    )


def _load_receipt_reservation(
    contract: WorkerContract,
    path: Path,
    request: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    details = _lstat_no_symlink(path, "receipt reservation")
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_size > MAX_RECEIPT_BYTES
    ):
        raise RemoteWorkerError("receipt reservation metadata is invalid")
    payload = _read_receipt_bytes(path, MAX_RECEIPT_BYTES)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("receipt reservation is invalid JSON") from error
    expected_fields = {
        "schema_id",
        "schema_version",
        "request_id",
        "semantic_sha256",
        "command_sha256",
        "reserved_bytes",
        "reservation_sha256",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_fields
        or canonical_json(document) != payload
        or document.get("schema_id")
        != "cascadia.r2-map.remote-receipt-reservation.v1"
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("reserved_bytes") != contract.max_receipt_bytes
        or document.get("reservation_sha256")
        != document_sha256(document, "reservation_sha256")
        or (request is not None and document.get("request_id") != request["request_id"])
        or (
            request is not None
            and document.get("semantic_sha256") != request["semantic_sha256"]
        )
        or (
            request is not None
            and document.get("command_sha256") != request["command_sha256"]
        )
    ):
        raise RemoteWorkerError("receipt reservation identity is invalid")
    return document


def _active_receipt_reservations(contract: WorkerContract) -> Dict[str, int]:
    directory = _safe_path(
        contract,
        "control/receipt-reservations",
        "receipt reservation directory",
    )
    if not directory.exists():
        return {"entries": 0, "bytes": 0}
    count = 0
    apparent = 0
    with os.scandir(str(directory)) as scanner:
        for entry in scanner:
            if _remove_recoverable_atomic_partial(
                contract,
                entry,
                mode=0o600,
                maximum=MAX_RECEIPT_BYTES,
                label="receipt reservation",
            ):
                apparent += entry.stat(follow_symlinks=False).st_size
                continue
            if not entry.name.endswith(".json"):
                raise RemoteWorkerError("receipt reservation namespace is invalid")
            document = _load_receipt_reservation(contract, Path(entry.path))
            if entry.name != "%s.json" % document["request_id"]:
                raise RemoteWorkerError("receipt reservation filename differs")
            count += 1
            apparent += Path(entry.path).lstat().st_size
            if count > contract.max_receipt_entries:
                raise RemoteWorkerError("receipt reservations exceed their entry bound")
    return {"entries": count, "bytes": apparent}


def _reserve_receipt_capacity_locked(
    contract: WorkerContract, request: Mapping[str, Any]
) -> Path:
    path = _receipt_reservation_path(contract, request["request_id"])
    if path.exists():
        _load_receipt_reservation(contract, path, request)
        return path
    receipts = _receipt_budget_state(contract)
    reservations = _active_receipt_reservations(contract)
    pressure_recovery = _matching_run_recovery_under_pressure(contract, request)
    if (
        receipts["receipt_entries"] + reservations["entries"] + 1
        > contract.max_receipt_entries
        or receipts["receipt_apparent_bytes"]
        + (reservations["entries"] + 1) * contract.max_receipt_bytes
        > contract.receipt_budget_bytes
        or (
            not pressure_recovery
            and shutil.disk_usage(str(contract.root)).free
            - (reservations["entries"] + 1) * contract.max_receipt_bytes
            < contract.min_free_bytes
        )
    ):
        raise RemoteWorkerError("no durable command-receipt capacity remains")
    document: Dict[str, Any] = {
        "schema_id": "cascadia.r2-map.remote-receipt-reservation.v1",
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "semantic_sha256": request["semantic_sha256"],
        "command_sha256": request["command_sha256"],
        "reserved_bytes": contract.max_receipt_bytes,
    }
    document["reservation_sha256"] = document_sha256(document, "reservation_sha256")
    _atomic_write_recoverable(
        path, canonical_json(document), 0o600, "receipt reservation"
    )
    return path


def _release_receipt_reservation(path: Optional[Path]) -> None:
    if path is None:
        return
    with suppress(FileNotFoundError):
        os.unlink(str(path))
        _fsync_directory(path.parent)


def _expected_data_reservation_identity(
    request: Mapping[str, Any],
) -> Optional[Tuple[Any, Any]]:
    arguments = request.get("arguments")
    if arguments is None:
        # Receipt recovery deliberately carries only stable request identity.
        return None
    if not isinstance(arguments, dict):
        raise RemoteWorkerError("data reservation request arguments are invalid")
    operation = request.get("operation")
    if operation in {"put-file", "put-stream", "publish-status"}:
        return (
            arguments.get("relative"),
            arguments.get("size", arguments.get("max_bytes")),
        )
    if operation in {"transaction-put", "transaction-import"}:
        return arguments.get("relative"), arguments.get("size")
    if operation in {"run-command", "run-controller"}:
        return (
            arguments.get("output_relative"),
            arguments.get("_test_max_run_bytes", MAX_RUN_BYTES),
        )
    raise RemoteWorkerError("operation is not authorized to own a data reservation")


def _load_data_reservation(
    contract: WorkerContract,
    path: Path,
    request: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    expected_identity = (
        None if request is None else _expected_data_reservation_identity(request)
    )
    details = _lstat_no_symlink(path, "data reservation")
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_size > MAX_DATA_RESERVATION_BYTES
    ):
        raise RemoteWorkerError("data reservation metadata is invalid")
    payload = _read_receipt_bytes(path, MAX_DATA_RESERVATION_BYTES)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("data reservation is invalid JSON") from error
    expected_fields = {
        "schema_id",
        "schema_version",
        "request_id",
        "semantic_sha256",
        "command_sha256",
        "target_relative",
        "payload_bytes",
        "journal_bytes",
        "reserved_bytes",
        "reservation_sha256",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_fields
        or canonical_json(document) != payload
        or document.get("schema_id") != DATA_RESERVATION_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or not isinstance(document.get("payload_bytes"), int)
        or isinstance(document.get("payload_bytes"), bool)
        or document.get("payload_bytes", -1) < 0
        or document.get("journal_bytes") != MAX_PUT_JOURNAL_BYTES
        or document.get("reserved_bytes")
        != document.get("payload_bytes", -1) + MAX_PUT_JOURNAL_BYTES
        or document.get("reservation_sha256")
        != document_sha256(document, "reservation_sha256")
        or (request is not None and document.get("request_id") != request["request_id"])
        or (
            request is not None
            and document.get("semantic_sha256") != request["semantic_sha256"]
        )
        or (
            request is not None
            and document.get("command_sha256") != request["command_sha256"]
        )
        or (
            expected_identity is not None
            and document.get("target_relative") != expected_identity[0]
        )
        or (
            expected_identity is not None
            and document.get("payload_bytes") != expected_identity[1]
        )
    ):
        raise RemoteWorkerError("data reservation identity is invalid")
    _relative_parts(document.get("target_relative"), "data reservation target")
    return document


def _active_data_reservations(contract: WorkerContract) -> Dict[str, int]:
    directory = _safe_path(
        contract,
        "control/data-reservations",
        "data reservation directory",
    )
    if not directory.exists():
        return {"entries": 0, "apparent_bytes": 0, "reserved_bytes": 0}
    count = 0
    apparent = 0
    reserved = 0
    with os.scandir(str(directory)) as scanner:
        for entry in scanner:
            if _remove_recoverable_atomic_partial(
                contract,
                entry,
                mode=0o600,
                maximum=MAX_DATA_RESERVATION_BYTES,
                label="data reservation",
            ):
                apparent += entry.stat(follow_symlinks=False).st_size
                continue
            if not entry.name.endswith(".json"):
                raise RemoteWorkerError("data reservation namespace is invalid")
            document = _load_data_reservation(contract, Path(entry.path))
            if entry.name != "%s.json" % document["request_id"]:
                raise RemoteWorkerError("data reservation filename differs")
            count += 1
            apparent += Path(entry.path).lstat().st_size
            reserved += document["reserved_bytes"]
            if count > contract.max_receipt_entries:
                raise RemoteWorkerError("data reservations exceed their entry bound")
    return {"entries": count, "apparent_bytes": apparent, "reserved_bytes": reserved}


def _reserve_data_capacity_locked(
    contract: WorkerContract,
    request: Mapping[str, Any],
    storage: Mapping[str, Any],
    *,
    reserved_payload_bytes: Optional[int] = None,
    target_relative: Optional[str] = None,
) -> Path:
    path = _data_reservation_path(contract, request["request_id"])
    if path.exists():
        _load_data_reservation(contract, path, request)
        return path
    arguments = request["arguments"]
    size = (
        reserved_payload_bytes
        if reserved_payload_bytes is not None
        else arguments.get("size", arguments.get("max_bytes"))
    )
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise RemoteWorkerError("data reservation payload size is invalid")
    reserved_bytes = size + MAX_PUT_JOURNAL_BYTES
    _require_mutation_headroom(
        contract,
        storage,
        incoming_data_bytes=reserved_bytes,
        extra_physical_bytes=MAX_DATA_RESERVATION_BYTES,
    )
    document: Dict[str, Any] = {
        "schema_id": DATA_RESERVATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "semantic_sha256": request["semantic_sha256"],
        "command_sha256": request["command_sha256"],
        "target_relative": (
            target_relative if target_relative is not None else arguments.get("relative")
        ),
        "payload_bytes": size,
        "journal_bytes": MAX_PUT_JOURNAL_BYTES,
        "reserved_bytes": reserved_bytes,
    }
    document["reservation_sha256"] = document_sha256(document, "reservation_sha256")
    payload = canonical_json(document)
    if len(payload) > MAX_DATA_RESERVATION_BYTES:
        raise RemoteWorkerError("data reservation exceeds its bounded size")
    _atomic_write_recoverable(path, payload, 0o600, "data reservation")
    _load_data_reservation(contract, path, request)
    return path


def _release_data_reservation(path: Optional[Path]) -> None:
    if path is None:
        return
    with suppress(FileNotFoundError):
        os.unlink(str(path))
        _fsync_directory(path.parent)


def _release_matching_data_reservation(
    contract: WorkerContract,
    request: Mapping[str, Any],
    path: Optional[Path] = None,
) -> None:
    """Release only the reservation cryptographically bound to this request."""
    candidate = path or _data_reservation_path(contract, request["request_id"])
    try:
        _load_data_reservation(contract, candidate, request)
    except FileNotFoundError:
        return
    _release_data_reservation(candidate)


def _remove_request_staging_file(path: Optional[Path]) -> None:
    """Unlink one exact request staging entry without following its contents."""
    if path is None:
        return
    try:
        details = os.lstat(str(path))
    except FileNotFoundError:
        return
    if stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode):
        raise RemoteWorkerError("request staging path unexpectedly became a directory")
    os.unlink(str(path))
    _fsync_directory(path.parent)


def _current_file_identity(path: Path) -> Optional[str]:
    try:
        details = _lstat_no_symlink(path, "CAS target")
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode):
        raise RemoteWorkerError("CAS target is not a regular file")
    digest, _ = file_sha256(path)
    return digest


def _put_journal_path(contract: WorkerContract, request_id: Any) -> Path:
    safe = _validate_identifier(request_id, "put journal request")
    return _safe_path(
        contract,
        "control/transactions/put-%s.journal.json" % safe,
        "put journal",
        create_parents=True,
    )


def _put_commit_context(
    contract: WorkerContract,
    request: Mapping[str, Any],
    target: Path,
    staging: Path,
    *,
    expected_current: str,
    current_size: int,
    expected_sha256: str,
    size: int,
    mode: int,
    storage_precommit: Mapping[str, Any],
    storage_staged: Mapping[str, Any],
    receipt_reservation_apparent_bytes: int,
    data_reservation_apparent_bytes: int,
) -> Dict[str, Any]:
    backup = target.parent / (".%s.%s.backup" % (target.name, request["request_id"]))
    journal_path = _put_journal_path(contract, request["request_id"])
    for path, label in ((backup, "put backup"), (journal_path, "put journal")):
        if path.exists() or path.is_symlink():
            raise RemoteWorkerError("%s already exists" % label)
    document: Dict[str, Any] = {
        "schema_id": PUT_JOURNAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "semantic_sha256": request["semantic_sha256"],
        "command_sha256": request["command_sha256"],
        "relative": target.relative_to(contract.root).as_posix(),
        "staging_relative": staging.relative_to(contract.root).as_posix(),
        "backup_relative": backup.relative_to(contract.root).as_posix(),
        "expected_current": expected_current,
        "previous_size": current_size,
        "sha256": expected_sha256,
        "size": size,
        "mode": "%04o" % mode,
        "storage_precommit": dict(storage_precommit),
        "storage_staged": dict(storage_staged),
        "receipt_reservation_apparent_bytes": receipt_reservation_apparent_bytes,
        "data_reservation_apparent_bytes": data_reservation_apparent_bytes,
    }
    document["journal_sha256"] = document_sha256(document, "journal_sha256")
    journal_payload = canonical_json(document)
    if len(journal_payload) > MAX_PUT_JOURNAL_BYTES:
        raise RemoteWorkerError("put journal exceeds its bounded size")
    _atomic_write(journal_path, journal_payload, 0o600)
    return {
        "journal": journal_path,
        "journal_document": document,
        "target": target,
        "staging": staging,
        "backup": backup,
        "new_sha256": expected_sha256,
        "new_size": size,
        "new_mode": mode,
        "expected_current": expected_current,
    }


def _rollback_put_commit(context: Mapping[str, Any]) -> None:
    target = context["target"]
    staging = context["staging"]
    backup = context["backup"]
    journal = context["journal"]
    if target.exists():
        current = _current_file_identity(target)
        if current == context["new_sha256"]:
            os.unlink(str(target))
            _fsync_directory(target.parent)
        elif context["expected_current"] != "absent" and current == context["expected_current"]:
            pass
        else:
            raise RemoteWorkerError("put rollback refuses to remove a changed target")
    if backup.exists():
        if target.exists():
            if _current_file_identity(target) == context["expected_current"]:
                raise RemoteWorkerError("put rollback found both original target and backup")
            raise RemoteWorkerError("put rollback target reappeared before backup restore")
        os.rename(str(backup), str(target))
        _fsync_directory(target.parent)
    with suppress(FileNotFoundError):
        os.unlink(str(staging))
        _fsync_directory(staging.parent)
    with suppress(FileNotFoundError):
        os.unlink(str(journal))
        _fsync_directory(journal.parent)
    _release_data_reservation(context.get("data_reservation"))


def _finalize_put_commit(context: Mapping[str, Any]) -> None:
    target = context["target"]
    details = _lstat_no_symlink(target, "committed put target")
    digest, size = file_sha256(target)
    if (
        not stat.S_ISREG(details.st_mode)
        or digest != context["new_sha256"]
        or size != context["new_size"]
        or stat.S_IMODE(details.st_mode) != context["new_mode"]
    ):
        raise RemoteWorkerError("committed put target changed before finalization")
    for path in (context["staging"], context["backup"]):
        with suppress(FileNotFoundError):
            os.unlink(str(path))
            _fsync_directory(path.parent)
    journal = context["journal"]
    os.unlink(str(journal))
    _fsync_directory(journal.parent)
    _release_data_reservation(context.get("data_reservation"))


def _load_put_commit_context(
    contract: WorkerContract, request: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    journal = _put_journal_path(contract, request["request_id"])
    try:
        details = _lstat_no_symlink(journal, "put journal")
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or details.st_size > MAX_PUT_JOURNAL_BYTES
    ):
        raise RemoteWorkerError("put journal metadata is invalid")
    payload = _read_receipt_bytes(journal, MAX_PUT_JOURNAL_BYTES)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("put journal is invalid JSON") from error
    expected_fields = {
        "schema_id",
        "schema_version",
        "request_id",
        "semantic_sha256",
        "command_sha256",
        "relative",
        "staging_relative",
        "backup_relative",
        "expected_current",
        "previous_size",
        "sha256",
        "size",
        "mode",
        "storage_precommit",
        "storage_staged",
        "receipt_reservation_apparent_bytes",
        "data_reservation_apparent_bytes",
        "journal_sha256",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_fields
        or canonical_json(document) != payload
        or document.get("schema_id") != PUT_JOURNAL_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("request_id") != request["request_id"]
        or document.get("semantic_sha256") != request["semantic_sha256"]
        or document.get("command_sha256") != request["command_sha256"]
        or document.get("journal_sha256") != document_sha256(document, "journal_sha256")
        or document.get("mode") not in {"0400", "0600"}
        or not isinstance(document.get("size"), int)
        or not isinstance(document.get("previous_size"), int)
        or not isinstance(document.get("receipt_reservation_apparent_bytes"), int)
        or not isinstance(document.get("data_reservation_apparent_bytes"), int)
        or document.get("receipt_reservation_apparent_bytes", -1) < 0
        or document.get("data_reservation_apparent_bytes", -1) < 0
    ):
        raise RemoteWorkerError("put journal does not bind the exact request")
    target = _safe_path(contract, document["relative"], "put journal target")
    staging = _safe_path(contract, document["staging_relative"], "put journal staging")
    backup = _safe_path(contract, document["backup_relative"], "put journal backup")
    data_reservation = _data_reservation_path(contract, request["request_id"])
    if staging.parent != target.parent or backup.parent != target.parent:
        raise RemoteWorkerError("put journal staging paths differ from the target parent")
    _load_data_reservation(contract, data_reservation, request)
    return {
        "journal": journal,
        "journal_document": document,
        "target": target,
        "staging": staging,
        "backup": backup,
        "data_reservation": data_reservation,
        "new_sha256": document["sha256"],
        "new_size": document["size"],
        "new_mode": int(document["mode"], 8),
        "expected_current": document["expected_current"],
    }


def _cleanup_put_journal_partials(
    contract: WorkerContract, request: Mapping[str, Any]
) -> int:
    directory = _safe_path(contract, "control/transactions", "transaction directory")
    prefix = ".put-%s.journal.json." % request["request_id"]
    removed = 0
    with os.scandir(str(directory)) as scanner:
        for entry in scanner:
            if not (entry.name.startswith(prefix) and entry.name.endswith(".tmp")):
                continue
            details = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or details.st_dev != contract.expected_root_device
                or details.st_size > MAX_PUT_JOURNAL_BYTES
            ):
                raise RemoteWorkerError("partial put journal metadata is invalid")
            os.unlink(entry.path)
            removed += 1
    if removed:
        _fsync_directory(directory)
    return removed


def _recovered_put_result(
    contract: WorkerContract,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    document = context["journal_document"]
    target = context["target"]
    details = _lstat_no_symlink(target, "recovered put target")
    digest, size = file_sha256(target)
    if (
        not stat.S_ISREG(details.st_mode)
        or digest != context["new_sha256"]
        or size != context["new_size"]
        or stat.S_IMODE(details.st_mode) != context["new_mode"]
        or context["staging"].exists()
    ):
        raise RemoteWorkerError("recovered put target does not match its durable intent")
    storage_precommit = document["storage_precommit"]
    storage_staged = document["storage_staged"]
    storage_transaction = verify_root(contract, measure_size=True)
    _require_receipt_capacity(contract, storage_transaction)
    journal_bytes = _lstat_no_symlink(context["journal"], "put journal").st_size
    backup_bytes = (
        _lstat_no_symlink(context["backup"], "put backup").st_size
        if context["backup"].exists()
        else 0
    )
    projected_data = storage_staged["campaign_data_apparent_bytes"] - document["previous_size"]
    if (
        storage_transaction["campaign_data_apparent_bytes"]
        < storage_staged["campaign_data_apparent_bytes"] + journal_bytes
    ):
        raise RemoteWorkerError("recovered put transaction accounting differs")
    return {
        "schema_id": PUT_RESULT_SCHEMA,
        "schema_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "relative": document["relative"],
        "sha256": document["sha256"],
        "size": document["size"],
        "mode": oct(context["new_mode"]),
        "previous_sha256": (
            None if document["expected_current"] == "absent" else document["expected_current"]
        ),
        "projected_campaign_bytes": (
            storage_staged["campaign_apparent_bytes"]
            - document["previous_size"]
            - document["receipt_reservation_apparent_bytes"]
            - document["data_reservation_apparent_bytes"]
            + contract.max_receipt_bytes
        ),
        "projected_data_bytes": projected_data,
        "projected_free_bytes": (
            storage_staged["free_bytes"]
            + document["previous_size"]
            + document["receipt_reservation_apparent_bytes"]
            + document["data_reservation_apparent_bytes"]
            - contract.max_receipt_bytes
        ),
        "receipt_capacity_reserved_bytes": contract.max_receipt_bytes,
        "journal_bytes": journal_bytes,
        "backup_bytes": backup_bytes,
        "transaction_overhead_bytes": journal_bytes,
        "receipt_reservation_apparent_bytes": document[
            "receipt_reservation_apparent_bytes"
        ],
        "data_reservation_apparent_bytes": document[
            "data_reservation_apparent_bytes"
        ],
        "storage_precommit": storage_precommit,
        "storage_staged": storage_staged,
        "storage_transaction": storage_transaction,
    }


def _put_file(
    contract: WorkerContract,
    arguments: Mapping[str, Any],
    source: BinaryIO,
    *,
    status_only: bool = False,
    lock_held: bool = False,
    request: Optional[Mapping[str, Any]] = None,
    commit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    relative = arguments.get("relative")
    parts = _relative_parts(relative, "upload relative path")
    if parts[0] not in MUTABLE_FILE_TOP_LEVELS:
        raise RemoteWorkerError("upload target top-level is not authorized")
    if parts[0] == "control" and (
        (
            len(parts) > 1
            and parts[1]
            in {
                "bin",
                "locks",
                "receipts",
                "receipt-reservations",
                "data-reservations",
                "transactions",
                "run-states",
                "run-supervisors",
            }
        )
        or parts[1:] == ("dashboard-status.json",)
    ) and not (status_only and parts[1:] == ("dashboard-status.json",)):
        raise RemoteWorkerError("upload target is in a worker-reserved control namespace")
    if status_only and relative != "control/dashboard-status.json":
        raise RemoteWorkerError("status publication target is not canonical")
    size = arguments.get("size")
    if not isinstance(size, int) or size < 0:
        raise RemoteWorkerError("upload size is invalid")
    if status_only and size > MAX_STATUS_BYTES:
        raise RemoteWorkerError("status projection exceeds 64 KiB")
    expected_sha256 = _validate_sha256(arguments.get("sha256"), "upload SHA-256")
    expected_current = arguments.get("expected_current")
    if expected_current != "absent":
        _validate_sha256(expected_current, "expected current SHA-256")
    mutable = arguments.get("mutable") is True
    mode = 0o600 if mutable else 0o400
    path = _safe_path(contract, relative, "upload target", create_parents=True)
    if status_only:
        payload = source.read(size + 1)
        if len(payload) != size or source.read(1):
            raise RemoteWorkerError("status upload size differs from declaration")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise RemoteWorkerError("status upload SHA-256 mismatch")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteWorkerError("status projection is not valid JSON") from error
        if not isinstance(decoded, dict):
            raise RemoteWorkerError("status projection must be a JSON object")
        source = _BytesReader(payload)
    if request is not None and lock_held:
        raise RemoteWorkerError("streaming upload cannot inherit the global worker lock")
    context: Optional[Dict[str, Any]] = None
    data_reservation: Optional[Path] = None
    data_reservation_attempted = False
    staging: Optional[Path] = None
    lock_fd: Optional[int] = None
    try:
        if not lock_held:
            lock_fd, _ = _global_lock(contract)
        try:
            current = _current_file_identity(path)
            if (expected_current == "absent" and current is not None) or (
                expected_current != "absent" and current != expected_current
            ):
                raise RemoteWorkerError("atomic upload compare-and-swap precondition failed")
            storage = verify_root(contract, measure_size=True)
            _require_receipt_capacity(contract, storage)
            current_size = (
                0 if current is None else _lstat_no_symlink(path, "CAS target").st_size
            )
            if request is None:
                projected_data = storage["campaign_data_apparent_bytes"] - current_size + size
                if (
                    projected_data > contract.max_data_bytes
                    or storage["free_bytes"] - size - contract.max_receipt_bytes
                    < contract.min_free_bytes
                ):
                    raise RemoteWorkerError("atomic upload exceeds the campaign capacity reserve")
                _atomic_stream(path, source, size, expected_sha256, mode)
                storage_staged = storage
                storage_after = verify_root(contract, measure_size=True)
                if storage_after["campaign_data_apparent_bytes"] != projected_data:
                    raise RemoteWorkerError("atomic upload postcommit capacity proof failed")
            else:
                data_reservation_attempted = True
                data_reservation = _reserve_data_capacity_locked(contract, request, storage)
                staging = path.parent / (
                    ".%s.%s.put-staging" % (path.name, request["request_id"])
                )
                if staging.exists() or staging.is_symlink():
                    details = _lstat_no_symlink(staging, "stale put staging")
                    if (
                        not stat.S_ISREG(details.st_mode)
                        or details.st_nlink != 1
                        or details.st_dev != contract.expected_root_device
                    ):
                        raise RemoteWorkerError("stale put staging metadata is invalid")
                    os.unlink(str(staging))
                    _fsync_directory(staging.parent)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
    except BaseException:
        if request is not None and data_reservation_attempted:
            cleanup_fd, _ = _global_lock(contract)
            try:
                _remove_request_staging_file(staging)
                _release_matching_data_reservation(contract, request, data_reservation)
            finally:
                os.close(cleanup_fd)
        raise

    if request is not None:
        assert staging is not None
        try:
            staging = _stage_stream(
                path,
                source,
                size,
                expected_sha256,
                mode,
                temporary=staging,
            )
            lock_fd, _ = _global_lock(contract)
            try:
                commit_current = _current_file_identity(path)
                if commit_current != current:
                    raise RemoteWorkerError("atomic upload CAS changed while stdin was in flight")
                storage_staged = verify_root(contract, measure_size=True)
                own_reservation = _load_data_reservation(contract, data_reservation, request)
                other_reserved = (
                    storage_staged["data_reservation_reserved_bytes"]
                    - own_reservation["reserved_bytes"]
                )
                if (
                    other_reserved < 0
                    or storage_staged["campaign_data_apparent_bytes"]
                    + other_reserved
                    + MAX_PUT_JOURNAL_BYTES
                    > contract.max_data_bytes
                    or storage_staged["free_bytes"]
                    - other_reserved
                    - MAX_PUT_JOURNAL_BYTES
                    - storage_staged["receipt_reservation_entries"]
                    * contract.max_receipt_bytes
                    < contract.min_free_bytes
                ):
                    raise RemoteWorkerError(
                        "staged upload exceeds its durable capacity reservation"
                    )
                receipt_reservation = _receipt_reservation_path(
                    contract, request["request_id"]
                )
                receipt_reservation_bytes = (
                    _lstat_no_symlink(receipt_reservation, "receipt reservation").st_size
                    if receipt_reservation.exists()
                    else 0
                )
                data_reservation_bytes = _lstat_no_symlink(
                    data_reservation, "data reservation"
                ).st_size
                context = _put_commit_context(
                    contract,
                    request,
                    path,
                    staging,
                    expected_current=expected_current,
                    current_size=current_size,
                    expected_sha256=expected_sha256,
                    size=size,
                    mode=mode,
                    storage_precommit=storage,
                    storage_staged=storage_staged,
                    receipt_reservation_apparent_bytes=receipt_reservation_bytes,
                    data_reservation_apparent_bytes=data_reservation_bytes,
                )
                context["data_reservation"] = data_reservation
                if current is not None:
                    os.rename(str(path), str(context["backup"]))
                    _fsync_directory(path.parent)
                os.rename(str(staging), str(path))
                _fsync_directory(path.parent)
                storage_after = verify_root(contract, measure_size=True)
                journal_size = _lstat_no_symlink(context["journal"], "put journal").st_size
                if (
                    storage_after["campaign_apparent_bytes"]
                    != storage_staged["campaign_apparent_bytes"] + journal_size
                    or storage_after["campaign_data_apparent_bytes"]
                    != storage_staged["campaign_data_apparent_bytes"] + journal_size
                    or storage_after["free_bytes"] < contract.min_free_bytes
                ):
                    raise RemoteWorkerError("atomic upload postcommit capacity proof failed")
            finally:
                os.close(lock_fd)
        except BaseException:
            rollback_fd, _ = _global_lock(contract)
            try:
                if context is not None:
                    _rollback_put_commit(context)
                elif staging is not None:
                    with suppress(FileNotFoundError):
                        os.unlink(str(staging))
                        _fsync_directory(staging.parent)
                _release_matching_data_reservation(contract, request, data_reservation)
                data_reservation = None
            finally:
                os.close(rollback_fd)
            raise
        projected_data = storage_staged["campaign_data_apparent_bytes"] - current_size
        if commit_context is not None and context is not None:
            commit_context.update(context)
    journal_bytes = (
        0 if context is None else _lstat_no_symlink(context["journal"], "put journal").st_size
    )
    backup_bytes = 0 if context is None or current is None else current_size
    receipt_reservation_bytes = (
        0
        if context is None
        else context["journal_document"]["receipt_reservation_apparent_bytes"]
    )
    data_reservation_bytes = (
        0
        if context is None
        else context["journal_document"]["data_reservation_apparent_bytes"]
    )
    return {
        "schema_id": PUT_RESULT_SCHEMA,
        "schema_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "relative": relative,
        "sha256": expected_sha256,
        "size": size,
        "mode": oct(mode),
        "previous_sha256": current,
        "projected_campaign_bytes": (
            storage_staged["campaign_apparent_bytes"]
            - current_size
            - receipt_reservation_bytes
            - data_reservation_bytes
            + contract.max_receipt_bytes
        ),
        "projected_data_bytes": projected_data,
        "projected_free_bytes": (
            storage_staged["free_bytes"]
            + current_size
            + receipt_reservation_bytes
            + data_reservation_bytes
            - contract.max_receipt_bytes
        ),
        "receipt_capacity_reserved_bytes": contract.max_receipt_bytes,
        "receipt_reservation_apparent_bytes": receipt_reservation_bytes,
        "data_reservation_apparent_bytes": data_reservation_bytes,
        "journal_bytes": journal_bytes,
        "backup_bytes": backup_bytes,
        "transaction_overhead_bytes": journal_bytes,
        "storage_precommit": storage,
        "storage_staged": storage_staged,
        "storage_transaction": storage_after,
    }


def _put_unknown_stream(
    contract: WorkerContract,
    arguments: Mapping[str, Any],
    source: BinaryIO,
    *,
    request: Optional[Mapping[str, Any]] = None,
    commit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    relative = arguments.get("relative")
    parts = _relative_parts(relative, "stream relative path")
    if parts[0] not in {"logs", "reports", "control"}:
        raise RemoteWorkerError("unknown-size stream target top-level is unauthorized")
    if parts[0] == "control" and (
        len(parts) < 2
        or parts[1]
        in {
            "bin",
            "locks",
            "receipts",
            "receipt-reservations",
            "data-reservations",
            "transactions",
            "run-states",
            "run-supervisors",
        }
        or parts[1:] == ("dashboard-status.json",)
    ):
        raise RemoteWorkerError("stream target is in a worker-reserved control namespace")
    max_bytes = arguments.get("max_bytes")
    if not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > MAX_UNKNOWN_STREAM_BYTES:
        raise RemoteWorkerError("unknown-size stream bound is invalid")
    expected_current = arguments.get("expected_current", "absent")
    if expected_current != "absent":
        _validate_sha256(expected_current, "expected current SHA-256")
    if request is None:
        raise RemoteWorkerError("unknown-size stream requires a durable request identity")
    path = _safe_path(contract, relative, "stream target", create_parents=True)
    data_reservation: Optional[Path] = None
    data_reservation_attempted = False
    temporary: Optional[Path] = None
    try:
        lock_fd, _ = _global_lock(contract)
        try:
            initial = _current_file_identity(path)
            if (expected_current == "absent" and initial is not None) or (
                expected_current != "absent" and initial != expected_current
            ):
                raise RemoteWorkerError("stream compare-and-swap precondition failed")
            current_size = (
                0 if initial is None else _lstat_no_symlink(path, "stream target").st_size
            )
            storage = verify_root(contract, measure_size=True)
            _require_receipt_capacity(contract, storage)
            data_reservation_attempted = True
            data_reservation = _reserve_data_capacity_locked(contract, request, storage)
            temporary = path.parent / (
                ".%s.%s.stream-staging" % (path.name, request["request_id"])
            )
            if temporary.exists() or temporary.is_symlink():
                details = _lstat_no_symlink(temporary, "stale stream staging")
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                    or details.st_dev != contract.expected_root_device
                ):
                    raise RemoteWorkerError("stale stream staging metadata is invalid")
                os.unlink(str(temporary))
                _fsync_directory(temporary.parent)
        finally:
            os.close(lock_fd)
    except BaseException:
        if data_reservation_attempted:
            cleanup_fd, _ = _global_lock(contract)
            try:
                _remove_request_staging_file(temporary)
                _release_matching_data_reservation(contract, request, data_reservation)
            finally:
                os.close(cleanup_fd)
        raise
    assert temporary is not None and data_reservation is not None
    context: Optional[Dict[str, Any]] = None
    try:
        temporary, digest, size = _write_unknown_stream_temporary(
            path,
            source,
            max_bytes,
            0o400,
            temporary=temporary,
        )
        lock_fd, _ = _global_lock(contract)
        try:
            current = _current_file_identity(path)
            if current != initial:
                raise RemoteWorkerError("stream CAS changed while stdin was in flight")
            storage_staged = verify_root(contract, measure_size=True)
            own_reservation = _load_data_reservation(contract, data_reservation, request)
            other_reserved = (
                storage_staged["data_reservation_reserved_bytes"]
                - own_reservation["reserved_bytes"]
            )
            if (
                other_reserved < 0
                or storage_staged["campaign_data_apparent_bytes"]
                + other_reserved
                + MAX_PUT_JOURNAL_BYTES
                > contract.max_data_bytes
            ):
                raise RemoteWorkerError("stream commit exceeds its data reservation")
            receipt_reservation = _receipt_reservation_path(contract, request["request_id"])
            receipt_reservation_bytes = (
                _lstat_no_symlink(receipt_reservation, "receipt reservation").st_size
                if receipt_reservation.exists()
                else 0
            )
            data_reservation_bytes = _lstat_no_symlink(
                data_reservation, "data reservation"
            ).st_size
            context = _put_commit_context(
                contract,
                request,
                path,
                temporary,
                expected_current=expected_current,
                current_size=current_size,
                expected_sha256=digest,
                size=size,
                mode=0o400,
                storage_precommit=storage,
                storage_staged=storage_staged,
                receipt_reservation_apparent_bytes=receipt_reservation_bytes,
                data_reservation_apparent_bytes=data_reservation_bytes,
            )
            context["data_reservation"] = data_reservation
            if current is not None:
                os.rename(str(path), str(context["backup"]))
                _fsync_directory(path.parent)
            os.rename(str(temporary), str(path))
            _fsync_directory(path.parent)
            storage_after = verify_root(contract, measure_size=True)
            journal_bytes = _lstat_no_symlink(context["journal"], "put journal").st_size
            if (
                storage_after["campaign_apparent_bytes"]
                != storage_staged["campaign_apparent_bytes"] + journal_bytes
                or storage_after["campaign_data_apparent_bytes"]
                != storage_staged["campaign_data_apparent_bytes"] + journal_bytes
            ):
                raise RemoteWorkerError("stream postcommit capacity accounting differs")
        finally:
            os.close(lock_fd)
    except BaseException:
        rollback_fd, _ = _global_lock(contract)
        try:
            if context is not None:
                _rollback_put_commit(context)
            else:
                _remove_request_staging_file(temporary)
                _release_matching_data_reservation(contract, request, data_reservation)
        finally:
            os.close(rollback_fd)
        raise
    projected_data = storage_staged["campaign_data_apparent_bytes"] - current_size
    if commit_context is not None:
        commit_context.update(context)
    return {
        "schema_id": PUT_RESULT_SCHEMA,
        "schema_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "relative": relative,
        "sha256": digest,
        "size": size,
        "mode": "0o400",
        "max_bytes": max_bytes,
        "previous_sha256": current,
        "projected_data_bytes": projected_data,
        "projected_campaign_bytes": (
            storage_staged["campaign_apparent_bytes"]
            - current_size
            - receipt_reservation_bytes
            - data_reservation_bytes
            + contract.max_receipt_bytes
        ),
        "projected_free_bytes": (
            storage_staged["free_bytes"]
            + current_size
            + receipt_reservation_bytes
            + data_reservation_bytes
            - contract.max_receipt_bytes
        ),
        "receipt_capacity_reserved_bytes": contract.max_receipt_bytes,
        "receipt_reservation_apparent_bytes": receipt_reservation_bytes,
        "data_reservation_apparent_bytes": data_reservation_bytes,
        "journal_bytes": journal_bytes,
        "backup_bytes": current_size if current is not None else 0,
        "transaction_overhead_bytes": journal_bytes,
        "storage_precommit": storage,
        "storage_staged": storage_staged,
        "storage_transaction": storage_after,
    }


class _BytesReader:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        end = min(self._offset + size, len(self._payload))
        result = self._payload[self._offset : end]
        self._offset = end
        return result


def _lock_path(contract: WorkerContract, name: Any) -> Path:
    safe = _validate_identifier(name, "lock name")
    return _safe_path(contract, "control/locks/%s.json" % safe, "lease lock")


def _lease_history_path(contract: WorkerContract, name: Any, lease_epoch: Any) -> Path:
    safe_name = _validate_identifier(name, "lock name")
    safe_epoch = _validate_identifier(lease_epoch, "lease epoch")
    return _safe_path(
        contract,
        "control/locks/history/%s/%s.json" % (safe_name, safe_epoch),
        "lease history",
        create_parents=True,
    )


def _lease_pending_path(contract: WorkerContract, name: Any) -> Path:
    safe = _validate_identifier(name, "lock name")
    return _safe_path(
        contract,
        "control/locks/.%s.pending.json" % safe,
        "pending lease transition",
    )


def _load_json_file(path: Path) -> Dict[str, Any]:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read(MAX_MANIFEST_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise RemoteWorkerError("JSON control object exceeds its limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("JSON control object is malformed") from error
    if not isinstance(value, dict):
        raise RemoteWorkerError("JSON control object must be an object")
    return value


LEASE_FIELDS = {
    "schema_id",
    "name",
    "owner",
    "token",
    "revision",
    "active",
    "released",
    "lease_epoch",
    "request_id",
    "semantic_sha256",
    "command_sha256",
    "issued_unix_ms",
    "expires_unix_ms",
    "lease_sha256",
}


def _load_lease_document(
    contract: WorkerContract,
    path: Path,
    name: str,
    *,
    request: Optional[Mapping[str, Any]] = None,
    lease_epoch: Optional[str] = None,
) -> Dict[str, Any]:
    details = _lstat_no_symlink(path, "lease state")
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_size > MAX_MANIFEST_BYTES
    ):
        raise RemoteWorkerError("lease state metadata is invalid")
    document = _load_json_file(path)
    if (
        set(document) != LEASE_FIELDS
        or document.get("schema_id") != "cascadia.r2-map.remote-lease.v2"
        or document.get("name") != name
        or document.get("lease_sha256")
        != document_sha256(document, "lease_sha256")
        or not isinstance(document.get("revision"), int)
        or isinstance(document.get("revision"), bool)
        or document.get("revision", 0) < 1
        or not isinstance(document.get("active"), bool)
        or not isinstance(document.get("released"), bool)
        or document.get("active") == document.get("released")
        or not isinstance(document.get("issued_unix_ms"), int)
        or not isinstance(document.get("expires_unix_ms"), int)
        or document.get("expires_unix_ms", -1) < document.get("issued_unix_ms", 0)
        or SHA256.fullmatch(str(document.get("token"))) is None
    ):
        raise RemoteWorkerError("lease state identity is invalid")
    if lease_epoch is not None and document.get("lease_epoch") != lease_epoch:
        raise RemoteWorkerError("lease history epoch identity differs")
    if request is not None and (
        document.get("request_id") != request["request_id"]
        or document.get("semantic_sha256") != request["semantic_sha256"]
        or document.get("command_sha256") != request["command_sha256"]
    ):
        raise RemoteWorkerError("lease history request identity differs")
    return document


def _recover_pending_lease_transition(
    contract: WorkerContract, name: str, path: Path
) -> Optional[Dict[str, Any]]:
    pending_path = _lease_pending_path(contract, name)
    try:
        pending = _load_lease_document(contract, pending_path, name)
    except FileNotFoundError:
        try:
            return _load_lease_document(contract, path, name)
        except FileNotFoundError:
            return None
    history_path = _lease_history_path(contract, name, pending["lease_epoch"])
    try:
        history = _load_lease_document(
            contract,
            history_path,
            name,
            lease_epoch=pending["lease_epoch"],
        )
    except FileNotFoundError:
        _atomic_write(history_path, canonical_json(pending), 0o600)
        history = pending
    if history != pending:
        raise RemoteWorkerError("pending lease transition differs from durable history")
    try:
        current = _load_lease_document(contract, path, name)
    except FileNotFoundError:
        current = None
    if current is not None and current["revision"] > pending["revision"]:
        raise RemoteWorkerError("pending lease transition is older than current state")
    if current is not None and current["revision"] == pending["revision"]:
        if current != pending:
            raise RemoteWorkerError("pending lease transition conflicts with current state")
    else:
        _atomic_write(path, canonical_json(pending), 0o600)
        current = pending
    os.unlink(str(pending_path))
    _fsync_directory(pending_path.parent)
    return current


def _lease_operation(
    contract: WorkerContract,
    operation: str,
    arguments: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    path = _lock_path(contract, arguments.get("name"))
    owner = _validate_identifier(arguments.get("owner"), "lock owner")
    token = arguments.get("token")
    if operation != "lock-acquire":
        _validate_sha256(token, "lock token")
    lease_seconds = arguments.get("lease_seconds", 0)
    lease_epoch = _validate_identifier(arguments.get("lease_epoch"), "lease epoch")
    if not isinstance(lease_seconds, int) or not 10 <= lease_seconds <= 3600:
        raise RemoteWorkerError("lock lease must be between 10 and 3600 seconds")
    now = time.time_ns() // 1_000_000
    lock_fd, _ = _global_lock(contract)
    try:
        current = _recover_pending_lease_transition(
            contract, arguments["name"], path
        )
        history_path = _lease_history_path(
            contract, arguments["name"], lease_epoch
        )
        try:
            history = _load_lease_document(
                contract,
                history_path,
                arguments["name"],
                request=request,
                lease_epoch=lease_epoch,
            )
        except FileNotFoundError:
            history = None
        if history is not None:
            # Delayed retries return their immutable epoch result without
            # rewinding a newer current lease state.
            if current is None or current["revision"] < history["revision"]:
                _atomic_write(path, canonical_json(history), 0o600)
            elif current["revision"] == history["revision"] and current != history:
                raise RemoteWorkerError("lease history conflicts with current state")
            return history
        if operation == "lock-acquire":
            if (
                current is not None
                and current.get("active") is True
                and int(current.get("expires_unix_ms", 0)) > now
            ):
                raise RemoteWorkerError("lease lock is held")
            token = hashlib.sha256(os.urandom(32)).hexdigest()
            revision = 1 if current is None else int(current.get("revision", 0)) + 1
        else:
            if (
                current is None
                or current.get("active") is not True
                or current.get("token") != token
                or current.get("owner") != owner
            ):
                raise RemoteWorkerError("lease lock token/owner does not match")
            if operation == "lock-release":
                revision = int(current["revision"]) + 1
                expires = now
                active = False
                released = True
            else:
                if int(current.get("expires_unix_ms", 0)) <= now:
                    raise RemoteWorkerError("lease lock expired before renewal")
                revision = int(current["revision"]) + 1
                expires = now + lease_seconds * 1000
                active = True
                released = False
        if operation == "lock-acquire":
            expires = now + lease_seconds * 1000
            active = True
            released = False
        document = {
            "schema_id": "cascadia.r2-map.remote-lease.v2",
            "name": arguments["name"],
            "owner": owner,
            "token": token,
            "revision": revision,
            "active": active,
            "released": released,
            "lease_epoch": lease_epoch,
            "request_id": request["request_id"],
            "semantic_sha256": request["semantic_sha256"],
            "command_sha256": request["command_sha256"],
            "issued_unix_ms": now,
            "expires_unix_ms": expires,
        }
        document["lease_sha256"] = document_sha256(document, "lease_sha256")
        pending_path = _lease_pending_path(contract, arguments["name"])
        if pending_path.exists() or history_path.exists():
            raise RemoteWorkerError("lease transition namespace changed while locked")
        encoded = canonical_json(document)
        # Pending is the write-ahead intent. Any interruption is reconciled
        # before the next operation; history then preserves this epoch forever.
        _atomic_write(pending_path, encoded, 0o600)
        _atomic_write(history_path, encoded, 0o600)
        _atomic_write(path, encoded, 0o600)
        os.unlink(str(pending_path))
        _fsync_directory(pending_path.parent)
        return document
    finally:
        os.close(lock_fd)


def _validate_transaction_manifest(payload: bytes) -> Dict[str, Any]:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise RemoteWorkerError("transaction manifest exceeds 2 MiB")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkerError("transaction manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_id") != TRANSACTION_SCHEMA:
        raise RemoteWorkerError("transaction manifest schema is invalid")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RemoteWorkerError("transaction manifest version is invalid")
    _validate_identifier(manifest.get("transaction_id"), "transaction_id")
    target_parts = _relative_parts(manifest.get("target_relative"), "transaction target")
    if target_parts[0] not in IMMUTABLE_TRANSACTION_TOP_LEVELS:
        raise RemoteWorkerError("transaction target top-level is not immutable-authorized")
    if target_parts[0] == "runs":
        if len(target_parts) != 4 or target_parts[2] != "checkpoints":
            raise RemoteWorkerError(
                "run transaction target must be runs/RUN/checkpoints/CHECKPOINT"
            )
        _validate_identifier(target_parts[1], "checkpoint run identifier")
        _validate_identifier(target_parts[3], "checkpoint identifier")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RemoteWorkerError("transaction must contain at least one object")
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for value in objects:
        allowed_fields = {"relative", "sha256", "size"}
        if (
            not isinstance(value, dict)
            or not allowed_fields.issubset(value)
            or set(value) - (allowed_fields | {"mode"})
        ):
            raise RemoteWorkerError("transaction object descriptor is invalid")
        relative = value["relative"]
        _relative_parts(relative, "transaction object")
        if relative == ".r2-map-transaction.json" or relative in seen:
            raise RemoteWorkerError("transaction object path is duplicated/reserved")
        seen.add(relative)
        size = value["size"]
        if not isinstance(size, int) or size < 0:
            raise RemoteWorkerError("transaction object size is invalid")
        mode = value.get("mode", "0400")
        if mode not in {"0400", "0500"}:
            raise RemoteWorkerError("transaction object mode is invalid")
        normalized_descriptor = {
            "relative": relative,
            "sha256": _validate_sha256(value["sha256"], "object SHA-256"),
            "size": size,
        }
        if mode != "0400":
            normalized_descriptor["mode"] = mode
        normalized.append(normalized_descriptor)
    if normalized != sorted(normalized, key=lambda item: item["relative"]):
        raise RemoteWorkerError("transaction objects must be sorted canonically")
    if manifest.get("objects") != normalized:
        raise RemoteWorkerError("transaction objects are not canonical")
    expected = document_sha256(manifest, "manifest_sha256")
    if manifest.get("manifest_sha256") != expected:
        raise RemoteWorkerError("transaction manifest hash is invalid")
    if canonical_json(manifest) != payload:
        raise RemoteWorkerError("transaction manifest bytes are not canonical JSON")
    return manifest


def _transaction_root(contract: WorkerContract, transaction_id: Any) -> Path:
    safe = _validate_identifier(transaction_id, "transaction_id")
    return _safe_path(contract, "control/transactions/%s.staging" % safe, "transaction")


def _transaction_outcome_path(contract: WorkerContract, transaction_id: Any) -> Path:
    safe = _validate_identifier(transaction_id, "transaction outcome")
    return _safe_path(
        contract,
        "control/transactions/%s.outcome.json" % safe,
        "transaction outcome",
        create_parents=True,
    )


def _transaction_outcome(
    manifest: Mapping[str, Any], outcome: str
) -> Dict[str, Any]:
    if outcome not in {"commit", "abort"}:
        raise RemoteWorkerError("transaction outcome is invalid")
    document: Dict[str, Any] = {
        "schema_id": TRANSACTION_OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "transaction_id": manifest["transaction_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "target_relative": manifest["target_relative"],
        "object_count": len(manifest["objects"]),
        "outcome": outcome,
    }
    document["outcome_sha256"] = document_sha256(document, "outcome_sha256")
    return document


def _load_transaction_outcome(
    contract: WorkerContract, transaction_id: Any
) -> Optional[Dict[str, Any]]:
    path = _transaction_outcome_path(contract, transaction_id)
    try:
        document = _load_json_file(path)
    except FileNotFoundError:
        return None
    if (
        set(document)
        != {
            "schema_id",
            "schema_version",
            "transaction_id",
            "manifest_sha256",
            "target_relative",
            "object_count",
            "outcome",
            "outcome_sha256",
        }
        or document.get("schema_id") != TRANSACTION_OUTCOME_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("transaction_id") != transaction_id
        or document.get("outcome") not in {"commit", "abort"}
        or document.get("outcome_sha256")
        != document_sha256(document, "outcome_sha256")
    ):
        raise RemoteWorkerError("transaction outcome identity is invalid")
    _validate_sha256(document.get("manifest_sha256"), "transaction outcome manifest")
    _relative_parts(document.get("target_relative"), "transaction outcome target")
    return document


def _cleanup_transaction_outcome_temporaries(
    contract: WorkerContract, transaction_id: Any
) -> None:
    """Recover exact atomic-write residue without touching another transaction."""
    path = _transaction_outcome_path(contract, transaction_id)
    temporary = path.parent / (".%s.pending.tmp" % path.name)
    try:
        details = os.lstat(str(temporary))
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid)
        != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o400
        or details.st_size > MAX_DATA_RESERVATION_BYTES
    ):
        raise RemoteWorkerError("transaction outcome temporary metadata is invalid")
    os.unlink(str(temporary))
    _fsync_directory(path.parent)


def _validate_transaction_cleanup_tree(
    contract: WorkerContract, path: Path, label: str
) -> None:
    """Prove a transaction-owned tree can be removed without following aliases."""
    root = _lstat_no_symlink(path, label)
    if (
        not stat.S_ISDIR(root.st_mode)
        or root.st_dev != contract.expected_root_device
        or (root.st_uid, root.st_gid) != (contract.expected_uid, contract.expected_gid)
    ):
        raise RemoteWorkerError("%s identity is invalid" % label)
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(str(current)) as scanner:
            children = list(scanner)
        for entry in children:
            child = Path(entry.path)
            details = entry.stat(follow_symlinks=False)
            if details.st_dev != root.st_dev or (
                details.st_uid,
                details.st_gid,
            ) != (contract.expected_uid, contract.expected_gid):
                raise RemoteWorkerError("%s crosses its owned device" % label)
            if stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode):
                stack.append(child)
            elif stat.S_ISREG(details.st_mode) and details.st_nlink == 1:
                continue
            else:
                raise RemoteWorkerError("%s contains an unsafe entry" % label)


def _transaction_tree(contract: WorkerContract, manifest: Mapping[str, Any]) -> Path:
    target_relative = manifest["target_relative"]
    target = _safe_path(contract, target_relative, "transaction target", create_parents=True)
    transaction_id = _validate_identifier(manifest["transaction_id"], "transaction_id")
    staging_name = ".%s.r2map-%s.staging" % (target.name, transaction_id)
    relative = target.parent.relative_to(contract.root) / staging_name
    return _safe_path(contract, relative.as_posix(), "transaction tree")


def _transaction_begin(
    contract: WorkerContract, arguments: Mapping[str, Any], source: BinaryIO
) -> Dict[str, Any]:
    size = arguments.get("size")
    expected_sha256 = _validate_sha256(arguments.get("sha256"), "manifest SHA-256")
    if not isinstance(size, int) or size < 2 or size > MAX_MANIFEST_BYTES:
        raise RemoteWorkerError("transaction manifest size is invalid")
    payload = source.read(size + 1)
    if len(payload) != size or source.read(1):
        raise RemoteWorkerError("transaction manifest upload size differs")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RemoteWorkerError("transaction manifest transport hash differs")
    manifest = _validate_transaction_manifest(payload)
    staging = _transaction_root(contract, manifest["transaction_id"])
    target = _safe_path(
        contract, manifest["target_relative"], "transaction target", create_parents=True
    )
    tree = _transaction_tree(contract, manifest)
    lock_fd, _ = _global_lock(contract)
    try:
        _cleanup_transaction_outcome_temporaries(contract, manifest["transaction_id"])
        if _load_transaction_outcome(contract, manifest["transaction_id"]) is not None:
            raise RemoteWorkerError("transaction identifier already has a terminal outcome")
        if staging.exists() != tree.exists():
            for partial in (staging, tree):
                if not partial.exists():
                    continue
                if not stat.S_ISDIR(_lstat_no_symlink(partial, "partial transaction").st_mode):
                    raise RemoteWorkerError("partial transaction staging is unsafe")
                if partial == staging:
                    for entry in partial.iterdir():
                        if (
                            not entry.name.startswith(".transaction-manifest.json.")
                            or not entry.name.endswith(".tmp")
                            or not stat.S_ISREG(
                                _lstat_no_symlink(entry, "partial transaction manifest").st_mode
                            )
                        ):
                            raise RemoteWorkerError(
                                "partial transaction staging is not recoverable"
                            )
                elif any(partial.iterdir()):
                    raise RemoteWorkerError("partial transaction tree is not empty")
                _remove_verified_tree(partial, diagnostic_root=contract.root)
        for active, label in (
            (staging, "transaction staging"),
            (tree, "transaction tree"),
        ):
            if not active.exists():
                continue
            details = _lstat_no_symlink(active, label)
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_dev != contract.expected_root_device
                or (details.st_uid, details.st_gid)
                != (contract.expected_uid, contract.expected_gid)
                or stat.S_IMODE(details.st_mode) != 0o700
            ):
                raise RemoteWorkerError("%s identity is invalid" % label)
        if target.exists():
            raise RemoteWorkerError("immutable transaction target already exists")
        if staging.exists():
            manifest_path = staging / "transaction-manifest.json"
            if not manifest_path.exists():
                for entry in staging.iterdir():
                    if (
                        not entry.name.startswith(".transaction-manifest.json.")
                        or not entry.name.endswith(".tmp")
                        or not stat.S_ISREG(
                            _lstat_no_symlink(entry, "partial transaction manifest").st_mode
                        )
                    ):
                        raise RemoteWorkerError("incomplete transaction staging is unsafe")
                if any(tree.iterdir()):
                    raise RemoteWorkerError("incomplete transaction tree is not empty")
                _remove_verified_tree(staging, diagnostic_root=contract.root)
                _remove_verified_tree(tree, diagnostic_root=contract.root)
            else:
                manifest_details = _lstat_no_symlink(
                    manifest_path, "existing transaction manifest"
                )
                if (
                    not stat.S_ISREG(manifest_details.st_mode)
                    or manifest_details.st_nlink != 1
                    or manifest_details.st_dev != contract.expected_root_device
                    or (manifest_details.st_uid, manifest_details.st_gid)
                    != (contract.expected_uid, contract.expected_gid)
                    or stat.S_IMODE(manifest_details.st_mode) != 0o400
                ):
                    raise RemoteWorkerError("existing transaction manifest metadata differs")
                _validate_transaction_cleanup_tree(contract, tree, "transaction tree")
                existing_manifest = _load_json_file(manifest_path)
                if existing_manifest != manifest:
                    raise RemoteWorkerError("existing transaction manifest identity differs")
                storage_present = _storage_capacity_state(contract)
                return {
                    "transaction_id": manifest["transaction_id"],
                    "target_relative": manifest["target_relative"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "object_count": len(manifest["objects"]),
                    "recovered": True,
                    "storage_precommit": storage_present,
                    "storage_postcommit": storage_present,
                }
        storage_before = _storage_capacity_state(contract)
        _require_mutation_headroom(
            contract, storage_before, incoming_data_bytes=len(payload)
        )
        try:
            os.mkdir(str(staging), 0o700)
            os.mkdir(str(tree), 0o700)
            _atomic_write(staging / "transaction-manifest.json", payload, 0o400)
            _fsync_directory(staging)
            _fsync_directory(staging.parent)
            _fsync_directory(tree.parent)
            storage_after = _storage_capacity_state(contract)
            if (
                storage_after["campaign_data_apparent_bytes"]
                != storage_before["campaign_data_apparent_bytes"] + len(payload)
            ):
                raise RemoteWorkerError("transaction begin capacity accounting differs")
        except BaseException:
            if tree.exists():
                _remove_verified_tree(tree, diagnostic_root=contract.root)
            if staging.exists():
                _remove_verified_tree(staging, diagnostic_root=contract.root)
            raise
    finally:
        os.close(lock_fd)
    return {
        "transaction_id": manifest["transaction_id"],
        "target_relative": manifest["target_relative"],
        "manifest_sha256": manifest["manifest_sha256"],
        "object_count": len(manifest["objects"]),
        "recovered": False,
        "storage_precommit": storage_before,
        "storage_postcommit": storage_after,
    }


def _load_transaction(contract: WorkerContract, transaction_id: Any) -> Tuple[Path, Dict[str, Any]]:
    staging = _transaction_root(contract, transaction_id)
    try:
        details = _lstat_no_symlink(staging, "transaction staging")
    except FileNotFoundError as error:
        raise RemoteWorkerError("transaction staging does not exist") from error
    if not stat.S_ISDIR(details.st_mode):
        raise RemoteWorkerError("transaction staging is not a directory")
    manifest = _load_json_file(staging / "transaction-manifest.json")
    payload = canonical_json(manifest)
    validated = _validate_transaction_manifest(payload)
    if validated["transaction_id"] != transaction_id:
        raise RemoteWorkerError("transaction identity differs from staging path")
    return staging, validated


def _transaction_put(
    contract: WorkerContract,
    arguments: Mapping[str, Any],
    source: BinaryIO,
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    reservation: Optional[Path] = None
    temporary: Optional[Path] = None
    reservation_attempted = False
    lock_fd: Optional[int] = None
    try:
        lock_fd, _ = _global_lock(contract)
        try:
            _staging, manifest = _load_transaction(contract, arguments.get("transaction_id"))
            relative = arguments.get("relative")
            descriptors = {item["relative"]: item for item in manifest["objects"]}
            if relative not in descriptors:
                raise RemoteWorkerError("transaction object is absent from the frozen manifest")
            descriptor = descriptors[relative]
            if (
                arguments.get("size") != descriptor["size"]
                or arguments.get("sha256") != descriptor["sha256"]
            ):
                raise RemoteWorkerError("transaction object transport differs from manifest")
            tree = _transaction_tree(contract, manifest)
            path = tree / Path(*_relative_parts(relative, "transaction object"))
            _safe_transaction_path(tree, path, create_parents=True)
            mode = 0o500 if descriptor.get("mode") == "0500" else 0o400
            if path.exists():
                details = _lstat_no_symlink(path, "transaction object")
                digest, size = file_sha256(path)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                    or stat.S_IMODE(details.st_mode) != mode
                    or digest != descriptor["sha256"]
                    or size != descriptor["size"]
                ):
                    raise RemoteWorkerError("existing transaction object identity differs")
                reservation = _data_reservation_path(contract, request["request_id"])
                if reservation.exists():
                    _release_matching_data_reservation(contract, request, reservation)
                storage_present = _storage_capacity_state(contract)
                recovered_result = {
                    **dict(descriptor),
                    "recovered": True,
                    "storage_precommit": storage_present,
                    "storage_staged": storage_present,
                    "storage_postcommit": storage_present,
                }
                os.close(lock_fd)
                lock_fd = None
                replay_digest = hashlib.sha256()
                remaining = descriptor["size"]
                while remaining:
                    chunk = source.read(min(1 << 20, remaining))
                    if not chunk:
                        raise RemoteWorkerError("recovered transaction body ended early")
                    replay_digest.update(chunk)
                    remaining -= len(chunk)
                if source.read(1) or replay_digest.hexdigest() != descriptor["sha256"]:
                    raise RemoteWorkerError("recovered transaction body identity differs")
                return recovered_result
            storage_before = _storage_capacity_state(contract)
            reservation_attempted = True
            reservation = _reserve_data_capacity_locked(contract, request, storage_before)
            temporary = path.parent / (
                ".%s.%s.transaction-staging" % (path.name, request["request_id"])
            )
            if temporary.exists() or temporary.is_symlink():
                details = _lstat_no_symlink(temporary, "stale transaction object staging")
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                    or details.st_dev != contract.expected_root_device
                ):
                    raise RemoteWorkerError("transaction object staging metadata is invalid")
                os.unlink(str(temporary))
                _fsync_directory(temporary.parent)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
    except BaseException:
        if reservation_attempted:
            cleanup_fd, _ = _global_lock(contract)
            try:
                _remove_request_staging_file(temporary)
                _release_matching_data_reservation(contract, request, reservation)
            finally:
                os.close(cleanup_fd)
        raise
    assert temporary is not None and reservation is not None
    try:
        temporary = _stage_stream(
            path,
            source,
            descriptor["size"],
            descriptor["sha256"],
            mode,
            temporary=temporary,
        )
        lock_fd, _ = _global_lock(contract)
        try:
            _staging, current_manifest = _load_transaction(
                contract, arguments.get("transaction_id")
            )
            if current_manifest != manifest or path.exists():
                raise RemoteWorkerError("transaction changed while object bytes were in flight")
            storage_staged = _storage_capacity_state(contract)
            own = _load_data_reservation(contract, reservation, request)
            _require_mutation_headroom(
                contract,
                storage_staged,
                incoming_data_bytes=0,
                data_reservation_credit=own["reserved_bytes"],
            )
            os.rename(str(temporary), str(path))
            _fsync_directory(path.parent)
            _release_data_reservation(reservation)
            storage_after = _storage_capacity_state(contract)
            expected_data = storage_staged["campaign_data_apparent_bytes"]
            if storage_after["campaign_data_apparent_bytes"] != expected_data:
                raise RemoteWorkerError("transaction object capacity accounting differs")
        finally:
            os.close(lock_fd)
    except BaseException:
        cleanup_fd, _ = _global_lock(contract)
        try:
            _remove_request_staging_file(temporary)
            _release_matching_data_reservation(contract, request, reservation)
        finally:
            os.close(cleanup_fd)
        raise
    return {
        **dict(descriptor),
        "recovered": False,
        "storage_precommit": storage_before,
        "storage_staged": storage_staged,
        "storage_postcommit": storage_after,
    }


def _transaction_import(
    contract: WorkerContract,
    arguments: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    source_relative = arguments.get("source_relative")
    source_parts = _relative_parts(source_relative, "transaction import source")
    if source_parts[0] != "build":
        raise RemoteWorkerError("transaction import source must be below build/")
    source_path = _safe_path(contract, source_relative, "transaction import source")
    details = _lstat_no_symlink(source_path, "transaction import source")
    if not stat.S_ISREG(details.st_mode):
        raise RemoteWorkerError("transaction import source is not a regular file")
    descriptor = os.open(str(source_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            result = _transaction_put(contract, arguments, source, request)
    finally:
        os.close(descriptor)
    return {**result, "source_relative": source_relative}


def _safe_transaction_path(tree: Path, path: Path, *, create_parents: bool) -> None:
    try:
        relative = path.relative_to(tree)
    except ValueError as error:
        raise RemoteWorkerError("transaction path escapes staging tree") from error
    current = tree
    root_stat = _lstat_no_symlink(tree, "transaction tree")
    for part in relative.parts[:-1]:
        current = current / part
        try:
            details = _lstat_no_symlink(current, "transaction ancestor")
        except FileNotFoundError:
            if not create_parents:
                raise
            os.mkdir(str(current), 0o700)
            _fsync_directory(current.parent)
            details = _lstat_no_symlink(current, "transaction ancestor")
        if not stat.S_ISDIR(details.st_mode) or details.st_dev != root_stat.st_dev:
            raise RemoteWorkerError("transaction ancestor is unsafe")


def _verify_transaction_object_set(
    contract: WorkerContract,
    tree: Path,
    manifest: Mapping[str, Any],
    *,
    allow_provenance: bool,
) -> None:
    root = _lstat_no_symlink(tree, "transaction tree")
    if not stat.S_ISDIR(root.st_mode) or root.st_dev != contract.expected_root_device:
        raise RemoteWorkerError("transaction tree identity is invalid")
    observed: set[str] = set()
    for current_root, directories, files in os.walk(str(tree), followlinks=False):
        for name in directories:
            details = _lstat_no_symlink(Path(current_root) / name, "transaction directory")
            if not stat.S_ISDIR(details.st_mode) or details.st_dev != root.st_dev:
                raise RemoteWorkerError("transaction tree contains an unsafe directory")
        for name in files:
            path = Path(current_root) / name
            details = _lstat_no_symlink(path, "transaction file")
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or details.st_dev != root.st_dev
            ):
                raise RemoteWorkerError("transaction tree contains an unsafe file")
            observed.add(path.relative_to(tree).as_posix())
    expected = {item["relative"] for item in manifest["objects"]}
    permitted = expected | ({".r2-map-transaction.json"} if allow_provenance else set())
    if not expected.issubset(observed) or not observed.issubset(permitted):
        raise RemoteWorkerError("transaction tree object set differs from its manifest")


def _verify_transaction_payloads(
    contract: WorkerContract, tree: Path, manifest: Mapping[str, Any]
) -> None:
    for descriptor in manifest["objects"]:
        path = tree / Path(*_relative_parts(descriptor["relative"], "transaction object"))
        _safe_transaction_path(tree, path, create_parents=False)
        details = _lstat_no_symlink(path, "transaction object")
        expected_mode = 0o500 if descriptor.get("mode") == "0500" else 0o400
        digest, size = file_sha256(path)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_dev != contract.expected_root_device
            or (details.st_uid, details.st_gid)
            != (contract.expected_uid, contract.expected_gid)
            or stat.S_IMODE(details.st_mode) != expected_mode
            or digest != descriptor["sha256"]
            or size != descriptor["size"]
        ):
            raise RemoteWorkerError("transaction object failed commit verification")


def _verify_committed_transaction_tree(
    contract: WorkerContract, target: Path, manifest: Mapping[str, Any]
) -> None:
    details = _lstat_no_symlink(target, "committed transaction target")
    if not stat.S_ISDIR(details.st_mode) or stat.S_IMODE(details.st_mode) != 0o500:
        raise RemoteWorkerError("committed transaction target mode/type differs")
    _verify_transaction_object_set(contract, target, manifest, allow_provenance=True)
    provenance = target / ".r2-map-transaction.json"
    if provenance.read_bytes() != canonical_json(manifest):
        raise RemoteWorkerError("committed transaction provenance differs")
    _verify_transaction_payloads(contract, target, manifest)


def _transaction_commit(contract: WorkerContract, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    lock_fd, _ = _global_lock(contract)
    try:
        return _transaction_commit_locked(contract, arguments)
    finally:
        os.close(lock_fd)


def _transaction_commit_locked(
    contract: WorkerContract, arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    transaction_id = arguments.get("transaction_id")
    expected_manifest = _validate_sha256(arguments.get("manifest_sha256"), "manifest SHA-256")
    outcome_path = _transaction_outcome_path(contract, transaction_id)
    _cleanup_transaction_outcome_temporaries(contract, transaction_id)
    outcome = _load_transaction_outcome(contract, transaction_id)
    recovered = outcome is not None
    if outcome is not None and outcome["outcome"] != "commit":
        raise RemoteWorkerError("transaction already has an abort outcome")
    try:
        staging, manifest = _load_transaction(contract, transaction_id)
    except RemoteWorkerError:
        if outcome is None:
            raise
        target = _safe_path(contract, outcome["target_relative"], "committed target")
        manifest = _load_json_file(target / ".r2-map-transaction.json")
        manifest = _validate_transaction_manifest(canonical_json(manifest))
        staging = _transaction_root(contract, transaction_id)
    if manifest["manifest_sha256"] != expected_manifest:
        raise RemoteWorkerError("commit manifest precondition failed")
    expected_outcome = _transaction_outcome(manifest, "commit")
    if outcome is not None and outcome != expected_outcome:
        raise RemoteWorkerError("commit outcome differs from the transaction manifest")
    outcome_payload = canonical_json(expected_outcome)
    tree = _transaction_tree(contract, manifest)
    target = _safe_path(
        contract, manifest["target_relative"], "transaction target", create_parents=True
    )
    provenance_payload = canonical_json(manifest)
    if target.exists():
        if tree.exists():
            raise RemoteWorkerError("committed target and transaction tree both exist")
        target_details = _lstat_no_symlink(target, "committed transaction target")
        if not stat.S_ISDIR(target_details.st_mode) or stat.S_IMODE(
            target_details.st_mode
        ) not in {0o500, 0o700}:
            raise RemoteWorkerError("recovering transaction target mode/type differs")
        _verify_transaction_object_set(contract, target, manifest, allow_provenance=True)
        if (target / ".r2-map-transaction.json").read_bytes() != provenance_payload:
            raise RemoteWorkerError("recovering transaction provenance differs")
        _verify_transaction_payloads(contract, target, manifest)
        if staging.exists():
            _validate_transaction_cleanup_tree(
                contract, staging, "committed transaction staging"
            )
        storage_before = _storage_capacity_state(contract)
        if outcome is None:
            _require_mutation_headroom(
                contract, storage_before, incoming_data_bytes=len(outcome_payload)
            )
            _atomic_write_recoverable(
                outcome_path, outcome_payload, 0o400, "transaction outcome"
            )
            outcome = expected_outcome
        if stat.S_IMODE(target_details.st_mode) == 0o700:
            os.chmod(str(target), 0o500, follow_symlinks=False)
            _fsync_directory(target)
            _fsync_directory(target.parent)
        _verify_committed_transaction_tree(contract, target, manifest)
        if staging.exists():
            _remove_verified_tree(staging, diagnostic_root=contract.root)
        storage_after = _storage_capacity_state(contract)
        return {
            "transaction_id": transaction_id,
            "target_relative": manifest["target_relative"],
            "manifest_sha256": expected_manifest,
            "object_count": len(manifest["objects"]),
            "committed": True,
            "recovered": True,
            "outcome_sha256": expected_outcome["outcome_sha256"],
            "storage_precommit": storage_before,
            "storage_with_staging": storage_before,
            "storage_postcommit": storage_after,
        }
    _verify_transaction_object_set(contract, tree, manifest, allow_provenance=True)
    _verify_transaction_payloads(contract, tree, manifest)
    if staging.exists():
        _validate_transaction_cleanup_tree(contract, staging, "transaction staging")
    provenance = tree / ".r2-map-transaction.json"
    provenance_present = provenance.exists()
    if provenance_present:
        provenance_details = _lstat_no_symlink(provenance, "transaction provenance")
        if (
            not stat.S_ISREG(provenance_details.st_mode)
            or provenance_details.st_nlink != 1
            or provenance_details.st_dev != contract.expected_root_device
            or (provenance_details.st_uid, provenance_details.st_gid)
            != (contract.expected_uid, contract.expected_gid)
            or stat.S_IMODE(provenance_details.st_mode) != 0o400
            or provenance.read_bytes() != provenance_payload
        ):
            raise RemoteWorkerError("transaction provenance identity differs")
    storage_before = _storage_capacity_state(contract)
    expected_outcome_delta = 0 if outcome is not None else len(outcome_payload)
    expected_provenance_delta = 0 if provenance_present else len(provenance_payload)
    _require_mutation_headroom(
        contract,
        storage_before,
        incoming_data_bytes=expected_outcome_delta + expected_provenance_delta,
    )
    # This is the terminal choice point. Every validation and capacity check is
    # complete before the outcome is made durable; all following mutations are
    # idempotently recoverable from that outcome.
    if outcome is None:
        _atomic_write_recoverable(
            outcome_path, outcome_payload, 0o400, "transaction outcome"
        )
        outcome = expected_outcome
    if not provenance_present:
        _atomic_write(provenance, provenance_payload, 0o400)
    for current_root, directories, files in os.walk(str(tree), topdown=False, followlinks=False):
        for name in files:
            path = Path(current_root) / name
            details = _lstat_no_symlink(path, "transaction final file")
            if not stat.S_ISREG(details.st_mode):
                raise RemoteWorkerError("transaction tree contains a special file")
            relative = path.relative_to(tree).as_posix()
            descriptor = next(
                (item for item in manifest["objects"] if item["relative"] == relative), None
            )
            final_mode = 0o500 if descriptor and descriptor.get("mode") == "0500" else 0o400
            os.chmod(str(path), final_mode, follow_symlinks=False)
        for name in directories:
            path = Path(current_root) / name
            details = _lstat_no_symlink(path, "transaction final directory")
            if not stat.S_ISDIR(details.st_mode):
                raise RemoteWorkerError("transaction tree contains a special directory")
            os.chmod(str(path), 0o500, follow_symlinks=False)
            _fsync_directory(path)
    # macOS requires the directory being renamed to remain owner-writable even
    # when source and destination share a parent.  The hidden staging name is
    # therefore kept at 0700 until the complete tree is atomically renamed.
    # All descendants and provenance are already final and read-only; the
    # visible target is made 0500 before commit success is acknowledged.
    os.chmod(str(tree), 0o700, follow_symlinks=False)
    _fsync_directory(tree)
    storage_with_staging = _storage_capacity_state(contract)
    if (
        storage_with_staging["campaign_data_apparent_bytes"]
        != storage_before["campaign_data_apparent_bytes"]
        + expected_outcome_delta
        + expected_provenance_delta
    ):
        raise RemoteWorkerError("transaction provenance capacity accounting differs")
    os.rename(str(tree), str(target))
    os.chmod(str(target), 0o500, follow_symlinks=False)
    _fsync_directory(target)
    _fsync_directory(target.parent)
    if staging.exists():
        _remove_verified_tree(staging, diagnostic_root=contract.root)
    storage_after = _storage_capacity_state(contract)
    _verify_committed_transaction_tree(contract, target, manifest)
    return {
        "transaction_id": transaction_id,
        "target_relative": manifest["target_relative"],
        "manifest_sha256": expected_manifest,
        "object_count": len(manifest["objects"]),
        "committed": True,
        "recovered": recovered,
        "outcome_sha256": expected_outcome["outcome_sha256"],
        "storage_precommit": storage_before,
        "storage_with_staging": storage_with_staging,
        "storage_postcommit": storage_after,
    }


def _transaction_abort(contract: WorkerContract, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    transaction_id = arguments.get("transaction_id")
    expected_manifest = _validate_sha256(arguments.get("manifest_sha256"), "manifest SHA-256")
    lock_fd, _ = _global_lock(contract)
    try:
        _cleanup_transaction_outcome_temporaries(contract, transaction_id)
        outcome = _load_transaction_outcome(contract, transaction_id)
        if outcome is not None and outcome["outcome"] != "abort":
            raise RemoteWorkerError("transaction already has a commit outcome")
        if outcome is None:
            staging, manifest = _load_transaction(contract, transaction_id)
            if manifest["manifest_sha256"] != expected_manifest:
                raise RemoteWorkerError("abort manifest precondition failed")
            outcome = _transaction_outcome(manifest, "abort")
            recovered = False
        else:
            if outcome["manifest_sha256"] != expected_manifest:
                raise RemoteWorkerError("abort outcome manifest precondition failed")
            staging = _transaction_root(contract, transaction_id)
            manifest = None
            recovered = True
        target = _safe_path(contract, outcome["target_relative"], "aborted target")
        target_owned_by_other = False
        if target.exists():
            try:
                target_manifest_value = _load_json_file(
                    target / ".r2-map-transaction.json"
                )
                target_manifest = _validate_transaction_manifest(
                    canonical_json(target_manifest_value)
                )
            except (FileNotFoundError, RemoteWorkerError) as error:
                raise RemoteWorkerError(
                    "aborted transaction target has no valid owner provenance"
                ) from error
            if (
                target_manifest["target_relative"] != outcome["target_relative"]
                or target_manifest["transaction_id"] == transaction_id
            ):
                raise RemoteWorkerError("aborted transaction owns the visible target")
            _verify_committed_transaction_tree(contract, target, target_manifest)
            target_owned_by_other = True
        if manifest is not None:
            tree = _transaction_tree(contract, manifest)
        else:
            target_parent = target.parent
            tree = _safe_path(
                contract,
                (
                    target_parent.relative_to(contract.root)
                    / (".%s.r2map-%s.staging" % (target.name, transaction_id))
                ).as_posix(),
                "aborted transaction tree",
            )
        if tree.exists():
            _validate_transaction_cleanup_tree(
                contract, tree, "aborted transaction tree"
            )
        if staging.exists():
            _validate_transaction_cleanup_tree(
                contract, staging, "aborted transaction staging"
            )
        if not recovered:
            outcome_payload = canonical_json(outcome)
            storage_before = _storage_capacity_state(contract)
            _require_mutation_headroom(
                contract, storage_before, incoming_data_bytes=len(outcome_payload)
            )
            # The terminal abort choice is persisted only after both deletion
            # targets have passed complete no-follow validation.
            _atomic_write_recoverable(
                _transaction_outcome_path(contract, transaction_id),
                outcome_payload,
                0o400,
                "transaction outcome",
            )
        if tree.exists():
            _remove_verified_tree(tree, diagnostic_root=contract.root)
        if staging.exists():
            _remove_verified_tree(staging, diagnostic_root=contract.root)
    finally:
        os.close(lock_fd)
    return {
        "transaction_id": transaction_id,
        "aborted": True,
        "recovered": recovered,
        "target_owned_by_other_transaction": target_owned_by_other,
        "outcome_sha256": outcome["outcome_sha256"],
    }


def _validate_absolute_under_root(contract: WorkerContract, value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RemoteWorkerError("%s must be absolute" % label)
    try:
        relative = path.relative_to(contract.root)
    except ValueError as error:
        raise RemoteWorkerError("%s escapes the john2 campaign root" % label) from error
    _relative_parts(relative.as_posix(), label)
    return path


def _validate_run_argument(contract: WorkerContract, cwd: Path, value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value or len(value.encode("utf-8")) > 8192:
        raise RemoteWorkerError("run argument is invalid")
    candidates = [value]
    if "=" in value:
        candidates.append(value.split("=", 1)[1])
    for candidate in candidates:
        if candidate.startswith("/"):
            _validate_absolute_under_root(contract, candidate, "run argument path")
        elif "/" in candidate or candidate in {".", ".."} or candidate.startswith("."):
            resolved = (cwd / candidate).resolve(strict=False)
            try:
                resolved.relative_to(contract.root)
            except ValueError as error:
                raise RemoteWorkerError(
                    "relative run argument escapes the campaign root"
                ) from error
    return value


def _validate_controller_source(
    contract: WorkerContract,
    executable: Path,
    cwd: Path,
    arguments: Mapping[str, Any],
    argv: Sequence[Any],
) -> Path:
    try:
        relative = executable.relative_to(contract.root)
    except ValueError as error:
        raise RemoteWorkerError("controller executable escapes the campaign root") from error
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != "source"
        or parts[2:] != ("tools", "r2_map_expert_iteration.py")
        or cwd != contract.root.joinpath(*parts[:2])
    ):
        raise RemoteWorkerError("controller executable/cwd is outside one source freeze")
    source_root = cwd
    provenance_path = source_root / ".r2-map-transaction.json"
    provenance = _load_json_file(provenance_path)
    payload = canonical_json(provenance)
    validated = _validate_transaction_manifest(payload)
    expected_manifest = _validate_sha256(
        arguments.get("source_manifest_sha256"), "controller source manifest SHA-256"
    )
    if (
        validated["manifest_sha256"] != expected_manifest
        or validated["target_relative"] != relative.parent.parent.as_posix()
        or not any(
            item["relative"] == "tools/r2_map_expert_iteration.py" and item.get("mode") == "0500"
            for item in validated["objects"]
        )
    ):
        raise RemoteWorkerError("controller source freeze identity differs")
    if len(argv) < 2 or argv[1] not in CONTROLLER_COMMANDS:
        raise RemoteWorkerError("controller subcommand is not authorized")
    return source_root


def _bounded_tool_output(argv: Sequence[str]) -> Tuple[str, str]:
    completed = subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=30,
        close_fds=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr)[:4096].decode("utf-8", errors="replace")
        raise RemoteWorkerError("executable inspection tool failed: %s" % detail.strip())
    if (
        len(completed.stdout) > MAX_EPHEMERAL_MANIFEST_BYTES
        or len(completed.stderr) > MAX_EPHEMERAL_MANIFEST_BYTES
    ):
        raise RemoteWorkerError("executable inspection output exceeded 64 KiB")
    return (
        completed.stdout.decode("utf-8", errors="strict").strip(),
        completed.stderr.decode("utf-8", errors="strict").strip(),
    )


def _parse_designated_requirement(output: str) -> str:
    for line in output.splitlines():
        normalized = line.strip()
        # macOS 15 prefixes requirement-dump records with ``# `` while older
        # codesign releases emit the same record without it.
        if normalized.startswith("# "):
            normalized = normalized[2:]
        if normalized.startswith("designated =>"):
            return normalized.partition("=>")[2].strip()
    return ""


def _inspect_executable(contract: WorkerContract, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    relative = arguments.get("relative")
    parts = _relative_parts(relative, "executable relative path")
    if parts[0] not in {"build", "bundles"}:
        raise RemoteWorkerError("executable inspection is restricted to build/ or bundles/")
    path = _safe_path(contract, relative, "executable")
    details = _lstat_no_symlink(path, "executable")
    if not stat.S_ISREG(details.st_mode) or not details.st_mode & stat.S_IXUSR:
        raise RemoteWorkerError("runtime artifact is not an owner-executable regular file")
    digest, size = file_sha256(path)
    if size <= 0 or size > MAX_EPHEMERAL_RUNTIME_BYTES - MAX_EPHEMERAL_MANIFEST_BYTES:
        raise RemoteWorkerError("runtime artifact exceeds the 64 MiB packet budget")
    expected_sha256 = _validate_sha256(arguments.get("sha256"), "executable SHA-256")
    if arguments.get("size") != size or expected_sha256 != digest:
        raise RemoteWorkerError("runtime artifact differs from its expected identity")
    file_stdout, _ = _bounded_tool_output(("/usr/bin/file", "-b", str(path)))
    arches_stdout, _ = _bounded_tool_output(("/usr/bin/lipo", "-archs", str(path)))
    if arches_stdout.split() != ["arm64"] or not (
        "Mach-O 64-bit executable arm64" in file_stdout
        or "Mach-O 64-bit bundle arm64" in file_stdout
    ):
        raise RemoteWorkerError("runtime artifact is not a thin arm64 Mach-O executable")
    verify_stdout, verify_stderr = _bounded_tool_output(
        ("/usr/bin/codesign", "--verify", "--strict", "--verbose=4", str(path))
    )
    details_stdout, details_stderr = _bounded_tool_output(
        ("/usr/bin/codesign", "-d", "--verbose=4", str(path))
    )
    requirement_stdout, requirement_stderr = _bounded_tool_output(
        ("/usr/bin/codesign", "-d", "-r-", str(path))
    )
    codesign_detail = "\n".join(value for value in (details_stdout, details_stderr) if value)
    normalized_codesign_detail = "\n".join(
        line for line in codesign_detail.splitlines() if not line.startswith("Executable=")
    )
    parsed = {}
    for line in codesign_detail.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"CDHash", "Identifier", "TeamIdentifier", "Signature"}:
            parsed[key] = value
    requirement_output = "\n".join(
        value for value in (requirement_stdout, requirement_stderr) if value
    )
    designated = _parse_designated_requirement(requirement_output)
    cdhash = parsed.get("CDHash", "")
    if (
        not cdhash
        or any(character not in "0123456789abcdefABCDEF" for character in cdhash)
        or not designated
    ):
        raise RemoteWorkerError("codesign detail omitted a valid CDHash or designated requirement")
    return {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.ephemeral-executable-inspection.v1",
        "relative": relative,
        "sha256": digest,
        "size": size,
        "mode": "%04o" % stat.S_IMODE(details.st_mode),
        "mach_o_arches": ["arm64"],
        "file_description": file_stdout,
        "codesign": {
            "verified": True,
            "strict": True,
            "cdhash": cdhash.lower(),
            "identifier": parsed.get("Identifier"),
            "team_identifier": parsed.get("TeamIdentifier"),
            "signature": parsed.get("Signature"),
            "designated_requirement": designated,
            "designated_requirement_sha256": hashlib.sha256(designated.encode("utf-8")).hexdigest(),
            "verify_output_sha256": hashlib.sha256(
                (verify_stdout + "\n" + verify_stderr).encode("utf-8")
            ).hexdigest(),
            "detail_output_sha256": hashlib.sha256(codesign_detail.encode("utf-8")).hexdigest(),
            "portable_detail_sha256": hashlib.sha256(
                normalized_codesign_detail.encode("utf-8")
            ).hexdigest(),
        },
    }


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise RemoteWorkerError("cannot inspect the owned process group") from error
    return True


def _wait_owned_process_group(
    process: subprocess.Popen[bytes], timeout_seconds: int
) -> Tuple[int, bool, bool]:
    """Wait once, then bound TERM/KILL and prove no descendant group survives."""

    timed_out = False
    descendants_reaped = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 124
    if timed_out or _process_group_exists(process.pid):
        descendants_reaped = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        if process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _process_group_exists(process.pid):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise RemoteWorkerError("run process leader resisted bounded reap") from error
            deadline = time.monotonic() + 5
            while _process_group_exists(process.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
        if _process_group_exists(process.pid):
            raise RemoteWorkerError("run process group survived bounded TERM/KILL")
    return return_code, timed_out, descendants_reaped


RUN_STATE_FIELDS = {
    "schema_id",
    "schema_version",
    "request_id",
    "semantic_sha256",
    "command_sha256",
    "run_id",
    "controller_mode",
    "cwd_relative",
    "argv_sha256",
    "output_relative",
    "tmp_relative",
    "build_relative",
    "cache_relative",
    "reservation_relative",
    "timeout_seconds",
    "max_run_bytes",
    "campaign_bytes_before",
    "storage_precommit",
    "started_unix_ms",
    "phase",
    "supervisor_pid",
    "supervisor_identity_sha256",
    "workload_pid",
    "workload_identity_sha256",
    "execution",
    "result",
    "state_sha256",
}


def _run_state_path(contract: WorkerContract, request_id: Any) -> Path:
    safe = _validate_identifier(request_id, "run request identifier")
    return _safe_path(
        contract,
        "control/run-states/%s.json" % safe,
        "run state",
        create_parents=True,
    )


def _run_supervisor_path(contract: WorkerContract, request_id: Any) -> Path:
    safe = _validate_identifier(request_id, "run request identifier")
    return _safe_path(
        contract,
        "control/run-supervisors/%s.json" % safe,
        "run supervisor configuration",
        create_parents=True,
    )


def _write_run_state(path: Path, document: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(document)
    value["state_sha256"] = document_sha256(value, "state_sha256")
    _atomic_write_recoverable(path, canonical_json(value), 0o600, "run state")
    return value


def _run_state_transition_lock(
    contract: WorkerContract, request_id: Any
) -> Tuple[int, Path]:
    safe = _validate_identifier(request_id, "run transition request")
    offset = int.from_bytes(
        hashlib.sha256(("run-state\x00" + safe).encode("ascii")).digest()[:8],
        "big",
    ) % ((1 << 63) - 1)
    path = _safe_path(
        contract,
        "control/locks/run-state-ranges.lock",
        "run transition lock",
        create_parents=True,
    )
    descriptor = os.open(
        str(path),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    details = os.fstat(descriptor)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid)
        != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RemoteWorkerError("run transition lock metadata is invalid")
    fcntl.lockf(descriptor, fcntl.LOCK_EX, 1, offset, os.SEEK_SET)
    return descriptor, path


def _transition_run_state(
    contract: WorkerContract,
    state_path: Path,
    request: Mapping[str, Any],
    expected: Mapping[str, Any],
    **changes: Any,
) -> Dict[str, Any]:
    lock_fd, _ = _run_state_transition_lock(contract, request["request_id"])
    try:
        current = _load_run_state(contract, state_path, request)
        if current["state_sha256"] != expected["state_sha256"]:
            raise RemoteWorkerError("run state transition lost its compare-and-swap")
        return _write_run_state(state_path, {**current, **changes})
    finally:
        os.close(lock_fd)


def _load_run_state(
    contract: WorkerContract,
    path: Path,
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    details = _lstat_no_symlink(path, "run state")
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid) != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_size > MAX_MANIFEST_BYTES
    ):
        raise RemoteWorkerError("run state metadata is invalid")
    state = _load_json_file(path)
    if (
        set(state) != RUN_STATE_FIELDS
        or state.get("schema_id") != RUN_STATE_SCHEMA
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("request_id") != request["request_id"]
        or state.get("semantic_sha256") != request["semantic_sha256"]
        or state.get("command_sha256") != request["command_sha256"]
        or state.get("state_sha256") != document_sha256(state, "state_sha256")
        or state.get("phase")
        not in {
            "reserved",
            "prepared",
            "supervisor-starting",
            "running",
            "recovering",
            "completed",
            "finalized",
        }
        or not isinstance(state.get("controller_mode"), bool)
        or not isinstance(state.get("timeout_seconds"), int)
        or not isinstance(state.get("max_run_bytes"), int)
        or isinstance(state.get("max_run_bytes"), bool)
        or not 1 <= state.get("max_run_bytes", 0) <= MAX_RUN_BYTES
        or not isinstance(state.get("campaign_bytes_before"), int)
        or not isinstance(state.get("storage_precommit"), dict)
        or not isinstance(state.get("started_unix_ms"), int)
        or not isinstance(state.get("run_id"), str)
        or not isinstance(state.get("argv_sha256"), str)
        or SHA256.fullmatch(state["argv_sha256"]) is None
        or (state.get("execution") is not None and not isinstance(state["execution"], dict))
        or (state.get("result") is not None and not isinstance(state["result"], dict))
    ):
        raise RemoteWorkerError("run state identity is invalid")
    for field in (
        "output_relative",
        "tmp_relative",
        "build_relative",
        "cache_relative",
        "reservation_relative",
    ):
        _relative_parts(state.get(field), "run state %s" % field)
    if state["phase"] == "finalized" and state["result"] is None:
        raise RemoteWorkerError("finalized run state omits its result")
    return state


def _matching_run_recovery_under_pressure(
    contract: WorkerContract, request: Mapping[str, Any]
) -> bool:
    if request.get("operation") not in {"run-command", "run-controller"}:
        return False
    try:
        state = _load_run_state(
            contract,
            _run_state_path(contract, request["request_id"]),
            request,
        )
    except (FileNotFoundError, RemoteWorkerError):
        return False
    if state["phase"] == "finalized":
        return False
    reservation = _data_reservation_path(contract, request["request_id"])
    try:
        _load_data_reservation(contract, reservation, request)
    except (FileNotFoundError, RemoteWorkerError):
        return False
    return True


def _process_identity_sha256(pid: int) -> Optional[str]:
    if not isinstance(pid, int) or pid <= 0:
        return None
    completed = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "lstart="],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return hashlib.sha256(str(pid).encode("ascii") + b"\x00" + completed.stdout).hexdigest()


def _terminate_exact_process_group(
    pid: Optional[int],
    identity_sha256: Optional[str],
    label: str,
    *,
    marker: Optional[str] = None,
) -> bool:
    if pid is None or identity_sha256 is None:
        return False
    observed = _process_identity_sha256(pid)
    if observed is None:
        if not _process_group_exists(pid):
            return False
        if marker is None:
            raise RemoteWorkerError("%s leader vanished while its PGID remains" % label)
        completed = subprocess.run(
            ["/bin/ps", "eww", "-axo", "pgid=,command="],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
        needle = ("CASCADIA_R2_RUN_MARKER=" + marker).encode("utf-8")
        members = []
        for line in completed.stdout.splitlines():
            group, separator, command = line.strip().partition(b" ")
            if separator and group.isdigit() and int(group) == pid:
                members.append(command)
        if completed.returncode != 0 or not members or not any(
            needle in command for command in members
        ):
            raise RemoteWorkerError("%s PGID may have been reused" % label)
    elif observed != identity_sha256:
        raise RemoteWorkerError("%s PID identity was reused" % label)
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while _process_group_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(pid):
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while _process_group_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _process_group_exists(pid):
        raise RemoteWorkerError("%s survived bounded TERM/KILL" % label)
    return True


RUN_SUPERVISOR_FIELDS = {
    "schema_id",
    "schema_version",
    "token_sha256",
    "parent_pid",
    "request",
    "root",
    "expected_uid",
    "expected_gid",
    "expected_root_device",
    "expected_root_inode",
    "expected_identity_sha256",
    "worker_sha256",
    "state_relative",
    "argv",
    "cwd_relative",
    "environment",
    "sandbox_profile",
    "timeout_seconds",
    "max_run_bytes",
    "min_free_bytes",
    "config_sha256",
}


def _load_run_supervisor_config(
    path: Path, authorization: bytes
) -> Tuple[WorkerContract, Dict[str, Any]]:
    details = _lstat_no_symlink(path, "run supervisor configuration")
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) != 0o400
        or details.st_size > MAX_REQUEST_BYTES
    ):
        raise RemoteWorkerError("run supervisor configuration metadata is invalid")
    config = _load_json_file(path)
    if (
        set(config) != RUN_SUPERVISOR_FIELDS
        or config.get("schema_id") != RUN_SUPERVISOR_SCHEMA
        or config.get("schema_version") != SCHEMA_VERSION
        or len(authorization) != 32
        or config.get("token_sha256") != hashlib.sha256(authorization).hexdigest()
        or config.get("config_sha256") != document_sha256(config, "config_sha256")
        or not isinstance(config.get("parent_pid"), int)
        or not isinstance(config.get("request"), dict)
        or config.get("worker_sha256") != config["request"].get("worker_sha256")
        or SHA256.fullmatch(str(config.get("worker_sha256"))) is None
        or not isinstance(config.get("argv"), list)
        or not config["argv"]
        or not all(isinstance(value, str) for value in config["argv"])
        or not isinstance(config.get("environment"), dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in config["environment"].items()
        )
        or not isinstance(config.get("sandbox_profile"), str)
        or not isinstance(config.get("timeout_seconds"), int)
        or not isinstance(config.get("max_run_bytes"), int)
        or config.get("max_run_bytes", 0) <= 0
        or not isinstance(config.get("min_free_bytes"), int)
    ):
        raise RemoteWorkerError("run supervisor configuration identity is invalid")
    root = Path(str(config["root"]))
    root_details = _lstat_no_symlink(root, "run supervisor root")
    if (
        root_details.st_uid != config["expected_uid"]
        or root_details.st_gid != config["expected_gid"]
        or root_details.st_dev != config["expected_root_device"]
        or root_details.st_ino != config["expected_root_inode"]
        or details.st_uid != config["expected_uid"]
        or details.st_gid != config["expected_gid"]
        or details.st_dev != config["expected_root_device"]
    ):
        raise RemoteWorkerError("run supervisor root identity differs")
    contract = WorkerContract(
        root=root,
        expected_host=PRODUCTION_HOST,
        expected_user=PRODUCTION_USER,
        expected_uid=config["expected_uid"],
        expected_gid=config["expected_gid"],
        expected_root_device=config["expected_root_device"],
        expected_root_inode=config["expected_root_inode"],
        expected_identity_sha256=config["expected_identity_sha256"],
        min_free_bytes=config["min_free_bytes"],
        max_campaign_bytes=max(config["max_run_bytes"] * 4, config["max_run_bytes"] + 1),
        max_data_bytes=max(config["max_run_bytes"] * 4 - 1, config["max_run_bytes"]),
        receipt_budget_bytes=1,
        max_receipt_bytes=1,
        max_receipt_entries=1,
    )
    expected_path = _run_supervisor_path(contract, config["request"].get("request_id"))
    if path != expected_path:
        raise RemoteWorkerError("run supervisor configuration path differs")
    worker_payload = Path(__file__).read_bytes()
    if hashlib.sha256(worker_payload).hexdigest() != config["worker_sha256"]:
        raise RemoteWorkerError("run supervisor executable differs from the frozen worker")
    return contract, config


def _run_tree_bytes(contract: WorkerContract, state: Mapping[str, Any]) -> int:
    total = 0
    for field in ("tmp_relative", "build_relative", "cache_relative"):
        path = _safe_path(contract, state[field], "run monitor tree")
        if path.exists():
            total += _apparent_size(
                path,
                allowed_symlink_prefixes=(path,),
                diagnostic_root=contract.root,
            )
    return total


def _run_supervisor(config_path: Path, authorization_fd: int) -> int:
    """Persistent watchdog that owns one workload after its SSH worker exits."""
    authorization = bytearray()
    try:
        while len(authorization) < 33:
            chunk = os.read(authorization_fd, 33 - len(authorization))
            if not chunk:
                break
            authorization.extend(chunk)
    finally:
        os.close(authorization_fd)
    contract, config = _load_run_supervisor_config(config_path, bytes(authorization))
    request = config["request"]
    state_path = _safe_path(contract, config["state_relative"], "run state")
    state = _load_run_state(contract, state_path, request)
    if state["phase"] != "prepared":
        raise RemoteWorkerError("run supervisor requires one prepared intent")
    supervisor_pid = os.getpid()
    supervisor_identity = _process_identity_sha256(supervisor_pid)
    if supervisor_identity is None or os.getpgrp() != supervisor_pid:
        raise RemoteWorkerError("run supervisor does not own a fresh process group")
    state = _transition_run_state(
        contract,
        state_path,
        request,
        state,
        phase="supervisor-starting",
        supervisor_pid=supervisor_pid,
        supervisor_identity_sha256=supervisor_identity,
    )
    cwd = _safe_path(contract, config["cwd_relative"], "supervisor run cwd")
    run_tmp = _safe_path(contract, state["tmp_relative"], "supervisor run tmp")
    stdout_path = run_tmp / "stdout.log"
    stderr_path = run_tmp / "stderr.log"
    started = time.monotonic_ns()
    resource_exceeded = False
    resource_reason: Optional[str] = None
    timed_out = False
    descendants_reaped = False
    monitor_samples = 0
    aggregate_monitor_samples = 0
    aggregate_margin = min(512 * (1 << 20), max(1, config["max_run_bytes"] // 8))
    next_aggregate_scan = 0.0
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        os.chmod(stdout_path, 0o400, follow_symlinks=False)
        os.chmod(stderr_path, 0o400, follow_symlinks=False)
        gate_read, gate_write = os.pipe()
        gate_token = os.urandom(32)
        workload_pid = os.fork()
        if workload_pid == 0:
            try:
                os.close(gate_write)
                os.setsid()
                received = bytearray()
                while len(received) < 33:
                    chunk = os.read(gate_read, 33 - len(received))
                    if not chunk:
                        break
                    received.extend(chunk)
                os.close(gate_read)
                if bytes(received) != gate_token:
                    os._exit(126)
                null_fd = os.open("/dev/null", os.O_RDONLY)
                os.dup2(null_fd, 0)
                os.close(null_fd)
                os.dup2(stdout.fileno(), 1)
                os.dup2(stderr.fileno(), 2)
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (config["max_run_bytes"], config["max_run_bytes"]),
                )
                os.chdir(cwd)
                os.execve(
                    "/usr/bin/sandbox-exec",
                    [
                        "/usr/bin/sandbox-exec",
                        "-p",
                        config["sandbox_profile"],
                        *config["argv"],
                    ],
                    config["environment"],
                )
            except BaseException as error:
                with suppress(OSError):
                    os.write(
                        stderr.fileno(),
                        ("run gate/exec failure: %s: %s\n" % (type(error).__name__, error)).encode(
                            "utf-8", errors="replace"
                        )[:4096],
                    )
                os._exit(127)
        os.close(gate_read)
        workload_identity = _process_identity_sha256(workload_pid)
        if workload_identity is None:
            os.close(gate_write)
            with suppress(ProcessLookupError):
                os.killpg(workload_pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                os.waitpid(workload_pid, 0)
            raise RemoteWorkerError("run workload identity was not observable")
        try:
            state = _transition_run_state(
                contract,
                state_path,
                request,
                state,
                phase="running",
                workload_pid=workload_pid,
                workload_identity_sha256=workload_identity,
            )
            view = memoryview(gate_token)
            while view:
                written = os.write(gate_write, view)
                if written <= 0:
                    raise RemoteWorkerError("run workload gate write made no progress")
                view = view[written:]
        finally:
            os.close(gate_write)
        deadline = time.monotonic() + config["timeout_seconds"]
        child_status: Optional[int] = None
        while child_status is None:
            observed_pid, observed_status = os.waitpid(workload_pid, os.WNOHANG)
            if observed_pid == workload_pid:
                child_status = observed_status
                break
            monitor_samples += 1
            run_bytes = os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size
            free_bytes = shutil.disk_usage(str(contract.root)).free
            if run_bytes > config["max_run_bytes"]:
                resource_exceeded = True
                resource_reason = "stream_limit"
                break
            if free_bytes < config["min_free_bytes"]:
                resource_exceeded = True
                resource_reason = "free_space_floor"
                break
            now = time.monotonic()
            if now >= next_aggregate_scan:
                aggregate_monitor_samples += 1
                aggregate_bytes = _run_tree_bytes(contract, state)
                if aggregate_bytes > config["max_run_bytes"] - aggregate_margin:
                    resource_exceeded = True
                    resource_reason = "aggregate_limit"
                    break
                ratio = aggregate_bytes / config["max_run_bytes"]
                interval = 0.25 if ratio >= 0.75 else 1.0 if ratio >= 0.5 else 5.0
                next_aggregate_scan = now + interval
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.25)
        if resource_exceeded or timed_out:
            with suppress(ProcessLookupError):
                os.killpg(workload_pid, signal.SIGTERM)
            stop_deadline = time.monotonic() + 5
            while child_status is None and time.monotonic() < stop_deadline:
                observed_pid, observed_status = os.waitpid(workload_pid, os.WNOHANG)
                if observed_pid == workload_pid:
                    child_status = observed_status
                    break
                time.sleep(0.05)
            if child_status is None:
                with suppress(ProcessLookupError):
                    os.killpg(workload_pid, signal.SIGKILL)
                _observed_pid, child_status = os.waitpid(workload_pid, 0)
            descendants_reaped = True
            exit_code = 125 if resource_exceeded else 124
        else:
            assert child_status is not None
            if os.WIFEXITED(child_status):
                exit_code = os.WEXITSTATUS(child_status)
            elif os.WIFSIGNALED(child_status):
                exit_code = 128 + os.WTERMSIG(child_status)
            else:
                exit_code = 126
        if _process_group_exists(workload_pid):
            descendants_reaped = True
            with suppress(ProcessLookupError):
                os.killpg(workload_pid, signal.SIGTERM)
            group_deadline = time.monotonic() + 5
            while _process_group_exists(workload_pid) and time.monotonic() < group_deadline:
                time.sleep(0.05)
            if _process_group_exists(workload_pid):
                with suppress(ProcessLookupError):
                    os.killpg(workload_pid, signal.SIGKILL)
            group_deadline = time.monotonic() + 5
            while _process_group_exists(workload_pid) and time.monotonic() < group_deadline:
                time.sleep(0.05)
        if _process_group_exists(workload_pid):
            raise RemoteWorkerError("run workload group survived supervisor completion")
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    stdout_sha, stdout_size = file_sha256(stdout_path)
    stderr_sha, stderr_size = file_sha256(stderr_path)
    run_bytes = _run_tree_bytes(contract, state)
    execution = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "resource_exceeded": resource_exceeded,
        "resource_reason": resource_reason,
        "descendants_reaped": descendants_reaped,
        "duration_ms": (time.monotonic_ns() - started) // 1_000_000,
        "run_bytes": run_bytes,
        "monitor_samples": monitor_samples,
        "aggregate_monitor_samples": aggregate_monitor_samples,
        "aggregate_margin_bytes": aggregate_margin,
        "stdout_sha256": stdout_sha,
        "stdout_size": stdout_size,
        "stderr_sha256": stderr_sha,
        "stderr_size": stderr_size,
    }
    _transition_run_state(
        contract,
        state_path,
        request,
        state,
        phase="completed",
        execution=execution,
    )
    return 0


def _ensure_run_log(path: Path) -> None:
    try:
        details = _lstat_no_symlink(path, "run log")
    except FileNotFoundError:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
        return
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) != 0o400
    ):
        raise RemoteWorkerError("run log metadata is invalid")


def _complete_interrupted_run(
    contract: WorkerContract,
    state_path: Path,
    state: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    resource_exceeded: bool = False,
) -> Dict[str, Any]:
    lock_fd, _ = _run_state_transition_lock(contract, request["request_id"])
    try:
        current = _load_run_state(contract, state_path, request)
        if current["phase"] in {"completed", "finalized"}:
            return current
        if current["state_sha256"] != state["state_sha256"]:
            state = current
        workload_reaped = _terminate_exact_process_group(
            state.get("workload_pid"),
            state.get("workload_identity_sha256"),
            "orphan run workload",
            marker=request["request_id"],
        )
        supervisor_reaped = _terminate_exact_process_group(
            state.get("supervisor_pid"),
            state.get("supervisor_identity_sha256"),
            "orphan run supervisor",
            marker=request["request_id"],
        )
        run_tmp = _safe_path(contract, state["tmp_relative"], "interrupted run tmp")
        if not run_tmp.exists():
            os.mkdir(str(run_tmp), 0o700)
            _fsync_directory(run_tmp.parent)
        stdout_path = run_tmp / "stdout.log"
        stderr_path = run_tmp / "stderr.log"
        _ensure_run_log(stdout_path)
        _ensure_run_log(stderr_path)
        stdout_sha, stdout_size = file_sha256(stdout_path)
        stderr_sha, stderr_size = file_sha256(stderr_path)
        run_bytes = _run_tree_bytes(contract, state)
        execution = {
            "exit_code": 125 if resource_exceeded else 126,
            "timed_out": False,
            "resource_exceeded": resource_exceeded,
            "resource_reason": (
                "recovery_deadline" if resource_exceeded else None
            ),
            "descendants_reaped": workload_reaped or supervisor_reaped,
            "duration_ms": max(
                0,
                time.time_ns() // 1_000_000 - int(state["started_unix_ms"]),
            ),
            "run_bytes": run_bytes,
            "monitor_samples": 0,
            "stdout_sha256": stdout_sha,
            "stdout_size": stdout_size,
            "stderr_sha256": stderr_sha,
            "stderr_size": stderr_size,
            "supervisor_interrupted": True,
            "workload_started": state.get("workload_pid") is not None,
        }
        return _write_run_state(
            state_path,
            {**state, "phase": "completed", "execution": execution},
        )
    finally:
        os.close(lock_fd)


def _await_run_completion(
    contract: WorkerContract,
    state_path: Path,
    request: Mapping[str, Any],
    *,
    spawned_supervisor: Optional[subprocess.Popen[bytes]] = None,
) -> Dict[str, Any]:
    initial = _load_run_state(contract, state_path, request)
    deadline = time.monotonic() + initial["timeout_seconds"] + 30
    prepared_grace = time.monotonic() + 5
    while True:
        state = _load_run_state(contract, state_path, request)
        if state["phase"] in {"completed", "finalized"}:
            if spawned_supervisor is not None:
                with suppress(subprocess.TimeoutExpired):
                    spawned_supervisor.wait(timeout=5)
            return state
        supervisor_pid = state.get("supervisor_pid")
        supervisor_identity = state.get("supervisor_identity_sha256")
        supervisor_alive = (
            isinstance(supervisor_pid, int)
            and isinstance(supervisor_identity, str)
            and _process_identity_sha256(supervisor_pid) == supervisor_identity
        )
        if (
            state["phase"] == "prepared"
            and time.monotonic() < prepared_grace
            and (spawned_supervisor is None or spawned_supervisor.poll() is None)
        ):
            time.sleep(0.05)
            continue
        if supervisor_alive and time.monotonic() < deadline:
            time.sleep(0.25)
            continue
        if supervisor_alive:
            _terminate_exact_process_group(
                supervisor_pid,
                supervisor_identity,
                "expired run supervisor",
                marker=request["request_id"],
            )
        return _complete_interrupted_run(
            contract,
            state_path,
            state,
            request,
            resource_exceeded=time.monotonic() >= deadline,
        )


def _verify_run_log(
    contract: WorkerContract,
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> None:
    details = _lstat_no_symlink(path, "run output log")
    digest, size = file_sha256(path)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid)
        != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o400
        or digest != expected_sha256
        or size != expected_size
    ):
        raise RemoteWorkerError("run output log identity differs")


def _cleanup_run_supervisor_config(
    contract: WorkerContract, request_id: str
) -> None:
    supervisor_path = _run_supervisor_path(contract, request_id)
    try:
        details = _lstat_no_symlink(supervisor_path, "run supervisor configuration")
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_dev != contract.expected_root_device
        or (details.st_uid, details.st_gid)
        != (contract.expected_uid, contract.expected_gid)
        or stat.S_IMODE(details.st_mode) != 0o400
    ):
        raise RemoteWorkerError("run supervisor configuration cleanup is unsafe")
    os.unlink(str(supervisor_path))
    _fsync_directory(supervisor_path.parent)


def _finalize_run_state(
    contract: WorkerContract,
    state_path: Path,
    state: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    if state["phase"] == "finalized":
        result = dict(state["result"])
        output = _safe_path(contract, state["output_relative"], "finalized run output")
        _verify_run_log(
            contract,
            output / "stdout.log",
            result["stdout_sha256"],
            result["stdout_size"],
        )
        _verify_run_log(
            contract,
            output / "stderr.log",
            result["stderr_sha256"],
            result["stderr_size"],
        )
        reservation = _data_reservation_path(contract, request["request_id"])
        if reservation.exists():
            _release_matching_data_reservation(contract, request, reservation)
        _cleanup_run_supervisor_config(contract, request["request_id"])
        return result
    if state["phase"] != "completed" or not isinstance(state.get("execution"), dict):
        raise RemoteWorkerError("run is not complete enough to finalize")
    execution = state["execution"]
    required_execution = {
        "exit_code",
        "timed_out",
        "resource_exceeded",
        "descendants_reaped",
        "duration_ms",
        "run_bytes",
        "monitor_samples",
        "stdout_sha256",
        "stdout_size",
        "stderr_sha256",
        "stderr_size",
    }
    if not required_execution.issubset(execution) or (
        execution["run_bytes"] > state["max_run_bytes"]
        and not execution.get("resource_exceeded")
    ):
        raise RemoteWorkerError("run execution result is invalid or over budget")
    run_tmp = _safe_path(contract, state["tmp_relative"], "run tmp")
    output = _safe_path(
        contract, state["output_relative"], "run output", create_parents=True
    )
    output_staging = output.parent / (
        ".%s.%s.run-output-staging" % (output.name, request["request_id"])
    )
    if output.exists() and output_staging.exists():
        raise RemoteWorkerError("run output and its staging directory both exist")
    if not output.exists():
        if not output_staging.exists():
            os.mkdir(str(output_staging), 0o700)
            _fsync_directory(output_staging.parent)
        else:
            _validate_transaction_cleanup_tree(
                contract, output_staging, "run output staging"
            )
        for name in ("stdout", "stderr"):
            source = run_tmp / (name + ".log")
            destination = output_staging / (name + ".log")
            digest = execution[name + "_sha256"]
            size = execution[name + "_size"]
            if destination.exists():
                _verify_run_log(contract, destination, digest, size)
            else:
                _verify_run_log(contract, source, digest, size)
                os.rename(str(source), str(destination))
                _fsync_directory(output_staging)
        os.rename(str(output_staging), str(output))
        _fsync_directory(output.parent)
    _verify_run_log(
        contract,
        output / "stdout.log",
        execution["stdout_sha256"],
        execution["stdout_size"],
    )
    _verify_run_log(
        contract,
        output / "stderr.log",
        execution["stderr_sha256"],
        execution["stderr_size"],
    )
    capacity_lock, _ = _global_lock(contract)
    try:
        reservation = _data_reservation_path(contract, request["request_id"])
        reservation_document = _load_data_reservation(contract, reservation, request)
        if execution.get("resource_exceeded"):
            storage_before_output = {
                **state["storage_precommit"],
                "resource_exceeded_at_finalize": True,
            }
        else:
            storage_before_output = _storage_capacity_state(contract)
            _require_mutation_headroom(
                contract,
                storage_before_output,
                incoming_data_bytes=0,
                data_reservation_credit=reservation_document["reserved_bytes"],
            )
    finally:
        os.close(capacity_lock)
    resource_cleanup: List[Dict[str, Any]] = []
    if run_tmp.exists():
        if execution.get("resource_exceeded"):
            inventory = _failed_run_tree_inventory(contract, run_tmp)
            _remove_failed_run_tree(run_tmp)
            resource_cleanup.append({"tree": "tmp", **inventory})
        else:
            _remove_verified_tree(
                run_tmp,
                allow_contained_symlinks=True,
                diagnostic_root=contract.root,
            )
    if not execution.get("workload_started", True):
        for field in ("build_relative", "cache_relative"):
            partial = _safe_path(contract, state[field], "unlaunched run partial")
            if partial.exists():
                inventory = _failed_run_tree_inventory(contract, partial)
                _remove_failed_run_tree(partial)
                resource_cleanup.append({"tree": field.removesuffix("_relative"), **inventory})
    if execution.get("resource_exceeded"):
        for field in ("build_relative", "cache_relative"):
            pressure_tree = _safe_path(
                contract, state[field], "resource-exceeded run cleanup"
            )
            if pressure_tree.exists():
                inventory = _failed_run_tree_inventory(contract, pressure_tree)
                _remove_failed_run_tree(pressure_tree)
                resource_cleanup.append({"tree": field.removesuffix("_relative"), **inventory})
        capacity_lock, _ = _global_lock(contract)
        try:
            storage_before_output = _storage_capacity_state(contract)
            reservation = _data_reservation_path(contract, request["request_id"])
            reservation_document = _load_data_reservation(
                contract, reservation, request
            )
            _require_mutation_headroom(
                contract,
                storage_before_output,
                incoming_data_bytes=0,
                data_reservation_credit=reservation_document["reserved_bytes"],
            )
        finally:
            os.close(capacity_lock)
    campaign_bytes_after = _campaign_apparent_size(contract)
    usage = shutil.disk_usage(str(contract.root))
    if campaign_bytes_after > contract.max_campaign_bytes or usage.free < contract.min_free_bytes:
        raise RemoteWorkerError("remote run violated the campaign capacity reserve")
    result = {
        "run_id": state["run_id"],
        "cwd_relative": state["cwd_relative"],
        "argv_sha256": state["argv_sha256"],
        "output_relative": state["output_relative"],
        "exit_code": execution["exit_code"],
        "timed_out": execution["timed_out"],
        "resource_exceeded": execution["resource_exceeded"],
        "resource_reason": execution.get("resource_reason"),
        "descendants_reaped": execution["descendants_reaped"],
        "supervisor_interrupted": execution.get("supervisor_interrupted", False),
        "duration_ms": execution["duration_ms"],
        "run_bytes": execution["run_bytes"],
        "max_run_bytes": state["max_run_bytes"],
        "campaign_bytes_before": state["campaign_bytes_before"],
        "campaign_bytes_after": campaign_bytes_after,
        "campaign_bytes_delta": campaign_bytes_after - state["campaign_bytes_before"],
        "storage_precommit": state["storage_precommit"],
        "storage_before_output": storage_before_output,
        "free_bytes_after": usage.free,
        "temporary_cleaned": True,
        "controller_mode": state["controller_mode"],
        "stdout_sha256": execution["stdout_sha256"],
        "stdout_size": execution["stdout_size"],
        "stderr_sha256": execution["stderr_sha256"],
        "stderr_size": execution["stderr_size"],
        "monitor_samples": execution["monitor_samples"],
        "aggregate_monitor_samples": execution.get("aggregate_monitor_samples", 0),
        "aggregate_margin_bytes": execution.get("aggregate_margin_bytes", 0),
        "resource_cleanup": resource_cleanup,
    }
    finalized = _transition_run_state(
        contract,
        state_path,
        request,
        state,
        phase="finalized",
        result=result,
    )
    capacity_lock, _ = _global_lock(contract)
    try:
        _release_matching_data_reservation(
            contract,
            request,
            _data_reservation_path(contract, request["request_id"]),
        )
    finally:
        os.close(capacity_lock)
    _cleanup_run_supervisor_config(contract, request["request_id"])
    return dict(finalized["result"])


def _run_command(
    contract: WorkerContract,
    arguments: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    controller: bool = False,
) -> Dict[str, Any]:
    run_id = _validate_identifier(arguments.get("run_id"), "run_id")
    cwd_relative = arguments.get("cwd_relative")
    cwd = _safe_path(contract, cwd_relative, "run cwd")
    cwd_details = _lstat_no_symlink(cwd, "run cwd")
    if not stat.S_ISDIR(cwd_details.st_mode):
        raise RemoteWorkerError("run cwd is not a directory")
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not argv or len(argv) > 256:
        raise RemoteWorkerError("run argv is invalid")
    executable = argv[0]
    if not isinstance(executable, str) or not executable.startswith("/"):
        raise RemoteWorkerError("run executable must be absolute")
    executable_path = Path(executable)
    if executable not in SAFE_SYSTEM_EXECUTABLES:
        relative_executable = _validate_absolute_under_root(
            contract, executable, "run executable"
        ).relative_to(contract.root)
        executable_path = _safe_path(contract, relative_executable.as_posix(), "run executable")
        details = _lstat_no_symlink(executable_path, "run executable")
        if not stat.S_ISREG(details.st_mode) or not details.st_mode & stat.S_IXUSR:
            raise RemoteWorkerError("root-contained run executable is not executable")
    controller_source = None
    if controller:
        if executable in SAFE_SYSTEM_EXECUTABLES:
            raise RemoteWorkerError("controller must be a frozen root-contained executable")
        controller_source = _validate_controller_source(
            contract, executable_path, cwd, arguments, argv
        )
    validated_argv = [
        executable,
        *(_validate_run_argument(contract, cwd, value) for value in argv[1:]),
    ]
    user_environment = arguments.get("environment", {})
    if not isinstance(user_environment, dict) or any(
        key not in SAFE_ENVIRONMENT_KEYS or not isinstance(value, str)
        for key, value in user_environment.items()
    ):
        raise RemoteWorkerError("run environment contains an unauthorized key/value")
    if any(
        len(value.encode("utf-8")) > 4096
        or "/" in value
        or ".." in value
        or "@" in value
        or "\x00" in value
        for value in user_environment.values()
    ):
        raise RemoteWorkerError("run environment value could redirect output outside the root")
    python_path_relatives = arguments.get("python_path_relatives", [])
    if (
        not isinstance(python_path_relatives, list)
        or len(python_path_relatives) > 8
        or any(not isinstance(value, str) for value in python_path_relatives)
    ):
        raise RemoteWorkerError("run Python path list is invalid")
    python_paths = []
    for relative in python_path_relatives:
        path = _safe_path(contract, relative, "run Python path")
        if not stat.S_ISDIR(_lstat_no_symlink(path, "run Python path").st_mode):
            raise RemoteWorkerError("run Python path is not a directory")
        python_paths.append(str(path))
        if controller_source is not None:
            try:
                path.relative_to(controller_source)
            except ValueError as error:
                raise RemoteWorkerError(
                    "controller Python path escapes its source freeze"
                ) from error
    if controller_source is not None and not python_paths:
        raise RemoteWorkerError("controller requires a Python path inside its source freeze")
    timeout_seconds = arguments.get("timeout_seconds")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 86400:
        raise RemoteWorkerError("run timeout is invalid")
    run_max_bytes = arguments.get("_test_max_run_bytes", MAX_RUN_BYTES)
    if (
        not isinstance(run_max_bytes, int)
        or isinstance(run_max_bytes, bool)
        or not 1 <= run_max_bytes <= MAX_RUN_BYTES
        or (
            "_test_max_run_bytes" in arguments
            and contract.root == PRODUCTION_ROOT
        )
    ):
        raise RemoteWorkerError("run byte ceiling is invalid")
    output_relative = arguments.get("output_relative")
    output_parts = _relative_parts(output_relative, "run output")
    if output_parts[0] not in RUN_OUTPUT_TOP_LEVELS:
        raise RemoteWorkerError("run output top-level is unauthorized")
    if controller and output_parts[:2] != ("reports", "controller-runs"):
        raise RemoteWorkerError("controller output must be below reports/controller-runs/")
    run_tmp = _safe_path(contract, "tmp/run-%s" % run_id, "run tmp")
    run_build = _safe_path(contract, "build/run-%s" % run_id, "run build")
    run_cache = _safe_path(contract, "cache/runs/run-%s" % run_id, "run cache")
    environment = {
        "HOME": str(run_cache / "home"),
        "TMPDIR": str(run_tmp),
        "CARGO_HOME": str(run_cache / "cargo-home"),
        "RUSTUP_HOME": str(run_cache / "rustup"),
        "CARGO_TARGET_DIR": str(run_build / "cargo-target"),
        "UV_CACHE_DIR": str(run_cache / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(run_cache / "uv-python"),
        "UV_TOOL_DIR": str(run_cache / "uv-tools"),
        "UV_TOOL_BIN_DIR": str(run_cache / "bin"),
        "UV_PROJECT_ENVIRONMENT": str(run_build / "uv-project-environment"),
        "PYTHONPYCACHEPREFIX": str(run_cache / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "%s:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        % (contract.root / "toolchains/bin"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.update(user_environment)
    environment["CASCADIA_R2_RUN_MARKER"] = request["request_id"]
    sandbox_profile = "\n".join(
        [
            "(version 1)",
            "(allow default)",
            *(["(deny network*)"] if controller else []),
            "(deny file-write*)",
            '(allow file-write* (literal "/dev/null"))',
            '(allow file-write* (subpath "%s"))' % run_tmp,
            '(allow file-write* (subpath "%s"))' % run_build,
            '(allow file-write* (subpath "%s"))' % run_cache,
            *(
                ['(allow file-write* (subpath "%s"))' % (contract.root / "control")]
                if controller
                else []
            ),
        ]
    )
    state_path = _run_state_path(contract, request["request_id"])
    argv_sha256 = hashlib.sha256(canonical_json(validated_argv)).hexdigest()
    try:
        existing_state = _load_run_state(contract, state_path, request)
    except FileNotFoundError:
        existing_state = None
    if existing_state is not None:
        if (
            existing_state["run_id"] != run_id
            or existing_state["controller_mode"] != controller
            or existing_state["cwd_relative"] != cwd_relative
            or existing_state["argv_sha256"] != argv_sha256
            or existing_state["output_relative"] != output_relative
            or existing_state["timeout_seconds"] != timeout_seconds
            or existing_state["max_run_bytes"] != run_max_bytes
        ):
            raise RemoteWorkerError("run state differs from the exact request")
        recovered = _await_run_completion(contract, state_path, request)
        return _finalize_run_state(contract, state_path, recovered, request)

    output = _safe_path(contract, output_relative, "run output", create_parents=True)
    output_staging = output.parent / (
        ".%s.%s.run-output-staging" % (output.name, request["request_id"])
    )
    if (
        any(path.exists() or path.is_symlink() for path in (run_tmp, run_build, run_cache))
        or output.exists()
        or output_staging.exists()
        or output_staging.is_symlink()
    ):
        raise RemoteWorkerError("run namespace already exists without a durable intent")
    run_reservation: Optional[Path] = None
    state_created = False
    try:
        capacity_lock, _ = _global_lock(contract)
        try:
            run_storage_precommit = _storage_capacity_state(contract)
            run_reservation = _reserve_data_capacity_locked(
                contract,
                request,
                run_storage_precommit,
                reserved_payload_bytes=run_max_bytes,
                target_relative=output_relative,
            )
        finally:
            os.close(capacity_lock)
        state = _write_run_state(
            state_path,
            {
                "schema_id": RUN_STATE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "request_id": request["request_id"],
                "semantic_sha256": request["semantic_sha256"],
                "command_sha256": request["command_sha256"],
                "run_id": run_id,
                "controller_mode": controller,
                "cwd_relative": cwd_relative,
                "argv_sha256": argv_sha256,
                "output_relative": output_relative,
                "tmp_relative": run_tmp.relative_to(contract.root).as_posix(),
                "build_relative": run_build.relative_to(contract.root).as_posix(),
                "cache_relative": run_cache.relative_to(contract.root).as_posix(),
                "reservation_relative": run_reservation.relative_to(contract.root).as_posix(),
                "timeout_seconds": timeout_seconds,
                "max_run_bytes": run_max_bytes,
                "campaign_bytes_before": run_storage_precommit[
                    "campaign_apparent_bytes"
                ],
                "storage_precommit": run_storage_precommit,
                "started_unix_ms": time.time_ns() // 1_000_000,
                "phase": "reserved",
                "supervisor_pid": None,
                "supervisor_identity_sha256": None,
                "workload_pid": None,
                "workload_identity_sha256": None,
                "execution": None,
                "result": None,
            },
        )
        state_created = True
        for path in (run_tmp, run_build, run_cache):
            os.mkdir(str(path), 0o700)
            _fsync_directory(path.parent)
        state = _transition_run_state(
            contract,
            state_path,
            request,
            state,
            phase="prepared",
        )
    except BaseException:
        if not state_created:
            for path in (run_tmp, run_build, run_cache):
                if path.exists():
                    _remove_verified_tree(
                        path,
                        allow_contained_symlinks=True,
                        diagnostic_root=contract.root,
                    )
            capacity_lock, _ = _global_lock(contract)
            try:
                _release_matching_data_reservation(contract, request, run_reservation)
            finally:
                os.close(capacity_lock)
        raise

    supervisor_path = _run_supervisor_path(contract, request["request_id"])
    supervisor_authorization = os.urandom(32)
    config = {
        "schema_id": RUN_SUPERVISOR_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "token_sha256": hashlib.sha256(supervisor_authorization).hexdigest(),
        "parent_pid": os.getpid(),
        "request": dict(request),
        "root": str(contract.root),
        "expected_uid": contract.expected_uid,
        "expected_gid": contract.expected_gid,
        "expected_root_device": contract.expected_root_device,
        "expected_root_inode": contract.expected_root_inode,
        "expected_identity_sha256": contract.expected_identity_sha256,
        "worker_sha256": request["worker_sha256"],
        "state_relative": state_path.relative_to(contract.root).as_posix(),
        "argv": validated_argv,
        "cwd_relative": cwd_relative,
        "environment": environment,
        "sandbox_profile": sandbox_profile,
        "timeout_seconds": timeout_seconds,
        "max_run_bytes": run_max_bytes,
        "min_free_bytes": contract.min_free_bytes,
    }
    config["config_sha256"] = document_sha256(config, "config_sha256")
    _atomic_write_recoverable(
        supervisor_path,
        canonical_json(config),
        0o400,
        "run supervisor configuration",
    )
    supervisor: Optional[subprocess.Popen[bytes]] = None
    authorization_read, authorization_write = os.pipe()
    try:
        supervisor = subprocess.Popen(
            [
                "/usr/bin/python3",
                str(Path(__file__).resolve()),
                "--run-supervisor-config",
                str(supervisor_path),
                "--run-supervisor-auth-fd",
                str(authorization_read),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "CASCADIA_R2_RUN_MARKER": request["request_id"]},
            close_fds=True,
            pass_fds=(authorization_read,),
            start_new_session=True,
        )
        os.close(authorization_read)
        authorization_read = -1
        view = memoryview(supervisor_authorization)
        while view:
            written = os.write(authorization_write, view)
            if written <= 0:
                raise RemoteWorkerError("run supervisor authorization made no progress")
            view = view[written:]
    except BaseException:
        if authorization_read >= 0:
            os.close(authorization_read)
        os.close(authorization_write)
        completed = _complete_interrupted_run(
            contract, state_path, state, request
        )
        return _finalize_run_state(contract, state_path, completed, request)
    os.close(authorization_write)
    completed = _await_run_completion(
        contract,
        state_path,
        request,
        spawned_supervisor=supervisor,
    )
    return _finalize_run_state(contract, state_path, completed, request)


def dispatch(
    contract: WorkerContract,
    request: Mapping[str, Any],
    source: BinaryIO,
    *,
    global_lock_held: bool = False,
    put_commit_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bytes]:
    operation = request.get("operation")
    arguments = request["arguments"]
    if operation == "provision":
        return provision_layout(contract), b""
    if operation == "protocol-info":
        return protocol_info(), b""
    if operation == "query-receipt":
        return _query_receipt(contract, arguments), b""
    if operation == "preflight":
        proof = verify_root(contract)
        proof["layout_verified"] = provision_layout(contract)["layout"]
        proof["atomic_probe"] = _atomic_preflight_probe(contract)
        return proof, b""
    if operation == "open-object":
        return {"object_token": _object_token(contract, arguments.get("relative"))}, b""
    if operation == "read-range":
        payload, result = _read_range(contract, arguments)
        return result, payload
    if operation == "put-file":
        return (
            _put_file(
                contract,
                arguments,
                source,
                lock_held=global_lock_held,
                request=request,
                commit_context=put_commit_context,
            ),
            b"",
        )
    if operation == "put-stream":
        return (
            _put_unknown_stream(
                contract,
                arguments,
                source,
                request=request,
                commit_context=put_commit_context,
            ),
            b"",
        )
    if operation == "publish-status":
        return (
            _put_file(
                contract,
                arguments,
                source,
                status_only=True,
                lock_held=global_lock_held,
                request=request,
                commit_context=put_commit_context,
            ),
            b"",
        )
    if operation in {"lock-acquire", "lock-renew", "lock-release"}:
        return _lease_operation(contract, operation, arguments, request), b""
    if operation == "transaction-begin":
        return _transaction_begin(contract, arguments, source), b""
    if operation == "transaction-put":
        return _transaction_put(contract, arguments, source, request), b""
    if operation == "transaction-import":
        return _transaction_import(contract, arguments, request), b""
    if operation == "transaction-commit":
        return _transaction_commit(contract, arguments), b""
    if operation == "transaction-abort":
        return _transaction_abort(contract, arguments), b""
    if operation == "run-command":
        return _run_command(contract, arguments, request), b""
    if operation == "run-controller":
        return _run_command(contract, arguments, request, controller=True), b""
    if operation == "run-cleanup-prepare":
        return _prepare_run_cleanup(contract, arguments), b""
    if operation == "run-cleanup-commit":
        return _commit_run_cleanup(contract, arguments), b""
    if operation == "failed-run-cleanup-prepare":
        return _prepare_failed_run_cleanup(contract, arguments), b""
    if operation == "failed-run-cleanup-commit":
        return _commit_failed_run_cleanup(contract, arguments), b""
    if operation == "inspect-executable":
        return _inspect_executable(contract, arguments), b""
    raise RemoteWorkerError("operation is not authorized")


def run(encoded_request: str, expected_worker_sha256: str) -> int:
    global _OUTER_GLOBAL_LOCK_FD
    try:
        actual_worker_sha256 = _worker_source_sha256(expected_worker_sha256)
    except Exception:
        sys.stderr.write("remote worker source identity drifted\n")
        return 70
    if actual_worker_sha256 != expected_worker_sha256:
        # There is no authenticated command yet, so emitting unframed stderr is intentional.
        sys.stderr.write("remote worker source identity drifted\n")
        return 70
    try:
        request = _request_from_base64(encoded_request, actual_worker_sha256)
    except RemoteWorkerError as error:
        sys.stderr.write("request rejected: %s\n" % error)
        return 64
    identity_sha256 = PRODUCTION_IDENTITY_SHA256
    transaction_lock_fd: Optional[int] = None
    request_identity_lock_fd: Optional[int] = None
    request_lock_fd: Optional[int] = None
    receipt_reservation: Optional[Path] = None
    put_context: Dict[str, Any] = {}
    replayed_receipt: Optional[Dict[str, Any]] = None
    response_disposition = "committed"
    put_operation = request["operation"] in {"put-file", "put-stream", "publish-status"}
    stable_mutator = request["operation"] in STABLE_MUTATING_OPERATIONS
    operation_handled = False
    pending_put_recovered = False
    try:
        try:
            # Full identity is checked for every operation. Size walking is deferred to preflight.
            low_free_recovery = _matching_run_recovery_under_pressure(
                PRODUCTION_CONTRACT, request
            )
            proof = verify_root(
                PRODUCTION_CONTRACT,
                measure_size=False,
                allow_low_free_recovery=low_free_recovery,
            )
            identity_sha256 = proof["host_identity_sha256"]
            if stable_mutator:
                request_identity_lock_fd, _ = _request_identity_lock(
                    PRODUCTION_CONTRACT, request["request_id"]
                )
            if request["operation"] in RESOURCE_LOCKED_OPERATIONS:
                request_lock_fd, _ = _request_lock(PRODUCTION_CONTRACT, request)
            if stable_mutator:
                transaction_lock_fd, _ = _global_lock(PRODUCTION_CONTRACT)
                replayed_receipt = _load_replay_receipt_locked(PRODUCTION_CONTRACT, request)
                if replayed_receipt is not None:
                    orphan_reservation = _receipt_reservation_path(
                        PRODUCTION_CONTRACT, request["request_id"]
                    )
                    if orphan_reservation.exists():
                        _load_receipt_reservation(
                            PRODUCTION_CONTRACT, orphan_reservation, request
                        )
                        _release_receipt_reservation(orphan_reservation)
                if replayed_receipt is None:
                    receipt_reservation = _reserve_receipt_capacity_locked(
                        PRODUCTION_CONTRACT, request
                    )
                os.close(transaction_lock_fd)
                transaction_lock_fd = None
                if replayed_receipt is not None:
                    _drain_replay_request_body(
                        request, sys.stdin.buffer, replayed_receipt["result"]
                    )
            if put_operation:
                recovery_lock, _ = _global_lock(PRODUCTION_CONTRACT)
                try:
                    _cleanup_put_journal_partials(PRODUCTION_CONTRACT, request)
                    pending = _load_put_commit_context(PRODUCTION_CONTRACT, request)
                    if replayed_receipt is not None:
                        if pending is not None:
                            _finalize_put_commit(pending)
                        result = dict(replayed_receipt["result"])
                        payload = b""
                        status = str(replayed_receipt["status"])
                        response_disposition = "replayed"
                        operation_handled = True
                    elif pending is not None:
                        try:
                            result = _recovered_put_result(PRODUCTION_CONTRACT, pending)
                        except Exception:
                            _rollback_put_commit(pending)
                        else:
                            put_context.update(pending)
                            payload = b""
                            status = "ok"
                            operation_handled = True
                            pending_put_recovered = True
                finally:
                    os.close(recovery_lock)
                if pending_put_recovered:
                    try:
                        _drain_replay_request_body(request, sys.stdin.buffer, result)
                    except BaseException:
                        rollback_lock, _ = _global_lock(PRODUCTION_CONTRACT)
                        try:
                            _rollback_put_commit(put_context)
                            put_context.clear()
                        finally:
                            os.close(rollback_lock)
                        raise
                if replayed_receipt is not None and pending is None:
                    cleanup_lock, _ = _global_lock(PRODUCTION_CONTRACT)
                    try:
                        orphan_data = _data_reservation_path(
                            PRODUCTION_CONTRACT, request["request_id"]
                        )
                        if orphan_data.exists():
                            _load_data_reservation(PRODUCTION_CONTRACT, orphan_data, request)
                            _release_data_reservation(orphan_data)
                    finally:
                        os.close(cleanup_lock)
            elif replayed_receipt is not None:
                result = dict(replayed_receipt["result"])
                payload = b""
                status = str(replayed_receipt["status"])
                response_disposition = "replayed"
                operation_handled = True
            if not operation_handled:
                result, payload = dispatch(
                    PRODUCTION_CONTRACT,
                    request,
                    sys.stdin.buffer,
                    global_lock_held=False,
                    put_commit_context=put_context,
                )
                status = "ok"
                operation_handled = True
        except Exception as error:
            replayed_receipt = None
            response_disposition = "committed"
            result = {"error_type": type(error).__name__, "error": str(error)[:4096]}
            payload = b""
            status = "error"

        result = dict(result)
        if replayed_receipt is None:
            result["payload_size"] = len(payload)
            result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
            receipt = _receipt(request, identity_sha256, status, result)
        else:
            receipt = replayed_receipt

        should_persist = (
            replayed_receipt is None
            and request["operation"] not in {"protocol-info", "query-receipt"}
            and not (stable_mutator and status != "ok")
        )
        if should_persist:
            try:
                if transaction_lock_fd is None:
                    transaction_lock_fd, _ = _global_lock(PRODUCTION_CONTRACT)
                _persist_receipt_locked(PRODUCTION_CONTRACT, receipt)
                _release_receipt_reservation(receipt_reservation)
                receipt_reservation = None
            except Exception as error:
                persisted = None
                if put_context:
                    with suppress(Exception):
                        persisted = _load_replay_receipt_locked(PRODUCTION_CONTRACT, request)
                if persisted != receipt:
                    if put_context:
                        with suppress(Exception):
                            _rollback_put_commit(put_context)
                    sys.stderr.write("receipt persistence failed: %s\n" % error)
                    return 75
        if put_context:
            try:
                _finalize_put_commit(put_context)
            except Exception as error:
                sys.stderr.write("put finalization failed: %s\n" % error)
                return 75
        if receipt_reservation is not None:
            if transaction_lock_fd is None:
                transaction_lock_fd, _ = _global_lock(PRODUCTION_CONTRACT)
            _release_receipt_reservation(receipt_reservation)
            receipt_reservation = None

        header = {
            "schema_id": "cascadia.r2-map.remote-frame.v1",
            "status": status,
            "response_disposition": response_disposition,
            "payload_size": len(payload),
            "payload_sha256": result["payload_sha256"],
            "receipt_size": len(canonical_json(receipt)),
        }
        frame = memoryview(encode_frame(header, payload, receipt))
        while frame:
            written = sys.stdout.buffer.write(frame)
            if written is None or written <= 0:
                raise RemoteWorkerError("worker response write made no progress")
            frame = frame[written:]
        sys.stdout.buffer.flush()
        return 0 if status == "ok" else 74
    finally:
        _OUTER_GLOBAL_LOCK_FD = None
        if receipt_reservation is not None:
            if transaction_lock_fd is None:
                transaction_lock_fd, _ = _global_lock(PRODUCTION_CONTRACT)
            _release_receipt_reservation(receipt_reservation)
        if transaction_lock_fd is not None:
            os.close(transaction_lock_fd)
        if request_lock_fd is not None:
            os.close(request_lock_fd)
        if request_identity_lock_fd is not None:
            os.close(request_identity_lock_fd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-sha256")
    parser.add_argument("--request-base64")
    parser.add_argument("--run-supervisor-config", type=Path)
    parser.add_argument("--run-supervisor-auth-fd", type=int)
    arguments = parser.parse_args(argv)
    if arguments.run_supervisor_config is not None:
        if (
            arguments.run_supervisor_auth_fd is None
            or arguments.worker_sha256 is not None
            or arguments.request_base64 is not None
        ):
            parser.error("run supervisor arguments are mutually exclusive")
        return _run_supervisor(
            arguments.run_supervisor_config,
            arguments.run_supervisor_auth_fd,
        )
    if arguments.worker_sha256 is None or arguments.request_base64 is None:
        parser.error("worker SHA-256 and request are required")
    return run(arguments.request_base64, arguments.worker_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
