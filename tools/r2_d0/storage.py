"""Physical John1 active-storage identity gates for every D0 write boundary."""

from __future__ import annotations

import hashlib
import os
import plistlib
import pwd
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from .canonical import D0Error, canonical_json

CANONICAL_ROOT = Path("/Users/johnherrick/cascadia-bench/r2-map-v1")
CANONICAL_ROOT_DEVICE = 16_777_230
CANONICAL_ROOT_INODE = 19_437_371
CANONICAL_IDENTITY_SHA256 = (
    "76bb13283e488e5d51261d61d967596c12e85d0e7393c228193121c263038c42"
)
CANONICAL_UID = 501
CANONICAL_GID = 20
CANONICAL_USER = "johnherrick"
FROZEN_LEGACY_JOHN2_ROOT = Path("/Users/john2/cascadia-bench/r2-map-v1")
MIN_FREE_BYTES = 64 * 1024**3
MAX_CAMPAIGN_BYTES = 64 * 1024**3
MAX_ENTRIES = 500_000


def _symlink_boundary(path: Path, allowed_prefixes: tuple[Path, ...]) -> Path | None:
    """Return the smallest audited tree a campaign symlink may resolve within."""

    for prefix in allowed_prefixes:
        try:
            relative = path.relative_to(prefix)
        except ValueError:
            continue
        # Build, tmp, and cache/runs each contain independently owned run trees.
        # A link in one run must not become a hidden dependency on another run.
        if prefix.name in {"runs", "build", "tmp"} and relative.parts:
            return prefix / relative.parts[0]
        return prefix
    return None


def _command(argv: list[str]) -> bytes:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            timeout=60,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise D0Error(f"storage identity command failed: {argv[0]}") from error
    if result.returncode != 0 or len(result.stdout) > 1024 * 1024 or result.stderr:
        raise D0Error(f"storage identity command differs: {argv[0]}")
    return result.stdout


def _platform_uuid() -> str:
    text = _command(["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]).decode()
    values = [
        line.split("=", 1)[1].strip().strip('"')
        for line in text.splitlines()
        if "IOPlatformUUID" in line and "=" in line
    ]
    if len(values) != 1:
        raise D0Error("storage platform UUID is unavailable")
    return values[0]


def _data_volume() -> dict[str, Any]:
    try:
        return plistlib.loads(
            _command(["/usr/sbin/diskutil", "info", "-plist", "/System/Volumes/Data"])
        )
    except (plistlib.InvalidFileException, ValueError) as error:
        raise D0Error("Data-volume identity is invalid") from error


def _no_follow_directory(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise D0Error(f"storage path cannot be inspected: {path}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise D0Error(f"storage path is not a real directory: {path}")
    return value


def _campaign_apparent_size(root: Path, root_device: int) -> tuple[int, int]:
    apparent = 0
    entries = 0
    pending = [root]
    allowed_symlink_prefixes = (
        root / "cache/runs",
        root / "build",
        root / "toolchains",
    )
    while pending:
        directory = pending.pop()
        try:
            scanner = os.scandir(directory)
        except OSError as error:
            raise D0Error("canonical storage cannot be enumerated") from error
        with scanner:
            for child in scanner:
                path = Path(child.path)
                try:
                    details = child.stat(follow_symlinks=False)
                except OSError as error:
                    raise D0Error("canonical storage entry cannot be inspected") from error
                entries += 1
                if entries > MAX_ENTRIES:
                    raise D0Error("canonical storage exceeds its audited entry limit")
                if stat.S_ISLNK(details.st_mode):
                    boundary = _symlink_boundary(path, allowed_symlink_prefixes)
                    if boundary is None:
                        raise D0Error("canonical storage contains an unauthorized symlink")
                    try:
                        resolved = path.resolve(strict=True)
                    except FileNotFoundError as error:
                        raise D0Error("canonical storage contains a dangling symlink") from error
                    except RuntimeError as error:
                        raise D0Error("canonical storage contains a symlink loop") from error
                    try:
                        resolved.relative_to(boundary)
                    except ValueError as error:
                        raise D0Error(
                            "canonical storage symlink escapes its audited run boundary"
                        ) from error
                    try:
                        resolved_details = os.stat(resolved)
                    except OSError as error:
                        raise D0Error(
                            "canonical storage symlink target cannot be inspected"
                        ) from error
                    if resolved_details.st_dev != root_device:
                        raise D0Error(
                            "canonical storage symlink target crosses a device boundary"
                        )
                    apparent += details.st_size
                    continue
                if details.st_dev != root_device:
                    raise D0Error("canonical storage crosses a device boundary")
                if stat.S_ISDIR(details.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(details.st_mode):
                    apparent += details.st_size
                else:
                    raise D0Error("canonical storage contains a special file")
    return apparent, entries


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _storage_write_probe(control: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Prove write/fsync/rename/reopen/unlink durability without retaining state."""

    observed = _no_follow_directory(control)
    if (
        observed.st_uid != CANONICAL_UID
        or observed.st_gid != CANONICAL_GID
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise D0Error("canonical John1 control directory metadata drifted")
    nonce = f"{os.getpid()}-{time.time_ns()}"
    partial = control / f".d0-storage-probe-{nonce}.partial"
    committed = control / f".d0-storage-probe-{nonce}.committed"
    payload_document = {
        "schema_id": "cascadia.r2-map.d0-storage-write-probe.v1",
        "schema_version": 1,
        "host_identity_sha256": hashlib.sha256(canonical_json(identity)).hexdigest(),
        "nonce": nonce,
    }
    payload = canonical_json(payload_document)
    descriptor = -1
    try:
        descriptor = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position:])
            if written <= 0:
                raise D0Error("canonical John1 storage probe made a short write")
            position += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(partial, committed)
        _fsync_directory(control)
        committed_stat = committed.lstat()
        if (
            not stat.S_ISREG(committed_stat.st_mode)
            or committed_stat.st_uid != CANONICAL_UID
            or committed_stat.st_gid != CANONICAL_GID
            or committed_stat.st_nlink != 1
            or stat.S_IMODE(committed_stat.st_mode) != 0o400
            or committed_stat.st_dev != observed.st_dev
            or committed_stat.st_size != len(payload)
        ):
            raise D0Error("canonical John1 storage probe metadata differs after rename")
        reopened = os.open(committed, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            reread = bytearray()
            while len(reread) <= len(payload):
                chunk = os.read(reopened, min(64 * 1024, len(payload) + 1 - len(reread)))
                if not chunk:
                    break
                reread.extend(chunk)
        finally:
            os.close(reopened)
        if bytes(reread) != payload:
            raise D0Error("canonical John1 storage probe bytes differ after reopen")
        committed.unlink()
        _fsync_directory(control)
        if partial.exists() or partial.is_symlink() or committed.exists() or committed.is_symlink():
            raise D0Error("canonical John1 storage probe cleanup is incomplete")
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        cleanup_failed = False
        for path in (partial, committed):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
        try:
            _fsync_directory(control)
        except OSError:
            cleanup_failed = True
        if cleanup_failed:
            raise D0Error("canonical John1 storage probe failed and could not clean up") from error
        raise
    return {
        "schema_id": "cascadia.r2-map.d0-storage-write-probe-receipt.v1",
        "schema_version": 1,
        "directory": str(control),
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "write_fsync": True,
        "atomic_rename": True,
        "directory_fsync_after_rename": True,
        "no_follow_reread": True,
        "cleanup_unlink": True,
        "directory_fsync_after_cleanup": True,
        "status": "pass",
    }


def verify_canonical_storage(*, measure_size: bool = True) -> dict[str, Any]:
    """Revalidate John1's internal-APFS active root before an authoritative write."""

    root = _no_follow_directory(CANONICAL_ROOT)
    if (
        stat.S_IMODE(root.st_mode) != 0o700
        or (root.st_uid, root.st_gid) != (CANONICAL_UID, CANONICAL_GID)
        or (root.st_dev, root.st_ino) != (CANONICAL_ROOT_DEVICE, CANONICAL_ROOT_INODE)
        or pwd.getpwuid(os.getuid()).pw_name != CANONICAL_USER
    ):
        raise D0Error("canonical John1 storage root identity drifted")
    for ancestor in (
        Path("/Users"),
        Path("/Users/johnherrick"),
        Path("/Users/johnherrick/cascadia-bench"),
        CANONICAL_ROOT,
    ):
        if _no_follow_directory(ancestor).st_dev != root.st_dev:
            raise D0Error("canonical John1 storage crosses the Data volume")
    info = _data_volume()
    identity = {
        "device_identifier": info.get("DeviceIdentifier"),
        "filesystem": info.get("FilesystemType"),
        "gid": os.getgid(),
        "hostname": os.uname().nodename,
        "platform_uuid": _platform_uuid(),
        "protocol": info.get("BusProtocol"),
        "root_device": root.st_dev,
        "root_inode": root.st_ino,
        "uid": os.getuid(),
        "user": pwd.getpwuid(os.getuid()).pw_name,
        "volume_uuid": info.get("VolumeUUID"),
    }
    digest = hashlib.sha256(canonical_json(identity)).hexdigest()
    if (
        digest != CANONICAL_IDENTITY_SHA256
        or identity["filesystem"] != "apfs"
        or identity["protocol"] != "Apple Fabric"
    ):
        raise D0Error("canonical John1 platform-volume identity drifted")
    physical = {
        "filesystem": info.get("FilesystemType"),
        "protocol": info.get("BusProtocol"),
        "internal": info.get("Internal"),
        "removable": info.get("Removable"),
        "solid_state": info.get("SolidState"),
    }
    if physical != {
        "filesystem": "apfs",
        "protocol": "Apple Fabric",
        "internal": True,
        "removable": False,
        "solid_state": True,
    }:
        raise D0Error("canonical John1 storage is not internal solid-state Apple Fabric APFS")
    usage = shutil.disk_usage(CANONICAL_ROOT)
    if usage.free < MIN_FREE_BYTES:
        raise D0Error("canonical John1 free-space reserve is below 64 GiB")
    apparent, entries = (
        _campaign_apparent_size(CANONICAL_ROOT, root.st_dev) if measure_size else (None, None)
    )
    if apparent is not None and apparent > MAX_CAMPAIGN_BYTES:
        raise D0Error("canonical John1 campaign exceeds 64 GiB")
    composite = {
        "host": identity,
        "physical_storage": physical,
        "root": {
            "path": str(CANONICAL_ROOT),
            "device": root.st_dev,
            "inode": root.st_ino,
            "uid": root.st_uid,
            "gid": root.st_gid,
            "mode": "0700",
        },
    }
    write_probe = _storage_write_probe(CANONICAL_ROOT / "control", composite)
    receipt = {
        "schema_id": "cascadia.r2-map.d0-john1-active-storage-receipt.v1",
        "schema_version": 1,
        "root": str(CANONICAL_ROOT),
        "root_device": root.st_dev,
        "root_inode": root.st_ino,
        "root_uid": root.st_uid,
        "root_gid": root.st_gid,
        "root_mode": "0700",
        "host_identity_sha256": digest,
        "filesystem": identity["filesystem"],
        "protocol": identity["protocol"],
        "internal": physical["internal"],
        "removable": physical["removable"],
        "solid_state": physical["solid_state"],
        "platform_volume_identity": composite,
        "platform_volume_identity_sha256": hashlib.sha256(canonical_json(composite)).hexdigest(),
        "free_bytes": usage.free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "campaign_apparent_bytes": apparent,
        "campaign_entries": entries,
        "maximum_campaign_bytes": MAX_CAMPAIGN_BYTES,
        "write_probe": write_probe,
        "status": "pass",
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    return receipt


def verify_canonical_commit_boundary(destination: Path) -> dict[str, Any] | None:
    """Revalidate John1 active storage immediately before an atomic commit."""

    try:
        destination.relative_to(FROZEN_LEGACY_JOHN2_ROOT)
    except ValueError:
        pass
    else:
        raise D0Error("the former John2 campaign root is frozen legacy state")
    try:
        destination.relative_to(CANONICAL_ROOT)
    except ValueError:
        return None
    if pwd.getpwuid(os.getuid()).pw_name != CANONICAL_USER:
        raise D0Error("only John1 may commit beneath the canonical active campaign root")
    return verify_canonical_storage(measure_size=True)
