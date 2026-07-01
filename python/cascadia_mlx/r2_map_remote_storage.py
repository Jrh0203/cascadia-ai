# ruff: noqa: E501
"""Strict SSH-backed storage/control client for the john2 R2-MAP root.

This is the only supported bridge between john1 compute and john2 campaign
storage.  It deliberately exposes bounded object windows and streaming atomic
publication rather than a mounted or synchronized tree.  No method silently
falls back to a local path.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import pwd
import re
import resource
import signal
import shlex
import shutil
import socket
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

import blake3

from cascadia_mlx import r2_map_remote_worker as worker

REMOTE_HOST_ALIAS = "john2"
REMOTE_HOSTNAME = "100.100.43.38"
REMOTE_USER = "john2"
REMOTE_ROOT = worker.PRODUCTION_ROOT
REMOTE_IDENTITY_SHA256 = worker.PRODUCTION_IDENTITY_SHA256
JOHN1_STAGING_ROOT = Path("/private/tmp")
JOHN1_HOSTNAME = "Johns-Mac-mini.local"
JOHN1_USER = "johnherrick"
JOHN1_UID = 501
JOHN1_GID = 20
JOHN1_REPOSITORY_ROOT = Path("/Users/johnherrick/cascadia")
JOHN1_DASHBOARD_API_PATH = JOHN1_REPOSITORY_ROOT / "target/release/cascadia-api"
DASHBOARD_API_MAX_BYTES = 64 * (1 << 20)
DASHBOARD_API_BUNDLE_SCHEMA = "cascadia.r2-map.dashboard-api-bundle.v1"
EPHEMERAL_RUNTIME_MANIFEST_SCHEMA = "cascadia.r2-map.john1-ephemeral-runtime.v1"
SSH_BINARY = Path("/usr/bin/ssh")
JOHN2_SSH_IDENTITY = Path("/Users/johnherrick/.ssh/john2_codex")
JOHN2_SSH_KNOWN_HOSTS = Path(
    "/Users/johnherrick/.config/cascadia-r2-d0/ssh/john2-known-hosts"
)
JOHN2_SSH_KNOWN_HOSTS_BYTES = (
    b"100.100.43.38 ssh-ed25519 "
    b"AAAAC3NzaC1lZDI1NTE5AAAAIJNlNTjAXURkqR7jt4h8AEaVSro7Gw91curGznTylmA/\n"
)
JOHN2_SSH_KNOWN_HOSTS_SHA256 = (
    "2a06d98b8e80614e69d16841f3126a7964a7e7fd387bc57f0597347265d23dc3"
)
JOHN2_SSH_HOST_KEY_FINGERPRINT = "SHA256:kXhUzhc/d5W4/L+H9FWntyvCGPaUL9AYjVqBzhwolNw"
JOHN1_JOHN2_PUBLIC_KEY_BYTES = (
    b"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDAYfd7JtG91R6n8eMlKzUjjnknXCLBGdzD43Pmp57vk "
    b"john1-codex-to-john2\n"
)
JOHN1_JOHN2_PUBLIC_KEY_SHA256 = (
    "c10d445c9d64b9a5cc6f069582964884765eb791771e8e13c8f36878a9285d92"
)
JOHN1_JOHN2_KEY_FINGERPRINT = "SHA256:Gu0TXn6/ngJK1xrzmkhvxqoCzB5Xrsyfiz6+AhIrkR4"
WORKER_LOCAL_PATH = Path(worker.__file__).resolve()
MAX_STDERR_BYTES = 64 * (1 << 10)

BOOTSTRAP_SCHEMA = "cascadia.r2-map.remote-worker-bootstrap.v1"

# This program is sent as the `python3 -c` body only during content-addressed
# worker installation. It verifies the frozen host/root identity before it
# reads stdin, then installs the worker by fsync+rename. Normal operations use
# only the installed, hash-checked worker.
BOOTSTRAP_PROGRAM = r"""
import hashlib,json,os,plistlib,pwd,stat,subprocess,sys,time,uuid
root=sys.argv[1]; expected_identity=sys.argv[2]; expected_device=int(sys.argv[3]); expected_inode=int(sys.argv[4]); expected_sha=sys.argv[5]; expected_size=int(sys.argv[6]); command_sha=sys.argv[7]
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")
def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try: os.fsync(fd)
    finally: os.close(fd)
def lstat_safe(path):
    value=os.lstat(path)
    if stat.S_ISLNK(value.st_mode): raise RuntimeError("bootstrap path is a symlink")
    return value
if root!="/Users/john2/cascadia-bench/r2-map-v1": raise RuntimeError("bootstrap root drifted")
root_stat=lstat_safe(root)
if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode)!=0o700 or (root_stat.st_uid,root_stat.st_gid)!=(501,20): raise RuntimeError("bootstrap root mode/owner is unsafe")
if (root_stat.st_dev,root_stat.st_ino)!=(expected_device,expected_inode): raise RuntimeError("bootstrap root identity drifted")
for path in ("/Users","/Users/john2","/Users/john2/cascadia-bench",root):
    value=lstat_safe(path)
    if value.st_dev!=root_stat.st_dev: raise RuntimeError("bootstrap crosses the Data volume")
ioreg=subprocess.check_output(["/usr/sbin/ioreg","-rd1","-c","IOPlatformExpertDevice"],text=True)
platform=[line.split("=",1)[1].strip().strip('"') for line in ioreg.splitlines() if "IOPlatformUUID" in line]
if len(platform)!=1: raise RuntimeError("platform identity unavailable")
info=plistlib.loads(subprocess.check_output(["/usr/sbin/diskutil","info","-plist","/System/Volumes/Data"]))
identity={"device_identifier":info.get("DeviceIdentifier"),"filesystem":info.get("FilesystemType"),"gid":os.getgid(),"hostname":os.uname().nodename,"platform_uuid":platform[0],"protocol":info.get("BusProtocol"),"root_device":root_stat.st_dev,"root_inode":root_stat.st_ino,"uid":os.getuid(),"user":pwd.getpwuid(os.getuid()).pw_name,"volume_uuid":info.get("VolumeUUID")}
identity_sha=hashlib.sha256(canonical(identity)).hexdigest()
if identity_sha!=expected_identity: raise RuntimeError("bootstrap host identity drifted")
payload=sys.stdin.buffer.read(expected_size+1)
if len(payload)!=expected_size or hashlib.sha256(payload).hexdigest()!=expected_sha: raise RuntimeError("bootstrap worker payload mismatch")
control=os.path.join(root,"control"); bindir=os.path.join(control,"bin")
for path in (control,bindir):
    try: os.mkdir(path,0o700); fsync_dir(os.path.dirname(path))
    except FileExistsError:
        value=lstat_safe(path)
        if not stat.S_ISDIR(value.st_mode) or stat.S_IMODE(value.st_mode)!=0o700 or (value.st_uid,value.st_gid)!=(501,20): raise RuntimeError("bootstrap directory is unsafe")
target=os.path.join(bindir,"r2-map-remote-worker-%s.py"%expected_sha)
try:
    value=lstat_safe(target)
except FileNotFoundError:
    value=None
if value is not None:
    with open(target,"rb") as source: existing=source.read()
    if not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode)!=0o500 or hashlib.sha256(existing).hexdigest()!=expected_sha or len(existing)!=expected_size: raise RuntimeError("installed worker identity drifted")
else:
    temporary=os.path.join(bindir,".%s.tmp"%uuid.uuid4().hex)
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o500)
    try:
        view=memoryview(payload)
        while view:
            count=os.write(fd,view)
            if count<=0: raise RuntimeError("short bootstrap write")
            view=view[count:]
        os.fsync(fd); os.fchmod(fd,0o500)
    finally: os.close(fd)
    os.rename(temporary,target); fsync_dir(bindir)
receipt={"schema_id":"cascadia.r2-map.remote-worker-bootstrap.v1","status":"ok","command_sha256":command_sha,"host":"john2","host_identity_sha256":identity_sha,"root":root,"worker_path":target,"worker_sha256":expected_sha,"worker_size":expected_size,"completed_unix_ms":time.time_ns()//1000000 if hasattr(time,"time_ns") else int(time.time()*1000)}
receipt["receipt_sha256"]=hashlib.sha256(canonical(receipt)).hexdigest()
sys.stdout.buffer.write(canonical(receipt)+b"\n")
""".replace(
    "import hashlib,json,os,plistlib,pwd,stat,subprocess,sys,uuid",
    "import hashlib,json,os,plistlib,pwd,stat,subprocess,sys,time,uuid",
)


class RemoteStorageError(RuntimeError):
    """Base class for a remote storage contract failure."""


class RemoteTransportError(RemoteStorageError):
    """SSH failed before an authenticated worker receipt was available."""


class RemoteProtocolError(RemoteStorageError):
    """The worker response was malformed, mismatched, or unverified."""


class RemoteOperationError(RemoteStorageError):
    """The authenticated worker rejected an operation."""

    def __init__(self, message: str, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = dict(receipt)


class ChunkSource(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...


def _ssh_key_fingerprint(payload: bytes, *, known_host: bool) -> str:
    fields = payload.rstrip(b"\n").split()
    algorithm_index = 1 if known_host else 0
    key_index = algorithm_index + 1
    try:
        if fields[algorithm_index] != b"ssh-ed25519":
            raise ValueError
        raw = base64.b64decode(fields[key_index], validate=True)
    except (IndexError, ValueError, TypeError) as error:
        raise RemoteTransportError("pinned SSH key is not canonical Ed25519") from error
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(
        "ascii"
    )


def _secure_owner_directory(path: Path) -> list[Path]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    current = Path("/")
    created: list[Path] = []
    try:
        for component in path.parts[1:]:
            current = current / component
            made = False
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno != errno.ENOENT:
                    raise RemoteTransportError(
                        f"SSH trust directory component is unsafe: {current}"
                    ) from error
                os.mkdir(component, 0o700, dir_fd=descriptor)
                made = True
                child = os.open(component, flags, dir_fd=descriptor)
            details = os.fstat(child)
            if not stat.S_ISDIR(details.st_mode):
                os.close(child)
                raise RemoteTransportError(f"SSH trust ancestor is not a directory: {current}")
            if made:
                os.fchmod(child, 0o700)
                details = os.fstat(child)
                created.append(current)
            if current == path and (
                details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) != 0o700
            ):
                os.close(child)
                raise RemoteTransportError("SSH trust directory owner or mode differs")
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return created


def _read_exact_ssh_file(
    path: Path, expected: bytes, modes: set[int], label: str
) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) <= len(expected):
            chunk = os.read(descriptor, len(expected) + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) not in modes
        or any(getattr(before, field) != getattr(after, field) for field in stable)
        or bytes(payload) != expected
    ):
        raise RemoteTransportError(f"{label} metadata or bytes differ")
    return {
        "path": str(path),
        "sha256": content_sha256(bytes(payload)),
        "size": len(payload),
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "uid": before.st_uid,
        "gid": before.st_gid,
        "nlink": before.st_nlink,
    }


def _provision_and_verify_ssh_trust() -> dict[str, Any]:
    if pwd.getpwuid(os.getuid()).pw_name != JOHN1_USER:
        raise RemoteTransportError("John2 SSH trust may only be provisioned by johnherrick")
    created_directories = _secure_owner_directory(JOHN2_SSH_KNOWN_HOSTS.parent)
    created_file = False
    temporary: Path | None = None
    try:
        try:
            os.lstat(JOHN2_SSH_KNOWN_HOSTS)
        except FileNotFoundError:
            temporary = JOHN2_SSH_KNOWN_HOSTS.parent / (
                f".{JOHN2_SSH_KNOWN_HOSTS.name}.partial-{os.getpid()}-{time.time_ns()}"
            )
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                position = 0
                while position < len(JOHN2_SSH_KNOWN_HOSTS_BYTES):
                    written = os.write(descriptor, JOHN2_SSH_KNOWN_HOSTS_BYTES[position:])
                    if written <= 0:
                        raise RemoteTransportError("known-host pin write made no progress")
                    position += written
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(temporary, JOHN2_SSH_KNOWN_HOSTS, follow_symlinks=False)
            except FileExistsError:
                os.unlink(temporary)
                temporary = None
            else:
                os.unlink(temporary)
                temporary = None
                created_file = True
            directory = os.open(
                JOHN2_SSH_KNOWN_HOSTS.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        known_hosts = _read_exact_ssh_file(
            JOHN2_SSH_KNOWN_HOSTS,
            JOHN2_SSH_KNOWN_HOSTS_BYTES,
            {0o600},
            "John2 known-host pin",
        )
        public = _read_exact_ssh_file(
            JOHN2_SSH_IDENTITY.with_suffix(".pub"),
            JOHN1_JOHN2_PUBLIC_KEY_BYTES,
            {0o600, 0o644},
            "John1-to-John2 public key",
        )
        private = os.lstat(JOHN2_SSH_IDENTITY)
        if (
            not stat.S_ISREG(private.st_mode)
            or stat.S_ISLNK(private.st_mode)
            or private.st_nlink != 1
            or private.st_uid != os.getuid()
            or stat.S_IMODE(private.st_mode) not in {0o400, 0o600}
        ):
            raise RemoteTransportError("John1-to-John2 private-key metadata differs")
        derived = subprocess.run(
            ["/usr/bin/ssh-keygen", "-y", "-f", str(JOHN2_SSH_IDENTITY)],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        ).stdout.rstrip(b"\n")
        if derived.split()[:2] != JOHN1_JOHN2_PUBLIC_KEY_BYTES.rstrip(b"\n").split()[:2]:
            raise RemoteTransportError("John1-to-John2 private/public key pair differs")
        host_fingerprint = _ssh_key_fingerprint(JOHN2_SSH_KNOWN_HOSTS_BYTES, known_host=True)
        key_fingerprint = _ssh_key_fingerprint(JOHN1_JOHN2_PUBLIC_KEY_BYTES, known_host=False)
        if (
            known_hosts["sha256"] != JOHN2_SSH_KNOWN_HOSTS_SHA256
            or public["sha256"] != JOHN1_JOHN2_PUBLIC_KEY_SHA256
            or host_fingerprint != JOHN2_SSH_HOST_KEY_FINGERPRINT
            or key_fingerprint != JOHN1_JOHN2_KEY_FINGERPRINT
        ):
            raise RemoteTransportError("John2 SSH pin digest or fingerprint differs")
        return {
            "known_hosts": known_hosts,
            "host_key_fingerprint": host_fingerprint,
            "public_key": public,
            "public_key_fingerprint": key_fingerprint,
            "private_key_mode": f"{stat.S_IMODE(private.st_mode):04o}",
            "known_hosts_created": created_file,
        }
    except BaseException:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()
        if created_file:
            with suppress(FileNotFoundError):
                JOHN2_SSH_KNOWN_HOSTS.unlink()
        for directory in reversed(created_directories):
            with suppress(OSError):
                directory.rmdir()
        raise


@dataclass(frozen=True)
class RemoteResult:
    payload: bytes
    receipt: dict[str, Any]
    input_sha256: str
    input_size: int

    @property
    def result(self) -> dict[str, Any]:
        value = self.receipt["result"]
        if not isinstance(value, dict):
            raise RemoteProtocolError("receipt result is not an object")
        return value

    @property
    def storage_receipt(self) -> dict[str, str]:
        request_id = self.receipt.get("request_id")
        receipt_sha256 = self.receipt.get("receipt_sha256")
        if (
            not isinstance(request_id, str)
            or not worker.IDENTIFIER.fullmatch(request_id)
            or not isinstance(receipt_sha256, str)
        ):
            raise RemoteProtocolError("storage receipt locator is invalid")
        _validate_sha256(receipt_sha256, "storage receipt SHA-256")
        return {
            "storage_receipt_relative": f"control/receipts/{request_id}.json",
            "storage_receipt_sha256": receipt_sha256,
        }


@dataclass(frozen=True)
class TransactionObject:
    relative: str
    size: int
    sha256: str
    mode: int = 0o400

    def to_dict(self) -> dict[str, Any]:
        _validate_relative(self.relative, "transaction object")
        _validate_sha256(self.sha256, "transaction object SHA-256")
        if self.size < 0:
            raise ValueError("transaction object size cannot be negative")
        if self.mode not in {0o400, 0o500}:
            raise ValueError("transaction object mode must be 0400 or 0500")
        result: dict[str, Any] = {
            "relative": self.relative,
            "sha256": self.sha256,
            "size": self.size,
        }
        if self.mode == 0o500:
            result["mode"] = "0500"
        return result


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def content_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def document_sha256(document: Mapping[str, Any], field: str) -> str:
    value = dict(document)
    value.pop(field, None)
    return content_sha256(canonical_json(value))


def _validate_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_relative(value: str, label: str) -> str:
    if not value or len(value.encode()) > 1024 or "\x00" in value or "\\" in value:
        raise ValueError(f"{label} must be a bounded POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be canonical and contained")
    return value


def _validate_storage_receipt_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a receipt-relative path")
    _validate_relative(value, label)
    parts = PurePosixPath(value).parts
    if (
        len(parts) != 3
        or parts[:2] != ("control", "receipts")
        or not parts[2].startswith("req-")
        or not parts[2].endswith(".json")
    ):
        raise ValueError(f"{label} is not a canonical persisted receipt locator")
    return value


def _validate_source_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    freeze = dict(value)
    if set(freeze) != {
        "target_relative",
        "manifest_sha256",
        "storage_receipt_relative",
        "storage_receipt_sha256",
    }:
        raise ValueError("source freeze identity fields are invalid")
    target = _validate_relative(str(freeze["target_relative"]), "source freeze target")
    if PurePosixPath(target).parts[0] != "source":
        raise ValueError("source freeze target must be an immutable source transaction")
    _validate_sha256(str(freeze["manifest_sha256"]), "source freeze manifest SHA-256")
    _validate_storage_receipt_relative(
        freeze["storage_receipt_relative"], "source freeze storage receipt"
    )
    _validate_sha256(str(freeze["storage_receipt_sha256"]), "source freeze storage receipt SHA-256")
    return freeze


def build_transaction_manifest(
    transaction_id: str,
    target_relative: str,
    objects: Sequence[TransactionObject],
) -> dict[str, Any]:
    if not worker.IDENTIFIER.fullmatch(transaction_id):
        raise ValueError("transaction_id is not a safe identifier")
    _validate_relative(target_relative, "transaction target")
    descriptors = sorted((item.to_dict() for item in objects), key=lambda item: item["relative"])
    if not descriptors or len({item["relative"] for item in descriptors}) != len(descriptors):
        raise ValueError("transaction objects must be non-empty and unique")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": worker.TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "target_relative": target_relative,
        "objects": descriptors,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    return manifest


class SshTransport:
    """One fail-closed OpenSSH connection per authenticated operation."""

    def __init__(
        self,
        host_alias: str = REMOTE_HOST_ALIAS,
        timeout_seconds: int = 15,
        *,
        compression: bool = False,
    ):
        if host_alias != REMOTE_HOST_ALIAS:
            raise ValueError("only the frozen john2 endpoint is authorized")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("SSH timeout must be between 1 and 120 seconds")
        if not isinstance(compression, bool):
            raise TypeError("SSH compression must be an explicit boolean")
        self.host_alias = host_alias
        self.timeout_seconds = timeout_seconds
        self.compression = compression
        self._active_lock = threading.Lock()
        self._active_processes: dict[int, subprocess.Popen[bytes]] = {}

    def _remember_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_lock:
            self._active_processes[id(process)] = process

    def _forget_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_lock:
            self._active_processes.pop(id(process), None)

    def cancel_active(self) -> None:
        """Boundedly terminate every SSH process currently owned by this transport."""
        with self._active_lock:
            active = list(self._active_processes.values())
        failures: list[BaseException] = []
        for process in active:
            try:
                self._terminate_and_reap(process)
            except BaseException as error:
                failures.append(error)
            finally:
                self._forget_process(process)
        if failures:
            raise RemoteTransportError("one or more active SSH groups survived cancellation") from failures[0]

    @property
    def base_argv(self) -> list[str]:
        return [
            str(SSH_BINARY),
            "-F",
            "/dev/null",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            f"ConnectTimeout={self.timeout_seconds}",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            f"HostName={REMOTE_HOSTNAME}",
            "-o",
            "HostKeyAlgorithms=ssh-ed25519",
            "-o",
            f"HostKeyAlias={REMOTE_HOSTNAME}",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"IdentityFile={JOHN2_SSH_IDENTITY}",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "PubkeyAcceptedAlgorithms=ssh-ed25519",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "RequestTTY=no",
            "-o",
            f"User={REMOTE_USER}",
            "-o",
            f"UserKnownHostsFile={JOHN2_SSH_KNOWN_HOSTS}",
            "-o",
            f"Compression={'yes' if self.compression else 'no'}",
            self.host_alias,
        ]

    def verify_local_configuration(self) -> dict[str, str]:
        trust = _provision_and_verify_ssh_trust()
        completed = subprocess.run(
            [str(SSH_BINARY), "-G", *self.base_argv[1:]],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        configuration: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition(" ")
            if separator and key not in configuration:
                configuration[key] = value.strip()
        if (
            configuration.get("hostname") != REMOTE_HOSTNAME
            or configuration.get("user") != REMOTE_USER
            or configuration.get("controlmaster") != "false"
            or configuration.get("controlpath") not in {None, "none"}
            or configuration.get("updatehostkeys") != "false"
            or configuration.get("hostkeyalias") != REMOTE_HOSTNAME
            or configuration.get("hostkeyalgorithms") != "ssh-ed25519"
            or configuration.get("pubkeyacceptedalgorithms") != "ssh-ed25519"
            or configuration.get("preferredauthentications") != "publickey"
            or configuration.get("userknownhostsfile") != str(JOHN2_SSH_KNOWN_HOSTS)
            or configuration.get("globalknownhostsfile") != "/dev/null"
            or configuration.get("identitiesonly") != "yes"
        ):
            raise RemoteTransportError(
                "SSH no-persistence configuration or frozen john2 endpoint drifted"
            )
        identity = os.path.expanduser(configuration.get("identityfile", ""))
        if identity != str(JOHN2_SSH_IDENTITY):
            raise RemoteTransportError("SSH alias no longer uses the dedicated john2 identity")
        return {
            "alias": self.host_alias,
            "compression": "yes" if self.compression else "no",
            "hostname": configuration["hostname"],
            "user": configuration["user"],
            "identityfile": identity,
            "controlmaster": "no",
            "controlpath": "none",
            "updatehostkeys": "no",
            "known_hosts": str(JOHN2_SSH_KNOWN_HOSTS),
            "known_hosts_sha256": str(trust["known_hosts"]["sha256"]),
            "host_key_fingerprint": str(trust["host_key_fingerprint"]),
            "public_key_fingerprint": str(trust["public_key_fingerprint"]),
        }

    def run(
        self,
        remote_argv: Sequence[str],
        chunks: Iterable[bytes] = (),
        *,
        timeout_seconds: int | None = None,
    ) -> tuple[int, bytes, bytes, str, int]:
        remote_command = shlex.join(list(remote_argv))
        process = subprocess.Popen(
            [*self.base_argv, remote_command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._remember_process(process)
        try:
            return self._run_owned_process(
                process,
                chunks,
                timeout_seconds=timeout_seconds,
            )
        finally:
            self._forget_process(process)

    def _run_owned_process(
        self,
        process: subprocess.Popen[bytes],
        chunks: Iterable[bytes],
        *,
        timeout_seconds: int | None,
    ) -> tuple[int, bytes, bytes, str, int]:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._terminate_and_reap(process)
            raise RemoteTransportError("SSH pipes were not created")
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        writer_error: list[BaseException] = []
        local_digest = hashlib.sha256()
        local_size = 0

        def write_input() -> None:
            nonlocal local_size
            try:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("SSH stream chunks must be bytes")
                    if not chunk:
                        continue
                    view = memoryview(chunk)
                    while view:
                        written = process.stdin.write(view)
                        if written is None or written <= 0:
                            raise BrokenPipeError("SSH stdin write made no progress")
                        view = view[written:]
                    local_digest.update(chunk)
                    local_size += len(chunk)
                process.stdin.close()
            except BaseException as error:
                writer_error.append(error)
                with suppress(OSError):
                    process.stdin.close()

        def read_all(source: BinaryIO, destination: list[bytes]) -> None:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    return
                destination.append(chunk)

        threads = [
            threading.Thread(target=write_input, daemon=True),
            threading.Thread(target=read_all, args=(process.stdout, stdout_parts), daemon=True),
            threading.Thread(target=read_all, args=(process.stderr, stderr_parts), daemon=True),
        ]
        for thread in threads:
            thread.start()
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds + 300
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self._terminate_and_reap(process)
            raise RemoteTransportError("SSH operation exceeded its local timeout") from error
        finally:
            for thread in threads:
                thread.join(timeout=5)
            if any(thread.is_alive() for thread in threads):
                self._terminate_and_reap(process)
                raise RemoteTransportError("SSH I/O thread survived bounded process reap")
        if writer_error and (
            not stdout_parts or not isinstance(writer_error[0], (BrokenPipeError, OSError))
        ):
            raise RemoteTransportError(f"SSH input stream failed: {writer_error[0]}")
        return (
            return_code,
            b"".join(stdout_parts),
            b"".join(stderr_parts),
            local_digest.hexdigest(),
            local_size,
        )

    @staticmethod
    def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                group_exists = False
            except PermissionError:
                group_exists = True
            else:
                group_exists = True
            if group_exists:
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGTERM)
        else:
            with suppress(OSError):
                process.kill()
        leader_reaped = False
        try:
            process.wait(timeout=2)
            leader_reaped = True
        except subprocess.TimeoutExpired:
            pass
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                group_exists = False
            except PermissionError:
                group_exists = True
            else:
                group_exists = True
            if group_exists:
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
        elif not leader_reaped:
            with suppress(OSError):
                process.kill()
        if not leader_reaped:
            try:
                process.wait(timeout=5)
                leader_reaped = True
            except subprocess.TimeoutExpired as error:
                raise RemoteTransportError("SSH process resisted bounded reap") from error
        if isinstance(pid, int) and pid > 0:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    return
                except PermissionError:
                    pass
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
                time.sleep(0.05)
            raise RemoteTransportError("SSH process descendants survived bounded reap")


class RemoteStorageClient:
    def __init__(self, transport: SshTransport | None = None):
        self.transport = transport or SshTransport()
        self.worker_bytes = WORKER_LOCAL_PATH.read_bytes()
        self.worker_sha256 = content_sha256(self.worker_bytes)
        self.worker_relative = f"control/bin/r2-map-remote-worker-{self.worker_sha256}.py"
        self.worker_remote_path = REMOTE_ROOT / self.worker_relative
        self._protocol_verified = False

    def install_worker(self) -> dict[str, Any]:
        configuration = self.transport.verify_local_configuration()
        descriptor = {
            "schema_id": BOOTSTRAP_SCHEMA,
            "root": str(REMOTE_ROOT),
            "host_identity_sha256": REMOTE_IDENTITY_SHA256,
            "worker_sha256": self.worker_sha256,
            "worker_size": len(self.worker_bytes),
        }
        command_sha256 = content_sha256(canonical_json(descriptor))
        argv = [
            "/usr/bin/python3",
            "-c",
            BOOTSTRAP_PROGRAM,
            str(REMOTE_ROOT),
            REMOTE_IDENTITY_SHA256,
            str(worker.PRODUCTION_ROOT_DEVICE),
            str(worker.PRODUCTION_ROOT_INODE),
            self.worker_sha256,
            str(len(self.worker_bytes)),
            command_sha256,
        ]
        return_code, stdout, stderr, stream_sha256, stream_size = self.transport.run(
            argv, [self.worker_bytes]
        )
        if return_code != 0:
            raise RemoteTransportError(_transport_message("worker bootstrap failed", stderr))
        try:
            receipt = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteProtocolError("worker bootstrap did not return canonical JSON") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_id") != BOOTSTRAP_SCHEMA
            or receipt.get("status") != "ok"
            or receipt.get("command_sha256") != command_sha256
            or receipt.get("worker_sha256") != self.worker_sha256
            or receipt.get("worker_size") != len(self.worker_bytes)
            or receipt.get("host_identity_sha256") != REMOTE_IDENTITY_SHA256
            or receipt.get("root") != str(REMOTE_ROOT)
            or receipt.get("receipt_sha256") != document_sha256(receipt, "receipt_sha256")
            or stream_sha256 != self.worker_sha256
            or stream_size != len(self.worker_bytes)
        ):
            raise RemoteProtocolError("worker bootstrap receipt failed verification")
        return {**receipt, "ssh": configuration}

    def _request(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if request_id is not None and (
            not request_id.startswith("req-")
            or worker.IDENTIFIER.fullmatch(request_id) is None
        ):
            raise ValueError("remote request id is not a safe req- identifier")
        normalized_arguments = dict(arguments)
        semantic_sha256 = worker.request_semantic_sha256(operation, normalized_arguments)
        generated_id = (
            "req-%s-%s" % (operation, semantic_sha256[:32])
            if operation in worker.STABLE_MUTATING_OPERATIONS
            else f"req-{uuid.uuid4().hex}"
        )
        request: dict[str, Any] = {
            "schema_version": 1,
            "schema_id": worker.COMMAND_SCHEMA,
            "request_id": request_id or generated_id,
            "issued_unix_ms": time.time_ns() // 1_000_000,
            "root": str(REMOTE_ROOT),
            "worker_sha256": self.worker_sha256,
            "operation": operation,
            "arguments": normalized_arguments,
            "semantic_sha256": semantic_sha256,
        }
        request["command_sha256"] = worker.request_command_sha256(request)
        return request

    def execute(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        chunks: Iterable[bytes] = (),
        *,
        timeout_seconds: int | None = None,
        request_id: str | None = None,
    ) -> RemoteResult:
        mutating_operations = {
            "put-file",
            "publish-status",
            "put-stream",
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
        }
        if operation in mutating_operations and not self._protocol_verified:
            negotiation = self.execute("protocol-info", {})
            expected = {
                **worker.protocol_info(),
                "payload_size": 0,
                "payload_sha256": content_sha256(b""),
            }
            if negotiation.payload or negotiation.result != expected:
                raise RemoteProtocolError("remote worker protocol negotiation differs")
            self._protocol_verified = True
        request = self._request(operation, arguments, request_id=request_id)
        if operation in worker.STABLE_MUTATING_OPERATIONS:
            query = self.execute(
                "query-receipt",
                {
                    "request_id": request["request_id"],
                    "semantic_sha256": request["semantic_sha256"],
                    "command_sha256": request["command_sha256"],
                    "operation": operation,
                },
            )
            if set(query.result) != {
                "found",
                "receipt",
                "journal_present",
                "receipt_reservation_present",
                "data_reservation_present",
                "payload_size",
                "payload_sha256",
            }:
                raise RemoteProtocolError("receipt query fields differ")
            found = query.result.get("found")
            stored = query.result.get("receipt")
            if found is True:
                if (
                    not isinstance(stored, dict)
                    or stored.get("schema_id") != worker.RECEIPT_SCHEMA
                    or stored.get("request_id") != request["request_id"]
                    or stored.get("semantic_sha256") != request["semantic_sha256"]
                    or stored.get("command_sha256") != request["command_sha256"]
                    or stored.get("operation") != operation
                    or stored.get("status") != "ok"
                    or stored.get("receipt_sha256")
                    != document_sha256(stored, "receipt_sha256")
                    or stored.get("result", {}).get("payload_size") != 0
                    or stored.get("result", {}).get("payload_sha256")
                    != content_sha256(b"")
                    or query.result.get("journal_present") is not False
                    or query.result.get("receipt_reservation_present") is not False
                    or query.result.get("data_reservation_present") is not False
                ):
                    raise RemoteProtocolError("queried receipt binding differs")
                if operation in {
                    "put-file",
                    "publish-status",
                    "transaction-begin",
                    "transaction-put",
                }:
                    input_size = int(arguments["size"])
                    input_sha256 = str(arguments["sha256"])
                elif operation == "put-stream":
                    input_size = int(stored["result"]["size"])
                    input_sha256 = str(stored["result"]["sha256"])
                else:
                    input_size = 0
                    input_sha256 = content_sha256(b"")
                if input_size or operation in {
                    "put-file",
                    "put-stream",
                    "publish-status",
                    "transaction-begin",
                    "transaction-put",
                }:
                    replay_digest = hashlib.sha256()
                    replay_size = 0
                    maximum = (
                        int(arguments["max_bytes"])
                        if operation == "put-stream"
                        else input_size
                    )
                    for chunk in chunks:
                        if not isinstance(chunk, bytes):
                            raise TypeError("SSH stream chunks must be bytes")
                        replay_size += len(chunk)
                        if replay_size > maximum:
                            raise RemoteProtocolError("cached replay body exceeds its bound")
                        replay_digest.update(chunk)
                    if replay_size != input_size or replay_digest.hexdigest() != input_sha256:
                        raise RemoteProtocolError("cached replay body identity differs")
                return RemoteResult(
                    payload=b"",
                    receipt=stored,
                    input_sha256=input_sha256,
                    input_size=input_size,
                )
            if found is not False or stored is not None:
                raise RemoteProtocolError("receipt query result differs")
        encoded = base64.urlsafe_b64encode(canonical_json(request)).rstrip(b"=").decode("ascii")
        argv = [
            "/usr/bin/python3",
            str(self.worker_remote_path),
            "--worker-sha256",
            self.worker_sha256,
            "--request-base64",
            encoded,
        ]
        return_code, stdout, stderr, input_sha256, input_size = self.transport.run(
            argv, chunks, timeout_seconds=timeout_seconds
        )
        try:
            result = self._decode_response(
                stdout,
                request,
                input_sha256=input_sha256,
                input_size=input_size,
            )
        except RemoteProtocolError as error:
            if return_code != 0:
                raise RemoteTransportError(
                    _transport_message("remote worker transport failed", stderr)
                ) from error
            raise
        if result.receipt["status"] != "ok":
            message = str(result.receipt.get("result", {}).get("error", "remote operation failed"))
            raise RemoteOperationError(message, result.receipt)
        if return_code != 0:
            raise RemoteProtocolError("SSH returned nonzero for a successful authenticated receipt")
        return result

    def _decode_response(
        self,
        encoded: bytes,
        request: Mapping[str, Any],
        *,
        input_sha256: str,
        input_size: int,
    ) -> RemoteResult:
        if len(encoded) < worker.FRAME_PREFIX.size + worker.FRAME_SUFFIX.size:
            raise RemoteProtocolError("worker response is shorter than the frame minimum")
        magic, header_size = worker.FRAME_PREFIX.unpack_from(encoded)
        if magic != worker.FRAME_MAGIC or header_size > (1 << 20):
            raise RemoteProtocolError("worker frame magic/header bound is invalid")
        header_start = worker.FRAME_PREFIX.size
        header_end = header_start + header_size
        try:
            header = json.loads(encoded[header_start:header_end])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteProtocolError("worker frame header is invalid") from error
        if (
            not isinstance(header, dict)
            or header.get("schema_id") != "cascadia.r2-map.remote-frame.v1"
            or header.get("response_disposition") not in {"committed", "replayed"}
        ):
            raise RemoteProtocolError("worker frame header schema is invalid")
        payload_size = header.get("payload_size")
        if not isinstance(payload_size, int) or not 0 <= payload_size <= worker.MAX_RANGE_BYTES:
            raise RemoteProtocolError("worker payload size is outside the protocol bound")
        payload_start = header_end
        payload_end = payload_start + payload_size
        if payload_end + worker.FRAME_SUFFIX.size > len(encoded):
            raise RemoteProtocolError("worker payload is truncated")
        (receipt_size,) = worker.FRAME_SUFFIX.unpack_from(encoded, payload_end)
        receipt_start = payload_end + worker.FRAME_SUFFIX.size
        receipt_end = receipt_start + receipt_size
        if receipt_end != len(encoded) or receipt_size > (1 << 20):
            raise RemoteProtocolError("worker receipt length is invalid")
        payload = encoded[payload_start:payload_end]
        try:
            receipt = json.loads(encoded[receipt_start:receipt_end])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteProtocolError("worker receipt JSON is invalid") from error
        if (
            header.get("payload_sha256") != content_sha256(payload)
            or header.get("receipt_size") != receipt_size
            or not isinstance(receipt, dict)
            or receipt.get("schema_id") != worker.RECEIPT_SCHEMA
            or receipt.get("request_id") != request["request_id"]
            or receipt.get("semantic_sha256") != request["semantic_sha256"]
            or receipt.get("command_sha256") != request["command_sha256"]
            or receipt.get("operation") != request["operation"]
            or receipt.get("host") != REMOTE_HOST_ALIAS
            or receipt.get("host_identity_sha256") != REMOTE_IDENTITY_SHA256
            or receipt.get("root") != str(REMOTE_ROOT)
            or receipt.get("receipt_sha256") != document_sha256(receipt, "receipt_sha256")
            or (
                header.get("response_disposition") == "replayed"
                and request["operation"] not in worker.STABLE_MUTATING_OPERATIONS
            )
            or receipt.get("result", {}).get("payload_size") != payload_size
            or receipt.get("result", {}).get("payload_sha256") != content_sha256(payload)
        ):
            raise RemoteProtocolError("worker frame/receipt binding failed verification")
        return RemoteResult(
            payload=payload,
            receipt=receipt,
            input_sha256=input_sha256,
            input_size=input_size,
        )

    def provision(self) -> dict[str, Any]:
        response = self.execute("provision", {})
        return {
            **response.result,
            **response.storage_receipt,
        }

    def preflight(self) -> dict[str, Any]:
        response = self.execute("preflight", {}, timeout_seconds=600)
        return {
            **response.result,
            **response.storage_receipt,
        }

    def open_object(self, relative: str) -> dict[str, Any]:
        return self.open_object_with_receipt(relative)["object_token"]

    def open_object_with_receipt(self, relative: str) -> dict[str, Any]:
        _validate_relative(relative, "object")
        response = self.execute("open-object", {"relative": relative})
        token = response.result.get("object_token")
        if not isinstance(token, dict) or token.get("token_sha256") != document_sha256(
            token, "token_sha256"
        ):
            raise RemoteProtocolError("object token failed local verification")
        return {
            "object_token": token,
            **response.storage_receipt,
        }

    def read_range(
        self,
        object_token: Mapping[str, Any],
        offset: int,
        length: int,
        *,
        max_bytes: int = worker.MAX_RANGE_BYTES,
    ) -> bytes:
        return self.read_range_with_receipt(object_token, offset, length, max_bytes=max_bytes)[
            "payload"
        ]

    def read_range_with_receipt(
        self,
        object_token: Mapping[str, Any],
        offset: int,
        length: int,
        *,
        max_bytes: int = worker.MAX_RANGE_BYTES,
    ) -> dict[str, Any]:
        if not 0 <= length <= max_bytes <= worker.MAX_RANGE_BYTES:
            raise ValueError("range length/max_bytes exceeds the 64 MiB transport bound")
        result = self.execute(
            "read-range",
            {
                "object_token": dict(object_token),
                "offset": offset,
                "length": length,
                "max_bytes": max_bytes,
            },
        )
        if (
            result.result.get("payload_sha256") != content_sha256(result.payload)
            or result.result.get("object_token_sha256") != object_token.get("token_sha256")
            or result.result.get("offset") != offset
            or result.result.get("length") != length
        ):
            raise RemoteProtocolError("range receipt payload identity differs")
        return {
            "payload": result.payload,
            "payload_sha256": content_sha256(result.payload),
            "object_token_sha256": object_token.get("token_sha256"),
            "offset": offset,
            "length": length,
            **result.storage_receipt,
        }

    def iter_object(
        self, object_token: Mapping[str, Any], *, window_bytes: int = worker.MAX_RANGE_BYTES
    ) -> Iterator[bytes]:
        if not 1 <= window_bytes <= worker.MAX_RANGE_BYTES:
            raise ValueError("object window must be between 1 byte and 64 MiB")
        size = object_token.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError("object token size is invalid")
        for offset in range(0, size, window_bytes):
            yield self.read_range(
                object_token, offset, min(window_bytes, size - offset), max_bytes=window_bytes
            )

    def iter_object_with_receipts(
        self, object_token: Mapping[str, Any], *, window_bytes: int = worker.MAX_RANGE_BYTES
    ) -> Iterator[dict[str, Any]]:
        if not 1 <= window_bytes <= worker.MAX_RANGE_BYTES:
            raise ValueError("object window must be between 1 byte and 64 MiB")
        size = object_token.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError("object token size is invalid")
        for offset in range(0, size, window_bytes):
            yield self.read_range_with_receipt(
                object_token, offset, min(window_bytes, size - offset), max_bytes=window_bytes
            )

    def put_bytes(
        self,
        relative: str,
        payload: bytes,
        *,
        expected_current: str = "absent",
        mutable: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return self.put_stream(
            relative,
            [payload],
            size=len(payload),
            sha256=content_sha256(payload),
            expected_current=expected_current,
            mutable=mutable,
            request_id=request_id,
        )

    def put_stream(
        self,
        relative: str,
        chunks: Iterable[bytes],
        *,
        size: int,
        sha256: str,
        expected_current: str = "absent",
        mutable: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        _validate_relative(relative, "upload")
        _validate_sha256(sha256, "upload SHA-256")
        if size < 0:
            raise ValueError("upload size cannot be negative")
        response = self.execute(
            "put-file",
            {
                "relative": relative,
                "size": size,
                "sha256": sha256,
                "expected_current": expected_current,
                "mutable": mutable,
            },
            chunks,
            timeout_seconds=3600,
            request_id=request_id,
        )
        result = response.result
        if (
            result.get("sha256") != sha256
            or result.get("size") != size
            or response.input_sha256 != sha256
            or response.input_size != size
        ):
            raise RemoteProtocolError("upload receipt differs from requested object")
        return {**result, **response.storage_receipt}

    def put_unknown_stream(
        self,
        relative: str,
        chunks: Iterable[bytes],
        *,
        max_bytes: int,
        expected_current: str = "absent",
    ) -> dict[str, Any]:
        _validate_relative(relative, "unknown-size stream")
        if not 1 <= max_bytes <= worker.MAX_UNKNOWN_STREAM_BYTES:
            raise ValueError("unknown-size stream bound exceeds the 1 GiB server limit")
        response = self.execute(
            "put-stream",
            {
                "relative": relative,
                "max_bytes": max_bytes,
                "expected_current": expected_current,
            },
            chunks,
            timeout_seconds=3600,
        )
        result = response.result
        if (
            result.get("sha256") != response.input_sha256
            or result.get("size") != response.input_size
        ):
            raise RemoteProtocolError("unknown-size stream differs across the SSH boundary")
        return {**result, **response.storage_receipt}

    def publish_status(self, payload: bytes, *, expected_current: str = "absent") -> dict[str, Any]:
        if len(payload) > worker.MAX_STATUS_BYTES:
            raise ValueError("dashboard projection exceeds 64 KiB")
        response = self.execute(
            "publish-status",
            {
                "relative": "control/dashboard-status.json",
                "size": len(payload),
                "sha256": content_sha256(payload),
                "expected_current": expected_current,
                "mutable": True,
            },
            [payload],
        )
        if response.input_sha256 != content_sha256(payload) or response.input_size != len(payload):
            raise RemoteProtocolError("status publication changed across the SSH boundary")
        return {
            **response.result,
            **response.storage_receipt,
        }

    def acquire_lock(
        self,
        name: str,
        owner: str,
        lease_seconds: int = 300,
        *,
        lease_epoch: str,
    ) -> dict[str, Any]:
        response = self.execute(
            "lock-acquire",
            {
                "name": name,
                "owner": owner,
                "lease_seconds": lease_seconds,
                "lease_epoch": lease_epoch,
            },
        )
        return {
            **response.result,
            **response.storage_receipt,
        }

    def renew_lock(
        self,
        name: str,
        owner: str,
        token: str,
        lease_seconds: int = 300,
        *,
        lease_epoch: str,
    ) -> dict[str, Any]:
        response = self.execute(
            "lock-renew",
            {
                "name": name,
                "owner": owner,
                "token": token,
                "lease_seconds": lease_seconds,
                "lease_epoch": lease_epoch,
            },
        )
        return {
            **response.result,
            **response.storage_receipt,
        }

    def release_lock(
        self,
        name: str,
        owner: str,
        token: str,
        lease_seconds: int = 300,
        *,
        lease_epoch: str,
    ) -> dict[str, Any]:
        response = self.execute(
            "lock-release",
            {
                "name": name,
                "owner": owner,
                "token": token,
                "lease_seconds": lease_seconds,
                "lease_epoch": lease_epoch,
            },
        )
        return {
            **response.result,
            **response.storage_receipt,
        }

    def begin_transaction(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        payload = canonical_json(manifest)
        if manifest.get("manifest_sha256") != document_sha256(manifest, "manifest_sha256"):
            raise ValueError("transaction manifest identity is invalid")
        response = self.execute(
            "transaction-begin",
            {"size": len(payload), "sha256": content_sha256(payload)},
            [payload],
        )
        if response.input_sha256 != content_sha256(payload) or response.input_size != len(payload):
            raise RemoteProtocolError("transaction manifest changed across the SSH boundary")
        return {
            **response.result,
            **response.storage_receipt,
        }

    def put_transaction_object(
        self,
        transaction_id: str,
        object_descriptor: TransactionObject,
        chunks: Iterable[bytes],
    ) -> dict[str, Any]:
        descriptor = object_descriptor.to_dict()
        response = self.execute(
            "transaction-put",
            {"transaction_id": transaction_id, **descriptor},
            chunks,
            timeout_seconds=3600,
        )
        if (
            response.input_sha256 != object_descriptor.sha256
            or response.input_size != object_descriptor.size
        ):
            raise RemoteProtocolError("transaction object changed across the SSH boundary")
        return {
            **response.result,
            **response.storage_receipt,
        }

    def import_transaction_object(
        self,
        transaction_id: str,
        object_descriptor: TransactionObject,
        *,
        source_relative: str,
    ) -> dict[str, Any]:
        _validate_relative(source_relative, "transaction import source")
        descriptor = object_descriptor.to_dict()
        response = self.execute(
            "transaction-import",
            {
                "transaction_id": transaction_id,
                **descriptor,
                "source_relative": source_relative,
            },
            timeout_seconds=3600,
        )
        return {
            **response.result,
            **response.storage_receipt,
        }

    def commit_transaction(self, transaction_id: str, manifest_sha256: str) -> dict[str, Any]:
        response = self.execute(
            "transaction-commit",
            {"transaction_id": transaction_id, "manifest_sha256": manifest_sha256},
            timeout_seconds=3600,
        )
        return {
            **response.result,
            **response.storage_receipt,
        }

    def abort_transaction(self, transaction_id: str, manifest_sha256: str) -> dict[str, Any]:
        response = self.execute(
            "transaction-abort",
            {"transaction_id": transaction_id, "manifest_sha256": manifest_sha256},
        )
        return {
            **response.result,
            **response.storage_receipt,
        }

    def run_remote(
        self,
        *,
        run_id: str,
        cwd_relative: str,
        argv: Sequence[str],
        output_relative: str,
        timeout_seconds: int,
        environment: Mapping[str, str] | None = None,
        python_path_relatives: Sequence[str] = (),
    ) -> dict[str, Any]:
        response = self.execute(
            "run-command",
            {
                "run_id": run_id,
                "cwd_relative": cwd_relative,
                "argv": list(argv),
                "output_relative": output_relative,
                "timeout_seconds": timeout_seconds,
                "environment": dict(environment or {}),
                "python_path_relatives": list(python_path_relatives),
            },
            timeout_seconds=timeout_seconds + 120,
        )
        return {
            **response.result,
            "run_receipt_sha256": response.receipt["receipt_sha256"],
            **response.storage_receipt,
        }

    def run_controller(
        self,
        *,
        run_id: str,
        source_manifest_sha256: str,
        cwd_relative: str,
        executable_relative: str,
        arguments: Sequence[str],
        output_relative: str,
        timeout_seconds: int = 600,
        python_path_relatives: Sequence[str] = (),
    ) -> dict[str, Any]:
        _validate_sha256(source_manifest_sha256, "controller source manifest SHA-256")
        _validate_relative(executable_relative, "controller executable")
        executable = str(REMOTE_ROOT / executable_relative)
        response = self.execute(
            "run-controller",
            {
                "run_id": run_id,
                "source_manifest_sha256": source_manifest_sha256,
                "cwd_relative": cwd_relative,
                "argv": [executable, *arguments],
                "output_relative": output_relative,
                "timeout_seconds": timeout_seconds,
                "environment": {},
                "python_path_relatives": list(python_path_relatives),
            },
            timeout_seconds=timeout_seconds + 120,
        )
        if response.result.get("controller_mode") is not True:
            raise RemoteProtocolError("controller run receipt omitted controller mode")
        return {
            **response.result,
            "run_receipt_sha256": response.receipt["receipt_sha256"],
            **response.storage_receipt,
        }

    def open_ephemeral_run_outputs(
        self,
        *,
        run_id: str,
        manifest_relative: str,
        dataset_relative: str,
    ) -> dict[str, Any]:
        if not worker.IDENTIFIER.fullmatch(run_id):
            raise ValueError("ephemeral output run_id is not a safe identifier")
        required_prefix = f"build/run-{run_id}/"
        for relative, suffix, label in (
            (manifest_relative, ".json", "manifest"),
            (dataset_relative, ".r2map", "dataset"),
        ):
            _validate_relative(relative, f"ephemeral output {label}")
            if not relative.startswith(required_prefix) or not relative.endswith(suffix):
                raise ValueError(f"ephemeral output {label} is outside the exact run")
        manifest = self.open_object_with_receipt(manifest_relative)
        dataset = self.open_object_with_receipt(dataset_relative)
        if manifest["object_token"]["size"] > worker.MAX_MANIFEST_BYTES:
            raise RemoteProtocolError("ephemeral output manifest exceeds 2 MiB")
        if dataset["object_token"]["size"] > worker.MAX_UNKNOWN_STREAM_BYTES:
            raise RemoteProtocolError("ephemeral output dataset exceeds 1 GiB")
        return {
            "run_id": run_id,
            "manifest": manifest,
            "dataset": dataset,
        }

    def prepare_run_cleanup(
        self,
        *,
        run_id: str,
        manifest_object_token: Mapping[str, Any],
        dataset_object_token: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = self.execute(
            "run-cleanup-prepare",
            {
                "run_id": run_id,
                "manifest_object_token": dict(manifest_object_token),
                "dataset_object_token": dict(dataset_object_token),
            },
            timeout_seconds=3600,
        )
        cleanup_token = response.result.get("cleanup_token")
        if (
            not isinstance(cleanup_token, dict)
            or cleanup_token.get("schema_id") != worker.RUN_CLEANUP_TOKEN_SCHEMA
            or cleanup_token.get("cleanup_token_sha256")
            != document_sha256(cleanup_token, "cleanup_token_sha256")
        ):
            raise RemoteProtocolError("run cleanup token failed local verification")
        return {
            "cleanup_token": cleanup_token,
            **response.storage_receipt,
        }

    def commit_run_cleanup(self, cleanup_token: Mapping[str, Any]) -> dict[str, Any]:
        if cleanup_token.get("cleanup_token_sha256") != document_sha256(
            cleanup_token, "cleanup_token_sha256"
        ):
            raise ValueError("run cleanup token identity is invalid")
        response = self.execute(
            "run-cleanup-commit",
            {"cleanup_token": dict(cleanup_token)},
            timeout_seconds=3600,
        )
        if response.result.get("cleanup_token_sha256") != cleanup_token.get("cleanup_token_sha256"):
            raise RemoteProtocolError("run cleanup receipt differs from its CAS token")
        return {
            **response.result,
            **response.storage_receipt,
        }

    def prepare_failed_run_cleanup(self, *, run_id: str) -> dict[str, Any]:
        if not worker.IDENTIFIER.fullmatch(run_id):
            raise ValueError("failed-run cleanup run_id is not a safe identifier")
        response = self.execute(
            "failed-run-cleanup-prepare",
            {"run_id": run_id},
            timeout_seconds=3600,
        )
        cleanup_token = response.result.get("cleanup_token")
        if (
            not isinstance(cleanup_token, dict)
            or cleanup_token.get("schema_id") != worker.FAILED_RUN_CLEANUP_TOKEN_SCHEMA
            or cleanup_token.get("cleanup_token_sha256")
            != document_sha256(cleanup_token, "cleanup_token_sha256")
        ):
            raise RemoteProtocolError("failed-run cleanup token failed local verification")
        return {"cleanup_token": cleanup_token, **response.storage_receipt}

    def commit_failed_run_cleanup(self, cleanup_token: Mapping[str, Any]) -> dict[str, Any]:
        if cleanup_token.get(
            "schema_id"
        ) != worker.FAILED_RUN_CLEANUP_TOKEN_SCHEMA or cleanup_token.get(
            "cleanup_token_sha256"
        ) != document_sha256(cleanup_token, "cleanup_token_sha256"):
            raise ValueError("failed-run cleanup token identity is invalid")
        response = self.execute(
            "failed-run-cleanup-commit",
            {"cleanup_token": dict(cleanup_token)},
            timeout_seconds=3600,
        )
        if response.result.get("cleanup_token_sha256") != cleanup_token.get("cleanup_token_sha256"):
            raise RemoteProtocolError("failed-run cleanup receipt differs from its CAS token")
        return {**response.result, **response.storage_receipt}

    def inspect_remote_executable(self, relative: str, *, size: int, sha256: str) -> dict[str, Any]:
        response = self.execute(
            "inspect-executable",
            {"relative": relative, "size": size, "sha256": sha256},
        )
        opened = self.open_object_with_receipt(relative)
        token = opened["object_token"]
        if token["sha256"] != sha256 or token["size"] != size:
            raise RemoteProtocolError("executable changed between inspection and content read")
        sha_hasher = hashlib.sha256()
        blake3_hasher = blake3.blake3()
        read_receipts = []
        observed_size = 0
        for read in self.iter_object_with_receipts(token):
            payload = read["payload"]
            sha_hasher.update(payload)
            blake3_hasher.update(payload)
            observed_size += len(payload)
            read_receipts.append({key: value for key, value in read.items() if key != "payload"})
        if observed_size != size or sha_hasher.hexdigest() != sha256:
            raise RemoteProtocolError("executable content read differs from its inspection")
        return {
            **response.result,
            "blake3": blake3_hasher.hexdigest(),
            "content_open_receipt": {
                key: value for key, value in opened.items() if key != "object_token"
            },
            "content_read_receipts": read_receipts,
            "inspection_receipt_sha256": response.receipt["receipt_sha256"],
            **response.storage_receipt,
        }


def _transport_message(prefix: str, stderr: bytes) -> str:
    bounded = stderr[:MAX_STDERR_BYTES].decode("utf-8", errors="replace").strip()
    return f"{prefix}: {bounded}" if bounded else prefix


def bytes_chunks(source: BinaryIO, chunk_bytes: int = 1 << 20) -> Iterator[bytes]:
    if not 1 <= chunk_bytes <= worker.MAX_RANGE_BYTES:
        raise ValueError("chunk size is outside the bounded transport range")
    while chunk := source.read(chunk_bytes):
        yield chunk


def build_john1_runtime_manifest(
    *,
    packet_id: str,
    executable_relative: str,
    inspection: Mapping[str, Any],
    source_freeze: Mapping[str, Any],
    build_receipt_relative: str,
    build_receipt_sha256: str,
    output_prefix_relative: str,
    stdout_max_bytes: int,
    stderr_max_bytes: int,
    created_unix_ms: int,
) -> dict[str, Any]:
    if not worker.IDENTIFIER.fullmatch(packet_id):
        raise ValueError("runtime packet_id is not a safe identifier")
    _validate_relative(executable_relative, "runtime executable")
    executable_parts = PurePosixPath(executable_relative).parts
    if executable_parts[0] != "bundles":
        raise ValueError("runtime executable must be in an immutable bundles/ transaction")
    _validate_relative(output_prefix_relative, "runtime output prefix")
    if PurePosixPath(output_prefix_relative).parts[:2] != ("logs", "generation"):
        raise ValueError("runtime output prefix must be below logs/generation/")
    frozen_source = _validate_source_freeze(source_freeze)
    _validate_storage_receipt_relative(build_receipt_relative, "build receipt")
    _validate_sha256(build_receipt_sha256, "build receipt SHA-256")
    inspection_receipt_relative = _validate_storage_receipt_relative(
        inspection.get("storage_receipt_relative"), "inspection receipt"
    )
    inspection_receipt = _validate_sha256(
        str(inspection.get("inspection_receipt_sha256", "")),
        "inspection receipt SHA-256",
    )
    if (
        inspection.get("schema_id") != "cascadia.r2-map.ephemeral-executable-inspection.v1"
        or inspection.get("mach_o_arches") != ["arm64"]
        or inspection.get("codesign", {}).get("verified") is not True
        or not inspection.get("codesign", {}).get("designated_requirement")
        or inspection.get("codesign", {}).get("designated_requirement_sha256")
        != content_sha256(
            str(inspection.get("codesign", {}).get("designated_requirement", "")).encode()
        )
    ):
        raise ValueError("runtime executable inspection is not signed thin arm64 evidence")
    size = inspection.get("size")
    sha256 = inspection.get("sha256")
    executable_blake3 = inspection.get("blake3")
    if not isinstance(size, int) or not 0 < size < worker.MAX_EPHEMERAL_RUNTIME_BYTES:
        raise ValueError("runtime executable size is invalid")
    _validate_sha256(str(sha256), "runtime executable SHA-256")
    _validate_sha256(str(executable_blake3), "runtime executable BLAKE3")
    if not (
        1 <= stdout_max_bytes <= worker.MAX_UNKNOWN_STREAM_BYTES
        and 1 <= stderr_max_bytes <= worker.MAX_UNKNOWN_STREAM_BYTES
    ):
        raise ValueError("runtime stream bound exceeds the 1 GiB remote stream limit")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": EPHEMERAL_RUNTIME_MANIFEST_SCHEMA,
        "packet_id": packet_id,
        "canonical_storage_host": REMOTE_HOST_ALIAS,
        "canonical_storage_root": str(REMOTE_ROOT),
        "staging_host": "john1",
        "staging_root": str(JOHN1_STAGING_ROOT),
        "combined_packet_max_bytes": worker.MAX_EPHEMERAL_RUNTIME_BYTES,
        "executable": {
            "relative": executable_relative,
            "sha256": sha256,
            "blake3": executable_blake3,
            "size": size,
            "staged_mode": "0500",
            "mach_o_arches": ["arm64"],
            "file_description": inspection.get("file_description"),
            "codesign": inspection.get("codesign"),
        },
        "provenance": {
            "source_freeze": frozen_source,
            "source_freeze_identity_sha256": content_sha256(canonical_json(frozen_source)),
            "build_receipt_relative": build_receipt_relative,
            "build_receipt_sha256": build_receipt_sha256,
            "inspection_source_relative": inspection.get("relative"),
            "inspection_receipt_relative": inspection_receipt_relative,
            "inspection_receipt_sha256": inspection_receipt,
        },
        "output": {
            "prefix_relative": output_prefix_relative,
            "stdout_max_bytes": stdout_max_bytes,
            "stderr_max_bytes": stderr_max_bytes,
        },
        "created_unix_ms": created_unix_ms,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    encoded = canonical_json(manifest)
    if len(encoded) > worker.MAX_EPHEMERAL_MANIFEST_BYTES:
        raise ValueError("runtime manifest exceeds 64 KiB")
    if size + len(encoded) > worker.MAX_EPHEMERAL_RUNTIME_BYTES:
        raise ValueError("runtime executable plus manifest exceeds 64 MiB")
    return manifest


def validate_john1_runtime_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("schema_id") != EPHEMERAL_RUNTIME_MANIFEST_SCHEMA
        or manifest.get("canonical_storage_host") != REMOTE_HOST_ALIAS
        or manifest.get("canonical_storage_root") != str(REMOTE_ROOT)
        or manifest.get("staging_host") != "john1"
        or manifest.get("staging_root") != str(JOHN1_STAGING_ROOT)
        or manifest.get("combined_packet_max_bytes") != worker.MAX_EPHEMERAL_RUNTIME_BYTES
        or manifest.get("manifest_sha256") != document_sha256(manifest, "manifest_sha256")
    ):
        raise RemoteProtocolError("ephemeral runtime manifest identity is invalid")
    packet_id = manifest.get("packet_id")
    if not isinstance(packet_id, str) or not worker.IDENTIFIER.fullmatch(packet_id):
        raise RemoteProtocolError("ephemeral runtime packet identifier is invalid")
    executable = manifest.get("executable")
    output = manifest.get("output")
    provenance = manifest.get("provenance")
    if not all(isinstance(value, dict) for value in (executable, output, provenance)):
        raise RemoteProtocolError("ephemeral runtime manifest sections are invalid")
    relative = _validate_relative(str(executable.get("relative", "")), "runtime executable")
    if PurePosixPath(relative).parts[0] != "bundles":
        raise RemoteProtocolError("ephemeral runtime executable is not in bundles/")
    size = executable.get("size")
    if (
        not isinstance(size, int)
        or not 0 < size < worker.MAX_EPHEMERAL_RUNTIME_BYTES
        or executable.get("staged_mode") != "0500"
        or executable.get("mach_o_arches") != ["arm64"]
        or executable.get("codesign", {}).get("verified") is not True
    ):
        raise RemoteProtocolError("ephemeral runtime executable contract is invalid")
    _validate_sha256(str(executable.get("sha256", "")), "runtime executable SHA-256")
    _validate_sha256(str(executable.get("blake3", "")), "runtime executable BLAKE3")
    codesign = executable.get("codesign", {})
    designated = codesign.get("designated_requirement")
    if (
        not isinstance(designated, str)
        or not designated
        or codesign.get("designated_requirement_sha256") != content_sha256(designated.encode())
    ):
        raise RemoteProtocolError("ephemeral runtime designated requirement is invalid")
    try:
        frozen_source = _validate_source_freeze(provenance.get("source_freeze", {}))
    except ValueError as error:
        raise RemoteProtocolError("ephemeral runtime source freeze identity is invalid") from error
    if provenance.get("source_freeze_identity_sha256") != content_sha256(
        canonical_json(frozen_source)
    ):
        raise RemoteProtocolError("ephemeral runtime source freeze digest is invalid")
    for field in ("build_receipt_relative", "inspection_receipt_relative"):
        try:
            _validate_storage_receipt_relative(provenance.get(field), field)
        except ValueError as error:
            raise RemoteProtocolError("ephemeral runtime receipt locator is invalid") from error
    for field in ("build_receipt_sha256", "inspection_receipt_sha256"):
        _validate_sha256(str(provenance.get(field, "")), field)
    prefix = _validate_relative(str(output.get("prefix_relative", "")), "runtime output")
    if PurePosixPath(prefix).parts[:2] != ("logs", "generation"):
        raise RemoteProtocolError("ephemeral runtime output prefix is invalid")
    for field in ("stdout_max_bytes", "stderr_max_bytes"):
        bound = output.get(field)
        if not isinstance(bound, int) or not 1 <= bound <= worker.MAX_UNKNOWN_STREAM_BYTES:
            raise RemoteProtocolError("ephemeral runtime stream bound is invalid")
    encoded = canonical_json(manifest)
    if (
        len(encoded) > worker.MAX_EPHEMERAL_MANIFEST_BYTES
        or size + len(encoded) > worker.MAX_EPHEMERAL_RUNTIME_BYTES
    ):
        raise RemoteProtocolError("ephemeral runtime packet exceeds 64 MiB")
    return manifest


@dataclass(frozen=True)
class StagedJohn1Runtime:
    manifest: dict[str, Any]
    directory: Path
    executable: Path
    manifest_path: Path
    remote_manifest_sha256: str
    john1_staging_proof: dict[str, Any]


def _verify_john1_staging_host() -> dict[str, Any]:
    root = JOHN1_STAGING_ROOT
    details = os.lstat(root)
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o1777
        or not details.st_mode & stat.S_ISVTX
        or details.st_uid != 0
        or details.st_gid != 0
    ):
        raise RemoteStorageError("/private/tmp staging root mode/owner is unsafe")
    if (
        socket.gethostname() != JOHN1_HOSTNAME
        or pwd.getpwuid(os.getuid()).pw_name != JOHN1_USER
        or os.getuid() != JOHN1_UID
    ):
        raise RemoteStorageError("ephemeral runtime staging is authorized only on john1")
    usage = shutil.disk_usage(root)
    if usage.free < 1 * (1 << 30):
        raise RemoteStorageError("john1 lacks the 1 GiB ephemeral staging reserve")
    return {
        "hostname": socket.gethostname(),
        "user": JOHN1_USER,
        "uid": JOHN1_UID,
        "staging_root": str(root),
        "staging_device": details.st_dev,
        "free_bytes": usage.free,
    }


def _atomic_local_write(
    path: Path, chunks: Iterable[bytes], expected_size: int, sha256: str
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in chunks:
            size += len(chunk)
            if size > expected_size:
                raise RemoteProtocolError("ephemeral runtime download exceeded its identity")
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RemoteStorageError("short ephemeral staging write")
                view = view[written:]
            digest.update(chunk)
        if size != expected_size or digest.hexdigest() != sha256:
            raise RemoteProtocolError("ephemeral runtime download identity differs")
        # macOS inherits /private/tmp's wheel group even when this process has
        # staff as its effective GID. Bind the packet to the frozen john1
        # uid/gid explicitly before the durable rename.
        os.fchown(descriptor, JOHN1_UID, JOHN1_GID)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _verified_remote_publication(publication: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(publication, Mapping):
        raise RemoteProtocolError(f"{label} publication result is not an object")
    receipt_sha256 = publication.get("storage_receipt_sha256")
    receipt_relative = publication.get("storage_receipt_relative")
    try:
        _validate_sha256(str(receipt_sha256), f"{label} storage receipt SHA-256")
    except ValueError as error:
        raise RemoteProtocolError(
            f"{label} lacks an authenticated John2 storage receipt"
        ) from error
    if (
        not isinstance(receipt_relative, str)
        or not receipt_relative.startswith("control/receipts/req-")
        or not receipt_relative.endswith(".json")
    ):
        raise RemoteProtocolError(f"{label} lacks a canonical John2 storage receipt locator")
    return dict(publication)


def deploy_dashboard_api(
    client: RemoteStorageClient,
    *,
    bundle_relative: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Atomically deploy one immutable John2 dashboard API bundle on John1.

    This is intentionally not a generic fetch-to-file primitive. The only
    persistent local destination is the fixed dashboard executable, and the
    source must be one immutable transaction below ``bundles/dashboard-api-*``.
    """
    _validate_relative(bundle_relative, "dashboard API bundle executable")
    _validate_sha256(expected_sha256, "dashboard API SHA-256")
    parts = PurePosixPath(bundle_relative).parts
    if (
        len(parts) != 3
        or parts[0] != "bundles"
        or not parts[1].startswith("dashboard-api-")
        or not worker.IDENTIFIER.fullmatch(parts[1])
        or parts[2] != "cascadia-api"
    ):
        raise ValueError("dashboard API must be an immutable dashboard-api bundle executable")
    if (
        socket.gethostname() != JOHN1_HOSTNAME
        or pwd.getpwuid(os.getuid()).pw_name != JOHN1_USER
        or os.getuid() != JOHN1_UID
        or os.getgid() != JOHN1_GID
    ):
        raise RemoteStorageError("dashboard API deployment is authorized only on john1")

    bundle_root = "/".join(parts[:2])
    transaction_relative = f"{bundle_root}/.r2-map-transaction.json"
    manifest_open = client.open_object_with_receipt(transaction_relative)
    manifest_token = manifest_open["object_token"]
    if manifest_token["size"] > worker.MAX_MANIFEST_BYTES:
        raise RemoteProtocolError("dashboard API transaction manifest exceeds 2 MiB")
    manifest_read = client.read_range_with_receipt(
        manifest_token,
        0,
        manifest_token["size"],
        max_bytes=worker.MAX_MANIFEST_BYTES,
    )
    try:
        transaction = json.loads(manifest_read["payload"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteProtocolError("dashboard API transaction manifest is invalid JSON") from error
    if (
        not isinstance(transaction, dict)
        or transaction.get("schema_id") != worker.TRANSACTION_SCHEMA
        or transaction.get("target_relative") != bundle_root
        or transaction.get("manifest_sha256") != document_sha256(transaction, "manifest_sha256")
    ):
        raise RemoteProtocolError("dashboard API transaction identity is invalid")
    descriptors = transaction.get("objects")
    descriptor_by_relative = (
        {
            item.get("relative"): item
            for item in descriptors
            if isinstance(item, dict) and isinstance(item.get("relative"), str)
        }
        if isinstance(descriptors, list)
        else {}
    )
    if set(descriptor_by_relative) != {"cascadia-api", "dashboard-api-manifest.json"}:
        raise RemoteProtocolError("dashboard API transaction has unexpected objects")
    descriptor = descriptor_by_relative["cascadia-api"]
    provenance_descriptor = descriptor_by_relative["dashboard-api-manifest.json"]
    if (
        descriptor.get("mode") != "0500"
        or descriptor.get("sha256") != expected_sha256
        or not isinstance(descriptor.get("size"), int)
        or not 0 < descriptor["size"] <= DASHBOARD_API_MAX_BYTES
        or "mode" in provenance_descriptor
        or not isinstance(provenance_descriptor.get("size"), int)
        or not 0 < provenance_descriptor["size"] <= worker.MAX_EPHEMERAL_MANIFEST_BYTES
    ):
        raise RemoteProtocolError("dashboard API transaction descriptor is invalid")

    provenance_relative = f"{bundle_root}/dashboard-api-manifest.json"
    provenance_open = client.open_object_with_receipt(provenance_relative)
    provenance_token = provenance_open["object_token"]
    if (
        provenance_token.get("sha256") != provenance_descriptor.get("sha256")
        or provenance_token.get("size") != provenance_descriptor.get("size")
        or provenance_token.get("mode") != 0o400
    ):
        raise RemoteProtocolError("dashboard API provenance object differs from transaction")
    provenance_read = client.read_range_with_receipt(
        provenance_token,
        0,
        provenance_token["size"],
        max_bytes=worker.MAX_EPHEMERAL_MANIFEST_BYTES,
    )
    try:
        provenance = json.loads(provenance_read["payload"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteProtocolError("dashboard API provenance is invalid JSON") from error
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != 1
        or provenance.get("schema_id") != DASHBOARD_API_BUNDLE_SCHEMA
        or provenance.get("bundle_target_relative") != bundle_root
        or provenance.get("manifest_sha256") != document_sha256(provenance, "manifest_sha256")
    ):
        raise RemoteProtocolError("dashboard API provenance identity is invalid")
    build_section = provenance.get("build")
    inspection_section = provenance.get("inspection")
    if not isinstance(build_section, dict) or not isinstance(inspection_section, dict):
        raise RemoteProtocolError("dashboard API provenance sections are invalid")
    try:
        _validate_source_freeze(provenance.get("source_freeze", {}))
        _validate_source_freeze(provenance.get("build_script_freeze", {}))
        build_receipt_relative = _validate_storage_receipt_relative(
            build_section.get("storage_receipt_relative"),
            "dashboard API build receipt",
        )
        _validate_sha256(
            str(build_section.get("storage_receipt_sha256", "")),
            "dashboard API build receipt SHA-256",
        )
        inspection_receipt_relative = _validate_storage_receipt_relative(
            inspection_section.get("storage_receipt_relative"),
            "dashboard API inspection receipt",
        )
        _validate_sha256(
            str(inspection_section.get("storage_receipt_sha256", "")),
            "dashboard API inspection receipt SHA-256",
        )
    except (TypeError, ValueError) as error:
        raise RemoteProtocolError("dashboard API provenance receipt is invalid") from error
    provenance_executable = provenance.get("executable")
    if (
        not isinstance(provenance_executable, dict)
        or provenance_executable.get("relative") != "cascadia-api"
        or provenance_executable.get("sha256") != descriptor["sha256"]
        or provenance_executable.get("size") != descriptor["size"]
        or not isinstance(provenance_executable.get("blake3"), str)
        or not worker.SHA256.fullmatch(provenance_executable["blake3"])
        or provenance_executable.get("mach_o_arches") != ["arm64"]
        or not isinstance(provenance_executable.get("codesign"), dict)
        or provenance_executable["codesign"].get("verified") is not True
        or not isinstance(build_section.get("run_id"), str)
        or not worker.IDENTIFIER.fullmatch(build_section["run_id"])
        or build_receipt_relative != build_section["storage_receipt_relative"]
        or inspection_receipt_relative != inspection_section["storage_receipt_relative"]
    ):
        raise RemoteProtocolError("dashboard API provenance executable/build is invalid")

    executable_open = client.open_object_with_receipt(bundle_relative)
    executable_token = executable_open["object_token"]
    if (
        executable_token.get("sha256") != expected_sha256
        or executable_token.get("size") != descriptor["size"]
        or executable_token.get("mode") != 0o500
    ):
        raise RemoteProtocolError("dashboard API object differs from its transaction")

    destination = JOHN1_DASHBOARD_API_PATH
    repository = JOHN1_REPOSITORY_ROOT.resolve(strict=True)
    if repository != JOHN1_REPOSITORY_ROOT:
        raise RemoteStorageError("john1 repository path is not canonical")
    current = repository
    for part in ("target", "release"):
        current = current / part
        if current.exists():
            details = os.lstat(current)
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise RemoteStorageError("dashboard API destination ancestor is unsafe")
        else:
            os.mkdir(current, 0o700)
            os.chown(current, JOHN1_UID, JOHN1_GID, follow_symlinks=False)
            _fsync_local_directory(current.parent)
    if destination.exists() or destination.is_symlink():
        details = os.lstat(destination)
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_uid != JOHN1_UID
            or details.st_gid != JOHN1_GID
            or details.st_nlink != 1
        ):
            raise RemoteStorageError("existing dashboard API destination is unsafe")
        previous_sha256 = content_sha256(destination.read_bytes())
    else:
        previous_sha256 = None

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.new")
    descriptor_fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o500,
    )
    digest = hashlib.sha256()
    blake3_digest = blake3.blake3()
    observed_size = 0
    range_receipts = []
    try:
        for read in client.iter_object_with_receipts(executable_token):
            payload = read["payload"]
            observed_size += len(payload)
            if observed_size > descriptor["size"]:
                raise RemoteProtocolError("dashboard API download exceeded its descriptor")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor_fd, view)
                if written <= 0:
                    raise RemoteStorageError("short dashboard API deployment write")
                view = view[written:]
            digest.update(payload)
            blake3_digest.update(payload)
            range_receipts.append({key: value for key, value in read.items() if key != "payload"})
        if (
            observed_size != descriptor["size"]
            or digest.hexdigest() != expected_sha256
            or blake3_digest.hexdigest() != provenance_executable["blake3"]
        ):
            raise RemoteProtocolError("dashboard API download identity differs")
        os.fchown(descriptor_fd, JOHN1_UID, JOHN1_GID)
        os.fchmod(descriptor_fd, 0o500)
        os.fsync(descriptor_fd)
    except BaseException:
        os.close(descriptor_fd)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor_fd)
    os.replace(temporary, destination)
    _fsync_local_directory(destination.parent)
    final = os.lstat(destination)
    if (
        not stat.S_ISREG(final.st_mode)
        or stat.S_IMODE(final.st_mode) != 0o500
        or final.st_uid != JOHN1_UID
        or final.st_gid != JOHN1_GID
        or final.st_nlink != 1
        or final.st_size != descriptor["size"]
        or content_sha256(destination.read_bytes()) != expected_sha256
    ):
        raise RemoteStorageError("deployed dashboard API failed final verification")
    result = {
        "schema_id": "cascadia.r2-map.dashboard-api-deployment.v1",
        "bundle_relative": bundle_relative,
        "bundle_manifest_sha256": transaction["manifest_sha256"],
        "bundle_provenance_sha256": provenance["manifest_sha256"],
        "sha256": expected_sha256,
        "size": descriptor["size"],
        "destination": str(destination),
        "previous_sha256": previous_sha256,
        "manifest_open_receipt": {
            key: value for key, value in manifest_open.items() if key != "object_token"
        },
        "manifest_read_receipt": {
            key: value for key, value in manifest_read.items() if key != "payload"
        },
        "provenance_open_receipt": {
            key: value for key, value in provenance_open.items() if key != "object_token"
        },
        "provenance_read_receipt": {
            key: value for key, value in provenance_read.items() if key != "payload"
        },
        "executable_open_receipt": {
            key: value for key, value in executable_open.items() if key != "object_token"
        },
        "executable_range_receipts": range_receipts,
        "completed_unix_ms": time.time_ns() // 1_000_000,
    }
    result["deployment_sha256"] = document_sha256(result, "deployment_sha256")
    remote = _publish_verified_bytes(
        client,
        f"control/dashboard-deployments/{parts[1]}-{result['completed_unix_ms']}.json",
        canonical_json(result),
        "dashboard API deployment",
    )
    return {**result, "deployment_remote": remote}


def _fsync_local_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_verified_bytes(
    client: RemoteStorageClient,
    relative: str,
    payload: bytes,
    label: str,
) -> dict[str, Any]:
    return _verified_remote_publication(client.put_bytes(relative, payload), label)


def _parse_designated_requirement(output: str) -> str:
    for line in output.splitlines():
        normalized = line.strip()
        if normalized.startswith("# "):
            normalized = normalized[2:]
        if normalized.startswith("designated =>"):
            return normalized.partition("=>")[2].strip()
    return ""


def _codesign_fields(path: Path) -> dict[str, Any]:
    file_result = subprocess.run(
        ["/usr/bin/file", "-b", path], check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    arches = subprocess.run(
        ["/usr/bin/lipo", "-archs", path], check=True, capture_output=True, text=True, timeout=30
    ).stdout.split()
    verify = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--strict", "--verbose=4", path],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    details = subprocess.run(
        ["/usr/bin/codesign", "-d", "--verbose=4", path],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    requirement = subprocess.run(
        ["/usr/bin/codesign", "-d", "-r-", path],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    parsed: dict[str, str] = {}
    detail_output = "\n".join(value for value in (details.stdout, details.stderr) if value)
    normalized_detail = "\n".join(
        line for line in detail_output.splitlines() if not line.startswith("Executable=")
    )
    for line in detail_output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"CDHash", "Identifier", "TeamIdentifier", "Signature"}:
            parsed[key] = value
    requirement_output = "\n".join(
        value for value in (requirement.stdout, requirement.stderr) if value
    )
    designated = _parse_designated_requirement(requirement_output)
    if not designated:
        raise RemoteProtocolError("staged runtime omitted its designated requirement")
    return {
        "file_description": file_result,
        "mach_o_arches": arches,
        "cdhash": parsed.get("CDHash", "").lower(),
        "identifier": parsed.get("Identifier"),
        "team_identifier": parsed.get("TeamIdentifier"),
        "signature": parsed.get("Signature"),
        "designated_requirement": designated,
        "designated_requirement_sha256": content_sha256(designated.encode()),
        "verify_output_sha256": content_sha256((verify.stdout + "\n" + verify.stderr).encode()),
        "detail_output_sha256": content_sha256(detail_output.encode()),
        "portable_detail_sha256": content_sha256(normalized_detail.encode()),
    }


def stage_john1_runtime(client: RemoteStorageClient, manifest_relative: str) -> StagedJohn1Runtime:
    host_proof = _verify_john1_staging_host()
    host_proof["startup_cleanup_receipts"] = cleanup_stale_john1_runtime_directories(client)
    remote_reads: list[dict[str, Any]] = []
    manifest_open = client.open_object_with_receipt(manifest_relative)
    manifest_token = manifest_open["object_token"]
    remote_reads.append(
        {
            "kind": "open-manifest",
            "object_token_sha256": manifest_token["token_sha256"],
            "storage_receipt_relative": manifest_open["storage_receipt_relative"],
            "storage_receipt_sha256": manifest_open["storage_receipt_sha256"],
        }
    )
    if manifest_token["size"] > worker.MAX_EPHEMERAL_MANIFEST_BYTES:
        raise RemoteProtocolError("remote runtime manifest exceeds 64 KiB")
    manifest_read = client.read_range_with_receipt(
        manifest_token,
        0,
        manifest_token["size"],
        max_bytes=worker.MAX_EPHEMERAL_MANIFEST_BYTES,
    )
    manifest_bytes = manifest_read["payload"]
    remote_reads.append(
        {key: value for key, value in manifest_read.items() if key != "payload"}
        | {"kind": "read-manifest"}
    )
    try:
        raw_manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteProtocolError("remote runtime manifest JSON is invalid") from error
    if not isinstance(raw_manifest, dict):
        raise RemoteProtocolError("remote runtime manifest must be an object")
    manifest = validate_john1_runtime_manifest(raw_manifest)
    executable_identity = manifest["executable"]
    executable_open = client.open_object_with_receipt(executable_identity["relative"])
    executable_token = executable_open["object_token"]
    remote_reads.append(
        {
            "kind": "open-executable",
            "object_token_sha256": executable_token["token_sha256"],
            "storage_receipt_relative": executable_open["storage_receipt_relative"],
            "storage_receipt_sha256": executable_open["storage_receipt_sha256"],
        }
    )
    if (
        executable_token["sha256"] != executable_identity["sha256"]
        or executable_token["size"] != executable_identity["size"]
        or executable_token["mode"] != 0o500
    ):
        raise RemoteProtocolError("remote executable object differs from its signed manifest")
    packet_id = manifest["packet_id"]
    directory = JOHN1_STAGING_ROOT / f"cascadia-r2-map-runtime-{packet_id}"
    if directory.exists() or directory.is_symlink():
        raise RemoteStorageError("registered john1 staging directory already exists")
    os.mkdir(directory, 0o700)
    os.chown(directory, JOHN1_UID, JOHN1_GID, follow_symlinks=False)
    os.chmod(directory, 0o700, follow_symlinks=False)
    manifest_path = directory / "runtime-manifest.json"
    executable_path = directory / "cascadia-r2-runtime"
    try:
        _atomic_local_write(
            manifest_path,
            [manifest_bytes],
            len(manifest_bytes),
            manifest_token["sha256"],
        )

        def executable_chunks() -> Iterator[bytes]:
            for read in client.iter_object_with_receipts(executable_token):
                remote_reads.append(
                    {key: value for key, value in read.items() if key != "payload"}
                    | {"kind": "read-executable"}
                )
                yield read["payload"]

        _atomic_local_write(
            executable_path,
            executable_chunks(),
            executable_token["size"],
            executable_token["sha256"],
        )
        os.chmod(executable_path, 0o500, follow_symlinks=False)
        staged_bytes = executable_path.read_bytes()
        staged_sha256 = content_sha256(staged_bytes)
        staged_blake3 = blake3.blake3(staged_bytes).hexdigest()
        if (
            staged_sha256 != executable_token["sha256"]
            or staged_blake3 != executable_identity["blake3"]
        ):
            raise RemoteProtocolError("staged executable bytes changed after chmod")
        observed = _codesign_fields(executable_path)
        extended_attributes = subprocess.run(
            ["/usr/bin/xattr", executable_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.splitlines()
        signed = executable_identity["codesign"]
        if (
            observed["mach_o_arches"] != ["arm64"]
            or "Mach-O 64-bit" not in observed["file_description"]
            or observed["cdhash"] != signed["cdhash"]
            or observed["identifier"] != signed["identifier"]
            or observed["team_identifier"] != signed["team_identifier"]
            or observed["signature"] != signed["signature"]
            or observed["designated_requirement"] != signed["designated_requirement"]
            or observed["designated_requirement_sha256"] != signed["designated_requirement_sha256"]
            or observed["portable_detail_sha256"] != signed["portable_detail_sha256"]
            or not set(extended_attributes).issubset({"com.apple.provenance"})
        ):
            raise RemoteProtocolError("staged runtime Mach-O/codesign identity differs")
        parent = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        validated_manifest = _validate_exact_staging_directory(directory)
        if validated_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RemoteProtocolError("staged runtime manifest changed during exact inventory")
        staging_inventory = _inventory_staging_tree(directory)
    except BaseException as error:
        cleanup = _cleanup_partial_staging_directory(directory)
        receipt = {
            "schema_id": "cascadia.r2-map.john1-stage-failure-cleanup.v1",
            "packet_id": packet_id,
            "staging_path": str(directory),
            "failure_type": type(error).__name__,
            "cleanup": cleanup,
            "completed_unix_ms": time.time_ns() // 1_000_000,
        }
        receipt["cleanup_receipt_sha256"] = document_sha256(receipt, "cleanup_receipt_sha256")
        try:
            _publish_verified_bytes(
                client,
                f"control/staging-cleanups/stage-failure-{packet_id}-"
                f"{receipt['completed_unix_ms']}.json",
                canonical_json(receipt),
                "stage failure cleanup",
            )
        except BaseException as publication_error:
            raise RemoteStorageError(
                "stage failure cleanup receipt publication failed"
            ) from publication_error
        raise
    host_proof["staged_inventory"] = staging_inventory
    host_proof["remote_read_receipts"] = remote_reads
    return StagedJohn1Runtime(
        manifest=manifest,
        directory=directory,
        executable=executable_path,
        manifest_path=manifest_path,
        remote_manifest_sha256=manifest_token["sha256"],
        john1_staging_proof=host_proof,
    )


def _inventory_staging_tree(directory: Path) -> dict[str, Any]:
    entries = []
    total = 0
    for root, directories, files in os.walk(directory, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = Path(root) / name
            details = os.lstat(path)
            relative = path.relative_to(directory).as_posix()
            if details.st_uid != JOHN1_UID or details.st_gid != JOHN1_GID:
                raise RemoteStorageError("staging tree contains a wrong-owner entry")
            if stat.S_ISLNK(details.st_mode):
                raise RemoteStorageError("staging tree contains a symlink")
            elif stat.S_ISDIR(details.st_mode):
                entries.append({"relative": relative, "kind": "directory"})
            elif stat.S_ISREG(details.st_mode):
                if details.st_nlink != 1:
                    raise RemoteStorageError("staging tree contains a hard-linked file")
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        digest.update(chunk)
                total += details.st_size
                entries.append(
                    {
                        "relative": relative,
                        "kind": "file",
                        "size": details.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            else:
                raise RemoteStorageError("staging tree contains a special file")
            if len(entries) > 10_000 or total > worker.MAX_UNKNOWN_STREAM_BYTES:
                raise RemoteStorageError("staging cleanup inventory exceeds its safety bound")
    return {
        "entries": entries,
        "bytes": total,
        "tree_sha256": content_sha256(canonical_json(entries)),
    }


def _validate_exact_staging_directory(directory: Path) -> dict[str, Any]:
    details = os.lstat(directory)
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o700
        or details.st_uid != JOHN1_UID
        or details.st_gid != JOHN1_GID
    ):
        raise RemoteStorageError("stale staging directory mode/owner is invalid")
    entries = {entry.name: entry for entry in os.scandir(directory)}
    if set(entries) != {"runtime-manifest.json", "cascadia-r2-runtime"}:
        raise RemoteStorageError("stale staging directory is not the exact two-file packet")
    manifest_path = directory / "runtime-manifest.json"
    executable_path = directory / "cascadia-r2-runtime"
    manifest_stat = os.lstat(manifest_path)
    executable_stat = os.lstat(executable_path)
    if (
        stat.S_ISLNK(manifest_stat.st_mode)
        or not stat.S_ISREG(manifest_stat.st_mode)
        or stat.S_IMODE(manifest_stat.st_mode) != 0o400
        or manifest_stat.st_size > worker.MAX_EPHEMERAL_MANIFEST_BYTES
        or manifest_stat.st_uid != JOHN1_UID
        or manifest_stat.st_gid != JOHN1_GID
        or manifest_stat.st_nlink != 1
        or stat.S_ISLNK(executable_stat.st_mode)
        or not stat.S_ISREG(executable_stat.st_mode)
        or stat.S_IMODE(executable_stat.st_mode) != 0o500
        or executable_stat.st_size > worker.MAX_EPHEMERAL_RUNTIME_BYTES
        or executable_stat.st_uid != JOHN1_UID
        or executable_stat.st_gid != JOHN1_GID
        or executable_stat.st_nlink != 1
    ):
        raise RemoteStorageError("stale staging packet file type/mode/size is invalid")
    try:
        raw_manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteStorageError("stale staging manifest is malformed") from error
    if not isinstance(raw_manifest, dict):
        raise RemoteStorageError("stale staging manifest is not an object")
    manifest = validate_john1_runtime_manifest(raw_manifest)
    if directory.name != f"cascadia-r2-map-runtime-{manifest['packet_id']}":
        raise RemoteStorageError("stale staging directory and packet identities differ")
    executable = manifest["executable"]
    observed_codesign = _codesign_fields(executable_path)
    extended_attributes = subprocess.run(
        ["/usr/bin/xattr", executable_path],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    if (
        content_sha256(executable_path.read_bytes()) != executable["sha256"]
        or blake3.blake3(executable_path.read_bytes()).hexdigest() != executable["blake3"]
        or executable_stat.st_size != executable["size"]
        or observed_codesign["mach_o_arches"] != ["arm64"]
        or observed_codesign["cdhash"] != executable["codesign"]["cdhash"]
        or observed_codesign["designated_requirement"]
        != executable["codesign"]["designated_requirement"]
        or observed_codesign["designated_requirement_sha256"]
        != executable["codesign"]["designated_requirement_sha256"]
        or observed_codesign["portable_detail_sha256"]
        != executable["codesign"]["portable_detail_sha256"]
        or not set(extended_attributes).issubset({"com.apple.provenance"})
    ):
        raise RemoteStorageError("stale staging executable identity is invalid")
    return manifest


def _remove_staging_tree(directory: Path) -> dict[str, Any]:
    inventory = _inventory_staging_tree(directory)
    os.chmod(directory, 0o700, follow_symlinks=False)
    shutil.rmtree(directory)
    parent = os.open(JOHN1_STAGING_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return {**inventory, "removed": not directory.exists()}


def _cleanup_partial_staging_directory(directory: Path) -> dict[str, Any]:
    details = os.lstat(directory)
    inherited_gid = os.lstat(JOHN1_STAGING_ROOT).st_gid
    if (
        directory.parent != JOHN1_STAGING_ROOT
        or stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != JOHN1_UID
        or details.st_gid not in {JOHN1_GID, inherited_gid}
    ):
        raise RemoteStorageError("partial staging cleanup target is unsafe")
    temporary = re.compile(r"\.(?:runtime-manifest\.json|cascadia-r2-runtime)\.[0-9a-f]{32}\.tmp\Z")
    for entry in os.scandir(directory):
        item = Path(entry.path)
        item_stat = os.lstat(item)
        if (
            stat.S_ISLNK(item_stat.st_mode)
            or not stat.S_ISREG(item_stat.st_mode)
            or item_stat.st_uid != JOHN1_UID
            or item_stat.st_gid not in {JOHN1_GID, inherited_gid}
            or item_stat.st_nlink != 1
            or (
                entry.name not in {"runtime-manifest.json", "cascadia-r2-runtime"}
                and temporary.fullmatch(entry.name) is None
            )
        ):
            raise RemoteStorageError("partial staging directory contains an unknown entry")
        if item_stat.st_gid != JOHN1_GID:
            os.chown(item, JOHN1_UID, JOHN1_GID, follow_symlinks=False)
    if details.st_gid != JOHN1_GID:
        os.chown(directory, JOHN1_UID, JOHN1_GID, follow_symlinks=False)
    return _remove_staging_tree(directory)


def cleanup_john1_runtime_directory(
    directory: Path, *, _already_validated: bool = False
) -> dict[str, Any]:
    expected_prefix = "cascadia-r2-map-runtime-"
    if directory.parent != JOHN1_STAGING_ROOT or not directory.name.startswith(expected_prefix):
        raise RemoteStorageError("cleanup target is outside the registered john1 staging namespace")
    details = os.lstat(directory)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise RemoteStorageError("cleanup target is not a real staging directory")
    if not _already_validated:
        _validate_exact_staging_directory(directory)
    return _remove_staging_tree(directory)


def cleanup_stale_john1_runtime_directories(
    client: RemoteStorageClient,
) -> list[dict[str, Any]]:
    _verify_john1_staging_host()
    prefix = "cascadia-r2-map-runtime-"
    candidates = sorted(
        (
            Path(entry.path)
            for entry in os.scandir(JOHN1_STAGING_ROOT)
            if entry.name.startswith(prefix)
        ),
        key=lambda path: path.name,
    )
    if len(candidates) > 16:
        raise RemoteStorageError("too many stale john1 runtime directories")
    receipts = []
    for directory in candidates:
        manifest = _validate_exact_staging_directory(directory)
        inventory = cleanup_john1_runtime_directory(directory, _already_validated=True)
        receipt = {
            "schema_id": "cascadia.r2-map.john1-startup-cleanup.v1",
            "packet_id": manifest["packet_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "staging_path": str(directory),
            "cleanup": inventory,
            "completed_unix_ms": time.time_ns() // 1_000_000,
        }
        receipt["cleanup_receipt_sha256"] = document_sha256(receipt, "cleanup_receipt_sha256")
        remote = _publish_verified_bytes(
            client,
            f"control/staging-cleanups/startup-{manifest['packet_id']}-"
            f"{receipt['completed_unix_ms']}.json",
            canonical_json(receipt),
            "startup cleanup",
        )
        receipts.append({"receipt": receipt, "remote": remote})
    return receipts


def execute_john1_runtime(
    client: RemoteStorageClient,
    *,
    manifest_relative: str,
    run_id: str,
    arguments: Sequence[str] = (),
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    if not worker.IDENTIFIER.fullmatch(run_id):
        raise ValueError("ephemeral runtime run_id is not a safe identifier")
    if not 1 <= timeout_seconds <= 86400:
        raise ValueError("ephemeral runtime timeout is invalid")
    for value in arguments:
        if (
            not isinstance(value, str)
            or "\x00" in value
            or "/" in value
            or ".." in value
            or len(value.encode()) > 4096
        ):
            raise ValueError("ephemeral runtime argument could address local storage")
    staged = stage_john1_runtime(client, manifest_relative)
    packet_id = staged.manifest["packet_id"]
    output = staged.manifest["output"]
    prefix = output["prefix_relative"]
    stdout_relative = f"{prefix}/{run_id}.stdout"
    stderr_relative = f"{prefix}/{run_id}.stderr"
    sandbox_profile = "\n".join(
        (
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny file-write*)",
            '(allow file-write* (literal "/dev/null"))',
        )
    )
    environment = {
        "HOME": "/var/empty",
        "TMPDIR": "/var/empty",
        "XDG_CACHE_HOME": "/var/empty",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "LANG": "C",
    }

    def limit_child() -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (worker.MAX_EPHEMERAL_RUNTIME_BYTES, worker.MAX_EPHEMERAL_RUNTIME_BYTES),
        )

    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-p", sandbox_profile, staged.executable, *arguments],
            cwd=staged.directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            preexec_fn=limit_child,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            SshTransport._terminate_and_reap(process)
            raise RemoteStorageError("ephemeral runtime pipes were not created")
    except BaseException as error:
        cleanup_error: BaseException | None = None
        cleanup: dict[str, Any] | None = None
        try:
            cleanup = cleanup_john1_runtime_directory(staged.directory)
        except BaseException as caught_cleanup_error:
            cleanup_error = caught_cleanup_error
        receipt = {
            "schema_id": "cascadia.r2-map.john1-ephemeral-cleanup.v1",
            "packet_id": packet_id,
            "run_id": run_id,
            "staging_path": str(staged.directory),
            "launch_error_type": type(error).__name__,
            "cleanup": cleanup,
            "cleanup_error_type": (None if cleanup_error is None else type(cleanup_error).__name__),
            "completed_unix_ms": time.time_ns() // 1_000_000,
        }
        receipt["cleanup_receipt_sha256"] = document_sha256(receipt, "cleanup_receipt_sha256")
        _publish_verified_bytes(
            client,
            f"control/staging-cleanups/{packet_id}-{run_id}.json",
            canonical_json(receipt),
            "launch failure cleanup",
        )
        if cleanup_error is not None:
            raise RemoteStorageError("ephemeral runtime launch cleanup failed") from cleanup_error
        raise
    assert process.stdout is not None and process.stderr is not None
    stream_results: dict[str, dict[str, Any]] = {}
    stream_errors: list[BaseException] = []
    stream_failed = threading.Event()
    termination_lock = threading.Lock()
    terminated = False

    def terminate_runtime() -> None:
        nonlocal terminated
        with termination_lock:
            if terminated:
                return
            terminated = True
            SshTransport._terminate_and_reap(process)

    def upload_stream(name: str, source: BinaryIO, relative: str, limit: int) -> None:
        try:
            stream_results[name] = _verified_remote_publication(
                client.put_unknown_stream(
                    relative,
                    bytes_chunks(source),
                    max_bytes=limit,
                ),
                f"ephemeral runtime {name}",
            )
        except BaseException as error:
            stream_errors.append(error)
            stream_failed.set()
            terminate_runtime()

    threads = [
        threading.Thread(
            target=upload_stream,
            args=("stdout", process.stdout, stdout_relative, output["stdout_max_bytes"]),
            daemon=True,
        ),
        threading.Thread(
            target=upload_stream,
            args=("stderr", process.stderr, stderr_relative, output["stderr_max_bytes"]),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    deadline = time.monotonic() + timeout_seconds
    while True:
        if stream_failed.is_set():
            terminate_runtime()
            exit_code = process.wait(timeout=1)
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            terminate_runtime()
            exit_code = 124
            break
        try:
            exit_code = process.wait(timeout=min(0.2, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    descendants_reaped = False
    process_pid = getattr(process, "pid", None)
    if isinstance(process_pid, int) and process_pid > 0:
        try:
            os.killpg(process_pid, 0)
        except ProcessLookupError:
            pass
        else:
            descendants_reaped = True
            terminate_runtime()
    for thread in threads:
        # Each SSH upload has its own bounded transport timeout. Do not publish or
        # clean staging while an output consumer can still mutate remote state.
        thread.join(timeout=30)
    if any(thread.is_alive() for thread in threads):
        terminate_runtime()
        client.transport.cancel_active()
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            raise RemoteStorageError(
                "output upload thread survived runtime and SSH process-group cancellation"
            )
        stream_errors.append(RemoteStorageError("output upload required forced cancellation"))
    stream_complete = not stream_errors and set(stream_results) == {"stdout", "stderr"}
    completed_ms = time.time_ns() // 1_000_000
    execution = {
        "schema_id": "cascadia.r2-map.john1-ephemeral-execution.v1",
        "packet_id": packet_id,
        "run_id": run_id,
        "manifest_sha256": staged.manifest["manifest_sha256"],
        "remote_manifest_sha256": staged.remote_manifest_sha256,
        "john1_staging_proof": staged.john1_staging_proof,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "descendants_reaped": descendants_reaped,
        "duration_ms": (time.monotonic_ns() - started) // 1_000_000,
        "completed_unix_ms": completed_ms,
        "stdout": stream_results.get("stdout"),
        "stderr": stream_results.get("stderr"),
        "stream_error_types": [type(error).__name__ for error in stream_errors],
        "stream_complete": stream_complete,
    }
    execution["execution_sha256"] = document_sha256(execution, "execution_sha256")
    execution_relative = f"control/ephemeral-executions/{packet_id}-{run_id}.json"
    cleanup_error: BaseException | None = None
    cleanup: dict[str, Any] | None = None
    try:
        cleanup = cleanup_john1_runtime_directory(staged.directory)
    except BaseException as error:
        cleanup_error = error
    cleanup_receipt = {
        "schema_id": "cascadia.r2-map.john1-ephemeral-cleanup.v1",
        "packet_id": packet_id,
        "run_id": run_id,
        "execution_sha256": execution["execution_sha256"],
        "staging_path": str(staged.directory),
        "cleanup": cleanup,
        "cleanup_error_type": None if cleanup_error is None else type(cleanup_error).__name__,
        "completed_unix_ms": time.time_ns() // 1_000_000,
    }
    cleanup_receipt["cleanup_receipt_sha256"] = document_sha256(
        cleanup_receipt, "cleanup_receipt_sha256"
    )
    publication_errors: list[BaseException] = []
    execution_remote: dict[str, Any] | None = None
    cleanup_remote: dict[str, Any] | None = None
    try:
        execution_remote = _publish_verified_bytes(
            client,
            execution_relative,
            canonical_json(execution),
            "ephemeral execution",
        )
    except BaseException as error:
        publication_errors.append(error)
    try:
        cleanup_remote = _publish_verified_bytes(
            client,
            f"control/staging-cleanups/{packet_id}-{run_id}.json",
            canonical_json(cleanup_receipt),
            "ephemeral cleanup",
        )
    except BaseException as error:
        publication_errors.append(error)
    if publication_errors:
        raise RemoteStorageError(
            "ephemeral runtime receipt publication failed"
        ) from publication_errors[0]
    if not stream_complete:
        detail = stream_errors[0] if stream_errors else RuntimeError("missing stream receipt")
        raise RemoteStorageError(f"ephemeral runtime output streaming failed: {detail}") from detail
    if cleanup_error is not None:
        raise RemoteStorageError("ephemeral runtime cleanup failed") from cleanup_error
    return {
        "execution": execution,
        "execution_remote": execution_remote,
        "cleanup": cleanup_receipt,
        "cleanup_remote": cleanup_remote,
    }
