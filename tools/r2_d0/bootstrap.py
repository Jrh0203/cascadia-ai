"""Trust-on-first-use helper packaging and one-shot public-key bootstrap."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import pwd
import shutil
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import (
    BOOTSTRAP_PACKET_SCHEMA,
    CAMPAIGN_ID,
    REJECTED_HELPER_ARCHIVE_SHA256,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    sha256_bytes,
    validate_bootstrap_packet,
)
from .inventory import secure_owner_directory
from .signing import normalize_public_key, public_key_fingerprint

HELPER_MANIFEST_SCHEMA = "cascadia.r2-map.d0-helper-source-manifest.v1"
BOOTSTRAP_RECEIPT_SCHEMA = "cascadia.r2-map.d0-bootstrap-receipt.v1"
MAX_HELPER_BYTES = 16 * 1024 * 1024
MAX_HELPER_FILES = 64
RECORD_SIZE = 10 * 1024
HELPER_ENTRYPOINT = "r2_map_d0_runtime.py"
HELPER_SOURCE_PATHS = (
    "r2_map_d0_runtime.py",
    "r2_d0/__init__.py",
    "r2_d0/authorization.py",
    "r2_d0/artifacts.py",
    "r2_d0/aggregate.py",
    "r2_d0/bootstrap.py",
    "r2_d0/bundle.py",
    "r2_d0/canonical.py",
    "r2_d0/cli.py",
    "r2_d0/closure.py",
    "r2_d0/dashboard.py",
    "r2_d0/inventory.py",
    "r2_d0/ingress.py",
    "r2_d0/runtime.py",
    "r2_d0/signing.py",
    "r2_d0/storage.py",
    "r2_d0/transport.py",
)
HOST_USERS = {"john1": "johnherrick", "john2": "john2", "john3": "john3"}

# Frozen from the reviewed helper closure rather than discovered from the
# controller interpreter. Apple system Python 3.9 does not expose
# sys.stdlib_module_names, and environment-dependent discovery would make the
# source audit non-reproducible. Any new top-level import therefore requires an
# explicit source-review change here.
ALLOWED_HELPER_IMPORTS = frozenset(
    {
        "__future__",
        "argparse",
        "ast",
        "base64",
        "binascii",
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "errno",
        "fcntl",
        "gzip",
        "hashlib",
        "io",
        "ipaddress",
        "json",
        "os",
        "pathlib",
        "plistlib",
        "pwd",
        "r2_d0",
        "re",
        "selectors",
        "shlex",
        "shutil",
        "signal",
        "ssl",
        "stat",
        "struct",
        "subprocess",
        "sys",
        "tarfile",
        "threading",
        "time",
        "typing",
        "urllib",
        "uuid",
    }
)


def _audit_standard_library_only(path: str, payload: bytes) -> None:
    try:
        tree = ast.parse(payload, filename=path)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise D0Error(f"helper source is not valid Python: {path}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            names = [(node.module or "").split(".", 1)[0]]
        else:
            continue
        for name in names:
            if name not in ALLOWED_HELPER_IMPORTS:
                raise D0Error(f"helper imports a non-standard-library module: {name}")


def build_helper_archive(source_root: Path) -> tuple[bytes, dict[str, Any]]:
    """Package only the reviewed standalone helper closure as deterministic USTAR."""

    files: dict[str, bytes] = {}
    identities: list[dict[str, Any]] = []
    for relative in HELPER_SOURCE_PATHS:
        path = source_root / relative
        try:
            observed = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            raise D0Error(f"cannot read helper source: {relative}") from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or len(payload) > MAX_HELPER_BYTES
        ):
            raise D0Error(f"helper source metadata is unsafe: {relative}")
        _audit_standard_library_only(relative, payload)
        files[relative] = payload
        identities.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": sha256_bytes(payload),
                "mode": "0555" if relative == HELPER_ENTRYPOINT else "0444",
            }
        )
    manifest: dict[str, Any] = {
        "schema_id": HELPER_MANIFEST_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "entrypoint": HELPER_ENTRYPOINT,
        "files": identities,
        "standard_library_only": True,
        "project_imports": False,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    files["helper-source-manifest.json"] = canonical_json(manifest)
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(files):
            payload = files[relative]
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = 0o555 if relative == HELPER_ENTRYPOINT else 0o444
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    value = output.getvalue()
    if len(value) > MAX_HELPER_BYTES or len(value) % RECORD_SIZE:
        raise D0Error("helper archive exceeds its size or record contract")
    return value, {
        "archive_size": len(value),
        "archive_sha256": sha256_bytes(value),
        "manifest": manifest,
    }


def _helper_members(value: bytes) -> tuple[dict[str, tuple[bytes, int]], dict[str, Any]]:
    if not value or len(value) > MAX_HELPER_BYTES:
        raise D0Error("helper archive size differs")
    files: dict[str, tuple[bytes, int]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(value), mode="r:") as archive:
            for item in archive:
                if (
                    len(files) >= MAX_HELPER_FILES
                    or not item.isfile()
                    or item.name in files
                    or item.name.startswith("/")
                    or ".." in PurePosixPath(item.name).parts
                    or item.uid != 0
                    or item.gid != 0
                    or item.mtime != 0
                    or item.mode not in {0o444, 0o555}
                ):
                    raise D0Error("helper archive has an unsafe member")
                stream = archive.extractfile(item)
                if stream is None:
                    raise D0Error("helper archive member is unreadable")
                payload = stream.read(MAX_HELPER_BYTES + 1)
                if len(payload) != item.size:
                    raise D0Error("helper archive member size differs")
                files[item.name] = (payload, item.mode)
    except (OSError, tarfile.TarError) as error:
        raise D0Error("helper archive is invalid") from error
    manifest_entry = files.pop("helper-source-manifest.json", None)
    if manifest_entry is None or manifest_entry[1] != 0o444:
        raise D0Error("helper source manifest is absent")
    manifest = load_canonical_json(
        manifest_entry[0], maximum=MAX_HELPER_BYTES, label="helper source manifest"
    )
    if (
        manifest.get("schema_id") != HELPER_MANIFEST_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("entrypoint") != HELPER_ENTRYPOINT
        or manifest.get("standard_library_only") is not True
        or manifest.get("project_imports") is not False
        or manifest.get("manifest_sha256") != document_sha256(manifest, "manifest_sha256")
    ):
        raise D0Error("helper source manifest identity differs")
    identities = manifest.get("files")
    if not isinstance(identities, list):
        raise D0Error("helper source identities are absent")
    expected_paths: set[str] = set()
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {"path", "size", "sha256", "mode"}:
            raise D0Error("helper source identity differs")
        path = identity["path"]
        entry = files.get(path) if isinstance(path, str) else None
        expected_mode = "0555" if entry is not None and entry[1] == 0o555 else "0444"
        if (
            entry is None
            or identity["size"] != len(entry[0])
            or identity["sha256"] != sha256_bytes(entry[0])
            or identity["mode"] != expected_mode
        ):
            raise D0Error("helper source bytes differ")
        _audit_standard_library_only(path, entry[0])
        expected_paths.add(path)
    if expected_paths != set(files) or expected_paths != set(HELPER_SOURCE_PATHS):
        raise D0Error("helper archive source closure differs")
    return files, manifest


def verify_helper_archive(value: bytes) -> dict[str, Any]:
    if sha256_bytes(value) == REJECTED_HELPER_ARCHIVE_SHA256:
        raise D0Error("obsolete D0 helper archive is explicitly rejected")
    files, manifest = _helper_members(value)
    return {
        "archive_size": len(value),
        "archive_sha256": sha256_bytes(value),
        "file_count": len(files),
        "manifest_sha256": manifest["manifest_sha256"],
        "entrypoint": HELPER_ENTRYPOINT,
        "status": "pass",
    }


def _owner_private_parent(path: Path) -> list[Path]:
    return secure_owner_directory(path)


def _atomic_key(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise D0Error("campaign public-key destination already exists")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        position = 0
        while position < len(value):
            position += os.write(descriptor, value[position:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise


def install_bootstrap_artifacts(
    *,
    helper_archive: bytes,
    public_key: bytes,
    helper_destination: Path,
    key_destination: Path,
    receipt_destination: Path,
    receipt_payload: bytes,
) -> dict[str, Any]:
    """Install already-authorized helper/key bytes as an all-or-rollback transaction."""

    normalized_key = normalize_public_key(public_key)
    helper_verification = verify_helper_archive(helper_archive)
    if helper_destination.exists() or helper_destination.is_symlink():
        raise D0Error("bootstrap helper destination already exists")
    if key_destination.exists() or key_destination.is_symlink():
        raise D0Error("campaign public-key destination already exists")
    if receipt_destination.exists() or receipt_destination.is_symlink():
        raise D0Error("bootstrap receipt destination already exists")
    created_parents: list[Path] = []
    installed_helper = False
    installed_key = False
    installed_receipt = False
    staging: Path | None = None
    try:
        created_parents.extend(_owner_private_parent(helper_destination.parent))
        created_parents.extend(_owner_private_parent(key_destination.parent))
        created_parents.extend(_owner_private_parent(receipt_destination.parent))
        staging = helper_destination.with_name(f".{helper_destination.name}.partial-{os.getpid()}")
        staging.mkdir(mode=0o700)
        files, _ = _helper_members(helper_archive)
        for relative in sorted(files):
            payload, archive_mode = files[relative]
            target = staging / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                archive_mode,
            )
            try:
                os.fchmod(descriptor, archive_mode)
                position = 0
                while position < len(payload):
                    position += os.write(descriptor, payload[position:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.replace(staging, helper_destination)
        staging = None
        installed_helper = True
        _atomic_key(key_destination, normalized_key)
        installed_key = True
        _atomic_key(receipt_destination, receipt_payload)
        installed_receipt = True
        for directory_path in (helper_destination, helper_destination.parent):
            directory = os.open(directory_path, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if installed_receipt:
            receipt_destination.unlink(missing_ok=True)
        if installed_key:
            key_destination.unlink(missing_ok=True)
        if installed_helper:
            shutil.rmtree(helper_destination)
        for directory in reversed(created_parents):
            with contextlib.suppress(OSError):
                directory.rmdir()
        raise
    return {
        "helper": helper_verification,
        "helper_destination": str(helper_destination),
        "public_key_sha256": sha256_bytes(normalized_key),
        "public_key_fingerprint": public_key_fingerprint(normalized_key),
        "public_key_destination": str(key_destination),
        "receipt_destination": str(receipt_destination),
    }


def apply_bootstrap(
    packet_bytes: bytes,
    *,
    authorized_packet_sha256: str,
    helper_archive: bytes,
    public_key: bytes,
    current_user: str | None = None,
    now_unix_ms: int | None = None,
) -> dict[str, Any]:
    """Apply exactly one root-authorized helper/key bootstrap transaction."""

    if sha256_bytes(packet_bytes) != authorized_packet_sha256:
        raise D0Error("bootstrap packet is not the root-authorized content hash")
    packet = load_canonical_json(packet_bytes, maximum=1024 * 1024, label="bootstrap packet")
    validate_bootstrap_packet(packet)
    now = now_unix_ms if now_unix_ms is not None else time.time_ns() // 1_000_000
    if not packet["issued_unix_ms"] <= now <= packet["expires_unix_ms"]:
        raise D0Error("bootstrap packet is outside its validity window")
    expected_user = HOST_USERS[packet["host"]]
    kernel_user = pwd.getpwuid(os.getuid()).pw_name
    if current_user is not None and current_user != kernel_user:
        raise D0Error("bootstrap caller-supplied owner differs from the kernel UID")
    if kernel_user != expected_user:
        raise D0Error("bootstrap packet is addressed to a different host owner")
    normalized_key = normalize_public_key(public_key)
    if (
        packet["schema_id"] != BOOTSTRAP_PACKET_SCHEMA
        or packet["helper"]["size"] != len(helper_archive)
        or packet["helper"]["sha256"] != sha256_bytes(helper_archive)
        or packet["public_key"]["openssh_sha256"] != sha256_bytes(normalized_key)
        or packet["public_key"]["fingerprint"] != public_key_fingerprint(normalized_key)
    ):
        raise D0Error("bootstrap artifact identity differs")
    helper_verification = verify_helper_archive(helper_archive)
    if packet["helper"]["entrypoint"] != helper_verification["entrypoint"]:
        raise D0Error("bootstrap helper entrypoint differs")
    helper_destination = Path(packet["destinations"]["helper"])
    key_destination = Path(packet["destinations"]["public_key"])
    receipt_destination = Path(packet["destinations"]["receipt"])
    receipt: dict[str, Any] = {
        "schema_id": BOOTSTRAP_RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "host": packet["host"],
        "packet_content_sha256": authorized_packet_sha256,
        "packet_sha256": packet["packet_sha256"],
        "helper_archive_sha256": helper_verification["archive_sha256"],
        "helper_manifest_sha256": helper_verification["manifest_sha256"],
        "helper_destination": str(helper_destination),
        "public_key_sha256": sha256_bytes(normalized_key),
        "public_key_fingerprint": public_key_fingerprint(normalized_key),
        "public_key_destination": str(key_destination),
        "receipt_destination": str(receipt_destination),
        "installed_unix_ms": now,
        "runtime_installed": False,
        "runtime_invoked": False,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    install_bootstrap_artifacts(
        helper_archive=helper_archive,
        public_key=normalized_key,
        helper_destination=helper_destination,
        key_destination=key_destination,
        receipt_destination=receipt_destination,
        receipt_payload=canonical_json(receipt),
    )
    return receipt
