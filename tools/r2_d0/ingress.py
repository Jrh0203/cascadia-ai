"""Verified immutable result ingress into John1's active campaign root."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .bundle import verify_result_bundle
from .canonical import CAMPAIGN_ID, D0_RUN_ID, D0Error, canonical_json, document_sha256
from .inventory import secure_owner_directory
from .storage import CANONICAL_ROOT, verify_canonical_commit_boundary, verify_canonical_storage

INGRESS_SCHEMA = "cascadia.r2-map.d0-john1-result-ingress.v1"
ACTIVE_HOSTS = {"john1", "john2", "john3"}


def _safe_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value in {".", ".."}
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in value)
    ):
        raise D0Error(f"result ingress {label} is not a safe component")
    return value


def result_ingress_relative(verification: Mapping[str, Any]) -> Path:
    """Derive the sole John1 destination from signed bundle identity."""

    manifest = verification.get("manifest")
    report = verification.get("report")
    if not isinstance(manifest, Mapping) or not isinstance(report, Mapping):
        raise D0Error("result ingress bundle identity is incomplete")
    if (
        manifest.get("run_id") != D0_RUN_ID
        or manifest.get("host") not in ACTIVE_HOSTS
        or manifest.get("cycle_id") not in {"qualification", "final-live"}
    ):
        raise D0Error("result ingress bundle identity differs")
    return Path(
        "reports",
        "infrastructure",
        D0_RUN_ID,
        "incoming",
        _safe_component(manifest["host"], "host"),
        _safe_component(manifest["cycle_id"], "cycle"),
        _safe_component(report.get("phase"), "phase"),
        _safe_component(report.get("operation"), "operation"),
        _safe_component(verification.get("report_sha256"), "report digest"),
    )


def _read_fixed_file(path: Path, *, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) != 0o400
            or details.st_size > maximum
        ):
            raise D0Error("result ingress installed file metadata differs")
        result = bytearray()
        while len(result) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        if len(result) != details.st_size:
            raise D0Error("result ingress installed file changed while reading")
        return bytes(result)
    finally:
        os.close(descriptor)


def _write_fixed_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise D0Error("result ingress made a short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_result_ingress(
    archive: bytes,
    *,
    public_key: bytes,
    campaign_root: Path = CANONICAL_ROOT,
    storage_verifier: Callable[..., Mapping[str, Any]] = verify_canonical_storage,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_result_bundle,
    commit_verifier: Callable[[Path], Mapping[str, Any] | None] = (
        verify_canonical_commit_boundary
    ),
) -> dict[str, Any]:
    """Verify and atomically install one signed host bundle on John1.

    Workers never write this tree. The root orchestrator supplies bytes obtained
    through a bounded transport, then this transaction independently verifies
    the signatures, exact archive, active storage, and destination identity.
    """

    if not isinstance(archive, bytes) or not archive:
        raise D0Error("result ingress archive is empty")
    canonical_root = campaign_root.resolve(strict=False)
    if canonical_root != CANONICAL_ROOT.resolve(strict=False):
        raise D0Error("result ingress campaign root is not John1's active root")
    storage = dict(storage_verifier(measure_size=True))
    if storage.get("status") != "pass" or storage.get("root") != str(canonical_root):
        raise D0Error("result ingress active-storage verification did not pass")
    verification = dict(bundle_verifier(archive, public_key=public_key))
    relative = result_ingress_relative(verification)
    destination = canonical_root / relative
    receipt: dict[str, Any] = {
        "schema_id": INGRESS_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "source_host": verification["manifest"]["host"],
        "destination_relative": relative.as_posix(),
        "archive_size": len(archive),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "manifest_sha256": verification["manifest"]["manifest_sha256"],
        "packet_sha256": verification["packet"]["packet_sha256"],
        "report_sha256": verification["report_sha256"],
        "storage_identity_sha256": storage["host_identity_sha256"],
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    receipt_bytes = canonical_json(receipt)
    expected = {"bundle.tar": archive, "ingress-receipt.json": receipt_bytes}
    secure_owner_directory(destination.parent)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise D0Error("result ingress destination is unsafe")
        observed_names = {child.name for child in destination.iterdir()}
        if observed_names != set(expected):
            raise D0Error("result ingress destination contains a different file set")
        for name, payload in expected.items():
            if _read_fixed_file(destination / name, maximum=max(len(payload), 1)) != payload:
                raise D0Error("result ingress destination contains different bytes")
        commit_verifier(destination)
        return {"receipt": receipt, "disposition": "present"}
    staging = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise D0Error("result ingress staging path already exists")
    staging.mkdir(mode=0o700)
    try:
        for name, payload in expected.items():
            _write_fixed_file(staging / name, payload)
        commit_verifier(destination)
        os.replace(staging, destination)
        parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"receipt": receipt, "disposition": "installed"}
