#!/usr/bin/env python3
"""Signed, atomic transfer of one sealed D0 predecessor to its execution host."""

from __future__ import annotations

import argparse
import os
import pwd
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

HELPER_ROOT = Path.home() / ".local/libexec/cascadia-r2-d0/v1"
sys.path.insert(0, str(HELPER_ROOT))

from r2_d0.bundle import verify_result_bundle  # noqa: E402
from r2_d0.canonical import (  # noqa: E402
    CAMPAIGN_ID,
    D0_RUN_ID,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    safe_relative,
    sha256_bytes,
)
from r2_d0.closure import (  # noqa: E402
    build_materialization_receipt,
    validate_materialization_receipt,
)
from r2_d0.signing import load_public_key, verify_stdin  # noqa: E402

SCHEMA = "cascadia.r2-map.d0-predecessor-transfer-authorization.v1"
MAX_JSON = 4 * 1024 * 1024
MAX_BUNDLE = 2 * 1024 * 1024 * 1024
HOST_USERS = {"john1": "johnherrick", "john2": "john2", "john3": "john3"}
HOST_HOMES = {host: f"/Users/{user}" for host, user in HOST_USERS.items()}


def _read_regular(path: Path, maximum: int, label: str, *, require_owner: bool = True) -> bytes:
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot inspect {label}") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (require_owner and observed.st_uid != os.getuid())
        or observed.st_mode & 0o022
        or observed.st_size > maximum
    ):
        raise D0Error(f"{label} metadata is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != observed.st_size:
        raise D0Error(f"{label} changed while reading")
    return payload


def _write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _local_storage_identity(root: Path, target_host: str) -> str:
    try:
        observed = root.lstat()
    except OSError as error:
        raise D0Error("target output root cannot be inspected") from error
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_mode & 0o022
        or root.resolve() != root
    ):
        raise D0Error("target output root metadata is unsafe")
    identity = {
        "target_host": target_host,
        "root": str(root),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
    }
    return sha256_bytes(canonical_json(identity))


def _self_sha256() -> str:
    return sha256_bytes(_read_regular(Path(__file__).resolve(), MAX_JSON, "transfer installer"))


def _load_receipt(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = _read_regular(path, MAX_JSON, "source materialization receipt")
    return payload, validate_materialization_receipt(
        load_canonical_json(payload, maximum=MAX_JSON, label="source materialization receipt")
    )


def _destination_relative(source_host: str, target_host: str, report_sha256: str) -> str:
    """Return the runtime's canonical target-relative predecessor namespace."""

    if source_host not in HOST_USERS or target_host not in HOST_USERS:
        raise D0Error("predecessor transfer host differs")
    if not isinstance(report_sha256, str) or len(report_sha256) != 64:
        raise D0Error("predecessor transfer report SHA-256 differs")
    if source_host == target_host:
        return f"receipts/{report_sha256}"
    return f"dependencies/{source_host}/{report_sha256}"


def render_authorization(args: argparse.Namespace) -> dict[str, Any]:
    archive = _read_regular(args.archive, MAX_BUNDLE, "sealed predecessor bundle")
    public_key = load_public_key(args.public_key)
    verification = verify_result_bundle(archive, public_key=public_key)
    source_receipt_bytes, source_receipt = _load_receipt(args.source_materialization_receipt)
    packet = verification["packet"]
    report = verification["report"]
    if (
        source_receipt["source_host"] != packet["host"]
        or source_receipt["target_host"] != "john1"
        or source_receipt["bundle_sha256"] != sha256_bytes(archive)
        or source_receipt["bundle_size"] != len(archive)
        or source_receipt["manifest_sha256"] != verification["manifest"]["manifest_sha256"]
        or source_receipt["packet_sha256"] != packet["packet_sha256"]
        or source_receipt["report_sha256"] != report["report_sha256"]
    ):
        raise D0Error("source materialization binding differs")
    destination_relative = _destination_relative(
        packet["host"], args.target_host, report["report_sha256"]
    )
    target_output_root = (
        Path(HOST_HOMES[args.target_host])
        / ".local/share/cascadia-r2/results"
        / D0_RUN_ID
    )
    issued = time.time_ns() // 1_000_000
    authorization: dict[str, Any] = {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "source_control_host": "john1",
        "source_host": packet["host"],
        "target_host": args.target_host,
        "cycle_id": packet["cycle_id"],
        "phase": report["phase"],
        "operation": report["operation"],
        "packet_sha256": packet["packet_sha256"],
        "report_sha256": report["report_sha256"],
        "bundle_sha256": sha256_bytes(archive),
        "bundle_size": len(archive),
        "manifest_sha256": verification["manifest"]["manifest_sha256"],
        "source_materialization_receipt_sha256": source_receipt["receipt_sha256"],
        "source_materialization_receipt_file_sha256": sha256_bytes(source_receipt_bytes),
        "installer_sha256": _self_sha256(),
        "target_output_root": str(target_output_root),
        "destination_relative": destination_relative,
        "destination": str(target_output_root / destination_relative),
        "transport": "direct-john1-control-edge",
        "peer_credentials_present": False,
        "issued_unix_ms": issued,
        "expires_unix_ms": issued + 24 * 60 * 60 * 1000,
        "protected_seed_values_opened": False,
    }
    authorization["authorization_sha256"] = document_sha256(
        authorization, "authorization_sha256"
    )
    _write_new(args.out, canonical_json(authorization))
    return authorization


def _validate_authorization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise D0Error("predecessor transfer authorization is not an object")
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "source_control_host",
        "source_host",
        "target_host",
        "cycle_id",
        "phase",
        "operation",
        "packet_sha256",
        "report_sha256",
        "bundle_sha256",
        "bundle_size",
        "manifest_sha256",
        "source_materialization_receipt_sha256",
        "source_materialization_receipt_file_sha256",
        "installer_sha256",
        "target_output_root",
        "destination_relative",
        "destination",
        "transport",
        "peer_credentials_present",
        "issued_unix_ms",
        "expires_unix_ms",
        "protected_seed_values_opened",
        "authorization_sha256",
    }
    if set(value) != required:
        raise D0Error("predecessor transfer authorization shape differs")
    if (
        value["schema_id"] != SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["source_control_host"] != "john1"
        or value["source_host"] not in HOST_USERS
        or value["target_host"] not in HOST_USERS
        or value["transport"] != "direct-john1-control-edge"
        or value["peer_credentials_present"] is not False
        or value["protected_seed_values_opened"] is not False
        or value["authorization_sha256"]
        != document_sha256(value, "authorization_sha256")
    ):
        raise D0Error("predecessor transfer authorization identity differs")
    for field in (
        "packet_sha256",
        "report_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "source_materialization_receipt_sha256",
        "source_materialization_receipt_file_sha256",
        "installer_sha256",
    ):
        if not isinstance(value[field], str) or len(value[field]) != 64:
            raise D0Error(f"predecessor transfer {field} differs")
    if not isinstance(value["bundle_size"], int) or value["bundle_size"] <= 0:
        raise D0Error("predecessor transfer bundle size differs")
    now = time.time_ns() // 1_000_000
    if not value["issued_unix_ms"] <= now <= value["expires_unix_ms"]:
        raise D0Error("predecessor transfer authorization is outside its validity window")
    relative = safe_relative(value["destination_relative"], "transfer destination")
    expected_root = (
        Path(HOST_HOMES[value["target_host"]])
        / ".local/share/cascadia-r2/results"
        / D0_RUN_ID
    )
    if (
        Path(value["target_output_root"]) != expected_root
        or Path(value["destination"]) != expected_root / relative
        or relative
        != _destination_relative(
            value["source_host"], value["target_host"], value["report_sha256"]
        )
    ):
        raise D0Error("predecessor transfer destination differs")
    return value


def install(args: argparse.Namespace) -> dict[str, Any]:
    authorization_bytes = _read_regular(args.authorization, MAX_JSON, "transfer authorization")
    signature_bytes = _read_regular(args.signature, MAX_JSON, "transfer authorization signature")
    authorization = _validate_authorization(
        load_canonical_json(
            authorization_bytes, maximum=MAX_JSON, label="transfer authorization"
        )
    )
    signature = load_canonical_json(
        signature_bytes, maximum=MAX_JSON, label="transfer authorization signature"
    )
    public_key = load_public_key(args.public_key)
    verify_stdin(public_key, authorization_bytes, signature)
    if sha256_bytes(authorization_bytes) != args.confirm_authorization_file_sha256:
        raise D0Error("transfer authorization file confirmation differs")
    if authorization["installer_sha256"] != _self_sha256():
        raise D0Error("transfer installer identity differs")
    local_user = pwd.getpwuid(os.getuid()).pw_name
    if HOST_USERS[authorization["target_host"]] != local_user:
        raise D0Error("transfer authorization target account differs")
    archive = _read_regular(args.archive, MAX_BUNDLE, "transferred predecessor bundle")
    source_receipt_bytes, source_receipt = _load_receipt(args.source_materialization_receipt)
    verification = verify_result_bundle(archive, public_key=public_key)
    packet = verification["packet"]
    report = verification["report"]
    if (
        sha256_bytes(archive) != authorization["bundle_sha256"]
        or len(archive) != authorization["bundle_size"]
        or verification["manifest"]["manifest_sha256"]
        != authorization["manifest_sha256"]
        or packet["host"] != authorization["source_host"]
        or packet["packet_sha256"] != authorization["packet_sha256"]
        or report["report_sha256"] != authorization["report_sha256"]
        or report["operation"] != authorization["operation"]
        or report["phase"] != authorization["phase"]
        or sha256_bytes(source_receipt_bytes)
        != authorization["source_materialization_receipt_file_sha256"]
        or source_receipt["receipt_sha256"]
        != authorization["source_materialization_receipt_sha256"]
        or source_receipt["target_host"] != "john1"
        or source_receipt["bundle_sha256"] != authorization["bundle_sha256"]
    ):
        raise D0Error("transferred predecessor lineage differs")
    destination = Path(authorization["destination"])
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    storage_identity_sha256 = _local_storage_identity(
        Path(authorization["target_output_root"]), authorization["target_host"]
    )
    staging = destination.with_name(f".{destination.name}.partial-transfer")
    if destination.exists():
        existing_archive = _read_regular(
            destination / "bundle.tar", MAX_BUNDLE, "installed predecessor bundle"
        )
        existing_receipt = validate_materialization_receipt(
            load_canonical_json(
                _read_regular(
                    destination / "materialization-receipt.json",
                    MAX_JSON,
                    "installed predecessor receipt",
                ),
                maximum=MAX_JSON,
                label="installed predecessor receipt",
            )
        )
        if (
            sha256_bytes(existing_archive) != authorization["bundle_sha256"]
            or existing_receipt["transport_receipt_sha256"]
            != authorization["authorization_sha256"]
            or existing_receipt["target_host"] != authorization["target_host"]
        ):
            raise D0Error("existing predecessor materialization differs")
        return {"status": "pass", "disposition": "already-installed", "receipt": existing_receipt}
    if staging.exists() or staging.is_symlink():
        raise D0Error("predecessor transfer staging already exists")
    staging.mkdir(mode=0o700)
    try:
        _write_new(staging / "bundle.tar", archive)
        materialized = time.time_ns() // 1_000_000
        receipt_bytes = build_materialization_receipt(
            source_host=authorization["source_host"],
            target_host=authorization["target_host"],
            packet_sha256=authorization["packet_sha256"],
            report_sha256=authorization["report_sha256"],
            operation=authorization["operation"],
            bundle_size=authorization["bundle_size"],
            bundle_sha256=authorization["bundle_sha256"],
            manifest_sha256=authorization["manifest_sha256"],
            persistence_evidence_sha256=verification["persistence_evidence"]["evidence_sha256"],
            transport_receipt_sha256=authorization["authorization_sha256"],
            storage_identity_sha256=storage_identity_sha256,
            destination_relative=authorization["destination_relative"],
            materialized_unix_ms=materialized,
        )
        _write_new(staging / "materialization-receipt.json", receipt_bytes)
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    receipt = validate_materialization_receipt(
        load_canonical_json(
            _read_regular(
                destination / "materialization-receipt.json",
                MAX_JSON,
                "installed predecessor receipt",
            ),
            maximum=MAX_JSON,
            label="installed predecessor receipt",
        )
    )
    return {"status": "pass", "disposition": "installed", "receipt": receipt}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render-authorization")
    render.add_argument("--archive", type=Path, required=True)
    render.add_argument("--source-materialization-receipt", type=Path, required=True)
    render.add_argument("--public-key", type=Path, required=True)
    render.add_argument("--target-host", choices=tuple(HOST_USERS), required=True)
    render.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("install")
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--signature", type=Path, required=True)
    execute.add_argument("--public-key", type=Path, required=True)
    execute.add_argument("--archive", type=Path, required=True)
    execute.add_argument("--source-materialization-receipt", type=Path, required=True)
    execute.add_argument("--confirm-authorization-file-sha256", required=True)
    execute.add_argument("--execute", action="store_true", required=True)
    return root


def main() -> int:
    try:
        arguments = parser().parse_args()
        result = (
            render_authorization(arguments)
            if arguments.command == "render-authorization"
            else install(arguments)
        )
        sys.stdout.buffer.write(canonical_json(result))
        return 0
    except (D0Error, OSError, ValueError) as error:
        sys.stderr.write(f"r2-d0-predecessor-transfer: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
