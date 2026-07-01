#!/usr/bin/env python3
"""Render and verify the signed three-host SSH maintenance restore gate."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2_d0.canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    D0Error,
    canonical_json,
    document_sha256,
    sha256_bytes,
)
from r2_d0.signing import load_public_key, verify_stdin

OPEN_SCHEMA = "cascadia.r2-map.d0-ssh-maintenance-window.v2"
RESTORE_SCHEMA = "cascadia.r2-map.d0-ssh-maintenance-restore.v1"
HOSTS = ("john1", "john2", "john3")
SOURCE_IPS = ("100.98.107.61", "100.98.16.59")
MAX_JSON = 1024 * 1024


def _read(path: Path, label: str) -> bytes:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
        or observed.st_size > MAX_JSON
    ):
        raise D0Error(f"{label} metadata or size is unsafe")
    return path.read_bytes()


def _json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read(path, label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error(f"{label} is not valid JSON") from error
    if payload not in {canonical_json(value), canonical_json(value) + b"\n"}:
        raise D0Error(f"{label} is not canonical JSON")
    if not isinstance(value, dict):
        raise D0Error(f"{label} is not an object")
    return payload, value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.fchmod(descriptor, 0o400)
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise D0Error(f"{label} differs")
    try:
        int(value, 16)
    except ValueError as error:
        raise D0Error(f"{label} differs") from error
    return value


def _open_host(value: Any) -> dict[str, Any]:
    required = {
        "host",
        "authorized_keys_path",
        "backup_path",
        "original_sha256",
        "original_size",
        "active_sha256",
        "active_size",
        "mode",
        "source_ip_denies",
        "changed_unix_ms",
        "original_preserved",
        "active_differs",
        "status",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("maintenance-window host evidence fields differ")
    home = {
        "john1": "/Users/johnherrick",
        "john2": "/Users/john2",
        "john3": "/Users/john3",
    }.get(value["host"])
    if (
        home is None
        or value["authorized_keys_path"] != f"{home}/.ssh/authorized_keys"
        or value["backup_path"] != f"{home}/.ssh/authorized_keys.cascadia-v2-original"
        or value["mode"] != "0600"
        or value["source_ip_denies"] != list(SOURCE_IPS)
        or not isinstance(value["changed_unix_ms"], int)
        or value["changed_unix_ms"] <= 0
        or not isinstance(value["original_size"], int)
        or value["original_size"] <= 0
        or not isinstance(value["active_size"], int)
        or value["active_size"] <= 0
        or value["original_preserved"] is not True
        or value["active_differs"] is not True
        or value["status"] != "pass"
    ):
        raise D0Error("maintenance-window host evidence differs")
    _sha256(value["original_sha256"], "maintenance original SHA-256")
    _sha256(value["active_sha256"], "maintenance active SHA-256")
    if value["original_sha256"] == value["active_sha256"]:
        raise D0Error("maintenance deny did not change authorized_keys")
    return value


def validate_open(value: Any) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "purpose",
        "source_ip_denies",
        "supersedes_receipt_sha256",
        "hosts",
        "restore_gate",
        "project_code_executed",
        "protected_seed_values_opened",
        "status",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("maintenance-window receipt fields differ")
    hosts = value["hosts"]
    if (
        value["schema_id"] != OPEN_SCHEMA
        or value["schema_version"] != 2
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["purpose"] != "exclusive D0 control window against duplicate inbound controllers"
        or value["source_ip_denies"] != list(SOURCE_IPS)
        or not isinstance(value["supersedes_receipt_sha256"], str)
        or len(value["supersedes_receipt_sha256"]) != 64
        or not isinstance(hosts, list)
        or [item.get("host") for item in hosts if isinstance(item, dict)] != list(HOSTS)
        or any(_open_host(item) is not item for item in hosts)
        or value["restore_gate"]
        != {
            "required_before": "Phase 8 final goal completion",
            "restore_exact_original_bytes": True,
            "restore_mode": "0600",
            "remove_backup_after_verified_restore": True,
            "signed_restore_receipt_required": True,
        }
        or value["project_code_executed"] is not False
        or value["protected_seed_values_opened"] is not False
        or value["status"] != "active"
        or value["receipt_sha256"] != document_sha256(value, "receipt_sha256")
    ):
        raise D0Error("maintenance-window receipt identity differs")
    _sha256(value["supersedes_receipt_sha256"], "superseded maintenance receipt SHA-256")
    return value


def render_open(args: argparse.Namespace) -> dict[str, Any]:
    rows = [_open_host(_json(path, "maintenance host evidence")[1]) for path in args.host]
    rows.sort(key=lambda item: item["host"])
    receipt: dict[str, Any] = {
        "schema_id": OPEN_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "purpose": "exclusive D0 control window against duplicate inbound controllers",
        "source_ip_denies": list(SOURCE_IPS),
        "supersedes_receipt_sha256": args.supersedes_receipt_sha256,
        "hosts": rows,
        "restore_gate": {
            "required_before": "Phase 8 final goal completion",
            "restore_exact_original_bytes": True,
            "restore_mode": "0600",
            "remove_backup_after_verified_restore": True,
            "signed_restore_receipt_required": True,
        },
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "active",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    validate_open(receipt)
    _write_new(args.out, canonical_json(receipt))
    return receipt


def _restore_host(value: Any, original: dict[str, Any]) -> dict[str, Any]:
    required = {
        "host",
        "authorized_keys_path",
        "restored_sha256",
        "restored_size",
        "mode",
        "source_ip_denies_absent",
        "backup_absent",
        "restored_unix_ms",
        "status",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("maintenance restore host evidence fields differ")
    if (
        value["host"] != original["host"]
        or value["authorized_keys_path"] != original["authorized_keys_path"]
        or value["restored_sha256"] != original["original_sha256"]
        or value["restored_size"] != original["original_size"]
        or value["mode"] != "0600"
        or value["source_ip_denies_absent"] is not True
        or value["backup_absent"] is not True
        or not isinstance(value["restored_unix_ms"], int)
        or value["restored_unix_ms"] <= original["changed_unix_ms"]
        or value["status"] != "pass"
    ):
        raise D0Error("maintenance restore did not reproduce the original state")
    return value


def _verified_open(args: argparse.Namespace) -> tuple[bytes, dict[str, Any], bytes]:
    payload, value = _json(args.open_receipt, "maintenance-window receipt")
    opened = validate_open(value)
    signature_bytes, signature = _json(args.open_signature, "maintenance-window signature")
    public_key = load_public_key(args.public_key)
    verify_stdin(public_key, payload, signature)
    return payload, opened, signature_bytes


def render_restore(args: argparse.Namespace) -> dict[str, Any]:
    open_payload, opened, open_signature = _verified_open(args)
    evidence = [_json(path, "maintenance restore evidence")[1] for path in args.host]
    evidence.sort(key=lambda item: item.get("host", ""))
    if len(evidence) != len(opened["hosts"]):
        raise D0Error("maintenance restore host count differs")
    rows = [
        _restore_host(value, original)
        for value, original in zip(  # noqa: B905 -- Apple system Python is 3.9.
            evidence, opened["hosts"]
        )
    ]
    receipt: dict[str, Any] = {
        "schema_id": RESTORE_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "maintenance_receipt_sha256": opened["receipt_sha256"],
        "maintenance_receipt_file_sha256": sha256_bytes(open_payload),
        "maintenance_signature_file_sha256": sha256_bytes(open_signature),
        "hosts": rows,
        "restore_gate_satisfied": True,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    validate_restore(
        receipt, opened=opened, open_payload=open_payload, open_signature=open_signature
    )
    _write_new(args.out, canonical_json(receipt))
    return receipt


def validate_restore(
    value: Any,
    *,
    opened: dict[str, Any],
    open_payload: bytes,
    open_signature: bytes,
) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "maintenance_receipt_sha256",
        "maintenance_receipt_file_sha256",
        "maintenance_signature_file_sha256",
        "hosts",
        "restore_gate_satisfied",
        "project_code_executed",
        "protected_seed_values_opened",
        "status",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("maintenance restore receipt fields differ")
    hosts = value["hosts"]
    if (
        value["schema_id"] != RESTORE_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["maintenance_receipt_sha256"] != opened["receipt_sha256"]
        or value["maintenance_receipt_file_sha256"] != sha256_bytes(open_payload)
        or value["maintenance_signature_file_sha256"] != sha256_bytes(open_signature)
        or not isinstance(hosts, list)
        or len(hosts) != len(opened["hosts"])
        or any(
            _restore_host(item, original) is not item
            for item, original in zip(  # noqa: B905 -- Apple system Python is 3.9.
                hosts, opened["hosts"]
            )
        )
        or value["restore_gate_satisfied"] is not True
        or value["project_code_executed"] is not False
        or value["protected_seed_values_opened"] is not False
        or value["status"] != "pass"
        or value["receipt_sha256"] != document_sha256(value, "receipt_sha256")
    ):
        raise D0Error("maintenance restore receipt identity differs")
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    open_payload, opened, open_signature = _verified_open(args)
    restore_payload, restore_value = _json(args.restore_receipt, "maintenance restore receipt")
    restored = validate_restore(
        restore_value,
        opened=opened,
        open_payload=open_payload,
        open_signature=open_signature,
    )
    restore_signature_bytes, restore_signature = _json(
        args.restore_signature, "maintenance restore signature"
    )
    public_key = load_public_key(args.public_key)
    verify_stdin(public_key, restore_payload, restore_signature)
    return {
        "status": "pass",
        "restore_gate_satisfied": True,
        "maintenance_receipt_sha256": opened["receipt_sha256"],
        "restore_receipt_sha256": restored["receipt_sha256"],
        "restore_signature_file_sha256": sha256_bytes(restore_signature_bytes),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    opened = commands.add_parser("render-open")
    opened.add_argument("--host", type=Path, action="append", required=True)
    opened.add_argument("--supersedes-receipt-sha256", required=True)
    opened.add_argument("--out", type=Path, required=True)
    restore = commands.add_parser("render-restore")
    restore.add_argument("--open-receipt", type=Path, required=True)
    restore.add_argument("--open-signature", type=Path, required=True)
    restore.add_argument("--public-key", type=Path, required=True)
    restore.add_argument("--host", type=Path, action="append", required=True)
    restore.add_argument("--out", type=Path, required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--open-receipt", type=Path, required=True)
    verification.add_argument("--open-signature", type=Path, required=True)
    verification.add_argument("--restore-receipt", type=Path, required=True)
    verification.add_argument("--restore-signature", type=Path, required=True)
    verification.add_argument("--public-key", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        result = {
            "render-open": render_open,
            "render-restore": render_restore,
            "verify": verify,
        }[args.command](args)
    except (D0Error, OSError, KeyError, ValueError) as error:
        sys.stderr.write(f"r2-d0-maintenance-window: {error}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
