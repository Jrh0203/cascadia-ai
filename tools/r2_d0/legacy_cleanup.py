"""Manifest-bound, crash-resumable cleanup of an archived John3 legacy tree."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

PACKET_SCHEMA = "cascadia.r2-map.john3-legacy-cleanup-packet.v1"
RECEIPT_SCHEMA = "cascadia.r2-map.john3-legacy-cleanup-receipt.v1"
SOURCE_ROOT = Path("/Users/john3/cascadia-bench/r2-map-v1")


class LegacyCleanupError(RuntimeError):
    """The cleanup packet or exact manifest-bound deletion failed."""


def _load_archive_helper(expected_sha256: str) -> ModuleType:
    """Load the exact staged sibling without relying on import search paths."""

    path = Path(__file__).absolute().with_name("legacy_archive_stream.py")
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise LegacyCleanupError("legacy archive helper path is unsafe")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LegacyCleanupError("legacy archive helper hash differs")
    specification = importlib.util.spec_from_file_location(
        "cascadia_r2_legacy_archive_stream_bound",
        path,
    )
    if specification is None or specification.loader is None:
        raise LegacyCleanupError("legacy archive helper cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash_document(value: Mapping[str, Any], field: str) -> str:
    document = dict(value)
    document.pop(field, None)
    return hashlib.sha256(_canonical(document)).hexdigest()


def build_cleanup_packet(
    *,
    manifest_file_sha256: str,
    manifest_sha256: str,
    root_tree_sha256: str,
    archive_plan_sha256: str,
    john2_commit_receipt_sha256: str,
    john1_reopen_receipt_sha256: str,
    archive_sha256: str,
    archive_size: int,
    goal_sha256: str,
    cleanup_helper_sha256: str,
    archive_helper_sha256: str,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema_id": PACKET_SCHEMA,
        "schema_version": 1,
        "campaign_id": "r2-map-expert-iteration-v1",
        "operation_id": "john3-legacy-native-workspace-cleanup-v1",
        "target_host": "john3",
        "source_root": str(SOURCE_ROOT),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": manifest_sha256,
        "root_tree_sha256": root_tree_sha256,
        "archive_plan_sha256": archive_plan_sha256,
        "john2_commit_receipt_sha256": john2_commit_receipt_sha256,
        "john1_reopen_receipt_sha256": john1_reopen_receipt_sha256,
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "goal_sha256": goal_sha256,
        "cleanup_helper_sha256": cleanup_helper_sha256,
        "archive_helper_sha256": archive_helper_sha256,
        "authorization": {
            "authorized_by": "root-orchestrator",
            "clear_uf_immutable": True,
            "delete_exact_manifest_tree": True,
            "delete_any_other_path": False,
            "external_ssd_authorized": False,
            "john4_authorized": False,
        },
    }
    packet["packet_sha256"] = _hash_document(packet, "packet_sha256")
    return validate_cleanup_packet(packet)


def validate_cleanup_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "operation_id",
        "target_host",
        "source_root",
        "manifest_file_sha256",
        "manifest_sha256",
        "root_tree_sha256",
        "archive_plan_sha256",
        "john2_commit_receipt_sha256",
        "john1_reopen_receipt_sha256",
        "archive_sha256",
        "archive_size",
        "goal_sha256",
        "cleanup_helper_sha256",
        "archive_helper_sha256",
        "authorization",
        "packet_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise LegacyCleanupError("legacy cleanup packet fields differ")
    for field in (
        "manifest_file_sha256",
        "manifest_sha256",
        "root_tree_sha256",
        "archive_plan_sha256",
        "john2_commit_receipt_sha256",
        "john1_reopen_receipt_sha256",
        "archive_sha256",
        "goal_sha256",
        "cleanup_helper_sha256",
        "archive_helper_sha256",
        "packet_sha256",
    ):
        digest = value[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise LegacyCleanupError(f"legacy cleanup {field} differs")
    authorization = value["authorization"]
    if (
        value["schema_id"] != PACKET_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != "r2-map-expert-iteration-v1"
        or value["operation_id"] != "john3-legacy-native-workspace-cleanup-v1"
        or value["target_host"] != "john3"
        or value["source_root"] != str(SOURCE_ROOT)
        or not isinstance(value["archive_size"], int)
        or isinstance(value["archive_size"], bool)
        or value["archive_size"] <= 0
        or authorization
        != {
            "authorized_by": "root-orchestrator",
            "clear_uf_immutable": True,
            "delete_exact_manifest_tree": True,
            "delete_any_other_path": False,
            "external_ssd_authorized": False,
            "john4_authorized": False,
        }
        or value["packet_sha256"] != _hash_document(value, "packet_sha256")
    ):
        raise LegacyCleanupError("legacy cleanup packet identity differs")
    return dict(value)


def encode_cleanup_packet(value: Mapping[str, Any]) -> bytes:
    return _canonical(validate_cleanup_packet(value))


def _write_once(path: Path, payload: bytes, *, mode: int = 0o400) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            raise LegacyCleanupError("legacy cleanup durable document collision")
        return "present"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.write(descriptor, payload)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return "installed"


def _remaining_entries(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    observed: list[dict[str, Any]] = []
    pending = [(root, ".")]
    while pending:
        path, relative = pending.pop()
        details = os.lstat(path)
        item = expected.get(relative)
        if item is None:
            raise LegacyCleanupError("legacy cleanup staging contains an unexpected path")
        kind = "directory" if stat.S_ISDIR(details.st_mode) else "file"
        if (
            item["type"] != kind
            or details.st_dev != item["device"]
            or details.st_ino != item["inode"]
            or stat.S_IMODE(details.st_mode) != int(item["mode"], 8)
            or details.st_size != item["size"]
            or details.st_mtime_ns != item["mtime_ns"]
            or getattr(details, "st_flags", item["flags_before"])
            not in {item["flags_before"], item["flags_after"]}
        ):
            raise LegacyCleanupError(f"legacy cleanup staging metadata differs: {relative}")
        if kind == "file":
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            digest = hashlib.sha256()
            try:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            if digest.hexdigest() != item["sha256"]:
                raise LegacyCleanupError(f"legacy cleanup staging content differs: {relative}")
        else:
            children = sorted(os.scandir(path), key=lambda entry: entry.name, reverse=True)
            for child in children:
                details = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(details.st_mode):
                    raise LegacyCleanupError("legacy cleanup staging contains a symlink")
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                pending.append((Path(child.path), child_relative))
        observed.append(item)
    return observed


def execute_cleanup(
    packet: Mapping[str, Any],
    manifest_payload: bytes,
    *,
    source_root: Path = SOURCE_ROOT,
    receipt_root: Path = Path("/Users/john3/.config/cascadia-r2/legacy-cleanup-receipts"),
    current_host: str | None = None,
    flag_setter: Callable[[Path, int], None] | None = None,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    validated = validate_cleanup_packet(packet)
    archive_helper = _load_archive_helper(validated["archive_helper_sha256"])
    host = getpass.getuser() if current_host is None else current_host
    if host not in {"john3"}:
        raise LegacyCleanupError("legacy cleanup may execute only as John3")
    if source_root.resolve(strict=False) != Path(validated["source_root"]).resolve(strict=False):
        raise LegacyCleanupError("legacy cleanup source root differs from its packet")
    if hashlib.sha256(manifest_payload).hexdigest() != validated["manifest_file_sha256"]:
        raise LegacyCleanupError("legacy cleanup manifest file hash differs")
    try:
        manifest = archive_helper.load_manifest(manifest_payload)
    except archive_helper.LegacyArchiveError as error:
        raise LegacyCleanupError(str(error)) from error
    if (
        manifest["manifest_sha256"] != validated["manifest_sha256"]
        or manifest["totals"]["root_tree_sha256"] != validated["root_tree_sha256"]
    ):
        raise LegacyCleanupError("legacy cleanup manifest binding differs")
    packet_sha256 = validated["packet_sha256"]
    receipt_path = receipt_root / f"{packet_sha256}.complete.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        if (
            receipt.get("receipt_sha256") != _hash_document(receipt, "receipt_sha256")
            or receipt.get("packet_sha256") != packet_sha256
            or receipt.get("status") != "pass"
            or source_root.exists()
        ):
            raise LegacyCleanupError("legacy cleanup replay state differs")
        return receipt
    intent = {
        "schema_id": "cascadia.r2-map.john3-legacy-cleanup-intent.v1",
        "schema_version": 1,
        "packet_sha256": packet_sha256,
        "manifest_file_sha256": validated["manifest_file_sha256"],
        "source_root": str(source_root),
        "source_delete_authorized": True,
    }
    intent["intent_sha256"] = _hash_document(intent, "intent_sha256")
    intent_path = receipt_root / f"{packet_sha256}.intent.json"
    _write_once(intent_path, _canonical(intent), mode=0o400)
    staging = source_root.with_name(f".{source_root.name}.cleanup-{packet_sha256}")
    fault = fault_injector or (lambda _point: None)
    setter = flag_setter
    if setter is None:
        if not hasattr(os, "chflags"):
            raise LegacyCleanupError("legacy cleanup requires no-follow chflags")

        def set_flags(path: Path, flags: int) -> None:
            os.chflags(path, flags, follow_symlinks=False)

        setter = set_flags
    if source_root.exists() and staging.exists():
        raise LegacyCleanupError("legacy cleanup source and staging both exist")
    if source_root.exists():
        try:
            archive_helper.verify_frozen_source(source_root, manifest)
        except archive_helper.LegacyArchiveError:
            # A crash after flag clearing but before rename is recoverable only
            # when every other frozen identity still matches.
            _remaining_entries(source_root, manifest)
        fault("after-verify")
        for entry in sorted(manifest["entries"], key=lambda item: item["path"], reverse=True):
            path = source_root if entry["path"] == "." else source_root / entry["path"]
            setter(path, entry["flags_before"])
        _remaining_entries(source_root, manifest)
        fault("after-clear-flags")
        os.replace(source_root, staging)
        parent = os.open(source_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        fault("after-rename")
    if not staging.exists():
        raise LegacyCleanupError("legacy cleanup source and staging are both absent")
    remaining = _remaining_entries(staging, manifest)
    files = sorted(
        (item for item in remaining if item["type"] == "file"),
        key=lambda item: (item["path"].count("/"), item["path"]),
        reverse=True,
    )
    directories = sorted(
        (item for item in remaining if item["type"] == "directory" and item["path"] != "."),
        key=lambda item: (item["path"].count("/"), item["path"]),
        reverse=True,
    )
    for index, entry in enumerate(files):
        (staging / entry["path"]).unlink()
        if index == 0:
            fault("after-first-delete")
    for entry in directories:
        (staging / entry["path"]).rmdir()
    staging.rmdir()
    parent = os.open(source_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    if source_root.exists() or staging.exists():
        raise LegacyCleanupError("legacy cleanup did not remove its exact tree")
    receipt: dict[str, Any] = {
        "schema_id": RECEIPT_SCHEMA,
        "schema_version": 1,
        "packet_sha256": packet_sha256,
        "manifest_sha256": validated["manifest_sha256"],
        "root_tree_sha256": validated["root_tree_sha256"],
        "archive_sha256": validated["archive_sha256"],
        "archive_plan_sha256": validated["archive_plan_sha256"],
        "john2_commit_receipt_sha256": validated["john2_commit_receipt_sha256"],
        "john1_reopen_receipt_sha256": validated["john1_reopen_receipt_sha256"],
        "deleted_entry_count": manifest["totals"]["entry_count"],
        "deleted_unique_regular_bytes": manifest["totals"]["unique_regular_bytes"],
        "source_root_absent": True,
        "completed_unix_ms": time.time_ns() // 1_000_000,
        "status": "pass",
    }
    receipt["receipt_sha256"] = _hash_document(receipt, "receipt_sha256")
    _write_once(receipt_path, _canonical(receipt), mode=0o400)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--confirm-packet-sha256", required=True)
    arguments = parser.parse_args()
    try:
        packet = json.loads(arguments.packet.read_bytes())
        validated = validate_cleanup_packet(packet)
        if arguments.confirm_packet_sha256 != validated["packet_sha256"]:
            raise LegacyCleanupError("legacy cleanup confirmation differs")
        result = execute_cleanup(validated, arguments.manifest.read_bytes())
    except (OSError, json.JSONDecodeError, LegacyCleanupError) as error:
        print(f"legacy cleanup failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
