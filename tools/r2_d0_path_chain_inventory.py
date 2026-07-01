#!/usr/bin/env python3
"""Read-only, exact path-chain inventory for a signed D0 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

MAX_AUTHORIZATION_BYTES = 1024 * 1024
MAX_ROOTS = 128


class InventoryError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_sha256(value: dict[str, Any], field: str) -> str:
    return sha256_bytes(canonical_json({key: item for key, item in value.items() if key != field}))


def _sha256_file(path: Path, expected_size: int) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_size:
            raise InventoryError(f"file changed before hashing: {path}")
        total = 0
        while total < expected_size:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size - total))
            if not chunk:
                raise InventoryError(f"short read while hashing: {path}")
            digest.update(chunk)
            total += len(chunk)
        if os.read(descriptor, 1):
            raise InventoryError(f"file grew while hashing: {path}")
        closed = os.fstat(descriptor)
        if (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise InventoryError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character-device"
    if stat.S_ISBLK(mode):
        return "block-device"
    if stat.S_ISFIFO(mode):
        return "fifo"
    return "other"


def path_record(path: Path) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "present": False}
    record: dict[str, Any] = {
        "path": str(path),
        "present": True,
        "type": _kind(observed.st_mode),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "mode": format(stat.S_IMODE(observed.st_mode), "04o"),
        "nlink": observed.st_nlink,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
    }
    if stat.S_ISLNK(observed.st_mode):
        record["symlink_target"] = os.readlink(path)
    return record


def path_chain(path: Path) -> list[dict[str, Any]]:
    current = Path("/")
    result = [path_record(current)]
    for part in path.parts[1:]:
        current /= part
        result.append(path_record(current))
    return result


def _directory_listing(path: Path) -> dict[str, Any]:
    entries = []
    for name in sorted(os.listdir(path), key=os.fsencode):
        child = path / name
        record = path_record(child)
        record.pop("path", None)
        record["name"] = name
        entries.append(record)
    encoded = canonical_json(entries)
    return {
        "entry_count": len(entries),
        "listing_sha256": sha256_bytes(encoded),
        "entries": entries,
    }


def _homebrew_owner(path: Path, max_hash_bytes: int) -> dict[str, Any] | None:
    parts = path.parts
    try:
        index = parts.index("Cellar")
        formula, version = parts[index + 1 : index + 3]
    except (ValueError, IndexError):
        return None
    keg = Path(*parts[: index + 3])
    receipt = keg / "INSTALL_RECEIPT.json"
    receipt_record = path_record(receipt)
    if receipt_record.get("type") == "file":
        size = receipt_record["size"]
        if size > max_hash_bytes:
            raise InventoryError(f"Homebrew receipt exceeds hash bound: {receipt}")
        receipt_record["content_sha256"] = _sha256_file(receipt, size)
    return {
        "formula": formula,
        "version": version,
        "keg_root": str(keg),
        "install_receipt": receipt_record,
    }


def inspect_root(path: Path, max_hash_bytes: int) -> dict[str, Any]:
    chain = path_chain(path)
    symlink_ancestors = [item for item in chain[:-1] if item.get("type") == "symlink"]
    resolved = Path(os.path.realpath(path))
    resolved_record = path_record(resolved)
    content: dict[str, Any] | None = None
    if resolved_record.get("type") == "file":
        size = resolved_record["size"]
        if size > max_hash_bytes:
            raise InventoryError(f"authorized file exceeds hash bound: {resolved}")
        content = {"kind": "full-file", "sha256": _sha256_file(resolved, size)}
    elif resolved_record.get("type") == "directory":
        content = {"kind": "immediate-directory", **_directory_listing(resolved)}
    return {
        "requested_root": str(path),
        "path_chain": chain,
        "symlink_ancestors": symlink_ancestors,
        "root_is_symlink": bool(chain and chain[-1].get("type") == "symlink"),
        "resolved_path": str(resolved),
        "resolved_record": resolved_record,
        "resolved_content": content,
        "homebrew_owner": _homebrew_owner(resolved, max_hash_bytes),
    }


def snapshot(roots: list[Path], max_hash_bytes: int) -> dict[str, Any]:
    reports = [inspect_root(path, max_hash_bytes) for path in roots]
    value = {"roots": reports}
    value["snapshot_sha256"] = sha256_bytes(canonical_json(value))
    return value


def swap_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "vm.swapusage"],
        capture_output=True,
        check=False,
        timeout=10,
    )
    output = completed.stdout.decode("ascii", "replace")
    return {
        "returncode": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "zero": completed.returncode == 0
        and "total = 0.00M" in output
        and "used = 0.00M" in output
        and "free = 0.00M" in output,
    }


def run(authorization: dict[str, Any]) -> dict[str, Any]:
    if (
        authorization.get("schema_id")
        != "cascadia.r2-map.d0-path-chain-inventory-authorization.v1"
        or authorization.get("authorization_sha256")
        != document_sha256(authorization, "authorization_sha256")
        or authorization.get("host") != "john2"
        or authorization.get("status") != "authorized-once"
        or authorization.get("read_only") is not True
        or authorization.get("project_code_executed") is not False
        or authorization.get("protected_seed_values_opened") is not False
        or authorization.get("scanner_executed") is not False
        or authorization.get("qualification_claimed") is not False
        or int(authorization.get("expires_unix_ms", 0)) <= time.time_ns() // 1_000_000
    ):
        raise InventoryError("path-chain authorization identity or expiry differs")
    raw_roots = authorization.get("roots")
    max_hash_bytes = authorization.get("max_file_hash_bytes")
    if (
        not isinstance(raw_roots, list)
        or not 1 <= len(raw_roots) <= MAX_ROOTS
        or len(set(raw_roots)) != len(raw_roots)
        or any(not isinstance(item, str) or not Path(item).is_absolute() for item in raw_roots)
        or not isinstance(max_hash_bytes, int)
        or isinstance(max_hash_bytes, bool)
        or not 1 <= max_hash_bytes <= 1024**3
    ):
        raise InventoryError("path-chain root or hash bound differs")
    roots = [Path(item) for item in raw_roots]
    swap_before = swap_snapshot()
    before = snapshot(roots, max_hash_bytes)
    after = snapshot(roots, max_hash_bytes)
    swap_after = swap_snapshot()
    if before != after:
        raise InventoryError("path-chain inventory mutated or changed during the diagnostic")
    if swap_before["zero"] is not True or swap_after["zero"] is not True:
        raise InventoryError("path-chain diagnostic requires zero swap")
    result: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.d0-path-chain-inventory-result.v1",
        "schema_version": 1,
        "campaign_id": authorization["campaign_id"],
        "run_id": authorization["run_id"],
        "host": "john2",
        "authorization_sha256": authorization["authorization_sha256"],
        "failure_report_sha256": authorization["failure_report_sha256"],
        "before": before,
        "after": after,
        "nonmutation_proven": True,
        "swap_before": swap_before,
        "swap_after": swap_after,
        "read_only": True,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "scanner_executed": False,
        "qualification_claimed": False,
        "status": "pass-diagnostic",
    }
    result["result_sha256"] = document_sha256(result, "result_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = args.authorization.read_bytes()
        if len(payload) > MAX_AUTHORIZATION_BYTES:
            raise InventoryError("path-chain authorization exceeds its byte bound")
        authorization = json.loads(payload)
        if not isinstance(authorization, dict):
            raise InventoryError("path-chain authorization is not an object")
        print(json.dumps(run(authorization), sort_keys=True, separators=(",", ":")))
        return 0
    except (InventoryError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"r2-d0-path-chain-inventory: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
