#!/usr/bin/env python3
"""Render an immutable quarantine manifest for unowned D0 executor artifacts."""

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

SCHEMA = "cascadia.r2-map.d0-collision-quarantine.v1"
MAX_FILE = 2 * 1024 * 1024 * 1024


def _read(path: Path) -> bytes:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
        or observed.st_size > MAX_FILE
    ):
        raise D0Error(f"quarantine artifact metadata is unsafe: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        remaining = MAX_FILE + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != observed.st_size:
        raise D0Error(f"quarantine artifact changed while reading: {path}")
    return payload


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _semantic_projection(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    projection: dict[str, Any] = {"schema_id": value.get("schema_id")}
    for field in (
        "packet_sha256",
        "report_sha256",
        "receipt_sha256",
        "authorization_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "supersession_sha256",
    ):
        item = value.get(field)
        if isinstance(item, str) and len(item) == 64:
            projection[field] = item
    return projection


def render(args: argparse.Namespace) -> dict[str, Any]:
    campaign_root = args.campaign_root.resolve()
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in args.quarantine_root:
        resolved = root.resolve()
        if campaign_root not in resolved.parents:
            raise D0Error("quarantine root escapes the canonical campaign")
        for path in sorted(resolved.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink() or path in seen:
                raise D0Error("quarantine contains a symlink or duplicate path")
            seen.add(path)
            payload = _read(path)
            artifacts.append(
                {
                    "path": str(path),
                    "size": len(payload),
                    "file_sha256": sha256_bytes(payload),
                    "semantic": _semantic_projection(payload),
                }
            )
    accepted = list(args.accepted_report)
    quarantined = list(args.quarantined_report)
    if (
        len(set(accepted)) != len(accepted)
        or len(set(quarantined)) != len(quarantined)
        or set(accepted) & set(quarantined)
        or not quarantined
        or not artifacts
    ):
        raise D0Error("quarantine report classification differs")
    document: dict[str, Any] = {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "incident_kind": "unowned-local-app-server-executor-collision",
        "disposition": "preserved-in-place-excluded-from-accepted-lineage",
        "accepted_report_sha256s": accepted,
        "quarantined_report_sha256s": quarantined,
        "quarantined_artifacts": artifacts,
        "quarantined_artifact_count": len(artifacts),
        "canonical_state_mutation_authorized": False,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "quarantined",
    }
    document["incident_sha256"] = document_sha256(document, "incident_sha256")
    _write_new(args.out, canonical_json(document))
    return document


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--campaign-root", type=Path, required=True)
    value.add_argument("--quarantine-root", type=Path, action="append", required=True)
    value.add_argument("--accepted-report", action="append", required=True)
    value.add_argument("--quarantined-report", action="append", required=True)
    value.add_argument("--out", type=Path, required=True)
    return value


def main() -> int:
    try:
        result = render(parser().parse_args())
    except (D0Error, OSError) as error:
        sys.stderr.write(f"r2-d0-collision-incident: {error}\n")
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
