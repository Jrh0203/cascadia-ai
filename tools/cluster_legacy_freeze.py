#!/usr/bin/env python3
"""Freeze the idle legacy queue as read-only migration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from cluster_research_queue import load_queue, queue_summary


class LegacyFreezeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_atomic(path: Path, value: dict[str, Any], mode: int = 0o444) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise LegacyFreezeError(f"existing freeze artifact differs: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_legacy_queue(queue: Path, *, freeze_id: str, observed_unix_ms: int) -> dict[str, Any]:
    state = load_queue(queue)
    summary = queue_summary(state)["summary"]
    if summary["ready"] != 0 or summary["running"] != 0:
        raise LegacyFreezeError("legacy queue must have zero ready and running tasks")
    root = queue.parent
    freeze_root = root / "legacy-freeze" / freeze_id
    if freeze_root.exists():
        raise LegacyFreezeError(f"freeze directory already exists: {freeze_root}")
    evidence_files = {queue}
    telemetry = root / "telemetry-v1.jsonl"
    if telemetry.is_file():
        evidence_files.add(telemetry)
    for relative in ("queue-archive", "queue-events", "experiment-specs"):
        directory = root / relative
        if directory.is_dir():
            evidence_files.update(path for path in directory.rglob("*") if path.is_file())
    entries = []
    for path in sorted(evidence_files):
        entries.append(
            {
                "path": path.relative_to(root.parent.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    freeze_root.mkdir(parents=True)
    snapshot = freeze_root / "research-queue-v1.json"
    shutil.copyfile(queue, snapshot)
    os.chmod(snapshot, 0o444)
    manifest = {
        "schema_id": "cascadia.cluster.legacy-queue-freeze.v1",
        "schema_version": 1,
        "freeze_id": freeze_id,
        "observed_unix_ms": observed_unix_ms,
        "campaign_id": state["campaign_id"],
        "queue_sha256": _sha256(queue),
        "snapshot_sha256": _sha256(snapshot),
        "summary": summary,
        "historical_files": entries,
        "new_submissions_disabled": True,
        "rollback_requires_explicit_flag": True,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    _write_atomic(freeze_root / "manifest.json", manifest)
    marker = {
        "schema_id": "cascadia.cluster.legacy-queue-write-freeze.v1",
        "schema_version": 1,
        "freeze_id": freeze_id,
        "queue_sha256": manifest["queue_sha256"],
        "manifest": str((freeze_root / "manifest.json").resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "new_submissions_disabled": True,
    }
    _write_atomic(root / "legacy-queue-freeze-v1.json", marker)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    arguments = parser.parse_args()
    try:
        manifest = freeze_legacy_queue(
            arguments.queue.resolve(strict=True),
            freeze_id=arguments.freeze_id,
            observed_unix_ms=time.time_ns() // 1_000_000,
        )
    except (LegacyFreezeError, OSError, ValueError) as error:
        parser.exit(2, f"legacy queue freeze refused: {error}\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
