#!/usr/bin/env python3
"""Render an immutable manifest for one exact D0 helper-rotation package."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2_d0.canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    D0Error,
    canonical_json,
    document_sha256,
    sha256_bytes,
)

FILES = (
    "accepted-lineage.json",
    "authorization-signature.json",
    "authorization.json",
    "campaign-public-key",
    "current-bootstrap-receipt.json",
    "installer.py",
    "new-bootstrap-packet.json",
    "new-helper.tar",
    "old-helper.tar",
)


def _read(path: Path) -> bytes:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
        or observed.st_size > 16 * 1024 * 1024
    ):
        raise D0Error(f"rotation package input is unsafe: {path.name}")
    return path.read_bytes()


def _write_new(path: Path, payload: bytes) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("john1", "john2", "john3"), required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        observed = sorted(path.name for path in args.package_root.iterdir())
        if observed != sorted(FILES):
            raise D0Error("rotation package file set differs")
        entries = []
        for name in FILES:
            payload = _read(args.package_root / name)
            entries.append({"name": name, "size": len(payload), "sha256": sha256_bytes(payload)})
        manifest = {
            "schema_id": "cascadia.r2-map.d0-helper-rotation-package.v2",
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "run_id": D0_RUN_ID,
            "host": args.host,
            "files": entries,
        }
        manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
        _write_new(args.out, canonical_json(manifest))
    except (D0Error, OSError) as error:
        sys.stderr.write(f"r2-d0-rotation-package: {error}\n")
        return 2
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
