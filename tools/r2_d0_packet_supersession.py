#!/usr/bin/env python3
"""Render a signed-record payload superseding one pre-claim D0 packet."""

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
    load_canonical_json,
    sha256_bytes,
    validate_work_packet,
)

SCHEMA = "cascadia.r2-map.d0-preclaim-packet-supersession.v1"
FILES = (
    "work-packet.json",
    "work-packet-signature.json",
    "control-envelope.json",
    "ready-receipt.json",
)


def _read(path: Path) -> bytes:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
        or observed.st_size > 4 * 1024 * 1024
    ):
        raise D0Error(f"supersession input is unsafe: {path}")
    return path.read_bytes()


def _identity(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = []
    payloads = {}
    for name in FILES:
        payload = _read(root / name)
        payloads[name] = payload
        entries.append({"name": name, "size": len(payload), "sha256": sha256_bytes(payload)})
    packet = validate_work_packet(
        load_canonical_json(
            payloads["work-packet.json"], maximum=4 * 1024 * 1024, label="work packet"
        )
    )
    return packet, entries


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.fchmod(descriptor, 0o400)
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def render(args: argparse.Namespace) -> dict[str, Any]:
    old, old_files = _identity(args.old)
    new, new_files = _identity(args.new)
    if (
        old["schema_version"] != 9
        or new["schema_version"] != 10
        or old["campaign_id"] != CAMPAIGN_ID
        or new["campaign_id"] != CAMPAIGN_ID
        or old["run_id"] != D0_RUN_ID
        or new["run_id"] != D0_RUN_ID
        or any(old[field] != new[field] for field in ("cycle_id", "host", "phase", "role"))
        or old["allowed_operations"] != new["allowed_operations"]
        or old["predecessors"] != new["predecessors"]
        or old["policy"] != new["policy"]
        or old["helper_sha256"] == new["helper_sha256"]
        or not new["helper_transitions"]
        or new["helper_transitions"][-1]["document"]["to_helper_sha256"] != new["helper_sha256"]
    ):
        raise D0Error("preclaim packet supersession lineage differs")
    document: dict[str, Any] = {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "old_packet_sha256": old["packet_sha256"],
        "old_helper_sha256": old["helper_sha256"],
        "old_files": old_files,
        "old_remote_state": {
            "envelope_installed": True,
            "claim_absent": True,
            "completion_absent": True,
            "pending_transaction_absent": True,
        },
        "new_packet_sha256": new["packet_sha256"],
        "new_helper_sha256": new["helper_sha256"],
        "new_files": new_files,
        "helper_transition_sha256s": [
            item["document"]["transition_sha256"] for item in new["helper_transitions"]
        ],
        "reason": "legacy helper could not authorize signed historical predecessors",
        "old_packet_replay_authorized": False,
        "new_packet_dispatch_authorized": True,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "superseded-preclaim",
    }
    document["supersession_sha256"] = document_sha256(document, "supersession_sha256")
    _write_new(args.out, canonical_json(document))
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    try:
        value = render(parser.parse_args())
    except (D0Error, OSError) as error:
        sys.stderr.write(f"r2-d0-packet-supersession: {error}\n")
        return 2
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
