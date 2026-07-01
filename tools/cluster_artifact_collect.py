#!/usr/bin/env python3
"""Collect remote task artifacts onto the coordinator with checksum proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


class CollectError(RuntimeError):
    """Raised when a remote artifact cannot be collected exactly."""


LOCAL_COORDINATOR_HOSTS = frozenset({"john1", "localhost", "127.0.0.1", "::1"})


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_remote(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise CollectError(f"remote path lacks host prefix: {value}")
    host, path = value.split(":", 1)
    if not host or not path.startswith("/"):
        raise CollectError(f"invalid remote path: {value}")
    return host, path


def remote_sha256(host: str, path: str) -> str:
    completed = subprocess.run(
        ["ssh", host, "shasum", "-a", "256", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CollectError(f"remote checksum failed on {host}: {completed.stderr.strip()}")
    fields = completed.stdout.strip().split()
    if len(fields) < 2 or len(fields[0]) != 64:
        raise CollectError(f"invalid remote checksum output on {host}")
    return fields[0]


def collect(artifacts: list[tuple[str, Path]]) -> dict[str, Any]:
    if not artifacts:
        raise CollectError("at least one artifact is required")
    destinations = [str(destination) for _source, destination in artifacts]
    if len(destinations) != len(set(destinations)):
        raise CollectError("local artifact destinations must be unique")
    started = time.time_ns() // 1_000_000
    reports = []
    for source, destination in artifacts:
        host, remote_path = split_remote(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if host in LOCAL_COORDINATOR_HOSTS:
            source_path = Path(remote_path)
            if not source_path.is_file():
                raise CollectError(f"local coordinator artifact is missing: {source_path}")
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
            remote_digest = sha256(source_path)
        else:
            completed = subprocess.run(
                ["rsync", "-a", source, str(destination)],
                check=False,
            )
            if completed.returncode != 0:
                raise CollectError(
                    f"command exited {completed.returncode}: "
                    f"{shlex.join(['rsync', '-a', source, str(destination)])}"
                )
            remote_digest = remote_sha256(host, remote_path)
        local_digest = sha256(destination)
        if local_digest != remote_digest:
            raise CollectError(f"checksum mismatch while collecting {source}")
        reports.append(
            {
                "source": source,
                "destination": str(destination),
                "sha256": local_digest,
            }
        )
    return {
        "schema_version": 1,
        "artifacts": reports,
        "all_artifacts_match": True,
        "started_unix_ms": started,
        "completed_unix_ms": time.time_ns() // 1_000_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        nargs=2,
        metavar=("REMOTE_SOURCE", "LOCAL_DESTINATION"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect([(source, Path(destination)) for source, destination in args.artifact])
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
