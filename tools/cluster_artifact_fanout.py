#!/usr/bin/env python3
"""Retrieve one immutable run and verify checksummed fan-out to remote hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


class FanoutError(RuntimeError):
    """Raised when transfer or cross-host verification fails."""


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


def tree_sha256(root: Path) -> dict[str, str]:
    """Hash every regular file beneath a tree using stable relative paths."""
    if not root.is_dir():
        raise FanoutError(f"artifact tree is not a directory: {root}")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FanoutError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256(path)
    if not hashes:
        raise FanoutError(f"artifact tree is empty: {root}")
    return hashes


def split_remote(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise FanoutError(f"remote path lacks host prefix: {value}")
    host, path = value.split(":", 1)
    if not host or not path.startswith("/"):
        raise FanoutError(f"invalid remote path: {value}")
    return host, path


def remote_sha256(host: str, path: str) -> str:
    completed = subprocess.run(
        ["ssh", host, "shasum", "-a", "256", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise FanoutError(f"remote checksum failed on {host}: {completed.stderr.strip()}")
    fields = completed.stdout.strip().split()
    if len(fields) < 2 or len(fields[0]) != 64:
        raise FanoutError(f"invalid remote checksum output on {host}")
    return fields[0]


def remote_tree_sha256(host: str, root: str) -> dict[str, str]:
    script = """
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit("remote artifact tree is not a directory")
hashes = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"remote artifact tree contains a symlink: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
if not hashes:
    raise SystemExit("remote artifact tree is empty")
print(json.dumps(hashes, sort_keys=True))
""".strip()
    remote_command = shlex.join(["python3", "-c", script, root])
    completed = subprocess.run(
        ["ssh", host, remote_command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise FanoutError(
            f"remote tree checksum failed on {host}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        hashes = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise FanoutError(f"invalid remote tree checksum output on {host}") from error
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64
            for path, digest in hashes.items()
        )
    ):
        raise FanoutError(f"invalid remote tree checksum payload on {host}")
    return hashes


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise FanoutError(f"command exited {completed.returncode}: {shlex.join(command)}")


def fanout(
    *,
    source: str,
    local_root: Path,
    destinations: list[str],
    required_files: list[str],
    verify_tree: bool = False,
) -> dict[str, Any]:
    if not source.endswith("/") or not destinations:
        raise FanoutError("source must end in / and at least one destination is required")
    if any(not value.endswith("/") for value in destinations):
        raise FanoutError("every destination must end in /")
    if len(required_files) != len(set(required_files)):
        raise FanoutError("required files must be unique")
    if not required_files and not verify_tree:
        raise FanoutError("required files or whole-tree verification is required")
    if any(Path(value).is_absolute() or ".." in Path(value).parts for value in required_files):
        raise FanoutError("required files must remain beneath the run root")
    started = time.time_ns() // 1_000_000
    source_path = None if ":" in source else Path(source.rstrip("/"))
    same_local_tree = source_path is not None and source_path.resolve() == local_root.resolve()
    if not same_local_tree:
        if local_root.exists():
            if any(local_root.iterdir()):
                raise FanoutError(f"local artifact destination is not empty: {local_root}")
        else:
            local_root.mkdir(parents=True)
        _run(["rsync", "-a", source, f"{local_root}/"])
    elif not local_root.is_dir():
        raise FanoutError(f"local artifact source is missing: {local_root}")
    local_hashes = {}
    for relative in required_files:
        path = local_root / relative
        if not path.is_file():
            raise FanoutError(f"required artifact is missing: {path}")
        local_hashes[relative] = sha256(path)
    local_tree_hashes = tree_sha256(local_root) if verify_tree else None
    destination_reports = []
    for destination in destinations:
        host, remote_root = split_remote(destination)
        _run(["rsync", "-a", "--delete", f"{local_root}/", destination])
        remote_hashes = {
            relative: remote_sha256(
                host,
                f"{remote_root.rstrip('/')}/{relative}",
            )
            for relative in required_files
        }
        if remote_hashes != local_hashes:
            raise FanoutError(f"checksum mismatch after fan-out to {host}")
        remote_tree_hashes = remote_tree_sha256(host, remote_root) if verify_tree else None
        if remote_tree_hashes != local_tree_hashes:
            raise FanoutError(f"whole-tree checksum mismatch after fan-out to {host}")
        destination_reports.append(
            {
                "host": host,
                "root": remote_root,
                "sha256": remote_hashes,
                "tree_sha256": remote_tree_hashes,
            }
        )
    return {
        "schema_version": 1,
        "source": source,
        "local_root": str(local_root),
        "required_files": required_files,
        "local_sha256": local_hashes,
        "local_tree_sha256": local_tree_hashes,
        "whole_tree_verified": verify_tree,
        "destinations": destination_reports,
        "started_unix_ms": started,
        "completed_unix_ms": time.time_ns() // 1_000_000,
        "all_destinations_match": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--destination", action="append", required=True)
    parser.add_argument("--required-file", action="append", default=[])
    parser.add_argument(
        "--verify-tree",
        action="store_true",
        help="Require every regular file and checksum to match on each host",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fanout(
        source=args.source,
        local_root=args.local_root,
        destinations=args.destination,
        required_files=args.required_file,
        verify_tree=args.verify_tree,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
