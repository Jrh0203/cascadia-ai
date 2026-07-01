#!/usr/bin/env python3
"""Collect immutable remote artifact trees with whole-tree checksum proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

LOCAL_COORDINATOR_HOSTS = frozenset({"john1", "localhost", "127.0.0.1", "::1"})


class TreeCollectError(RuntimeError):
    """Raised when an immutable artifact tree cannot be collected exactly."""


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
    if not root.is_dir():
        raise TreeCollectError(f"artifact tree is not a directory: {root}")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TreeCollectError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256(path)
    if not hashes:
        raise TreeCollectError(f"artifact tree is empty: {root}")
    return hashes


def tree_manifest_sha256(hashes: dict[str, str]) -> str:
    encoded = json.dumps(
        hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def split_remote(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise TreeCollectError(f"remote tree lacks host prefix: {value}")
    host, path = value.split(":", 1)
    if not host or not path.startswith("/"):
        raise TreeCollectError(f"invalid remote tree: {value}")
    return host, path


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
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise TreeCollectError(f"remote tree checksum failed on {host}: {detail}")
    try:
        hashes = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise TreeCollectError(f"invalid remote tree checksum output on {host}") from error
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64
            for path, digest in hashes.items()
        )
    ):
        raise TreeCollectError(f"invalid remote tree checksum payload on {host}")
    return hashes


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise TreeCollectError(f"command exited {completed.returncode}: {shlex.join(command)}")


def _validate_destinations(trees: list[tuple[str, Path]]) -> None:
    if not trees:
        raise TreeCollectError("at least one artifact tree is required")
    destinations = [destination.resolve(strict=False) for _, destination in trees]
    if len(destinations) != len(set(destinations)):
        raise TreeCollectError("local artifact-tree destinations must be unique")
    for index, left in enumerate(destinations):
        for right in destinations[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise TreeCollectError("local artifact-tree destinations must not be nested")


def _copy_to_staging(source: str, destination: Path) -> tuple[dict[str, str], bool]:
    host, remote_root = split_remote(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.collecting-{uuid.uuid4().hex}")
    reused = False
    try:
        if host in LOCAL_COORDINATOR_HOSTS:
            source_path = Path(remote_root)
            source_before = tree_sha256(source_path)
            if source_path.resolve() == destination.resolve() and destination.is_dir():
                local_hashes = source_before
                source_after = tree_sha256(source_path)
                reused = True
            else:
                shutil.copytree(source_path, staging, copy_function=shutil.copy2)
                source_after = tree_sha256(source_path)
                local_hashes = tree_sha256(staging)
        else:
            source_before = remote_tree_sha256(host, remote_root)
            staging.mkdir()
            _run(["rsync", "-a", source.rstrip("/") + "/", f"{staging}/"])
            source_after = remote_tree_sha256(host, remote_root)
            local_hashes = tree_sha256(staging)
        if source_before != source_after:
            raise TreeCollectError(f"source tree changed during collection: {source}")
        if local_hashes != source_after:
            raise TreeCollectError(f"whole-tree checksum mismatch collecting {source}")
        if not reused and destination.exists():
            existing_hashes = tree_sha256(destination)
            if existing_hashes != local_hashes:
                raise TreeCollectError(
                    f"immutable destination already exists with different content: {destination}"
                )
            reused = True
        if not reused:
            os.replace(staging, destination)
        return local_hashes, reused
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def collect_trees(trees: list[tuple[str, Path]]) -> dict[str, Any]:
    _validate_destinations(trees)
    started = time.time_ns() // 1_000_000
    reports = []
    for source, destination in trees:
        hashes, reused = _copy_to_staging(source, destination)
        reports.append(
            {
                "source": source,
                "destination": str(destination),
                "file_count": len(hashes),
                "byte_count": sum((destination / relative).stat().st_size for relative in hashes),
                "tree_manifest_sha256": tree_manifest_sha256(hashes),
                "tree_sha256": hashes,
                "reused": reused,
            }
        )
    return {
        "schema_version": 1,
        "trees": reports,
        "all_trees_match": True,
        "started_unix_ms": started,
        "completed_unix_ms": time.time_ns() // 1_000_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree",
        action="append",
        nargs=2,
        metavar=("REMOTE_SOURCE", "LOCAL_DESTINATION"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_trees([(source, Path(destination)) for source, destination in args.tree])
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
