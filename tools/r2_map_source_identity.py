#!/usr/bin/env python3
"""Compute the same complete V2 source identity embedded by Rust provenance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import blake3

SOURCE_ROOTS = (
    "Cargo.toml",
    "Cargo.lock",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "python/cascadia_mlx",
    "apps/web/src",
    "legacy/crates/cascadia-core",
    "legacy/crates/cascadia-ai",
    "crates/cascadia-game",
    "crates/cascadia-sim",
    "crates/cascadia-data",
    "crates/cascadia-model",
    "crates/cascadia-eval",
    "crates/cascadia-search",
    "crates/cascadia-api",
    "crates/cascadia-cli-v2",
    "crates/cascadia-differential",
    "crates/cascadia-provenance",
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def source_identity(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    files: list[Path] = []
    for relative in SOURCE_ROOTS:
        root = repository / relative
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    files.sort()
    digest = blake3.blake3()
    entries = []
    for path in files:
        relative = path.relative_to(repository).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(payload)
        entries.append(
            {
                "path": relative.decode(),
                "bytes": len(payload),
                "blake3": blake3.blake3(payload).hexdigest(),
            }
        )
    status = _git(repository, "status", "--porcelain=v1")
    return {
        "schema_version": 1,
        "schema_id": "cascadia.v2-source-identity.v1",
        "git_revision": _git(repository, "rev-parse", "HEAD"),
        "git_dirty": status == "unavailable" or bool(status),
        "git_status_blake3": blake3.blake3(status.encode()).hexdigest(),
        "v2_source_blake3": digest.hexdigest(),
        "files": len(entries),
        "entries": entries,
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    value = source_identity(arguments.repository)
    if arguments.output is not None:
        write_atomic(arguments.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
