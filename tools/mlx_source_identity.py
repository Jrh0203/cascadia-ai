#!/usr/bin/env python3
"""Create a deterministic identity for the complete local MLX runtime source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

SOURCE_FILES = ("pyproject.toml", "uv.lock")
SOURCE_ROOT = Path("python/cascadia_mlx")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collect_source_identity(repository: Path, *, host: str | None = None) -> dict[str, Any]:
    repository = repository.resolve()
    paths = [repository / relative for relative in SOURCE_FILES]
    package_root = repository / SOURCE_ROOT
    paths.extend(
        path
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"MLX source identity is missing files: {missing}")

    entries = []
    bundle = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(repository).as_posix()
        payload = path.read_bytes()
        encoded = relative.encode()
        bundle.update(len(encoded).to_bytes(4, "little"))
        bundle.update(encoded)
        bundle.update(len(payload).to_bytes(8, "little"))
        bundle.update(payload)
        entries.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return {
        "schema_version": 1,
        "identity_kind": "complete-mlx-runtime-source-v1",
        "host": host or socket.gethostname().split(".")[0],
        "files": len(entries),
        "bundle_sha256": bundle.hexdigest(),
        "entries": entries,
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_source_identity(args.repository, host=args.host)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
