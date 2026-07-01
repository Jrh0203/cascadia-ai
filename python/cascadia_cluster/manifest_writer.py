"""Create the strict output manifest required by the cluster result importer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .results import MANIFEST_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    root: Path,
    *,
    command: list[str],
    application_metadata: dict,
    protocol_version: str,
) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir() or not command:
        raise ValueError("manifest root must be a directory and command must be nonempty")
    files = []
    for path in sorted(root.rglob("*")):
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise ValueError(f"output symlinks are forbidden: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    value = {
        "schema_id": "cascadia.cluster.output-manifest.v1",
        "protocol_version": protocol_version,
        "command": command,
        "files": files,
        "application_metadata": application_metadata,
    }
    descriptor, name = tempfile.mkstemp(prefix=".manifest.", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / MANIFEST_NAME)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--protocol-version", default="cascadia-cluster-map-v1")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    metadata = json.loads(arguments.metadata_json)
    if not isinstance(metadata, dict):
        raise SystemExit("application metadata must be a JSON object")
    write_manifest(
        arguments.root,
        command=command,
        application_metadata=metadata,
        protocol_version=arguments.protocol_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
