#!/usr/bin/env python3
"""Project John1's canonical R2-MAP status into a bounded serving mirror.

The canonical payload lives in the owner-private active-campaign root on
John1's internal APFS volume.  This process opens that one registered file
with no-follow semantics, validates its ownership, mode, device, size, and
schema, then atomically replaces the bounded read-only source-tree projection.
It never enumerates the campaign tree, uses SSH, or touches external storage.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import blake3

REPOSITORY = Path(__file__).resolve().parents[1]
CANONICAL_HOST = "john1"
CANONICAL_ROOT = Path("/Users/johnherrick/cascadia-bench/r2-map-v1")
CANONICAL_PATH = CANONICAL_ROOT / "control/dashboard-status.json"
SERVING_PATH = REPOSITORY / "artifacts/cluster/r2-map-dashboard-serving-projection-v2.json"
STATUS_SCHEMA_ID = "cascadia.r2-map.dashboard-status.v1"
PROJECTION_SCHEMA_ID = "cascadia.r2-map.dashboard-serving-projection.v2"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
ALLOWED_HOSTS = {"john1", "john2", "john3"}
MAX_CANONICAL_BYTES = 64 << 10
MAX_PROJECTION_BYTES = 64 << 10


class DashboardFetchError(RuntimeError):
    """The authenticated fetch or serving projection contract failed."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def validate_canonical_payload(payload: bytes) -> dict[str, Any]:
    """Validate the source envelope before the stricter Rust semantic reader."""
    if not payload:
        raise DashboardFetchError("canonical dashboard status is empty")
    if len(payload) > MAX_CANONICAL_BYTES:
        raise DashboardFetchError(
            f"canonical dashboard status is {len(payload)} bytes; maximum is {MAX_CANONICAL_BYTES}"
        )
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DashboardFetchError(f"canonical dashboard status is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DashboardFetchError("canonical dashboard status must be a JSON object")
    if value.get("schema_id") != STATUS_SCHEMA_ID or value.get("schema_version") != 1:
        raise DashboardFetchError("canonical dashboard status schema identity is invalid")
    if value.get("campaign_id") != CAMPAIGN_ID:
        raise DashboardFetchError("canonical dashboard status campaign identity is invalid")
    updated = value.get("updated_unix_ms")
    if isinstance(updated, bool) or not isinstance(updated, int) or updated <= 0:
        raise DashboardFetchError("canonical dashboard status update timestamp is invalid")
    hosts = value.get("hosts")
    if not isinstance(hosts, Mapping) or set(hosts) != ALLOWED_HOSTS:
        raise DashboardFetchError(
            "canonical dashboard status must name exactly john1, john2, john3"
        )
    if _contains_forbidden_host(value):
        raise DashboardFetchError("canonical dashboard status may not contain john4")
    return value


def _contains_forbidden_host(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "john4"
    if isinstance(value, Mapping):
        return any(
            str(key).lower() == "john4" or _contains_forbidden_host(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(_contains_forbidden_host(item) for item in value)
    return False


def build_serving_projection(payload: bytes, *, fetched_unix_ms: int) -> dict[str, Any]:
    status = validate_canonical_payload(payload)
    if isinstance(fetched_unix_ms, bool) or fetched_unix_ms <= 0:
        raise DashboardFetchError("fetch timestamp must be positive")
    projection = {
        "schema_version": 1,
        "schema_id": PROJECTION_SCHEMA_ID,
        "canonical_host": CANONICAL_HOST,
        "canonical_path": str(CANONICAL_PATH),
        "canonical_blake3": blake3.blake3(payload).hexdigest(),
        "canonical_updated_unix_ms": status["updated_unix_ms"],
        "fetched_unix_ms": fetched_unix_ms,
        "canonical_payload": payload.decode("utf-8"),
    }
    encoded = encode_projection(projection)
    if len(encoded) > MAX_PROJECTION_BYTES:
        raise DashboardFetchError(
            f"serving projection is {len(encoded)} bytes; maximum is {MAX_PROJECTION_BYTES}"
        )
    return projection


def encode_projection(projection: Mapping[str, Any]) -> bytes:
    return json.dumps(projection, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def fetch_canonical_payload() -> bytes:
    """Read the registered local source without following links or crossing devices."""
    expected_uid = os.getuid()
    home_device = os.lstat(Path.home()).st_dev
    directory_chain = (
        CANONICAL_ROOT.parent,
        CANONICAL_ROOT,
        CANONICAL_PATH.parent,
    )
    root_device: int | None = None
    for directory in directory_chain:
        try:
            metadata = os.lstat(directory)
        except OSError as error:
            raise DashboardFetchError(
                f"canonical dashboard directory {directory} is unavailable: {error}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise DashboardFetchError(
                f"canonical dashboard directory {directory} is not a real directory"
            )
        if metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise DashboardFetchError(
                f"canonical dashboard directory {directory} ownership or mode differs"
            )
        if root_device is None:
            root_device = metadata.st_dev
        if metadata.st_dev != root_device or metadata.st_dev != home_device:
            raise DashboardFetchError("canonical dashboard path crossed a filesystem boundary")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(CANONICAL_PATH, flags)
    except OSError as error:
        raise DashboardFetchError(f"canonical dashboard status is unavailable: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DashboardFetchError("canonical dashboard status is not a regular file")
        if before.st_uid != expected_uid or stat.S_IMODE(before.st_mode) & 0o077:
            raise DashboardFetchError("canonical dashboard status ownership or mode differs")
        if before.st_dev != root_device:
            raise DashboardFetchError("canonical dashboard status crossed a filesystem boundary")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            payload = handle.read(MAX_CANONICAL_BYTES + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DashboardFetchError("canonical dashboard status changed while being read")
    finally:
        os.close(descriptor)
    if len(payload) > MAX_CANONICAL_BYTES:
        raise DashboardFetchError(f"canonical dashboard status exceeds {MAX_CANONICAL_BYTES} bytes")
    validate_canonical_payload(payload)
    return payload


def write_projection(path: Path, projection: Mapping[str, Any]) -> int:
    encoded = encode_projection(projection)
    if len(encoded) > MAX_PROJECTION_BYTES:
        raise DashboardFetchError(
            f"serving projection is {len(encoded)} bytes; maximum is {MAX_PROJECTION_BYTES}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            os.fsync(directory)
        finally:
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()
            fcntl.flock(directory, fcntl.LOCK_UN)
    finally:
        os.close(directory)
    return len(encoded)


def fetch_and_publish(path: Path = SERVING_PATH) -> dict[str, Any]:
    payload = fetch_canonical_payload()
    fetched_unix_ms = time.time_ns() // 1_000_000
    projection = build_serving_projection(payload, fetched_unix_ms=fetched_unix_ms)
    written = write_projection(path, projection)
    return {
        "path": str(path),
        "bytes": written,
        "canonical_host": CANONICAL_HOST,
        "canonical_path": str(CANONICAL_PATH),
        "canonical_blake3": projection["canonical_blake3"],
        "canonical_updated_unix_ms": projection["canonical_updated_unix_ms"],
        "fetched_unix_ms": fetched_unix_ms,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=SERVING_PATH)
    result.add_argument("--watch", action="store_true")
    result.add_argument("--interval-seconds", type=int, default=10, choices=range(5, 31))
    return result


def main() -> int:
    arguments = parser().parse_args()
    expected = SERVING_PATH.resolve(strict=False)
    if arguments.output.resolve(strict=False) != expected:
        print(
            f"R2-MAP dashboard fetch refused unregistered output {arguments.output}; "
            f"expected {SERVING_PATH}",
            file=sys.stderr,
        )
        return 2
    while True:
        try:
            result = fetch_and_publish(arguments.output)
        except DashboardFetchError as error:
            print(f"R2-MAP dashboard fetch failed: {error}", file=sys.stderr, flush=True)
            if not arguments.watch:
                return 2
        else:
            print(json.dumps(result, sort_keys=True), flush=True)
            if not arguments.watch:
                return 0
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
