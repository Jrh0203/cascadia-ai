#!/usr/bin/env python3
"""Launch one John1 R2-MAP MLX tool with zero local filesystem writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_local_write_guard import (  # noqa: E402
    ATTESTATION_ENVIRONMENT,
    JOHN1_MLX_INTERPRETER,
    PROFILE_ENVIRONMENT,
    SANDBOX_PROFILE,
    SANDBOX_PROFILE_SHA256,
)

SCHEMA = "cascadia.r2-map.john1-local-write-attestation.v1"
MAX_STDOUT_BYTES = 8 << 20
MAX_STDERR_BYTES = 256 << 10
ALLOWED_TOOLS = {
    "packing-sweep": REPOSITORY / "tools/r2_map_john1_packing_sweep.py",
    "train": REPOSITORY / "tools/r2_map_john1_train.py",
}
PUBLISHER = REPOSITORY / "tools/r2_map_john1_publish_write_attestation.py"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _stat_record(path: Path, root: Path) -> bytes:
    details = path.lstat()
    relative = "." if path == root else path.relative_to(root).as_posix()
    value = [
        relative,
        stat.S_IFMT(details.st_mode),
        stat.S_IMODE(details.st_mode),
        details.st_dev,
        details.st_ino,
        details.st_uid,
        details.st_gid,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    ]
    if stat.S_ISLNK(details.st_mode):
        value.append(os.readlink(path))
    return _canonical_json(value) + b"\n"


def _snapshot_root(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    entries = 0
    apparent_bytes = 0
    try:
        root_details = path.lstat()
    except FileNotFoundError:
        digest.update(b"absent\n")
        return {"path": str(path), "state": "absent", "entries": 0, "sha256": digest.hexdigest()}
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        digest.update(_stat_record(path, path))
        return {
            "path": str(path),
            "state": "present",
            "entries": 1,
            "apparent_bytes": (
                root_details.st_size if stat.S_ISREG(root_details.st_mode) else 0
            ),
            "sha256": digest.hexdigest(),
        }
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        files.sort()
        for candidate in [current_path, *(current_path / name for name in directories + files)]:
            record = _stat_record(candidate, path)
            digest.update(record)
            details = candidate.lstat()
            entries += 1
            if stat.S_ISREG(details.st_mode):
                apparent_bytes += details.st_size
        directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
    return {
        "path": str(path),
        "state": "present",
        "entries": entries,
        "apparent_bytes": apparent_bytes,
        "sha256": digest.hexdigest(),
    }
def _snapshot_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    lexical = {
        os.path.abspath(os.fspath(path)): Path(os.path.abspath(os.fspath(path)))
        for path in paths
    }
    unique = sorted(lexical.values(), key=str)
    return [_snapshot_root(path) for path in unique]


def _prefixed_paths(parent: Path, prefixes: tuple[str, ...]) -> tuple[Path, ...]:
    if not parent.is_dir():
        return ()
    return tuple(
        sorted(
            (
                Path(entry.path)
                for entry in os.scandir(parent)
                if any(entry.name.startswith(prefix) for prefix in prefixes)
            ),
            key=str,
        )
    )


def _snapshot_scope(run_id: str, tool: str) -> tuple[Path, ...]:
    home = Path.home()
    return (
        Path(f"/private/tmp/r2-map-{run_id}"),
        Path(f"/private/var/empty/r2-map-{run_id}"),
        Path(f"/private/var/empty/r2-map-sweep-{run_id}"),
        home / ".ssh",
        home / ".mlx",
        home / ".cache/mlx",
        home / ".python_history",
        home / "Library/Caches/mlx",
        home / "Library/Caches/com.apple.Metal",
        home / "Library/Logs" / f"r2-map-{run_id}-{tool}",
        REPOSITORY,
        *_prefixed_paths(
            Path("/private/tmp"),
            ("r2-map-", "cascadia-r2-map-runtime-"),
        ),
        *_prefixed_paths(
            Path("/private/var/empty"),
            ("r2-map-", "cascadia-r2-map-runtime-"),
        ),
    )


def _bounded_child(
    argv: list[str],
    *,
    environment: dict[str, str],
    stdin_payload: bytes | None = None,
) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(
        argv,
        cwd=REPOSITORY,
        env=environment,
        stdin=subprocess.PIPE if stdin_payload is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("sandbox child pipes were not created")
    if stdin_payload is not None:
        assert process.stdin is not None
        process.stdin.write(stdin_payload)
        process.stdin.close()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, ("stdout", MAX_STDOUT_BYTES))
    selector.register(process.stderr, selectors.EVENT_READ, ("stderr", MAX_STDERR_BYTES))
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    while selector.get_map():
        for key, _ in selector.select(timeout=1.0):
            stream_name, limit = key.data
            chunk = os.read(key.fileobj.fileno(), 1 << 15)
            if not chunk:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            buffers[stream_name].extend(chunk)
            if len(buffers[stream_name]) > limit:
                process.kill()
                process.wait()
                raise RuntimeError(f"sandbox child {stream_name} exceeded its in-memory bound")
    return process.wait(), bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _environment(attestation_relative: str) -> dict[str, str]:
    return {
        "HOME": str(Path.home()),
        "TMPDIR": "/private/var/empty",
        "XDG_CACHE_HOME": "/private/var/empty",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(PYTHON_ROOT),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "LANG": "C",
        PROFILE_ENVIRONMENT: SANDBOX_PROFILE_SHA256,
        ATTESTATION_ENVIRONMENT: attestation_relative,
    }


def _argument_value(arguments: list[str], name: str) -> str:
    if arguments.count(name) != 1:
        raise ValueError(f"required child argument must appear exactly once: {name}")
    try:
        index = arguments.index(name)
        value = arguments[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"required child argument is absent: {name}") from error
    if not value or value.startswith("-"):
        raise ValueError(f"required child argument is invalid: {name}")
    return value


def _attestation_relative(tool: str, run_id: str) -> str:
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not run_id or any(character not in safe for character in run_id):
        raise ValueError("run id is unsafe")
    if tool == "packing-sweep":
        return f"reports/w2-w3/{run_id}/local-write-attestation.json"
    return f"runs/{run_id}/local-write-attestation.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("tool", choices=sorted(ALLOWED_TOOLS))
    result.add_argument("arguments", nargs=argparse.REMAINDER)
    return result


def main() -> int:
    arguments = parser().parse_args()
    child_arguments = list(arguments.arguments)
    if child_arguments[:1] == ["--"]:
        child_arguments.pop(0)
    run_id = _argument_value(child_arguments, "--run-id")
    attestation_relative = _attestation_relative(arguments.tool, run_id)
    if socket.gethostname() != "Johns-Mac-mini.local" or os.getuid() != 501:
        raise SystemExit("zero-write launcher is authorized only on John1")
    resolved_repository_python = (REPOSITORY / ".venv/bin/python").resolve(strict=True)
    expected_python = Path(JOHN1_MLX_INTERPRETER)
    if (
        resolved_repository_python != expected_python
        or Path(sys.executable).resolve(strict=True) != expected_python
    ):
        raise SystemExit("zero-write launcher requires the repository MLX interpreter")
    for path in (SANDBOX_EXEC, ALLOWED_TOOLS[arguments.tool], PUBLISHER):
        details = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise SystemExit(f"required zero-write executable/source is unsafe: {path}")
    environment = _environment(attestation_relative)
    before = _snapshot_paths(_snapshot_scope(run_id, arguments.tool))
    started_ns = time.time_ns()
    child_argv = [
        str(SANDBOX_EXEC),
        "-p",
        SANDBOX_PROFILE,
        str(expected_python),
        "-B",
        str(ALLOWED_TOOLS[arguments.tool]),
        *child_arguments,
    ]
    return_code, stdout, stderr = _bounded_child(child_argv, environment=environment)
    after = _snapshot_paths(_snapshot_scope(run_id, arguments.tool))
    if before != after:
        raise SystemExit("John1 scoped filesystem snapshot changed during sandboxed work")
    if return_code != 0:
        sys.stderr.buffer.write(stderr[-MAX_STDERR_BYTES:])
        raise SystemExit(return_code)
    try:
        main_receipt = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("sandboxed tool did not return one bounded JSON receipt") from error
    if (
        not isinstance(main_receipt, dict)
        or main_receipt.get("local_write_attestation_relative") != attestation_relative
    ):
        raise SystemExit("sandboxed tool receipt does not bind its local-write attestation")
    attestation: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": SCHEMA,
        "run_id": run_id,
        "tool": arguments.tool,
        "profile_sha256": SANDBOX_PROFILE_SHA256,
        "sandbox_argv": child_argv,
        "sandbox_argv_sha256": hashlib.sha256(_canonical_json(child_argv)).hexdigest(),
        "main_receipt": main_receipt,
        "main_stdout_bytes": len(stdout),
        "main_stderr_bytes": len(stderr),
        "snapshot_contract": "lstat-tree-metadata-no-follow-v1",
        "snapshot_scope_excludes_legacy_ssd": True,
        "before": before,
        "after": after,
        "unchanged": True,
        "started_unix_ns": started_ns,
        "completed_unix_ns": time.time_ns(),
    }
    attestation["attestation_sha256"] = hashlib.sha256(_canonical_json(attestation)).hexdigest()
    publisher_argv = [
        str(SANDBOX_EXEC),
        "-p",
        SANDBOX_PROFILE,
        str(expected_python),
        "-B",
        str(PUBLISHER),
        "--run-id",
        run_id,
        "--relative",
        attestation_relative,
    ]
    publisher_code, publisher_stdout, publisher_stderr = _bounded_child(
        publisher_argv,
        environment=environment,
        stdin_payload=_canonical_json(attestation),
    )
    if publisher_code != 0:
        sys.stderr.buffer.write(publisher_stderr[-MAX_STDERR_BYTES:])
        raise SystemExit(publisher_code)
    publication = json.loads(publisher_stdout)
    sys.stdout.buffer.write(
        _canonical_json(
            {
                "schema_version": 1,
                "schema_id": "cascadia.r2-map.john1-zero-write-launch.v1",
                "status": "complete",
                "main_receipt": main_receipt,
                "attestation_relative": attestation_relative,
                "attestation_sha256": attestation["attestation_sha256"],
                "attestation_publication": publication,
            }
        )
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
