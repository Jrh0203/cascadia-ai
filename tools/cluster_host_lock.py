#!/usr/bin/env python3
"""Serialize heavyweight Cascadia work per host and hold macOS awake."""

# ruff: noqa: UP045 - cluster tools must run under the macOS system Python 3.9.

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_LOCK_PATH = Path("/tmp/cascadia-v2-heavy-workload.lock")


class LockTimeout(RuntimeError):
    """Raised when a host remains occupied beyond the authorized queue wait."""


def read_owner(path: Path) -> Optional[dict[str, Any]]:
    try:
        text = path.read_text().strip()
        return json.loads(text) if text else None
    except (OSError, json.JSONDecodeError):
        return None


def acquire_lock(path: Path, wait_seconds: float, poll_seconds: float) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise LockTimeout(
                    f"host workload lock remained busy: {json.dumps(read_owner(path))}"
                ) from None
            time.sleep(poll_seconds)


def write_owner(handle: Any, owner: dict[str, Any]) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(owner, handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def release_lock(handle: Any) -> None:
    handle.seek(0)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def append_event(path: Optional[Path], event: dict[str, Any]) -> None:
    line = json.dumps(event, sort_keys=True)
    print(line, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_locked(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("run requires a command after --")
    queued_at = time.time()
    handle = acquire_lock(args.lock_path, args.wait_seconds, args.poll_seconds)
    started_at = time.time()
    owner = {
        "schema_version": 1,
        "host": socket.gethostname().split(".")[0],
        "pid": os.getpid(),
        "name": args.name,
        "command": command,
        "queued_unix_seconds": queued_at,
        "started_unix_seconds": started_at,
    }
    write_owner(handle, owner)
    append_event(
        args.event_log,
        {
            **owner,
            "event": "started",
            "queued_seconds": started_at - queued_at,
        },
    )

    wrapped = command
    if not args.no_caffeinate and shutil.which("caffeinate"):
        wrapped = ["caffeinate", "-ims", *command]
    try:
        process = subprocess.Popen(wrapped, start_new_session=True)
    except Exception:
        append_event(
            args.event_log,
            {
                **owner,
                "event": "launch-failed",
                "ended_unix_seconds": time.time(),
            },
        )
        release_lock(handle)
        raise
    previous_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            os.killpg(process.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward)
    try:
        return_code = process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        ended_at = time.time()
        append_event(
            args.event_log,
            {
                **owner,
                "event": "finished",
                "ended_unix_seconds": ended_at,
                "elapsed_seconds": ended_at - started_at,
                "return_code": process.returncode,
            },
        )
        release_lock(handle)
    return return_code


def lock_status(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return {"busy": True, "owner": read_owner(path)}
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return {"busy": False, "owner": None}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--wait-seconds", type=float, default=300.0)
    run.add_argument("--poll-seconds", type=float, default=1.0)
    run.add_argument("--event-log", type=Path)
    run.add_argument("--no-caffeinate", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)

    subparsers.add_parser("status")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.subcommand == "status":
        print(json.dumps(lock_status(args.lock_path), indent=2, sort_keys=True))
        return 0
    try:
        return run_locked(args)
    except LockTimeout as error:
        print(str(error), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
