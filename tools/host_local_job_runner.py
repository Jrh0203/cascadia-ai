#!/usr/bin/env python3
"""Run one host-local job with durable typed lifecycle and progress state."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _progress_count(directory: Path | None, prefix: str, suffix: str) -> int | None:
    if directory is None:
        return None
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)
    )


def run(
    command: list[str],
    status_path: Path,
    progress_directory: Path | None,
    progress_prefix: str,
    progress_suffix: str,
    expected_items: int | None,
    poll_seconds: float,
    stdout_log: Path | None,
    stderr_log: Path | None,
) -> int:
    if not command:
        raise ValueError("a host-local command is required")
    if expected_items is not None and expected_items <= 0:
        raise ValueError("expected-items must be positive")
    for path in (stdout_log, stderr_log):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_log.open("ab", buffering=0) if stdout_log is not None else None
    stderr_handle = stderr_log.open("ab", buffering=0) if stderr_log is not None else None
    started = time.time()
    process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
    state: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "supervisor_pid": os.getpid(),
        "process_pid": process.pid,
        "command": command,
        "started_unix_seconds": started,
        "ended_unix_seconds": None,
        "exit_code": None,
        "stdout_log": str(stdout_log) if stdout_log is not None else None,
        "stderr_log": str(stderr_log) if stderr_log is not None else None,
        "progress": {
            "completed_items": _progress_count(
                progress_directory, progress_prefix, progress_suffix
            ),
            "expected_items": expected_items,
        },
    }
    _write_atomic(status_path, state)

    terminating_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal terminating_signal
        terminating_signal = signum
        if process.poll() is None:
            process.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        while process.poll() is None:
            state["progress"]["completed_items"] = _progress_count(
                progress_directory, progress_prefix, progress_suffix
            )
            _write_atomic(status_path, state)
            time.sleep(poll_seconds)
        exit_code = int(process.returncode)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()

    state.update(
        {
            "state": (
                "terminated"
                if terminating_signal is not None
                else "completed" if exit_code == 0 else "failed"
            ),
            "ended_unix_seconds": time.time(),
            "exit_code": exit_code,
            "terminating_signal": terminating_signal,
        }
    )
    state["progress"]["completed_items"] = _progress_count(
        progress_directory, progress_prefix, progress_suffix
    )
    _write_atomic(status_path, state)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--progress-directory", type=Path)
    parser.add_argument("--progress-prefix", default="")
    parser.add_argument("--progress-suffix", default="")
    parser.add_argument("--expected-items", type=int)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(
        run(
            command,
            args.status,
            args.progress_directory,
            args.progress_prefix,
            args.progress_suffix,
            args.expected_items,
            args.poll_seconds,
            args.stdout_log,
            args.stderr_log,
        )
    )


if __name__ == "__main__":
    main()
