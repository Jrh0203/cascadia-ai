#!/usr/bin/env python3
"""Run one Codex turn and verify both anonymous-pipe John2 stream sinks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

sys.dont_write_bytecode = True
CHUNK_BYTES = 1 << 20
MAX_PROMPT_BYTES = 64 * 1024
MAX_REMOTE_STREAM_BYTES = 1 << 30
SINK_FAILURE_EXIT = 74


class HeadlessTurnError(RuntimeError):
    """A local child or authenticated John2 sink failed."""


@dataclass
class PumpResult:
    bytes_written: int = 0
    sha256: str = hashlib.sha256(b"").hexdigest()
    error: BaseException | None = None


@dataclass
class CaptureResult:
    payload: bytes = b""
    overflow: bool = False
    error: BaseException | None = None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--codex-bin", type=Path, required=True)
    result.add_argument("--remote-python", type=Path, required=True)
    result.add_argument("--remote-tool", type=Path, required=True)
    result.add_argument("--python-root", type=Path, required=True)
    result.add_argument("--repository", type=Path, required=True)
    result.add_argument("--session-id", required=True)
    result.add_argument("--prompt-path", type=Path, required=True)
    result.add_argument("--events-relative", required=True)
    result.add_argument("--stderr-relative", required=True)
    result.add_argument("--events-max-bytes", type=int, required=True)
    result.add_argument("--stderr-max-bytes", type=int, required=True)
    result.add_argument("--heartbeat-pid", type=int, required=True)
    return result


def _sink_command(arguments: argparse.Namespace, relative: str, maximum: int) -> list[str]:
    return [
        str(arguments.remote_python),
        str(arguments.remote_tool),
        "put-stream",
        "--relative",
        relative,
        "--max-bytes",
        str(maximum),
        "--expected-current",
        "absent",
    ]


def _pump(source: BinaryIO, destination: BinaryIO, result: PumpResult) -> None:
    hasher = hashlib.sha256()
    try:
        while True:
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = destination.write(view)
                if written is None or written <= 0:
                    raise HeadlessTurnError("anonymous-pipe sink write made no progress")
                view = view[written:]
            destination.flush()
            hasher.update(chunk)
            result.bytes_written += len(chunk)
    except BaseException as error:  # retained for the parent thread
        result.error = error
    finally:
        try:
            destination.close()
        except BaseException as error:
            if result.error is None:
                result.error = error
        try:
            source.close()
        except BaseException as error:
            if result.error is None:
                result.error = error
        result.sha256 = hasher.hexdigest()


def _capture(source: BinaryIO, result: CaptureResult, *, maximum: int) -> None:
    """Drain a pipe without allowing its diagnostic payload to grow unbounded."""
    retained = bytearray()
    try:
        while True:
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            remaining = maximum + 1 - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(retained) > maximum or len(chunk) > remaining:
                result.overflow = True
        result.payload = bytes(retained[:maximum])
    except BaseException as error:  # retained for the parent thread
        result.error = error
    finally:
        try:
            source.close()
        except BaseException as error:
            if result.error is None:
                result.error = error


def _validate_sink_receipt(
    payload: bytes,
    *,
    relative: str,
    maximum_bytes: int,
    bytes_written: int,
    stream_sha256: str,
) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeadlessTurnError("remote stream sink returned invalid JSON") from error
    size = value.get("size") if isinstance(value, dict) else None
    receipt_relative = value.get("storage_receipt_relative") if isinstance(value, dict) else None
    semantic = hashlib.sha256(
        json.dumps(
            {
                "operation": "put-stream",
                "arguments": {
                    "relative": relative,
                    "max_bytes": maximum_bytes,
                    "expected_current": "absent",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    expected_receipt = f"control/receipts/req-put-stream-{semantic[:32]}.json"
    if (
        not isinstance(value, dict)
        or value.get("relative") != relative
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size != bytes_written
        or size > maximum_bytes
        or value.get("mode") != "0o400"
        or value.get("sha256") != stream_sha256
        or not _sha256(value.get("sha256"))
        or receipt_relative != expected_receipt
        or not _sha256(value.get("storage_receipt_sha256"))
    ):
        raise HeadlessTurnError("remote stream sink receipt identity differs")
    return value


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            group_exists = False
        except PermissionError:
            group_exists = True
        else:
            group_exists = True
        if group_exists:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                group_exists = False
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            group_exists = False
        except PermissionError:
            group_exists = True
        else:
            group_exists = True
        if group_exists:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                group_exists = False
    elif process.poll() is None:
        process.kill()
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise HeadlessTurnError("child process group resisted bounded reap") from error
    if isinstance(pid, int) and pid > 0:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                pass
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        raise HeadlessTurnError("child process descendants survived bounded reap")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_turn(arguments: argparse.Namespace) -> dict[str, Any]:
    for label, path in (
        ("Codex executable", arguments.codex_bin),
        ("remote Python", arguments.remote_python),
        ("remote tool", arguments.remote_tool),
        ("prompt", arguments.prompt_path),
    ):
        if not path.is_file():
            raise HeadlessTurnError(f"{label} is absent: {path}")
    for label, path in (
        ("Codex executable", arguments.codex_bin),
        ("remote Python", arguments.remote_python),
    ):
        if not os.access(path, os.X_OK):
            raise HeadlessTurnError(f"{label} is not owner-executable")
    if not arguments.repository.is_dir() or not arguments.python_root.is_dir():
        raise HeadlessTurnError("repository or Python source root is absent")
    if not arguments.session_id:
        raise HeadlessTurnError("Codex session identity is empty")
    if not all(
        1 <= value <= MAX_REMOTE_STREAM_BYTES
        for value in (arguments.events_max_bytes, arguments.stderr_max_bytes)
    ):
        raise HeadlessTurnError("stream bounds must be within 1 byte and 1 GiB")
    if arguments.heartbeat_pid <= 0 or not _pid_is_alive(arguments.heartbeat_pid):
        raise HeadlessTurnError("lock heartbeat is not alive before the turn")
    prompt_payload = arguments.prompt_path.read_bytes()
    if len(prompt_payload) > MAX_PROMPT_BYTES:
        raise HeadlessTurnError("headless turn prompt exceeds 64 KiB")
    prompt = prompt_payload.decode()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(arguments.python_root),
        }
    )
    sink_specs = (
        (arguments.events_relative, arguments.events_max_bytes),
        (arguments.stderr_relative, arguments.stderr_max_bytes),
    )
    if sink_specs[0][0] == sink_specs[1][0]:
        raise HeadlessTurnError("stdout and stderr remote paths must differ")
    sinks: list[subprocess.Popen[bytes]] = []
    codex: subprocess.Popen[bytes] | None = None
    pump_threads: list[threading.Thread] = []
    capture_threads: list[threading.Thread] = []
    pump_results = [PumpResult(), PumpResult()]
    stdout_captures: list[CaptureResult] = []
    stderr_captures: list[CaptureResult] = []
    heartbeat_failed = False
    try:
        for relative, maximum in sink_specs:
            sink = subprocess.Popen(
                _sink_command(arguments, relative, maximum),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                close_fds=True,
                start_new_session=True,
            )
            sinks.append(sink)
            assert sink.stdout is not None and sink.stderr is not None
            stdout_capture = CaptureResult()
            stderr_capture = CaptureResult()
            stdout_captures.append(stdout_capture)
            stderr_captures.append(stderr_capture)
            for source, capture, maximum_capture in (
                (sink.stdout, stdout_capture, 1 << 20),
                (sink.stderr, stderr_capture, 4096),
            ):
                thread = threading.Thread(
                    target=_capture,
                    args=(source, capture),
                    kwargs={"maximum": maximum_capture},
                    daemon=False,
                )
                thread.start()
                capture_threads.append(thread)
        codex = subprocess.Popen(
            [
                str(arguments.codex_bin),
                "exec",
                "--json",
                "--color",
                "never",
                "--sandbox",
                "danger-full-access",
                "--cd",
                str(arguments.repository),
                "--config",
                'approval_policy="never"',
                "resume",
                arguments.session_id,
                prompt,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
        )
        assert codex.stdout is not None and codex.stderr is not None
        for source, sink, result in zip(
            (codex.stdout, codex.stderr), sinks, pump_results, strict=True
        ):
            assert sink.stdin is not None
            thread = threading.Thread(
                target=_pump,
                args=(source, sink.stdin, result),
                daemon=False,
            )
            thread.start()
            pump_threads.append(thread)
        while codex.poll() is None:
            if not _pid_is_alive(arguments.heartbeat_pid):
                heartbeat_failed = True
                _terminate(codex)
                break
            if any(sink.poll() is not None for sink in sinks) or any(
                result.error is not None for result in pump_results
            ):
                _terminate(codex)
                break
            time.sleep(0.25)
        codex_exit = codex.wait()
        for thread in pump_threads:
            thread.join(timeout=30)
        if any(thread.is_alive() for thread in pump_threads):
            for sink in sinks:
                _terminate(sink)
            raise HeadlessTurnError("stream pump thread resisted bounded join")
        sink_results = []
        failures = []
        sink_exits = [sink.wait(timeout=30) for sink in sinks]
        for thread in capture_threads:
            thread.join(timeout=30)
        if any(thread.is_alive() for thread in capture_threads):
            raise HeadlessTurnError("sink capture thread resisted bounded join")
        for (
            (relative, maximum),
            sink_exit,
            pump,
            stdout_capture,
            stderr_capture,
        ) in zip(
            sink_specs,
            sink_exits,
            pump_results,
            stdout_captures,
            stderr_captures,
            strict=True,
        ):
            if (
                sink_exit != 0
                or pump.error is not None
                or stdout_capture.error is not None
                or stderr_capture.error is not None
                or stdout_capture.overflow
                or stderr_capture.overflow
            ):
                failures.append(
                    {
                        "relative": relative,
                        "exit_code": sink_exit,
                        "pump_error": None if pump.error is None else type(pump.error).__name__,
                        "capture_error": next(
                            (
                                type(value.error).__name__
                                for value in (stdout_capture, stderr_capture)
                                if value.error is not None
                            ),
                            None,
                        ),
                        "capture_overflow": (stdout_capture.overflow or stderr_capture.overflow),
                        "stderr": stderr_capture.payload.decode("utf-8", errors="replace"),
                    }
                )
                continue
            try:
                receipt = _validate_sink_receipt(
                    stdout_capture.payload,
                    relative=relative,
                    maximum_bytes=maximum,
                    bytes_written=pump.bytes_written,
                    stream_sha256=pump.sha256,
                )
            except HeadlessTurnError as error:
                failures.append(
                    {
                        "relative": relative,
                        "exit_code": sink_exit,
                        "pump_error": None,
                        "stderr": str(error),
                    }
                )
            else:
                sink_results.append(
                    {
                        "relative": relative,
                        "bytes": pump.bytes_written,
                        "sha256": receipt["sha256"],
                        "storage_receipt_relative": receipt["storage_receipt_relative"],
                        "storage_receipt_sha256": receipt["storage_receipt_sha256"],
                    }
                )
        if failures:
            raise HeadlessTurnError(
                "one or more John2 stream sinks failed: " + json.dumps(failures, sort_keys=True)
            )
        if heartbeat_failed:
            raise HeadlessTurnError("remote lock heartbeat exited during the Codex turn")
        return {
            "codex_exit_code": codex_exit,
            "sinks_verified": True,
            "sinks": sink_results,
        }
    finally:
        if codex is not None:
            _terminate(codex)
        for sink in sinks:
            _terminate(sink)
        for thread in pump_threads:
            thread.join(timeout=10)
        for thread in capture_threads:
            thread.join(timeout=10)


def main() -> int:
    try:
        result = run_turn(parser().parse_args())
    except (
        HeadlessTurnError,
        OSError,
        UnicodeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return SINK_FAILURE_EXIT
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
