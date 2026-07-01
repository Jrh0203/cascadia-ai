#!/usr/bin/env python3
"""One-shot capture wrapper for a signed, read-only D0 diagnostic command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class CaptureError(RuntimeError):
    pass


AUTHORIZED_SCHEMAS = {
    "cascadia.r2-map.d0-path-chain-inventory-authorization.v1",
    "cascadia.r2-map.d0-post-failure-network-authorization.v1",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_sha256(value: dict[str, Any], field: str) -> str:
    return sha256_bytes(canonical_json({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, maximum: int) -> bytes:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
        or observed.st_size > maximum
    ):
        raise CaptureError(f"unsafe capture input: {path}")
    value = path.read_bytes()
    if len(value) != observed.st_size:
        raise CaptureError(f"capture input changed while reading: {path}")
    return value


def write_new(path: Path, value: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def swap_sample() -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "vm.swapusage"],
        capture_output=True,
        check=False,
        timeout=10,
    )
    output = completed.stdout.decode("ascii", "replace")
    return {
        "unix_ms": time.time_ns() // 1_000_000,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "zero": completed.returncode == 0
        and "total = 0.00M" in output
        and "used = 0.00M" in output
        and "free = 0.00M" in output,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    authorization_bytes = read_regular(args.authorization, 1024 * 1024)
    authorization = json.loads(authorization_bytes)
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_id") not in AUTHORIZED_SCHEMAS
        or authorization.get("authorization_sha256")
        != document_sha256(authorization, "authorization_sha256")
        or authorization.get("status") != "authorized-once"
        or authorization.get("capture_runner_sha256")
        != sha256_bytes(read_regular(Path(__file__).resolve(), 1024 * 1024))
        or authorization.get("probe_sha256") != sha256_bytes(read_regular(args.probe, 1024 * 1024))
        or authorization.get("public_key_sha256")
        != sha256_bytes(read_regular(args.public_key, 4096))
        or authorization.get("remote_output_root") != str(args.output_root)
        or int(authorization.get("expires_unix_ms", 0)) <= time.time_ns() // 1_000_000
    ):
        raise CaptureError("capture authorization identity or expiry differs")
    expected_command = [
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        str(args.probe),
        "--authorization",
        str(args.authorization),
    ]
    if authorization.get("command") != expected_command:
        raise CaptureError("capture command differs")
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.time_ns() // 1_000_000
    claim = {
        "authorization_sha256": authorization["authorization_sha256"],
        "command": expected_command,
        "started_unix_ms": started,
        "status": "claimed-once",
    }
    write_new(args.output_root / "claim.json", canonical_json(claim))
    before = swap_sample()
    if before["zero"] is not True:
        raise CaptureError("capture requires zero swap before launch")
    environment = {
        "HOME": "/Users/john2",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TZ": "UTC",
    }
    if authorization.get("schema_id") == "cascadia.r2-map.d0-post-failure-network-authorization.v1":
        environment.update(
            {
                "COLIMA_CACHE_HOME": "/Users/john2/Library/Caches/cascadia-r2/colima",
                "COLIMA_HOME": "/Users/john2/.local/share/cascadia-r2/colima",
                "COLIMA_PROFILE": "cascadia-r2",
                "DOCKER_CONFIG": "/Users/john2/.config/cascadia-r2/docker",
                "DOCKER_HOST": "unix:///Users/john2/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock",
                "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            }
        )
    process = subprocess.Popen(
        expected_command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    samples: list[dict[str, Any]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.25):
            samples.append(swap_sample())

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    stdout, stderr = process.communicate()
    stop.set()
    thread.join(timeout=2)
    after = swap_sample()
    swap_zero = before["zero"] and after["zero"] and all(item["zero"] for item in samples)
    write_new(args.output_root / "stdout.json", stdout)
    write_new(args.output_root / "stderr.bin", stderr)
    finished = time.time_ns() // 1_000_000
    state: dict[str, Any] = {
        **claim,
        "pid": process.pid,
        "finished_unix_ms": finished,
        "returncode": process.returncode,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "swap_before": before,
        "swap_after": after,
        "swap_samples": samples,
        "swap_zero_throughout": swap_zero,
    }
    try:
        result = json.loads(stdout)
        parse_error = None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        result = None
        parse_error = type(error).__name__
    passed = (
        process.returncode == 0
        and not stderr
        and isinstance(result, dict)
        and result.get("status") == "pass-diagnostic"
        and result.get("authorization_sha256") == authorization["authorization_sha256"]
        and result.get("result_sha256") == document_sha256(result, "result_sha256")
        and swap_zero is True
    )
    if not passed:
        failure = {
            "authorization_sha256": authorization["authorization_sha256"],
            "finished_unix_ms": finished,
            "parse_error": parse_error,
            "returncode": process.returncode,
            "stderr_sha256": sha256_bytes(stderr),
            "stderr_size": len(stderr),
            "stdout_sha256": sha256_bytes(stdout),
            "stdout_size": len(stdout),
            "swap_zero_throughout": swap_zero,
            "status": "captured-fail",
        }
        failure["failure_sha256"] = document_sha256(failure, "failure_sha256")
        state.update(failure)
        write_new(args.output_root / "failure.json", canonical_json(failure))
        write_new(args.output_root / "runner-state.json", canonical_json(state))
        raise CaptureError("read-only diagnostic did not pass; raw output persisted")
    state.update(
        {
            "result_sha256": result["result_sha256"],
            "status": "captured-pass",
            "swap_zero_throughout": True,
        }
    )
    write_new(args.output_root / "runner-state.json", canonical_json(state))
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    try:
        print(json.dumps(run(parser.parse_args()), sort_keys=True, separators=(",", ":")))
        return 0
    except (CaptureError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"r2-d0-readonly-capture: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
