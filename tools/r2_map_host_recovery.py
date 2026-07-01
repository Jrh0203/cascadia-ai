#!/usr/bin/env python3
"""Measure and publish John1's fail-closed post-restart recovery gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_apfs_lifecycle import (  # noqa: E402
    build_apfs_bootstrap_safety_receipt,
    build_host_safety_receipt,
)
from cascadia_mlx.r2_map_contracts import CAMPAIGN_ROOT, content_sha256  # noqa: E402

PROOF_SCHEMA = "cascadia.r2-map.host-recovery-proof.v1"
APFS_BOOTSTRAP_PROOF_SCHEMA = "cascadia.r2-map.apfs-bootstrap-proof.v2"
QUIET_CPU_PERCENT = 5.0
RECOVERY_RSS_BYTES = 256 * (1 << 20)
_SWAP = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMG])\b")
_UNIT = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


class HostRecoveryError(RuntimeError):
    """Host recovery telemetry is unavailable or malformed."""


def _syspolicyd() -> dict[str, int | float]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,%cpu=,rss=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4 and fields[3] == "/usr/libexec/syspolicyd":
            matches.append(
                {
                    "pid": int(fields[0]),
                    "cpu_percent": float(fields[1]),
                    "rss_bytes": int(fields[2]) * 1024,
                }
            )
    if len(matches) > 1:
        raise HostRecoveryError("multiple syspolicyd processes were observed")
    return matches[0] if matches else {"pid": 0, "cpu_percent": 0.0, "rss_bytes": 0}


def _swap_used_bytes() -> int:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = _SWAP.search(completed.stdout)
    if match is None:
        raise HostRecoveryError("vm.swapusage output is unrecognized")
    return round(float(match.group(1)) * _UNIT[match.group(2)])


def _memory_pressure_level() -> int:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = int(completed.stdout.strip())
    except ValueError as error:
        raise HostRecoveryError("memory pressure level output is unrecognized") from error
    if result not in {1, 2, 4}:
        raise HostRecoveryError("memory pressure level is outside normal/warn/critical")
    return result


def observe(*, window_seconds: int, interval_seconds: int) -> tuple[dict[str, Any], dict]:
    if window_seconds < 60 or interval_seconds <= 0 or window_seconds % interval_seconds:
        raise ValueError("recovery window must be >=60 seconds and divisible by interval")
    started_ms = time.time_ns() // 1_000_000
    baseline_swap = _swap_used_bytes()
    samples = []
    for index in range(window_seconds // interval_seconds + 1):
        process = _syspolicyd()
        samples.append(
            {
                "offset_seconds": index * interval_seconds,
                "observed_unix_ms": time.time_ns() // 1_000_000,
                **process,
                "system_swap_used_bytes": _swap_used_bytes(),
            }
        )
        if index * interval_seconds < window_seconds:
            time.sleep(interval_seconds)
    max_cpu = max(float(sample["cpu_percent"]) for sample in samples)
    max_rss = max(int(sample["rss_bytes"]) for sample in samples)
    max_swap = max(int(sample["system_swap_used_bytes"]) for sample in samples)
    quiet = (
        max_cpu <= QUIET_CPU_PERCENT
        and max_rss <= RECOVERY_RSS_BYTES
        and max_swap <= baseline_swap
    )
    proof = {
        "schema_version": 1,
        "schema_id": PROOF_SCHEMA,
        "host": "john1",
        "started_unix_ms": started_ms,
        "completed_unix_ms": time.time_ns() // 1_000_000,
        "window_seconds": window_seconds,
        "interval_seconds": interval_seconds,
        "cpu_quiet_threshold_percent": QUIET_CPU_PERCENT,
        "rss_recovery_threshold_bytes": RECOVERY_RSS_BYTES,
        "system_swap_baseline_bytes": baseline_swap,
        "maximum_cpu_percent": max_cpu,
        "maximum_rss_bytes": max_rss,
        "maximum_system_swap_used_bytes": max_swap,
        "system_swap_delta_bytes": max(max_swap - baseline_swap, 0),
        "quiet_window_passed": quiet,
        "samples": samples,
    }
    proof["proof_sha256"] = content_sha256(proof, hash_field="proof_sha256")
    detail = (
        f"{window_seconds}-second recovery window: max syspolicyd CPU {max_cpu:.1f}%, "
        f"max RSS {max_rss} bytes, max positive swap delta "
        f"{proof['system_swap_delta_bytes']} bytes; proof {proof['proof_sha256']}"
    )
    receipt = build_host_safety_receipt(
        status="safe" if quiet else "blocked-host-recovery",
        observed_unix_ms=int(proof["completed_unix_ms"]),
        syspolicyd_rss_bytes=max_rss,
        system_swap_baseline_bytes=baseline_swap,
        system_swap_observed_bytes=max_swap,
        quiet_window_passed=quiet,
        detail=detail,
    )
    return proof, receipt


def observe_apfs_bootstrap(
    *, window_seconds: int, interval_seconds: int
) -> tuple[dict[str, Any], dict]:
    """Measure only the bounded built-in-tool APFS bootstrap gate.

    This deliberately omits the strict CPU/256-MiB thresholds. Its receipt is
    incapable of authorizing builds or runtime execution.
    """
    if window_seconds != 60 or interval_seconds <= 0 or window_seconds % interval_seconds:
        raise ValueError("APFS bootstrap window must be exactly 60 divisible seconds")
    started_ms = time.time_ns() // 1_000_000
    baseline_swap = _swap_used_bytes()
    samples = []
    for index in range(window_seconds // interval_seconds + 1):
        process = _syspolicyd()
        samples.append(
            {
                "offset_seconds": index * interval_seconds,
                "observed_unix_ms": time.time_ns() // 1_000_000,
                **process,
                "system_swap_used_bytes": _swap_used_bytes(),
                "memory_pressure_level": _memory_pressure_level(),
                "backing_free_bytes": shutil.disk_usage(CAMPAIGN_ROOT).free,
            }
        )
        if index * interval_seconds < window_seconds:
            time.sleep(interval_seconds)
    maximum_rss = max(int(sample["rss_bytes"]) for sample in samples)
    maximum_swap = max(int(sample["system_swap_used_bytes"]) for sample in samples)
    maximum_pressure = max(int(sample["memory_pressure_level"]) for sample in samples)
    minimum_free = min(int(sample["backing_free_bytes"]) for sample in samples)
    safe = (
        maximum_rss < 4 * (1 << 30)
        and maximum_swap <= baseline_swap
        and maximum_pressure == 1
        and minimum_free >= 140 * (1 << 30)
    )
    completed_ms = time.time_ns() // 1_000_000
    proof = {
        "schema_version": 1,
        "schema_id": APFS_BOOTSTRAP_PROOF_SCHEMA,
        "host": "john1",
        "started_unix_ms": started_ms,
        "completed_unix_ms": completed_ms,
        "window_seconds": window_seconds,
        "interval_seconds": interval_seconds,
        "maximum_syspolicyd_rss_bytes": maximum_rss,
        "hard_stop_rss_bytes": 4 * (1 << 30),
        "system_swap_baseline_bytes": baseline_swap,
        "maximum_system_swap_used_bytes": maximum_swap,
        "system_swap_delta_bytes": max(maximum_swap - baseline_swap, 0),
        "maximum_memory_pressure_level": maximum_pressure,
        "required_memory_pressure_level": 1,
        "minimum_observed_backing_free_bytes": minimum_free,
        "minimum_backing_free_bytes": 140 * (1 << 30),
        "quiet_window_passed": safe,
        "runtime_authorized": False,
        "samples": samples,
    }
    proof["proof_sha256"] = content_sha256(proof, hash_field="proof_sha256")
    detail = (
        f"60-second APFS-bootstrap-only window: max syspolicyd RSS {maximum_rss}, "
        f"swap delta {proof['system_swap_delta_bytes']}, max memory pressure level "
        f"{maximum_pressure}, minimum backing free {minimum_free}; proof "
        f"{proof['proof_sha256']}; runtime remains unauthorized"
    )
    receipt = build_apfs_bootstrap_safety_receipt(
        status="apfs-bootstrap-safe" if safe else "blocked-host-recovery",
        started_unix_ms=started_ms,
        completed_unix_ms=completed_ms,
        maximum_syspolicyd_rss_bytes=maximum_rss,
        system_swap_baseline_bytes=baseline_swap,
        maximum_system_swap_used_bytes=maximum_swap,
        maximum_memory_pressure_level=maximum_pressure,
        minimum_observed_backing_free_bytes=minimum_free,
        detail=detail,
    )
    return proof, receipt


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--interval-seconds", type=int, default=5)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--mode", choices=("strict",), default="strict")
    arguments = parser.parse_args()
    proof, receipt = observe(
        window_seconds=arguments.window_seconds, interval_seconds=arguments.interval_seconds
    )
    if arguments.publish:
        raise SystemExit(
            "local host-recovery publication is disabled; stream the receipt atomically "
            "to john2 with r2_map_remote_storage"
        )
    print(json.dumps({"proof": proof, "receipt": receipt}, sort_keys=True, indent=2))
    return 0 if proof["quiet_window_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
