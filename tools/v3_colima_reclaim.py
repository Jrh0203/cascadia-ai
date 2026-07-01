#!/usr/bin/env python3
"""Reclaim completed V3 worker inputs retained by macOS VirtioFS.

Bacalhau removes each completed job's materialized model inputs, but the
Virtualization.framework VirtioFS process can retain deleted file descriptors
until the dedicated Colima VM exits. Promotion increments stage many copies of
large model bundles, so trimming the guest filesystem alone is insufficient.
This module performs a bounded, evidence-gated restart only after every item in
an increment is terminal and its artifacts have been reconciled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

COLIMA = Path("/opt/homebrew/bin/colima")
DOCKER = Path("/opt/homebrew/bin/docker")
COLIMA_HOME = Path("/Users/johnherrick/.local/share/cascadia-r2/colima")
PROFILE = "cascadia-r2"
DOCKER_SOCKET = COLIMA_HOME / PROFILE / "docker.sock"
MIN_FREE_BYTES = 50 * 1024**3
CONTROL_SERVICES = {"cascadia-minio", "cascadia-registry"}
REMOTE_WORKERS = ("john2", "john3")
REMOTE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
REMOTE_COLIMA = "/opt/homebrew/bin/colima"
REMOTE_DOCKER = "/opt/homebrew/bin/docker"
REMOTE_FREE = re.compile(r"^free_kib=(\d+)$", re.MULTILINE)


class ColimaReclaimError(ValueError):
    """The worker VM was not safe to restart or did not recover cleanly."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ColimaReclaimError(f"{path} must contain a JSON object")
    return value


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_completion(completion: Path) -> dict[str, Any]:
    evidence = _read(completion)
    work_items = evidence.get("work_items")
    if (
        evidence.get("passed") is not True
        or not isinstance(work_items, int)
        or isinstance(work_items, bool)
        or work_items <= 0
        or evidence.get("succeeded") != work_items
    ):
        raise ColimaReclaimError("promotion increment is not fully reconciled")
    return evidence


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value["COLIMA_HOME"] = str(COLIMA_HOME)
    value["DOCKER_HOST"] = f"unix://{DOCKER_SOCKET}"
    # LaunchAgents receive a minimal PATH. Colima invokes Docker helpers while
    # starting the profile, so binding the top-level binaries by absolute path
    # is not sufficient on its own.
    required = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")
    existing = tuple(part for part in value.get("PATH", "").split(":") if part)
    value["PATH"] = ":".join(dict.fromkeys((*required, *existing)))
    return value


def _invoke(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_environment(),
    )


def _running_containers(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> set[str]:
    completed = _invoke(
        runner,
        [str(DOCKER), "ps", "--format", "{{.Names}}"],
        timeout=30,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def _trim_worker_disk(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    sleeper: Callable[[float], None],
) -> subprocess.CompletedProcess[str]:
    error: BaseException | None = None
    command = [
        str(COLIMA),
        "ssh",
        "-p",
        PROFILE,
        "--",
        "sudo",
        "fstrim",
        "-av",
    ]
    for attempt in range(3):
        try:
            return _invoke(runner, command)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as caught:
            error = caught
            if attempt < 2:
                sleeper(2.0)
    raise ColimaReclaimError("guest trim failed after three attempts") from error


def reclaim_completed_increment(
    completion: Path,
    receipt: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Restart the dedicated VM after a fully reconciled promotion increment."""

    if receipt.is_file():
        existing = _read(receipt)
        if existing.get("passed") is True:
            return existing
        raise ColimaReclaimError(f"existing reclaim receipt is not passing: {receipt}")

    evidence = _validate_completion(completion)
    work_items = evidence["work_items"]
    if not COLIMA.is_file():
        raise ColimaReclaimError(f"Colima binary is missing: {COLIMA}")
    if not DOCKER.is_file():
        raise ColimaReclaimError(f"Docker binary is missing: {DOCKER}")

    running_before = _running_containers(runner)
    unexpected = running_before - CONTROL_SERVICES
    if unexpected:
        raise ColimaReclaimError(
            "worker VM still has non-control containers: " + ", ".join(sorted(unexpected))
        )
    if running_before != CONTROL_SERVICES:
        raise ColimaReclaimError("worker VM control services are not both running")

    free_before = disk_usage(completion.parent).free
    trim = _trim_worker_disk(runner, sleeper)
    _invoke(runner, [str(COLIMA), "stop", "-p", PROFILE], timeout=180)
    free_after_stop = disk_usage(completion.parent).free
    _invoke(runner, [str(COLIMA), "start", "-p", PROFILE], timeout=300)

    running_after: set[str] | None = None
    for _ in range(60):
        try:
            running_after = _running_containers(runner)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            running_after = None
        if running_after == CONTROL_SERVICES:
            break
        sleeper(1.0)
    if running_after != CONTROL_SERVICES:
        raise ColimaReclaimError("worker VM control services did not recover")

    free_after = disk_usage(completion.parent).free
    if free_after < MIN_FREE_BYTES:
        raise ColimaReclaimError(
            f"worker reclaim left {free_after} bytes free; need {MIN_FREE_BYTES}"
        )
    value = {
        "schema_id": "cascadia-v3-colima-worker-reclaim-v1",
        "passed": True,
        "completion": str(completion.resolve()),
        "work_items": work_items,
        "free_bytes_before": free_before,
        "free_bytes_after_stop": free_after_stop,
        "free_bytes_after": free_after,
        "reclaimed_bytes": max(0, free_after - free_before),
        "minimum_free_bytes": MIN_FREE_BYTES,
        "trim_output": trim.stdout.strip(),
        "running_containers_before": sorted(running_before),
        "running_containers_after": sorted(running_after),
        "recovery_guarantee": "all-increment-items-terminal-before-dedicated-vm-restart",
    }
    _write_atomic(receipt, value)
    return value


def _remote(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    host: str,
    script: str,
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return runner(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            f"export PATH={REMOTE_PATH}; {script}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _remote_free_kib(
    runner: Callable[..., subprocess.CompletedProcess[str]], host: str
) -> int:
    completed = _remote(
        runner,
        host,
        "printf 'free_kib=%s\\n' \"$(df -k \"$HOME\" | awk 'NR==2 {print $4}')\"",
        timeout=30,
    )
    match = REMOTE_FREE.search(completed.stdout)
    if match is None:
        raise ColimaReclaimError(f"{host} did not report parseable free space")
    return int(match.group(1))


def _remote_running_containers(
    runner: Callable[..., subprocess.CompletedProcess[str]], host: str
) -> list[str]:
    completed = _remote(
        runner,
        host,
        f"{REMOTE_DOCKER} ps --format '{{{{.Names}}}}'",
        timeout=30,
    )
    return sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())


def reclaim_remote_workers(
    completion: Path,
    receipt: Path,
    *,
    hosts: tuple[str, ...] = REMOTE_WORKERS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    fabric_probe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Release deleted VirtioFS inputs on idle John2/John3 worker VMs.

    The completion receipt proves every scheduler item is terminal and locally
    reconciled. Workers are restarted one at a time, only with no running
    Docker containers, and the full scheduler fabric is checked after each
    restart. This closes the remote half of the same lifecycle already enforced
    for John1's dedicated Colima profile.
    """

    if receipt.is_file():
        existing = _read(receipt)
        if existing.get("passed") is True:
            return existing
        raise ColimaReclaimError(f"existing remote reclaim receipt is not passing: {receipt}")
    evidence = _validate_completion(completion)
    if not hosts or len(set(hosts)) != len(hosts):
        raise ColimaReclaimError("remote worker list must be non-empty and unique")

    if fabric_probe is None:
        from v3_phase2_pipeline import _client, _validate_fabric

        def fabric_probe() -> None:
            client = _client(
                completion.parent / ".fleet-reclaim-client",
                completion.parent / ".fleet-reclaim-artifacts",
            )
            _validate_fabric(client.api.nodes())

    results = []
    for host in hosts:
        running_before: list[str] = []
        for _ in range(60):
            running_before = _remote_running_containers(runner, host)
            if not running_before:
                break
            sleeper(1.0)
        if running_before:
            raise ColimaReclaimError(
                f"{host} still has running containers: {', '.join(running_before)}"
            )
        free_before = _remote_free_kib(runner, host) * 1024
        trim = _remote(
            runner,
            host,
            f"{REMOTE_COLIMA} ssh -- sudo fstrim -av",
            timeout=180,
        )
        _remote(runner, host, f"{REMOTE_COLIMA} stop", timeout=180)
        free_after_stop = _remote_free_kib(runner, host) * 1024
        _remote(runner, host, f"{REMOTE_COLIMA} start", timeout=300)
        running_after: list[str] | None = None
        for _ in range(60):
            try:
                _remote(runner, host, f"{REMOTE_DOCKER} info >/dev/null", timeout=30)
                running_after = _remote_running_containers(runner, host)
                break
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                sleeper(1.0)
        if running_after is None:
            raise ColimaReclaimError(f"{host} Docker did not recover")
        if running_after:
            raise ColimaReclaimError(
                f"{host} unexpectedly started containers: {', '.join(running_after)}"
            )
        free_after = _remote_free_kib(runner, host) * 1024
        if free_after < MIN_FREE_BYTES:
            raise ColimaReclaimError(
                f"{host} reclaim left {free_after} bytes free; need {MIN_FREE_BYTES}"
            )
        fabric_error: Exception | None = None
        for _ in range(60):
            try:
                fabric_probe()
                fabric_error = None
                break
            except Exception as error:  # scheduler/client transports have several exception types
                fabric_error = error
                sleeper(1.0)
        if fabric_error is not None:
            raise ColimaReclaimError(
                f"scheduler fabric did not recover after restarting {host}"
            ) from fabric_error
        results.append(
            {
                "host": host,
                "free_bytes_before": free_before,
                "free_bytes_after_stop": free_after_stop,
                "free_bytes_after": free_after,
                "reclaimed_bytes": max(0, free_after - free_before),
                "trim_output": trim.stdout.strip(),
                "running_containers_before": running_before,
                "running_containers_after": running_after,
                "fabric_verified_after_restart": True,
            }
        )

    value = {
        "schema_id": "cascadia-v3-remote-worker-reclaim-v1",
        "passed": True,
        "completion": str(completion.resolve()),
        "work_items": evidence["work_items"],
        "minimum_free_bytes": MIN_FREE_BYTES,
        "workers": results,
        "reclaimed_bytes": sum(int(item["reclaimed_bytes"]) for item in results),
        "recovery_guarantee": "terminal-items-no-live-containers-serial-restart-fabric-verified",
    }
    _write_atomic(receipt, value)
    return value
