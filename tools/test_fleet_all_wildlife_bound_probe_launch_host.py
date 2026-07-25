from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

LAUNCHER = Path(
    "cascadiav3/scripts/fleet_all_wildlife_bound_probe_launch_host.sh"
).resolve()


def _environment(home: Path, *, tag: str = "launch_test") -> dict[str, str]:
    return {
        **os.environ,
        "HOME": str(home),
        "FLEET_TAG": tag,
        "SHARD_HOST": "john-test",
        "TASK_INDICES": "0,4",
        "WILDLIFE_VENV": ".venv",
        "TIME_LIMIT": "1",
        "TOTAL_TIME_LIMIT": "2",
        "SOLVER_WORKERS": "1",
        "HEARTBEAT_INTERVAL": "1",
    }


def _fake_worker(home: Path) -> Path:
    worker = (
        home
        / "cascadia"
        / "cascadiav3"
        / "scripts"
        / "fleet_all_wildlife_bound_probe_worker.sh"
    )
    worker.parent.mkdir(parents=True)
    worker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
sleep 2
"""
    )
    worker.chmod(0o755)
    return worker


def test_host_launcher_reports_live_process_pid(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _fake_worker(home)

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)],
        env=_environment(home),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    worker_pid = int(result.stdout.strip())
    os.kill(worker_pid, 0)
    pid_file = (
        home
        / "cascadia"
        / "cascadiav3"
        / "logs"
        / "all_wildlife_bound_launch_test_john-test.pid"
    )
    assert pid_file.read_text() == f"{worker_pid}\n"
    time.sleep(2.1)


def test_host_launcher_allows_rerun_without_receipt_checks(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _fake_worker(home)

    first = subprocess.run(
        ["/bin/bash", str(LAUNCHER)],
        env=_environment(home, tag="invalid_pid"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    second = subprocess.run(
        ["/bin/bash", str(LAUNCHER)],
        env=_environment(home, tag="invalid_pid"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert int(first.stdout) != int(second.stdout)
    time.sleep(2.1)
