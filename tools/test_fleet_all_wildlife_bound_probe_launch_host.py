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
        "SOURCE_REVISION": "revision",
        "TASKSET_SHA256": "taskset",
        "CATALOG_SHA256": "catalog",
        "PROBE_SOURCE_SHA256": "probe",
        "EXACT_SOURCE_SHA256": "exact",
        "EXACT_SUPPORT_SHA256": "support",
        "RULES_SOURCE_SHA256": "rules",
        "WILDLIFE_VENV": ".venv",
        "TIME_LIMIT": "1",
        "TOTAL_TIME_LIMIT": "2",
        "SOLVER_WORKERS": "1",
        "HEARTBEAT_INTERVAL": "1",
    }


def _fake_worker(home: Path, pid_value: str = "$$") -> Path:
    worker = (
        home
        / "cascadia"
        / "cascadiav3"
        / "scripts"
        / "fleet_all_wildlife_bound_probe_worker.sh"
    )
    worker.parent.mkdir(parents=True)
    worker.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
log_dir="$HOME/cascadia/cascadiav3/logs"
base="$log_dir/all_wildlife_bound_${{FLEET_TAG}}_${{SHARD_HOST}}"
printf '%s\\n' "{pid_value}" > "${{base}}.pid.tmp"
mv "${{base}}.pid.tmp" "${{base}}.pid"
printf 'heartbeat\\n' > "${{base}}.heartbeat.tmp"
mv "${{base}}.heartbeat.tmp" "${{base}}.heartbeat"
sleep 2
"""
    )
    worker.chmod(0o755)
    return worker


def test_host_launcher_requires_and_reports_live_worker_pid(tmp_path: Path) -> None:
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
    heartbeat = (
        home
        / "cascadia"
        / "cascadiav3"
        / "logs"
        / "all_wildlife_bound_launch_test_john-test.heartbeat"
    )
    assert heartbeat.read_text() == "heartbeat\n"
    time.sleep(2.1)


def test_host_launcher_rejects_nonnumeric_worker_pid(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _fake_worker(home, "not-a-pid")

    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)],
        env=_environment(home, tag="invalid_pid"),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 70
    assert "invalid PID" in result.stderr
    time.sleep(2.1)
