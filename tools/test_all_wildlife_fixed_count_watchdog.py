from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools import all_wildlife_fixed_count_watchdog as watchdog


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def test_inspect_host_requires_a_live_process_and_fresh_heartbeat(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        watchdog,
        "_run_on_host",
        lambda *_args, **_kwargs: _completed(
            "pid=123\n"
            "running=1\n"
            "stage=all_wildlife_fixed_count_shallow_20260726\n"
            "exit_code=\n"
            "heartbeat_epoch=1000\n"
        ),
    )

    status = watchdog.inspect_host(
        "pipeline",
        "john2",
        stale_after_seconds=120,
        now_epoch=1060,
    )

    assert status.reachable
    assert status.running
    assert status.healthy
    assert status.heartbeat_age_seconds == 60


def test_inspect_host_marks_stale_heartbeat_unhealthy(monkeypatch) -> None:
    monkeypatch.setattr(
        watchdog,
        "_run_on_host",
        lambda *_args, **_kwargs: _completed(
            "pid=123\n"
            "running=1\n"
            "stage=all_wildlife_fixed_count_shallow_20260726\n"
            "exit_code=\n"
            "heartbeat_epoch=1000\n"
        ),
    )

    status = watchdog.inspect_host(
        "pipeline",
        "john2",
        stale_after_seconds=120,
        now_epoch=1201,
    )

    assert not status.healthy
    assert "stale" in status.detail


def test_watchdog_relaunches_an_absent_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = {
        "pipeline_tag": "pipeline",
        "scope": {"cells": 10},
        "durability": {
            "chunk_size": 2,
            "central_sync_interval_seconds": 300,
        },
        "stages": [
            {
                "name": "shallow",
                "tag": "shallow",
                "restarts_per_cell": 1,
                "iterations_per_restart": 2,
                "base_seed": 3,
            },
            {
                "name": "production",
                "tag": "production",
                "restarts_per_cell": 4,
                "iterations_per_restart": 5,
                "base_seed": 3,
            },
            {"name": "best-of-stages", "tag": "best"},
        ],
        "shards": [{"host": "john1", "shard_index": 0, "shard_count": 1}],
        "search_threads_per_host": 8,
        "summary_paths": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    status_log = tmp_path / "status.jsonl"
    absent = watchdog.HostStatus(
        host="john1",
        reachable=True,
        running=False,
        pipeline_pid=None,
        stage="shallow",
        exit_code=101,
        heartbeat_epoch=1000,
        heartbeat_age_seconds=5000,
        heartbeat_chunk=10,
        progress_since_epoch=None,
        progress_age_seconds=None,
        progress_stalled=False,
        healthy=False,
        detail="pipeline process absent",
    )
    monkeypatch.setattr(watchdog, "inspect_host", lambda *_args, **_kwargs: absent)
    monkeypatch.setattr(watchdog, "launch_host", lambda *_args, **_kwargs: "456")
    monkeypatch.setattr(watchdog, "_sync_running", lambda: True)

    result = watchdog.run_watchdog(
        config_path,
        status_log,
        stale_after_seconds=1200,
        progress_stale_after_seconds=2700,
        restart=True,
    )

    assert result["actions"] == [
        {"host": "john1", "action": "launched_pipeline", "pid": "456"}
    ]
    assert json.loads(status_log.read_text())["actions"] == result["actions"]


def test_john1_launch_is_anchored_in_tmux(monkeypatch) -> None:
    config = {
        "pipeline_tag": "pipeline",
        "scope": {"cells": 10},
        "durability": {"chunk_size": 2},
        "stages": [
            {
                "name": "shallow",
                "tag": "shallow",
                "restarts_per_cell": 1,
                "iterations_per_restart": 2,
                "base_seed": 3,
            },
            {
                "name": "production",
                "tag": "production",
                "restarts_per_cell": 4,
                "iterations_per_restart": 5,
                "base_seed": 3,
            },
        ],
        "search_threads_per_host": 8,
    }
    observed = {}

    def fake_run(host: str, command: str, **_kwargs):
        observed["host"] = host
        observed["command"] = command
        return _completed("789\n")

    monkeypatch.setattr(watchdog, "_run_on_host", fake_run)

    assert (
        watchdog.launch_host(
            config,
            {"host": "john1", "shard_index": 0, "shard_count": 1},
        )
        == "789"
    )
    assert observed["host"] == "john1"
    assert "tmux new-session" in observed["command"]
    assert watchdog.PIPELINE_SCRIPT in observed["command"]


def test_progress_health_detects_a_live_solver_stuck_on_one_chunk() -> None:
    status = watchdog.HostStatus(
        host="john2",
        reachable=True,
        running=True,
        pipeline_pid=123,
        stage="shallow",
        exit_code=None,
        heartbeat_epoch=3699,
        heartbeat_age_seconds=1,
        heartbeat_chunk=42,
        progress_since_epoch=None,
        progress_age_seconds=None,
        progress_stalled=False,
        healthy=True,
        detail="pipeline and stage heartbeat are live",
    )
    previous = {
        "timestamp": "1970-01-01T00:16:40Z",
        "hosts": [
            {
                "host": "john2",
                "stage": "shallow",
                "heartbeat_chunk": 42,
                "progress_since_epoch": 900,
            }
        ],
    }

    observed = watchdog._with_progress_health(
        status,
        previous,
        now_epoch=3700,
        progress_stale_after_seconds=2700,
    )

    assert not observed.healthy
    assert observed.progress_stalled
    assert observed.progress_age_seconds == 2800
    assert "chunk 42" in observed.detail


def test_progress_health_resets_when_the_chunk_advances() -> None:
    status = watchdog.HostStatus(
        host="john2",
        reachable=True,
        running=True,
        pipeline_pid=123,
        stage="shallow",
        exit_code=None,
        heartbeat_epoch=3699,
        heartbeat_age_seconds=1,
        heartbeat_chunk=43,
        progress_since_epoch=None,
        progress_age_seconds=None,
        progress_stalled=False,
        healthy=True,
        detail="pipeline and stage heartbeat are live",
    )
    previous = {
        "timestamp": "1970-01-01T00:16:40Z",
        "hosts": [
            {
                "host": "john2",
                "stage": "shallow",
                "heartbeat_chunk": 42,
                "progress_since_epoch": 900,
            }
        ],
    }

    observed = watchdog._with_progress_health(
        status,
        previous,
        now_epoch=3700,
        progress_stale_after_seconds=2700,
    )

    assert observed.healthy
    assert not observed.progress_stalled
    assert observed.progress_since_epoch == 3700
    assert observed.progress_age_seconds == 0
