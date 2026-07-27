#!/usr/bin/env python3
"""Monitor and recover the incremental fixed-count wildlife fleet pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY
    / "cascadiav3/fleet/all_wildlife_fixed_count_pipeline_20260726.json"
)
DEFAULT_STATUS_LOG = (
    REPOSITORY
    / "cascadiav3/logs/all_wildlife_fixed_count_watchdog_20260726.jsonl"
)
PIPELINE_LAUNCHER = (
    "cascadiav3/scripts/"
    "fleet_all_wildlife_fixed_count_candidates_pipeline_launch_host.sh"
)
PIPELINE_SCRIPT = (
    "cascadiav3/scripts/fleet_all_wildlife_fixed_count_candidates_pipeline.sh"
)
SYNC_SCRIPT = (
    "cascadiav3/scripts/"
    "fleet_all_wildlife_fixed_count_candidates_pipeline_sync.sh"
)


@dataclass(frozen=True)
class HostStatus:
    host: str
    reachable: bool
    running: bool
    pipeline_pid: int | None
    stage: str
    exit_code: int | None
    heartbeat_epoch: int | None
    heartbeat_age_seconds: int | None
    heartbeat_chunk: int | None
    progress_since_epoch: int | None
    progress_age_seconds: int | None
    progress_stalled: bool
    healthy: bool
    detail: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _shell_join_environment(values: dict[str, str | int]) -> str:
    return " ".join(
        f"{name}={shlex.quote(str(value))}" for name, value in values.items()
    )


def _run_on_host(
    host: str,
    command: str,
    *,
    timeout: int = 45,
) -> subprocess.CompletedProcess[str]:
    if host == "john1":
        argv = ["/bin/bash", "-lc", command]
    else:
        argv = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            command,
        ]
    return subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value or not value.lstrip("-").isdigit():
        return None
    return int(value)


def _status_command(pipeline_tag: str, host: str) -> str:
    quoted_tag = shlex.quote(pipeline_tag)
    quoted_host = shlex.quote(host)
    return f"""
set -u
root="$HOME/cascadia"
log_dir="$root/cascadiav3/logs"
tag={quoted_tag}
host={quoted_host}
pid_file="$log_dir/all_wildlife_fixed_pipeline_${{tag}}_${{host}}.pid"
exit_file="$log_dir/all_wildlife_fixed_pipeline_${{tag}}_${{host}}.exit"
stage_file="$log_dir/all_wildlife_fixed_pipeline_${{tag}}_${{host}}.stage"
pid=""
running=0
if [ -f "$pid_file" ]; then
  pid="$(tr -cd '0-9' < "$pid_file")"
fi
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$command" in
    *{PIPELINE_SCRIPT}*) running=1 ;;
  esac
fi
stage=""
if [ -f "$stage_file" ]; then
  stage="$(tr -d '\\r\\n' < "$stage_file")"
fi
exit_code=""
if [ -f "$exit_file" ]; then
  exit_code="$(tr -cd '0-9-' < "$exit_file")"
fi
heartbeat_epoch=""
heartbeat_chunk=""
if [ -n "$stage" ] && [ "$stage" != "complete" ]; then
  heartbeat="$log_dir/all_wildlife_fixed_${{stage}}_${{host}}.heartbeat"
  if [ -f "$heartbeat" ]; then
    heartbeat_epoch="$(stat -f '%m' "$heartbeat" 2>/dev/null || true)"
    heartbeat_chunk="$(sed -n 's/.* chunk=\\([0-9][0-9]*\\).*/\\1/p' "$heartbeat")"
  fi
fi
printf 'pid=%s\\nrunning=%s\\nstage=%s\\nexit_code=%s\\nheartbeat_epoch=%s\\nheartbeat_chunk=%s\\n' \
  "$pid" "$running" "$stage" "$exit_code" "$heartbeat_epoch" "$heartbeat_chunk"
"""


def inspect_host(
    pipeline_tag: str,
    host: str,
    *,
    stale_after_seconds: int,
    now_epoch: int | None = None,
) -> HostStatus:
    now_epoch = int(time.time()) if now_epoch is None else now_epoch
    try:
        result = _run_on_host(host, _status_command(pipeline_tag, host))
    except (OSError, subprocess.TimeoutExpired) as error:
        return HostStatus(
            host=host,
            reachable=False,
            running=False,
            pipeline_pid=None,
            stage="",
            exit_code=None,
            heartbeat_epoch=None,
            heartbeat_age_seconds=None,
            heartbeat_chunk=None,
            progress_since_epoch=None,
            progress_age_seconds=None,
            progress_stalled=False,
            healthy=False,
            detail=str(error),
        )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"status command exited {result.returncode}"
        return HostStatus(
            host=host,
            reachable=False,
            running=False,
            pipeline_pid=None,
            stage="",
            exit_code=None,
            heartbeat_epoch=None,
            heartbeat_age_seconds=None,
            heartbeat_chunk=None,
            progress_since_epoch=None,
            progress_age_seconds=None,
            progress_stalled=False,
            healthy=False,
            detail=detail,
        )

    fields = {}
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            fields[name] = value
    running = fields.get("running") == "1"
    stage = fields.get("stage", "")
    heartbeat_epoch = _parse_optional_int(fields.get("heartbeat_epoch"))
    heartbeat_age = (
        max(0, now_epoch - heartbeat_epoch)
        if heartbeat_epoch is not None
        else None
    )
    complete = stage == "complete"
    heartbeat_fresh = (
        heartbeat_age is not None and heartbeat_age <= stale_after_seconds
    )
    healthy = complete or (running and heartbeat_fresh)
    if complete:
        detail = "pipeline complete"
    elif not running:
        detail = "pipeline process absent"
    elif heartbeat_epoch is None:
        detail = "pipeline running without a stage heartbeat"
    elif not heartbeat_fresh:
        detail = f"stage heartbeat stale by {heartbeat_age} seconds"
    else:
        detail = "pipeline and stage heartbeat are live"
    return HostStatus(
        host=host,
        reachable=True,
        running=running,
        pipeline_pid=_parse_optional_int(fields.get("pid")),
        stage=stage,
        exit_code=_parse_optional_int(fields.get("exit_code")),
        heartbeat_epoch=heartbeat_epoch,
        heartbeat_age_seconds=heartbeat_age,
        heartbeat_chunk=_parse_optional_int(fields.get("heartbeat_chunk")),
        progress_since_epoch=None,
        progress_age_seconds=None,
        progress_stalled=False,
        healthy=healthy,
        detail=detail,
    )


def _timestamp_epoch(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _previous_status(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _with_progress_health(
    status: HostStatus,
    previous: dict[str, Any] | None,
    *,
    now_epoch: int,
    progress_stale_after_seconds: int,
) -> HostStatus:
    if (
        not status.running
        or status.stage == "complete"
        or status.heartbeat_chunk is None
    ):
        return status

    previous_host = None
    if previous is not None:
        previous_host = next(
            (
                host
                for host in previous.get("hosts", ())
                if isinstance(host, dict) and host.get("host") == status.host
            ),
            None,
        )
    same_work = (
        previous_host is not None
        and previous_host.get("stage") == status.stage
        and previous_host.get("heartbeat_chunk") == status.heartbeat_chunk
    )
    if same_work:
        progress_since = _parse_optional_int(
            str(previous_host.get("progress_since_epoch", ""))
        )
        if progress_since is None:
            progress_since = _timestamp_epoch(previous.get("timestamp"))
    else:
        progress_since = now_epoch
    if progress_since is None:
        progress_since = now_epoch
    progress_age = max(0, now_epoch - progress_since)
    stalled = progress_age >= progress_stale_after_seconds
    if stalled:
        return replace(
            status,
            progress_since_epoch=progress_since,
            progress_age_seconds=progress_age,
            progress_stalled=True,
            healthy=False,
            detail=(
                f"pipeline has remained on chunk {status.heartbeat_chunk} "
                f"for {progress_age} seconds"
            ),
        )
    return replace(
        status,
        progress_since_epoch=progress_since,
        progress_age_seconds=progress_age,
    )


def _pipeline_environment(config: dict[str, Any], shard: dict[str, Any]) -> dict[str, str | int]:
    stages = {stage["name"]: stage for stage in config["stages"]}
    return {
        "PIPELINE_TAG": config["pipeline_tag"],
        "SHARD_HOST": shard["host"],
        "SHARD_INDEX": shard["shard_index"],
        "SHARD_COUNT": shard["shard_count"],
        "SHALLOW_TAG": stages["shallow"]["tag"],
        "PRODUCTION_TAG": stages["production"]["tag"],
        "CHUNK_SIZE": config["durability"]["chunk_size"],
        "TOTAL_CELLS": config["scope"]["cells"],
        "SEARCH_THREADS": config["search_threads_per_host"],
        "BASE_SEED": stages["shallow"]["base_seed"],
        "HEARTBEAT_INTERVAL": 15,
        "SHALLOW_RESTARTS": stages["shallow"]["restarts_per_cell"],
        "SHALLOW_ITERATIONS": stages["shallow"]["iterations_per_restart"],
        "PRODUCTION_RESTARTS": stages["production"]["restarts_per_cell"],
        "PRODUCTION_ITERATIONS": stages["production"]["iterations_per_restart"],
    }


def launch_host(config: dict[str, Any], shard: dict[str, Any]) -> str:
    environment = _shell_join_environment(_pipeline_environment(config, shard))
    if shard["host"] == "john1":
        session = "cascadia-fixed-count-pipeline-john1"
        log = (
            "$HOME/cascadia/cascadiav3/logs/"
            f"all_wildlife_fixed_pipeline_{config['pipeline_tag']}_john1.log"
        )
        foreground = (
            f"cd \"$HOME/cascadia\" && {environment} "
            f"/bin/bash {shlex.quote(PIPELINE_SCRIPT)} >> \"{log}\" 2>&1"
        )
        command = (
            f"/opt/homebrew/bin/tmux kill-session -t {shlex.quote(session)} "
            f">/dev/null 2>&1 || true; "
            f"/opt/homebrew/bin/tmux new-session -d -s {shlex.quote(session)} "
            f"{shlex.quote(foreground)}; "
            f"/opt/homebrew/bin/tmux display-message -p -t {shlex.quote(session)} "
            "'#{pane_pid}'"
        )
    else:
        command = (
            f"cd \"$HOME/cascadia\" && {environment} "
            f"/bin/bash {shlex.quote(PIPELINE_LAUNCHER)}"
        )
    result = _run_on_host(shard["host"], command, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"{shard['host']} launch exited {result.returncode}"
        )
    return result.stdout.strip()


def terminate_host(config: dict[str, Any], status: HostStatus) -> None:
    if status.pipeline_pid is None:
        return
    tag = shlex.quote(config["pipeline_tag"])
    host = shlex.quote(status.host)
    pid = status.pipeline_pid
    command = f"""
set -eu
root="$HOME/cascadia"
pid={pid}
command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
case "$command" in
  *{PIPELINE_SCRIPT}*) ;;
  *) exit 66 ;;
esac
kill -TERM "$pid"
for unused in $(jot 40 2>/dev/null || seq 1 40); do
  if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi
  sleep 0.25
done
echo "pipeline $pid did not exit after TERM for tag {tag} host {host}" >&2
exit 70
"""
    result = _run_on_host(status.host, command, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"{status.host} termination exited {result.returncode}"
        )


def _sync_running() -> bool:
    result = subprocess.run(
        ["/usr/bin/pgrep", "-f", SYNC_SCRIPT],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def launch_sync(config: dict[str, Any]) -> int:
    stages = {stage["name"]: stage for stage in config["stages"]}
    best_stage = stages["best-of-stages"]
    environment = os.environ.copy()
    environment.update(
        {
            "SHALLOW_TAG": stages["shallow"]["tag"],
            "PRODUCTION_TAG": stages["production"]["tag"],
            "BEST_TAG": best_stage["tag"],
            "REMOTE_HOSTS": " ".join(
                shard["host"]
                for shard in config["shards"]
                if shard["host"] != "john1"
            ),
            "SYNC_INTERVAL": str(
                config["durability"]["central_sync_interval_seconds"]
            ),
            "CHUNK_SIZE": str(config["durability"]["chunk_size"]),
        }
    )
    log_path = (
        REPOSITORY
        / "cascadiav3/logs/all_wildlife_fixed_count_pipeline_sync.log"
    )
    pid_path = (
        REPOSITORY
        / "cascadiav3/logs/all_wildlife_fixed_count_pipeline_sync.pid"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            ["/bin/bash", str(REPOSITORY / SYNC_SCRIPT)],
            cwd=REPOSITORY,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    temporary = pid_path.with_suffix(".pid.tmp")
    temporary.write_text(f"{process.pid}\n")
    os.replace(temporary, pid_path)
    return process.pid


def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None


def _append_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def run_watchdog(
    config_path: Path,
    status_log: Path,
    *,
    stale_after_seconds: int,
    progress_stale_after_seconds: int,
    restart: bool,
) -> dict[str, Any]:
    config = json.loads(config_path.read_bytes())
    now_epoch = int(time.time())
    previous = _previous_status(status_log)
    actions: list[dict[str, Any]] = []
    statuses = [
        _with_progress_health(
            inspect_host(
                config["pipeline_tag"],
                shard["host"],
                stale_after_seconds=stale_after_seconds,
                now_epoch=now_epoch,
            ),
            previous,
            now_epoch=now_epoch,
            progress_stale_after_seconds=progress_stale_after_seconds,
        )
        for shard in config["shards"]
    ]
    status_by_host = {status.host: status for status in statuses}

    if restart:
        for shard in config["shards"]:
            status = status_by_host[shard["host"]]
            if not status.reachable or status.stage == "complete" or status.healthy:
                continue
            try:
                if status.running:
                    terminate_host(config, status)
                    actions.append(
                        {
                            "host": status.host,
                            "action": "terminated_stale_pipeline",
                            "pid": status.pipeline_pid,
                        }
                    )
                launched_pid = launch_host(config, shard)
                actions.append(
                    {
                        "host": status.host,
                        "action": "launched_pipeline",
                        "pid": launched_pid,
                    }
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                actions.append(
                    {
                        "host": status.host,
                        "action": "recovery_failed",
                        "error": str(error),
                    }
                )

    sync_running = _sync_running()
    if restart and not sync_running and not all(
        status.stage == "complete" for status in statuses
    ):
        try:
            sync_pid = launch_sync(config)
            sync_running = True
            actions.append(
                {
                    "host": "john1",
                    "action": "launched_sync",
                    "pid": sync_pid,
                }
            )
        except OSError as error:
            actions.append(
                {
                    "host": "john1",
                    "action": "sync_recovery_failed",
                    "error": str(error),
                }
            )

    summaries = {
        name: _read_summary(REPOSITORY / relative)
        for name, relative in config.get("summary_paths", {}).items()
    }
    payload = {
        "schema": "all-wildlife-fixed-count-watchdog-v1",
        "timestamp": _utc_now(),
        "restart_enabled": restart,
        "stale_after_seconds": stale_after_seconds,
        "progress_stale_after_seconds": progress_stale_after_seconds,
        "hosts": [asdict(status) for status in statuses],
        "sync_running": sync_running,
        "summaries": summaries,
        "actions": actions,
    }
    _append_status(status_log, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--status-log", type=Path, default=DEFAULT_STATUS_LOG)
    parser.add_argument("--stale-after-seconds", type=int, default=20 * 60)
    parser.add_argument(
        "--progress-stale-after-seconds",
        type=int,
        default=45 * 60,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="record health without restarting unhealthy workers",
    )
    args = parser.parse_args()
    if args.stale_after_seconds < 60:
        parser.error("--stale-after-seconds must be at least 60")
    if args.progress_stale_after_seconds < 5 * 60:
        parser.error("--progress-stale-after-seconds must be at least 300")
    payload = run_watchdog(
        args.config,
        args.status_log,
        stale_after_seconds=args.stale_after_seconds,
        progress_stale_after_seconds=args.progress_stale_after_seconds,
        restart=not args.check_only,
    )
    print(json.dumps(payload, sort_keys=True))
    failed_recoveries = {
        "recovery_failed",
        "sync_recovery_failed",
    }
    return int(
        any(action["action"] in failed_recoveries for action in payload["actions"])
    )


if __name__ == "__main__":
    sys.exit(main())
