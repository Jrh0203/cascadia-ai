#!/usr/bin/env python3
"""Publish a truthful heartbeat for the active 100k greedy bootstrap run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import tempfile
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

CAMPAIGN_ROOT = Path("/Users/johnherrick/cascadia-bench/r2-map-v1")
STATUS_PATH = CAMPAIGN_ROOT / "control/dashboard-status.json"
LOCK_PATH = CAMPAIGN_ROOT / "control/.dashboard-status.lock"
OUTPUT_ROOT = CAMPAIGN_ROOT / "generation/r2-map-bootstrap-100000-v1/john1"
RETURN_ROOT = CAMPAIGN_ROOT / "generation/r2-map-bootstrap-100000-v1"
AGGREGATE_RECEIPT = CAMPAIGN_ROOT / "datasets/r2-map-bootstrap-100000-v1/aggregate-receipt.json"
PACKING_RECEIPT = (
    CAMPAIGN_ROOT / "datasets/r2-map-bootstrap-100000-v1/packing-selection-receipt.json"
)
PACKING_PLAN = (
    CAMPAIGN_ROOT / "datasets/r2-map-bootstrap-100000-v1/focal-seat-one-epoch-packing-plan.json"
)
IMAGE_LOAD_RECEIPT = CAMPAIGN_ROOT / "control/images/john1-canonical-r2-map-image-v1.json"
TRAINING_RUN_ID = "r2-map-bootstrap-iteration0-value-v1"
TRAINING_RUN = CAMPAIGN_ROOT / f"runs/{TRAINING_RUN_ID}"
TRAINING_STEPS = 7_235
TRAINING_GROUP_BATCH_SIZE = 256
LAGGED_BENCHMARK_AGGREGATE = (
    CAMPAIGN_ROOT / "benchmarks/r2-map-lagged-greedy-baseline-aggregate-v1/report.json"
)
DOCKER_HOST = "unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock"
RUN_ID = "r2-map-bootstrap-100000-v1"
JOHN1_TARGET = 33_334
TOTAL_TARGET = 100_000
PER_HOST_RATE = 2.068787174
CLUSTER_RATE = 6.206361522
WORKER_TARGETS = (3334, 3334, 3334, 3334, 3333, 3333, 3333, 3333, 3333, 3333)
REMOTE_WORKER_TARGETS = (3334, 3334, 3334, 3333, 3333, 3333, 3333, 3333, 3333, 3333)
REMOTE_HOSTS = {
    "john2": (33_333, "33334-66666"),
    "john3": (33_333, "66667-99999"),
}
FOCAL_CAMPAIGN_PHASES = frozenset(
    {
        "r2-map-strength-blinded-smoke",
        "r2-map-strength-blinded-smoke-complete",
        "r2-map-fixed-250-comparison",
        "r2-map-fixed-250-comparison-complete",
    }
)


def focal_campaign_owns_status(status: dict[str, Any]) -> bool:
    """Yield permanently once the post-training focal gate owns the dashboard."""

    return status.get("phase") in FOCAL_CAMPAIGN_PHASES


def read_completed_games(root: Path = OUTPUT_ROOT) -> tuple[int, int, float | None]:
    completed = 0
    readable = 0
    first_created: int | None = None
    last_updated: int | None = None
    for worker, target in enumerate(WORKER_TARGETS):
        path = root / f"worker-{worker}/dataset.json"
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        value = manifest.get("completed_games")
        lease = manifest.get("lease", {})
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= target
            and lease.get("host_id") == "john1"
            and lease.get("game_count") == target
        ):
            completed += value
            readable += 1
            created = manifest.get("created_unix_seconds")
            updated = manifest.get("updated_unix_seconds")
            if isinstance(created, int) and isinstance(updated, int):
                first_created = created if first_created is None else min(first_created, created)
                last_updated = updated if last_updated is None else max(last_updated, updated)
    elapsed = 0 if first_created is None or last_updated is None else last_updated - first_created
    rate = completed / elapsed if completed > 0 and elapsed > 0 else None
    return completed, readable, rate


def read_returned_host(host: str) -> tuple[int, int]:
    root = RETURN_ROOT / host
    completed = readable = 0
    for worker, target in enumerate(REMOTE_WORKER_TARGETS):
        path = root / f"worker-{worker}/dataset.json"
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lease = manifest.get("lease", {})
        value = manifest.get("completed_games")
        if (
            lease.get("host_id") == host
            and lease.get("game_count") == target
            and value == target
            and manifest.get("primary_example_count") == target * 80
        ):
            completed += value
            readable += 1
    return completed, readable


def aggregate_is_complete() -> bool:
    try:
        receipt = json.loads(AGGREGATE_RECEIPT.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        receipt.get("schema_id") == "cascadia.r2-map.bootstrap-aggregate.v1"
        and receipt.get("result") == "pass"
        and receipt.get("games") == TOTAL_TARGET
        and receipt.get("primary_example_count") == TOTAL_TARGET * 80
        and receipt.get("worker_datasets") == 30
        and receipt.get("replay_shards") == 420
        and receipt.get("completion_audits") == 30
    )


def packing_is_complete() -> bool:
    try:
        receipt = json.loads(PACKING_RECEIPT.read_text())
        plan = json.loads(PACKING_PLAN.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        receipt.get("schema_id") == "cascadia.r2-map.bootstrap-packing-selection.v1"
        and receipt.get("result") == "pass"
        and plan.get("schema_id") == "r2-map-focal-seat-one-epoch-packing-v1"
        and plan.get("configured_group_batch_size") == TRAINING_GROUP_BATCH_SIZE
        and plan.get("maximum_candidates_per_batch") == TRAINING_GROUP_BATCH_SIZE
        and plan.get("steps") == TRAINING_STEPS
        and plan.get("draft_policy_targets") == 0
    )


def _training_process_active() -> bool:
    try:
        result = subprocess.run(
            [
                "/usr/bin/pgrep",
                "-f",
                f"cascadia_mlx.r2_map_train.*--run-id {TRAINING_RUN_ID}",
            ],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _elapsed_seconds(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    days = 0
    if "-" in value:
        raw_days, value = value.split("-", 1)
        try:
            days = int(raw_days)
        except ValueError:
            return None
    try:
        fields = [int(field) for field in value.split(":")]
    except ValueError:
        return None
    if len(fields) == 2:
        hours = 0
        minutes, seconds = fields
    elif len(fields) == 3:
        hours, minutes, seconds = fields
    else:
        return None
    if min(days, hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
        return None
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


def _training_process_metrics() -> dict[str, int | str | None]:
    try:
        pgrep = subprocess.run(
            [
                "/usr/bin/pgrep",
                "-f",
                f"cascadia_mlx.r2_map_train.*--run-id {TRAINING_RUN_ID}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw_pid = next((line.strip() for line in pgrep.stdout.splitlines() if line.strip()), None)
        if pgrep.returncode != 0 or raw_pid is None:
            return {
                "pid": None,
                "rss_bytes": None,
                "elapsed_seconds": None,
                "branch_id": None,
            }
        pid = int(raw_pid)
        ps = subprocess.run(
            ["/bin/ps", "-o", "rss=,etime=,command=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, StopIteration, ValueError, subprocess.TimeoutExpired):
        return {
            "pid": None,
            "rss_bytes": None,
            "elapsed_seconds": None,
            "branch_id": None,
        }
    fields = ps.stdout.split(None, 2)
    if ps.returncode != 0 or len(fields) != 3:
        return {
            "pid": pid,
            "rss_bytes": None,
            "elapsed_seconds": None,
            "branch_id": None,
        }
    try:
        rss_bytes = int(fields[0]) * 1024
    except ValueError:
        rss_bytes = None
    branch = re.search(r"(?:^|\s)--branch-id\s+(\S+)", fields[2])
    return {
        "pid": pid,
        "rss_bytes": rss_bytes,
        "elapsed_seconds": _elapsed_seconds(fields[1]),
        "branch_id": branch.group(1) if branch else None,
    }


def _system_swap_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])", result.stdout)
    if not match:
        return None
    multipliers = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    return round(float(match.group(1)) * multipliers[match.group(2)])


def _dashboard_resume_swap_baseline(
    run_dir: Path, branch_id: str, observed_swap_bytes: int
) -> int:
    if re.fullmatch(r"resume-step-[0-9]+-[0-9]+", branch_id) is None:
        raise ValueError("dashboard resume branch identity is malformed")
    path = run_dir / "dashboard-resource-baselines" / f"{branch_id}.json"
    try:
        value = json.loads(path.read_text())
        baseline = value.get("system_swap_bytes")
        if (
            value.get("schema_id")
            != "cascadia.r2-map.dashboard-resume-resource-baseline.v1"
            or value.get("branch_id") != branch_id
            or not isinstance(baseline, int)
            or isinstance(baseline, bool)
            or baseline < 0
        ):
            raise ValueError("dashboard resume resource baseline differs")
        return baseline
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_id": "cascadia.r2-map.dashboard-resume-resource-baseline.v1",
        "branch_id": branch_id,
        "observed_unix_ms": time.time_ns() // 1_000_000,
        "system_swap_bytes": observed_swap_bytes,
        "authority": "dashboard-only; trainer resource receipt remains authoritative",
    }
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _dashboard_resume_swap_baseline(run_dir, branch_id, observed_swap_bytes)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return observed_swap_bytes
    finally:
        temporary.unlink(missing_ok=True)


def _loss_tail(path: Path, maximum_records: int = 20) -> list[dict[str, Any]]:
    records: deque[dict[str, Any]] = deque(maxlen=maximum_records)
    try:
        with path.open("rb") as handle:
            for raw in handle:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        return []
    return list(records)


def _branch_start_step(branch_id: object) -> int:
    if not isinstance(branch_id, str):
        return 0
    match = re.fullmatch(r"resume-step-([0-9]+)-[0-9]+", branch_id)
    return int(match.group(1)) if match else 0


def read_training_state(
    run_dir: Path = TRAINING_RUN,
    *,
    process_active: bool | None = None,
) -> dict[str, Any]:
    active = _training_process_active() if process_active is None else process_active
    losses = _loss_tail(run_dir / "losses/loss-stream.jsonl")
    process = (
        _training_process_metrics()
        if active
        else {
            "pid": None,
            "rss_bytes": None,
            "elapsed_seconds": None,
            "branch_id": None,
        }
    )
    active_branch = process.get("branch_id")
    active_branch_losses = (
        [value for value in losses if value.get("branch_id") == active_branch]
        if isinstance(active_branch, str)
        else []
    )
    branch_start = _branch_start_step(active_branch)
    if active_branch_losses:
        latest = active_branch_losses[-1]
        visible_losses = [
            value
            for value in losses
            if value.get("branch_id") == active_branch
            or (
                isinstance(value.get("global_step"), int)
                and value["global_step"] <= branch_start
            )
        ]
    elif active and isinstance(active_branch, str):
        latest = {"global_step": branch_start, "branch_id": active_branch, "metrics": {}}
        visible_losses = [
            value
            for value in losses
            if isinstance(value.get("global_step"), int)
            and value["global_step"] <= branch_start
        ]
    else:
        latest = losses[-1] if losses else {}
        visible_losses = losses
    current_step = latest.get("global_step")
    # A verified resume branch may intentionally overlap uncheckpointed steps
    # from its parent. Retain the later append-only event for an overlapping
    # global step and restore the compact chart's strict ordering.
    samples_by_step = {
        value["global_step"]: {
            "step": value["global_step"],
            "train_total": value.get("metrics", {}).get("total_loss"),
            "validation_total": None,
        }
        for value in visible_losses
        if isinstance(value.get("global_step"), int)
    }
    loss_samples = [samples_by_step[step] for step in sorted(samples_by_step)]
    latest_verified = None
    try:
        pointer = json.loads((run_dir / "last_verified.json").read_text())
        checkpoint_id = pointer.get("checkpoint")
        manifest_blake3 = pointer.get("manifest_blake3")
        if isinstance(checkpoint_id, str) and checkpoint_id:
            latest_verified = {
                "id": checkpoint_id,
                "blake3": manifest_blake3
                if isinstance(manifest_blake3, str) and manifest_blake3
                else None,
            }
        match = re.search(r"step-([0-9]+)$", str(checkpoint_id))
        if match and not isinstance(current_step, int):
            current_step = int(match.group(1))
    except (OSError, json.JSONDecodeError):
        pass
    complete = False
    try:
        receipt = json.loads((run_dir / "training-command-receipt.json").read_text())
        complete = (
            receipt.get("schema_id") == "r2-map-training-command-receipt-v1"
            and receipt.get("run_id") == TRAINING_RUN_ID
            and receipt.get("final_step") == TRAINING_STEPS
        )
        if complete:
            # Loss events are intentionally sparse, so the last loss sample can
            # precede the terminal checkpoint. Once the bound command receipt
            # is complete, report its authoritative final step rather than the
            # last chart sample.
            current_step = TRAINING_STEPS
    except (OSError, json.JSONDecodeError):
        pass
    baseline_swap = None
    try:
        baseline = json.loads((run_dir / "resource-baseline.json").read_text())
        if isinstance(baseline.get("system_swap_bytes"), int):
            baseline_swap = baseline["system_swap_bytes"]
    except (OSError, json.JSONDecodeError):
        pass
    system_swap = _system_swap_bytes() if active else None
    if (
        active
        and isinstance(active_branch, str)
        and active_branch.startswith("resume-step-")
        and isinstance(system_swap, int)
    ):
        baseline_swap = _dashboard_resume_swap_baseline(
            run_dir, active_branch, system_swap
        )
    swap_delta = (
        system_swap - baseline_swap
        if system_swap is not None and baseline_swap is not None
        else None
    )
    elapsed_seconds = process["elapsed_seconds"]
    examples_per_second = None
    eta_seconds = None
    if (
        active
        and isinstance(current_step, int)
        and current_step > 0
        and isinstance(elapsed_seconds, int)
        and elapsed_seconds > 0
    ):
        completed_process_steps = current_step - _branch_start_step(
            active_branch if isinstance(active_branch, str) else latest.get("branch_id")
        )
        if completed_process_steps > 0:
            steps_per_second = completed_process_steps / elapsed_seconds
            examples_per_second = steps_per_second * TRAINING_GROUP_BATCH_SIZE
            eta_seconds = (TRAINING_STEPS - current_step) / steps_per_second
    return {
        "active": active,
        "complete": complete,
        "current_step": current_step,
        "latest_verified_checkpoint": latest_verified,
        "loss_samples": loss_samples,
        "run_exists": run_dir.exists(),
        "rss_bytes": process["rss_bytes"],
        "swap_delta_bytes": swap_delta,
        "examples_per_second": examples_per_second,
        "eta_seconds": eta_seconds,
    }


def apply_post_bootstrap_phase(status: dict[str, Any]) -> None:
    if not packing_is_complete():
        return
    training = read_training_state()
    status.setdefault("training", {}).update(
        {
            "active": training["active"],
            "current_step": training["current_step"],
            "total_steps": TRAINING_STEPS,
            "latest_verified_checkpoint": training["latest_verified_checkpoint"],
            "loss_samples": training["loss_samples"],
            "examples_per_second": training["examples_per_second"],
            "eta_seconds": training["eta_seconds"],
        }
    )
    john1 = status["hosts"]["john1"]
    if training["complete"]:
        status["phase"] = "bootstrap-training-complete"
        status["legal_next_transitions"] = [
            "verify-bootstrap-training-receipt",
            "freeze-terminal-bootstrap-checkpoint",
            "build-and-distribute-cross-architecture-image",
            "run-strength-blinded-20-pair-smoke",
        ]
        john1["intent"] = "control"
        phase = "complete"
    elif training["active"]:
        status["phase"] = "bootstrap-training"
        status["legal_next_transitions"] = [
            "monitor-bootstrap-training",
            "prepare-r2-vs-qualified-nnue-candidate-gate",
        ]
        john1["intent"] = "train"
        phase = "active"
    elif training["run_exists"]:
        status["phase"] = "bootstrap-training-recovery-required"
        status["legal_next_transitions"] = ["resume-from-last-verified-checkpoint"]
        john1["intent"] = "validate"
        phase = "recovery-required"
    elif not IMAGE_LOAD_RECEIPT.exists():
        status["phase"] = "bootstrap-training-image-supply"
        status["legal_next_transitions"] = [
            "build-and-test-canonical-image-on-john1",
            "export-and-load-exact-image-on-john2-john3",
        ]
        john1["intent"] = "control"
        for host in ("john2", "john3"):
            status["hosts"][host]["intent"] = "validate"
            status["hosts"][host]["detail"] = (
                "role=execution-only; phase=canonical-image-load; "
                "source_authority=john1; image_builder=john1"
            )
        phase = "image-supply"
    else:
        status["phase"] = "bootstrap-packing-selected"
        status["legal_next_transitions"] = ["train-bootstrap-iteration-0-on-john1"]
        john1["intent"] = "control"
        phase = "ready"
    john1["detail"] = (
        f"role=control+mlx; phase=bootstrap-training-{phase}; "
        f"run={TRAINING_RUN_ID}; current_step={training['current_step']}; "
        f"total_steps={TRAINING_STEPS}; group_batch_size={TRAINING_GROUP_BATCH_SIZE}; "
        f"last_verified={training['latest_verified_checkpoint']}; "
        f"resource_baseline_step=20"
    )
    john1["rss_bytes"] = training["rss_bytes"]
    john1["swap_delta_bytes"] = training["swap_delta_bytes"]
    john1["eta_seconds"] = training["eta_seconds"]
    if training["active"]:
        john1["throughput_games_per_second"] = None
    try:
        lagged = json.loads(LAGGED_BENCHMARK_AGGREGATE.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if (
        lagged.get("schema_id") != "cascadia.r2-map.lagged-greedy-benchmark-aggregate.v1"
        or lagged.get("games") != 5_000
        or lagged.get("seed_coverage") != "contiguous-exactly-once"
    ):
        return
    status["benchmark"].update(
        {
            "active": False,
            "classification": "pending",
            "stage": "cross-architecture-smoke-awaiting-candidate",
            "pairs_completed": 0,
            "pairs_total": 20,
            "eta_seconds": None,
            "throughput_games_per_second": None,
            # The v2 baseline has exact total percentiles and component means,
            # but the serving schema requires component percentiles as well.
            # Keep the typed dashboard honest by withholding the incomplete
            # focal bundle; the complete means remain in the bound aggregate
            # report and receipt named by the stage.
            "focal": None,
        }
    )
    for host in REMOTE_HOSTS:
        if status["phase"] != "bootstrap-training-image-supply":
            status["hosts"][host]["intent"] = "idle"
        status["hosts"][host]["benchmark_pairs_completed"] = 0
        status["hosts"][host]["benchmark_pairs_total"] = None


def read_container_states() -> dict[str, int]:
    names = [f"r2-bootstrap-100k-v1-john1-w{worker}" for worker in range(10)]
    command = [
        "docker",
        "inspect",
        "--format",
        "{{.Name}}\t{{.State.Status}}\t{{.State.ExitCode}}",
        *names,
    ]
    environment = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "DOCKER_HOST": DOCKER_HOST,
    }
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"running": 0, "successful": 0, "failed": 0, "unknown": 10, "memory_bytes": 0}
    running = successful = failed = 0
    running_names: list[str] = []
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        name, state, raw_exit = fields
        try:
            exit_code = int(raw_exit)
        except ValueError:
            continue
        if state == "running":
            running += 1
            running_names.append(name.removeprefix("/"))
        elif state == "exited" and exit_code == 0:
            successful += 1
        elif state in {"exited", "dead"}:
            failed += 1
    memory_bytes = 0
    if running_names:
        try:
            stats = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", *running_names],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=environment,
            )
            memory_bytes = sum(
                _memory_to_bytes(line.split()[0])
                for line in stats.stdout.splitlines()
                if line.split()
            )
        except (OSError, subprocess.TimeoutExpired):
            memory_bytes = 0
    return {
        "running": running,
        "successful": successful,
        "failed": failed,
        "unknown": max(0, 10 - running - successful - failed),
        "memory_bytes": memory_bytes,
    }


def _memory_to_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?i?B)", value)
    if match is None:
        return 0
    number = float(match.group(1))
    unit = match.group(2)
    multiplier = {
        "B": 1,
        "kB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "TB": 1_000_000_000_000,
        "KiB": 1 << 10,
        "MiB": 1 << 20,
        "GiB": 1 << 30,
        "TiB": 1 << 40,
    }.get(unit, 1)
    return int(number * multiplier)


def update_status(
    status: dict[str, Any],
    *,
    observed_completed: int,
    readable_manifests: int,
    states: dict[str, int],
    now_ms: int,
    observed_rate: float | None = None,
) -> dict[str, Any]:
    hosts = status["hosts"]
    previous = hosts["john1"].get("generation_games_completed", 0)
    if status.get("phase") not in {"bootstrap-generation", "bootstrap-generation-complete"}:
        previous = 0
    if not isinstance(previous, int) or isinstance(previous, bool):
        previous = 0
    completed = max(previous, observed_completed)
    completed = min(completed, JOHN1_TARGET)
    remaining = JOHN1_TARGET - completed
    rate = max(PER_HOST_RATE, observed_rate or 0.0)
    eta = int(remaining / rate) if remaining else 0
    detail = (
        f"role=control+execution+mlx; phase=bootstrap-generation; run={RUN_ID}; "
        f"topology=10x1; indices=0-33333; manifests={readable_manifests}/10; "
        f"containers_running={states['running']}; containers_successful={states['successful']}; "
        f"containers_failed={states['failed']}; memory_bytes={states.get('memory_bytes', 0)}; "
        f"completed={completed}; target={JOHN1_TARGET}; committed_rate={rate:.9f}"
    )
    hosts["john1"].update(
        {
            "intent": "generate" if remaining else "validate",
            "detail": detail,
            "generation_games_completed": completed,
            "generation_games_target": JOHN1_TARGET,
            "generation_seed_prefix": "global-game-index:0-33333",
            "eta_seconds": eta,
            "throughput_games_per_second": rate,
            "rss_bytes": states.get("memory_bytes", 0) or None,
        }
    )
    for host, (target, index_range) in REMOTE_HOSTS.items():
        returned, manifests = read_returned_host(host)
        if returned:
            hosts[host].update(
                {
                    "intent": "validate" if returned == target else "generate",
                    "detail": (
                        f"role=execution; phase=bootstrap-generation-return; run={RUN_ID}; "
                        f"indices={index_range}; manifests_returned={manifests}/10; "
                        f"completed={returned}; target={target}; authority=john1"
                    ),
                    "generation_games_completed": returned,
                    "generation_games_target": target,
                    "generation_seed_prefix": f"global-game-index:{index_range}",
                    "eta_seconds": 0 if returned == target else None,
                    "rss_bytes": None,
                }
            )
    total_completed = sum(
        max(0, int(host.get("generation_games_completed") or 0)) for host in hosts.values()
    )
    total_completed = min(total_completed, TOTAL_TARGET)
    benchmark = status["benchmark"]
    cluster_rate = rate + sum(
        float(hosts[host].get("throughput_games_per_second") or PER_HOST_RATE)
        for host in ("john2", "john3")
    )
    benchmark.update(
        {
            "active": total_completed < TOTAL_TARGET,
            "classification": "pending" if total_completed < TOTAL_TARGET else "promote",
            "stage": "greedy-bootstrap-generation-100000",
            "pairs_completed": total_completed,
            "pairs_total": TOTAL_TARGET,
            "eta_seconds": int((TOTAL_TARGET - total_completed) / cluster_rate)
            if total_completed < TOTAL_TARGET
            else 0,
            "throughput_games_per_second": cluster_rate,
            "peak_rss_bytes": max(
                int(benchmark.get("peak_rss_bytes") or 0), states.get("memory_bytes", 0)
            ),
        }
    )
    aggregate_complete = aggregate_is_complete()
    if aggregate_complete:
        status["phase"] = "bootstrap-validated"
        status["legal_next_transitions"] = [
            "select-bootstrap-packing-plan",
            "train-bootstrap-iteration-0-on-john1",
        ]
        benchmark.update(
            {
                "active": False,
                "classification": "promote",
                "stage": "greedy-bootstrap-aggregate-validated",
                "pairs_completed": TOTAL_TARGET,
                "pairs_total": TOTAL_TARGET,
                "eta_seconds": 0,
            }
        )
        hosts["john1"]["intent"] = "control"
        hosts["john1"]["detail"] = (
            f"role=control+mlx; phase=bootstrap-validated; run={RUN_ID}; "
            "datasets=30/30; games=100000; examples=8000000; shards=420; "
            "aggregate_receipt=pass"
        )
        for host in REMOTE_HOSTS:
            hosts[host]["intent"] = "idle"
        apply_post_bootstrap_phase(status)
    else:
        status["phase"] = (
            "bootstrap-generation"
            if total_completed < TOTAL_TARGET
            else "bootstrap-generation-complete"
        )
        status["legal_next_transitions"] = (
            ["monitor-bootstrap-generation", "collect-validated-remote-artifacts"]
            if total_completed < TOTAL_TARGET
            else [
                "aggregate-and-validate-bootstrap-100000",
                "train-bootstrap-iteration-0-on-john1",
            ]
        )
    status["stale_after_seconds"] = 30
    status["updated_unix_ms"] = now_ms
    return status


def publish_once() -> dict[str, Any]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = json.loads(STATUS_PATH.read_text())
        if focal_campaign_owns_status(status):
            return {
                "skipped": "focal-benchmark-owns-dashboard",
                "updated_unix_ms": status.get("updated_unix_ms"),
            }
        observed, readable, observed_rate = read_completed_games()
        states = read_container_states()
        updated = update_status(
            status,
            observed_completed=observed,
            readable_manifests=readable,
            states=states,
            now_ms=time.time_ns() // 1_000_000,
            observed_rate=observed_rate,
        )
        payload = json.dumps(updated, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".dashboard-status.bootstrap.", dir=STATUS_PATH.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), 0o600)
                os.fsync(handle.fileno())
            os.replace(temporary, STATUS_PATH)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "completed": updated["hosts"]["john1"]["generation_games_completed"],
        "running": states["running"],
        "updated_unix_ms": updated["updated_unix_ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.interval_seconds < 5:
        parser.error("--interval-seconds must be at least 5")
    while True:
        result = publish_once()
        if not arguments.watch:
            print(json.dumps(result, sort_keys=True))
            return 0
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
