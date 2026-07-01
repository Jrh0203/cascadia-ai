#!/usr/bin/env python3
"""Cascadia V3 two-part campaign controller with a hard human approval gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import blake3

CAMPAIGN_SCHEMA = "cascadia-v3-campaign-state-v1"
READINESS_SCHEMA = "cascadia-v3-part1-readiness-v1"
CAMPAIGN_ID = "cascadia-v3-radius7-stockfish-nnue-v1"
DEFAULT_ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
DEFAULT_DASHBOARD = Path(
    "/Users/johnherrick/cascadia/artifacts/cluster/r2-map-dashboard-serving-projection-v2.json"
)
V3_DASHBOARD_SCHEMA = "cascadia.v3.dashboard-status.v1"
V3_DASHBOARD_PATH = DEFAULT_ROOT / "control" / "dashboard-status.json"
MAX_BYTES = 40 * 1024**3
MIN_FREE_BYTES = 50 * 1024**3
COLIMA_BINARY = Path("/opt/homebrew/bin/colima")
COLIMA_HOME = Path("/Users/johnherrick/.local/share/cascadia-r2/colima")
COLIMA_PROFILE = "cascadia-r2"
EXPERT_CYCLES = 10
EXPLORATION = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.02]
FINAL_PHASES = {"final_protected_comparison", "final_all_v3_evaluation", "complete"}


class CampaignError(ValueError):
    """Raised when a transition or artifact violates the V3 campaign contract."""


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{path} must contain a JSON object")
    return value


def _state_path(root: Path) -> Path:
    return root / "control" / "campaign-state.json"


def _read_state(root: Path) -> dict[str, Any]:
    state = _read_json(_state_path(root))
    if state.get("schema_id") != CAMPAIGN_SCHEMA or state.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("campaign state identity is invalid")
    protected_opened = state.get("protected_seed_values_opened")
    if protected_opened not in (True, False) or (
        protected_opened and state.get("phase") not in FINAL_PHASES
    ):
        raise CampaignError("protected seed state is inconsistent with the final gate")
    recorded_hash = state.get("state_sha256")
    if recorded_hash is not None:
        payload = dict(state)
        payload.pop("state_sha256", None)
        if recorded_hash != _sha256(payload):
            raise CampaignError("campaign state checksum is invalid")
    return state


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _storage(root: Path) -> dict[str, int | bool]:
    usage = shutil.disk_usage(root if root.exists() else root.parent)
    campaign_bytes = _tree_bytes(root)
    return {
        "campaign_bytes": campaign_bytes,
        "campaign_limit_bytes": MAX_BYTES,
        "free_bytes": usage.free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "within_campaign_limit": campaign_bytes <= MAX_BYTES,
        "free_space_preserved": usage.free >= MIN_FREE_BYTES,
    }


def _trim_sparse_worker_disk() -> bool:
    """Return freed Docker blocks to APFS without deleting live data."""

    if not COLIMA_BINARY.is_file():
        return False
    environment = dict(os.environ)
    environment["COLIMA_HOME"] = str(COLIMA_HOME)
    try:
        result = subprocess.run(
            [
                str(COLIMA_BINARY),
                "ssh",
                "-p",
                COLIMA_PROFILE,
                "--",
                "sudo",
                "fstrim",
                "-v",
                "/var/lib/docker",
            ],
            check=False,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _assert_storage(root: Path) -> dict[str, int | bool]:
    storage = _storage(root)
    if storage["within_campaign_limit"] and not storage["free_space_preserved"]:
        # Colima's sparse disk grows as Bacalhau workers materialize inputs.
        # Docker removes terminal containers, but APFS cannot see those freed
        # ext4 blocks until the guest trims them. Reclaim once, then enforce the
        # exact same invariant against a fresh physical measurement.
        _trim_sparse_worker_disk()
        storage = _storage(root)
    if not storage["within_campaign_limit"] or not storage["free_space_preserved"]:
        raise CampaignError(f"V3 storage guard refused mutation: {storage}")
    return storage


def assert_capacity_for_write(root: Path, planned_bytes: int) -> dict[str, int | bool]:
    if not isinstance(planned_bytes, int) or isinstance(planned_bytes, bool) or planned_bytes < 0:
        raise CampaignError("planned write size must be a nonnegative integer")
    storage = _storage(root)
    projected_campaign = int(storage["campaign_bytes"]) + planned_bytes
    projected_free = int(storage["free_bytes"]) - planned_bytes
    if projected_campaign > MAX_BYTES or projected_free < MIN_FREE_BYTES:
        raise CampaignError(
            "V3 storage guard refused projected write: "
            f"campaign={projected_campaign}, free={projected_free}, planned={planned_bytes}"
        )
    return storage | {
        "planned_bytes": planned_bytes,
        "projected_campaign_bytes": projected_campaign,
        "projected_free_bytes": projected_free,
    }


def _host(intent: str, detail: str) -> dict[str, object]:
    return {
        "intent": intent,
        "detail": detail,
        "generation_games_completed": 0,
        "generation_games_target": None,
        "generation_seed_prefix": None,
        "benchmark_pairs_completed": 0,
        "benchmark_pairs_total": None,
        "eta_seconds": None,
        "throughput_games_per_second": None,
        "rss_bytes": None,
        "swap_delta_bytes": 0,
    }


def _dashboard(state: dict[str, Any]) -> dict[str, object]:
    phase = state["phase"]
    awaiting = phase == "awaiting_phase2_approval"
    detail = "Part 1 complete — awaiting John" if awaiting else state.get("detail", phase)
    runtime = state.get("runtime", {})
    intents = runtime.get("host_intents", {})
    training = runtime.get("training", {})
    now = _now_ms()
    return {
        "schema_version": 1,
        "schema_id": V3_DASHBOARD_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "updated_unix_ms": now,
        "stale_after_seconds": 120,
        "phase": phase,
        "legal_next_transitions": state.get("legal_next_transitions", []),
        "round_index": state.get("round_index"),
        "models": {"incumbent": None, "candidate": None, "opponent_pool": []},
        "hosts": {
            "john1": _host(intents.get("john1", "idle"), detail),
            "john2": _host(intents.get("john2", "idle"), detail),
            "john3": _host(intents.get("john3", "idle"), detail),
            "john4": _host("idle", "Dashboard-visible; excluded from V3 compute"),
        },
        "training": {
            "active": bool(training.get("active", False)),
            "latest_verified_checkpoint": training.get("latest_verified_checkpoint"),
            "current_step": training.get("current_step"),
            "total_steps": training.get("total_steps"),
            "eta_seconds": training.get("eta_seconds"),
            "examples_per_second": training.get("examples_per_second"),
            "loss_samples": training.get("loss_samples", []),
        },
        "benchmark": {
            "active": False,
            "stage": None,
            "pairs_completed": 0,
            "pairs_total": None,
            "eta_seconds": None,
            "throughput_games_per_second": None,
            "peak_rss_bytes": None,
            "swap_delta_bytes": 0,
            "focal": None,
            "paired_delta": None,
            "classification": "pending",
            "scheduler_work_items": None,
        },
    }


def _write_state(root: Path, state: dict[str, Any], dashboard: Path) -> None:
    path = _state_path(root)
    previous = _read_json(path) if path.exists() else None
    previous_hash = None
    previous_sequence = -1
    if previous is not None:
        previous_payload = dict(previous)
        previous_hash = previous_payload.pop("state_sha256", None) or _sha256(previous_payload)
        previous_sequence = int(previous.get("transition_sequence", -1))
    state.pop("state_sha256", None)
    state["transition_sequence"] = previous_sequence + 1
    state["previous_state_sha256"] = previous_hash
    state["updated_unix_ms"] = _now_ms()
    state["state_sha256"] = _sha256(state)
    transition = root / "control" / "transitions" / f"{state['transition_sequence']:08d}.json"
    if transition.exists():
        raise CampaignError(f"immutable campaign transition already exists: {transition}")
    _write_json_atomic(transition, state)
    _write_json_atomic(path, state)
    _write_dashboard(root, dashboard, state)


def _write_dashboard(root: Path, dashboard: Path, state: dict[str, Any]) -> None:
    status_path = root / "control" / "dashboard-status.json"
    status = _dashboard(state)
    _overlay_live_progress(root, status)
    payload = json.dumps(status, indent=2, sort_keys=True) + "\n"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_name(f".{status_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, status_path)
    projection = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.dashboard-serving-projection.v2",
        "canonical_host": "john1",
        "canonical_path": str(status_path),
        "canonical_blake3": blake3.blake3(payload.encode()).hexdigest(),
        "canonical_updated_unix_ms": status["updated_unix_ms"],
        "fetched_unix_ms": _now_ms(),
        "canonical_payload": payload,
    }
    _write_json_atomic(dashboard, projection)


def _overlay_live_progress(root: Path, status: dict[str, Any]) -> None:
    """Project scheduler aggregates without mutating the immutable state chain."""

    phase = str(status.get("phase", ""))
    if phase == "bootstrap_training" or re.fullmatch(r"cycle-\d{2}-training", phase):
        _overlay_training_progress(root, status, phase)
        return
    progress_path: Path | None = None
    total_units = 0
    completed_before_increment = 0
    increment_units = 0
    unit_name = "items"
    if phase == "bootstrap_collecting":
        progress_path = root / "phase2/bootstrap/collection/progress.json"
        total_units = 500_000
        unit_name = "games"
    elif phase == "bootstrap_labeling":
        progress_path = root / "phase2/bootstrap/labeling/progress.json"
        total_units = 120_000
        unit_name = "roots"
    else:
        match = re.fullmatch(r"cycle-(\d{2})-(collecting|labeling|promotion)", phase)
        if match is not None:
            cycle = int(match.group(1))
            stage = match.group(2)
            directory = root / f"phase2/cycles/cycle-{cycle:02d}"
            if stage == "collecting":
                progress_path = directory / "collection/progress.json"
                total_units = 10_000
                unit_name = "games"
            elif stage == "labeling":
                progress_path = directory / "labeling/progress.json"
                total_units = 2_500
                unit_name = "roots"
            else:
                candidates = sorted((directory / "promotion").glob("progress-*.json"))
                progress_path = candidates[-1] if candidates else None
                if progress_path is not None:
                    try:
                        _, start_text, end_text = progress_path.stem.split("-")
                        start = int(start_text)
                        end = int(end_text)
                        if not 0 <= start < end:
                            raise ValueError("invalid promotion progress range")
                    except ValueError:
                        start = 0
                        end = 0
                    completed_before_increment = start * 4
                    increment_units = (end - start) * 4
                    total_units = end * 4
                    unit_name = "tier-pairs"
    if progress_path is None or not progress_path.is_file():
        return
    try:
        progress = _read_json(progress_path)
        work_items = int(progress["work_items"])
        terminal = int(progress["terminal_items"])
        elapsed = float(progress["elapsed_seconds"])
    except (CampaignError, KeyError, TypeError, ValueError):
        return
    if work_items <= 0 or not 0 <= terminal <= work_items or elapsed < 0:
        return
    if increment_units == 0:
        increment_units = total_units
    increment_completed = increment_units * terminal // work_items
    completed = completed_before_increment + increment_completed
    throughput = increment_completed / elapsed if elapsed > 0 else None
    eta = (
        (total_units - completed) / throughput
        if throughput is not None and throughput > 0
        else None
    )
    hosts = status["hosts"]
    # Bacalhau owns placement, so these thirds are explicitly a fleet-progress
    # projection rather than fabricated host attribution. Their sum is exact.
    completed_parts = [completed // 3] * 3
    target_parts = [total_units // 3] * 3
    for index in range(completed % 3):
        completed_parts[index] += 1
    for index in range(total_units % 3):
        target_parts[index] += 1
    detail = (
        f"Scheduler aggregate: {completed:,}/{total_units:,} {unit_name}; "
        f"{terminal}/{work_items} immutable work items"
    )
    for index, name in enumerate(("john1", "john2", "john3")):
        hosts[name].update(
            {
                "detail": detail,
                "generation_games_completed": completed_parts[index],
                "generation_games_target": target_parts[index],
                "eta_seconds": eta,
                "throughput_games_per_second": throughput,
            }
        )
    status["benchmark"]["scheduler_work_items"] = {
        "completed": terminal,
        "total": work_items,
        "states": progress.get("status_counts", {}),
        "retry_attempts": 0,
        "utilization": None,
    }


def _overlay_training_progress(
    root: Path,
    status: dict[str, Any],
    phase: str,
) -> None:
    if phase == "bootstrap_training":
        run_directories = sorted(
            (root / "phase2/bootstrap/training/calibration").glob("calibration-*")
        ) + sorted((root / "phase2/bootstrap/training/origins").glob("bootstrap-origin-*"))
        target = 120_000_000
    else:
        cycle = int(phase.split("-")[1])
        run_directories = sorted(
            (root / f"phase2/cycles/cycle-{cycle:02d}/training").glob("origin-*")
        )
        target = 2_400_000
    completed = 0
    latest_loss: Path | None = None
    loss_samples: list[dict[str, Any]] = []
    elapsed = 0.0
    latest_checkpoint = None
    for directory in run_directories:
        report_path = directory / "training-report.json"
        loss_path = directory / "loss.json"
        examples = 0
        run_elapsed = 0.0
        if report_path.is_file():
            try:
                report = _read_json(report_path)
                examples = int(report.get("examples_seen", 0))
                run_elapsed = float(report.get("elapsed_seconds", 0.0))
            except (CampaignError, TypeError, ValueError):
                examples = 0
        if loss_path.is_file():
            try:
                samples = _read_json(loss_path).get("samples", [])
                if samples:
                    examples = max(examples, int(samples[-1]["examples"]))
                    if (
                        latest_loss is None
                        or loss_path.stat().st_mtime > latest_loss.stat().st_mtime
                    ):
                        latest_loss = loss_path
                        loss_samples = [
                            {
                                "step": int(sample["examples"]),
                                "train_total": float(sample["loss"]),
                                "validation_total": None,
                            }
                            for sample in samples[-100:]
                        ]
                        evaluation_path = directory / "evaluation.json"
                        if evaluation_path.is_file():
                            evaluation = _read_json(evaluation_path)
                            validation = evaluation.get("validation", {})
                            value = validation.get("quantized_power_loss")
                            if value is not None:
                                loss_samples[-1]["validation_total"] = float(value)
            except (CampaignError, KeyError, OSError, TypeError, ValueError):
                pass
        latest_path = directory / "latest.json"
        evaluation_path = directory / "evaluation.json"
        evaluation_passed = False
        if evaluation_path.is_file():
            try:
                evaluation_passed = _read_json(evaluation_path).get("passed") is True
            except (CampaignError, OSError, TypeError, ValueError):
                evaluation_passed = False
        checkpoint_integrity_passed = False
        if latest_path.is_file():
            try:
                checkpoint_name = _read_json(latest_path)["checkpoint"]
                checkpoint_state = _read_json(
                    directory / "checkpoints" / checkpoint_name / "state.json"
                )
                checkpoint_examples = int(checkpoint_state.get("examples_seen", 0))
                integrity_path = (
                    directory
                    / "checkpoint-integrity"
                    / f"{checkpoint_examples:012d}.json"
                )
                if integrity_path.is_file():
                    checkpoint_integrity_passed = (
                        _read_json(integrity_path).get("passed") is True
                    )
                if not checkpoint_integrity_passed:
                    checkpoint_integrity_passed = _atomic_checkpoint_is_verified(
                        directory,
                        checkpoint_name,
                        checkpoint_state,
                    )
            except (CampaignError, KeyError, OSError, TypeError, ValueError):
                checkpoint_integrity_passed = False
        if latest_path.is_file() and (evaluation_passed or checkpoint_integrity_passed):
            try:
                checkpoint = _read_json(latest_path)["checkpoint"]
                state_path = directory / "checkpoints" / checkpoint / "state.json"
                trainer_state = _read_json(state_path)
                examples = max(examples, int(trainer_state.get("examples_seen", 0)))
                run_elapsed = max(
                    run_elapsed, float(trainer_state.get("elapsed_seconds", 0.0))
                )
                checkpoint_path = directory / "checkpoints" / checkpoint
                checkpoint_examples = int(trainer_state.get("examples_seen", 0))
                if examples > checkpoint_examples and not report_path.is_file():
                    run_elapsed = max(
                        run_elapsed,
                        float(trainer_state.get("elapsed_seconds", 0.0))
                        + max(
                            0.0,
                            time.time()
                            - (checkpoint_path / "checkpoint.json").stat().st_mtime,
                        ),
                    )
                manifest = _read_json(checkpoint_path / "checkpoint.json")
                model = manifest.get("files", {}).get("model.safetensors", {})
                digest = model.get("blake3")
                latest_checkpoint = {
                    "id": f"{directory.name}/{checkpoint}",
                    "blake3": digest if isinstance(digest, str) else None,
                }
            except (CampaignError, KeyError, OSError, TypeError, ValueError):
                pass
        completed += examples
        elapsed += run_elapsed
    completed = min(completed, target)
    rate = completed / elapsed if completed > 0 and elapsed > 0 else None
    eta = (target - completed) / rate if rate is not None and rate > 0 else None
    status["training"].update(
        {
            "active": completed < target,
            "latest_verified_checkpoint": latest_checkpoint,
            "current_step": completed,
            "total_steps": target,
            "eta_seconds": eta,
            "examples_per_second": rate,
            "loss_samples": loss_samples,
        }
    )
    detail = f"MLX training: {completed:,}/{target:,} scheduled exposures"
    status["hosts"]["john1"].update({"intent": "train", "detail": detail, "eta_seconds": eta})
    for host in ("john2", "john3"):
        status["hosts"][host].update(
            {"intent": "benchmark", "detail": "Benchmarking previous frozen checkpoint"}
        )


def _atomic_checkpoint_is_verified(
    run_directory: Path,
    checkpoint_name: str,
    state: dict[str, Any],
) -> bool:
    """Validate a complete atomic expert-cycle checkpoint without rereading 1.2 GiB.

    Checkpoint publication computes BLAKE3 while writing and exposes the
    manifest only after every model, optimizer, and cursor file is atomically
    installed. The dashboard revalidates that frozen manifest, all file sizes,
    the clean pass-boundary cursor, and the run-manifest binding. Full content
    hashes are rechecked by the resume/evaluation loaders before scientific use.
    """

    checkpoint = run_directory / "checkpoints" / checkpoint_name
    try:
        manifest = _read_json(checkpoint / "checkpoint.json")
        run_manifest = _read_json(run_directory / "run-manifest.json")
        files = manifest["files"]
        metadata = manifest["metadata"]
        required = {"model.safetensors", "optimizer.safetensors", "state.json"}
        if (
            manifest.get("schema_version") != 1
            or manifest.get("checkpoint_id") != checkpoint_name
            or set(files) != required
            or metadata.get("run_manifest_blake3") != run_manifest.get("canonical_blake3")
            or int(metadata.get("examples_seen", -1)) != int(state.get("examples_seen", -2))
            or int(metadata.get("completed_pass", -1)) != int(state.get("schedule_block", -2))
            or int(state.get("batch_in_block", -1)) != 0
            or int(state.get("batch_in_epoch", -1)) != 0
        ):
            return False
        for name in required:
            identity = files[name]
            path = checkpoint / name
            if (
                not path.is_file()
                or not isinstance(identity, dict)
                or int(identity.get("bytes", -1)) != path.stat().st_size
                or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("blake3", "")))
            ):
                return False
    except (CampaignError, KeyError, OSError, TypeError, ValueError):
        return False
    return True


def refresh_dashboard(root: Path, dashboard: Path) -> dict[str, Any]:
    state = _read_state(root)
    _write_dashboard(root, dashboard, state)
    return {
        "campaign_id": state["campaign_id"],
        "phase": state["phase"],
        "dashboard": str(dashboard),
        "refreshed_unix_ms": _now_ms(),
    }


def set_part1_activity(
    root: Path,
    dashboard: Path,
    *,
    phase: str,
    detail: str,
    john1_intent: str,
    john2_intent: str,
    john3_intent: str,
    training_active: bool,
    current_step: int | None,
    total_steps: int | None,
    examples_per_second: float | None,
    eta_seconds: float | None,
) -> dict[str, Any]:
    state = _read_state(root)
    if state.get("part") != 1 or state.get("phase") == "awaiting_phase2_approval":
        raise CampaignError("Part 1 activity cannot modify a sealed or Phase 2 campaign")
    allowed_intents = {"idle", "control", "generate", "validate", "train", "benchmark"}
    intents = {
        "john1": john1_intent,
        "john2": john2_intent,
        "john3": john3_intent,
    }
    if any(value not in allowed_intents for value in intents.values()):
        raise CampaignError("Part 1 host intent is invalid")
    if training_active and john1_intent != "train":
        raise CampaignError("active MLX training requires john1 intent=train")
    if (current_step is None) != (total_steps is None):
        raise CampaignError("training step and total must be supplied together")
    if current_step is not None and not 0 <= current_step <= total_steps:
        raise CampaignError("training progress is invalid")
    state.update({"phase": phase, "detail": detail})
    state["runtime"] = {
        "host_intents": intents,
        "training": {
            "active": training_active,
            "latest_verified_checkpoint": None,
            "current_step": current_step,
            "total_steps": total_steps,
            "eta_seconds": eta_seconds,
            "examples_per_second": examples_per_second,
            "loss_samples": [],
        },
    }
    _write_state(root, state, dashboard)
    return state


def initialize(root: Path, dashboard: Path) -> dict[str, Any]:
    if root != DEFAULT_ROOT:
        raise CampaignError(f"V3 campaign root is frozen at {DEFAULT_ROOT}")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("control", "engineering", "profiles", "smoke", "models", "reports"):
        (root / name).mkdir(exist_ok=True)
    (root / "control" / "transitions").mkdir(exist_ok=True)
    storage = _assert_storage(root)
    path = _state_path(root)
    if path.exists():
        state = _read_state(root)
        _write_dashboard(root, dashboard, state)
        return state
    state = {
        "schema_id": CAMPAIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "part1_engineering",
        "detail": "Building and profiling the V3 engineering system",
        "part": 1,
        "round_index": None,
        "legal_next_transitions": ["record_part1_evidence"],
        "phase2_authorized": False,
        "protected_seed_values_opened": False,
        "scientific_training_started": False,
        "john4_compute_authorized": False,
        "topology": {
            "code_authority": "john1",
            "cpu_workers": {"john1": 9, "john2": 10, "john3": 10},
            "john4": "dashboard-only",
            "scheduler": "bacalhau",
            "manual_pair_sharding": False,
            "subagents": False,
        },
        "storage": storage,
        "runtime": {
            "host_intents": {"john1": "control", "john2": "idle", "john3": "idle"},
            "training": {"active": False, "loss_samples": []},
        },
        "created_unix_ms": _now_ms(),
    }
    _write_state(root, state, dashboard)
    return state


def _profile_gate(first: dict[str, Any], second: dict[str, Any], metric: str) -> tuple[bool, float]:
    baseline = float(first[metric])
    optimized = float(second[metric])
    ratio = optimized / baseline if baseline > 0 else 0.0
    return ratio >= 1.5, ratio


def qualify(
    root: Path,
    dashboard: Path,
    *,
    feature_manifest: Path,
    model_manifest: Path,
    mlx_profile_1: Path,
    mlx_profile_2: Path,
    game_profile_1: Path,
    game_profile_2: Path,
    game_profile_2_late: Path,
    game_profile_2_overflow: Path,
    engineering_corpus: Path,
    training_smoke: Path,
    direct_smoke: Path,
    r600_smoke: Path,
    parity: Path,
    docker_receipt: Path,
    recovery_receipt: Path,
    infrastructure_receipt: Path,
    capacity_projection: Path,
) -> dict[str, Any]:
    state = _read_state(root)
    if state["part"] != 1 or state["phase"] == "phase2_authorized":
        raise CampaignError("Part 1 qualification is not legal from the current state")
    storage = _assert_storage(root)
    artifacts = {
        name: _read_json(path)
        for name, path in {
            "feature_manifest": feature_manifest,
            "model_manifest": model_manifest,
            "mlx_profile_1": mlx_profile_1,
            "mlx_profile_2": mlx_profile_2,
            "game_profile_1": game_profile_1,
            "game_profile_2": game_profile_2,
            "game_profile_2_late": game_profile_2_late,
            "game_profile_2_overflow": game_profile_2_overflow,
            "engineering_corpus": engineering_corpus,
            "training_smoke": training_smoke,
            "direct_smoke": direct_smoke,
            "r600_smoke": r600_smoke,
            "parity": parity,
            "docker_receipt": docker_receipt,
            "recovery_receipt": recovery_receipt,
            "infrastructure_receipt": infrastructure_receipt,
            "capacity_projection": capacity_projection,
        }.items()
    }
    mlx_ok, mlx_speedup = _profile_gate(
        artifacts["mlx_profile_1"], artifacts["mlx_profile_2"], "examples_per_second"
    )
    game_ok, game_speedup = _profile_gate(
        artifacts["game_profile_1"], artifacts["game_profile_2"], "decisions_per_second"
    )
    hot_fraction = float(artifacts["engineering_corpus"]["hot_path_fraction"])
    peak_memory = int(artifacts["mlx_profile_2"]["peak_memory_bytes"])
    physical_memory = int(artifacts["mlx_profile_2"]["physical_memory_bytes"])
    projected_seconds = float(artifacts["capacity_projection"]["active_wall_seconds"])
    infrastructure = artifacts["infrastructure_receipt"]
    training_state = artifacts["training_smoke"].get("state", {})
    training_config = artifacts["training_smoke"].get("training_config", {})
    gates = {
        "engineering_corpus_exact_and_excluded": artifacts["engineering_corpus"].get("games")
        == 2_000
        and artifacts["engineering_corpus"].get("records") == 160_000
        and artifacts["engineering_corpus"].get("scientific_eligible") is False,
        "one_complete_160k_epoch": training_config.get("examples") == 160_000
        and training_state.get("epoch") == 1
        and artifacts["training_smoke"].get("metrics", {}).get("interrupted") is False,
        "direct_smoke_100_games": artifacts["direct_smoke"].get("games") == 100,
        "r600_smoke_8_games": artifacts["r600_smoke"].get("games") == 8,
        "radius7_hot_path_at_least_99_9_percent": hot_fraction >= 0.999,
        "overflow_exact": bool(artifacts["parity"].get("overflow_exact", False)),
        "mlx_speedup_at_least_1_5x": mlx_ok,
        "gameplay_speedup_at_least_1_5x": game_ok,
        "representative_late_and_overflow_profiled": (
            all(artifacts["game_profile_2_late"].get("radius7_hot_path", []))
            and bool(artifacts["game_profile_2_overflow"].get("radius7_hot_path"))
            and not any(artifacts["game_profile_2_overflow"]["radius7_hot_path"])
            and min(artifacts["game_profile_2_overflow"].get("overflow_entities", [0])) > 0
        ),
        "r600_at_most_45_seconds_per_game": float(
            artifacts["r600_smoke"].get("r600_seconds_per_game", float("inf"))
        )
        <= 45.0,
        "no_swap_growth": int(artifacts["mlx_profile_2"].get("swap_delta_bytes", -1)) == 0
        and int(artifacts["direct_smoke"].get("swap_delta_bytes", -1)) == 0
        and int(artifacts["r600_smoke"].get("swap_delta_bytes", -1)) == 0,
        "mlx_peak_below_70_percent": peak_memory <= physical_memory * 0.70,
        "part2_projected_at_most_6_5_days": projected_seconds <= 6.5 * 86400,
        "rust_mlx_quantized_bit_identity": bool(
            artifacts["parity"].get("rust_mlx_quantized_bit_identical", False)
        ),
        "float_quantized_top32_at_least_99_9_percent": float(
            artifacts["parity"].get("float_quantized_top32_agreement", 0.0)
        )
        >= 0.999,
        "docker_identity": bool(artifacts["docker_receipt"].get("passed", False)),
        "checkpoint_exact_continuation": bool(
            artifacts["recovery_receipt"].get("checkpoint_exact_continuation", False)
        ),
        "worker_failure_retry": bool(infrastructure.get("bacalhau_worker_retry", False)),
        "trainer_restart": bool(infrastructure.get("john1_trainer_restart", False)),
        "dashboard_john1_through_john4": infrastructure.get("dashboard_hosts")
        == ["john1", "john2", "john3", "john4"],
        "resource_and_shutdown_tests": all(
            bool(infrastructure.get(name, False))
            for name in ("disk_limit", "memory_limit", "clean_shutdown")
        ),
        "part2_storage_projected_at_most_40_gib": int(
            artifacts["capacity_projection"].get("projected_campaign_bytes", MAX_BYTES + 1)
        )
        <= MAX_BYTES,
        "protected_domain_isolated": (
            artifacts["capacity_projection"].get("protected_seed_values_opened") is False
            and infrastructure.get("scientific_data_guard") is True
        ),
        "storage": bool(storage["within_campaign_limit"] and storage["free_space_preserved"]),
    }
    readiness = {
        "schema_id": READINESS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "green" if all(gates.values()) else "red",
        "phase": "awaiting_phase2_approval",
        "gates": gates,
        "measurements": {
            "radius7_hot_path_fraction": hot_fraction,
            "mlx_speedup": mlx_speedup,
            "gameplay_speedup": game_speedup,
            "late_game_decisions_per_second": artifacts["game_profile_2_late"].get(
                "decisions_per_second"
            ),
            "overflow_decisions_per_second": artifacts["game_profile_2_overflow"].get(
                "decisions_per_second"
            ),
            "projected_part2_seconds": projected_seconds,
            "storage": storage,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in {
                "feature_manifest": feature_manifest,
                "model_manifest": model_manifest,
                "mlx_profile_1": mlx_profile_1,
                "mlx_profile_2": mlx_profile_2,
            "game_profile_1": game_profile_1,
            "game_profile_2": game_profile_2,
            "game_profile_2_late": game_profile_2_late,
            "game_profile_2_overflow": game_profile_2_overflow,
            "engineering_corpus": engineering_corpus,
            "training_smoke": training_smoke,
            "direct_smoke": direct_smoke,
            "r600_smoke": r600_smoke,
            "parity": parity,
            "docker_receipt": docker_receipt,
            "recovery_receipt": recovery_receipt,
            "infrastructure_receipt": infrastructure_receipt,
            "capacity_projection": capacity_projection,
            }.items()
        },
        "protected_seed_values_opened": False,
        "scientific_training_started": False,
        "created_unix_ms": _now_ms(),
    }
    readiness["readiness_sha256"] = _sha256(readiness)
    readiness_path = root / "reports" / "part1-readiness.json"
    _write_json_atomic(readiness_path, readiness)
    state.update(
        {
            "phase": "awaiting_phase2_approval",
            "detail": "Part 1 complete — awaiting John",
            "part": 1,
            "legal_next_transitions": ["authorize_phase2"],
            "readiness_path": str(readiness_path),
            "readiness_sha256": readiness["readiness_sha256"],
            "readiness_status": readiness["status"],
            "storage": storage,
            "runtime": {
                "host_intents": {"john1": "idle", "john2": "idle", "john3": "idle"},
                "training": {"active": False, "loss_samples": []},
            },
        }
    )
    _write_state(root, state, dashboard)
    return readiness


def authorize_phase2(
    root: Path,
    dashboard: Path,
    checksum: str,
    approved_by: str,
    accept_red_readiness: bool,
) -> dict[str, Any]:
    state = _read_state(root)
    if state.get("phase") != "awaiting_phase2_approval":
        raise CampaignError("Phase 2 authorization requires the mandatory Part 1 stop")
    readiness = _read_json(Path(state["readiness_path"]))
    expected = readiness.get("readiness_sha256")
    if checksum != expected or checksum != state.get("readiness_sha256"):
        raise CampaignError("Phase 2 approval checksum does not match readiness")
    check = dict(readiness)
    check.pop("readiness_sha256", None)
    if _sha256(check) != expected:
        raise CampaignError("readiness manifest checksum is internally invalid")
    if readiness.get("status") != "green" and not accept_red_readiness:
        raise CampaignError("red readiness requires an explicit accept-red-readiness override")
    if not approved_by.strip():
        raise CampaignError("approved-by must identify the approving user")
    state.update(
        {
            "phase": "phase2_authorized",
            "detail": "Phase 2 authorized; bootstrap may be scheduled",
            "part": 2,
            "legal_next_transitions": ["bootstrap_collecting"],
            "phase2_authorized": True,
            "approved_by": approved_by,
            "approved_unix_ms": _now_ms(),
            "approved_readiness_sha256": checksum,
        }
    )
    _write_state(root, state, dashboard)
    return state


def phase2_plan(root: Path) -> dict[str, Any]:
    state = _read_state(root)
    if state.get("phase2_authorized") is not True:
        raise CampaignError("Phase 2 plan is sealed until checksum-bound user approval")
    return {
        "schema_id": "cascadia-v3-phase2-plan-v2",
        "bootstrap_games": 500_000,
        "bootstrap_teacher_roots": 100_000,
        "bootstrap_validation_roots": 20_000,
        "cycles": [
            {
                "cycle": cycle,
                "games": 10_000,
                "teacher_roots": 2_500,
                "training_origins": 2,
                "exploration_epsilon": EXPLORATION[cycle - 1],
                "newest_model_seats_per_game": 1,
                "opponent_mix": {"qualified_v1": 0.8, "prior_v3": 0.2},
            }
            for cycle in range(1, EXPERT_CYCLES + 1)
        ],
        "john4_compute_authorized": False,
        "protected_final_opened": False,
    }


def _phase2_successor(phase: str) -> str | None:
    fixed = {
        "phase2_authorized": "bootstrap_collecting",
        "bootstrap_collecting": "bootstrap_labeling",
        "bootstrap_labeling": "bootstrap_training",
        "bootstrap_training": "cycle-01-collecting",
        "final_protected_comparison": "final_all_v3_evaluation",
        "final_all_v3_evaluation": "complete",
        "complete": None,
    }
    if phase in fixed:
        return fixed[phase]
    match = re.fullmatch(r"cycle-(\d{2})-(collecting|labeling|training|promotion)", phase)
    if match is None:
        raise CampaignError(f"unknown V3 Phase 2 state: {phase}")
    cycle = int(match.group(1))
    stage = match.group(2)
    if not 1 <= cycle <= EXPERT_CYCLES:
        raise CampaignError("expert cycle state is outside 1..=10")
    if stage == "collecting":
        return f"cycle-{cycle:02d}-labeling"
    if stage == "labeling":
        return f"cycle-{cycle:02d}-training"
    if stage == "training":
        return f"cycle-{cycle:02d}-promotion"
    return (
        f"cycle-{cycle + 1:02d}-collecting"
        if cycle < EXPERT_CYCLES
        else "final_protected_comparison"
    )


def _phase2_runtime(phase: str) -> dict[str, Any]:
    if phase.endswith("collecting"):
        intents = {"john1": "generate", "john2": "generate", "john3": "generate"}
        active = False
    elif phase.endswith("labeling"):
        intents = {"john1": "validate", "john2": "validate", "john3": "validate"}
        active = False
    elif phase.endswith("training") or phase == "bootstrap_training":
        intents = {"john1": "train", "john2": "benchmark", "john3": "benchmark"}
        active = True
    elif phase.endswith("promotion") or phase in FINAL_PHASES:
        intents = {"john1": "control", "john2": "benchmark", "john3": "benchmark"}
        active = False
    else:
        intents = {"john1": "control", "john2": "idle", "john3": "idle"}
        active = False
    return {"host_intents": intents, "training": {"active": active, "loss_samples": []}}


def advance_phase2(
    root: Path,
    dashboard: Path,
    *,
    destination: str,
    evidence: Path,
    evidence_sha256: str,
) -> dict[str, Any]:
    state = _read_state(root)
    if state.get("phase2_authorized") is not True or state.get("part") != 2:
        raise CampaignError("Phase 2 transition is sealed until checksum-bound approval")
    expected = _phase2_successor(str(state["phase"]))
    if destination != expected:
        raise CampaignError(f"illegal Phase 2 transition; expected {expected!r}")
    if not evidence.is_file():
        raise CampaignError("Phase 2 transition evidence is missing")
    observed = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if observed != evidence_sha256:
        raise CampaignError("Phase 2 transition evidence checksum differs")
    evidence_value = _read_json(evidence)
    if evidence_value.get("passed") is not True:
        raise CampaignError("Phase 2 transition evidence is not passing")
    storage = _assert_storage(root)
    state["phase"] = destination
    state["detail"] = destination.replace("-", " ")
    state["round_index"] = (
        int(destination.split("-")[1]) if destination.startswith("cycle-") else None
    )
    state["protected_seed_values_opened"] = destination in FINAL_PHASES
    state["scientific_training_started"] = state.get("scientific_training_started", False) or (
        destination == "bootstrap_training" or destination.endswith("-training")
    )
    successor = _phase2_successor(destination)
    state["legal_next_transitions"] = [successor] if successor is not None else []
    state["last_transition_evidence"] = {
        "path": str(evidence),
        "sha256": observed,
        "schema_id": evidence_value.get("schema_id"),
    }
    state["storage"] = storage
    state["runtime"] = _phase2_runtime(destination)
    _write_state(root, state, dashboard)
    return state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("status")
    subparsers.add_parser("refresh-dashboard")
    activity = subparsers.add_parser("set-part1-activity")
    activity.add_argument("--phase", required=True)
    activity.add_argument("--detail", required=True)
    activity.add_argument("--john1-intent", default="control")
    activity.add_argument("--john2-intent", default="idle")
    activity.add_argument("--john3-intent", default="idle")
    activity.add_argument("--training-active", action="store_true")
    activity.add_argument("--current-step", type=int)
    activity.add_argument("--total-steps", type=int)
    activity.add_argument("--examples-per-second", type=float)
    activity.add_argument("--eta-seconds", type=float)
    qualify_parser = subparsers.add_parser("qualify-part1")
    for name in (
        "feature-manifest",
        "model-manifest",
        "mlx-profile-1",
        "mlx-profile-2",
        "game-profile-1",
        "game-profile-2",
        "game-profile-2-late",
        "game-profile-2-overflow",
        "engineering-corpus",
        "training-smoke",
        "direct-smoke",
        "r600-smoke",
        "parity",
        "docker-receipt",
        "recovery-receipt",
        "infrastructure-receipt",
        "capacity-projection",
    ):
        qualify_parser.add_argument(f"--{name}", type=Path, required=True)
    authorize = subparsers.add_parser("authorize-phase2")
    authorize.add_argument("--readiness-sha256", required=True)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--accept-red-readiness", action="store_true")
    subparsers.add_parser("phase2-plan")
    advance = subparsers.add_parser("advance-phase2")
    advance.add_argument("--to", required=True)
    advance.add_argument("--evidence", type=Path, required=True)
    advance.add_argument("--evidence-sha256", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "init":
            result = initialize(args.root, args.dashboard)
        elif args.command == "status":
            result = _read_state(args.root)
        elif args.command == "refresh-dashboard":
            result = refresh_dashboard(args.root, args.dashboard)
        elif args.command == "set-part1-activity":
            result = set_part1_activity(
                args.root,
                args.dashboard,
                phase=args.phase,
                detail=args.detail,
                john1_intent=args.john1_intent,
                john2_intent=args.john2_intent,
                john3_intent=args.john3_intent,
                training_active=args.training_active,
                current_step=args.current_step,
                total_steps=args.total_steps,
                examples_per_second=args.examples_per_second,
                eta_seconds=args.eta_seconds,
            )
        elif args.command == "qualify-part1":
            result = qualify(
                args.root,
                args.dashboard,
                feature_manifest=args.feature_manifest,
                model_manifest=args.model_manifest,
                mlx_profile_1=args.mlx_profile_1,
                mlx_profile_2=args.mlx_profile_2,
                game_profile_1=args.game_profile_1,
                game_profile_2=args.game_profile_2,
                game_profile_2_late=args.game_profile_2_late,
                game_profile_2_overflow=args.game_profile_2_overflow,
                engineering_corpus=args.engineering_corpus,
                training_smoke=args.training_smoke,
                direct_smoke=args.direct_smoke,
                r600_smoke=args.r600_smoke,
                parity=args.parity,
                docker_receipt=args.docker_receipt,
                recovery_receipt=args.recovery_receipt,
                infrastructure_receipt=args.infrastructure_receipt,
                capacity_projection=args.capacity_projection,
            )
        elif args.command == "authorize-phase2":
            result = authorize_phase2(
                args.root,
                args.dashboard,
                args.readiness_sha256,
                args.approved_by,
                args.accept_red_readiness,
            )
        elif args.command == "phase2-plan":
            result = phase2_plan(args.root)
        elif args.command == "advance-phase2":
            result = advance_phase2(
                args.root,
                args.dashboard,
                destination=args.to,
                evidence=args.evidence,
                evidence_sha256=args.evidence_sha256,
            )
        else:
            raise AssertionError(args.command)
    except CampaignError as error:
        raise SystemExit(f"V3 campaign refused: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
