"""Bound V3 MLX checkpoint storage without weakening active recovery guarantees."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


class CheckpointLifecycleError(ValueError):
    """A checkpoint set is incomplete or unsafe to compact or retire."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    training = _read(run_dir / "training-report.json")
    evaluation = _read(run_dir / "evaluation.json")
    latest = _read(run_dir / "latest.json").get("checkpoint")
    checkpoint = run_dir / "checkpoints" / str(latest)
    if (
        training.get("passed") is not True
        or evaluation.get("passed") is not True
        or not isinstance(latest, str)
        or not (checkpoint / "checkpoint.json").is_file()
        or not (run_dir / "serving/model.json").is_file()
    ):
        raise CheckpointLifecycleError(f"run is not safe to compact: {run_dir}")
    return training, evaluation, checkpoint


def compact_completed_run(run_dir: Path) -> dict[str, Any]:
    """Keep one exact-resume checkpoint for a completed, evaluated active run."""
    receipt = run_dir / "checkpoint-compaction.json"
    if receipt.is_file() and _read(receipt).get("passed") is True:
        return _read(receipt)
    _, _, retained = _completed(run_dir)
    before = _tree_bytes(run_dir / "checkpoints") + _tree_bytes(run_dir / "swa")
    inventory = []
    for checkpoint in sorted((run_dir / "checkpoints").iterdir()):
        manifest = checkpoint / "checkpoint.json"
        if checkpoint.is_dir() and manifest.is_file():
            inventory.append(
                {
                    "checkpoint": checkpoint.name,
                    "bytes": _tree_bytes(checkpoint),
                    "manifest_sha256": _sha256(manifest),
                }
            )
            if checkpoint != retained:
                shutil.rmtree(checkpoint)
    shutil.rmtree(run_dir / "swa", ignore_errors=True)
    after = _tree_bytes(run_dir / "checkpoints")
    value = {
        "schema_id": "cascadia-v3-checkpoint-compaction-v1",
        "passed": True,
        "run_dir": str(run_dir.resolve()),
        "retained_checkpoint": retained.name,
        "retained_checkpoint_manifest_sha256": _sha256(retained / "checkpoint.json"),
        "inventory_before": inventory,
        "bytes_before": before,
        "bytes_after": after,
        "bytes_reclaimed": before - after,
        "recovery_guarantee": "latest-complete-exact-resume",
    }
    _write_atomic(receipt, value)
    return value


def retire_completed_run(run_dir: Path, *, reason: str) -> dict[str, Any]:
    """Retire optimizer state after immutable evaluation and serving export exist."""
    receipt = run_dir / "checkpoint-retirement.json"
    if receipt.is_file() and _read(receipt).get("passed") is True:
        value = _read(receipt)
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        shutil.rmtree(run_dir / "swa", ignore_errors=True)
        (run_dir / "latest.json").unlink(missing_ok=True)
        return value
    if not reason:
        raise CheckpointLifecycleError("checkpoint retirement reason is empty")
    _, _, latest = _completed(run_dir)
    inventory = []
    for checkpoint in sorted((run_dir / "checkpoints").iterdir()):
        manifest = checkpoint / "checkpoint.json"
        if checkpoint.is_dir() and manifest.is_file():
            inventory.append(
                {
                    "checkpoint": checkpoint.name,
                    "bytes": _tree_bytes(checkpoint),
                    "manifest_sha256": _sha256(manifest),
                }
            )
    swa_bytes = _tree_bytes(run_dir / "swa")
    retired_bytes = sum(item["bytes"] for item in inventory) + swa_bytes
    value = {
        "schema_id": "cascadia-v3-checkpoint-retirement-v1",
        "passed": True,
        "run_dir": str(run_dir.resolve()),
        "reason": reason,
        "latest_checkpoint": latest.name,
        "checkpoint_inventory": inventory,
        "swa_bytes": swa_bytes,
        "bytes_reclaimed": retired_bytes,
        "training_report_sha256": _sha256(run_dir / "training-report.json"),
        "evaluation_sha256": _sha256(run_dir / "evaluation.json"),
        "serving_manifest_sha256": _sha256(run_dir / "serving/model.json"),
        "scientific_outputs_preserved": True,
        "exact_resume_retired": True,
    }
    _write_atomic(receipt, value)
    shutil.rmtree(run_dir / "checkpoints")
    shutil.rmtree(run_dir / "swa", ignore_errors=True)
    (run_dir / "latest.json").unlink()
    return value


def retire_rejected_run(run_dir: Path, *, reason: str) -> dict[str, Any]:
    """Retire optimizer state for an immutably evaluated, rejected model."""
    receipt = run_dir / "checkpoint-retirement.json"
    if receipt.is_file() and _read(receipt).get("passed") is True:
        value = _read(receipt)
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        shutil.rmtree(run_dir / "swa", ignore_errors=True)
        (run_dir / "latest.json").unlink(missing_ok=True)
        return value
    if not reason:
        raise CheckpointLifecycleError("checkpoint retirement reason is empty")
    training = _read(run_dir / "training-report.json")
    failure_path = run_dir / "evaluation-failure.json"
    failure = _read(failure_path)
    latest_name = _read(run_dir / "latest.json").get("checkpoint")
    latest = run_dir / "checkpoints" / str(latest_name)
    serving = run_dir / "serving/model.json"
    if (
        training.get("passed") is not True
        or failure.get("passed") is not False
        or not isinstance(latest_name, str)
        or not (latest / "checkpoint.json").is_file()
        or not serving.is_file()
    ):
        raise CheckpointLifecycleError(f"rejected run is not safe to retire: {run_dir}")
    inventory = []
    for checkpoint in sorted((run_dir / "checkpoints").iterdir()):
        manifest = checkpoint / "checkpoint.json"
        if checkpoint.is_dir() and manifest.is_file():
            inventory.append(
                {
                    "checkpoint": checkpoint.name,
                    "bytes": _tree_bytes(checkpoint),
                    "manifest_sha256": _sha256(manifest),
                }
            )
    swa_bytes = _tree_bytes(run_dir / "swa")
    value = {
        "schema_id": "cascadia-v3-rejected-checkpoint-retirement-v1",
        "passed": True,
        "run_dir": str(run_dir.resolve()),
        "reason": reason,
        "latest_checkpoint": latest.name,
        "checkpoint_inventory": inventory,
        "swa_bytes": swa_bytes,
        "bytes_reclaimed": sum(item["bytes"] for item in inventory) + swa_bytes,
        "training_report_sha256": _sha256(run_dir / "training-report.json"),
        "evaluation_failure_sha256": _sha256(failure_path),
        "serving_manifest_sha256": _sha256(serving),
        "scientific_outputs_preserved": True,
        "exact_resume_retired": True,
        "evaluation_passed": False,
    }
    _write_atomic(receipt, value)
    shutil.rmtree(run_dir / "checkpoints")
    shutil.rmtree(run_dir / "swa", ignore_errors=True)
    (run_dir / "latest.json").unlink()
    return value
