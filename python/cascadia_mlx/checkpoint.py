"""Atomic, checksummed MLX checkpoints with exact optimizer resumption."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from cascadia_mlx.model import EntitySetValueModel, ModelConfig

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, incompatible, or corrupt."""


@dataclass
class TrainerState:
    """The exact next batch to execute."""

    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    elapsed_seconds: float = 0.0
    best_validation_mae: float | None = None
    best_ranking_loss: float | None = None
    best_top1_accuracy: float | None = None
    ranking_epochs_without_improvement: int = 0
    best_validation_loss: float | None = None
    best_validation_rmse: float | None = None
    value_epochs_without_improvement: int = 0


def save_checkpoint(
    run_dir: str | Path,
    model: Any,
    optimizer: optim.Optimizer,
    state: TrainerState,
) -> Path:
    """Atomically save weights, optimizer state, and trainer cursor."""
    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_id = (
        f"step-{state.global_step:09d}-epoch-{state.epoch:04d}-batch-{state.batch_in_epoch:06d}"
    )
    final_path = checkpoints / checkpoint_id
    temp_path = checkpoints / f".{checkpoint_id}.{uuid.uuid4().hex}.tmp"
    temp_path.mkdir()

    try:
        model_path = temp_path / "model.safetensors"
        optimizer_path = temp_path / "optimizer.safetensors"
        state_path = temp_path / "state.json"
        model.save_weights(str(model_path))
        mx.save_safetensors(
            optimizer_path,
            dict(tree_flatten(optimizer.state)),
            metadata={"schema_version": str(CHECKPOINT_SCHEMA_VERSION)},
        )
        _write_json(state_path, asdict(state))
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "model_config": model.config.to_dict(),
            "files": {
                path.name: {"blake3": _checksum(path), "bytes": path.stat().st_size}
                for path in (model_path, optimizer_path, state_path)
            },
        }
        _write_json(temp_path / "checkpoint.json", manifest)
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(temp_path, final_path)
        _write_json_atomic(run_dir / "latest.json", {"checkpoint": checkpoint_id})
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return final_path


def set_checkpoint_pointer(
    run_dir: str | Path,
    name: str,
    checkpoint: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically update a named pointer such as ``best``."""
    if not name.isidentifier():
        raise ValueError("checkpoint pointer name must be an identifier")
    run_dir = Path(run_dir)
    expected_parent = (run_dir / "checkpoints").resolve()
    if checkpoint.resolve().parent != expected_parent:
        raise ValueError("checkpoint pointer must target this run")
    value = {"checkpoint": checkpoint.name}
    if metadata:
        value.update(metadata)
    _write_json_atomic(run_dir / f"{name}.json", value)


def prune_checkpoints(run_dir: str | Path, *, keep_recent: int = 2) -> None:
    """Bound disk use while preserving exact latest, best, and recent recovery points."""
    if keep_recent < 0:
        raise ValueError("keep_recent cannot be negative")
    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir():
        return
    complete = sorted(
        path
        for path in checkpoints.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "checkpoint.json").is_file()
    )
    keep = set(complete[-keep_recent:]) if keep_recent else set()
    for pointer in ("latest", "best"):
        path = run_dir / f"{pointer}.json"
        if not path.is_file():
            continue
        try:
            checkpoint_name = json.loads(path.read_text())["checkpoint"]
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise CheckpointError(
                f"cannot prune around invalid {pointer} pointer: {error}"
            ) from error
        checkpoint = checkpoints / checkpoint_name
        if not (checkpoint / "checkpoint.json").is_file():
            raise CheckpointError(f"{pointer} pointer targets an incomplete checkpoint")
        keep.add(checkpoint)
    for checkpoint in complete:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)


def load_latest_checkpoint(
    run_dir: str | Path,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[EntitySetValueModel, optim.AdamW, TrainerState, Path]:
    """Load and verify the latest complete checkpoint."""
    model, optimizer, state, checkpoint_path = load_latest_checkpoint_with_factory(
        run_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_factory=lambda values: EntitySetValueModel(ModelConfig.from_dict(values)),
    )
    return model, optimizer, state, checkpoint_path


def load_latest_checkpoint_with_factory(
    run_dir: str | Path,
    *,
    learning_rate: float,
    weight_decay: float,
    model_factory: Callable[[dict[str, object]], Any],
) -> tuple[Any, optim.AdamW, TrainerState, Path]:
    """Load a verified checkpoint using its architecture-specific model factory."""
    return load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="latest",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_factory=model_factory,
    )


def load_checkpoint_pointer_with_factory(
    run_dir: str | Path,
    *,
    pointer: str,
    learning_rate: float,
    weight_decay: float,
    model_factory: Callable[[dict[str, object]], Any],
) -> tuple[Any, optim.AdamW, TrainerState, Path]:
    """Load a verified named checkpoint pointer such as ``latest`` or ``best``."""
    if not pointer.isidentifier():
        raise ValueError("checkpoint pointer name must be an identifier")
    run_dir = Path(run_dir)
    try:
        selected = json.loads((run_dir / f"{pointer}.json").read_text())
        checkpoint_path = run_dir / "checkpoints" / selected["checkpoint"]
        manifest = json.loads((checkpoint_path / "checkpoint.json").read_text())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read {pointer} checkpoint: {error}") from error
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint schema")
    for name, expected in manifest.get("files", {}).items():
        path = checkpoint_path / name
        try:
            if path.stat().st_size != expected["bytes"] or _checksum(path) != expected["blake3"]:
                raise CheckpointError(f"checkpoint file failed integrity validation: {name}")
        except OSError as error:
            raise CheckpointError(f"cannot read checkpoint file {name}: {error}") from error

    model = model_factory(manifest["model_config"])
    model.load_weights(str(checkpoint_path / "model.safetensors"))
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer.state = tree_unflatten(
        list(mx.load(checkpoint_path / "optimizer.safetensors").items())
    )
    try:
        state = TrainerState(**json.loads((checkpoint_path / "state.json").read_text()))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot decode trainer state: {error}") from error
    mx.eval(model.parameters(), optimizer.state)
    return model, optimizer, state, checkpoint_path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    _write_json(temp, value)
    os.replace(temp, path)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
