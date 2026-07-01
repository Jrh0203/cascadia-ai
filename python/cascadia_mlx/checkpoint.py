"""Atomic, checksummed MLX checkpoints with exact optimizer resumption."""

from __future__ import annotations

import fcntl
import io
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
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
    # V3 scheduled-training cursors. Defaults preserve every schema-v1
    # checkpoint written before the campaign scheduler existed.
    examples_seen: int = 0
    schedule_block: int = 0
    batch_in_block: int = 0
    broad_examples_seen: int = 0
    teacher_examples_seen: int = 0
    best_validation_mae: float | None = None
    best_ranking_loss: float | None = None
    best_top1_accuracy: float | None = None
    best_ranking_tiebreak_loss: float | None = None
    ranking_epochs_without_improvement: int = 0
    best_validation_loss: float | None = None
    best_validation_rmse: float | None = None
    value_epochs_without_improvement: int = 0


def save_checkpoint(
    run_dir: str | Path,
    model: Any,
    optimizer: optim.Optimizer,
    state: TrainerState,
    *,
    metadata: dict[str, Any] | None = None,
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
        if metadata is not None:
            manifest["metadata"] = metadata
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


# R2-MAP uses a deliberately separate schema and API surface. Existing schema-v1
# readers/writers above remain behaviorally unchanged for all current consumers.
R2_MAP_CHECKPOINT_SCHEMA_VERSION = 2
R2_MAP_CHECKPOINT_SCHEMA = "r2-map-checkpoint-v2"
R2_MAP_POINTER_SCHEMA = "r2-map-checkpoint-pointer-v1"
R2_MAP_POINTER_NAMES = (
    "latest_complete",
    "last_verified",
    "best_validation",
    "incumbent",
    "promoted",
)
R2_MAP_WRITE_STAGES = (
    "temp-created",
    "model-written",
    "optimizer-written",
    "state-written",
    "prediction-panel-written",
    "manifest-written",
    "checkpoint-committed",
    "latest-pointer-temp-written",
    "latest-pointer-committed",
)
_R2_MAP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


@dataclass(frozen=True)
class R2MapCheckpointIdentity:
    """Immutable scientific and execution identity of one checkpoint."""

    checkpoint_id: str
    run_id: str
    branch_id: str
    source_blake3: str
    dataset_blake3: str
    model_config_blake3: str
    training_config_blake3: str
    loss_contract_blake3: str

    def validate(self) -> None:
        for label, value in (
            ("checkpoint_id", self.checkpoint_id),
            ("run_id", self.run_id),
            ("branch_id", self.branch_id),
        ):
            if not _R2_MAP_ID_PATTERN.fullmatch(value):
                raise CheckpointError(f"invalid R2-MAP {label}: {value!r}")
        for label in (
            "source_blake3",
            "dataset_blake3",
            "model_config_blake3",
            "training_config_blake3",
            "loss_contract_blake3",
        ):
            _require_blake3(getattr(self, label), label)


@dataclass(frozen=True)
class R2MapResumeState:
    """Everything required to reproduce the exact next optimizer batch."""

    global_step: int
    epoch: int
    batch_in_epoch: int
    examples_seen: int
    cursor: dict[str, Any]
    sampler_state: dict[str, Any]
    rng_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    normalization: dict[str, Any]
    auxiliary_loss_weights: dict[str, float]
    dataset_contract: dict[str, Any]
    training_counters: dict[str, int]
    loss_stream: dict[str, Any]
    next_batch_identity: str
    validation: dict[str, float] | None = None

    def validate(self) -> None:
        for label, value in (
            ("global_step", self.global_step),
            ("epoch", self.epoch),
            ("batch_in_epoch", self.batch_in_epoch),
            ("examples_seen", self.examples_seen),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CheckpointError(f"R2-MAP {label} must be a nonnegative integer")
        for label, value in (
            ("cursor", self.cursor),
            ("sampler_state", self.sampler_state),
            ("rng_state", self.rng_state),
            ("scheduler_state", self.scheduler_state),
            ("normalization", self.normalization),
            ("auxiliary_loss_weights", self.auxiliary_loss_weights),
            ("dataset_contract", self.dataset_contract),
            ("training_counters", self.training_counters),
            ("loss_stream", self.loss_stream),
        ):
            if not isinstance(value, dict):
                raise CheckpointError(f"R2-MAP {label} must be a JSON object")
        base_dataset_contract = {
            "schema_version",
            "dataset_blake3",
            "d6_schema",
            "d6_cycle_epochs",
            "imitation_subset_schema",
            "imitation_subset_parts_per_million",
            "collection_kind",
            "example_count",
            "imitation_example_count",
            "market_decision_count",
            "market_policy_target_count",
        }
        packed_value_contract = {
            "adapter_contract_schema_version",
            "adapter_protocol_id",
            "pipe_protocol_id",
            "focal_seat_rule",
            "bootstrap_games",
            "bootstrap_focal_examples",
            "one_epoch_steps",
            "one_epoch_plan_blake3",
            "expanded_window_files",
            "bootstrap_objective",
            "bootstrap_policy_loss_weight",
        }
        contract_fields = set(self.dataset_contract)
        is_packed_value_contract = contract_fields == (
            base_dataset_contract | packed_value_contract
        )
        if (
            frozenset(contract_fields) not in {
                frozenset(base_dataset_contract),
                frozenset(base_dataset_contract | packed_value_contract),
            }
            or self.dataset_contract["schema_version"] != 1
            or self.dataset_contract["d6_cycle_epochs"] != 12
            or self.dataset_contract["imitation_subset_parts_per_million"] != 10_000
            or self.dataset_contract["collection_kind"]
            not in {"bootstrap", "iterative-training", "benchmark", "synthetic"}
        ):
            raise CheckpointError("R2-MAP checkpoint dataset contract differs")
        if is_packed_value_contract and (
            self.dataset_contract["adapter_contract_schema_version"] != 2
            or self.dataset_contract["adapter_protocol_id"]
            != "r2-map-focal-seat-bootstrap-value-pipe-adapter-v2"
            or self.dataset_contract["pipe_protocol_id"] != "r2-map-packed-batch-pipe-v1"
            or self.dataset_contract["focal_seat_rule"] != "global-game-index-mod-4"
            or self.dataset_contract["collection_kind"] != "bootstrap"
            or not isinstance(self.dataset_contract["bootstrap_games"], int)
            or isinstance(self.dataset_contract["bootstrap_games"], bool)
            or self.dataset_contract["bootstrap_games"] <= 0
            or self.dataset_contract["bootstrap_focal_examples"]
            != self.dataset_contract["bootstrap_games"] * 20
            or not isinstance(self.dataset_contract["one_epoch_steps"], int)
            or isinstance(self.dataset_contract["one_epoch_steps"], bool)
            or self.dataset_contract["one_epoch_steps"] <= 0
            or self.dataset_contract["expanded_window_files"] is not False
            or self.dataset_contract["bootstrap_objective"] != "selected-value-only-v1"
            or self.dataset_contract["bootstrap_policy_loss_weight"] != 0.0
            or self.auxiliary_loss_weights.get("bootstrap_policy") != 0.0
        ):
            raise CheckpointError("R2-MAP packed value-only dataset contract differs")
        _require_blake3(self.dataset_contract["dataset_blake3"], "dataset-contract dataset")
        if is_packed_value_contract:
            _require_blake3(
                self.dataset_contract["one_epoch_plan_blake3"],
                "dataset-contract one-epoch plan",
            )
        for name in (
            "d6_schema",
            "imitation_subset_schema",
        ):
            if not isinstance(self.dataset_contract[name], str) or not self.dataset_contract[name]:
                raise CheckpointError("R2-MAP checkpoint dataset schema identity is empty")
        for name in (
            "example_count",
            "imitation_example_count",
            "market_decision_count",
            "market_policy_target_count",
        ):
            value = self.dataset_contract[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CheckpointError("R2-MAP checkpoint dataset count is invalid")
        kind = self.dataset_contract["collection_kind"]
        if (
            self.dataset_contract["imitation_example_count"]
            > self.dataset_contract["example_count"]
            or self.dataset_contract["market_policy_target_count"]
            > self.dataset_contract["market_decision_count"]
            or (
                kind == "bootstrap"
                and self.dataset_contract["market_policy_target_count"]
                != self.dataset_contract["market_decision_count"]
            )
            or (
                kind in {"iterative-training", "benchmark"}
                and (
                    self.dataset_contract["imitation_example_count"] != 0
                    or self.dataset_contract["market_policy_target_count"] != 0
                )
            )
        ):
            raise CheckpointError("R2-MAP checkpoint policy exposure counts differ")
        required_counters = {
            "draft_groups",
            "draft_candidates",
            "padded_draft_candidates",
            "draft_policy_targets",
            "market_groups",
            "market_actions",
            "market_policy_targets",
        }
        if set(self.training_counters) != required_counters or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.training_counters.values()
        ):
            raise CheckpointError("R2-MAP checkpoint training counters differ")
        counters = self.training_counters
        if (
            self.examples_seen != counters["draft_groups"] + counters["market_groups"]
            or counters["draft_groups"] < self.global_step
            or counters["draft_candidates"] < counters["draft_groups"]
            or counters["padded_draft_candidates"] < counters["draft_candidates"]
            or counters["draft_policy_targets"] > counters["draft_groups"]
            or counters["market_actions"] < counters["market_groups"]
            or counters["market_policy_targets"] > counters["market_groups"]
        ):
            raise CheckpointError("R2-MAP checkpoint training counter algebra differs")
        if "epoch" in self.cursor and self.cursor["epoch"] != self.epoch:
            raise CheckpointError("R2-MAP checkpoint cursor epoch differs")
        if set(self.loss_stream) != {
            "relative_path",
            "offset_bytes",
            "prefix_blake3",
            "head_record_blake3",
        }:
            raise CheckpointError("R2-MAP loss-stream state has the wrong fields")
        if not isinstance(self.loss_stream["relative_path"], str):
            raise CheckpointError("R2-MAP loss stream requires a relative path")
        relative = Path(self.loss_stream["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointError("R2-MAP loss stream path must remain run-relative")
        offset = self.loss_stream["offset_bytes"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise CheckpointError("R2-MAP loss-stream offset must be nonnegative")
        _require_blake3(self.loss_stream["prefix_blake3"], "loss-stream prefix")
        head = self.loss_stream["head_record_blake3"]
        if head is not None:
            _require_blake3(head, "loss-stream head record")
        if (
            not isinstance(self.next_batch_identity, str)
            or not self.next_batch_identity
            or len(self.next_batch_identity) > 256
        ):
            raise CheckpointError(
                "R2-MAP next-batch identity must be a nonempty bounded string"
            )
        if self.validation is not None and (
            not isinstance(self.validation, dict)
            or not all(
                isinstance(key, str) and isinstance(value, int | float)
                for key, value in self.validation.items()
            )
        ):
            raise CheckpointError("R2-MAP validation metrics must be a numeric object")
        _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> R2MapResumeState:
        try:
            state = cls(**dict(value))
        except TypeError as error:
            raise CheckpointError(f"cannot decode R2-MAP resume state: {error}") from error
        state.validate()
        return state


@dataclass(frozen=True)
class LoadedR2MapCheckpoint:
    model: Any
    optimizer: optim.Optimizer
    state: R2MapResumeState
    identity: R2MapCheckpointIdentity
    checkpoint_path: Path
    manifest: dict[str, Any]
    prediction_panel: dict[str, mx.array]


@dataclass(frozen=True)
class R2MapCheckpointBundle:
    """One complete checkpoint represented only by bounded in-memory objects."""

    checkpoint_id: str
    manifest: dict[str, Any]
    objects: dict[str, bytes]

    @property
    def manifest_blake3(self) -> str:
        return blake3.blake3(self.objects["checkpoint.json"]).hexdigest()

    @property
    def total_bytes(self) -> int:
        return sum(len(value) for value in self.objects.values())


class _NamedBytesIO(io.BytesIO):
    def __init__(self, value: bytes = b"", *, name: str):
        super().__init__(value)
        self.name = name


def build_r2_map_checkpoint_bundle(
    model: Any,
    optimizer: optim.Optimizer,
    identity: R2MapCheckpointIdentity,
    state: R2MapResumeState,
    *,
    model_config: Mapping[str, Any],
    fixed_prediction_panel: Mapping[str, mx.array],
    prediction_panel_id: str,
) -> R2MapCheckpointBundle:
    """Serialize the schema-v2 checkpoint without filesystem I/O."""

    identity.validate()
    state.validate()
    if not _R2_MAP_ID_PATTERN.fullmatch(prediction_panel_id):
        raise CheckpointError("invalid R2-MAP fixed prediction panel id")
    config = dict(model_config)
    if _canonical_blake3(config) != identity.model_config_blake3:
        raise CheckpointError("R2-MAP model config hash differs from checkpoint identity")
    if not fixed_prediction_panel or any(
        not name or not isinstance(value, mx.array)
        for name, value in fixed_prediction_panel.items()
    ):
        raise CheckpointError("R2-MAP checkpoint requires named fixed-panel tensors")
    mx.eval(model.parameters(), optimizer.state, fixed_prediction_panel)

    model_buffer = _NamedBytesIO(name="model.safetensors")
    mx.save_safetensors(model_buffer, dict(tree_flatten(model.parameters())))
    optimizer_buffer = _NamedBytesIO(name="optimizer.safetensors")
    mx.save_safetensors(
        optimizer_buffer,
        dict(tree_flatten(optimizer.state)),
        metadata={"schema_version": str(R2_MAP_CHECKPOINT_SCHEMA_VERSION)},
    )
    panel_buffer = _NamedBytesIO(name="fixed-prediction-panel.safetensors")
    mx.save_safetensors(
        panel_buffer,
        dict(fixed_prediction_panel),
        metadata={"panel_id": prediction_panel_id},
    )
    payloads = {
        "model.safetensors": model_buffer.getvalue(),
        "optimizer.safetensors": optimizer_buffer.getvalue(),
        "state.json": _canonical_json(state.to_dict()) + b"\n",
        "fixed-prediction-panel.safetensors": panel_buffer.getvalue(),
    }
    files = {
        name: {"blake3": blake3.blake3(value).hexdigest(), "bytes": len(value)}
        for name, value in payloads.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": R2_MAP_CHECKPOINT_SCHEMA_VERSION,
        "schema_id": R2_MAP_CHECKPOINT_SCHEMA,
        "checkpoint_id": identity.checkpoint_id,
        "identity": asdict(identity),
        "model_config": config,
        "resume_state_blake3": _canonical_blake3(state.to_dict()),
        "prediction_panel": {
            "panel_id": prediction_panel_id,
            "tensor_shapes": {
                name: list(value.shape) for name, value in fixed_prediction_panel.items()
            },
            "tensor_dtypes": {
                name: str(value.dtype) for name, value in fixed_prediction_panel.items()
            },
        },
        "files": files,
    }
    manifest["manifest_identity_blake3"] = _canonical_blake3(manifest)
    payloads["checkpoint.json"] = _canonical_json(manifest) + b"\n"
    bundle = R2MapCheckpointBundle(identity.checkpoint_id, manifest, payloads)
    verify_r2_map_checkpoint_bundle(bundle)
    return bundle


def verify_r2_map_checkpoint_bundle(
    bundle: R2MapCheckpointBundle,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    loss_stream: bytes | None = None,
) -> tuple[dict[str, Any], R2MapResumeState, dict[str, mx.array]]:
    """Verify every in-memory object using the same immutable schema contract."""

    required = {
        "checkpoint.json",
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
        "fixed-prediction-panel.safetensors",
    }
    if set(bundle.objects) != required:
        raise CheckpointError("R2-MAP checkpoint bundle object set is incomplete")
    try:
        manifest = json.loads(bundle.objects["checkpoint.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot decode R2-MAP checkpoint manifest: {error}") from error
    if manifest != bundle.manifest or manifest.get("checkpoint_id") != bundle.checkpoint_id:
        raise CheckpointError("R2-MAP checkpoint bundle manifest identity differs")
    if (
        manifest.get("schema_version") != R2_MAP_CHECKPOINT_SCHEMA_VERSION
        or manifest.get("schema_id") != R2_MAP_CHECKPOINT_SCHEMA
    ):
        raise CheckpointError("unsupported R2-MAP checkpoint bundle schema")
    try:
        identity = R2MapCheckpointIdentity(**manifest["identity"])
    except (KeyError, TypeError) as error:
        raise CheckpointError(f"invalid R2-MAP checkpoint bundle identity: {error}") from error
    identity.validate()
    if identity.checkpoint_id != bundle.checkpoint_id:
        raise CheckpointError("R2-MAP checkpoint bundle id differs")
    if expected_identity is not None:
        for name, expected in expected_identity.items():
            if manifest["identity"].get(name) != expected:
                raise CheckpointError(f"R2-MAP immutable identity mismatch for {name}")
    claimed = manifest.get("manifest_identity_blake3")
    identity_manifest = dict(manifest)
    identity_manifest.pop("manifest_identity_blake3", None)
    if claimed != _canonical_blake3(identity_manifest):
        raise CheckpointError("R2-MAP checkpoint bundle manifest hash differs")
    if _canonical_blake3(manifest.get("model_config")) != identity.model_config_blake3:
        raise CheckpointError("R2-MAP checkpoint bundle model config hash differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != required - {"checkpoint.json"}:
        raise CheckpointError("R2-MAP checkpoint bundle file manifest is incomplete")
    for name, expected in files.items():
        payload = bundle.objects[name]
        if (
            expected != {"blake3": blake3.blake3(payload).hexdigest(), "bytes": len(payload)}
        ):
            raise CheckpointError(f"R2-MAP checkpoint bundle failed integrity: {name}")
    try:
        state = R2MapResumeState.from_dict(json.loads(bundle.objects["state.json"]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot decode R2-MAP checkpoint state: {error}") from error
    if manifest.get("resume_state_blake3") != _canonical_blake3(state.to_dict()):
        raise CheckpointError("R2-MAP checkpoint bundle resume state differs")
    panel_buffer = _NamedBytesIO(
        bundle.objects["fixed-prediction-panel.safetensors"],
        name="fixed-prediction-panel.safetensors",
    )
    panel = dict(mx.load(panel_buffer))
    panel_manifest = manifest.get("prediction_panel", {})
    if set(panel) != set(panel_manifest.get("tensor_shapes", {})):
        raise CheckpointError("R2-MAP checkpoint bundle panel names differ")
    for name, value in panel.items():
        if (
            list(value.shape) != panel_manifest["tensor_shapes"][name]
            or str(value.dtype) != panel_manifest["tensor_dtypes"][name]
        ):
            raise CheckpointError(f"R2-MAP checkpoint bundle panel differs: {name}")
    if loss_stream is not None:
        verify_loss_stream_prefix_bytes(loss_stream, state.loss_stream)
    return manifest, state, panel


def load_r2_map_checkpoint_bundle(
    bundle: R2MapCheckpointBundle,
    *,
    model_factory: Callable[[dict[str, object]], Any],
    optimizer_factory: Callable[[], optim.Optimizer],
    expected_identity: Mapping[str, Any] | None = None,
    loss_stream: bytes | None = None,
) -> LoadedR2MapCheckpoint:
    manifest, state, panel = verify_r2_map_checkpoint_bundle(
        bundle,
        expected_identity=expected_identity,
        loss_stream=loss_stream,
    )
    model = model_factory(manifest["model_config"])
    model_buffer = _NamedBytesIO(bundle.objects["model.safetensors"], name="model.safetensors")
    model.load_weights(list(mx.load(model_buffer).items()))
    optimizer = optimizer_factory()
    optimizer_buffer = _NamedBytesIO(
        bundle.objects["optimizer.safetensors"], name="optimizer.safetensors"
    )
    optimizer.state = tree_unflatten(list(mx.load(optimizer_buffer).items()))
    identity = R2MapCheckpointIdentity(**manifest["identity"])
    mx.eval(model.parameters(), optimizer.state, panel)
    return LoadedR2MapCheckpoint(
        model=model,
        optimizer=optimizer,
        state=state,
        identity=identity,
        checkpoint_path=Path(bundle.checkpoint_id),
        manifest=manifest,
        prediction_panel=panel,
    )


def build_r2_map_checkpoint_pointer_document(
    name: str,
    bundle: R2MapCheckpointBundle,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the existing pointer schema from an in-memory checkpoint."""
    if name not in R2_MAP_POINTER_NAMES:
        raise CheckpointError(f"unknown R2-MAP checkpoint pointer {name!r}")
    verify_r2_map_checkpoint_bundle(bundle)
    value: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": R2_MAP_POINTER_SCHEMA,
        "pointer": name,
        "checkpoint": bundle.checkpoint_id,
        "manifest_blake3": bundle.manifest_blake3,
        "checkpoint_identity_blake3": bundle.manifest["manifest_identity_blake3"],
    }
    if metadata is not None:
        value["metadata"] = dict(metadata)
    _canonical_json(value)
    return value


def validate_r2_map_checkpoint_pointer_document(
    value: Mapping[str, Any],
    *,
    name: str,
    bundle: R2MapCheckpointBundle,
) -> dict[str, Any]:
    expected = build_r2_map_checkpoint_pointer_document(
        name,
        bundle,
        metadata=value.get("metadata") if isinstance(value, Mapping) else None,
    )
    if dict(value) != expected:
        raise CheckpointError(f"R2-MAP {name} pointer differs from checkpoint bundle")
    return expected


def save_r2_map_checkpoint(
    run_dir: str | Path,
    model: Any,
    optimizer: optim.Optimizer,
    identity: R2MapCheckpointIdentity,
    state: R2MapResumeState,
    *,
    model_config: Mapping[str, Any],
    fixed_prediction_panel: Mapping[str, mx.array],
    prediction_panel_id: str,
    fault_injector: Callable[[str], None] | None = None,
) -> Path:
    """Durably commit one immutable schema-v2 R2-MAP checkpoint.

    The only mutable artifact is ``latest_complete.json``. A fault before the
    directory rename removes the temporary directory; a fault afterward can at
    worst leave a complete unpointed checkpoint.
    """
    identity.validate()
    state.validate()
    if not _R2_MAP_ID_PATTERN.fullmatch(prediction_panel_id):
        raise CheckpointError("invalid R2-MAP fixed prediction panel id")
    config = dict(model_config)
    observed_config_hash = _canonical_blake3(config)
    if observed_config_hash != identity.model_config_blake3:
        raise CheckpointError("R2-MAP model config hash differs from checkpoint identity")
    if not fixed_prediction_panel:
        raise CheckpointError("R2-MAP checkpoint requires a fixed prediction panel")
    for name, value in fixed_prediction_panel.items():
        if not name or not isinstance(value, mx.array):
            raise CheckpointError("R2-MAP prediction panel must contain named MLX tensors")

    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final_path = checkpoints / identity.checkpoint_id
    if final_path.exists():
        raise CheckpointError(
            f"immutable R2-MAP checkpoint id already exists: {identity.checkpoint_id}"
        )
    temporary = checkpoints / f".{identity.checkpoint_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    committed = False

    def stage(name: str) -> None:
        if fault_injector is not None:
            fault_injector(name)

    try:
        if temporary.stat().st_dev != checkpoints.stat().st_dev:
            raise CheckpointError("R2-MAP checkpoint temporary directory is on another volume")
        stage("temp-created")

        model_path = temporary / "model.safetensors"
        model.save_weights(str(model_path))
        _fsync_file(model_path)
        stage("model-written")

        optimizer_path = temporary / "optimizer.safetensors"
        mx.save_safetensors(
            optimizer_path,
            dict(tree_flatten(optimizer.state)),
            metadata={"schema_version": str(R2_MAP_CHECKPOINT_SCHEMA_VERSION)},
        )
        _fsync_file(optimizer_path)
        stage("optimizer-written")

        state_path = temporary / "state.json"
        _write_json_fsync(state_path, state.to_dict())
        stage("state-written")

        panel_path = temporary / "fixed-prediction-panel.safetensors"
        mx.save_safetensors(
            panel_path,
            dict(fixed_prediction_panel),
            metadata={"panel_id": prediction_panel_id},
        )
        _fsync_file(panel_path)
        stage("prediction-panel-written")

        files = {
            path.name: {"blake3": _checksum(path), "bytes": path.stat().st_size}
            for path in (model_path, optimizer_path, state_path, panel_path)
        }
        manifest: dict[str, Any] = {
            "schema_version": R2_MAP_CHECKPOINT_SCHEMA_VERSION,
            "schema_id": R2_MAP_CHECKPOINT_SCHEMA,
            "checkpoint_id": identity.checkpoint_id,
            "identity": asdict(identity),
            "model_config": config,
            "resume_state_blake3": _canonical_blake3(state.to_dict()),
            "prediction_panel": {
                "panel_id": prediction_panel_id,
                "tensor_shapes": {
                    name: list(value.shape) for name, value in fixed_prediction_panel.items()
                },
                "tensor_dtypes": {
                    name: str(value.dtype) for name, value in fixed_prediction_panel.items()
                },
            },
            "files": files,
        }
        manifest["manifest_identity_blake3"] = _canonical_blake3(manifest)
        manifest_path = temporary / "checkpoint.json"
        _write_json_fsync(manifest_path, manifest)
        _fsync_directory_strict(temporary)
        stage("manifest-written")

        lock_path = checkpoints / ".r2-map-checkpoint-write.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if final_path.exists():
                raise CheckpointError(
                    "immutable R2-MAP checkpoint id raced with another writer: "
                    f"{identity.checkpoint_id}"
                )
            os.replace(temporary, final_path)
            committed = True
            _fsync_directory_strict(checkpoints)
            stage("checkpoint-committed")
            set_r2_map_checkpoint_pointer(
                run_dir,
                "latest_complete",
                final_path,
                fault_injector=fault_injector,
                fault_stage_prefix="latest-pointer",
            )
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return final_path
    except Exception:
        if not committed:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_r2_map_checkpoint_files(
    checkpoint_path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    loss_stream_path: str | Path | None = None,
) -> tuple[dict[str, Any], R2MapResumeState, dict[str, mx.array]]:
    """Verify schema, identity, every file, and the captured loss-stream prefix."""
    checkpoint_path = Path(checkpoint_path)
    try:
        manifest = json.loads((checkpoint_path / "checkpoint.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read R2-MAP checkpoint manifest: {error}") from error
    if (
        manifest.get("schema_version") != R2_MAP_CHECKPOINT_SCHEMA_VERSION
        or manifest.get("schema_id") != R2_MAP_CHECKPOINT_SCHEMA
    ):
        raise CheckpointError("unsupported R2-MAP checkpoint schema")
    if manifest.get("checkpoint_id") != checkpoint_path.name:
        raise CheckpointError("R2-MAP checkpoint directory and immutable id differ")
    identity_value = manifest.get("identity")
    try:
        identity = R2MapCheckpointIdentity(**identity_value)
    except (TypeError, AttributeError) as error:
        raise CheckpointError(f"invalid R2-MAP checkpoint identity: {error}") from error
    identity.validate()
    if identity.checkpoint_id != checkpoint_path.name:
        raise CheckpointError("R2-MAP manifest identity and directory differ")
    if expected_identity is not None:
        for name, expected in expected_identity.items():
            if identity_value.get(name) != expected:
                raise CheckpointError(f"R2-MAP immutable identity mismatch for {name}")
    observed_manifest_identity = dict(manifest)
    claimed_manifest_identity = observed_manifest_identity.pop("manifest_identity_blake3", None)
    if claimed_manifest_identity != _canonical_blake3(observed_manifest_identity):
        raise CheckpointError("R2-MAP checkpoint manifest identity hash differs")
    if _canonical_blake3(manifest.get("model_config")) != identity.model_config_blake3:
        raise CheckpointError("R2-MAP checkpoint model config hash differs")
    files = manifest.get("files")
    required_files = {
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
        "fixed-prediction-panel.safetensors",
    }
    if not isinstance(files, dict) or set(files) != required_files:
        raise CheckpointError("R2-MAP checkpoint file manifest is incomplete")
    for name, expected in files.items():
        path = checkpoint_path / name
        try:
            if path.stat().st_size != expected["bytes"] or _checksum(path) != expected["blake3"]:
                raise CheckpointError(f"R2-MAP checkpoint file failed integrity: {name}")
        except (OSError, KeyError, TypeError) as error:
            raise CheckpointError(
                f"cannot verify R2-MAP checkpoint file {name}: {error}"
            ) from error
    try:
        state_value = json.loads((checkpoint_path / "state.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read R2-MAP resume state: {error}") from error
    state = R2MapResumeState.from_dict(state_value)
    if manifest.get("resume_state_blake3") != _canonical_blake3(state.to_dict()):
        raise CheckpointError("R2-MAP resume state identity differs")
    panel = dict(mx.load(checkpoint_path / "fixed-prediction-panel.safetensors"))
    panel_manifest = manifest.get("prediction_panel", {})
    if set(panel) != set(panel_manifest.get("tensor_shapes", {})):
        raise CheckpointError("R2-MAP fixed prediction panel tensor names differ")
    for name, value in panel.items():
        if list(value.shape) != panel_manifest["tensor_shapes"][name]:
            raise CheckpointError(f"R2-MAP fixed prediction panel shape differs: {name}")
        if str(value.dtype) != panel_manifest["tensor_dtypes"][name]:
            raise CheckpointError(f"R2-MAP fixed prediction panel dtype differs: {name}")
    if loss_stream_path is not None:
        verify_loss_stream_prefix(Path(loss_stream_path), state.loss_stream)
    return manifest, state, panel


def load_r2_map_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_factory: Callable[[dict[str, object]], Any],
    optimizer_factory: Callable[[], optim.Optimizer],
    expected_identity: Mapping[str, Any] | None = None,
    loss_stream_path: str | Path | None = None,
) -> LoadedR2MapCheckpoint:
    manifest, state, panel = verify_r2_map_checkpoint_files(
        checkpoint_path,
        expected_identity=expected_identity,
        loss_stream_path=loss_stream_path,
    )
    model = model_factory(manifest["model_config"])
    model.load_weights(str(Path(checkpoint_path) / "model.safetensors"))
    optimizer = optimizer_factory()
    optimizer.state = tree_unflatten(
        list(mx.load(Path(checkpoint_path) / "optimizer.safetensors").items())
    )
    identity = R2MapCheckpointIdentity(**manifest["identity"])
    mx.eval(model.parameters(), optimizer.state, panel)
    return LoadedR2MapCheckpoint(
        model=model,
        optimizer=optimizer,
        state=state,
        identity=identity,
        checkpoint_path=Path(checkpoint_path),
        manifest=manifest,
        prediction_panel=panel,
    )


def load_r2_map_checkpoint_pointer(
    run_dir: str | Path,
    pointer: str,
    **load_arguments: Any,
) -> LoadedR2MapCheckpoint:
    checkpoint = resolve_r2_map_checkpoint_pointer(run_dir, pointer)
    return load_r2_map_checkpoint(checkpoint, **load_arguments)


def set_r2_map_checkpoint_pointer(
    run_dir: str | Path,
    name: str,
    checkpoint: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    fault_injector: Callable[[str], None] | None = None,
    fault_stage_prefix: str | None = None,
) -> None:
    if name not in R2_MAP_POINTER_NAMES:
        raise CheckpointError(f"unknown R2-MAP checkpoint pointer {name!r}")
    run_dir = Path(run_dir)
    checkpoint = Path(checkpoint)
    expected_parent = (run_dir / "checkpoints").resolve()
    if checkpoint.resolve().parent != expected_parent:
        raise CheckpointError("R2-MAP checkpoint pointer must target this run")
    manifest, _, _ = verify_r2_map_checkpoint_files(checkpoint)
    value: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": R2_MAP_POINTER_SCHEMA,
        "pointer": name,
        "checkpoint": checkpoint.name,
        "manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "checkpoint_identity_blake3": manifest["manifest_identity_blake3"],
    }
    if metadata is not None:
        value["metadata"] = dict(metadata)
    _write_json_atomic_fsync(
        run_dir / f"{name}.json",
        value,
        fault_injector=fault_injector,
        fault_stage_prefix=fault_stage_prefix,
    )


def prune_r2_map_checkpoints(run_dir: str | Path, *, keep_recent: int = 2) -> None:
    """Prune only unpointed v2 checkpoints, preserving every semantic role."""
    if keep_recent < 0:
        raise ValueError("keep_recent cannot be negative")
    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir():
        return
    complete: list[tuple[int, str, Path]] = []
    for path in checkpoints.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        _, state, _ = verify_r2_map_checkpoint_files(path)
        complete.append((state.global_step, path.name, path))
    complete.sort()
    keep = {item[2] for item in complete[-keep_recent:]} if keep_recent else set()
    for pointer in R2_MAP_POINTER_NAMES:
        if (run_dir / f"{pointer}.json").exists():
            keep.add(resolve_r2_map_checkpoint_pointer(run_dir, pointer))
    for _, _, checkpoint in complete:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)
    _fsync_directory_strict(checkpoints)


def resolve_r2_map_checkpoint_pointer(run_dir: str | Path, name: str) -> Path:
    if name not in R2_MAP_POINTER_NAMES:
        raise CheckpointError(f"unknown R2-MAP checkpoint pointer {name!r}")
    run_dir = Path(run_dir)
    pointer_path = run_dir / f"{name}.json"
    try:
        value = json.loads(pointer_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read R2-MAP {name} pointer: {error}") from error
    if (
        value.get("schema_version") != 1
        or value.get("schema_id") != R2_MAP_POINTER_SCHEMA
        or value.get("pointer") != name
    ):
        raise CheckpointError(f"invalid R2-MAP {name} pointer schema")
    checkpoint = run_dir / "checkpoints" / str(value.get("checkpoint"))
    manifest, _, _ = verify_r2_map_checkpoint_files(checkpoint)
    if value.get("manifest_blake3") != _checksum(checkpoint / "checkpoint.json"):
        raise CheckpointError(f"R2-MAP {name} pointer manifest hash differs")
    if value.get("checkpoint_identity_blake3") != manifest["manifest_identity_blake3"]:
        raise CheckpointError(f"R2-MAP {name} pointer identity hash differs")
    return checkpoint


def verify_loss_stream_prefix(path: Path, loss_stream: Mapping[str, Any]) -> None:
    try:
        offset = int(loss_stream["offset_bytes"])
        expected = loss_stream["prefix_blake3"]
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"invalid R2-MAP loss-stream binding: {error}") from error
    try:
        with path.open("rb") as handle:
            prefix = handle.read(offset)
            if len(prefix) != offset:
                raise CheckpointError("R2-MAP loss stream is shorter than checkpoint offset")
    except OSError as error:
        raise CheckpointError(f"cannot read R2-MAP loss stream: {error}") from error
    if blake3.blake3(prefix).hexdigest() != expected:
        raise CheckpointError("R2-MAP loss stream prefix differs from checkpoint")


def verify_loss_stream_prefix_bytes(content: bytes, loss_stream: Mapping[str, Any]) -> None:
    """Verify a remotely streamed loss prefix without constructing a local file."""
    try:
        offset = int(loss_stream["offset_bytes"])
        expected = loss_stream["prefix_blake3"]
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"invalid R2-MAP loss-stream binding: {error}") from error
    if offset < 0 or len(content) < offset:
        raise CheckpointError("R2-MAP loss stream is shorter than checkpoint offset")
    if blake3.blake3(content[:offset]).hexdigest() != expected:
        raise CheckpointError("R2-MAP loss stream prefix differs from checkpoint")


def loss_stream_binding_bytes(
    content: bytes,
    *,
    relative_path: str,
    head_record_blake3: str | None,
) -> dict[str, Any]:
    """Bind one in-memory remote loss stream to exact resume state."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CheckpointError("R2-MAP loss stream path must remain run-relative")
    if head_record_blake3 is not None:
        _require_blake3(head_record_blake3, "loss-stream head record")
    return {
        "relative_path": relative.as_posix(),
        "offset_bytes": len(content),
        "prefix_blake3": blake3.blake3(content).hexdigest(),
        "head_record_blake3": head_record_blake3,
    }


def loss_stream_binding(
    path: str | Path,
    *,
    relative_to: str | Path,
    head_record_blake3: str | None,
) -> dict[str, Any]:
    path = Path(path)
    root = Path(relative_to).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise CheckpointError("R2-MAP loss stream escapes its run directory")
    try:
        relative = resolved.relative_to(root).as_posix()
        content = path.read_bytes()
    except OSError as error:
        raise CheckpointError(f"cannot bind R2-MAP loss stream: {error}") from error
    if head_record_blake3 is not None:
        _require_blake3(head_record_blake3, "loss-stream head record")
    return {
        "relative_path": relative,
        "offset_bytes": len(content),
        "prefix_blake3": blake3.blake3(content).hexdigest(),
        "head_record_blake3": head_record_blake3,
    }


def _require_blake3(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointError(f"R2-MAP {label} must be a lowercase BLAKE3 digest")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"R2-MAP metadata is not canonical JSON: {error}") from error


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(_canonical_json(value)).hexdigest()


def _write_json_fsync(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic_fsync(
    path: Path,
    value: Any,
    *,
    fault_injector: Callable[[str], None] | None = None,
    fault_stage_prefix: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json_fsync(temporary, value)
        if temporary.stat().st_dev != path.parent.stat().st_dev:
            raise CheckpointError("R2-MAP atomic JSON temporary file crossed volumes")
        if fault_injector is not None and fault_stage_prefix is not None:
            fault_injector(f"{fault_stage_prefix}-temp-written")
        os.replace(temporary, path)
        _fsync_directory_strict(path.parent)
        if fault_injector is not None and fault_stage_prefix is not None:
            fault_injector(f"{fault_stage_prefix}-committed")
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory_strict(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
