"""Joint MLX rollout-return and root-ranking fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_checkpoint_pointer_with_factory,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.legacy_nnue import (
    LegacySparseNnue,
    checksum_file,
    load_legacy_nnue_manifest,
    package_derived_legacy_nnue,
)
from cascadia_mlx.rollout_value_dataset import RolloutValueDataset
from cascadia_mlx.rollout_value_model import (
    ROLLOUT_ROOT_HUBER_DELTA,
    ROLLOUT_ROOT_SELECTED_WEIGHT,
    ROLLOUT_ROOT_TEACHER_TEMPERATURE,
    ROLLOUT_ROOT_TEACHER_WEIGHT,
    ROLLOUT_VALUE_HUBER_DELTA,
    RolloutValueNnue,
    RolloutValueNnueConfig,
    rollout_root_ranking_loss,
    rollout_value_loss,
)
from cascadia_mlx.rollout_value_train import (
    _append_json,
    _parent_identity,
    _validation_gates,
    _value_tensors,
    _verify_packaged_values,
    _write_json_atomic,
    evaluate_rollout_roots,
    evaluate_rollout_trajectories,
)
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest

ADR66_PARENT_MANIFEST_BLAKE3 = "dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d"
ADR66_TRAIN_MANIFEST_BLAKE3 = "5a041c73c15075e38d3106a77b09b1a33e6597c4d8eee5eea38446490c282ec0"
ADR66_SMOKE_MANIFEST_BLAKE3 = "139ae54f94001bccd24b7a3a0493952e260cb45139a5c505d3d317f4514d1115"
ADR66_TRAIN_FIRST_GAME = 94_000
ADR66_TRAIN_GAMES = 4
ADR66_VALIDATION_FIRST_GAME = 95_000
ADR66_VALIDATION_GAMES = 2
ADR66_MIN_TRAIN_TRAJECTORIES = 100_000
ADR66_MIN_VALIDATION_TRAJECTORIES = 40_000


@dataclass(frozen=True)
class JointRolloutTrainingConfig:
    parent_model_dir: Path
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    derived_model_dir: Path
    epochs: int = 12
    trajectory_batch_size: int = 512
    root_group_batch_size: int = 4
    learning_rate: float = 3e-6
    weight_decay: float = 0.0
    seed: int = 20_260_621
    checkpoint_steps: int = 500
    validation_patience: int = 4
    implementation_smoke: bool = False
    resume: bool = False

    def validate(self) -> None:
        if not 1 <= self.epochs <= 12:
            raise ValueError("ADR66 epochs must be between one and twelve")
        if self.implementation_smoke and self.epochs != 1:
            raise ValueError("ADR66 implementation smoke is exactly one epoch")
        if self.trajectory_batch_size != 512:
            raise ValueError("ADR66 trajectory batch size is frozen at 512")
        if self.root_group_batch_size != 4:
            raise ValueError("ADR66 root group batch size is frozen at four")
        if self.learning_rate != 3e-6:
            raise ValueError("ADR66 learning rate is frozen at 3e-6")
        if self.weight_decay != 0.0:
            raise ValueError("ADR66 weight decay is frozen at zero")
        if self.seed != 20_260_621:
            raise ValueError("ADR66 seed is frozen at 20260621")
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.validation_patience != 4:
            raise ValueError("ADR66 validation patience is frozen at four")


def train_joint_rollout(config: JointRolloutTrainingConfig) -> dict[str, Any]:
    config.validate()
    train_dataset = RolloutValueDataset(config.train_dataset)
    validation_dataset = RolloutValueDataset(config.validation_dataset)
    parent_manifest = load_legacy_nnue_manifest(config.parent_model_dir)
    _validate_adr66_inputs(config, train_dataset, validation_dataset)
    parent_identity_before = _parent_identity(config.parent_model_dir)
    parent = LegacySparseNnue.load(config.parent_model_dir)
    parent_policy = {
        "w3_policy": parent.tensors["w3_policy"],
        "b3_policy": parent.tensors["b3_policy"],
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _build_run_manifest(
        config,
        train_dataset,
        validation_dataset,
        parent_manifest,
        parent_identity_before,
    )

    if config.resume:
        validate_resume_manifest(
            config.run_dir,
            training=run_manifest["training"],
            datasets=run_manifest["datasets"],
            runtime=run_manifest["runtime"],
            source=run_manifest["source"],
        )
        model, optimizer, state, _ = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            model_factory=lambda values: RolloutValueNnue(RolloutValueNnueConfig.from_dict(values)),
        )
        if config.epochs < state.epoch:
            raise ValueError("resume epoch budget is behind the checkpoint")
        final_path = config.run_dir / "final-report.json"
        if final_path.exists():
            return json.loads(final_path.read_text())
    else:
        if (config.run_dir / "latest.json").exists() or (config.run_dir / "run.json").exists():
            raise ValueError("ADR66 run already exists; pass --resume")
        mx.random.seed(config.seed)
        model = RolloutValueNnue.from_parent(config.parent_model_dir)
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    value_loss_and_grad = nn.value_and_grad(model, rollout_value_loss)
    root_loss_and_grad = nn.value_and_grad(model, rollout_root_ranking_loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    started = time.perf_counter()
    stopped_early = state.ranking_epochs_without_improvement >= config.validation_patience
    for epoch in range(state.epoch, config.epochs) if not stopped_early else ():
        model.train()
        trajectory_loss = 0.0
        trajectory_batches = 0
        root_loss = 0.0
        root_batches = 0
        resume_batch = state.batch_in_epoch if epoch == state.epoch else 0
        epoch_batches = _epoch_batches(config, train_dataset, epoch)
        for batch_index, (kind, batch) in enumerate(epoch_batches):
            if batch_index < resume_batch:
                continue
            if kind == "trajectory":
                loss, gradients = value_loss_and_grad(model, batch)
            else:
                loss, gradients = root_loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise ValueError("ADR66 training produced a non-finite loss")
            if kind == "trajectory":
                trajectory_loss += loss_value
                trajectory_batches += 1
            else:
                root_loss += loss_value
                root_batches += 1
            state.global_step += 1
            state.epoch = epoch
            state.batch_in_epoch = batch_index + 1
            state.elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if state.global_step % config.checkpoint_steps == 0:
                save_checkpoint(config.run_dir, model, optimizer, state)
                prune_checkpoints(config.run_dir)

        validation_trajectory = evaluate_rollout_trajectories(
            model,
            parent_policy,
            validation_dataset,
            config.trajectory_batch_size,
        )
        validation_root = evaluate_rollout_roots(
            model,
            parent_policy,
            validation_dataset,
            config.trajectory_batch_size,
        )
        state.elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        improved = (
            state.best_ranking_loss is None
            or validation_root["selection_loss"] < state.best_ranking_loss
        )
        if improved:
            state.best_ranking_loss = validation_root["selection_loss"]
            state.best_validation_rmse = validation_trajectory["rmse"]
            state.ranking_epochs_without_improvement = 0
        else:
            state.ranking_epochs_without_improvement += 1
        state.epoch = epoch + 1
        state.batch_in_epoch = 0
        event = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_trajectory_huber_loss": trajectory_loss / max(trajectory_batches, 1),
            "train_root_ranking_loss": root_loss / max(root_batches, 1),
            "validation_trajectory": validation_trajectory,
            "validation_root": validation_root,
            "best_ranking_loss": state.best_ranking_loss,
            "epochs_without_improvement": state.ranking_epochs_without_improvement,
            "elapsed_seconds": state.elapsed_seconds,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
        if improved:
            set_checkpoint_pointer(
                config.run_dir,
                "best",
                checkpoint,
                {
                    "validation_trajectory": validation_trajectory,
                    "validation_root": validation_root,
                },
            )
        prune_checkpoints(config.run_dir)
        if state.ranking_epochs_without_improvement >= config.validation_patience:
            stopped_early = True
            break

    selected_model, _, selected_state, selected_checkpoint = load_checkpoint_pointer_with_factory(
        config.run_dir,
        pointer="best",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        model_factory=lambda values: RolloutValueNnue(RolloutValueNnueConfig.from_dict(values)),
    )
    parent_model = RolloutValueNnue.from_parent(config.parent_model_dir)
    parent_trajectory = evaluate_rollout_trajectories(
        parent_model,
        parent_policy,
        validation_dataset,
        config.trajectory_batch_size,
    )
    selected_trajectory = evaluate_rollout_trajectories(
        selected_model,
        parent_policy,
        validation_dataset,
        config.trajectory_batch_size,
    )
    parent_root = evaluate_rollout_roots(
        parent_model,
        parent_policy,
        validation_dataset,
        config.trajectory_batch_size,
    )
    selected_root = evaluate_rollout_roots(
        selected_model,
        parent_policy,
        validation_dataset,
        config.trajectory_batch_size,
    )
    parent_identity_after = _parent_identity(config.parent_model_dir)
    gates = _validation_gates(
        train_dataset,
        validation_dataset,
        parent_trajectory,
        selected_trajectory,
        parent_root,
        selected_root,
        parent_identity_before == parent_identity_after,
    )
    derivation = {
        "kind": "mlx-joint-return-ranking-finetune-v1",
        "parent_manifest_blake3": parent_identity_before["manifest_blake3"],
        "parent_model_blake3": parent_identity_before["model_blake3"],
        "train_dataset_manifest_blake3": train_dataset.manifest_blake3,
        "validation_dataset_manifest_blake3": validation_dataset.manifest_blake3,
        "run_manifest_blake3": checksum_file(config.run_dir / "run.json"),
        "selected_checkpoint": selected_checkpoint.name,
        "checkpoint_manifest_blake3": checksum_file(selected_checkpoint / "checkpoint.json"),
    }
    derived_manifest = package_derived_legacy_nnue(
        config.parent_model_dir,
        config.derived_model_dir,
        _value_tensors(selected_model),
        derivation,
    )
    _verify_packaged_values(config.derived_model_dir, selected_model, parent_policy)
    state.elapsed_seconds += time.perf_counter() - started
    final_report = {
        "schema_version": 1,
        "experiment_id": "exact-mlx-joint-return-ranking-finetune-v1-20260612",
        "status": "passed-validation" if gates["passed"] else "rejected-validation",
        "selected_checkpoint": selected_checkpoint.name,
        "selected_epoch": selected_state.epoch,
        "global_step": selected_state.global_step,
        "stopped_early": stopped_early,
        "parent_trajectory": parent_trajectory,
        "selected_trajectory": selected_trajectory,
        "parent_root": parent_root,
        "selected_root": selected_root,
        "gates": gates,
        "derived_artifact": {
            "path": str(config.derived_model_dir.resolve()),
            "manifest_blake3": checksum_file(config.derived_model_dir / "model.json"),
            "model_blake3": derived_manifest["files"]["model.safetensors"]["blake3"],
        },
        "parent_identity_before": parent_identity_before,
        "parent_identity_after": parent_identity_after,
        "elapsed_seconds": state.elapsed_seconds,
        "device": str(mx.default_device()),
    }
    _write_json_atomic(config.run_dir / "final-report.json", final_report)
    return final_report


def _epoch_batches(
    config: JointRolloutTrainingConfig,
    dataset: RolloutValueDataset,
    epoch: int,
) -> Iterator[tuple[str, object]]:
    for batch in dataset.batches(
        config.trajectory_batch_size,
        kind="trajectory",
        shuffle=True,
        seed=config.seed + epoch,
    ):
        yield "trajectory", batch
    for batch in dataset.root_groups(
        config.root_group_batch_size,
        shuffle=True,
        seed=config.seed + 10_000 + epoch,
    ):
        yield "root", batch


def _validate_adr66_inputs(
    config: JointRolloutTrainingConfig,
    train: RolloutValueDataset,
    validation: RolloutValueDataset,
) -> None:
    parent_manifest_hash = checksum_file(config.parent_model_dir / "model.json")
    if parent_manifest_hash != ADR66_PARENT_MANIFEST_BLAKE3:
        raise ValueError("ADR66 parent model manifest is not the preregistered artifact")
    if config.implementation_smoke:
        if (
            train.manifest_blake3 != ADR66_SMOKE_MANIFEST_BLAKE3
            or validation.manifest_blake3 != ADR66_SMOKE_MANIFEST_BLAKE3
            or train.split != "train"
            or validation.split != "train"
            or train.manifest["first_game_index"] != 93_000
            or train.manifest["completed_games"] != 1
            or train.manifest["teacher"]["rollouts"] != 32
        ):
            raise ValueError("ADR66 implementation smoke is not the authorized R32 artifact")
        return
    if train.manifest_blake3 != ADR66_TRAIN_MANIFEST_BLAKE3:
        raise ValueError("ADR66 train dataset is not the preregistered ADR65 split")
    if train.split != "train" or validation.split != "validation":
        raise ValueError("ADR66 datasets use incorrect splits")
    if train.manifest["teacher"] != validation.manifest["teacher"]:
        raise ValueError("ADR66 train and validation teachers differ")
    teacher = train.manifest["teacher"]
    if (
        teacher["rollouts"] != 600
        or teacher["trace_modulus"] != 8
        or teacher["parent_model_manifest_blake3"] != ADR66_PARENT_MANIFEST_BLAKE3
    ):
        raise ValueError("ADR66 teacher protocol differs from preregistration")
    if (
        train.manifest["first_game_index"] != ADR66_TRAIN_FIRST_GAME
        or train.manifest["completed_games"] != ADR66_TRAIN_GAMES
        or validation.manifest["first_game_index"] != ADR66_VALIDATION_FIRST_GAME
        or validation.manifest["completed_games"] != ADR66_VALIDATION_GAMES
    ):
        raise ValueError("ADR66 dataset game ranges differ from preregistration")
    if (
        train.trajectory_count < ADR66_MIN_TRAIN_TRAJECTORIES
        or validation.trajectory_count < ADR66_MIN_VALIDATION_TRAJECTORIES
        or train.root_count <= 0
        or validation.root_count <= 0
    ):
        raise ValueError("ADR66 datasets do not meet preregistered minimums")


def _build_run_manifest(
    config: JointRolloutTrainingConfig,
    train: RolloutValueDataset,
    validation: RolloutValueDataset,
    parent_manifest: dict[str, Any],
    parent_identity: dict[str, str],
) -> dict[str, Any]:
    training = asdict(config)
    for field in (
        "parent_model_dir",
        "train_dataset",
        "validation_dataset",
        "run_dir",
        "derived_model_dir",
    ):
        training[field] = str(getattr(config, field).resolve())
    training.update(
        {
            "trajectory_huber_delta": ROLLOUT_VALUE_HUBER_DELTA,
            "root_centered_huber_delta": ROLLOUT_ROOT_HUBER_DELTA,
            "root_selected_weight": ROLLOUT_ROOT_SELECTED_WEIGHT,
            "root_teacher_weight": ROLLOUT_ROOT_TEACHER_WEIGHT,
            "root_teacher_temperature": ROLLOUT_ROOT_TEACHER_TEMPERATURE,
        }
    )
    return {
        "schema_version": 1,
        "kind": "mlx-joint-return-ranking-finetune-v1",
        "training": training,
        "datasets": {
            "teacher": train.manifest["teacher"],
            "parent_manifest_blake3": parent_identity["manifest_blake3"],
            "parent_model_blake3": parent_identity["model_blake3"],
            "train_manifest_blake3": train.manifest_blake3,
            "validation_manifest_blake3": validation.manifest_blake3,
            "train_trajectory_records": train.trajectory_count,
            "train_root_records": train.root_count,
            "validation_trajectory_records": validation.trajectory_count,
            "validation_root_records": validation.root_count,
        },
        "parent": {
            "manifest": parent_manifest,
            **parent_identity,
        },
        "runtime": {
            "mlx_version": version("mlx"),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "device": str(mx.default_device()),
        },
        "source": source_provenance(Path(__file__).resolve().parents[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--derived-model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--trajectory-batch-size", type=int, default=512)
    parser.add_argument("--root-group-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20_260_621)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=4)
    parser.add_argument("--implementation-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_joint_rollout(JointRolloutTrainingConfig(**vars(args)))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
