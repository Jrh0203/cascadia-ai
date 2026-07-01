"""Resumable QAT trainer for the V3 NNUE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)
from mlx.utils import tree_map

from .contracts import V3MlxConfig
from .dataset import SparseWidths, synthetic_batch
from .model import V3Nnue, v3_loss
from .provenance import training_source_identity
from .stream import RustBatchStream

ENGINEERING_RUN_MANIFEST_SCHEMA = "cascadia-v3-engineering-training-run-v1"
SCIENTIFIC_RUN_MANIFEST_SCHEMA = "cascadia-v3-scientific-training-run-v1"
CAMPAIGN_ID = "cascadia-v3-radius7-stockfish-nnue-v1"


@dataclass(frozen=True)
class V3TrainingConfig:
    examples: int
    logical_batch_size: int
    microbatch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    widths: SparseWidths

    def validate(self) -> None:
        if min(self.examples, self.logical_batch_size, self.microbatch_size, self.epochs) <= 0:
            raise ValueError("training sizes and epochs must be positive")
        if self.logical_batch_size % self.microbatch_size:
            raise ValueError("logical batch must be divisible by microbatch size")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        self.widths.validate()


def _add_gradients(left: object | None, right: object) -> object:
    if left is None:
        return right
    return tree_map(lambda a, b: a + b, left, right)


def train_engineering_epoch(
    model: V3Nnue,
    optimizer: optim.Optimizer,
    config: V3TrainingConfig,
    state: TrainerState,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_every_steps: int = 0,
    checkpoint_metadata: dict[str, object] | None = None,
) -> tuple[TrainerState, dict[str, float]]:
    config.validate()
    loss_and_grad = nn.value_and_grad(model, v3_loss)
    started = time.perf_counter()
    total_loss = 0.0
    completed = 0
    logical_steps = math.ceil(config.examples / config.logical_batch_size)
    for step in range(state.global_step, logical_steps * config.epochs):
        epoch = step // logical_steps
        batch_in_epoch = step % logical_steps
        remaining = config.examples - batch_in_epoch * config.logical_batch_size
        logical_size = min(config.logical_batch_size, remaining)
        if logical_size <= 0:
            continue
        gradients: object | None = None
        step_loss = 0.0
        microbatches = 0
        for offset in range(0, logical_size, config.microbatch_size):
            size = min(config.microbatch_size, logical_size - offset)
            batch = synthetic_batch(
                model.config,
                size,
                config.seed
                + epoch * config.examples
                + batch_in_epoch * config.logical_batch_size
                + offset,
                config.widths,
            )
            value, gradient = loss_and_grad(model, batch)
            mx.eval(value, gradient)
            gradients = _add_gradients(gradients, gradient)
            step_loss += float(value.item())
            microbatches += 1
        assert gradients is not None
        gradient_divisor = microbatches
        gradients = tree_map(
            lambda value, divisor=gradient_divisor: value / divisor,
            gradients,
        )
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += step_loss / microbatches
        completed += logical_size
        state = TrainerState(
            epoch=epoch,
            batch_in_epoch=batch_in_epoch + 1,
            global_step=step + 1,
            elapsed_seconds=state.elapsed_seconds + (time.perf_counter() - started),
        )
        if (
            checkpoint_dir is not None
            and checkpoint_every_steps > 0
            and state.global_step % checkpoint_every_steps == 0
        ):
            save_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                state,
                metadata=checkpoint_metadata,
            )
        if batch_in_epoch + 1 == logical_steps:
            state = TrainerState(
                epoch=epoch + 1,
                batch_in_epoch=0,
                global_step=step + 1,
                elapsed_seconds=state.elapsed_seconds,
            )
    elapsed = time.perf_counter() - started
    return state, {
        "examples": float(completed),
        "elapsed_seconds": elapsed,
        "examples_per_second": completed / max(elapsed, 1e-9),
        "mean_loss": total_loss / max(state.global_step, 1),
    }


def train_native_epoch(
    model: V3Nnue,
    optimizer: optim.Optimizer,
    state: TrainerState,
    *,
    stream_binary: Path,
    inputs: list[Path],
    batch_size: int,
    epochs: int,
    checkpoint_dir: Path,
    checkpoint_every_steps: int,
    checkpoint_metadata: dict[str, object],
    stop_after_global_step: int | None = None,
    d6_cycle: bool = False,
    allow_scientific_data: bool = False,
    campaign_state: Path | None = None,
    cycle: int | None = None,
    teacher_lambda: float | None = None,
) -> tuple[TrainerState, dict[str, float]]:
    started = time.perf_counter()
    prior_elapsed = state.elapsed_seconds
    loss_and_grad = nn.value_and_grad(model, v3_loss)
    stream = RustBatchStream(
        stream_binary,
        inputs,
        model.config,
        batch_size=batch_size,
        epochs=epochs,
        allow_scientific_data=allow_scientific_data,
        d6_cycle=d6_cycle,
        campaign_state=campaign_state,
        cycle=cycle,
        teacher_lambda=teacher_lambda,
        expansion_threads=8,
    )
    skipped = 0
    examples = 0
    loss_sum = 0.0
    steps = 0
    interrupted = False
    checkpoint_seconds = 0.0
    try:
        for batch_index, batch in enumerate(stream):
            if batch_index < state.global_step:
                skipped += 1
                continue
            if stop_after_global_step is not None and state.global_step >= stop_after_global_step:
                interrupted = True
                break
            loss, gradients = loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
            rows = int(batch.targets.shape[0])
            examples += rows
            loss_sum += float(loss.item())
            steps += 1
            state = TrainerState(
                epoch=0,
                batch_in_epoch=batch_index + 1,
                global_step=batch_index + 1,
                elapsed_seconds=prior_elapsed + (time.perf_counter() - started),
            )
            if checkpoint_every_steps and state.global_step % checkpoint_every_steps == 0:
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    checkpoint_dir,
                    model,
                    optimizer,
                    state,
                    metadata=checkpoint_metadata,
                )
                prune_checkpoints(checkpoint_dir, keep_recent=2)
                checkpoint_seconds += time.perf_counter() - checkpoint_started
    finally:
        stream.close()
    elapsed = time.perf_counter() - started
    if not interrupted:
        state = TrainerState(
            epoch=epochs,
            batch_in_epoch=0,
            global_step=state.global_step,
            elapsed_seconds=prior_elapsed + elapsed,
        )
    return state, {
        "examples": float(examples),
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-9),
        "mean_loss": loss_sum / max(steps, 1),
        "skipped_resume_batches": float(skipped),
        "interrupted": interrupted,
        "next_batch_index": state.global_step,
        "checkpoint_seconds": checkpoint_seconds,
    }


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: asdict(item),
    ).encode()


def _validate_phase2_authorization(path: Path, cycle: int | None) -> dict[str, object]:
    authorization = json.loads(path.read_text())
    state_hash = authorization.pop("state_sha256", None)
    if state_hash != hashlib.sha256(_canonical(authorization)).hexdigest():
        raise ValueError("scientific training campaign state checksum is invalid")
    expected_phase = "bootstrap_training" if cycle is None else f"cycle-{cycle:02d}-training"
    readiness_path = Path(str(authorization.get("readiness_path", "")))
    readiness = json.loads(readiness_path.read_text()) if readiness_path.is_file() else {}
    readiness_hash = readiness.pop("readiness_sha256", None)
    if readiness_hash != hashlib.sha256(_canonical(readiness)).hexdigest():
        raise ValueError("scientific training readiness checksum is invalid")
    if (
        authorization.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("part") != 2
        or authorization.get("phase") != expected_phase
        or authorization.get("phase2_authorized") is not True
        or authorization.get("approved_readiness_sha256")
        != readiness_hash
        or readiness.get("schema_id") != "cascadia-v3-part1-readiness-v1"
        or readiness.get("campaign_id") != CAMPAIGN_ID
        or readiness.get("status") not in {"green", "red"}
    ):
        raise ValueError("scientific training requires the exact authorized campaign phase")
    return authorization


def _run_manifest(
    args: argparse.Namespace,
    model_config: V3MlxConfig,
    training: V3TrainingConfig,
) -> dict[str, object]:
    datasets = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "blake3": _checksum(path),
        }
        for path in args.dataset
    ]
    value: dict[str, object] = {
        "schema_id": (
            SCIENTIFIC_RUN_MANIFEST_SCHEMA
            if args.allow_scientific_data
            else ENGINEERING_RUN_MANIFEST_SCHEMA
        ),
        "scientific_eligible": args.allow_scientific_data,
        "dataset_class": "scientific" if args.allow_scientific_data else "engineering_smoke",
        "model_config": model_config.to_dict(),
        "training_config": asdict(training),
        "datasets": datasets,
        "stream_binary": (
            {
                "path": str(args.batch_stream_binary.resolve()),
                "bytes": args.batch_stream_binary.stat().st_size,
                "blake3": _checksum(args.batch_stream_binary),
            }
            if args.batch_stream_binary is not None
            else None
        ),
        "loader": {
            "schema_id": "cascadia-v3-native-csr-stream-v1" if datasets else "synthetic-v1",
            "deterministic_order": True,
            "cursor": "global_step",
            "epochs": args.epochs,
            "batch_size": args.logical_batch_size,
            "teacher_lambda": args.teacher_lambda,
        },
        "optimizer": {
            "kind": "adamw",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "training_source_identity": training_source_identity(),
        "rng": {
            "training_seed": args.seed,
            "stochastic_layers": False,
            "online_augmentation": args.d6_cycle,
            "d6_cycle": args.d6_cycle,
        },
    }
    if args.training_origin != "engineering-smoke":
        value["training_origin"] = args.training_origin
    if args.cycle is not None:
        value["expert_cycle"] = args.cycle
    if args.init_run_dir is not None:
        value["initial_checkpoint"] = {
            "run_dir": str(args.init_run_dir.resolve()),
            "latest": json.loads((args.init_run_dir / "latest.json").read_text()),
            "run_manifest_blake3": json.loads(
                (args.init_run_dir / "run-manifest.json").read_text()
            )["canonical_blake3"],
        }
    value["canonical_blake3"] = blake3.blake3(_canonical(value)).hexdigest()
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=asdict) + "\n")
    os.replace(temporary, path)


def _bind_run_manifest(run_dir: Path, proposed: dict[str, object], resume: bool) -> str:
    path = run_dir / "run-manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != proposed:
            raise ValueError("resume refused: V3 run manifest differs from the checkpoint origin")
    elif resume:
        raise ValueError("resume refused: V3 run manifest is missing")
    else:
        _write_json_atomic(path, proposed)
    return str(proposed["canonical_blake3"])


def _verify_checkpoint_binding(checkpoint: Path, run_manifest_blake3: str) -> None:
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    metadata = manifest.get("metadata", {})
    if metadata.get("run_manifest_blake3") != run_manifest_blake3:
        raise ValueError("resume refused: checkpoint is not bound to this V3 run manifest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=160_000)
    parser.add_argument("--logical-batch-size", type=int, default=16_384)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=73_001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-run-dir", type=Path)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--batch-stream-binary", type=Path)
    parser.add_argument("--d6-cycle", action="store_true")
    parser.add_argument("--allow-scientific-data", action="store_true")
    parser.add_argument("--campaign-state", type=Path)
    parser.add_argument("--training-origin", default="engineering-smoke")
    parser.add_argument("--cycle", type=int)
    parser.add_argument(
        "--teacher-lambda",
        type=float,
        help="teacher/realized target blend for teacher-labeled rows (0..1)",
    )
    parser.add_argument(
        "--stop-after-global-step",
        type=int,
        help="controlled interruption after this exact global step (engineering recovery smoke)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.dataset and args.batch_stream_binary is None:
        raise SystemExit("--batch-stream-binary is required with --dataset")
    if args.allow_scientific_data:
        if args.campaign_state is None:
            raise SystemExit("scientific training requires --campaign-state")
        try:
            _validate_phase2_authorization(args.campaign_state, args.cycle)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"scientific training authorization refused: {error}") from error
    elif args.campaign_state is not None:
        raise SystemExit("--campaign-state is only valid with --allow-scientific-data")
    if args.cycle is not None and not 1 <= args.cycle <= 10:
        raise SystemExit("--cycle must be in 1..=10")
    if args.cycle is not None and not args.allow_scientific_data:
        raise SystemExit("expert --cycle requires scientific training authorization")
    if args.teacher_lambda is not None and not 0.0 <= args.teacher_lambda <= 1.0:
        raise SystemExit("--teacher-lambda must be within [0, 1]")
    if args.resume and args.init_run_dir is not None:
        raise SystemExit("--resume and --init-run-dir are mutually exclusive")
    if args.init_run_dir is not None and not (args.init_run_dir / "latest.json").is_file():
        raise SystemExit("--init-run-dir has no verified checkpoint")
    for path in [args.feature_manifest, *args.dataset]:
        if not path.is_file():
            raise SystemExit(f"required V3 input does not exist: {path}")
    if args.batch_stream_binary is not None and not args.batch_stream_binary.is_file():
        raise SystemExit(f"V3 batch stream binary does not exist: {args.batch_stream_binary}")
    feature = json.loads(args.feature_manifest.read_text())
    model_config = V3MlxConfig(
        opportunity_feature_rows=feature["opportunity_feature_rows"],
        opportunity_training_factor_rows=feature["opportunity_training_factor_rows"],
    )
    training = V3TrainingConfig(
        examples=args.examples,
        logical_batch_size=args.logical_batch_size,
        microbatch_size=args.microbatch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        widths=SparseWidths(),
    )
    training.validate()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stop_after_global_step is not None and args.stop_after_global_step <= 0:
        raise SystemExit("--stop-after-global-step must be positive")
    run_manifest = _run_manifest(args, model_config, training)
    run_manifest_blake3 = _bind_run_manifest(args.run_dir, run_manifest, args.resume)
    if args.resume and (args.run_dir / "latest.json").is_file():
        model, optimizer, state, loaded_checkpoint = load_latest_checkpoint_with_factory(
            args.run_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
        )
        _verify_checkpoint_binding(loaded_checkpoint, run_manifest_blake3)
    elif args.init_run_dir is not None:
        origin_manifest = json.loads((args.init_run_dir / "run-manifest.json").read_text())
        origin_optimizer = origin_manifest["optimizer"]
        model, _, _, _ = load_latest_checkpoint_with_factory(
            args.init_run_dir,
            learning_rate=float(origin_optimizer["learning_rate"]),
            weight_decay=float(origin_optimizer["weight_decay"]),
            model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
        )
        if model.config != model_config:
            raise SystemExit("initial checkpoint architecture differs from this V3 run")
        optimizer = optim.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        state = TrainerState()
    else:
        # Origin initialization is part of the resumable scientific contract,
        # not merely a data-loader seed. Without this, two executions of the
        # same immutable run manifest begin from different parameter tensors.
        mx.random.seed(args.seed)
        model = V3Nnue(model_config)
        optimizer = optim.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        state = TrainerState()
    checkpoint_metadata = {
        "dataset_class": "scientific" if args.allow_scientific_data else "engineering_smoke",
        "scientific_eligible": args.allow_scientific_data,
        "run_manifest_blake3": run_manifest_blake3,
        "teacher_lambda": args.teacher_lambda,
    }
    if args.dataset:
        assert args.batch_stream_binary is not None
        state, metrics = train_native_epoch(
            model,
            optimizer,
            state,
            stream_binary=args.batch_stream_binary,
            inputs=args.dataset,
            batch_size=args.logical_batch_size,
            epochs=args.epochs,
            checkpoint_dir=args.run_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            checkpoint_metadata=checkpoint_metadata,
            stop_after_global_step=args.stop_after_global_step,
            d6_cycle=args.d6_cycle,
            allow_scientific_data=args.allow_scientific_data,
            campaign_state=args.campaign_state,
            cycle=args.cycle,
            teacher_lambda=args.teacher_lambda,
        )
    else:
        state, metrics = train_engineering_epoch(
            model,
            optimizer,
            training,
            state,
            checkpoint_dir=args.run_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            checkpoint_metadata=checkpoint_metadata,
        )
    checkpoint_metadata["exact_next_global_step"] = state.global_step
    checkpoint_started = time.perf_counter()
    checkpoint = save_checkpoint(
        args.run_dir,
        model,
        optimizer,
        state,
        metadata=checkpoint_metadata,
    )
    metrics["final_checkpoint_seconds"] = time.perf_counter() - checkpoint_started
    report = {
        "schema_id": (
            "cascadia-v3-scientific-training-v1"
            if args.allow_scientific_data
            else "cascadia-v3-engineering-training-v1"
        ),
        "scientific_eligible": args.allow_scientific_data,
        "training_origin": args.training_origin,
        "expert_cycle": args.cycle,
        "model_config": model_config.to_dict(),
        "training_config": asdict(training),
        "state": asdict(state),
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "run_manifest": str(args.run_dir / "run-manifest.json"),
        "run_manifest_blake3": run_manifest_blake3,
        "mlx_device": str(mx.default_device()),
    }
    report_name = (
        "training-report.json"
        if args.allow_scientific_data
        else "engineering-training-report.json"
    )
    _write_json_atomic(args.run_dir / report_name, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
