#!/usr/bin/env python3
"""Run the fail-closed 100-step packed-pipe MLX production gate.

The gate first performs one explicitly unmeasured initialization step so native
MLX can load and compile before the zero-swap baseline is frozen.  It then
constructs a fresh model at step zero, checkpoints after measured step 99,
tears down every producer, resumes from the verified checkpoint with fresh
producers, and executes measured step 100.  Measured wall time therefore
includes compact replay, Rust encoding, pipe transport, MLX
forward/backward/update, checkpoint publication, verification, and exact cursor
replay without conflating one-time process initialization with training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
from cascadia_mlx.r2_map_pipe_dataset import R2MapPackedPipeDatasetAdapter
from cascadia_mlx.r2_map_train import R2MapTrainer, R2MapTrainerConfig
from cascadia_mlx.r2_map_training_resources import (
    R2MapTrainingResourceMonitor,
    system_swap_used_bytes,
    validate_training_resource_receipt,
)
from cascadia_mlx.r2_map_verify import verify_r2_map_checkpoint


def _canonical_blake3(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return blake3.blake3(payload).hexdigest()


def _adapter(arguments: argparse.Namespace) -> R2MapPackedPipeDatasetAdapter:
    return R2MapPackedPipeDatasetAdapter(
        index=arguments.compact_index,
        shard_root=arguments.compact_shard_root,
        exporter=arguments.compact_exporter,
        validated_aggregate_receipt=arguments.validated_aggregate_receipt,
        validated_packing_receipt=arguments.validated_packing_receipt,
        group_batch_size=arguments.group_batch_size,
        maximum_candidates_per_batch=arguments.maximum_candidates_per_batch,
        sampler_seed=arguments.seed,
    )


def _config(
    arguments: argparse.Namespace,
    adapter: R2MapPackedPipeDatasetAdapter,
) -> R2MapTrainerConfig:
    return R2MapTrainerConfig(
        run_dir=arguments.run_dir,
        run_id=arguments.run_id,
        branch_id="main",
        source_blake3=arguments.source_blake3,
        dataset_blake3=adapter.dataset_blake3,
        adapter_protocol_id=adapter.protocol_id,
        group_batch_size=arguments.group_batch_size,
        maximum_candidates_per_batch=arguments.maximum_candidates_per_batch,
        learning_rate=3e-5,
        minimum_learning_rate=3e-6,
        warmup_steps=10,
        schedule_steps=arguments.production_steps,
        loss_event_interval_steps=10,
        seed=arguments.seed,
        auxiliary_loss_weights={
            "components": 0.25,
            "bootstrap_policy": 0.0,
            "opponent_next_action": 0.05,
            "market_survival": 0.05,
            "market_decision_policy": 0.10,
        },
    )


def _assert_finite(record: dict[str, Any] | None) -> None:
    if record is None:
        return
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or any(
        not isinstance(value, int | float) or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise RuntimeError("packed-pipe benchmark produced a non-finite metric")


def _wait_for_stable_swap_baseline(
    *,
    consecutive_samples: int = 5,
    sample_seconds: float = 2.0,
    maximum_wait_seconds: float = 60.0,
) -> dict[str, Any]:
    """Require macOS swap usage to settle before freezing the measured gate."""
    started = time.monotonic()
    history: list[int] = []
    stable = 0
    previous: int | None = None
    while time.monotonic() - started <= maximum_wait_seconds:
        observed = system_swap_used_bytes()
        history.append(observed)
        stable = stable + 1 if observed == previous else 1
        previous = observed
        if stable >= consecutive_samples:
            return {
                "stable_swap_bytes": observed,
                "samples": history,
                "elapsed_seconds": time.monotonic() - started,
            }
        time.sleep(sample_seconds)
    raise RuntimeError("system swap did not stabilize before the measured gate")


def _materialize_optimizer_state(trainer: R2MapTrainer) -> None:
    """Allocate lazy Adam moments before the measured zero-swap boundary."""
    trainer.optimizer.init(trainer.model.trainable_parameters())
    mx.eval(trainer.model.parameters(), trainer.optimizer.state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-index", type=Path, required=True)
    parser.add_argument("--compact-shard-root", type=Path, required=True)
    parser.add_argument("--compact-exporter", type=Path, required=True)
    parser.add_argument("--validated-aggregate-receipt", type=Path, required=True)
    parser.add_argument("--validated-packing-receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-blake3", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--group-batch-size", type=int, default=128)
    parser.add_argument("--maximum-candidates-per-batch", type=int, default=16_384)
    parser.add_argument("--production-steps", type=int, default=37_807)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_260_618)
    arguments = parser.parse_args()
    if arguments.steps != 100:
        parser.error("the production gate is frozen at exactly 100 optimizer steps")
    if arguments.run_dir.exists():
        parser.error("benchmark run directory already exists; never overwrite gate evidence")
    arguments.run_dir.mkdir(parents=True)

    warmup_started = time.monotonic()
    with _adapter(arguments) as warmup_adapter:
        warmup_config = replace(
            _config(arguments, warmup_adapter),
            run_dir=arguments.run_dir / "initialization-warmup",
            run_id=f"{arguments.run_id}-initialization-warmup",
            loss_event_interval_steps=1,
        )
        warmup_trainer = R2MapTrainer(warmup_config, warmup_adapter)
        warmup_record = warmup_trainer.step()
        _assert_finite(warmup_record)
        if warmup_trainer.global_step != 1 or warmup_record is None:
            raise RuntimeError("packed-pipe initialization warmup did not complete one step")
    warmup_receipt = {
        "steps": 1,
        "elapsed_seconds": time.monotonic() - warmup_started,
        "batch_identity": warmup_record["batch_identity"],
        "total_loss": warmup_record["metrics"]["total_loss"],
        "measured": False,
        "checkpoint_written": False,
    }

    records: list[dict[str, Any]] = []

    with _adapter(arguments) as first_adapter:
        config = _config(arguments, first_adapter)
        trainer = R2MapTrainer(config, first_adapter)
        _materialize_optimizer_state(trainer)
        stable_swap_baseline = _wait_for_stable_swap_baseline()
        monitor = R2MapTrainingResourceMonitor.start()
        initial_sample = monitor.sample()
        started = time.monotonic()
        initial_batch_identity = trainer.peek_next_batch_identity()
        for _ in range(99):
            record = trainer.step()
            _assert_finite(record)
            if record is not None:
                records.append(record)
                monitor.sample()
        if trainer.global_step != 99:
            raise RuntimeError("packed-pipe benchmark did not reach step 99")
        checkpoint_99 = trainer.save_checkpoint(validation=None)
        verification_99 = verify_r2_map_checkpoint(
            checkpoint_99,
            run_dir=arguments.run_dir,
            adapter=first_adapter,
            mark_last_verified=True,
        )
        resume_batch_identity = trainer.peek_next_batch_identity()

    with _adapter(arguments) as resumed_adapter:
        config = _config(arguments, resumed_adapter)
        resumed = R2MapTrainer.resume(config, resumed_adapter, pointer="last_verified")
        if resumed.global_step != 99 or resumed.peek_next_batch_identity() != resume_batch_identity:
            raise RuntimeError("packed-pipe verified resume cursor or batch identity differs")
        record = resumed.step()
        _assert_finite(record)
        if record is not None:
            records.append(record)
        if resumed.global_step != 100:
            raise RuntimeError("packed-pipe benchmark did not reach step 100")
        checkpoint_100 = resumed.save_checkpoint(validation=None)
        verification_100 = verify_r2_map_checkpoint(
            checkpoint_100,
            run_dir=arguments.run_dir,
            adapter=resumed_adapter,
            mark_last_verified=True,
        )
        final_batch_identity = resumed.peek_next_batch_identity()
        training_counters = dict(resumed.training_counters)

    final_sample = monitor.sample()
    elapsed_seconds = time.monotonic() - started
    resource_receipt = validate_training_resource_receipt(monitor.receipt())
    receipt = {
        "schema_id": "r2-map-packed-pipe-100-step-gate-v1",
        "schema_version": 1,
        "result": "pass",
        "run_id": arguments.run_id,
        "source_blake3": arguments.source_blake3,
        "dataset_blake3": config.dataset_blake3,
        "image_id": arguments.image_id,
        "initialization_warmup": warmup_receipt,
        "stable_swap_baseline": stable_swap_baseline,
        "pipe_protocol_id": "r2-map-packed-batch-pipe-v1",
        "focal_seat_rule": "global-game-index-mod-4",
        "steps": 100,
        "restart_after_step": 99,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": 100.0 / elapsed_seconds,
        "seconds_per_step": elapsed_seconds / 100.0,
        "production_steps": arguments.production_steps,
        "projected_production_seconds": elapsed_seconds * arguments.production_steps / 100.0,
        "initial_batch_identity": initial_batch_identity,
        "resume_batch_identity": resume_batch_identity,
        "final_batch_identity": final_batch_identity,
        "loss_records": records,
        "training_counters": training_counters,
        "checkpoint_99": {
            "path": str(checkpoint_99),
            "manifest_blake3": verification_99["checkpoint_manifest_blake3"],
            "verification_id": verification_99["verification_id"],
        },
        "checkpoint_100": {
            "path": str(checkpoint_100),
            "manifest_blake3": verification_100["checkpoint_manifest_blake3"],
            "verification_id": verification_100["verification_id"],
        },
        "initial_resource_sample": initial_sample,
        "final_resource_sample": final_sample,
        "resource_receipt": resource_receipt,
        "expanded_window_files": False,
    }
    receipt["receipt_blake3"] = _canonical_blake3(receipt)
    destination = arguments.run_dir / "report.json"
    temporary = arguments.run_dir / "report.json.partial"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
