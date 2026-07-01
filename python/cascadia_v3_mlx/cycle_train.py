"""Exact three-pass MLX fine-tuner for one Cascadia V3 expert-cycle origin."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)

from .campaign_train import (
    CAMPAIGN_ID,
    LOSS_PROGRESS_EVERY_STEPS,
    MLX_CACHE_LIMIT_BYTES,
    _assert_storage,
    _bind_manifest,
    _dataset_identity,
    _ordered,
    _write_atomic,
)
from .checkpoint_eval import _optimizer_settings
from .contracts import FEATURE_SCALE, V3MlxConfig
from .model import (
    ACCUMULATOR_HEADROOM_COEFFICIENT,
    ACCUMULATOR_HEADROOM_LIMIT,
    CsrBatch,
    V3Nnue,
    v3_loss,
)
from .provenance import training_source_identity
from .stream import RustBatchStream

PASSES = (3e-5, 3e-5, 1e-5)
EXPOSURES_PER_PASS = 400_000
RUN_SCHEMA = "cascadia-v3-expert-cycle-origin-run-v1"
EXPANSION_THREAD_BUDGET = 9
# Relative full-pass cost for the four-source Cycle-1 topology. Later cycles
# add a 120K-example recent-replay source. Live Cycle-7 traces showed that its
# one-thread producer remained alone for 6-12 minutes at pass tails. Moving a
# thread from older_teacher to recent in Cycle 8 merely transferred that tail:
# the 40K older-teacher stream is expensive per row and became the sole active
# producer, while both broad streams had already exited. Current broad is the
# cheapest measured stream, so later cycles give it one thread and retain two
# for both teacher streams, recent replay, and older broad. Rayon indexed
# expansion is order-preserving, so this changes latency only; source quotas,
# order, augmentation, batches, and optimizer steps remain identical.
SOURCE_PREPROCESSING_COST = {
    "current_broad": 2,
    "current_teacher": 2,
    "recent": 1,
    "older_broad": 2,
    "older_teacher": 2,
}
LATER_CYCLE_PREPROCESSING_COST = {
    **SOURCE_PREPROCESSING_COST,
    "current_broad": 1,
    "recent": 2,
}
# A cycle's 2,500 labeled roots yield fewer than the scheduled 80,000
# candidate exposures in one traversal.  Replaying across the complete D6
# group is the registered online augmentation, and max_examples still stops
# every source at its exact quota.  Twelve traversals also guarantee that the
# per-source phase x score-quantile balancer can fill its quota without
# silently weakening the requested mixture.
SOURCE_REPLAY_EPOCHS = 12


class CycleTrainingError(ValueError):
    """The cycle data mix, parent, or resumable state is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _authorization(path: Path, cycle: int) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("state_sha256", None)
    if (
        recorded != hashlib.sha256(_canonical(value)).hexdigest()
        or value.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("part") != 2
        or value.get("phase") != f"cycle-{cycle:02d}-training"
        or value.get("phase2_authorized") is not True
        or value.get("protected_seed_values_opened") is not False
        or value.get("approved_readiness_sha256") != value.get("readiness_sha256")
    ):
        raise CycleTrainingError("cycle training is not checksum-authorized")
    value["state_sha256"] = recorded
    return value


def source_quotas(cycle: int) -> dict[str, int]:
    if not 1 <= cycle <= 10:
        raise CycleTrainingError("cycle is outside 1..=10")
    quotas = {
        "current_broad": 120_000,
        "current_teacher": 80_000,
        "recent": 120_000 if cycle > 1 else 0,
        "older_broad": 40_000 if cycle > 1 else 100_000,
        "older_teacher": 40_000 if cycle > 1 else 100_000,
    }
    if sum(quotas.values()) != EXPOSURES_PER_PASS or any(value % 32 for value in quotas.values()):
        raise AssertionError("cycle source quotas drifted from the exact 400K pass")
    return quotas


def source_thread_allocations(
    quotas: dict[str, int], total_threads: int = EXPANSION_THREAD_BUDGET
) -> dict[str, int]:
    """Share John1's preprocessing CPUs by measured producer work."""
    active = {source: quota for source, quota in quotas.items() if quota > 0}
    if not active or total_threads < len(active):
        raise CycleTrainingError("expansion thread budget cannot cover active sources")
    costs = (
        LATER_CYCLE_PREPROCESSING_COST
        if active.get("recent", 0) > 0
        else SOURCE_PREPROCESSING_COST
    )
    unknown = set(active) - set(costs)
    if unknown:
        raise CycleTrainingError(f"preprocessing cost is missing for sources: {sorted(unknown)}")
    allocations = {source: 1 for source in active}
    remaining = total_threads - len(active)
    # Greedy weighted apportionment is deterministic. Expansion preserves
    # source order across Rayon thread counts, so this changes only latency.
    while remaining:
        source = max(
            active,
            key=lambda name: (
                costs[name] / allocations[name],
                costs[name],
                name,
            ),
        )
        allocations[source] += 1
        remaining -= 1
    return allocations


def active_source_schedule(
    quotas: dict[str, int], allocations: dict[str, int]
) -> tuple[tuple[str, int, int], ...]:
    """Bind threads only to sources with a nonzero quota."""
    active = []
    for source, examples in quotas.items():
        if examples == 0:
            continue
        threads = allocations.get(source)
        if threads is None or threads <= 0:
            raise CycleTrainingError(f"active cycle source {source} has no thread allocation")
        active.append((source, examples, threads))
    if set(allocations) != {source for source, _, _ in active}:
        raise CycleTrainingError("cycle thread allocations contain an inactive source")
    return tuple(active)


def _source_paths(args: argparse.Namespace) -> dict[str, list[Path]]:
    return {
        "current_broad": args.current_broad,
        "current_teacher": args.current_teacher,
        "recent": args.recent,
        "older_broad": args.older_broad,
        "older_teacher": args.older_teacher,
    }


def _stream(
    *,
    args: argparse.Namespace,
    config: V3MlxConfig,
    source: str,
    paths: list[Path],
    examples: int,
    pass_index: int,
    boundaries: tuple[float, ...] | None,
    expansion_threads: int,
) -> RustBatchStream | None:
    if examples == 0:
        return None
    return RustBatchStream(
        args.batch_stream_binary,
        _ordered(paths, args.seed, pass_index, source),
        config,
        batch_size=args.batch_size,
        epochs=SOURCE_REPLAY_EPOCHS,
        allow_scientific_data=True,
        campaign_state=args.campaign_state,
        cycle=args.cycle,
        teacher_lambda=1.0 if "teacher" in source else None,
        max_examples=examples,
        d6_cycle=True,
        d6_offset=pass_index - 1,
        score_quantile_boundaries=boundaries,
        expansion_threads=expansion_threads,
    )


def _round_robin(
    streams: list[tuple[str, RustBatchStream]],
) -> Iterator[tuple[str, CsrBatch]]:
    active: list[tuple[str, Iterator[CsrBatch]]] = [
        (source, iter(stream)) for source, stream in streams
    ]
    while active:
        remaining = []
        for source, stream in active:
            try:
                yield source, next(stream)
                remaining.append((source, stream))
            except StopIteration:
                pass
        active = remaining


def _measure_boundaries(
    args: argparse.Namespace,
    config: V3MlxConfig,
    paths: dict[str, list[Path]],
    quotas: dict[str, int],
) -> tuple[float, ...]:
    targets = []
    phases = []
    streams = []
    try:
        for source, examples in quotas.items():
            stream = _stream(
                args=args,
                config=config,
                source=source,
                paths=paths[source],
                examples=examples,
                pass_index=1,
                boundaries=None,
                # The quantile census drains one source completely before
                # opening the next, so each sequential producer may use the
                # full preprocessing budget.
                expansion_threads=EXPANSION_THREAD_BUDGET,
            )
            if stream is not None:
                streams.append(stream)
                for batch in stream:
                    targets.append(np.asarray(batch.targets, dtype=np.float32))
                    phases.append(np.asarray(batch.phase_buckets, dtype=np.int32))
    finally:
        for stream in streams:
            stream.close()
    if sum(value.size for value in targets) != EXPOSURES_PER_PASS:
        raise CycleTrainingError("quantile census did not cover the exact scheduled mixture")
    values = np.concatenate(targets)
    phase_values = np.concatenate(phases)
    boundaries = []
    for phase in range(8):
        selected = values[phase_values == phase]
        if selected.size == 0:
            raise CycleTrainingError(f"score-to-go census has no phase {phase} rows")
        triple = tuple(float(value) for value in np.quantile(selected, (0.25, 0.5, 0.75)))
        if not triple[0] < triple[1] < triple[2]:
            raise CycleTrainingError(
                f"phase {phase} score-to-go quartiles are not distinct: {triple}"
            )
        boundaries.extend(triple)
    return tuple(boundaries)


def _load_parent(run_dir: Path) -> tuple[V3Nnue, dict[str, Any], Path]:
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    learning_rate, weight_decay = _optimizer_settings(manifest)
    model, _, _, checkpoint = load_latest_checkpoint_with_factory(
        run_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
    )
    return model, manifest, checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
    authorization = _authorization(args.campaign_state, args.cycle)
    paths = _source_paths(args)
    quotas = source_quotas(args.cycle)
    for source, examples in quotas.items():
        if examples and not paths[source]:
            raise CycleTrainingError(f"cycle source {source} is empty")
    required = [
        args.feature_manifest,
        args.batch_stream_binary,
        args.parent_run_dir / "run-manifest.json",
        *[path for values in paths.values() for path in values],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CycleTrainingError(f"cycle training inputs are missing: {missing[:3]}")
    _assert_storage(args.campaign_root, args.checkpoint_bytes)
    feature = json.loads(args.feature_manifest.read_text())
    config = V3MlxConfig(
        opportunity_feature_rows=feature["opportunity_feature_rows"],
        opportunity_training_factor_rows=feature["opportunity_training_factor_rows"],
    )
    parent, parent_manifest, parent_checkpoint = _load_parent(args.parent_run_dir)
    if parent.config != config:
        raise CycleTrainingError("cycle parent architecture differs from the fixed V3 model")
    boundaries = _measure_boundaries(args, config, paths, quotas)
    thread_allocations = source_thread_allocations(quotas)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_id": RUN_SCHEMA,
        "campaign_state_sha256": authorization["state_sha256"],
        "approved_readiness_sha256": authorization["approved_readiness_sha256"],
        "cycle": args.cycle,
        "origin": args.origin,
        "seed": args.seed,
        "base_learning_rate": PASSES[0],
        "weight_decay": args.weight_decay,
        "accumulator_headroom": {
            "limit_float_units": ACCUMULATOR_HEADROOM_LIMIT,
            "limit_integer_units": int(ACCUMULATOR_HEADROOM_LIMIT * FEATURE_SCALE),
            "coefficient": ACCUMULATOR_HEADROOM_COEFFICIENT,
            "scope": "maximum-absolute-own-and-field-accumulator-per-example",
        },
        "batch_size": args.batch_size,
        "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
        "loss_progress_every_steps": LOSS_PROGRESS_EVERY_STEPS,
        "exposures_per_pass": EXPOSURES_PER_PASS,
        "total_exposures": EXPOSURES_PER_PASS * len(PASSES),
        "passes": list(PASSES),
        "source_quotas_per_pass": quotas,
        "source_replay_epochs": SOURCE_REPLAY_EPOCHS,
        "source_expansion_threads": thread_allocations,
        "phase_score_cells": 32,
        "score_quantile_boundaries": boundaries,
        "feature_manifest": _dataset_identity([args.feature_manifest])[0],
        "batch_stream_binary": _dataset_identity([args.batch_stream_binary])[0],
        "sources": {source: _dataset_identity(values) for source, values in paths.items()},
        "parent": {
            "run_dir": str(args.parent_run_dir.resolve()),
            "run_manifest_blake3": parent_manifest.get("canonical_blake3"),
            "checkpoint": parent_checkpoint.name,
        },
        "training_source_identity": training_source_identity(),
    }
    manifest_hash = _bind_manifest(args.run_dir / "run-manifest.json", manifest, args.resume)
    if args.resume:
        model, optimizer, state, checkpoint = load_latest_checkpoint_with_factory(
            args.run_dir,
            learning_rate=PASSES[0],
            weight_decay=args.weight_decay,
            model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
        )
        checkpoint_value = json.loads((checkpoint / "checkpoint.json").read_text())
        if checkpoint_value.get("metadata", {}).get("run_manifest_blake3") != manifest_hash:
            raise CycleTrainingError("cycle checkpoint is not bound to this run manifest")
    else:
        model = parent
        optimizer = optim.AdamW(learning_rate=PASSES[0], weight_decay=args.weight_decay)
        state = TrainerState()
    loss_and_grad = nn.value_and_grad(model, v3_loss)
    loss_path = args.run_dir / "loss.json"
    samples = (
        list(json.loads(loss_path.read_text()).get("samples", []))
        if args.resume and loss_path.is_file()
        else []
    )
    started = time.perf_counter()
    for pass_offset in range(state.schedule_block, len(PASSES)):
        pass_index = pass_offset + 1
        optimizer.learning_rate = PASSES[pass_offset]
        opened = []
        try:
            for source, examples, expansion_threads in active_source_schedule(
                quotas, thread_allocations
            ):
                stream = _stream(
                    args=args,
                    config=config,
                    source=source,
                    paths=paths[source],
                    examples=examples,
                    pass_index=pass_index,
                    boundaries=boundaries,
                    expansion_threads=expansion_threads,
                )
                if stream is not None:
                    opened.append((source, stream))
            for batch_index, (source, batch) in enumerate(_round_robin(opened)):
                if batch_index < state.batch_in_block:
                    continue
                loss, gradients = loss_and_grad(model, batch)
                optimizer.update(model, gradients)
                mx.eval(loss, model.parameters(), optimizer.state)
                rows = int(batch.targets.shape[0])
                state.global_step += 1
                state.batch_in_block = batch_index + 1
                state.examples_seen += rows
                state.elapsed_seconds += time.perf_counter() - started
                started = time.perf_counter()
                samples.append(
                    {
                        "step": state.global_step,
                        "examples": state.examples_seen,
                        "pass": pass_index,
                        "source": source,
                        "loss": float(loss.item()),
                        "learning_rate": PASSES[pass_offset],
                    }
                )
                if len(samples) > 1_000:
                    samples = samples[-1_000:]
                if state.global_step % LOSS_PROGRESS_EVERY_STEPS == 0:
                    _write_atomic(loss_path, {"samples": samples})
        finally:
            for _, stream in opened:
                stream.close()
        expected = pass_index * EXPOSURES_PER_PASS
        if state.examples_seen != expected:
            raise CycleTrainingError(
                f"cycle pass {pass_index} emitted {state.examples_seen}, expected {expected}"
            )
        state.schedule_block = pass_index
        state.batch_in_block = 0
        _assert_storage(args.campaign_root, args.checkpoint_bytes)
        save_checkpoint(
            args.run_dir,
            model,
            optimizer,
            state,
            metadata={
                "run_manifest_blake3": manifest_hash,
                "completed_pass": pass_index,
                "examples_seen": state.examples_seen,
            },
        )
        prune_checkpoints(args.run_dir, keep_recent=2)
        _write_atomic(loss_path, {"samples": samples})
    report = {
        "schema_id": "cascadia-v3-expert-cycle-origin-training-report-v1",
        "passed": state.schedule_block == len(PASSES),
        "cycle": args.cycle,
        "origin": args.origin,
        "run_manifest_blake3": manifest_hash,
        "examples_seen": state.examples_seen,
        "completed_passes": state.schedule_block,
        "score_quantile_boundaries": boundaries,
        "elapsed_seconds": state.elapsed_seconds,
        "latest_loss": samples[-1] if samples else None,
    }
    _write_atomic(args.run_dir / "training-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--batch-stream-binary", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--current-broad", type=Path, action="append", default=[])
    parser.add_argument("--current-teacher", type=Path, action="append", default=[])
    parser.add_argument("--recent", type=Path, action="append", default=[])
    parser.add_argument("--older-broad", type=Path, action="append", default=[])
    parser.add_argument("--older-teacher", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--checkpoint-bytes", type=int, default=1280 * 1024**2)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.cycle <= 10 or args.seed < 0:
        raise SystemExit("cycle or seed is invalid")
    if args.batch_size <= 0 or args.batch_size % 32:
        raise SystemExit("batch size must be a positive multiple of 32")
    if args.weight_decay < 0:
        raise SystemExit("weight decay must be nonnegative")
    try:
        value = run(args)
    except (CycleTrainingError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
