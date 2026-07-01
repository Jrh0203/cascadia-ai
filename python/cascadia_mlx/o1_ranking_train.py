"""Authorized adapter-only training for ADR 0188."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import subprocess
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)
from cascadia_mlx.o1_ranking_afterstate_cache import O1RankingAfterstateCache
from cascadia_mlx.o1_ranking_cohort import O1RankingCohortCache
from cascadia_mlx.o1_ranking_dataset import (
    O1RankingBatch,
    O1RankingDataset,
)
from cascadia_mlx.o1_ranking_intent_cache import ARMS, O1RankingIntentCache
from cascadia_mlx.o1_ranking_metrics import evaluate_o1_ranking
from cascadia_mlx.o1_ranking_model import (
    O1IntentConditionedRanker,
    O1RankingModelConfig,
    o1_ranking_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.o1_ranking_protocol import (
    ADR_ID,
    CHECKPOINT_STEPS,
    EXPERIMENT_ID,
    GROUPS_PER_STEP,
    LEARNING_RATE,
    MAX_SMOKE_STEPS,
    METRIC_STEPS,
    MLX_CACHE_LIMIT_BYTES,
    PROTOCOL_ID,
    TRAINING_SEED,
    TRAINING_STEPS,
    WAVE_HOSTS,
    WEIGHT_DECAY,
    O1RankingTrainingProtocol,
    normalize_host,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM,
    R3ActionEditMlxCache,
)
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

ADAPTER_PARAMETER_ROOTS = frozenset(
    {
        "intent_projection",
        "intent_fusion",
        "intent_delta",
    }
)


@dataclass(frozen=True)
class O1RankingTrainingConfig:
    """One production, replay, or bounded-smoke invocation."""

    train_dataset: Path
    validation_dataset: Path
    r3_cache: Path
    s1_cache: Path
    cohort: Path
    afterstates: Path
    intent_cache: Path
    warm_start_checkpoint: Path
    run_dir: Path
    output: Path
    arm: str
    wave: str = "primary"
    resume: bool = False
    smoke_steps: int | None = None
    authorization: Path | None = None
    preflight: Path | None = None
    protocol: O1RankingTrainingProtocol = field(
        default_factory=O1RankingTrainingProtocol
    )

    @property
    def production(self) -> bool:
        return self.smoke_steps is None

    @property
    def target_steps(self) -> int:
        return TRAINING_STEPS if self.production else int(self.smoke_steps)

    def validate(self) -> None:
        self.protocol.validate()
        if self.arm not in ARMS:
            raise ValueError("O1 ranking training arm is unknown")
        if self.wave not in WAVE_HOSTS:
            raise ValueError("O1 ranking wave must be primary or rotated")
        if self.production:
            if self.authorization is None or self.preflight is None:
                raise ValueError(
                    "production O1 ranking training requires launch controls"
                )
        elif (
            self.smoke_steps is None
            or self.smoke_steps <= 0
            or self.smoke_steps > MAX_SMOKE_STEPS
            or self.resume
            or self.authorization is not None
            or self.preflight is not None
        ):
            raise ValueError(
                "bounded O1 ranking smoke must be fresh and at most 10 steps"
            )


@dataclass(frozen=True)
class O1RankingSurfaces:
    """All immutable data surfaces joined for one routed arm."""

    r3: R3ActionEditMlxCache
    s1: S1ExactSupplyCache
    cohort: O1RankingCohortCache
    afterstates: O1RankingAfterstateCache
    intent: O1RankingIntentCache
    train: O1RankingDataset
    validation: O1RankingDataset


def run_o1_ranking_training(
    config: O1RankingTrainingConfig,
) -> dict[str, Any]:
    """Train one arm while proving exact-R2 remains unchanged."""
    config.validate()
    mx.set_default_device(mx.gpu)
    runtime = runtime_identity()
    source = source_provenance(Path(__file__).resolve().parents[2])
    if config.production:
        require_production_runtime(runtime)
        expected_host = WAVE_HOSTS[config.wave][config.arm]
        if runtime["host"] != expected_host:
            raise ValueError(
                f"{config.wave} arm {config.arm} must run on "
                f"{expected_host}, not {runtime['host']}"
            )
    surfaces = load_experiment_surfaces(
        config,
        verify_checksums=not config.production,
    )
    warm_start = warm_start_identity(config.warm_start_checkpoint)
    cross_arm = cross_arm_initialization(config.warm_start_checkpoint)
    authorization_identity = experiment_authorization_identity(
        config=config,
        surfaces=surfaces,
        warm_start=warm_start,
        cross_arm=cross_arm,
        source=source,
    )
    controls = (
        validate_launch_controls(
            config,
            authorization_identity=authorization_identity,
            runtime=runtime,
            source=source,
        )
        if config.production
        else None
    )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    model_config = O1RankingModelConfig(arm=config.arm)
    run_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "wave": config.wave,
        "arm": config.arm,
        "target_steps": config.target_steps,
        "protocol": config.protocol.to_dict(),
        "inputs": input_identity(config, surfaces),
        "warm_start": warm_start,
        "cross_arm_initialization": cross_arm,
        "source": source,
        "runtime": runtime,
        "controls": controls,
    }
    parity_path = config.run_dir / "zero-init-prediction-parity.json"
    batch_trace_path = config.run_dir / "batch-trace.jsonl"
    metrics_path = config.run_dir / "metrics.jsonl"
    if config.resume:
        existing = _read_json(config.run_dir / "run.json", "O1 run manifest")
        if existing != run_manifest:
            raise ValueError("O1 ranking resume manifest differs from frozen run")
        model, optimizer, state, _checkpoint = (
            load_latest_checkpoint_with_factory(
                config.run_dir,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                model_factory=lambda values: O1IntentConditionedRanker(
                    O1RankingModelConfig.from_dict(values)
                ).freeze_base_for_adapter_training(),
            )
        )
        parity = _read_json(parity_path, "zero-init prediction parity")
        if parity.get("exact_array_equal") is not True:
            raise ValueError("saved O1 zero-init parity is invalid")
        loss_trace = _load_batch_trace(
            batch_trace_path,
            expected_steps=state.global_step,
        )
    else:
        if (
            (config.run_dir / "latest.json").exists()
            or batch_trace_path.exists()
            or metrics_path.exists()
            or parity_path.exists()
        ):
            raise ValueError("fresh O1 run directory contains stale artifacts")
        model = initialize_model(
            config.arm,
            config.warm_start_checkpoint,
        )
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        base = load_exact_r2_model(config.warm_start_checkpoint)
        parity = verify_zero_init_prediction_parity(
            base,
            model,
            surfaces.validation,
        )
        _write_json_atomic(parity_path, parity)
        _write_json_atomic(config.run_dir / "run.json", run_manifest)
        loss_trace = []
    validate_model_state(
        model,
        model_config=model_config,
        cross_arm=cross_arm,
        warm_start=warm_start,
    )

    loss_and_grad = nn.value_and_grad(model, o1_ranking_loss)
    measured_seconds = float(
        sum(float(event["elapsed_seconds"]) for event in loss_trace)
    )
    measured_candidates = int(
        sum(int(event["candidates"]) for event in loss_trace)
    )
    model.train()
    invocation_started = time.perf_counter()
    while state.global_step < config.target_steps:
        step = state.global_step
        started = time.perf_counter()
        batch = surfaces.train.deterministic_training_batch(
            step=step,
            seed=TRAINING_SEED,
            groups_per_step=GROUPS_PER_STEP,
        )
        cohort_batch_hash = scientific_batch_blake3(batch, surfaces.cohort)
        intent_batch_hash = intent_batch_blake3(batch)
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise ValueError(
                f"O1 ranking training produced nonfinite loss at step {step}"
            )
        candidates = int(
            np.asarray(batch.base.candidate_mask, dtype=np.bool_).sum()
        )
        measured_seconds += elapsed
        measured_candidates += candidates
        state.global_step += 1
        state.batch_in_epoch = state.global_step
        state.elapsed_seconds += elapsed
        event = {
            "schema_version": 1,
            "step": state.global_step,
            "cohort_batch_blake3": cohort_batch_hash,
            "intent_batch_blake3": intent_batch_hash,
            "loss": loss_value,
            "candidates": candidates,
            "elapsed_seconds": elapsed,
        }
        loss_trace.append(event)
        _append_json(batch_trace_path, event)
        if state.global_step % METRIC_STEPS == 0:
            metric_event = {
                "schema_version": 1,
                "step": state.global_step,
                "mean_recent_loss": float(
                    np.mean(
                        [
                            value["loss"]
                            for value in loss_trace[-METRIC_STEPS:]
                        ]
                    )
                ),
                "candidates_per_second": (
                    measured_candidates / max(measured_seconds, 1e-12)
                ),
                "validation_read": False,
                "peak_active_memory_bytes": int(mx.get_peak_memory()),
            }
            _append_json(metrics_path, metric_event)
            print(json.dumps(metric_event, sort_keys=True), flush=True)
        if state.global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(config.run_dir, model, optimizer, state)
            prune_checkpoints(config.run_dir)

    checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
    prune_checkpoints(config.run_dir)
    training_wall_seconds = time.perf_counter() - invocation_started
    training_peak_memory = int(mx.get_peak_memory())
    validate_frozen_base(
        model,
        expected_base_blake3=cross_arm["base_parameter_tensor_blake3"],
    )
    model.eval()
    evaluation = evaluate_o1_ranking(model, surfaces.validation)
    prediction_files = write_prediction_tensors(
        config.run_dir,
        evaluation.scores,
        evaluation.standard_errors,
    )
    prediction_identity = {
        name: {
            key: value
            for key, value in specification.items()
            if key != "path"
        }
        for name, specification in prediction_files.items()
    }
    loss_trace_id = scientific_loss_trace_blake3(loss_trace)
    model_identity = {
        "config": model.config.to_dict(),
        **cross_arm,
        "final_adapter_parameter_tensor_blake3": parameter_tensor_blake3(
            model
        ),
        "final_all_parameter_tensor_blake3": parameter_tensor_blake3(
            model,
            trainable_only=False,
        ),
        "final_base_parameter_tensor_blake3": base_parameter_tensor_blake3(
            model
        ),
    }
    scientific_metrics = {
        key: value
        for key, value in evaluation.metrics.items()
        if key != "performance"
    }
    replication_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": config.arm,
        "protocol": config.protocol.to_dict(),
        "inputs": input_identity(config, surfaces),
        "warm_start": warm_start,
        "zero_init_prediction_parity": parity,
        "model": model_identity,
        "global_step": state.global_step,
        "loss_trace_blake3": loss_trace_id,
        "metrics": scientific_metrics,
        "prediction_files": prediction_identity,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "wave": config.wave,
        "arm": config.arm,
        "host": runtime["host"],
        "protocol": config.protocol.to_dict(),
        "inputs": input_identity(config, surfaces),
        "warm_start": warm_start,
        "zero_init_prediction_parity": parity,
        "model": model_identity,
        "optimization": {
            "global_step": state.global_step,
            "candidates": measured_candidates,
            "training_seconds": measured_seconds,
            "training_wall_seconds": training_wall_seconds,
            "trainer_state_elapsed_seconds": state.elapsed_seconds,
            "candidates_per_second": (
                measured_candidates / max(measured_seconds, 1e-12)
            ),
            "training_peak_active_memory_bytes": training_peak_memory,
            "loss_trace_blake3": loss_trace_id,
            "loss_trace": loss_trace,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
            "model_blake3": _checksum(checkpoint / "model.safetensors"),
        },
        "metrics": evaluation.metrics,
        "prediction_files": prediction_files,
        "performance": {
            **evaluation.metrics["performance"],
            "peak_rss_bytes": peak_rss_bytes(),
            "swap_used_bytes": system_swap_used_bytes(),
        },
        "runtime": {
            **runtime,
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "source": source,
        "controls": controls,
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_read": False,
            "realized_future_refill_read": False,
            "policy_identity_read": False,
        },
        "claims": {
            "offline_validation_complete": config.production,
            "bounded_smoke_complete": not config.production,
            "base_parameters_frozen": True,
            "validation_read_during_training": False,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
        "replication_identity": replication_identity,
        "replication_id": _canonical_blake3(replication_identity),
    }
    report["scientific_identity"] = {
        key: report[key]
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "wave",
            "arm",
            "host",
            "protocol",
            "inputs",
            "warm_start",
            "zero_init_prediction_parity",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
            "prediction_files",
            "performance",
            "runtime",
            "source",
            "controls",
            "information_boundary",
            "claims",
            "replication_id",
        )
    }
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def load_experiment_surfaces(
    config: O1RankingTrainingConfig,
    *,
    verify_checksums: bool,
) -> O1RankingSurfaces:
    """Load and align every immutable open-data surface."""
    r3 = R3ActionEditMlxCache(
        config.r3_cache,
        verify_checksums=verify_checksums,
        verify_semantics=verify_checksums,
        require_complete=True,
    )
    s1 = S1ExactSupplyCache(
        config.s1_cache,
        verify_checksums=verify_checksums,
        verify_semantics=verify_checksums,
        require_complete=True,
    )
    cohort = O1RankingCohortCache(
        config.cohort,
        verify_checksums=verify_checksums,
        require_complete=True,
    )
    afterstates = O1RankingAfterstateCache(
        config.afterstates,
        cohort=cohort,
        verify_checksums=verify_checksums,
        verify_model_inputs=verify_checksums,
        require_complete=True,
    )
    intent = O1RankingIntentCache(
        config.intent_cache,
        cohort=cohort,
        afterstates=afterstates,
        verify_checksums=verify_checksums,
        verify_semantics=verify_checksums,
        require_complete=True,
    )
    proof_id = str(cohort.manifest["open_data_verification_id"])
    train_r3 = r3.bind_dataset(
        config.train_dataset,
        s1_cache=s1,
        verify_dataset_checksums=verify_checksums,
        preverified_open_data_proof_id=proof_id,
    )
    validation_r3 = r3.bind_dataset(
        config.validation_dataset,
        s1_cache=s1,
        verify_dataset_checksums=verify_checksums,
        preverified_open_data_proof_id=proof_id,
    )
    train = O1RankingDataset(
        train_r3,
        cohort=cohort,
        intent=intent,
        arm=config.arm,
    )
    validation = O1RankingDataset(
        validation_r3,
        cohort=cohort,
        intent=intent,
        arm=config.arm,
    )
    return O1RankingSurfaces(
        r3=r3,
        s1=s1,
        cohort=cohort,
        afterstates=afterstates,
        intent=intent,
        train=train,
        validation=validation,
    )


def warm_start_identity(checkpoint: Path) -> dict[str, Any]:
    manifest = _read_json(checkpoint / "checkpoint.json", "exact-R2 checkpoint")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise ValueError("exact-R2 checkpoint file manifest is incomplete")
    for name, expected in files.items():
        path = checkpoint / name
        if (
            not isinstance(expected, dict)
            or not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or _checksum(path) != expected.get("blake3")
        ):
            raise ValueError(f"exact-R2 checkpoint failed integrity: {name}")
    config = R3ActionEditModelConfig.from_dict(manifest["model_config"])
    if config.arm != CONTROL_ARM:
        raise ValueError("O1 warm start must be the exact-R2 control arm")
    identity = {
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_manifest_blake3": _checksum(
            checkpoint / "checkpoint.json"
        ),
        "model_blake3": _checksum(checkpoint / "model.safetensors"),
        "model_config": config.to_dict(),
    }
    identity["warm_start_id"] = _canonical_blake3(identity)
    return identity


def load_exact_r2_model(checkpoint: Path) -> R3ActionEditRanker:
    identity = warm_start_identity(checkpoint)
    config = R3ActionEditModelConfig.from_dict(identity["model_config"])
    model = R3ActionEditRanker(config)
    model.load_weights(str(checkpoint / "model.safetensors"))
    mx.eval(model.parameters())
    model.eval()
    return model


def initialize_model(
    arm: str,
    checkpoint: Path,
) -> O1IntentConditionedRanker:
    mx.random.seed(TRAINING_SEED)
    model = O1IntentConditionedRanker(O1RankingModelConfig(arm=arm))
    model.load_weights(
        str(checkpoint / "model.safetensors"),
        strict=False,
    )
    model.freeze_base_for_adapter_training()
    mx.eval(model.parameters())
    return model


def cross_arm_initialization(checkpoint: Path) -> dict[str, Any]:
    """Prove all four routed arms start from exactly the same tensors."""
    all_counts: dict[str, int] = {}
    all_layouts: dict[str, str] = {}
    all_tensors: dict[str, str] = {}
    adapter_counts: dict[str, int] = {}
    adapter_layouts: dict[str, str] = {}
    adapter_tensors: dict[str, str] = {}
    base_tensors: dict[str, str] = {}
    for arm in ARMS:
        model = initialize_model(arm, checkpoint)
        all_counts[arm] = parameter_count(model, trainable_only=False)
        all_layouts[arm] = parameter_layout_blake3(
            model,
            trainable_only=False,
        )
        all_tensors[arm] = parameter_tensor_blake3(
            model,
            trainable_only=False,
        )
        adapter_counts[arm] = parameter_count(model)
        adapter_layouts[arm] = parameter_layout_blake3(model)
        adapter_tensors[arm] = parameter_tensor_blake3(model)
        base_tensors[arm] = base_parameter_tensor_blake3(model)
    if any(
        len(set(values.values())) != 1
        for values in (
            all_counts,
            all_layouts,
            all_tensors,
            adapter_counts,
            adapter_layouts,
            adapter_tensors,
            base_tensors,
        )
    ):
        raise ValueError("O1 arm graph, adapter initialization, or base differs")
    identity = {
        "all_parameter_count": next(iter(all_counts.values())),
        "all_parameter_layout_blake3": next(iter(all_layouts.values())),
        "initial_all_parameter_tensor_blake3": next(
            iter(all_tensors.values())
        ),
        "adapter_parameter_count": next(iter(adapter_counts.values())),
        "adapter_parameter_layout_blake3": next(
            iter(adapter_layouts.values())
        ),
        "initial_adapter_parameter_tensor_blake3": next(
            iter(adapter_tensors.values())
        ),
        "base_parameter_tensor_blake3": next(iter(base_tensors.values())),
        "cross_arm_all_parameter_counts": all_counts,
        "cross_arm_all_parameter_layout_blake3": all_layouts,
        "cross_arm_initial_all_parameter_tensor_blake3": all_tensors,
        "cross_arm_adapter_parameter_counts": adapter_counts,
        "cross_arm_adapter_parameter_layout_blake3": adapter_layouts,
        "cross_arm_initial_adapter_parameter_tensor_blake3": adapter_tensors,
        "cross_arm_base_parameter_tensor_blake3": base_tensors,
    }
    identity["cross_arm_initialization_id"] = _canonical_blake3(identity)
    return identity


def verify_zero_init_prediction_parity(
    base: R3ActionEditRanker,
    treatment: O1IntentConditionedRanker,
    validation: O1RankingDataset,
) -> dict[str, Any]:
    batch = validation.batch([0])
    expected = base(batch)
    observed = treatment(batch)
    mx.eval(
        expected.scores,
        expected.standard_errors,
        observed.scores,
        observed.standard_errors,
    )
    expected_scores = np.asarray(expected.scores)
    observed_scores = np.asarray(observed.scores)
    expected_errors = np.asarray(expected.standard_errors)
    observed_errors = np.asarray(observed.standard_errors)
    exact = np.array_equal(
        expected_scores,
        observed_scores,
    ) and np.array_equal(expected_errors, observed_errors)
    if not exact:
        raise ValueError("zero-initialized O1 adapter changed exact-R2 outputs")
    digest = blake3.blake3()
    digest.update(expected_scores.astype("<f4", copy=False).tobytes())
    digest.update(expected_errors.astype("<f4", copy=False).tobytes())
    return {
        "exact_array_equal": True,
        "groups": 1,
        "candidates": expected_scores.shape[1],
        "prediction_blake3": digest.hexdigest(),
    }


def input_identity(
    config: O1RankingTrainingConfig,
    surfaces: O1RankingSurfaces,
) -> dict[str, Any]:
    return {
        "train_dataset_id": surfaces.train.r3.base.manifest["dataset_id"],
        "train_dataset_manifest_blake3": _checksum(
            config.train_dataset / "dataset.json"
        ),
        "validation_dataset_id": (
            surfaces.validation.r3.base.manifest["dataset_id"]
        ),
        "validation_dataset_manifest_blake3": _checksum(
            config.validation_dataset / "dataset.json"
        ),
        "r3_cache_id": surfaces.r3.cache_id,
        "r3_cache_manifest_blake3": _checksum(config.r3_cache / "cache.json"),
        "s1_cache_id": surfaces.s1.cache_id,
        "s1_cache_manifest_blake3": _checksum(config.s1_cache / "cache.json"),
        "cohort_id": surfaces.cohort.cache_id,
        "cohort_manifest_blake3": _checksum(config.cohort / "cache.json"),
        "afterstate_cache_id": surfaces.afterstates.cache_id,
        "afterstate_manifest_blake3": _checksum(
            config.afterstates / "cache.json"
        ),
        "intent_cache_id": surfaces.intent.cache_id,
        "intent_manifest_blake3": _checksum(
            config.intent_cache / "cache.json"
        ),
        "open_data_verification_id": (
            surfaces.cohort.manifest["open_data_verification_id"]
        ),
    }


def experiment_authorization_identity(
    *,
    config: O1RankingTrainingConfig,
    surfaces: O1RankingSurfaces,
    warm_start: dict[str, Any],
    cross_arm: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "authorized_arms": list(ARMS),
        "wave_hosts": WAVE_HOSTS,
        "protocol": O1RankingTrainingProtocol().to_dict(),
        "inputs": input_identity(config, surfaces),
        "warm_start": warm_start,
        "cross_arm_initialization": cross_arm,
        "source_blake3": source["v2_source_blake3"],
        "sealed_test_opened": False,
        "gameplay_run": False,
    }


def validate_launch_controls(
    config: O1RankingTrainingConfig,
    *,
    authorization_identity: dict[str, Any],
    runtime: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if config.authorization is None or config.preflight is None:
        raise ValueError("O1 ranking production launch controls are absent")
    authorization = _read_json(
        config.authorization,
        "O1 ranking authorization",
    )
    preflight = _read_json(config.preflight, "O1 ranking preflight")
    expected_authorization_id = _canonical_blake3(authorization_identity)
    preflight_identity = preflight.get("identity")
    checks = preflight.get("checks")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("approved") is not True
        or authorization.get("identity") != authorization_identity
        or authorization.get("authorization_id")
        != expected_authorization_id
    ):
        raise ValueError("O1 ranking authorization is stale or malformed")
    expected_host = WAVE_HOSTS[config.wave][config.arm]
    if (
        preflight.get("schema_version") != 1
        or preflight.get("experiment_id") != EXPERIMENT_ID
        or preflight.get("arm") != config.arm
        or preflight.get("wave") != config.wave
        or not isinstance(preflight_identity, dict)
        or _canonical_blake3(preflight_identity)
        != preflight.get("preflight_id")
        or preflight_identity.get("authorization_id")
        != expected_authorization_id
        or preflight_identity.get("arm") != config.arm
        or preflight_identity.get("wave") != config.wave
        or preflight_identity.get("host") != expected_host
        or preflight_identity.get("runtime") != runtime
        or preflight_identity.get("source_blake3")
        != source["v2_source_blake3"]
        or preflight_identity.get("input_bundle_id")
        != _canonical_blake3(authorization_identity["inputs"])
        or preflight_identity.get("warm_start_id")
        != authorization_identity["warm_start"]["warm_start_id"]
        or preflight_identity.get("cross_arm_initialization_id")
        != authorization_identity["cross_arm_initialization"][
            "cross_arm_initialization_id"
        ]
        or not isinstance(checks, dict)
        or any(
            value is not True
            for key, value in checks.items()
            if key != "production_training_started"
        )
        or checks.get("production_training_started") is not False
    ):
        raise ValueError("O1 ranking preflight is stale or incomplete")
    return {
        "authorization_id": expected_authorization_id,
        "preflight_id": preflight["preflight_id"],
        "input_bundle_id": preflight_identity["input_bundle_id"],
        "full_preflight_verification_reused": True,
    }


def validate_model_state(
    model: O1IntentConditionedRanker,
    *,
    model_config: O1RankingModelConfig,
    cross_arm: dict[str, Any],
    warm_start: dict[str, Any],
) -> None:
    if (
        model.config != model_config
        or parameter_count(model, trainable_only=False)
        != cross_arm["all_parameter_count"]
        or parameter_layout_blake3(model, trainable_only=False)
        != cross_arm["all_parameter_layout_blake3"]
        or parameter_count(model) != cross_arm["adapter_parameter_count"]
        or parameter_layout_blake3(model)
        != cross_arm["adapter_parameter_layout_blake3"]
        or base_parameter_tensor_blake3(model)
        != cross_arm["base_parameter_tensor_blake3"]
        or warm_start["model_blake3"] == ""
    ):
        raise ValueError("O1 model graph or frozen exact-R2 warm start drifted")


def validate_frozen_base(
    model: O1IntentConditionedRanker,
    *,
    expected_base_blake3: str,
) -> None:
    if base_parameter_tensor_blake3(model) != expected_base_blake3:
        raise ValueError("O1 adapter training mutated a frozen exact-R2 parameter")


def scientific_batch_blake3(
    batch: O1RankingBatch,
    cohort: O1RankingCohortCache,
) -> str:
    digest = blake3.blake3()
    digest.update(b"cascadia-v2-o1-ranking-scientific-batch-v1")
    rows = np.asarray(batch.cohort_rows, dtype=np.int64)
    split = cohort.split("train")
    for row in rows:
        digest.update(
            np.asarray(
                split.tensors["cohort_hashes"][int(row)],
                dtype=np.uint8,
            ).tobytes()
        )
    return digest.hexdigest()


def intent_batch_blake3(batch: O1RankingBatch) -> str:
    values = np.asarray(batch.intent_features, dtype="<f4")
    digest = blake3.blake3()
    digest.update(b"cascadia-v2-o1-ranking-routed-intent-batch-v1")
    digest.update(batch.arm.encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def scientific_loss_trace_blake3(
    trace: list[dict[str, Any]],
) -> str:
    digest = blake3.blake3()
    digest.update(b"cascadia-v2-o1-ranking-loss-trace-v1")
    for event in trace:
        digest.update(int(event["step"]).to_bytes(8, "little"))
        digest.update(str(event["cohort_batch_blake3"]).encode())
        digest.update(str(event["intent_batch_blake3"]).encode())
        digest.update(
            np.asarray([event["loss"]], dtype="<f8").tobytes()
        )
    return digest.hexdigest()


def base_parameter_tensor_blake3(
    model: O1IntentConditionedRanker,
) -> str:
    return selected_parameter_tensor_blake3(
        [
            (name, value)
            for name, value in tree_flatten(model.parameters())
            if name.split(".", 1)[0] not in ADAPTER_PARAMETER_ROOTS
        ]
    )


def selected_parameter_tensor_blake3(
    values: list[tuple[str, mx.array]],
) -> str:
    digest = blake3.blake3()
    for name, value in values:
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode()
        )
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()


def write_prediction_tensors(
    run_dir: Path,
    scores: np.ndarray,
    standard_errors: np.ndarray,
) -> dict[str, Any]:
    files = {}
    for name, values in (
        ("validation-scores.f32", scores),
        ("validation-standard-errors.f32", standard_errors),
    ):
        path = run_dir / name
        array = np.ascontiguousarray(values, dtype="<f4")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(array.tobytes(order="C"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        files[name] = {
            "path": str(path.resolve()),
            "dtype": "<f4",
            "shape": list(array.shape),
            "bytes": path.stat().st_size,
            "blake3": _checksum(path),
        }
    return files


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "default_device": str(mx.default_device()),
        "host": normalize_host(socket.gethostname().split(".")[0]),
    }


def require_production_runtime(runtime: dict[str, Any]) -> None:
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    if (
        runtime.get("machine") != "arm64"
        or "gpu" not in str(mx.default_device()).lower()
    ):
        raise ValueError("O1 production training requires Apple Silicon MLX GPU")


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def system_swap_used_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    marker = "used = "
    if marker not in output:
        return None
    raw = output.split(marker, 1)[1].split("M", 1)[0].strip()
    try:
        return int(float(raw) * 1024 * 1024)
    except ValueError:
        return None


def _load_batch_trace(
    path: Path,
    *,
    expected_steps: int,
) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load O1 batch trace: {error}") from error
    if (
        len(values) != expected_steps
        or [value.get("step") for value in values]
        != list(range(1, expected_steps + 1))
    ):
        raise ValueError("O1 batch trace does not match trainer cursor")
    return values


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one ADR 0188 O1 ranking arm"
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--afterstates", type=Path, required=True)
    parser.add_argument("--intent-cache", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--wave", choices=tuple(WAVE_HOSTS), default="primary")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--preflight", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_o1_ranking_training(
        O1RankingTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            r3_cache=args.r3_cache,
            s1_cache=args.s1_cache,
            cohort=args.cohort,
            afterstates=args.afterstates,
            intent_cache=args.intent_cache,
            warm_start_checkpoint=args.warm_start_checkpoint,
            run_dir=args.run_dir,
            output=args.output,
            arm=args.arm,
            wave=args.wave,
            resume=args.resume,
            smoke_steps=args.smoke_steps,
            authorization=args.authorization,
            preflight=args.preflight,
        )
    )
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "replication_id": report["replication_id"],
                "arm": report["arm"],
                "wave": report["wave"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
