"""Authorized adapter training for the exact-R2 opportunity query factorial."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
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
from cascadia_mlx.opportunity_cross_attention_mlx_benchmark import (
    run_isolated_opportunity_serving_benchmark,
)
from cascadia_mlx.opportunity_cross_attention_mlx_metrics import (
    evaluate_opportunity_cross_attention,
)
from cascadia_mlx.opportunity_cross_attention_mlx_model import (
    ARMS,
    OpportunityCrossAttentionModelConfig,
    OpportunityCrossAttentionRanker,
    opportunity_cross_attention_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)
from cascadia_mlx.opportunity_cross_attention_mlx_pairwise import (
    collect_decision_panel,
    panel_identity,
)
from cascadia_mlx.opportunity_cross_attention_mlx_protocol import (
    ADR_ID,
    ARM_HOSTS,
    CHECKPOINT_STEPS,
    EXPERIMENT_ID,
    LEARNING_RATE,
    MAX_SMOKE_STEPS,
    METRIC_STEPS,
    MLX_CACHE_LIMIT_BYTES,
    PROTOCOL_ID,
    RELATIONAL_DATA_ARM,
    TRAINING_SEED,
    TRAINING_STEPS,
    VALIDATION_PROBE_GROUPS,
    WEIGHT_DECAY,
    OpportunityCrossAttentionTrainingProtocol,
    normalize_host,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3ActionEditMlxCache
from cascadia_mlx.relational_substrate_mlx_cache import (
    RelationalSubstrateMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateModelConfig,
    RelationalSubstrateRanker,
)
from cascadia_mlx.relational_substrate_mlx_train import (
    scientific_batch_blake3,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

EXPECTED_WARM_START_EXPERIMENT = "relational-substrate-mlx-tournament-v1"
EXPECTED_WARM_START_PROTOCOL = "r5-s3-s5-matched-mlx-v1"
EXPECTED_WARM_START_ADR = "0161"
EXPECTED_WARM_START_STEPS = 3000
ADAPTER_PARAMETER_ROOTS = frozenset(
    {
        "memory_seat_embedding",
        "supply_token_projection",
        "supply_position_embedding",
        "candidate_query_projection",
        "parent_query_projection",
        "supply_cross_attention",
        "frontier_cross_attention",
        "context_projection",
        "context_delta",
    }
)


@dataclass(frozen=True)
class OpportunityCrossAttentionTrainingConfig:
    """One production or bounded-smoke arm invocation."""

    train_dataset: Path
    validation_dataset: Path
    r3_cache: Path
    relational_cache: Path
    s1_cache: Path
    r6_binary: Path
    warm_start_run_dir: Path
    warm_start_report: Path
    run_dir: Path
    output: Path
    arm: str
    resume: bool = False
    smoke_steps: int | None = None
    authorization: Path | None = None
    preflight: Path | None = None
    protocol: OpportunityCrossAttentionTrainingProtocol = field(
        default_factory=OpportunityCrossAttentionTrainingProtocol
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
            raise ValueError("opportunity training arm is unknown")
        if not self.r6_binary.is_file():
            raise ValueError("opportunity R6 replay binary is absent")
        if self.production:
            if self.authorization is None or self.preflight is None:
                raise ValueError(
                    "production opportunity training requires launch controls"
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
                "bounded opportunity smoke must be fresh and at most 10 steps"
            )


def run_opportunity_cross_attention_training(
    config: OpportunityCrossAttentionTrainingConfig,
) -> dict[str, Any]:
    """Train one arm while proving the inherited exact-R2 model stays frozen."""
    config.validate()
    require_complete = config.production
    source = source_provenance(Path(__file__).resolve().parents[2])
    mx.set_default_device(mx.gpu)
    runtime = _runtime_identity()
    if config.production:
        _require_production_runtime(runtime)
        actual_host = normalize_host(socket.gethostname().split(".")[0])
        if actual_host != ARM_HOSTS[config.arm]:
            raise ValueError(
                f"opportunity arm {config.arm} must run on "
                f"{ARM_HOSTS[config.arm]}, not {actual_host}"
            )

    r3 = R3ActionEditMlxCache(
        config.r3_cache,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
        require_complete=require_complete,
    )
    relational_cache = RelationalSubstrateMlxCache(
        config.relational_cache,
        r3_cache=r3,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
        require_complete=require_complete,
    )
    s1_cache = S1ExactSupplyCache(
        config.s1_cache,
        verify_checksums=not config.production,
        verify_semantics=not config.production,
        require_complete=require_complete,
    )
    open_data = open_data_verification_identity(
        cache=relational_cache,
        s1_cache=s1_cache,
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
    )
    (
        warm_start_model,
        warm_start,
        warm_start_checkpoint,
    ) = load_verified_warm_start(
        config.warm_start_run_dir,
        config.warm_start_report,
    )
    cross_arm = cross_arm_initialization(
        warm_start_model,
        warm_start_checkpoint=warm_start_checkpoint,
    )
    controls = (
        validate_launch_controls(
            config.authorization,
            config.preflight,
            arm=config.arm,
            r3_cache_id=r3.cache_id,
            relational_cache_id=relational_cache.cache_id,
            s1_cache_id=s1_cache.cache_id,
            r6_binary_blake3=_checksum(config.r6_binary),
            source=source,
            runtime=runtime,
            open_data_verification=open_data,
            warm_start=warm_start,
            cross_arm_initialization_proof=cross_arm,
        )
        if config.production
        else None
    )
    proof_id = (
        str(controls["open_data_verification_id"])
        if controls is not None
        else None
    )
    train = relational_cache.bind_dataset(
        config.train_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=not config.production,
        preverified_open_data_proof_id=proof_id,
    )
    validation = relational_cache.bind_dataset(
        config.validation_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=not config.production,
        preverified_open_data_proof_id=proof_id,
    )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.clear_cache()
    mx.reset_peak_memory()
    model_config = OpportunityCrossAttentionModelConfig(arm=config.arm)
    run_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "arm": config.arm,
        "data_arm": RELATIONAL_DATA_ARM,
        "target_steps": config.target_steps,
        "protocol": config.protocol.to_dict(),
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational_cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "r6_binary": {
            "path": str(config.r6_binary.resolve()),
            "blake3": _checksum(config.r6_binary),
        },
        "train_dataset_id": train.base.manifest["dataset_id"],
        "validation_dataset_id": validation.base.manifest["dataset_id"],
        "warm_start": warm_start,
        "source": source,
        "runtime": runtime,
        "cross_arm_initialization": cross_arm,
        "controls": controls,
    }
    if config.resume:
        existing = _read_json(
            config.run_dir / "run.json",
            "opportunity run manifest",
        )
        if existing != run_manifest:
            raise ValueError(
                "opportunity resume manifest differs from the frozen run"
            )
        model, optimizer, state, _ = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            model_factory=lambda values: OpportunityCrossAttentionRanker(
                OpportunityCrossAttentionModelConfig.from_dict(values)
            ),
        )
        model.freeze_base_for_adapter_training()
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError(
                "opportunity run already has checkpoints; pass --resume"
            )
        mx.random.seed(TRAINING_SEED)
        model = OpportunityCrossAttentionRanker(model_config)
        model.load_weights(
            str(warm_start_checkpoint / "model.safetensors"),
            strict=False,
        )
        model.freeze_base_for_adapter_training()
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
    _validate_model_state(model, model_config, cross_arm, warm_start)

    loss_and_grad = nn.value_and_grad(
        model,
        opportunity_cross_attention_loss,
    )
    metrics_path = config.run_dir / "metrics.jsonl"
    batch_trace_path = config.run_dir / "batch-trace.jsonl"
    parity_path = config.run_dir / "zero-init-prediction-parity.json"
    if config.resume:
        zero_init_parity = _read_json(
            parity_path,
            "zero-init prediction parity",
        )
        if zero_init_parity.get("exact_array_equal") is not True:
            raise ValueError("saved zero-init prediction parity is invalid")
        loss_trace = _load_batch_trace(
            batch_trace_path,
            expected_steps=state.global_step,
        )
    else:
        if batch_trace_path.exists() or metrics_path.exists() or parity_path.exists():
            raise ValueError(
                "fresh opportunity run contains stale training artifacts"
            )
        zero_init_parity = verify_zero_init_prediction_parity(
            warm_start_model,
            model,
            validation,
        )
        _write_json_atomic(parity_path, zero_init_parity)
        _write_json_atomic(config.run_dir / "run.json", run_manifest)
        loss_trace = []
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
        batch = train.deterministic_training_batch(
            step=step,
            seed=TRAINING_SEED,
            arm=RELATIONAL_DATA_ARM,
        )
        batch_identity = scientific_batch_blake3(batch)
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise ValueError(
                f"opportunity training produced nonfinite loss at step {step}"
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
            "batch_blake3": batch_identity,
            "loss": loss_value,
            "candidates": candidates,
            "elapsed_seconds": elapsed,
        }
        loss_trace.append(event)
        _append_json(batch_trace_path, event)
        if state.global_step % METRIC_STEPS == 0:
            probe = evaluate_opportunity_cross_attention(
                model,
                validation,
                rows=_validation_probe_rows(validation.group_count),
                prediction_panel_size=16,
            )
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
                "validation_probe": probe,
                "peak_active_memory_bytes": int(mx.get_peak_memory()),
            }
            _append_json(metrics_path, metric_event)
            print(json.dumps(metric_event, sort_keys=True), flush=True)
            model.train()
        if state.global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(config.run_dir, model, optimizer, state)
            prune_checkpoints(config.run_dir)

    checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
    prune_checkpoints(config.run_dir)
    training_peak_memory = int(mx.get_peak_memory())
    training_wall_seconds = time.perf_counter() - invocation_started
    _validate_frozen_base(model, warm_start)
    model.eval()
    if config.production:
        validation_metrics = evaluate_opportunity_cross_attention(
            model,
            validation,
        )
        paired_panel = collect_decision_panel(model, validation)
        paired_panel_id = panel_identity(paired_panel)
        benchmark_rows = np.arange(
            validation.group_count,
            dtype=np.int64,
        )
        warmup_iterations = 5
        steady_iterations = 30
        verification_source = "cluster-preflight"
    else:
        benchmark_rows = np.arange(
            min(validation.group_count, 5),
            dtype=np.int64,
        )
        validation_metrics = evaluate_opportunity_cross_attention(
            model,
            validation,
            rows=benchmark_rows,
            prediction_panel_size=16,
        )
        paired_panel = None
        paired_panel_id = None
        warmup_iterations = 1
        steady_iterations = 3
        verification_source = "in-process-full"
    performance = run_isolated_opportunity_serving_benchmark(
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
        r3_cache=config.r3_cache,
        relational_cache=config.relational_cache,
        s1_cache=config.s1_cache,
        r6_binary=config.r6_binary,
        run_dir=config.run_dir,
        checkpoint=checkpoint,
        arm=config.arm,
        global_step=state.global_step,
        open_data_verification=open_data,
        verification_source=verification_source,
        warmup_iterations=warmup_iterations,
        steady_iterations=steady_iterations,
        decision_rows=benchmark_rows,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "arm": config.arm,
        "host": normalize_host(socket.gethostname().split(".")[0]),
        "data_arm": RELATIONAL_DATA_ARM,
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational_cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "r6_binary": {
            "path": str(config.r6_binary.resolve()),
            "blake3": _checksum(config.r6_binary),
        },
        "protocol": config.protocol.to_dict(),
        "warm_start": warm_start,
        "zero_init_prediction_parity": zero_init_parity,
        "model": {
            "config": model.config.to_dict(),
            **cross_arm,
            "final_adapter_parameter_tensor_blake3": (
                parameter_tensor_blake3(model)
            ),
            "final_all_parameter_tensor_blake3": _all_parameter_tensor_blake3(
                model
            ),
            "final_base_parameter_tensor_blake3": (
                _base_parameter_tensor_blake3(model)
            ),
        },
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
            "loss_trace": loss_trace,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "manifest_blake3": _checksum(
                checkpoint / "checkpoint.json"
            ),
            "model_blake3": _checksum(checkpoint / "model.safetensors"),
        },
        "metrics": validation_metrics,
        "paired_panel": paired_panel,
        "paired_panel_id": paired_panel_id,
        "performance": performance,
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
            "future_refill_read": False,
        },
        "claims": {
            "offline_comparison_complete": config.production,
            "bounded_smoke_complete": not config.production,
            "base_parameters_frozen": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = {
        key: report[key]
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "arm",
            "host",
            "data_arm",
            "r3_cache_id",
            "relational_cache_id",
            "s1_cache_id",
            "r6_binary",
            "protocol",
            "warm_start",
            "zero_init_prediction_parity",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
            "paired_panel",
            "paired_panel_id",
            "performance",
            "runtime",
            "source",
            "controls",
            "information_boundary",
            "claims",
        )
    }
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def load_verified_warm_start(
    run_dir: Path,
    report_path: Path,
) -> tuple[RelationalSubstrateRanker, dict[str, Any], Path]:
    """Load the final C0 checkpoint and bind it to its immutable report."""
    report = _read_json(report_path, "C0 warm-start report")
    scientific_identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPECTED_WARM_START_EXPERIMENT
        or report.get("protocol_id") != EXPECTED_WARM_START_PROTOCOL
        or report.get("adr") != EXPECTED_WARM_START_ADR
        or report.get("mode") != "production"
        or report.get("arm") != RELATIONAL_DATA_ARM
        or not isinstance(scientific_identity, dict)
        or _canonical_blake3(scientific_identity) != report.get("report_id")
        or report.get("claims", {}).get("offline_comparison_complete")
        is not True
        or report.get("information_boundary", {}).get("sealed_test_opened")
        is not False
    ):
        raise ValueError("C0 warm-start report is incomplete or malformed")
    model, _optimizer, state, checkpoint = load_latest_checkpoint_with_factory(
        run_dir,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        model_factory=lambda values: RelationalSubstrateRanker(
            RelationalSubstrateModelConfig.from_dict(values)
        ),
    )
    expected = report.get("checkpoint")
    if (
        not isinstance(expected, dict)
        or state.global_step != EXPECTED_WARM_START_STEPS
        or model.config.arm != RELATIONAL_DATA_ARM
        or _checksum(checkpoint / "checkpoint.json")
        != expected.get("manifest_blake3")
        or _checksum(checkpoint / "model.safetensors")
        != expected.get("model_blake3")
        or report.get("optimization", {}).get("global_step")
        != state.global_step
    ):
        raise ValueError("C0 checkpoint does not match its final report")
    model.eval()
    identity = {
        "experiment_id": report["experiment_id"],
        "protocol_id": report["protocol_id"],
        "adr": report["adr"],
        "report_id": report["report_id"],
        "arm": report["arm"],
        "global_step": state.global_step,
        "checkpoint": {
            "checkpoint_id": checkpoint.name,
            "manifest_blake3": expected["manifest_blake3"],
            "model_blake3": expected["model_blake3"],
        },
        "model_config": model.config.to_dict(),
        "parameter_count": _all_parameter_count(model),
        "parameter_layout_blake3": _all_parameter_layout_blake3(model),
        "base_parameter_tensor_blake3": _all_parameter_tensor_blake3(model),
    }
    identity["warm_start_id"] = _canonical_blake3(identity)
    return model, identity, checkpoint


def cross_arm_initialization(
    warm_start_model: RelationalSubstrateRanker,
    *,
    warm_start_checkpoint: Path,
) -> dict[str, Any]:
    """Prove all four arms begin from the same graph and exact tensors."""
    total_counts: dict[str, int] = {}
    total_layouts: dict[str, str] = {}
    total_tensors: dict[str, str] = {}
    adapter_counts: dict[str, int] = {}
    adapter_layouts: dict[str, str] = {}
    adapter_tensors: dict[str, str] = {}
    base_tensors: dict[str, str] = {}
    checkpoint = warm_start_checkpoint / "model.safetensors"
    for arm in ARMS:
        mx.random.seed(TRAINING_SEED)
        candidate = OpportunityCrossAttentionRanker(
            OpportunityCrossAttentionModelConfig(arm=arm)
        )
        candidate.load_weights(str(checkpoint), strict=False)
        total_counts[arm] = _all_parameter_count(candidate)
        total_layouts[arm] = _all_parameter_layout_blake3(candidate)
        total_tensors[arm] = _all_parameter_tensor_blake3(candidate)
        base_tensors[arm] = _base_parameter_tensor_blake3(candidate)
        candidate.freeze_base_for_adapter_training()
        adapter_counts[arm] = parameter_count(candidate)
        adapter_layouts[arm] = parameter_layout_blake3(candidate)
        adapter_tensors[arm] = parameter_tensor_blake3(candidate)
    expected_base = _all_parameter_tensor_blake3(warm_start_model)
    if (
        len(set(total_counts.values())) != 1
        or len(set(total_layouts.values())) != 1
        or len(set(total_tensors.values())) != 1
        or len(set(adapter_counts.values())) != 1
        or len(set(adapter_layouts.values())) != 1
        or len(set(adapter_tensors.values())) != 1
        or set(base_tensors.values()) != {expected_base}
    ):
        raise ValueError(
            "opportunity arm graph, initialization, or warm start differs"
        )
    return {
        "total_parameter_count": next(iter(total_counts.values())),
        "total_parameter_layout_blake3": next(iter(total_layouts.values())),
        "initial_all_parameter_tensor_blake3": next(
            iter(total_tensors.values())
        ),
        "adapter_parameter_count": next(iter(adapter_counts.values())),
        "adapter_parameter_layout_blake3": next(
            iter(adapter_layouts.values())
        ),
        "initial_adapter_parameter_tensor_blake3": next(
            iter(adapter_tensors.values())
        ),
        "base_parameter_tensor_blake3": expected_base,
        "cross_arm_total_parameter_counts": total_counts,
        "cross_arm_total_parameter_layout_blake3": total_layouts,
        "cross_arm_initial_all_parameter_tensor_blake3": total_tensors,
        "cross_arm_adapter_parameter_counts": adapter_counts,
        "cross_arm_adapter_parameter_layout_blake3": adapter_layouts,
        "cross_arm_initial_adapter_parameter_tensor_blake3": adapter_tensors,
        "cross_arm_base_parameter_tensor_blake3": base_tensors,
    }


def verify_zero_init_prediction_parity(
    base: RelationalSubstrateRanker,
    treatment: OpportunityCrossAttentionRanker,
    validation: object,
) -> dict[str, Any]:
    """Prove the zero-initialized adapter starts at exact C0 predictions."""
    batch = validation.batch(
        [0],
        arm=RELATIONAL_DATA_ARM,
        transform_ids=[0],
    )
    count = min(
        64,
        int(np.asarray(batch.base.candidate_mask, dtype=np.bool_)[0].sum()),
    )
    base_parent = base.encode_parent(batch)
    treatment_parent = treatment.encode_parent(batch)
    expected = base.predict(
        batch,
        candidate_slice=slice(0, count),
        parent_state=base_parent,
    )
    observed = treatment.predict(
        batch,
        candidate_slice=slice(0, count),
        parent_state=treatment_parent,
    )
    mx.eval(expected.scores, observed.scores)
    expected_scores = np.asarray(expected.scores)
    observed_scores = np.asarray(observed.scores)
    if not np.array_equal(expected_scores, observed_scores):
        raise ValueError(
            "zero-initialized opportunity adapter changed C0 predictions"
        )
    payload = expected_scores.astype("<f4", copy=False).tobytes()
    return {
        "exact_array_equal": True,
        "candidates": count,
        "scores_blake3": blake3.blake3(payload).hexdigest(),
    }


def validate_launch_controls(
    authorization_path: Path | None,
    preflight_path: Path | None,
    *,
    arm: str,
    r3_cache_id: str,
    relational_cache_id: str,
    s1_cache_id: str,
    r6_binary_blake3: str,
    source: dict[str, Any],
    runtime: dict[str, Any],
    open_data_verification: dict[str, Any],
    warm_start: dict[str, Any],
    cross_arm_initialization_proof: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless authorization and host preflight match exactly."""
    if authorization_path is None or preflight_path is None:
        raise ValueError("opportunity production launch controls are absent")
    authorization = _read_json(
        authorization_path,
        "opportunity authorization",
    )
    preflight = _read_json(preflight_path, "opportunity preflight")
    authorization_identity = authorization.get("identity")
    preflight_identity = preflight.get("identity")
    verification_id = open_data_verification_id(
        open_data_verification
    )
    preflight_checks = preflight.get("checks")
    expected_source = source.get("v2_source_blake3")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization_identity, dict)
        or _canonical_blake3(authorization_identity)
        != authorization.get("authorization_id")
        or authorization_identity.get("protocol_id") != PROTOCOL_ID
        or authorization_identity.get("r3_cache_id") != r3_cache_id
        or authorization_identity.get("relational_cache_id")
        != relational_cache_id
        or authorization_identity.get("s1_cache_id") != s1_cache_id
        or authorization_identity.get("r6_binary_blake3")
        != r6_binary_blake3
        or authorization_identity.get("authorized_arms") != list(ARMS)
        or authorization_identity.get("arm_hosts") != ARM_HOSTS
        or authorization_identity.get("source_blake3") != expected_source
        or authorization_identity.get("protocol")
        != OpportunityCrossAttentionTrainingProtocol().to_dict()
        or authorization_identity.get("open_data_verification")
        != open_data_verification
        or authorization_identity.get("open_data_verification_id")
        != verification_id
        or authorization_identity.get("warm_start") != warm_start
        or authorization_identity.get("cross_arm_initialization")
        != cross_arm_initialization_proof
    ):
        raise ValueError(
            "opportunity production authorization is stale or malformed"
        )
    if (
        preflight.get("schema_version") != 1
        or preflight.get("experiment_id") != EXPERIMENT_ID
        or preflight.get("arm") != arm
        or not isinstance(preflight_identity, dict)
        or _canonical_blake3(preflight_identity)
        != preflight.get("preflight_id")
        or preflight_identity.get("authorization_id")
        != authorization.get("authorization_id")
        or preflight_identity.get("r3_cache_id") != r3_cache_id
        or preflight_identity.get("relational_cache_id")
        != relational_cache_id
        or preflight_identity.get("s1_cache_id") != s1_cache_id
        or preflight_identity.get("r6_binary_blake3")
        != r6_binary_blake3
        or preflight_identity.get("arm") != arm
        or preflight_identity.get("host") != ARM_HOSTS[arm]
        or preflight_identity.get("runtime") != runtime
        or preflight_identity.get("source_blake3") != expected_source
        or preflight_identity.get("open_data_verification_id")
        != verification_id
        or preflight_identity.get("warm_start_id")
        != warm_start["warm_start_id"]
        or preflight_identity.get("mlx_gpu_verified") is not True
        or preflight_identity.get("open_data_only_verified") is not True
        or preflight_identity.get("warm_start_verified") is not True
        or preflight_identity.get("initialization_parity_verified")
        is not True
        or preflight_identity.get("zero_init_prediction_parity_verified")
        is not True
        or preflight_identity.get("smoke_replay_verified") is not True
        or not isinstance(preflight_checks, dict)
        or any(
            value is not True
            for key, value in preflight_checks.items()
            if key != "production_training_started"
        )
        or preflight_checks.get("production_training_started") is not False
    ):
        raise ValueError(
            "opportunity arm preflight is stale or incomplete"
        )
    return {
        "authorization_id": authorization["authorization_id"],
        "preflight_id": preflight["preflight_id"],
        "open_data_verification_id": verification_id,
        "full_preflight_verification_reused": True,
    }


def _validate_model_state(
    model: OpportunityCrossAttentionRanker,
    model_config: OpportunityCrossAttentionModelConfig,
    cross_arm: dict[str, Any],
    warm_start: dict[str, Any],
) -> None:
    if (
        model.config != model_config
        or _all_parameter_count(model)
        != cross_arm["total_parameter_count"]
        or _all_parameter_layout_blake3(model)
        != cross_arm["total_parameter_layout_blake3"]
        or parameter_count(model)
        != cross_arm["adapter_parameter_count"]
        or parameter_layout_blake3(model)
        != cross_arm["adapter_parameter_layout_blake3"]
        or _base_parameter_tensor_blake3(model)
        != warm_start["base_parameter_tensor_blake3"]
    ):
        raise ValueError(
            "opportunity model graph or frozen warm start drifted"
        )


def _validate_frozen_base(
    model: OpportunityCrossAttentionRanker,
    warm_start: dict[str, Any],
) -> None:
    if (
        _base_parameter_tensor_blake3(model)
        != warm_start["base_parameter_tensor_blake3"]
    ):
        raise ValueError("opportunity training mutated a frozen C0 parameter")


def _all_parameter_count(model: object) -> int:
    return sum(int(value.size) for _, value in tree_flatten(model.parameters()))


def _all_parameter_layout_blake3(model: object) -> str:
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(model.parameters())
    ]
    return _canonical_blake3(layout)


def _all_parameter_tensor_blake3(model: object) -> str:
    return _selected_parameter_tensor_blake3(
        tree_flatten(model.parameters())
    )


def _base_parameter_tensor_blake3(model: object) -> str:
    selected = [
        (name, value)
        for name, value in tree_flatten(model.parameters())
        if name.split(".", 1)[0] not in ADAPTER_PARAMETER_ROOTS
    ]
    return _selected_parameter_tensor_blake3(selected)


def _selected_parameter_tensor_blake3(
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


def _validation_probe_rows(group_count: int) -> np.ndarray:
    count = min(VALIDATION_PROBE_GROUPS, group_count)
    return np.unique(
        np.linspace(0, group_count - 1, count, dtype=np.int64)
    )


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "default_device": str(mx.default_device()),
        "host": normalize_host(socket.gethostname().split(".")[0]),
    }


def _require_production_runtime(runtime: dict[str, Any]) -> None:
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    if (
        runtime.get("machine") != "arm64"
        or "gpu" not in str(mx.default_device()).lower()
    ):
        raise ValueError(
            "opportunity production training requires Apple Silicon MLX GPU"
        )


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
        while block := handle.read(1024 * 1024):
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


def _load_batch_trace(
    path: Path,
    *,
    expected_steps: int,
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read opportunity batch trace: {error}") from error
    if (
        len(events) != expected_steps
        or any(not isinstance(event, dict) for event in events)
        or [event.get("step") for event in events]
        != list(range(1, expected_steps + 1))
        or any(
            event.get("schema_version") != 1
            or not isinstance(event.get("batch_blake3"), str)
            or len(event["batch_blake3"]) != 64
            or not math.isfinite(float(event.get("loss", math.nan)))
            or int(event.get("candidates", 0)) <= 0
            or float(event.get("elapsed_seconds", 0.0)) <= 0.0
            for event in events
        )
    ):
        raise ValueError(
            "opportunity batch trace does not match the resume checkpoint"
        )
    return events


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one opportunity query-conditioning arm"
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument("--relational-cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--r6-binary", type=Path, required=True)
    parser.add_argument("--warm-start-run-dir", type=Path, required=True)
    parser.add_argument("--warm-start-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--preflight", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_opportunity_cross_attention_training(
        OpportunityCrossAttentionTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            r3_cache=args.r3_cache,
            relational_cache=args.relational_cache,
            s1_cache=args.s1_cache,
            r6_binary=args.r6_binary,
            warm_start_run_dir=args.warm_start_run_dir,
            warm_start_report=args.warm_start_report,
            run_dir=args.run_dir,
            output=args.output,
            arm=args.arm,
            resume=args.resume,
            smoke_steps=args.smoke_steps,
            authorization=args.authorization,
            preflight=args.preflight,
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
