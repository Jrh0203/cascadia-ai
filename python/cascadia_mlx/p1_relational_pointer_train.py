"""Matched MLX training for exact-R2 selected-prefix pointer rankers."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import shutil
import socket
import time
import uuid
from dataclasses import asdict, dataclass
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
    load_checkpoint_pointer_with_factory,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    LEARNING_RATE,
    STAGE_BATCH_SIZES,
    STAGE_EPOCHS,
    STAGE_WIDTHS,
    STAGES,
    WEIGHT_DECAY,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.p1_relational_pointer_data import (
    DEFAULT_FACTOR_CACHE,
    DEFAULT_R3_CACHE,
    PointerParentBatch,
    PointerStageBatch,
    RelationalPointerCorpus,
    validate_pointer_batch,
)
from cascadia_mlx.p1_relational_pointer_model import (
    PointerParentEncoding,
    RelationalPointerModelConfig,
    RelationalPointerRanker,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    relational_pointer_loss,
    trainable_parameter_names,
)
from cascadia_mlx.run_manifest import source_provenance

SCHEMA_VERSION = 1
EXPERIMENT_ID = "p1-relational-selected-prefix-pointer-pilot-v1"
PROTOCOL_ID = "matched-mlx-selected-prefix-pointer-pilot-v1"
ADR_ID = "0175"
FOUNDATION_EXPERIMENT_ID = "p1-relational-hierarchical-pointer-foundation-v1"
FOUNDATION_PROTOCOL_ID = "exact-r2-selected-prefix-pointer-alignment-v1"
FOUNDATION_ADR_ID = "0174"
FOUNDATION_PASS = "p1_relational_pointer_foundation_passed"
AUTHORIZED_SUCCESSOR = "matched-mlx-selected-prefix-pointer-pilot"

EXPECTED_WARM_START_EXPERIMENT = "relational-substrate-mlx-tournament-v1"
EXPECTED_WARM_START_PROTOCOL = "r5-s3-s5-matched-mlx-v1"
EXPECTED_WARM_START_ADR = "0161"
EXPECTED_WARM_START_CHECKPOINT = (
    "step-000003000-epoch-0000-batch-003000"
)
EXPECTED_WARM_START_MANIFEST_BLAKE3 = (
    "a7e31e2713a2afd642f7143fee3d9071c9776ee88ca7bbed61564d6e7b12b9d3"
)
EXPECTED_WARM_START_MODEL_BLAKE3 = (
    "eadcfbd5d0f02d642e7003431809b9ae8c41f0c3faf12c57d6da84a18acc5b89"
)
EXPECTED_WARM_START_PARAMETER_BLAKE3 = (
    "563ff390fe5815e590e937b336f2c77100fee840bf1f5602cfb547d8356adbbf"
)
EXPECTED_PARENT_PARAMETER_BLAKE3 = (
    "51c54d58edd536c139e5ff3b92cefe85d45bdfe5177387a4affa904dce7f73cf"
)

STAGE_SEEDS = {
    "draft": 2026061675,
    "tile": 2026061676,
    "wildlife": 2026061677,
}
CHECKPOINT_BATCHES = 250
DEFAULT_WARM_START_CHECKPOINT = Path(
    "artifacts/experiments/relational-substrate-mlx-tournament-v1/"
    "runs/c0_exact_r2/checkpoints/"
    f"{EXPECTED_WARM_START_CHECKPOINT}"
)
DEFAULT_WARM_START_REPORT = Path(
    "artifacts/experiments/relational-substrate-mlx-tournament-v1/"
    "reports/c0_exact_r2.json"
)
DEFAULT_FOUNDATION_CLASSIFICATION = Path(
    "artifacts/experiments/"
    "p1-relational-hierarchical-pointer-foundation-v1/classification.json"
)


@dataclass(frozen=True)
class PointerStageTrainingProtocol:
    """Frozen matched comparison for one conditional pointer stage."""

    stage: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_batches: int = CHECKPOINT_BATCHES

    @classmethod
    def frozen(cls, stage: str) -> PointerStageTrainingProtocol:
        if stage not in STAGES:
            raise ValueError("pointer protocol names an unknown stage")
        return cls(
            stage=stage,
            seed=STAGE_SEEDS[stage],
            epochs=STAGE_EPOCHS[stage],
            batch_size=STAGE_BATCH_SIZES[stage],
        )

    def validate(self) -> None:
        if self != PointerStageTrainingProtocol.frozen(self.stage):
            raise ValueError("pointer stage protocol drifted")


@dataclass(frozen=True)
class PointerStageTrainingConfig:
    """Files and launch controls for one stage run."""

    stage: str
    run_dir: Path
    output: Path
    factor_cache: Path = DEFAULT_FACTOR_CACHE
    r3_cache: Path = DEFAULT_R3_CACHE
    warm_start_checkpoint: Path = DEFAULT_WARM_START_CHECKPOINT
    warm_start_report: Path = DEFAULT_WARM_START_REPORT
    foundation_classification: Path | None = DEFAULT_FOUNDATION_CLASSIFICATION
    bundle_id: str | None = None
    resume: bool = False
    smoke_batches: int | None = None

    @property
    def production(self) -> bool:
        return self.smoke_batches is None

    @property
    def protocol(self) -> PointerStageTrainingProtocol:
        return PointerStageTrainingProtocol.frozen(self.stage)

    def validate(self) -> None:
        self.protocol.validate()
        if self.run_dir == self.output:
            raise ValueError("pointer run directory and output path must differ")
        if self.production:
            if (
                self.foundation_classification is None
                or self.bundle_id is None
                or len(self.bundle_id) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.bundle_id
                )
            ):
                raise ValueError(
                    "production pointer training requires classification and bundle ID"
                )
        elif (
            self.smoke_batches is None
            or self.smoke_batches <= 0
            or self.smoke_batches > 10
            or self.resume
            or self.foundation_classification is not None
        ):
            raise ValueError(
                "bounded pointer smoke must be fresh, ungated, and at most 10 batches"
            )


@dataclass
class ParentEncodingMemoStats:
    requested_parents: int = 0
    encoded_parents: int = 0
    cache_hits: int = 0

    def report(self) -> dict[str, int | float]:
        return {
            "requested_parents": self.requested_parents,
            "encoded_parents": self.encoded_parents,
            "cache_hits": self.cache_hits,
            "hit_fraction": self.cache_hits / max(self.requested_parents, 1),
        }


class PointerParentEncodingMemo:
    """Reuse frozen C0 encodings within one fixed D6 schedule."""

    def __init__(self) -> None:
        self._values: dict[tuple[int, int], PointerParentEncoding] = {}
        self.stats = ParentEncodingMemoStats()

    def encoding(
        self,
        model: RelationalPointerRanker,
        batch: PointerStageBatch,
    ) -> PointerParentEncoding:
        keys = [
            (int(group_id), int(transform_id))
            for group_id, transform_id in zip(
                batch.parent_group_ids,
                batch.parent_transform_ids,
                strict=True,
            )
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("pointer batch contains duplicate parent identities")
        self.stats.requested_parents += len(keys)
        missing_positions = [
            index for index, key in enumerate(keys) if key not in self._values
        ]
        self.stats.cache_hits += len(keys) - len(missing_positions)
        if missing_positions:
            subset = _slice_parent_batch(batch.parent, missing_positions)
            encoded = model.encode_parent(subset)
            mx.eval(
                encoded.summary,
                encoded.active_tokens,
                encoded.active_mask,
                encoded.active_types,
            )
            for encoded_row, source_position in enumerate(missing_positions):
                key = keys[source_position]
                self._values[key] = PointerParentEncoding(
                    summary=encoded.summary[encoded_row : encoded_row + 1],
                    active_tokens=encoded.active_tokens[
                        encoded_row : encoded_row + 1
                    ],
                    active_mask=encoded.active_mask[
                        encoded_row : encoded_row + 1
                    ],
                    active_types=encoded.active_types[
                        encoded_row : encoded_row + 1
                    ],
                )
            self.stats.encoded_parents += len(missing_positions)
        values = [self._values[key] for key in keys]
        return PointerParentEncoding(
            summary=mx.concatenate([value.summary for value in values], axis=0),
            active_tokens=mx.concatenate(
                [value.active_tokens for value in values],
                axis=0,
            ),
            active_mask=mx.concatenate(
                [value.active_mask for value in values],
                axis=0,
            ),
            active_types=mx.concatenate(
                [value.active_types for value in values],
                axis=0,
            ),
        )


def run_pointer_stage_training(
    config: PointerStageTrainingConfig,
) -> dict[str, Any]:
    """Train, select, and evaluate one exact selected-prefix pointer stage."""
    overall_started = time.perf_counter()
    config.validate()
    protocol = config.protocol
    mx.set_default_device(mx.gpu)
    allocator = configure_mlx_memory()
    runtime = _runtime_identity()
    if config.production:
        _require_production_runtime(runtime)
    source = source_provenance(Path(__file__).resolve().parents[2])
    foundation = (
        require_foundation_classification(config.foundation_classification)
        if config.production
        else None
    )
    train = RelationalPointerCorpus(
        split="train",
        factor_cache=config.factor_cache,
        r3_cache=config.r3_cache,
        verify_r3_checksums=not config.production,
        verify_r3_semantics=not config.production,
    )
    validation_manifest = _read_json(
        config.factor_cache / "validation/manifest.json",
        "validation factor manifest",
    )
    model, warm_start = load_verified_c0_parent(
        checkpoint_dir=config.warm_start_checkpoint,
        report_path=config.warm_start_report,
        stage=config.stage,
        seed=protocol.seed,
    )
    initial_pointer_hash = parameter_tensor_blake3(
        model,
        trainable_only=True,
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "stage": config.stage,
        "bundle_id": config.bundle_id,
        "protocol": asdict(protocol),
        "smoke_batches": config.smoke_batches,
        "factor_cache": {
            "train_payload_blake3": train.factor.manifest["payload_blake3"],
            "validation_payload_blake3": validation_manifest["payload_blake3"],
            "train_dataset_manifest_blake3": train.factor.manifest[
                "dataset_manifest_blake3"
            ],
            "validation_dataset_manifest_blake3": validation_manifest[
                "dataset_manifest_blake3"
            ],
        },
        "r3_cache_id": train.r3.cache_id,
        "warm_start": warm_start,
        "foundation": foundation,
        "model_config": model.config.to_dict(),
        "initial_pointer_parameter_tensor_blake3": initial_pointer_hash,
        "source": source,
        "runtime": runtime,
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used_for_selection": False,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_read": False,
            "future_refill_read": False,
        },
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_path = config.run_dir / "run.json"
    if config.resume:
        existing = _read_json(run_path, "pointer run manifest")
        if existing != run_manifest:
            raise ValueError("pointer resume manifest differs from frozen run")
        model, optimizer, state, latest = (
            load_latest_checkpoint_with_factory(
                config.run_dir,
                learning_rate=protocol.learning_rate,
                weight_decay=protocol.weight_decay,
                model_factory=_model_factory,
            )
        )
        model.freeze_parent_for_pointer_training()
        _require_frozen_parent(model)
        resume_metadata = _read_json(
            latest / "checkpoint.json",
            "latest pointer checkpoint",
        ).get("metadata", {})
        _reconcile_epoch_checkpoint(
            config.run_dir,
            latest,
            resume_metadata,
        )
    else:
        if any(config.run_dir.iterdir()):
            raise ValueError("fresh pointer run directory is not empty")
        _write_json_atomic(run_path, run_manifest)
        optimizer = optim.AdamW(
            learning_rate=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
        )
        state = TrainerState()
        latest = None
        resume_metadata = {}

    target_epochs = protocol.epochs if config.production else 1
    events = _read_metric_events(config.run_dir / "metrics.jsonl")
    best_key = max(
        (
            tuple(float(value) for value in event["selection_key"])
            for event in events
        ),
        default=None,
    )
    training_started = time.perf_counter()
    starting_elapsed = float(state.elapsed_seconds)
    parent_hash = parameter_tensor_blake3(model, parent_only=True)
    if parent_hash != EXPECTED_PARENT_PARAMETER_BLAKE3:
        raise ValueError("pointer parent changed before optimization")
    loss_and_grad = nn.value_and_grad(model, relational_pointer_loss)
    epoch_loss_sum = 0.0
    epoch_batches = 0
    if (
        latest is not None
        and resume_metadata.get("kind") == "mid-epoch"
        and int(resume_metadata.get("epoch", -1)) == state.epoch
    ):
        epoch_loss_sum = float(resume_metadata["epoch_loss_sum"])
        epoch_batches = int(resume_metadata["epoch_batches"])

    while state.epoch < target_epochs:
        epoch = state.epoch
        skip_batches = state.batch_in_epoch
        parent_memo = PointerParentEncodingMemo()
        model.train()
        items_seen = 0
        queries_seen = 0
        executed_batches = epoch_batches
        for batch_index, batch in enumerate(
            train.iter_stage_batches(
                stage=config.stage,
                batch_size=protocol.batch_size,
                shuffle=True,
                seed=protocol.seed,
                epoch=epoch,
                d6_augment=True,
            )
        ):
            if config.smoke_batches is not None and batch_index >= config.smoke_batches:
                break
            if batch_index < skip_batches:
                continue
            validate_pointer_batch(batch, stage=config.stage)
            parent_encoding = parent_memo.encoding(model, batch)
            loss, gradients = loss_and_grad(model, batch, parent_encoding)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value) or not _tree_finite(
                model.parameters()
            ) or not _tree_finite(optimizer.state):
                raise RuntimeError("pointer training became nonfinite")
            epoch_loss_sum += loss_value
            epoch_batches += 1
            executed_batches += 1
            queries_seen += int(batch.item_mask.shape[0])
            items_seen += int(np.asarray(batch.item_mask).sum())
            state.global_step += 1
            state.batch_in_epoch = batch_index + 1
            state.elapsed_seconds = (
                starting_elapsed + time.perf_counter() - training_started
            )
            if (
                config.production
                and state.global_step % protocol.checkpoint_batches == 0
            ):
                latest = save_checkpoint(
                    config.run_dir,
                    model,
                    optimizer,
                    state,
                    metadata={
                        "kind": "mid-epoch",
                        "epoch": epoch,
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_batches": epoch_batches,
                    },
                )
                prune_checkpoints(config.run_dir, keep_recent=2)

        if executed_batches == 0:
            raise RuntimeError("pointer epoch executed no batches")
        _require_frozen_parent(model)
        train_metrics = evaluate_pointer_stage(
            model=model,
            corpus=train,
            stage=config.stage,
            batch_size=protocol.batch_size,
            max_batches=config.smoke_batches,
        )
        selection_key = calibrated_stage_selection_key(train_metrics)
        selected = best_key is None or selection_key > best_key
        event_identity = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "stage": config.stage,
            "epoch": epoch + 1,
            "global_step": state.global_step,
            "train_loss": epoch_loss_sum / max(epoch_batches, 1),
            "train": train_metrics,
            "selection_key": list(selection_key),
            "selected": selected,
            "pointer_parameter_tensor_blake3": parameter_tensor_blake3(
                model,
                trainable_only=True,
            ),
            "parent_parameter_tensor_blake3": parameter_tensor_blake3(
                model,
                parent_only=True,
            ),
        }
        event = {
            "schema_version": SCHEMA_VERSION,
            "scientific_identity": event_identity,
            "scientific_blake3": _canonical_blake3(event_identity),
            "runtime": {
                "elapsed_seconds": state.elapsed_seconds,
                "training_queries_seen_after_resume": queries_seen,
                "training_items_seen_after_resume": items_seen,
                "parent_encoding_memo": parent_memo.stats.report(),
            },
        }
        state.epoch = epoch + 1
        state.batch_in_epoch = 0
        latest = save_checkpoint(
            config.run_dir,
            model,
            optimizer,
            state,
            metadata={
                "kind": "epoch-complete",
                "event": event,
                "selected": selected,
            },
        )
        _append_metric_event_once(
            config.run_dir / "metrics.jsonl",
            {**event, "checkpoint": latest.name},
        )
        if selected:
            best_key = selection_key
            set_checkpoint_pointer(
                config.run_dir,
                "best",
                latest,
                metadata={
                    "selection_key": list(selection_key),
                    "epoch": epoch + 1,
                },
            )
        prune_checkpoints(config.run_dir, keep_recent=2)
        print(
            json.dumps(
                {**event, "checkpoint": latest.name},
                sort_keys=True,
            ),
            flush=True,
        )
        epoch_loss_sum = 0.0
        epoch_batches = 0
        mx.clear_cache()

    best_model, best_optimizer, _best_state, best_checkpoint = (
        load_checkpoint_pointer_with_factory(
            config.run_dir,
            pointer="best",
            learning_rate=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
            model_factory=_model_factory,
        )
    )
    del best_optimizer
    best_model.freeze_parent_for_pointer_training()
    _require_frozen_parent(best_model)
    train_metrics = evaluate_pointer_stage(
        model=best_model,
        corpus=train,
        stage=config.stage,
        batch_size=protocol.batch_size,
        max_batches=config.smoke_batches,
    )
    validation = RelationalPointerCorpus(
        split="validation",
        factor_cache=config.factor_cache,
        r3_cache=config.r3_cache,
        verify_r3_checksums=False,
        verify_r3_semantics=False,
    )
    validation_metrics = evaluate_pointer_stage(
        model=best_model,
        corpus=validation,
        stage=config.stage,
        batch_size=protocol.batch_size,
        max_batches=config.smoke_batches,
    )
    checkpoint_manifest = _read_json(
        best_checkpoint / "checkpoint.json",
        "selected pointer checkpoint",
    )
    weights = best_checkpoint / "model.safetensors"
    published = publish_selected_checkpoint(
        run_dir=config.run_dir,
        checkpoint=best_checkpoint,
    )
    report_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "stage": config.stage,
        "bundle_id": config.bundle_id,
        "protocol": asdict(protocol),
        "foundation": foundation,
        "warm_start": warm_start,
        "selected_checkpoint": {
            "name": best_checkpoint.name,
            "manifest_blake3": _checksum(
                best_checkpoint / "checkpoint.json"
            ),
            "model_blake3": _checksum(weights),
            "published": published,
        },
        "model": {
            "config": best_model.config.to_dict(),
            "total_parameter_count": _all_parameter_count(best_model),
            "trainable_pointer_parameter_count": parameter_count(best_model),
            "total_parameter_layout_blake3": parameter_layout_blake3(
                best_model
            ),
            "pointer_parameter_layout_blake3": parameter_layout_blake3(
                best_model,
                trainable_only=True,
            ),
            "final_pointer_parameter_tensor_blake3": (
                parameter_tensor_blake3(
                    best_model,
                    trainable_only=True,
                )
            ),
            "frozen_parent_parameter_tensor_blake3": (
                parameter_tensor_blake3(
                    best_model,
                    parent_only=True,
                )
            ),
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "selection": {
            "source": "open-train-only",
            "key": list(calibrated_stage_selection_key(train_metrics)),
            "validation_used_for_selection": False,
        },
        "information_boundary": run_manifest["information_boundary"],
        "claims": {
            "bounded_smoke_complete": not config.production,
            "offline_stage_comparison_complete": config.production,
            "integrated_proposal_measured": False,
            "gameplay_strength_measured": False,
            "progress_to_100_claimed": False,
        },
        "source": source,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific_identity": report_identity,
        "scientific_blake3": _canonical_blake3(report_identity),
        "runtime": {
            **runtime,
            "elapsed_seconds": time.perf_counter() - overall_started,
            "optimization_and_evaluation_seconds": (
                starting_elapsed + time.perf_counter() - training_started
            ),
            "resource_usage": _resource_usage(),
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
            "selected_checkpoint_manifest": checkpoint_manifest,
        },
    }
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def publish_selected_checkpoint(
    *,
    run_dir: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    """Atomically expose the selected model at a collection-stable path."""
    expected_parent = (run_dir / "checkpoints").resolve()
    if checkpoint.resolve().parent != expected_parent:
        raise ValueError("selected pointer checkpoint is outside its run")
    source_manifest = checkpoint / "checkpoint.json"
    source_model = checkpoint / "model.safetensors"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(source_manifest),
        "model_blake3": _checksum(source_model),
    }
    destination = run_dir / "selected"
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        existing = _read_json(selection_path, "published pointer selection")
        if existing != identity:
            raise ValueError("published pointer selection collides with another model")
        if (
            _checksum(destination / "checkpoint.json")
            != identity["checkpoint_manifest_blake3"]
            or _checksum(destination / "model.safetensors")
            != identity["model_blake3"]
        ):
            raise ValueError("published pointer selection failed integrity")
        return identity
    temporary = run_dir / f".selected.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        shutil.copy2(source_manifest, temporary / "checkpoint.json")
        shutil.copy2(source_model, temporary / "model.safetensors")
        _write_json_atomic(temporary / "selection.json", identity)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return identity


def calibrated_stage_selection_key(
    metrics: dict[str, Any],
) -> tuple[float, float, float]:
    """Match ADR 0115 train-only checkpoint selection exactly."""
    return (
        float(metrics["target_factor_recall"]),
        float(metrics["exact_query_fraction"]),
        -float(metrics["rank_mean_absolute_error"]),
    )


def evaluate_pointer_stage(
    *,
    model: RelationalPointerRanker,
    corpus: RelationalPointerCorpus,
    stage: str,
    batch_size: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Score one open split once with deterministic identity orientation."""
    if stage not in STAGES or batch_size <= 0:
        raise ValueError("pointer evaluation request is invalid")
    model.eval()
    memo = PointerParentEncodingMemo()
    target_total = 0
    target_hits = 0
    exact = 0
    queries = 0
    items = 0
    absolute_error = 0.0
    ranked_items = 0
    finite = True
    batches = 0
    for batch in corpus.iter_stage_batches(
        stage=stage,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        epoch=0,
        d6_augment=False,
    ):
        if max_batches is not None and batches >= max_batches:
            break
        validate_pointer_batch(batch, stage=stage)
        encoding = memo.encoding(model, batch)
        scores = model(batch, parent_encoding=encoding)
        mx.eval(scores)
        values = np.asarray(scores)
        item_mask = np.asarray(batch.item_mask)
        targets = np.asarray(batch.target)
        rank_mask = np.asarray(batch.expected_rank_mask)
        ranks = np.asarray(batch.expected_rank)
        source_items = batch.source_item_indices
        valid_scores = values[item_mask]
        finite &= bool(np.all(np.isfinite(valid_scores)))
        for row in range(values.shape[0]):
            valid = np.flatnonzero(item_mask[row])
            ordering = valid[
                np.lexsort(
                    (
                        source_items[row, valid],
                        -values[row, valid],
                    )
                )
            ]
            width = min(STAGE_WIDTHS[stage], len(valid))
            selected = ordering[:width]
            quota = int(np.sum(targets[row, valid]))
            hits = int(np.sum(targets[row, selected]))
            target_total += quota
            target_hits += hits
            exact += int(hits == quota)
            queries += 1
        ranked = item_mask & rank_mask
        absolute_error += float(
            np.sum(
                np.abs(
                    values[ranked]
                    + np.log1p(ranks[ranked])
                )
            )
        )
        ranked_items += int(np.sum(ranked))
        items += int(np.sum(item_mask))
        batches += 1
    expected_queries = int(corpus.factor.manifest["queries"][stage])
    expected_items = int(corpus.factor.manifest["items"][stage])
    return {
        "queries": queries,
        "items": items,
        "batches": batches,
        "target_factors": target_total,
        "target_hits": target_hits,
        "target_factor_recall": target_hits / max(target_total, 1),
        "exact_query_fraction": exact / max(queries, 1),
        "rank_mean_absolute_error": absolute_error / max(ranked_items, 1),
        "all_scores_finite": finite,
        "all_queries_scored_once": queries == expected_queries,
        "all_items_scored_once": items == expected_items,
        "expected_queries": expected_queries,
        "expected_items": expected_items,
        "parent_encoding_memo": memo.stats.report(),
        "d6_augmentation_enabled": False,
    }


def require_foundation_classification(path: Path | None) -> dict[str, Any]:
    """Authorize production only from the exact crossed ADR 0174 pass."""
    if path is None:
        raise ValueError("pointer foundation classification path is absent")
    report = _read_json(path, "pointer foundation classification")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or not isinstance(identity, dict)
        or report.get("scientific_blake3") != _canonical_blake3(identity)
        or identity.get("experiment_id") != FOUNDATION_EXPERIMENT_ID
        or identity.get("protocol_id") != FOUNDATION_PROTOCOL_ID
        or identity.get("adr") != FOUNDATION_ADR_ID
        or identity.get("passed") is not True
        or identity.get("classification") != FOUNDATION_PASS
        or identity.get("authorized_successor") != AUTHORIZED_SUCCESSOR
    ):
        raise ValueError("pointer foundation did not authorize production training")
    return {
        "classification_scientific_blake3": report["scientific_blake3"],
        "classification": identity["classification"],
        "authorized_successor": identity["authorized_successor"],
        "source": identity.get("source"),
        "split_scientific_blake3": {
            split: values["scientific_blake3"]
            for split, values in identity["splits"].items()
        },
    }


def load_verified_c0_parent(
    *,
    checkpoint_dir: Path,
    report_path: Path,
    stage: str,
    seed: int,
) -> tuple[RelationalPointerRanker, dict[str, Any]]:
    """Load only the exact C0 parent tensors into a fresh pointer head."""
    if stage not in STAGES or seed != STAGE_SEEDS[stage]:
        raise ValueError("pointer warm-start stage or seed drifted")
    checkpoint_dir = checkpoint_dir.resolve()
    report_path = report_path.resolve()
    if checkpoint_dir.name != EXPECTED_WARM_START_CHECKPOINT:
        raise ValueError("pointer warm-start checkpoint name drifted")
    manifest_path = checkpoint_dir / "checkpoint.json"
    model_path = checkpoint_dir / "model.safetensors"
    if _checksum(manifest_path) != EXPECTED_WARM_START_MANIFEST_BLAKE3:
        raise ValueError("pointer warm-start checkpoint manifest drifted")
    manifest = _read_json(manifest_path, "C0 checkpoint manifest")
    model_metadata = manifest.get("files", {}).get("model.safetensors", {})
    if (
        model_metadata.get("blake3") != EXPECTED_WARM_START_MODEL_BLAKE3
        or model_path.stat().st_size != int(model_metadata.get("bytes", -1))
        or _checksum(model_path) != EXPECTED_WARM_START_MODEL_BLAKE3
    ):
        raise ValueError("pointer warm-start model file drifted")
    report = _read_json(report_path, "C0 final report")
    checkpoint = report.get("checkpoint", {})
    model_report = report.get("model", {})
    if (
        report.get("experiment_id") != EXPECTED_WARM_START_EXPERIMENT
        or report.get("protocol_id") != EXPECTED_WARM_START_PROTOCOL
        or report.get("adr") != EXPECTED_WARM_START_ADR
        or Path(str(checkpoint.get("path", ""))).name
        != EXPECTED_WARM_START_CHECKPOINT
        or checkpoint.get("manifest_blake3")
        != EXPECTED_WARM_START_MANIFEST_BLAKE3
        or checkpoint.get("model_blake3")
        != EXPECTED_WARM_START_MODEL_BLAKE3
        or model_report.get("config") != manifest.get("model_config")
        or model_report.get("final_parameter_tensor_blake3")
        != EXPECTED_WARM_START_PARAMETER_BLAKE3
    ):
        raise ValueError("pointer warm-start report identity drifted")

    mx.random.seed(seed)
    model = RelationalPointerRanker(
        RelationalPointerModelConfig(stage=stage)
    )
    current = dict(tree_flatten(model.parameters()))
    parent_names = {
        name for name in current if name.startswith("parent_encoder.")
    }
    source_weights = [
        (name, value)
        for name, value in mx.load(model_path).items()
        if name.startswith("parent_encoder.")
    ]
    if {name for name, _value in source_weights} != parent_names:
        raise ValueError("C0 checkpoint parent tensor set drifted")
    for name, value in source_weights:
        if value.shape != current[name].shape:
            raise ValueError(f"C0 parent tensor shape drifted: {name}")
    model.load_weights(source_weights, strict=False)
    mx.eval(model.parameters())
    model.freeze_parent_for_pointer_training()
    _require_frozen_parent(model)
    warm_start = {
        "experiment_id": EXPECTED_WARM_START_EXPERIMENT,
        "protocol_id": EXPECTED_WARM_START_PROTOCOL,
        "adr": EXPECTED_WARM_START_ADR,
        "checkpoint": EXPECTED_WARM_START_CHECKPOINT,
        "manifest_blake3": EXPECTED_WARM_START_MANIFEST_BLAKE3,
        "model_blake3": EXPECTED_WARM_START_MODEL_BLAKE3,
        "full_parameter_tensor_blake3": EXPECTED_WARM_START_PARAMETER_BLAKE3,
        "parent_parameter_tensor_blake3": (
            EXPECTED_PARENT_PARAMETER_BLAKE3
        ),
        "report_blake3": _checksum(report_path),
    }
    warm_start["warm_start_id"] = _canonical_blake3(warm_start)
    return model, warm_start


def _require_frozen_parent(model: RelationalPointerRanker) -> None:
    names = trainable_parameter_names(model)
    if (
        not names
        or any(name.startswith("parent_encoder.") for name in names)
        or parameter_tensor_blake3(model, parent_only=True)
        != EXPECTED_PARENT_PARAMETER_BLAKE3
    ):
        raise ValueError("pointer C0 parent is not exact and frozen")


def _slice_parent_batch(
    parent: PointerParentBatch,
    positions: list[int],
) -> PointerParentBatch:
    indices = mx.array(np.asarray(positions, dtype=np.int32))
    return PointerParentBatch(
        r2_token_features=mx.take(parent.r2_token_features, indices, axis=0),
        r2_token_types=mx.take(parent.r2_token_types, indices, axis=0),
        r2_token_mask=mx.take(parent.r2_token_mask, indices, axis=0),
        relational_values=mx.take(parent.relational_values, indices, axis=0),
        relational_classes=mx.take(parent.relational_classes, indices, axis=0),
        relational_mask=mx.take(parent.relational_mask, indices, axis=0),
        market_features=mx.take(parent.market_features, indices, axis=0),
        market_mask=mx.take(parent.market_mask, indices, axis=0),
        player_features=mx.take(parent.player_features, indices, axis=0),
        player_mask=mx.take(parent.player_mask, indices, axis=0),
        global_features=mx.take(parent.global_features, indices, axis=0),
    )


def _model_factory(values: dict[str, object]) -> RelationalPointerRanker:
    return RelationalPointerRanker(
        RelationalPointerModelConfig.from_dict(values)
    )


def _reconcile_epoch_checkpoint(
    run_dir: Path,
    checkpoint: Path,
    metadata: object,
) -> None:
    if not isinstance(metadata, dict) or metadata.get("kind") != "epoch-complete":
        return
    event = metadata.get("event")
    if not isinstance(event, dict):
        raise ValueError("epoch checkpoint lacks its exact metric event")
    _append_metric_event_once(
        run_dir / "metrics.jsonl",
        {**event, "checkpoint": checkpoint.name},
    )
    if metadata.get("selected") is True:
        identity = event["scientific_identity"]
        set_checkpoint_pointer(
            run_dir,
            "best",
            checkpoint,
            metadata={
                "selection_key": identity["selection_key"],
                "epoch": identity["epoch"],
            },
        )


def _read_metric_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"pointer metrics JSONL is invalid at line {line_number}"
            ) from error
        identity = event.get("scientific_identity")
        if (
            not isinstance(identity, dict)
            or event.get("scientific_blake3") != _canonical_blake3(identity)
        ):
            raise ValueError("pointer metric event identity drifted")
        events.append(
            {
                **event,
                "selection_key": identity["selection_key"],
            }
        )
    epochs = [int(event["scientific_identity"]["epoch"]) for event in events]
    if epochs != sorted(set(epochs)):
        raise ValueError("pointer metric epochs are duplicated or out of order")
    return events


def _append_metric_event_once(path: Path, event: dict[str, Any]) -> None:
    existing = _read_metric_events(path)
    epoch = int(event["scientific_identity"]["epoch"])
    for prior in existing:
        if int(prior["scientific_identity"]["epoch"]) != epoch:
            continue
        if prior["scientific_blake3"] != event["scientific_blake3"]:
            raise ValueError("pointer epoch replay changed scientific metrics")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _all_parameter_count(model: RelationalPointerRanker) -> int:
    return sum(int(value.size) for _name, value in tree_flatten(model.parameters()))


def _tree_finite(tree: object) -> bool:
    for _name, value in tree_flatten(tree):
        array = mx.asarray(value)
        mx.eval(array)
        if not bool(mx.all(mx.isfinite(array)).item()):
            return False
    return True


def _runtime_identity() -> dict[str, Any]:
    return {
        "host": socket.gethostname().split(".")[0],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "default_device": str(mx.default_device()),
    }


def _require_production_runtime(runtime: dict[str, Any]) -> None:
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    if (
        runtime.get("machine") != "arm64"
        or "gpu" not in str(mx.default_device()).lower()
        or not math.isfinite(float(probe.item()))
    ):
        raise ValueError("production pointer training requires Apple MLX GPU")


def _resource_usage() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        maximum_rss *= 1024
    return {
        "maximum_rss_bytes": maximum_rss,
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }


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
        for block in iter(lambda: handle.read(1 << 20), b""):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-cache", type=Path, default=DEFAULT_FACTOR_CACHE)
    parser.add_argument("--r3-cache", type=Path, default=DEFAULT_R3_CACHE)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        default=DEFAULT_WARM_START_CHECKPOINT,
    )
    parser.add_argument(
        "--warm-start-report",
        type=Path,
        default=DEFAULT_WARM_START_REPORT,
    )
    parser.add_argument(
        "--foundation-classification",
        type=Path,
        default=DEFAULT_FOUNDATION_CLASSIFICATION,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--bundle-id")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_pointer_stage_training(
        PointerStageTrainingConfig(
            stage=args.stage,
            run_dir=args.run_dir,
            output=args.output,
            factor_cache=args.factor_cache,
            r3_cache=args.r3_cache,
            warm_start_checkpoint=args.warm_start_checkpoint,
            warm_start_report=args.warm_start_report,
            foundation_classification=(
                None
                if args.smoke_batches is not None
                else args.foundation_classification
            ),
            bundle_id=args.bundle_id,
            resume=args.resume,
            smoke_batches=args.smoke_batches,
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
