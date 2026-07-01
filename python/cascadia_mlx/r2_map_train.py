"""Adapter-driven, exactly resumable R2-MAP training foundation.

No replay file format is assumed here.  A versioned adapter deterministically
maps cursor and sampler state to the next supervised batch, which lets John2's
trajectory contract land independently while checkpoint/recovery behavior is
already testable with a synthetic adapter.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from cascadia_mlx.checkpoint import (
    CheckpointError,
    R2MapCheckpointBundle,
    R2MapCheckpointIdentity,
    R2MapResumeState,
    build_r2_map_checkpoint_bundle,
    load_r2_map_checkpoint_bundle,
    load_r2_map_checkpoint_pointer,
    loss_stream_binding,
    loss_stream_binding_bytes,
    save_r2_map_checkpoint,
    set_r2_map_checkpoint_pointer,
    verify_r2_map_checkpoint_bundle,
    verify_r2_map_checkpoint_files,
)
from cascadia_mlx.r2_map_contracts import (
    CAMPAIGN_ROOT,
    require_local_storage_authority,
)
from cascadia_mlx.r2_map_model import (
    R2MapBatch,
    R2MapModel,
    R2MapModelConfig,
    R2MapPrediction,
    R2MapPublicState,
)
from cascadia_mlx.r2_map_training_contract import (
    R2MapAdapterStep,
    R2MapSupervisedBatch,
    R2MapTrainingAdapter,
)
from cascadia_mlx.r2_map_verify import (
    prediction_panel,
    validate_verification_receipt,
    validate_verification_receipt_value,
)

__all__ = [
    "R2MapAdapterStep",
    "R2MapSupervisedBatch",
    "R2MapTrainer",
    "R2MapTrainerConfig",
    "R2MapTrainingAdapter",
]

TRAINER_SCHEMA = "r2-map-trainer-v7"
LOSS_CONTRACT = "r2-map-selected-action-conditioned-multitask-loss-v3"
LOSS_STREAM_SCHEMA = "r2-map-branch-aware-loss-stream-v1"
SCHEDULER_SCHEMA = "r2-map-cosine-scheduler-v1"
PRIMARY_VALIDATION_METRIC = "primary_score_to_go_loss"
MLX_CACHE_LIMIT_BYTES = 1 << 30
BOOTSTRAP_POLICY_CANDIDATE_CHUNK = 128
TRAINING_COUNTER_NAMES = (
    "draft_groups",
    "draft_candidates",
    "padded_draft_candidates",
    "draft_policy_targets",
    "market_groups",
    "market_actions",
    "market_policy_targets",
)
AUXILIARY_TASK_ORDER = (
    ("components", "score_components"),
    ("bootstrap_policy", "bootstrap_policy"),
    ("opponent_next_action", "opponent_next_action"),
    ("market_survival", "market_survival"),
    ("market_decision_policy", "market_decision_policy"),
)


@dataclass(frozen=True)
class R2MapTrainerConfig:
    run_dir: Path
    run_id: str
    branch_id: str
    source_blake3: str
    dataset_blake3: str
    adapter_protocol_id: str
    group_batch_size: int = 2
    maximum_candidates_per_batch: int = 16_384
    packing_report_binding: dict[str, Any] | None = None
    model_config: R2MapModelConfig = field(default_factory=R2MapModelConfig)
    learning_rate: float = 3e-5
    minimum_learning_rate: float = 3e-6
    weight_decay: float = 1e-4
    warmup_steps: int = 10
    schedule_steps: int = 1_000
    loss_event_interval_steps: int = 20
    mlx_cache_limit_bytes: int = MLX_CACHE_LIMIT_BYTES
    seed: int = 20260618
    panel_id: str = "r2-map-fixed-panel-v1.1"
    normalization: dict[str, Any] | None = None
    auxiliary_loss_weights: dict[str, float] | None = None

    def normalized(self) -> dict[str, Any]:
        values = self.normalization or {
            "score_scale": 100.0,
            "component_scales": [
                23.0,
                23.0,
                23.0,
                23.0,
                23.0,
                30.0,
                28.0,
                28.0,
                28.0,
                40.0,
                20.0,
            ],
        }
        if not isinstance(values.get("score_scale"), int | float) or values["score_scale"] <= 0:
            raise ValueError("R2-MAP score normalization must be positive")
        components = values.get("component_scales")
        if (
            not isinstance(components, list)
            or len(components) != 11
            or not all(isinstance(value, int | float) and value > 0 for value in components)
        ):
            raise ValueError("R2-MAP component normalization must contain 11 positive scales")
        return json.loads(json.dumps(values))

    def auxiliary_weights(self) -> dict[str, float]:
        values = self.auxiliary_loss_weights or {
            "components": 0.25,
            "bootstrap_policy": 0.10,
            "opponent_next_action": 0.05,
            "market_survival": 0.05,
            "market_decision_policy": 0.10,
        }
        if set(values) != {
            "components",
            "bootstrap_policy",
            "opponent_next_action",
            "market_survival",
            "market_decision_policy",
        } or not all(isinstance(value, int | float) and value >= 0 for value in values.values()):
            raise ValueError("R2-MAP auxiliary-loss weight contract drifted")
        return {key: float(value) for key, value in values.items()}

    def validate(self) -> None:
        self.model_config.validate()
        for label, value in (
            ("run_id", self.run_id),
            ("branch_id", self.branch_id),
            ("adapter_protocol_id", self.adapter_protocol_id),
            ("panel_id", self.panel_id),
        ):
            if not value or not isinstance(value, str):
                raise ValueError(f"R2-MAP trainer requires {label}")
        for label, value in (
            ("source_blake3", self.source_blake3),
            ("dataset_blake3", self.dataset_blake3),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"R2-MAP trainer {label} must be a BLAKE3 digest")
        if not 0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("R2-MAP learning-rate bounds are invalid")
        if self.weight_decay < 0 or self.warmup_steps < 0 or self.schedule_steps <= 0:
            raise ValueError("R2-MAP optimizer schedule is invalid")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (
                self.group_batch_size,
                self.maximum_candidates_per_batch,
            )
        ):
            raise ValueError("R2-MAP batch-packing limits are invalid")
        if self.warmup_steps >= self.schedule_steps:
            raise ValueError("R2-MAP warmup must be shorter than the schedule")
        if (
            not isinstance(self.loss_event_interval_steps, int)
            or isinstance(self.loss_event_interval_steps, bool)
            or not 1 <= self.loss_event_interval_steps <= 25
        ):
            raise ValueError("R2-MAP loss-event interval must be between 1 and 25 steps")
        if self.mlx_cache_limit_bytes != MLX_CACHE_LIMIT_BYTES:
            raise ValueError("R2-MAP MLX cache limit must remain at the frozen 1-GiB bound")
        self.packing_binding()
        self.normalized()
        self.auxiliary_weights()

    def packing_binding(self) -> dict[str, Any] | None:
        if self.packing_report_binding is None:
            return None
        expected = {
            "report_relative",
            "report_sha256",
            "report_object_sha256",
            "report_object_token_sha256",
            "publication_receipt_relative",
            "publication_receipt_object_sha256",
            "publication_receipt_sha256",
            "local_write_attestation_relative",
            "local_write_attestation_object_sha256",
            "local_write_attestation_object_token_sha256",
            "local_write_attestation_sha256",
            "local_write_attestation_publication_receipt_relative",
            "local_write_attestation_publication_receipt_object_sha256",
            "local_write_attestation_publication_receipt_object_token_sha256",
            "local_write_attestation_publication_receipt_sha256",
            "bootstrap_phase_barrier_identity_sha256",
            "bootstrap_phase_barrier_sha256",
            "bootstrap_phase_barrier_publication_receipt_sha256",
            "bootstrap_controller_state_sha256",
            "bootstrap_generation_manifest_payload_sha256",
            "bootstrap_generation_manifest_identity_sha256",
            "bootstrap_generation_manifest_publication_receipt_sha256",
            "selected_group_batch_size",
            "maximum_candidates_per_batch",
            "schedule_steps",
            "epochs",
        }
        value = self.packing_report_binding
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("R2-MAP packing-report binding schema differs")
        for name in (
            "report_relative",
            "publication_receipt_relative",
            "local_write_attestation_relative",
            "local_write_attestation_publication_receipt_relative",
        ):
            path = value[name]
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or ".." in Path(path).parts
            ):
                raise ValueError("R2-MAP packing-report path binding is invalid")
        for name in (
            "report_sha256",
            "report_object_sha256",
            "report_object_token_sha256",
            "publication_receipt_object_sha256",
            "publication_receipt_sha256",
            "local_write_attestation_object_sha256",
            "local_write_attestation_object_token_sha256",
            "local_write_attestation_sha256",
            "local_write_attestation_publication_receipt_object_sha256",
            "local_write_attestation_publication_receipt_object_token_sha256",
            "local_write_attestation_publication_receipt_sha256",
            "bootstrap_phase_barrier_identity_sha256",
            "bootstrap_phase_barrier_sha256",
            "bootstrap_phase_barrier_publication_receipt_sha256",
            "bootstrap_controller_state_sha256",
            "bootstrap_generation_manifest_payload_sha256",
            "bootstrap_generation_manifest_identity_sha256",
            "bootstrap_generation_manifest_publication_receipt_sha256",
        ):
            digest = value[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("R2-MAP packing-report digest binding is invalid")
        if (
            value["selected_group_batch_size"] != self.group_batch_size
            or value["maximum_candidates_per_batch"] != self.maximum_candidates_per_batch
            or value["schedule_steps"] != self.schedule_steps
            or value["epochs"] != 12
            or value["local_write_attestation_publication_receipt_relative"]
            != (
                "control/receipts/req-john1-attestation-"
                f"{value['local_write_attestation_sha256'][:32]}.json"
            )
        ):
            raise ValueError("R2-MAP packing report differs from the trainer schedule")
        return json.loads(json.dumps(value, allow_nan=False))

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": TRAINER_SCHEMA,
            "run_id": self.run_id,
            "source_blake3": self.source_blake3,
            "dataset_blake3": self.dataset_blake3,
            "adapter_protocol_id": self.adapter_protocol_id,
            "group_batch_size": self.group_batch_size,
            "maximum_candidates_per_batch": self.maximum_candidates_per_batch,
            "packing_report_binding": self.packing_binding(),
            "model_config": self.model_config.to_dict(),
            "learning_rate": self.learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "schedule_steps": self.schedule_steps,
            "loss_event_interval_steps": self.loss_event_interval_steps,
            "mlx_cache_limit_bytes": self.mlx_cache_limit_bytes,
            "seed": self.seed,
            "panel_id": self.panel_id,
            "normalization": self.normalized(),
            "auxiliary_loss_weights": self.auxiliary_weights(),
        }


def _selected_r2_map_loss_components(
    model: R2MapModel,
    batch: R2MapSupervisedBatch,
    *,
    normalization: dict[str, Any],
) -> dict[str, mx.array]:
    groups, _ = batch.validate()
    selected_inputs = _selected_r2_map_batch(
        batch.inputs,
        batch.selected_action_index,
    )
    selected_prediction = model(selected_inputs)
    weights = batch.score_target_mask.astype(mx.float32)
    selected_weights = _selected_candidate(weights[..., None], batch.selected_action_index)
    candidate_count = mx.sum(selected_weights)
    score_scale = float(normalization["score_scale"])
    selected_score_targets = _selected_candidate(
        batch.score_to_go_targets[..., None],
        batch.selected_action_index,
    )
    draft_numerator = mx.sum(
        mx.square(
            (selected_prediction.predicted_score_to_go[:, 0] - selected_score_targets[:, 0])
            / score_scale
        )
        * selected_weights[:, 0]
    )
    market_numerator = mx.array(0.0, dtype=mx.float32)
    market_count = mx.array(0.0, dtype=mx.float32)
    market_policy = mx.array(0.0, dtype=mx.float32)
    if batch.market_decisions is not None:
        market = batch.market_decisions
        market_prediction = model.score_market_decisions(market.inputs)
        market_weights = market.score_target_mask.astype(mx.float32)
        market_count = mx.sum(market_weights)
        market_numerator = mx.sum(
            mx.square(
                (market_prediction.predicted_score_to_go - market.score_to_go_targets) / score_scale
            )
            * market_weights
        )
        market_policy = _masked_group_mean(
            nn.losses.cross_entropy(
                market_prediction.bootstrap_policy_logits,
                market.selected_action_index,
                reduction="none",
            ),
            market.policy_target_mask,
        )
    primary = (draft_numerator + market_numerator) / mx.maximum(candidate_count + market_count, 1.0)
    market_loss = market_numerator / mx.maximum(market_count, 1.0)
    component_scale = mx.array(normalization["component_scales"], dtype=mx.float32)
    selected_component_targets = _selected_candidate(
        batch.score_component_targets,
        batch.selected_action_index,
    )
    selected_component_weights = _selected_candidate(
        weights[..., None],
        batch.selected_action_index,
    )
    components = mx.sum(
        mx.square(
            (
                selected_prediction.predicted_score_components_to_go[:, 0, :]
                - selected_component_targets
            )
            / component_scale
        )
        * selected_component_weights
    ) / (mx.maximum(candidate_count, 1.0) * 11.0)
    opponent = _opponent_loss(selected_prediction, batch, preselected=True)
    survival = _survival_loss(selected_prediction, batch, preselected=True)
    return {
        "primary_score_to_go": primary,
        "score_components": components,
        "opponent_next_action": opponent,
        "market_survival": survival,
        "market_decision_score_to_go": market_loss,
        "market_decision_policy": market_policy,
        "groups": mx.array(float(groups), dtype=mx.float32),
        "market_decision_groups": market_count,
    }


def r2_map_loss_components(
    model: R2MapModel,
    batch: R2MapSupervisedBatch,
    *,
    normalization: dict[str, Any],
) -> dict[str, mx.array]:
    losses = _selected_r2_map_loss_components(
        model,
        batch,
        normalization=normalization,
    )
    losses["bootstrap_policy"] = _bootstrap_policy_loss(model, batch)
    return losses


def protected_r2_map_gradients(
    model: R2MapModel,
    batch: R2MapSupervisedBatch,
    *,
    normalization: dict[str, Any],
    auxiliary_weights: dict[str, float],
) -> tuple[mx.array, Any, dict[str, mx.array], dict[str, dict[str, dict[str, float | bool]]]]:
    """Keep the primary gradient exact and project only conflicting auxiliaries."""

    def objective(candidate: R2MapModel, values: R2MapSupervisedBatch, name: str) -> mx.array:
        if name == "bootstrap_policy":
            return _bootstrap_policy_loss(candidate, values)
        return _selected_r2_map_loss_components(candidate, values, normalization=normalization)[
            name
        ]

    primary_value_and_gradient = nn.value_and_grad(
        model,
        lambda candidate, values: objective(candidate, values, "primary_score_to_go"),
    )
    primary_loss, primary_gradients = primary_value_and_gradient(model, batch)
    mx.eval(primary_loss, primary_gradients)

    combined = dict(tree_flatten(primary_gradients))
    losses = {"primary_score_to_go": primary_loss}
    diagnostics: dict[str, dict[str, dict[str, float | bool]]] = {}
    total_loss = primary_loss
    for weight_name, loss_name in AUXILIARY_TASK_ORDER:
        if loss_name == "bootstrap_policy" and not _has_bootstrap_policy_targets(batch):
            value = mx.array(0.0, dtype=mx.float32)
            gradients = tree_unflatten(
                [(name, mx.zeros_like(gradient)) for name, gradient in combined.items()]
            )
        elif loss_name == "bootstrap_policy" and _requires_chunked_policy(batch):
            value, gradients = _chunked_bootstrap_policy_value_and_grad(model, batch)
        else:
            value_and_gradient = nn.value_and_grad(
                model,
                lambda candidate, values, selected=loss_name: objective(
                    candidate, values, selected
                ),
            )
            value, gradients = value_and_gradient(model, batch)
        mx.eval(value, gradients)
        projected, task_diagnostics = project_conflicting_auxiliary_gradients(
            primary_gradients,
            gradients,
        )
        weight = auxiliary_weights[weight_name]
        for name, gradient in tree_flatten(projected):
            combined[name] = combined[name] + weight * gradient
        losses[loss_name] = value
        diagnostics[loss_name] = task_diagnostics
        total_loss = total_loss + weight * value
        # Materialize the fixed-order accumulator before releasing this task's
        # graph. Holding all auxiliary backward graphs simultaneously makes
        # peak memory scale with the number of heads on wide imitation screens.
        mx.eval(combined)
    mx.eval(total_loss)
    return total_loss, tree_unflatten(list(combined.items())), losses, diagnostics


def project_conflicting_auxiliary_gradients(
    primary_gradients: Any,
    auxiliary_gradients: Any,
) -> tuple[Any, dict[str, dict[str, float | bool]]]:
    """Project a conflicting auxiliary off the primary, independently by module."""
    primary = dict(tree_flatten(primary_gradients))
    auxiliary = dict(tree_flatten(auxiliary_gradients))
    if primary.keys() != auxiliary.keys():
        raise ValueError("R2-MAP primary and auxiliary gradient layouts differ")
    modules: dict[str, list[str]] = {}
    for name in primary:
        modules.setdefault(name.split(".", 1)[0], []).append(name)
    projected = dict(auxiliary)
    diagnostics: dict[str, dict[str, float | bool]] = {}
    for module, names in sorted(modules.items()):
        dot = sum(mx.sum(primary[name] * auxiliary[name]) for name in names)
        primary_square = sum(mx.sum(mx.square(primary[name])) for name in names)
        auxiliary_square = sum(mx.sum(mx.square(auxiliary[name])) for name in names)
        mx.eval(dot, primary_square, auxiliary_square)
        dot_value = float(dot.item())
        primary_norm = math.sqrt(max(float(primary_square.item()), 0.0))
        auxiliary_norm = math.sqrt(max(float(auxiliary_square.item()), 0.0))
        denominator = primary_norm * auxiliary_norm
        cosine = dot_value / denominator if denominator > 0 else 0.0
        conflict = dot_value < 0.0 and primary_norm > 0.0
        if conflict:
            coefficient = dot / primary_square
            for name in names:
                projected[name] = auxiliary[name] - coefficient * primary[name]
        diagnostics[module] = {
            "primary_norm": primary_norm,
            "auxiliary_norm": auxiliary_norm,
            "cosine": cosine,
            "projected": conflict,
        }
    return tree_unflatten(list(projected.items())), diagnostics


class R2MapTrainer:
    """Small control object whose durable state always names the next batch."""

    def __init__(
        self,
        config: R2MapTrainerConfig,
        adapter: R2MapTrainingAdapter,
        *,
        in_memory: bool = False,
    ):
        config.validate()
        _validate_adapter(config, adapter)
        self.mlx_previous_cache_limit_bytes = int(mx.set_cache_limit(config.mlx_cache_limit_bytes))
        self.config = config
        self.adapter = adapter
        self.run_dir = config.run_dir
        self.in_memory = in_memory
        self.loss_path: Path | None
        self.loss_content: bytes
        if in_memory:
            self.loss_path = None
            self.loss_content = b""
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.loss_path = self.run_dir / "losses/loss-stream.jsonl"
            self.loss_path.parent.mkdir(parents=True, exist_ok=True)
            self.loss_path.touch(exist_ok=True)
            if self.loss_path.stat().st_size:
                raise CheckpointError(
                    "R2-MAP new training cannot attach to an existing loss stream; use resume"
                )
            self.loss_content = b""
        mx.random.seed(config.seed)
        self.model = R2MapModel(config.model_config)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.cursor, self.sampler_state = adapter.initial_state(config.seed)
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.examples_seen = 0
        self.training_counters = {name: 0 for name in TRAINING_COUNTER_NAMES}
        self.loss_head: str | None = None
        self._set_learning_rate(self._learning_rate(0))

    @classmethod
    def resume(
        cls,
        config: R2MapTrainerConfig,
        adapter: R2MapTrainingAdapter,
        *,
        pointer: str = "last_verified",
    ) -> R2MapTrainer:
        config.validate()
        _validate_adapter(config, adapter)
        instance = cls.__new__(cls)
        instance.mlx_previous_cache_limit_bytes = int(
            mx.set_cache_limit(config.mlx_cache_limit_bytes)
        )
        instance.config = config
        instance.adapter = adapter
        instance.run_dir = config.run_dir
        instance.loss_path = config.run_dir / "losses/loss-stream.jsonl"
        instance.in_memory = False
        instance.loss_content = b""
        expected = {
            "run_id": config.run_id,
            "source_blake3": config.source_blake3,
            "dataset_blake3": config.dataset_blake3,
            "model_config_blake3": _canonical_blake3(config.model_config.to_dict()),
            "training_config_blake3": _canonical_blake3(config.identity_dict()),
            "loss_contract_blake3": _loss_contract_blake3(),
        }
        loaded = load_r2_map_checkpoint_pointer(
            config.run_dir,
            pointer,
            model_factory=lambda values: R2MapModel(R2MapModelConfig.from_dict(values)),
            optimizer_factory=lambda: optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            expected_identity=expected,
            loss_stream_path=instance.loss_path,
        )
        _validate_resume_branch(
            instance.loss_path,
            loaded.state.loss_stream,
            checkpoint_branch=loaded.identity.branch_id,
            requested_branch=config.branch_id,
        )
        instance.model = loaded.model
        instance.optimizer = loaded.optimizer
        instance.cursor = dict(loaded.state.cursor)
        instance.sampler_state = dict(loaded.state.sampler_state)
        instance.global_step = loaded.state.global_step
        instance.epoch = loaded.state.epoch
        instance.batch_in_epoch = loaded.state.batch_in_epoch
        instance.examples_seen = loaded.state.examples_seen
        instance.training_counters = dict(loaded.state.training_counters)
        instance.loss_head = loaded.state.loss_stream["head_record_blake3"]
        _validate_resume_dataset_contract(adapter, loaded.state)
        instance._verify_recorded_next_batch(loaded.state)
        _restore_mlx_rng(loaded.state.rng_state)
        expected_scheduler = _scheduler_state(config, instance.global_step)
        if loaded.state.scheduler_state != expected_scheduler:
            raise CheckpointError("R2-MAP scheduler state differs from deterministic schedule")
        instance._set_learning_rate(float(expected_scheduler["learning_rate"]))
        return instance

    @classmethod
    def resume_from_bundle(
        cls,
        config: R2MapTrainerConfig,
        adapter: R2MapTrainingAdapter,
        *,
        bundle: R2MapCheckpointBundle,
        loss_content: bytes,
    ) -> R2MapTrainer:
        """Resume exactly from verified remote objects without local filesystem I/O."""
        config.validate()
        _validate_adapter(config, adapter)
        previous_cache_limit = int(mx.set_cache_limit(config.mlx_cache_limit_bytes))
        expected = {
            "run_id": config.run_id,
            "source_blake3": config.source_blake3,
            "dataset_blake3": config.dataset_blake3,
            "model_config_blake3": _canonical_blake3(config.model_config.to_dict()),
            "training_config_blake3": _canonical_blake3(config.identity_dict()),
            "loss_contract_blake3": _loss_contract_blake3(),
        }
        loaded = load_r2_map_checkpoint_bundle(
            bundle,
            model_factory=lambda values: R2MapModel(R2MapModelConfig.from_dict(values)),
            optimizer_factory=lambda: optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            expected_identity=expected,
            loss_stream=loss_content,
        )
        _validate_resume_branch_bytes(
            loss_content,
            loaded.state.loss_stream,
            checkpoint_branch=loaded.identity.branch_id,
            requested_branch=config.branch_id,
        )
        instance = cls.__new__(cls)
        instance.mlx_previous_cache_limit_bytes = previous_cache_limit
        instance.config = config
        instance.adapter = adapter
        instance.run_dir = config.run_dir
        instance.in_memory = True
        instance.loss_path = None
        instance.loss_content = bytes(loss_content)
        instance.model = loaded.model
        instance.optimizer = loaded.optimizer
        instance.cursor = dict(loaded.state.cursor)
        instance.sampler_state = dict(loaded.state.sampler_state)
        instance.global_step = loaded.state.global_step
        instance.epoch = loaded.state.epoch
        instance.batch_in_epoch = loaded.state.batch_in_epoch
        instance.examples_seen = loaded.state.examples_seen
        instance.training_counters = dict(loaded.state.training_counters)
        instance.loss_head = loaded.state.loss_stream["head_record_blake3"]
        _validate_resume_dataset_contract(adapter, loaded.state)
        instance._verify_recorded_next_batch(loaded.state)
        _restore_mlx_rng(loaded.state.rng_state)
        expected_scheduler = _scheduler_state(config, instance.global_step)
        if loaded.state.scheduler_state != expected_scheduler:
            raise CheckpointError("R2-MAP scheduler state differs from deterministic schedule")
        instance._set_learning_rate(float(expected_scheduler["learning_rate"]))
        return instance

    def peek_next_batch_identity(self) -> str:
        return self.adapter.training_batch(self.cursor, self.sampler_state).batch.batch_identity

    def _verify_recorded_next_batch(self, state: R2MapResumeState) -> None:
        actual = self.peek_next_batch_identity()
        if actual != state.next_batch_identity:
            raise CheckpointError(
                "R2-MAP exact next batch differs from the checkpoint resume state"
            )

    def step(self) -> dict[str, Any] | None:
        adapter_step = self.adapter.training_batch(self.cursor, self.sampler_state)
        batch = adapter_step.batch
        groups, _ = batch.validate()
        padded_draft_candidates = groups * batch.inputs.candidate_mask.shape[1]
        oversize_exact_screen = padded_draft_candidates > self.config.maximum_candidates_per_batch
        if oversize_exact_screen:
            mx.clear_cache()
        market_groups = (
            0 if batch.market_decisions is None else batch.market_decisions.validate()[0]
        )
        draft_candidates = int(np.asarray(batch.inputs.candidate_mask).sum())
        draft_policy_targets = int(np.asarray(batch.bootstrap_policy_mask).sum())
        market_actions = (
            0
            if batch.market_decisions is None
            else int(np.asarray(batch.market_decisions.inputs.action_mask).sum())
        )
        market_policy_targets = (
            0
            if batch.market_decisions is None
            else int(np.asarray(batch.market_decisions.policy_target_mask).sum())
        )
        learning_rate = self._learning_rate(self.global_step)
        self._set_learning_rate(learning_rate)
        normalization = self.config.normalized()
        auxiliary_weights = self.config.auxiliary_weights()
        raw_losses = r2_map_loss_components(
            self.model,
            batch,
            normalization=normalization,
        )
        total_before = (
            raw_losses["primary_score_to_go"]
            + auxiliary_weights["components"] * raw_losses["score_components"]
            + auxiliary_weights["bootstrap_policy"] * raw_losses["bootstrap_policy"]
            + auxiliary_weights["opponent_next_action"] * raw_losses["opponent_next_action"]
            + auxiliary_weights["market_survival"] * raw_losses["market_survival"]
            + auxiliary_weights["market_decision_policy"] * raw_losses["market_decision_policy"]
        )
        mx.eval(total_before, raw_losses)
        loss, gradients, protected_losses, gradient_diagnostics = protected_r2_map_gradients(
            self.model,
            batch,
            normalization=normalization,
            auxiliary_weights=auxiliary_weights,
        )
        self.optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        if oversize_exact_screen:
            mx.clear_cache()

        prior_epoch = int(self.cursor.get("epoch", self.epoch))
        self.cursor = dict(adapter_step.next_cursor)
        self.sampler_state = dict(adapter_step.next_sampler_state)
        self.global_step += 1
        next_epoch = int(self.cursor.get("epoch", prior_epoch))
        if next_epoch < prior_epoch:
            raise RuntimeError("R2-MAP adapter epoch moved backward")
        self.epoch = next_epoch
        self.batch_in_epoch = 0 if next_epoch != prior_epoch else self.batch_in_epoch + 1
        self.examples_seen += groups + market_groups
        for name, value in {
            "draft_groups": groups,
            "draft_candidates": draft_candidates,
            "padded_draft_candidates": groups * batch.inputs.candidate_mask.shape[1],
            "draft_policy_targets": draft_policy_targets,
            "market_groups": market_groups,
            "market_actions": market_actions,
            "market_policy_targets": market_policy_targets,
        }.items():
            self.training_counters[name] += int(value)
        metrics = {
            "total_loss": float(total_before.item()),
            "primary_score_to_go_loss": float(raw_losses["primary_score_to_go"].item()),
            "score_components_loss": float(raw_losses["score_components"].item()),
            "bootstrap_policy_loss": float(raw_losses["bootstrap_policy"].item()),
            "opponent_next_action_loss": float(raw_losses["opponent_next_action"].item()),
            "market_survival_loss": float(raw_losses["market_survival"].item()),
            "market_decision_score_to_go_loss": float(
                raw_losses["market_decision_score_to_go"].item()
            ),
            "market_decision_groups": float(market_groups),
            "draft_candidates": float(draft_candidates),
            "padded_draft_candidates": float(padded_draft_candidates),
            "draft_padding_efficiency": float(draft_candidates) / float(padded_draft_candidates),
            "draft_policy_targets": float(draft_policy_targets),
            "market_actions": float(market_actions),
            "market_policy_targets": float(market_policy_targets),
            "market_decision_policy_loss": float(raw_losses["market_decision_policy"].item()),
            "learning_rate": learning_rate,
        }
        for task, modules in gradient_diagnostics.items():
            for module, diagnostic in modules.items():
                prefix = f"gradient/{task}/{module}"
                metrics[f"{prefix}/primary_norm"] = float(diagnostic["primary_norm"])
                metrics[f"{prefix}/auxiliary_norm"] = float(diagnostic["auxiliary_norm"])
                metrics[f"{prefix}/cosine"] = float(diagnostic["cosine"])
                metrics[f"{prefix}/projected"] = float(bool(diagnostic["projected"]))
        if set(protected_losses) != {
            "primary_score_to_go",
            "score_components",
            "bootstrap_policy",
            "opponent_next_action",
            "market_survival",
            "market_decision_policy",
        }:
            raise RuntimeError("R2-MAP protected loss task order drifted")
        if self.global_step % self.config.loss_event_interval_steps != 0:
            return None
        if self.in_memory:
            self.loss_content, record = append_loss_record_bytes(
                self.loss_content,
                branch_id=self.config.branch_id,
                global_step=self.global_step,
                batch_identity=batch.batch_identity,
                metrics=metrics,
                parent_record_blake3=self.loss_head,
            )
        else:
            assert self.loss_path is not None
            record = append_loss_record(
                self.loss_path,
                branch_id=self.config.branch_id,
                global_step=self.global_step,
                batch_identity=batch.batch_identity,
                metrics=metrics,
                parent_record_blake3=self.loss_head,
            )
        self.loss_head = record["record_blake3"]
        return record

    def validation_metrics(self) -> dict[str, float]:
        losses: list[float] = []
        for batch in self.adapter.validation_batches():
            batch.validate()
            loss = r2_map_loss_components(
                self.model,
                batch,
                normalization=self.config.normalized(),
            )["primary_score_to_go"]
            mx.eval(loss)
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError("R2-MAP validation adapter returned no batches")
        return {PRIMARY_VALIDATION_METRIC: math.fsum(losses) / len(losses)}

    def save_checkpoint(
        self,
        *,
        validation: dict[str, float] | None = None,
        fault_injector: Any | None = None,
    ) -> Path:
        if self.in_memory:
            raise CheckpointError(
                "in-memory training must publish checkpoint_bundle() through remote storage"
            )
        assert self.loss_path is not None
        checkpoint_id = f"{self.config.run_id}.{self.config.branch_id}.step-{self.global_step:09d}"
        identity = R2MapCheckpointIdentity(
            checkpoint_id=checkpoint_id,
            run_id=self.config.run_id,
            branch_id=self.config.branch_id,
            source_blake3=self.config.source_blake3,
            dataset_blake3=self.config.dataset_blake3,
            model_config_blake3=_canonical_blake3(self.config.model_config.to_dict()),
            training_config_blake3=_canonical_blake3(self.config.identity_dict()),
            loss_contract_blake3=_loss_contract_blake3(),
        )
        resume = R2MapResumeState(
            global_step=self.global_step,
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            examples_seen=self.examples_seen,
            cursor=dict(self.cursor),
            sampler_state=dict(self.sampler_state),
            rng_state=_capture_mlx_rng(),
            scheduler_state=_scheduler_state(self.config, self.global_step),
            normalization=self.config.normalized(),
            auxiliary_loss_weights=self.config.auxiliary_weights(),
            dataset_contract=_adapter_dataset_contract(self.adapter),
            training_counters=dict(self.training_counters),
            loss_stream=loss_stream_binding(
                self.loss_path,
                relative_to=self.run_dir,
                head_record_blake3=self.loss_head,
            ),
            next_batch_identity=self.peek_next_batch_identity(),
            validation=validation,
        )
        panel = prediction_panel(
            self.model,
            self.adapter.fixed_prediction_batch(self.config.panel_id),
        )
        return save_r2_map_checkpoint(
            self.run_dir,
            self.model,
            self.optimizer,
            identity,
            resume,
            model_config=self.config.model_config.to_dict(),
            fixed_prediction_panel=panel,
            prediction_panel_id=self.config.panel_id,
            fault_injector=fault_injector,
        )

    def checkpoint_bundle(
        self,
        *,
        validation: dict[str, float] | None = None,
    ) -> R2MapCheckpointBundle:
        """Build one complete checkpoint without writing a local file."""
        checkpoint_id, identity = self._checkpoint_identity()
        content = self.loss_content
        if not self.in_memory:
            assert self.loss_path is not None
            content = self.loss_path.read_bytes()
        resume = self._resume_state(
            loss_stream_binding_bytes(
                content,
                relative_path="losses/loss-stream.jsonl",
                head_record_blake3=self.loss_head,
            ),
            validation=validation,
        )
        panel = prediction_panel(
            self.model,
            self.adapter.fixed_prediction_batch(self.config.panel_id),
        )
        bundle = build_r2_map_checkpoint_bundle(
            self.model,
            self.optimizer,
            identity,
            resume,
            model_config=self.config.model_config.to_dict(),
            fixed_prediction_panel=panel,
            prediction_panel_id=self.config.panel_id,
        )
        if bundle.checkpoint_id != checkpoint_id:
            raise CheckpointError("in-memory checkpoint identity drifted")
        return bundle

    def _checkpoint_identity(self) -> tuple[str, R2MapCheckpointIdentity]:
        checkpoint_id = f"{self.config.run_id}.{self.config.branch_id}.step-{self.global_step:09d}"
        return checkpoint_id, R2MapCheckpointIdentity(
            checkpoint_id=checkpoint_id,
            run_id=self.config.run_id,
            branch_id=self.config.branch_id,
            source_blake3=self.config.source_blake3,
            dataset_blake3=self.config.dataset_blake3,
            model_config_blake3=_canonical_blake3(self.config.model_config.to_dict()),
            training_config_blake3=_canonical_blake3(self.config.identity_dict()),
            loss_contract_blake3=_loss_contract_blake3(),
        )

    def _resume_state(
        self,
        loss_stream: dict[str, Any],
        *,
        validation: dict[str, float] | None,
    ) -> R2MapResumeState:
        return R2MapResumeState(
            global_step=self.global_step,
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            examples_seen=self.examples_seen,
            cursor=dict(self.cursor),
            sampler_state=dict(self.sampler_state),
            rng_state=_capture_mlx_rng(),
            scheduler_state=_scheduler_state(self.config, self.global_step),
            normalization=self.config.normalized(),
            auxiliary_loss_weights=self.config.auxiliary_weights(),
            dataset_contract=_adapter_dataset_contract(self.adapter),
            training_counters=dict(self.training_counters),
            loss_stream=loss_stream,
            next_batch_identity=self.peek_next_batch_identity(),
            validation=validation,
        )

    def _learning_rate(self, step: int) -> float:
        return _learning_rate(self.config, step)

    def _set_learning_rate(self, value: float) -> None:
        self.optimizer.learning_rate = mx.array(value, dtype=mx.float32)


def append_loss_record(
    path: str | Path,
    *,
    branch_id: str,
    global_step: int,
    batch_identity: str,
    metrics: dict[str, float],
    parent_record_blake3: str | None,
) -> dict[str, Any]:
    """Append one fsynced hash-chained record without ever truncating history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        records = _parse_loss_stream(handle.read())
        branch_records = [record for record in records if record["branch_id"] == branch_id]
        expected_parent = (
            branch_records[-1]["record_blake3"] if branch_records else parent_record_blake3
        )
        if branch_records and parent_record_blake3 != expected_parent:
            raise CheckpointError("R2-MAP loss branch parent does not match its current head")
        known_hashes = {record["record_blake3"] for record in records}
        if (
            not branch_records
            and parent_record_blake3 is not None
            and parent_record_blake3 not in known_hashes
        ):
            raise CheckpointError("R2-MAP loss branch parent is absent from the stream")
        record: dict[str, Any] = {
            "schema_version": 1,
            "schema_id": LOSS_STREAM_SCHEMA,
            "stream_sequence": len(records),
            "branch_id": branch_id,
            "branch_sequence": len(branch_records),
            "global_step": global_step,
            "batch_identity": batch_identity,
            "parent_record_blake3": expected_parent,
            "metrics": metrics,
        }
        record["record_blake3"] = _canonical_blake3(record)
        encoded = _canonical_json(record) + b"\n"
        handle.seek(0, os.SEEK_END)
        os.write(handle.fileno(), encoded)
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def append_loss_record_bytes(
    content: bytes,
    *,
    branch_id: str,
    global_step: int,
    batch_identity: str,
    metrics: dict[str, float],
    parent_record_blake3: str | None,
) -> tuple[bytes, dict[str, Any]]:
    """Append one hash-chained record to bounded in-memory remote state."""
    try:
        records = _parse_loss_stream(content.decode())
    except UnicodeDecodeError as error:
        raise CheckpointError("R2-MAP loss stream is not UTF-8") from error
    branch_records = [record for record in records if record["branch_id"] == branch_id]
    expected_parent = (
        branch_records[-1]["record_blake3"] if branch_records else parent_record_blake3
    )
    if branch_records and parent_record_blake3 != expected_parent:
        raise CheckpointError("R2-MAP loss branch parent does not match its current head")
    known_hashes = {record["record_blake3"] for record in records}
    if (
        not branch_records
        and parent_record_blake3 is not None
        and parent_record_blake3 not in known_hashes
    ):
        raise CheckpointError("R2-MAP loss branch parent is absent from the stream")
    record: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": LOSS_STREAM_SCHEMA,
        "stream_sequence": len(records),
        "branch_id": branch_id,
        "branch_sequence": len(branch_records),
        "global_step": global_step,
        "batch_identity": batch_identity,
        "parent_record_blake3": expected_parent,
        "metrics": metrics,
    }
    record["record_blake3"] = _canonical_blake3(record)
    encoded = content + _canonical_json(record) + b"\n"
    _parse_loss_stream(encoded.decode())
    return encoded, record


def validate_loss_stream(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return _parse_loss_stream(path.read_text())


def validate_loss_stream_bytes(content: bytes) -> list[dict[str, Any]]:
    try:
        return _parse_loss_stream(content.decode())
    except UnicodeDecodeError as error:
        raise CheckpointError("R2-MAP loss stream is not UTF-8") from error


def select_best_validation_checkpoint(
    run_dir: str | Path,
    *,
    checkpoint_paths: Sequence[Path] | None = None,
) -> Path:
    """Select by primary validation loss, then lower step, then manifest hash."""
    run_dir = Path(run_dir)
    candidates = list(checkpoint_paths or sorted((run_dir / "checkpoints").iterdir()))
    ranked: list[tuple[float, int, str, Path]] = []
    for path in candidates:
        if not path.is_dir() or path.name.startswith("."):
            continue
        receipt = run_dir / "verifications" / f"{path.name}.json"
        if not receipt.is_file():
            continue
        validate_verification_receipt(receipt, checkpoint_path=path)
        _, state, _ = verify_r2_map_checkpoint_files(path)
        validation = state.validation
        if validation is None:
            continue
        if PRIMARY_VALIDATION_METRIC not in validation:
            raise CheckpointError(
                "R2-MAP best checkpoint selection requires primary validation loss"
            )
        value = float(validation[PRIMARY_VALIDATION_METRIC])
        if not math.isfinite(value):
            raise CheckpointError("R2-MAP validation selection metric is not finite")
        manifest_blake3 = blake3.blake3((path / "checkpoint.json").read_bytes()).hexdigest()
        ranked.append((value, state.global_step, manifest_blake3, path))
    if not ranked:
        raise CheckpointError("no verified R2-MAP validation checkpoint is selectable")
    metric, global_step, manifest_blake3, selected = min(ranked)
    set_r2_map_checkpoint_pointer(
        run_dir,
        "best_validation",
        selected,
        metadata={
            PRIMARY_VALIDATION_METRIC: metric,
            "global_step": global_step,
            "checkpoint_manifest_blake3": manifest_blake3,
            "selection_tiebreak": "global-step-then-checkpoint-manifest-blake3",
        },
    )
    return selected


def select_best_validation_checkpoint_bundle(
    candidates: Sequence[tuple[R2MapCheckpointBundle, Mapping[str, Any]]],
) -> R2MapCheckpointBundle:
    """Apply the frozen primary-loss/step/hash ordering without filesystem I/O."""
    ranked: list[tuple[float, int, str, R2MapCheckpointBundle]] = []
    for bundle, receipt in candidates:
        _, state, _ = verify_r2_map_checkpoint_bundle(bundle)
        validate_verification_receipt_value(
            receipt,
            checkpoint_id=bundle.checkpoint_id,
            checkpoint_manifest_blake3=bundle.manifest_blake3,
            expected_dataset_contract_blake3=_canonical_blake3(state.dataset_contract),
        )
        validation = state.validation
        if validation is None or PRIMARY_VALIDATION_METRIC not in validation:
            raise CheckpointError(
                "R2-MAP best checkpoint selection requires primary validation loss"
            )
        metric = float(validation[PRIMARY_VALIDATION_METRIC])
        if not math.isfinite(metric):
            raise CheckpointError("R2-MAP validation selection metric is not finite")
        ranked.append((metric, state.global_step, bundle.manifest_blake3, bundle))
    if not ranked:
        raise CheckpointError("no verified R2-MAP validation checkpoint is selectable")
    return min(ranked, key=lambda item: item[:3])[3]


def _opponent_loss(
    prediction: R2MapPrediction,
    batch: R2MapSupervisedBatch,
    *,
    preselected: bool = False,
) -> mx.array:
    output = prediction.opponent_next_action
    tile_slot = _training_auxiliary_candidate(
        output.tile_slot_logits, batch.selected_action_index, preselected=preselected
    )
    wildlife_slot = _training_auxiliary_candidate(
        output.wildlife_slot_logits, batch.selected_action_index, preselected=preselected
    )
    draft_kind = _training_auxiliary_candidate(
        output.draft_kind_logits, batch.selected_action_index, preselected=preselected
    )
    drafted_wildlife = _training_auxiliary_candidate(
        output.drafted_wildlife_logits,
        batch.selected_action_index,
        preselected=preselected,
    )
    replace_three = _training_auxiliary_candidate(
        output.replace_three_logits,
        batch.selected_action_index,
        preselected=preselected,
    )
    paid_wipe_count = _training_auxiliary_candidate(
        output.paid_wipe_count_logits,
        batch.selected_action_index,
        preselected=preselected,
    )
    paid_wipe_masks = _training_auxiliary_candidate(
        output.paid_wipe_mask_logits,
        batch.selected_action_index,
        preselected=preselected,
    )
    losses = (
        _masked_cross_entropy(
            tile_slot,
            batch.opponent_tile_slot_targets,
            batch.opponent_valid_mask,
        ),
        _masked_cross_entropy(
            wildlife_slot,
            batch.opponent_wildlife_slot_targets,
            batch.opponent_valid_mask,
        ),
        _masked_cross_entropy(
            draft_kind,
            batch.opponent_draft_kind_targets,
            batch.opponent_valid_mask,
        ),
        _masked_cross_entropy(
            drafted_wildlife,
            batch.opponent_drafted_wildlife_targets,
            batch.opponent_valid_mask,
        ),
        _masked_cross_entropy(
            replace_three,
            batch.opponent_replace_three_targets,
            batch.opponent_valid_mask,
        ),
        _masked_cross_entropy(
            paid_wipe_count,
            batch.opponent_paid_wipe_count_targets,
            batch.opponent_valid_mask,
        ),
        _masked_cross_entropy(
            paid_wipe_masks,
            batch.opponent_paid_wipe_mask_targets,
            batch.opponent_paid_wipe_mask_valid,
        ),
    )
    return sum(losses) / len(losses)


def _survival_loss(
    prediction: R2MapPrediction,
    batch: R2MapSupervisedBatch,
    *,
    preselected: bool = False,
) -> mx.array:
    output = prediction.market_survival
    losses = (
        _masked_cross_entropy(
            _training_auxiliary_candidate(
                output.disposition_logits,
                batch.selected_action_index,
                preselected=preselected,
            ),
            batch.market_disposition_targets,
            batch.market_disposition_mask,
        ),
        _masked_cross_entropy(
            _training_auxiliary_candidate(
                output.pair_survival_logits,
                batch.selected_action_index,
                preselected=preselected,
            ),
            batch.market_pair_survival_targets,
            batch.market_pair_survival_mask,
        ),
        _masked_cross_entropy(
            _training_auxiliary_candidate(
                output.final_slot_logits,
                batch.selected_action_index,
                preselected=preselected,
            ),
            batch.market_final_slot_targets,
            batch.market_final_slot_mask,
        ),
    )
    return sum(losses) / len(losses)


def _requires_chunked_policy(batch: R2MapSupervisedBatch) -> bool:
    groups, candidates = batch.inputs.validate()
    return (
        groups == 1
        and candidates > BOOTSTRAP_POLICY_CANDIDATE_CHUNK
        and _has_bootstrap_policy_targets(batch)
    )


def _has_bootstrap_policy_targets(batch: R2MapSupervisedBatch) -> bool:
    return bool(np.asarray(batch.bootstrap_policy_mask).any())


def _bootstrap_policy_loss(
    model: R2MapModel,
    batch: R2MapSupervisedBatch,
) -> mx.array:
    if not _has_bootstrap_policy_targets(batch):
        return mx.array(0.0, dtype=mx.float32)
    if _requires_chunked_policy(batch):
        loss, _ = _chunked_bootstrap_policy_statistics(model, batch)
        return mx.array(loss, dtype=mx.float32)
    prediction = model.score_actions(batch.inputs)
    return _masked_group_mean(
        nn.losses.cross_entropy(
            prediction.bootstrap_policy_logits,
            batch.selected_action_index,
            reduction="none",
        ),
        batch.bootstrap_policy_mask,
    )


def _chunked_bootstrap_policy_statistics(
    model: R2MapModel,
    batch: R2MapSupervisedBatch,
) -> tuple[float, float]:
    groups, candidates = batch.inputs.validate()
    if groups != 1 or not bool(np.asarray(batch.bootstrap_policy_mask)[0]):
        raise ValueError("chunked bootstrap policy requires one supervised decision")
    selected = int(np.asarray(batch.selected_action_index)[0])
    log_normalizer = -math.inf
    selected_logit: float | None = None
    for start in range(0, candidates, BOOTSTRAP_POLICY_CANDIDATE_CHUNK):
        stop = min(start + BOOTSTRAP_POLICY_CANDIDATE_CHUNK, candidates)
        chunk = _slice_r2_map_batch_candidates(batch.inputs, start, stop)
        prediction = model.score_actions(chunk)
        masked = mx.where(
            chunk.candidate_mask,
            prediction.bootstrap_policy_logits,
            -mx.inf,
        )
        chunk_normalizer = mx.logsumexp(masked, axis=1)[0]
        mx.eval(chunk_normalizer, prediction.bootstrap_policy_logits)
        log_normalizer = float(np.logaddexp(log_normalizer, float(chunk_normalizer.item())))
        if start <= selected < stop:
            selected_logit = float(prediction.bootstrap_policy_logits[0, selected - start].item())
    if selected_logit is None or not math.isfinite(log_normalizer):
        raise ValueError("chunked bootstrap policy omitted the selected legal action")
    return log_normalizer - selected_logit, log_normalizer


def _chunked_bootstrap_policy_value_and_grad(
    model: R2MapModel,
    batch: R2MapSupervisedBatch,
) -> tuple[mx.array, Any]:
    loss_value, log_normalizer = _chunked_bootstrap_policy_statistics(model, batch)
    _, candidates = batch.inputs.validate()
    selected = int(np.asarray(batch.selected_action_index)[0])
    combined: dict[str, mx.array] | None = None
    for start in range(0, candidates, BOOTSTRAP_POLICY_CANDIDATE_CHUNK):
        stop = min(start + BOOTSTRAP_POLICY_CANDIDATE_CHUNK, candidates)
        chunk = _slice_r2_map_batch_candidates(batch.inputs, start, stop)
        local_selected = selected - start if start <= selected < stop else None

        def surrogate(
            candidate: R2MapModel,
            values: R2MapBatch,
            selected_in_chunk: int | None = local_selected,
        ) -> mx.array:
            prediction = candidate.score_actions(values)
            logits = prediction.bootstrap_policy_logits
            mass = mx.sum(
                mx.where(
                    values.candidate_mask,
                    mx.exp(logits - log_normalizer),
                    0.0,
                )
            )
            if selected_in_chunk is not None:
                mass = mass - logits[0, selected_in_chunk]
            return mass

        _, gradients = nn.value_and_grad(model, surrogate)(model, chunk)
        mx.eval(gradients)
        flattened = dict(tree_flatten(gradients))
        if combined is None:
            combined = flattened
        else:
            for name, gradient in flattened.items():
                combined[name] = combined[name] + gradient
            mx.eval(combined)
    if combined is None:
        raise ValueError("chunked bootstrap policy produced no candidate chunks")
    return mx.array(loss_value, dtype=mx.float32), tree_unflatten(list(combined.items()))


def _masked_cross_entropy(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    weights = mask.astype(mx.float32)
    return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), 1.0)


def _slice_r2_map_batch_candidates(batch: R2MapBatch, start: int, stop: int) -> R2MapBatch:
    _, candidates = batch.validate()
    if not 0 <= start < stop <= candidates:
        raise ValueError("R2-MAP candidate chunk bounds are invalid")

    def sliced(value: mx.array) -> mx.array:
        return value[:, start:stop]

    source = batch.candidates
    chunk = R2MapBatch(
        parent=batch.parent,
        candidates=R2MapPublicState(
            token_features=sliced(source.token_features),
            token_types=sliced(source.token_types),
            token_mask=sliced(source.token_mask),
            market_features=sliced(source.market_features),
            market_mask=sliced(source.market_mask),
            player_features=sliced(source.player_features),
            player_mask=sliced(source.player_mask),
            global_features=sliced(source.global_features),
        ),
        candidate_mask=sliced(batch.candidate_mask),
        action_features=sliced(batch.action_features),
        exact_afterstate_scores=sliced(batch.exact_afterstate_scores),
    )
    chunk.validate()
    return chunk


def _selected_r2_map_batch(batch: R2MapBatch, selected: mx.array) -> R2MapBatch:
    groups, _ = batch.validate()
    if tuple(selected.shape) != (groups,):
        raise ValueError("R2-MAP selected training index shape drifted")
    rows = mx.arange(groups)

    def gather(value: mx.array) -> mx.array:
        return value[rows, selected][:, None]

    candidates = batch.candidates
    return R2MapBatch(
        parent=batch.parent,
        candidates=R2MapPublicState(
            token_features=gather(candidates.token_features),
            token_types=gather(candidates.token_types),
            token_mask=gather(candidates.token_mask),
            market_features=gather(candidates.market_features),
            market_mask=gather(candidates.market_mask),
            player_features=gather(candidates.player_features),
            player_mask=gather(candidates.player_mask),
            global_features=gather(candidates.global_features),
        ),
        candidate_mask=mx.ones((groups, 1), dtype=mx.bool_),
        action_features=gather(batch.action_features),
        exact_afterstate_scores=gather(batch.exact_afterstate_scores),
    )


def _selected_candidate(values: mx.array, selected: mx.array) -> mx.array:
    if values.ndim < 3 or tuple(selected.shape) != (values.shape[0],):
        raise ValueError("R2-MAP selected-candidate auxiliary shape drifted")
    weights = mx.arange(values.shape[1])[None, :] == selected[:, None]
    for _ in values.shape[2:]:
        weights = weights[..., None]
    return mx.sum(values * weights, axis=1)


def _training_auxiliary_candidate(
    values: mx.array,
    selected: mx.array,
    *,
    preselected: bool,
) -> mx.array:
    if preselected:
        if values.ndim < 3 or values.shape[1] != 1 or values.shape[0] != selected.shape[0]:
            raise ValueError("R2-MAP preselected auxiliary shape drifted")
        return values[:, 0]
    return _selected_candidate(values, selected)


def _masked_group_mean(losses: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(mx.float32)
    return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), 1.0)


def _validate_adapter(config: R2MapTrainerConfig, adapter: R2MapTrainingAdapter) -> None:
    if adapter.protocol_id != config.adapter_protocol_id:
        raise ValueError("R2-MAP adapter protocol identity differs")
    if adapter.dataset_blake3 != config.dataset_blake3:
        raise ValueError("R2-MAP adapter dataset identity differs")
    if (
        getattr(adapter, "group_batch_size", None) != config.group_batch_size
        or getattr(adapter, "maximum_candidates_per_batch", None)
        != config.maximum_candidates_per_batch
    ):
        raise ValueError("R2-MAP adapter batch-packing contract differs")
    _adapter_dataset_contract(adapter)


def _adapter_dataset_contract(adapter: R2MapTrainingAdapter) -> dict[str, Any]:
    value = getattr(adapter, "dataset_contract", None)
    if not isinstance(value, dict) or value.get("dataset_blake3") != adapter.dataset_blake3:
        raise ValueError("R2-MAP adapter dataset contract is absent or inconsistent")
    return json.loads(json.dumps(value))


def _validate_resume_dataset_contract(
    adapter: R2MapTrainingAdapter, state: R2MapResumeState
) -> None:
    if _adapter_dataset_contract(adapter) != state.dataset_contract:
        raise CheckpointError("R2-MAP resume dataset contract differs")


def _learning_rate(config: R2MapTrainerConfig, step: int) -> float:
    if step < config.warmup_steps and config.warmup_steps:
        return config.learning_rate * float(step + 1) / config.warmup_steps
    progress = min(
        max((step - config.warmup_steps) / (config.schedule_steps - config.warmup_steps), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _scheduler_state(config: R2MapTrainerConfig, next_step: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema_id": SCHEDULER_SCHEMA,
        "next_step": next_step,
        "learning_rate": _learning_rate(config, next_step),
    }


def _capture_mlx_rng() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mlx_uint32_keys": [np.asarray(key).astype(np.uint32).tolist() for key in mx.random.state],
    }


def _restore_mlx_rng(value: dict[str, Any]) -> None:
    if set(value) != {"schema_version", "mlx_uint32_keys"} or value["schema_version"] != 1:
        raise CheckpointError("R2-MAP MLX RNG state schema differs")
    keys = value["mlx_uint32_keys"]
    if not isinstance(keys, list) or len(keys) != len(mx.random.state):
        raise CheckpointError("R2-MAP MLX RNG key count differs")
    for index, key in enumerate(keys):
        mx.random.state[index] = mx.array(key, dtype=mx.uint32)


def _validate_resume_branch(
    path: Path,
    binding: dict[str, Any],
    *,
    checkpoint_branch: str,
    requested_branch: str,
) -> None:
    content = path.read_bytes()
    offset = binding["offset_bytes"]
    records = _parse_loss_stream(content.decode())
    suffix_exists = len(content) != offset
    if checkpoint_branch == requested_branch and suffix_exists:
        raise CheckpointError(
            "R2-MAP same-branch resume would overwrite post-checkpoint loss history; fork a branch"
        )
    if checkpoint_branch != requested_branch and any(
        record["branch_id"] == requested_branch for record in records
    ):
        raise CheckpointError("R2-MAP requested resume branch already exists")


def _validate_resume_branch_bytes(
    content: bytes,
    binding: dict[str, Any],
    *,
    checkpoint_branch: str,
    requested_branch: str,
) -> None:
    try:
        records = _parse_loss_stream(content.decode())
    except UnicodeDecodeError as error:
        raise CheckpointError("R2-MAP loss stream is not UTF-8") from error
    offset = binding["offset_bytes"]
    if not isinstance(offset, int) or not 0 <= offset <= len(content):
        raise CheckpointError("R2-MAP checkpoint loss-stream offset is invalid")
    suffix_exists = len(content) != offset
    if checkpoint_branch == requested_branch and suffix_exists:
        raise CheckpointError(
            "R2-MAP same-branch resume would overwrite post-checkpoint loss history; fork a branch"
        )
    if checkpoint_branch != requested_branch and any(
        record["branch_id"] == requested_branch for record in records
    ):
        raise CheckpointError("R2-MAP requested resume branch already exists")


def _parse_loss_stream(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    if not text.endswith("\n"):
        raise CheckpointError("R2-MAP loss stream ends with an incomplete record")
    records: list[dict[str, Any]] = []
    branch_heads: dict[str, str] = {}
    branch_counts: dict[str, int] = {}
    known_hashes: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CheckpointError(f"invalid R2-MAP loss record {line_number}: {error}") from error
        claimed = record.pop("record_blake3", None)
        if claimed != _canonical_blake3(record):
            raise CheckpointError(f"R2-MAP loss record hash differs on line {line_number}")
        record["record_blake3"] = claimed
        branch = record.get("branch_id")
        if (
            record.get("schema_version") != 1
            or record.get("schema_id") != LOSS_STREAM_SCHEMA
            or record.get("stream_sequence") != len(records)
            or not isinstance(branch, str)
            or record.get("branch_sequence") != branch_counts.get(branch, 0)
        ):
            raise CheckpointError(f"R2-MAP loss record identity differs on line {line_number}")
        parent = record.get("parent_record_blake3")
        if branch in branch_heads:
            if parent != branch_heads[branch]:
                raise CheckpointError(f"R2-MAP loss branch chain differs on line {line_number}")
        elif parent is not None and parent not in known_hashes:
            raise CheckpointError(f"R2-MAP loss branch parent is unknown on line {line_number}")
        metrics = record.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or not all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in metrics.values()
            )
        ):
            raise CheckpointError(f"R2-MAP loss metrics are invalid on line {line_number}")
        branch_heads[branch] = claimed
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        known_hashes.add(claimed)
        records.append(record)
    return records


def _loss_contract_blake3() -> str:
    return _canonical_blake3(
        {
            "schema": LOSS_CONTRACT,
            "primary": "normalized-observed-draft-and-market-score-to-go-mse",
            "auxiliary_order": [
                "score-components",
                "bootstrap-policy",
                "opponent-next-action",
                "market-survival",
                "market-decision-policy",
            ],
        }
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(_canonical_json(value)).hexdigest()


def main() -> int:
    """Train or resume from compact replay windows; expanded streams are test-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--train-stream", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-stream", type=Path)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--panel-stream", type=Path)
    parser.add_argument("--allow-reference-expanded-streams", action="store_true")
    parser.add_argument("--maximum-reference-stream-bytes", type=int, default=1 << 30)
    parser.add_argument("--compact-index", type=Path)
    parser.add_argument("--compact-shard-root", type=Path)
    parser.add_argument("--compact-exporter", type=Path)
    parser.add_argument("--compact-window-dir", type=Path)
    parser.add_argument("--packed-pipe", action="store_true")
    parser.add_argument("--validated-aggregate-receipt", type=Path)
    parser.add_argument("--validated-packing-receipt", type=Path)
    parser.add_argument("--maximum-window-bytes", type=int, default=1 << 30)
    parser.add_argument("--maximum-prefetch-windows", type=int, choices=(0, 1), default=1)
    parser.add_argument("--fixed-panel-games", type=int, default=1)
    parser.add_argument("--production-game-projection", type=int, default=100_000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch-id", default="main")
    parser.add_argument("--source-blake3", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000,
        help="fixed-step recovery interval; the five-minute wall gate may checkpoint sooner",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=int,
        default=300,
        help="wall-clock recovery interval; must be no greater than five minutes",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=0,
        help=(
            "optional step validation interval; epoch boundaries and the final step always validate"
        ),
    )
    parser.add_argument(
        "--loss-event-every",
        type=int,
        choices=range(10, 26),
        default=20,
        help="append durable loss telemetry every 10-25 optimizer steps",
    )
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--maximum-candidates-per-batch", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--schedule-steps", type=int, default=1_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-pointer", default="last_verified")
    arguments = parser.parse_args()
    if (
        arguments.steps <= 0
        or arguments.checkpoint_every <= 0
        or not 1 <= arguments.checkpoint_seconds <= 300
        or arguments.validate_every < 0
    ):
        parser.error("steps/checkpoint-every must be positive and checkpoint-seconds <= 300")
    _require_local_john1_run_path(arguments.run_dir)

    from cascadia_mlx.r2_map_dataset import (
        R2MapCompactDatasetAdapter,
        R2MapDatasetAdapter,
        compact_storage_projection,
    )
    from cascadia_mlx.r2_map_pipe_dataset import R2MapPackedPipeDatasetAdapter
    from cascadia_mlx.r2_map_training_resources import (
        R2MapTrainingResourceMonitor,
        validate_training_resource_receipt,
    )
    from cascadia_mlx.r2_map_verify import verify_r2_map_checkpoint

    stream_values = (
        arguments.train_manifest,
        arguments.train_stream,
        arguments.validation_manifest,
        arguments.validation_stream,
        arguments.panel_manifest,
        arguments.panel_stream,
    )
    compact_required_values = (
        arguments.compact_index,
        arguments.compact_shard_root,
        arguments.compact_exporter,
        arguments.validated_aggregate_receipt,
        arguments.validated_packing_receipt,
    )
    projection = None
    if any(compact_required_values) or arguments.compact_window_dir:
        if not all(compact_required_values) or any(stream_values):
            parser.error("compact training requires every replay identity path and no streams")
        if arguments.packed_pipe:
            if arguments.compact_window_dir is not None:
                parser.error("packed-pipe training forbids an expanded window directory")
            adapter_context = R2MapPackedPipeDatasetAdapter(
                index=arguments.compact_index,
                shard_root=arguments.compact_shard_root,
                exporter=arguments.compact_exporter,
                validated_aggregate_receipt=arguments.validated_aggregate_receipt,
                validated_packing_receipt=arguments.validated_packing_receipt,
                group_batch_size=arguments.group_batch_size,
                maximum_candidates_per_batch=arguments.maximum_candidates_per_batch,
                sampler_seed=arguments.seed,
            )
            if not arguments.resume and arguments.steps != adapter_context.one_epoch_plan["steps"]:
                parser.error("initial packed-pipe training must run exactly one focal-seat epoch")
            if arguments.resume and arguments.steps > adapter_context.one_epoch_plan["steps"]:
                parser.error("packed-pipe resume cannot exceed one focal-seat epoch")
        else:
            if arguments.compact_window_dir is None:
                parser.error("legacy compact-window training requires a window directory")
            projection = compact_storage_projection(
                arguments.compact_index,
                target_games=arguments.production_game_projection,
                maximum_window_bytes=arguments.maximum_window_bytes,
                maximum_prefetch_windows=arguments.maximum_prefetch_windows,
            )
            if not projection.compact_fits_run_budget:
                parser.error("compact replay plus bounded windows exceeds the 40 GiB run budget")
            adapter_context = R2MapCompactDatasetAdapter(
                index=arguments.compact_index,
                shard_root=arguments.compact_shard_root,
                exporter=arguments.compact_exporter,
                window_root=arguments.compact_window_dir,
                validated_aggregate_receipt=arguments.validated_aggregate_receipt,
                validated_packing_receipt=arguments.validated_packing_receipt,
                group_batch_size=arguments.group_batch_size,
                maximum_candidates_per_batch=arguments.maximum_candidates_per_batch,
                maximum_window_bytes=arguments.maximum_window_bytes,
                maximum_prefetch_windows=arguments.maximum_prefetch_windows,
                fixed_panel_games=arguments.fixed_panel_games,
                require_ssd=True,
                sampler_seed=arguments.seed,
                packing_epochs=12,
            )
    else:
        if not all(stream_values):
            parser.error("training requires either compact replay paths or all six stream paths")
        if not arguments.allow_reference_expanded_streams:
            parser.error(
                "persistent expanded .r2map streams are reference/test-only; use compact replay"
            )
        expanded_bytes = sum(path.stat().st_size for path in stream_values if path is not None)
        if (
            arguments.maximum_reference_stream_bytes <= 0
            or expanded_bytes > arguments.maximum_reference_stream_bytes
        ):
            parser.error("reference expanded streams exceed the bounded 1-GiB default gate")
        adapter_context = R2MapDatasetAdapter.open(
            train_manifest=arguments.train_manifest,
            train_stream=arguments.train_stream,
            validation_manifest=arguments.validation_manifest,
            validation_stream=arguments.validation_stream,
            panel_manifest=arguments.panel_manifest,
            panel_stream=arguments.panel_stream,
            group_batch_size=arguments.group_batch_size,
            maximum_candidates_per_batch=arguments.maximum_candidates_per_batch,
        )

    resource_monitor = R2MapTrainingResourceMonitor.start()
    resource_monitor.sample()
    with adapter_context as adapter:
        config = R2MapTrainerConfig(
            run_dir=arguments.run_dir,
            run_id=arguments.run_id,
            branch_id=arguments.branch_id,
            source_blake3=arguments.source_blake3,
            dataset_blake3=adapter.dataset_blake3,
            adapter_protocol_id=adapter.protocol_id,
            group_batch_size=arguments.group_batch_size,
            maximum_candidates_per_batch=arguments.maximum_candidates_per_batch,
            learning_rate=arguments.learning_rate,
            minimum_learning_rate=arguments.minimum_learning_rate,
            warmup_steps=arguments.warmup_steps,
            schedule_steps=arguments.schedule_steps,
            loss_event_interval_steps=arguments.loss_event_every,
            seed=arguments.seed,
            auxiliary_loss_weights=(
                {
                    "components": 0.25,
                    "bootstrap_policy": 0.0,
                    "opponent_next_action": 0.05,
                    "market_survival": 0.05,
                    "market_decision_policy": 0.10,
                }
                if getattr(adapter, "bootstrap_value_only", False)
                else None
            ),
        )
        trainer = (
            R2MapTrainer.resume(config, adapter, pointer=arguments.resume_pointer)
            if arguments.resume
            else R2MapTrainer(config, adapter)
        )
        start_step = trainer.global_step
        last_validated_epoch = trainer.epoch
        last_checkpoint_monotonic = time.monotonic()
        checkpoints: list[dict[str, Any]] = []
        for _ in range(arguments.steps):
            trainer.step()
            if trainer.global_step % arguments.loss_event_every == 0:
                resource_monitor.sample()
            if (
                trainer.global_step % arguments.checkpoint_every == 0
                or time.monotonic() - last_checkpoint_monotonic >= arguments.checkpoint_seconds
                or trainer.global_step == start_step + arguments.steps
            ):
                resource_monitor.sample()
                final_step = trainer.global_step == start_step + arguments.steps
                validation_due = (
                    final_step
                    or trainer.epoch > last_validated_epoch
                    or (
                        arguments.validate_every > 0
                        and trainer.global_step % arguments.validate_every == 0
                    )
                )
                validation = trainer.validation_metrics() if validation_due else None
                if validation is not None:
                    last_validated_epoch = trainer.epoch
                checkpoint = trainer.save_checkpoint(validation=validation)
                verification = verify_r2_map_checkpoint(
                    checkpoint,
                    run_dir=arguments.run_dir,
                    adapter=adapter,
                    mark_last_verified=True,
                )
                checkpoints.append(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_manifest_blake3": verification["checkpoint_manifest_blake3"],
                        "verification_id": verification["verification_id"],
                        "validation": validation,
                    }
                )
                last_checkpoint_monotonic = time.monotonic()
        best_validation = select_best_validation_checkpoint(arguments.run_dir)
        resource_monitor.sample()
        result = {
            "schema_version": 1,
            "schema_id": "r2-map-training-command-receipt-v1",
            "run_id": arguments.run_id,
            "branch_id": arguments.branch_id,
            "dataset_blake3": adapter.dataset_blake3,
            "source_blake3": arguments.source_blake3,
            "start_step": start_step,
            "final_step": trainer.global_step,
            "examples_seen": trainer.examples_seen,
            "next_batch_identity": trainer.peek_next_batch_identity(),
            "checkpoints": checkpoints,
            "best_validation_checkpoint": best_validation.name,
            "storage_projection": None if projection is None else projection.to_dict(),
            "resource_receipt": validate_training_resource_receipt(resource_monitor.receipt()),
        }
        result["receipt_blake3"] = _canonical_blake3(result)
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _require_local_john1_run_path(path: Path) -> None:
    """Keep native MLX checkpoints on John1's primary campaign store."""

    require_local_storage_authority()
    required = CAMPAIGN_ROOT
    resolved_parent = path.parent.resolve(strict=True)
    resolved = resolved_parent / path.name
    try:
        resolved.relative_to(required.resolve(strict=True))
    except ValueError as error:
        raise ValueError(
            f"R2-MAP local run path must remain under john1 root {required}"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
