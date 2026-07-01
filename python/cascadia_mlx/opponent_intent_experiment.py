"""Matched O1 MLX learnability factorial with sealed policy-held-out testing."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import shutil
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
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
from cascadia_mlx.opponent_intent_dataset import (
    CombinedOpponentIntentDataset,
    OpponentIntentDataset,
    decode_opponent_intent_records,
)
from cascadia_mlx.opponent_intent_model import (
    ARMS,
    OpponentIntentModelConfig,
    OpponentIntentSurvivalModel,
    opponent_intent_loss,
    parameter_count,
    parameter_layout_blake3,
    parameter_tensor_blake3,
)

EXPERIMENT_ID = "o1-opponent-intent-mlx-factorial-v1"
ADR_ID = "0187"
PROTOCOL_ID = "o1-policy-heldout-matched-mlx-v1"
CORPUS_EXPERIMENT_ID = "o1-opponent-intent-policy-heldout-corpus-v1"
CORPUS_CLASSIFICATION = "policy_held_out_draft_survival_corpus_passed"

ROLES = tuple(
    f"{arm.split('-', 1)[0]}-{replica}" for replica in ("primary", "replay") for arm in ARMS
)
ROLE_ARMS = {
    role: next(arm for arm in ARMS if role.startswith(arm.split("-", 1)[0])) for role in ROLES
}
ARM_ROLES = {
    arm: (
        f"{arm.split('-', 1)[0]}-primary",
        f"{arm.split('-', 1)[0]}-replay",
    )
    for arm in ARMS
}

TRAINING_SEED = 2_026_061_704
TRAINING_STEPS = 5_120
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 640
METRIC_STEPS = 100
EVALUATION_BATCH_SIZE = 256
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_061_705
MLX_CACHE_LIMIT_BYTES = 1_073_741_824

MIN_VALIDATION_RELATIVE_BRIER_GAIN = 0.01
MAX_VALIDATION_NLL_REGRESSION = 0.005
MAX_VALIDATION_ECE_REGRESSION = 0.010
MIN_AUXILIARY_RELATIVE_NLL_GAIN = 0.02
MIN_TEST_RELATIVE_BRIER_GAIN = 0.005
MAX_TEST_NLL_REGRESSION = 0.005
MAX_TEST_ECE_REGRESSION = 0.015

HEAD_TARGETS = (
    ("tile_slot", "tile_slot_targets", 4),
    ("wildlife_slot", "wildlife_slot_targets", 4),
    ("draft_kind", "draft_kind_targets", 2),
    ("drafted_wildlife", "drafted_wildlife_targets", 5),
    ("replace_three", "replace_three_targets", 2),
)
RAW_ACTION_TARGET_FIELDS = {
    "tile_slot": "tile_slot",
    "wildlife_slot": "wildlife_slot",
    "draft_kind": "draft_kind",
    "drafted_wildlife": "drafted_wildlife",
    "replace_three": "replace_three_of_a_kind",
}


class OpponentIntentExperimentError(ValueError):
    """Raised when the frozen O1 factorial contract is violated."""


@dataclass(frozen=True)
class OpponentIntentProtocol:
    """Every scientific and optimization variable frozen before launch."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    training_steps: int = TRAINING_STEPS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    arms: tuple[str, ...] = ARMS
    primary_endpoint: str = "four-way-tile-disposition-multiclass-brier"
    primary_unit: str = "one-focal-decision-window"
    bootstrap_unit: str = "game"
    checkpoint_selection: str = "fixed-final-step"
    validation_during_training: bool = False
    test_open_rule: str = "eligible-noncontrol-validation-arm-selected"
    final_stress_role: str = "descriptive-only"
    policy_identity_observable: bool = False
    physical_tile_identity_observable: bool = False
    paid_wipe_claim_authorized: bool = False
    strategy_switch_claim_authorized: bool = False
    gameplay_claim_authorized: bool = False

    def validate(self) -> None:
        if self != OpponentIntentProtocol():
            raise OpponentIntentExperimentError("O1 protocol drifted")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["arms"] = list(self.arms)
        return value


@dataclass(frozen=True)
class OpponentIntentRunConfig:
    """One primary or crossed-host replay invocation."""

    train_datasets: tuple[Path, Path]
    validation_dataset: Path
    corpus_classification: Path
    authorization: Path
    run_dir: Path
    output: Path
    bundle_id: str
    role: str
    resume: bool = False
    smoke_steps: int | None = None
    protocol: OpponentIntentProtocol = field(default_factory=OpponentIntentProtocol)

    @property
    def production(self) -> bool:
        return self.smoke_steps is None

    @property
    def target_steps(self) -> int:
        return TRAINING_STEPS if self.production else int(self.smoke_steps)

    def validate(self) -> None:
        self.protocol.validate()
        if self.role not in ROLES:
            raise OpponentIntentExperimentError(f"unknown O1 factorial role: {self.role}")
        _require_digest(self.bundle_id, "bundle ID")
        if len(self.train_datasets) != 2:
            raise OpponentIntentExperimentError("O1 factorial requires both frozen train parts")
        if not self.production and (
            self.smoke_steps is None or not 1 <= self.smoke_steps <= 10 or self.resume
        ):
            raise OpponentIntentExperimentError(
                "bounded O1 smoke must be fresh and at most ten steps"
            )


def build_authorization(
    *,
    train_datasets: tuple[Path, Path],
    validation_dataset: Path,
    corpus_classification: Path,
    bundle_id: str,
) -> dict[str, Any]:
    """Bind corpus authorization, exact open roles, graph, and train priors."""
    _require_digest(bundle_id, "bundle ID")
    protocol = OpponentIntentProtocol()
    protocol.validate()
    corpus = _read_json(
        corpus_classification,
        "O1 corpus classification",
    )
    _validate_corpus_classification(corpus)
    train_parts = tuple(OpponentIntentDataset(path) for path in train_datasets)
    validation = OpponentIntentDataset(validation_dataset)
    if any(dataset.split != "train" for dataset in train_parts):
        raise OpponentIntentExperimentError("O1 authorization train roles are not train splits")
    if validation.split != "validation":
        raise OpponentIntentExperimentError("O1 authorization validation role is not validation")
    combined = CombinedOpponentIntentDataset(train_parts)
    priors = training_label_priors(combined)
    fingerprints = []
    for arm in ARMS:
        mx.random.seed(TRAINING_SEED)
        model = OpponentIntentSurvivalModel(OpponentIntentModelConfig(arm=arm))
        fingerprints.append(
            {
                "arm": arm,
                "parameter_count": parameter_count(model),
                "parameter_layout_blake3": parameter_layout_blake3(model),
                "initial_parameter_tensor_blake3": parameter_tensor_blake3(model),
            }
        )
    matched = {
        (
            value["parameter_count"],
            value["parameter_layout_blake3"],
            value["initial_parameter_tensor_blake3"],
        )
        for value in fingerprints
    }
    if len(matched) != 1:
        raise OpponentIntentExperimentError("O1 arms do not share one graph and initialization")
    count, layout, initial = next(iter(matched))
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol": protocol.to_dict(),
        "protocol_blake3": canonical_blake3(protocol.to_dict()),
        "bundle_id": bundle_id,
        "roles": list(ROLES),
        "role_arms": ROLE_ARMS,
        "corpus": {
            "experiment_id": corpus["experiment_id"],
            "classification": corpus["classification"],
            "classification_blake3": corpus["classification_blake3"],
            "matched_scientific_blake3": corpus["matched_scientific_blake3"],
            "authorization": corpus["authorization"],
        },
        "datasets": {
            "train": [dataset_identity(dataset) for dataset in train_parts],
            "validation": dataset_identity(validation),
        },
        "training_priors": priors,
        "training_priors_blake3": canonical_blake3(priors),
        "model": {
            "architecture": OpponentIntentModelConfig().architecture,
            "parameter_count": count,
            "parameter_layout_blake3": layout,
            "initial_parameter_tensor_blake3": initial,
            "arm_fingerprints": fingerprints,
        },
        "claim_boundary": {
            "validation_only_before_selection": True,
            "test_or_final_opened": False,
            "paid_wipe_intent": False,
            "strategy_switch": False,
            "champion_generalization": False,
            "gameplay_strength": False,
            "progress_to_100": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "approved": True,
        "authorization_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_authorization(
    path: Path,
    *,
    train_datasets: tuple[Path, Path],
    validation_dataset: Path,
    corpus_classification: Path,
    bundle_id: str,
    role: str,
) -> dict[str, Any]:
    observed = _read_json(path, "O1 factorial authorization")
    expected = build_authorization(
        train_datasets=train_datasets,
        validation_dataset=validation_dataset,
        corpus_classification=corpus_classification,
        bundle_id=bundle_id,
    )
    if observed != expected or role not in expected["identity"]["roles"]:
        raise OpponentIntentExperimentError(
            "O1 authorization does not match corpus, data, graph, or role"
        )
    return observed


def verify_authorization(
    *,
    path: Path,
    train_datasets: tuple[Path, Path],
    validation_dataset: Path,
    corpus_classification: Path,
    bundle_id: str,
    role: str,
) -> dict[str, Any]:
    """Rebuild authorization without creating an optimizer or run directory."""
    authorization = validate_authorization(
        path,
        train_datasets=train_datasets,
        validation_dataset=validation_dataset,
        corpus_classification=corpus_classification,
        bundle_id=bundle_id,
        role=role,
    )
    identity = authorization["identity"]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "role": role,
        "arm": ROLE_ARMS[role],
        "authorization_id": authorization["authorization_id"],
        "bundle_id": bundle_id,
        "protocol_blake3": identity["protocol_blake3"],
        "training_priors_blake3": identity["training_priors_blake3"],
        "initial_parameter_tensor_blake3": identity["model"]["initial_parameter_tensor_blake3"],
        "runtime": runtime_identity(),
        "run_directory_created": False,
        "optimizer_created": False,
        "passed": True,
    }


def training_label_priors(
    dataset: CombinedOpponentIntentDataset,
) -> dict[str, Any]:
    """Compute deterministic train-only frequency baselines."""
    disposition = np.ones((4, 4), dtype=np.float64)
    pair = np.ones((4, 2), dtype=np.float64)
    final_slot = np.ones((4, 4), dtype=np.float64)
    auxiliary = {
        name: np.ones((3, classes), dtype=np.float64) for name, _field, classes in HEAD_TARGETS
    }
    records_seen = 0
    for shard in dataset.shards:
        records = shard.records()
        for start in range(0, len(records), EVALUATION_BATCH_SIZE):
            batch = records[start : start + EVALUATION_BATCH_SIZE]
            survival = batch["survival_targets"]
            labels = survival["disposition"].astype(np.int64) - 1
            for slot in range(4):
                disposition[slot] += np.bincount(
                    labels[:, slot],
                    minlength=4,
                )
                survivors = labels[:, slot] == 3
                pair[slot] += np.bincount(
                    survival["pair_survives"][:, slot][survivors],
                    minlength=2,
                )
                final_slot[slot] += np.bincount(
                    survival["final_slot"][:, slot][survivors],
                    minlength=4,
                )
            actions = batch["opponent_targets"]["action"]
            for name, _batch_field, classes in HEAD_TARGETS:
                for opponent in range(3):
                    auxiliary[name][opponent] += np.bincount(
                        actions[RAW_ACTION_TARGET_FIELDS[name]][:, opponent],
                        minlength=classes,
                    )
            records_seen += len(batch)
    if records_seen != dataset.sample_count:
        raise OpponentIntentExperimentError("train-prior scan did not cover every O1 row")
    return {
        "records": records_seen,
        "laplace_pseudocount_per_class": 1,
        "disposition": _normalize_counts(disposition).tolist(),
        "pair_survival": _normalize_counts(pair).tolist(),
        "final_slot": _normalize_counts(final_slot).tolist(),
        "next_draft": {
            name: _normalize_counts(counts).tolist() for name, counts in auxiliary.items()
        },
    }


def run_experiment(
    config: OpponentIntentRunConfig,
) -> dict[str, Any]:
    """Train one frozen arm and evaluate every open-validation window."""
    config.validate()
    authorization = validate_authorization(
        config.authorization,
        train_datasets=config.train_datasets,
        validation_dataset=config.validation_dataset,
        corpus_classification=config.corpus_classification,
        bundle_id=config.bundle_id,
        role=config.role,
    )
    train_parts = tuple(OpponentIntentDataset(path) for path in config.train_datasets)
    train = CombinedOpponentIntentDataset(train_parts)
    validation = OpponentIntentDataset(config.validation_dataset)
    arm = ROLE_ARMS[config.role]
    model_config = OpponentIntentModelConfig(arm=arm)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    previous_cache_limit = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
    mx.set_default_device(mx.gpu)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(TRAINING_SEED)
    fresh_model = OpponentIntentSurvivalModel(model_config)
    initial_tensor = parameter_tensor_blake3(fresh_model)
    run_manifest = _run_manifest(
        config,
        authorization,
        model_config,
    )
    if config.resume:
        if _read_json(config.run_dir / "run.json", "run manifest") != run_manifest:
            raise OpponentIntentExperimentError("O1 resume manifest differs from the frozen run")
        model, optimizer, state, _checkpoint = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            model_factory=lambda values: OpponentIntentSurvivalModel(
                OpponentIntentModelConfig.from_dict(values)
            ),
        )
        if model.config != model_config:
            raise OpponentIntentExperimentError("O1 resume model configuration drifted")
    else:
        if (config.run_dir / "latest.json").exists():
            raise OpponentIntentExperimentError(
                "O1 run already contains checkpoints; pass --resume"
            )
        model = fresh_model
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    loss_and_grad = nn.value_and_grad(model, opponent_intent_loss)
    recent_losses: list[float] = []
    measured_examples = 0
    measured_seconds = 0.0
    metrics_path = config.run_dir / "metrics.jsonl"
    invocation_started = time.perf_counter()
    model.train()
    while state.global_step < config.target_steps:
        started = time.perf_counter()
        batch = train.deterministic_training_batch(
            step=state.global_step,
            seed=TRAINING_SEED,
            batch_size=BATCH_SIZE,
        )
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.perf_counter() - started
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise OpponentIntentExperimentError(
                f"O1 training produced nonfinite loss at step {state.global_step}"
            )
        batch_examples = int(batch.focal_turn.shape[0])
        state.global_step += 1
        state.epoch = state.global_step // train.batches_per_epoch
        state.batch_in_epoch = state.global_step % train.batches_per_epoch
        state.elapsed_seconds += elapsed
        measured_examples += batch_examples
        measured_seconds += elapsed
        recent_losses.append(loss_value)
        if state.global_step % METRIC_STEPS == 0:
            event = {
                "schema_version": 1,
                "role": config.role,
                "arm": arm,
                "global_step": state.global_step,
                "mean_loss": float(np.mean(recent_losses)),
                "examples_per_second": measured_examples / max(measured_seconds, 1e-12),
                "peak_active_memory_bytes": int(mx.get_peak_memory()),
            }
            _append_json(metrics_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            recent_losses.clear()
        if config.production and state.global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(config.run_dir, model, optimizer, state)
            prune_checkpoints(config.run_dir)

    checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
    prune_checkpoints(config.run_dir)
    final_model = config.run_dir / "final-model.safetensors"
    shutil.copy2(checkpoint / "model.safetensors", final_model)
    training_peak = int(mx.get_peak_memory())
    model.eval()
    validation_metrics, evidence = evaluate_model(
        model,
        validation,
        authorization["identity"]["training_priors"],
    )
    evidence_path = config.run_dir / "validation-evidence.npz"
    np.savez_compressed(evidence_path, **evidence)
    performance = benchmark_model(model, validation)
    total_training_examples = train.training_examples_for_steps(
        steps=state.global_step,
        seed=TRAINING_SEED,
        batch_size=BATCH_SIZE,
    )
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol_id": PROTOCOL_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "role": config.role,
        "arm": arm,
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "bundle_id": config.bundle_id,
        },
        "protocol": config.protocol.to_dict(),
        "corpus": authorization["identity"]["corpus"],
        "datasets": authorization["identity"]["datasets"],
        "training_priors_blake3": authorization["identity"]["training_priors_blake3"],
        "model": {
            "config": model.config.to_dict(),
            "parameter_count": parameter_count(model),
            "parameter_layout_blake3": parameter_layout_blake3(model),
            "initial_parameter_tensor_blake3": initial_tensor,
            "final_parameter_tensor_blake3": parameter_tensor_blake3(model),
            "final_model_file_blake3": _checksum(final_model),
        },
        "optimization": {
            "global_step": state.global_step,
            "training_examples": total_training_examples,
            "training_seconds": state.elapsed_seconds,
            "training_examples_per_second": total_training_examples
            / max(state.elapsed_seconds, 1e-12),
            "invocation_wall_seconds": time.perf_counter() - invocation_started,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        },
        "metrics": {"validation": validation_metrics},
        "evidence": {
            "validation_file": str(evidence_path.resolve()),
            "validation_file_blake3": _checksum(evidence_path),
            "validation_array_blake3": evidence_blake3(evidence),
        },
        "performance": {
            **performance,
            "training_peak_active_memory_bytes": training_peak,
            "peak_process_rss_bytes": peak_process_rss_bytes(),
        },
        "runtime": {
            **runtime_identity(),
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "integrity": {
            "same_parameter_graph_for_all_arms": True,
            "same_initialization_for_all_arms": True,
            "fixed_final_checkpoint": True,
            "validation_during_training": False,
            "policy_identity_observable": False,
            "physical_tile_identity_observable": False,
            "all_metrics_finite": _all_finite(validation_metrics),
            "test_or_final_data_opened": False,
        },
        "claims": {
            "offline_validation_complete": True,
            "paid_wipe_intent_measured": False,
            "strategy_switch_measured": False,
            "champion_generalization_measured": False,
            "gameplay_strength_measured": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = report_scientific_identity(report)
    report["report_id"] = canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "final-report.json", report)
    return report


def evaluate_model(
    model: OpponentIntentSurvivalModel,
    dataset: OpponentIntentDataset,
    priors: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate calibrated survival and authorized next-draft auxiliaries."""
    disposition_probabilities: list[np.ndarray] = []
    pair_probabilities: list[np.ndarray] = []
    final_slot_probabilities: list[np.ndarray] = []
    auxiliary_probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name, _field, _classes in HEAD_TARGETS
    }
    disposition_targets: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    final_slot_targets: list[np.ndarray] = []
    auxiliary_targets: dict[str, list[np.ndarray]] = {
        name: [] for name, _field, _classes in HEAD_TARGETS
    }
    games: list[np.ndarray] = []
    turns: list[np.ndarray] = []
    model.eval()
    for records in dataset.raw_batches(EVALUATION_BATCH_SIZE):
        batch = decode_opponent_intent_records(records)
        prediction = model(batch)
        arrays = [
            mx.softmax(prediction.disposition_logits, axis=-1),
            mx.softmax(prediction.pair_survival_logits, axis=-1),
            mx.softmax(prediction.final_slot_logits, axis=-1),
            mx.softmax(prediction.tile_slot_logits, axis=-1),
            mx.softmax(prediction.wildlife_slot_logits, axis=-1),
            mx.softmax(prediction.draft_kind_logits, axis=-1),
            mx.softmax(prediction.drafted_wildlife_logits, axis=-1),
            mx.softmax(prediction.replace_three_logits, axis=-1),
        ]
        mx.eval(*arrays)
        disposition_probabilities.append(np.asarray(arrays[0], dtype=np.float32))
        pair_probabilities.append(np.asarray(arrays[1], dtype=np.float32))
        final_slot_probabilities.append(np.asarray(arrays[2], dtype=np.float32))
        for index, (name, _field, _classes) in enumerate(HEAD_TARGETS):
            auxiliary_probabilities[name].append(np.asarray(arrays[index + 3], dtype=np.float32))
            auxiliary_targets[name].append(
                np.asarray(getattr(batch, f"{name}_targets"), dtype=np.int8)
            )
        disposition_targets.append(np.asarray(batch.disposition_targets, dtype=np.int8))
        pair_targets.append(np.asarray(batch.pair_survival_targets, dtype=np.int8))
        final_slot_targets.append(np.asarray(batch.final_slot_targets, dtype=np.int8))
        games.append(np.asarray(batch.game_index, dtype=np.int64))
        turns.append(np.asarray(batch.focal_turn, dtype=np.int8))
    if not disposition_probabilities:
        raise OpponentIntentExperimentError("O1 evaluation dataset is empty")
    probabilities = np.concatenate(disposition_probabilities)
    labels = np.concatenate(disposition_targets)
    pair_probs = np.concatenate(pair_probabilities)
    pair_labels = np.concatenate(pair_targets)
    final_probs = np.concatenate(final_slot_probabilities)
    final_labels = np.concatenate(final_slot_targets)
    game_values = np.concatenate(games)
    turn_values = np.concatenate(turns)
    auxiliary = {
        name: (
            np.concatenate(auxiliary_probabilities[name]),
            np.concatenate(auxiliary_targets[name]),
        )
        for name, _field, _classes in HEAD_TARGETS
    }
    primary = disposition_metrics(probabilities, labels)
    disposition_prior = np.asarray(
        priors["disposition"],
        dtype=np.float64,
    )
    prior_probabilities = np.broadcast_to(
        disposition_prior[None, :, :],
        probabilities.shape,
    )
    baseline = disposition_metrics(prior_probabilities, labels)
    survivor_mask = labels == 3
    pair_metrics = classification_metrics(
        pair_probs[survivor_mask],
        pair_labels[survivor_mask],
        classes=2,
    )
    pair_prior = np.asarray(priors["pair_survival"], dtype=np.float64)
    pair_prior_probabilities = np.broadcast_to(
        pair_prior[None, :, :],
        pair_probs.shape,
    )
    pair_baseline = classification_metrics(
        pair_prior_probabilities[survivor_mask],
        pair_labels[survivor_mask],
        classes=2,
    )
    final_metrics = classification_metrics(
        final_probs[survivor_mask],
        final_labels[survivor_mask],
        classes=4,
    )
    final_prior = np.asarray(priors["final_slot"], dtype=np.float64)
    final_prior_probabilities = np.broadcast_to(
        final_prior[None, :, :],
        final_probs.shape,
    )
    final_baseline = classification_metrics(
        final_prior_probabilities[survivor_mask],
        final_labels[survivor_mask],
        classes=4,
    )
    auxiliary_metrics = {}
    auxiliary_model_nll = []
    auxiliary_prior_nll = []
    for name, _field, classes in HEAD_TARGETS:
        head_probabilities, head_targets = auxiliary[name]
        metrics = classification_metrics(
            head_probabilities,
            head_targets,
            classes=classes,
        )
        prior = np.asarray(
            priors["next_draft"][name],
            dtype=np.float64,
        )
        prior_probabilities = np.broadcast_to(
            prior[None, :, :],
            head_probabilities.shape,
        )
        prior_metrics = classification_metrics(
            prior_probabilities,
            head_targets,
            classes=classes,
        )
        auxiliary_metrics[name] = {
            "model": metrics,
            "train_frequency_baseline": prior_metrics,
            "relative_nll_gain": (
                prior_metrics["negative_log_likelihood"] - metrics["negative_log_likelihood"]
            )
            / max(prior_metrics["negative_log_likelihood"], 1e-12),
        }
        auxiliary_model_nll.append(metrics["negative_log_likelihood"])
        auxiliary_prior_nll.append(prior_metrics["negative_log_likelihood"])
    phase = {}
    for name, lower, upper in (
        ("opening", 0, 19),
        ("early_middle", 19, 38),
        ("late_middle", 38, 57),
        ("endgame", 57, 76),
    ):
        selected = (turn_values >= lower) & (turn_values < upper)
        phase[name] = {
            "windows": int(np.sum(selected)),
            **disposition_metrics(
                probabilities[selected],
                labels[selected],
            ),
        }
    probe_rows = min(256, len(probabilities))
    probe_payload = np.concatenate(
        [
            probabilities[:probe_rows].reshape(-1),
            labels[:probe_rows].astype(np.float32).reshape(-1),
        ]
    ).astype("<f4", copy=False)
    metrics = {
        "windows": len(probabilities),
        "tile_labels": int(labels.size),
        "games": len(np.unique(game_values)),
        "disposition": primary,
        "pair_survival_among_survivors": pair_metrics,
        "final_slot_among_survivors": final_metrics,
        "next_draft": auxiliary_metrics,
        "mean_next_draft_negative_log_likelihood": float(np.mean(auxiliary_model_nll)),
        "mean_next_draft_train_frequency_nll": float(np.mean(auxiliary_prior_nll)),
        "mean_next_draft_relative_nll_gain": (
            float(np.mean(auxiliary_prior_nll)) - float(np.mean(auxiliary_model_nll))
        )
        / max(float(np.mean(auxiliary_prior_nll)), 1e-12),
        "train_frequency_baseline": {
            "disposition": baseline,
            "pair_survival_among_survivors": pair_baseline,
            "final_slot_among_survivors": final_baseline,
        },
        "phase": phase,
        "prediction_probe_rows": probe_rows,
        "prediction_probe_blake3": blake3.blake3(probe_payload.tobytes()).hexdigest(),
    }
    evidence = {
        "schema_version": np.asarray([1], dtype=np.int16),
        "game_index": game_values.astype("<i8", copy=False),
        "focal_turn": turn_values.astype("u1", copy=False),
        "disposition_probabilities": probabilities.astype("<f4", copy=False),
        "disposition_targets": labels.astype("u1", copy=False),
    }
    return metrics, evidence


def disposition_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(probabilities) == 0:
        return {
            "negative_log_likelihood": 0.0,
            "multiclass_brier": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "top_label_ece": 0.0,
            "survival_binary": {
                "brier": 0.0,
                "negative_log_likelihood": 0.0,
                "ece": 0.0,
                "auroc": 0.0,
            },
        }
    flat_probabilities = probabilities.reshape(-1, 4)
    flat_targets = targets.reshape(-1)
    primary = classification_metrics(
        flat_probabilities,
        flat_targets,
        classes=4,
    )
    survival_probabilities = probabilities[..., 3].reshape(-1)
    survival_targets = (targets == 3).astype(np.int8).reshape(-1)
    return {
        **primary,
        "survival_binary": binary_metrics(
            survival_probabilities,
            survival_targets,
        ),
    }


def classification_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    classes: int,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(
        -1,
        classes,
    )
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if len(targets) == 0:
        return {
            "negative_log_likelihood": 0.0,
            "multiclass_brier": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "top_label_ece": 0.0,
        }
    probabilities = probabilities / np.maximum(
        probabilities.sum(axis=-1, keepdims=True),
        1e-12,
    )
    clipped = np.clip(probabilities, 1e-8, 1.0)
    one_hot = np.eye(classes, dtype=np.float64)[targets]
    predictions = np.argmax(probabilities, axis=-1)
    f1_values = []
    for class_index in range(classes):
        predicted = predictions == class_index
        actual = targets == class_index
        true_positive = np.sum(predicted & actual)
        false_positive = np.sum(predicted & ~actual)
        false_negative = np.sum(~predicted & actual)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "negative_log_likelihood": float(
            -np.mean(np.log(clipped[np.arange(len(targets)), targets]))
        ),
        "multiclass_brier": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=-1))),
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(f1_values)),
        "top_label_ece": top_label_ece(
            probabilities,
            targets,
        ),
    }


def binary_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-8,
        1.0 - 1e-8,
    )
    targets = np.asarray(targets, dtype=np.int8)
    return {
        "brier": float(np.mean(np.square(probabilities - targets))),
        "negative_log_likelihood": float(
            np.mean(-targets * np.log(probabilities) - (1 - targets) * np.log(1 - probabilities))
        ),
        "ece": binary_ece(probabilities, targets),
        "auroc": binary_auroc(probabilities, targets),
    }


def top_label_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    confidence = np.max(probabilities, axis=-1)
    predictions = np.argmax(probabilities, axis=-1)
    correct = predictions == targets
    return _ece(confidence, correct.astype(np.float64), bins=bins)


def binary_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    confidence = np.where(
        probabilities >= 0.5,
        probabilities,
        1.0 - probabilities,
    )
    predictions = probabilities >= 0.5
    correct = predictions == targets.astype(bool)
    return _ece(confidence, correct.astype(np.float64), bins=bins)


def binary_auroc(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(
        probabilities,
        return_inverse=True,
        return_counts=True,
    )
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            selected = inverse == group
            ranks[selected] = np.mean(ranks[selected])
    positive_rank_sum = float(np.sum(ranks[targets == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def benchmark_model(
    model: OpponentIntentSurvivalModel,
    dataset: OpponentIntentDataset,
) -> dict[str, Any]:
    records = next(dataset.raw_batches(128))
    batch = decode_opponent_intent_records(records)
    for _ in range(5):
        prediction = model(batch)
        mx.eval(prediction.disposition_logits)
    mx.reset_peak_memory()
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        prediction = model(batch)
        mx.eval(prediction.disposition_logits)
    elapsed = time.perf_counter() - started
    return {
        "batch_windows": len(records),
        "steady_iterations": iterations,
        "steady_seconds": elapsed,
        "windows_per_second": len(records) * iterations / max(elapsed, 1e-12),
        "latency_milliseconds": elapsed / iterations * 1_000.0,
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def classify_reports(
    reports: dict[str, Path],
    evidence: dict[str, Path],
    models: dict[str, Path],
) -> dict[str, Any]:
    """Require exact replays, then select only a calibrated treatment arm."""
    if set(reports) != set(ROLES) or set(evidence) != set(ROLES) or set(models) != set(ROLES):
        raise OpponentIntentExperimentError(
            "O1 classification requires all eight reports, evidence files, and models"
        )
    loaded = {role: _read_json(path, f"{role} report") for role, path in reports.items()}
    loaded_evidence = {}
    artifact_evidence = {}
    for role in ROLES:
        report = loaded[role]
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or report.get("protocol_id") != PROTOCOL_ID
            or report.get("role") != role
            or report.get("arm") != ROLE_ARMS[role]
            or report.get("report_id") != canonical_blake3(report.get("scientific_identity"))
            or report.get("integrity", {}).get("all_metrics_finite") is not True
        ):
            raise OpponentIntentExperimentError(f"invalid O1 report for {role}")
        arrays = load_evidence(evidence[role])
        loaded_evidence[role] = arrays
        artifact_evidence[role] = {
            "model_matches_report": _checksum(models[role])
            == report["model"]["final_model_file_blake3"],
            "evidence_file_matches_report": _checksum(evidence[role])
            == report["evidence"]["validation_file_blake3"],
            "evidence_arrays_match_report": evidence_blake3(arrays)
            == report["evidence"]["validation_array_blake3"],
        }
    common_fields = (
        ("authorization", "authorization_id"),
        ("authorization", "bundle_id"),
        ("datasets", "train"),
        ("datasets", "validation"),
        ("training_priors_blake3", None),
        ("model", "parameter_count"),
        ("model", "parameter_layout_blake3"),
        ("model", "initial_parameter_tensor_blake3"),
    )
    parity = {}
    for section, field_name in common_fields:
        values = []
        for report in loaded.values():
            value = report[section] if field_name is None else report[section][field_name]
            values.append(canonical_blake3(value) if isinstance(value, (dict, list)) else value)
        parity[section if field_name is None else f"{section}.{field_name}"] = len(set(values)) == 1
    replay_parity = {}
    for arm, (primary_role, replay_role) in ARM_ROLES.items():
        primary = loaded[primary_role]
        replay = loaded[replay_role]
        replay_parity[arm] = {
            "final_parameter_tensor_exact": primary["model"]["final_parameter_tensor_blake3"]
            == replay["model"]["final_parameter_tensor_blake3"],
            "final_model_file_exact": primary["model"]["final_model_file_blake3"]
            == replay["model"]["final_model_file_blake3"],
            "prediction_evidence_exact": primary["evidence"]["validation_array_blake3"]
            == replay["evidence"]["validation_array_blake3"],
            "scientific_identity_exact": _role_neutral_scientific_identity(primary)
            == _role_neutral_scientific_identity(replay),
        }
    integrity_pass = (
        all(parity.values())
        and all(all(values.values()) for values in replay_parity.values())
        and all(all(values.values()) for values in artifact_evidence.values())
    )
    control_role = ARM_ROLES[ARMS[0]][0]
    control = loaded[control_role]["metrics"]["validation"]
    control_evidence = loaded_evidence[control_role]
    treatments = {}
    eligible = []
    for arm in ARMS[1:]:
        role = ARM_ROLES[arm][0]
        metrics = loaded[role]["metrics"]["validation"]
        comparison = paired_game_bootstrap(
            control_evidence,
            loaded_evidence[role],
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        control_brier = control["disposition"]["multiclass_brier"]
        treatment_brier = metrics["disposition"]["multiclass_brier"]
        gates = {
            "relative_brier_gain": treatment_brier
            <= control_brier * (1.0 - MIN_VALIDATION_RELATIVE_BRIER_GAIN),
            "game_clustered_ci_excludes_zero": comparison["confidence_95"][1] < 0.0,
            "negative_log_likelihood_noninferior": metrics["disposition"]["negative_log_likelihood"]
            <= control["disposition"]["negative_log_likelihood"] + MAX_VALIDATION_NLL_REGRESSION,
            "top_label_calibration_noninferior": metrics["disposition"]["top_label_ece"]
            <= control["disposition"]["top_label_ece"] + MAX_VALIDATION_ECE_REGRESSION,
            "survival_binary_brier_noninferior": metrics["disposition"]["survival_binary"]["brier"]
            <= control["disposition"]["survival_binary"]["brier"],
            "next_draft_auxiliary_learned_when_enabled": (
                arm == ARMS[1]
                or metrics["mean_next_draft_relative_nll_gain"] >= MIN_AUXILIARY_RELATIVE_NLL_GAIN
            ),
        }
        arm_eligible = integrity_pass and all(gates.values())
        if arm_eligible:
            eligible.append(arm)
        treatments[arm] = {
            "eligible": arm_eligible,
            "gates": gates,
            "metrics": metrics,
            "paired_game_bootstrap": comparison,
            "relative_brier_gain": (control_brier - treatment_brier) / max(control_brier, 1e-12),
        }
    selected = (
        min(
            eligible,
            key=lambda arm: (
                treatments[arm]["metrics"]["disposition"]["multiclass_brier"],
                treatments[arm]["metrics"]["disposition"]["negative_log_likelihood"],
                treatments[arm]["metrics"]["disposition"]["top_label_ece"],
                arm,
            ),
        )
        if eligible
        else None
    )
    classification = (
        "opponent_intent_validation_arm_selected"
        if selected is not None
        else "opponent_intent_validation_factorial_null"
    )
    scientific = {
        "classification": classification,
        "selected_arm": selected,
        "selected_primary_role": (ARM_ROLES[selected][0] if selected is not None else None),
        "integrity_pass": integrity_pass,
        "parity": parity,
        "replay_parity": replay_parity,
        "artifact_evidence": artifact_evidence,
        "control": control,
        "treatments": treatments,
        "claim_boundary": {
            "validation_complete": True,
            "sealed_test_authorized": selected is not None,
            "test_or_final_opened": False,
            "gameplay_strength_measured": False,
            "progress_to_100_claimed": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "classification_id": canonical_blake3(scientific),
        "scientific": scientific,
    }


def evaluate_selected(
    *,
    classification_path: Path,
    reports: dict[str, Path],
    models: dict[str, Path],
    authorization_path: Path,
    test_dataset: Path,
    final_stress_dataset: Path,
) -> dict[str, Any]:
    """Open test exactly once after validation selects a noncontrol arm."""
    classification = _read_json(
        classification_path,
        "O1 validation classification",
    )
    scientific = classification.get("scientific", {})
    selected = scientific.get("selected_arm")
    if selected is None:
        return {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "adr": ADR_ID,
            "classification": "opponent_intent_test_not_opened",
            "validation_classification_id": classification["classification_id"],
            "test_or_final_opened": False,
            "reason": "validation selected no eligible treatment arm",
        }
    if selected not in ARMS[1:]:
        raise OpponentIntentExperimentError("sealed-test selection is not a treatment arm")
    control_role = ARM_ROLES[ARMS[0]][0]
    selected_role = ARM_ROLES[selected][0]
    required_roles = {control_role, selected_role}
    if not required_roles.issubset(reports) or not required_roles.issubset(models):
        raise OpponentIntentExperimentError(
            "sealed test requires control and selected primary artifacts"
        )
    loaded_reports = {role: _read_json(reports[role], f"{role} report") for role in required_roles}
    authorization = _read_json(
        authorization_path,
        "O1 factorial authorization",
    )
    if any(
        report["authorization"]["authorization_id"] != authorization.get("authorization_id")
        for report in loaded_reports.values()
    ):
        raise OpponentIntentExperimentError(
            "sealed-test models do not share the supplied authorization"
        )
    control_model = load_final_model(
        loaded_reports[control_role],
        models[control_role],
    )
    selected_model = load_final_model(
        loaded_reports[selected_role],
        models[selected_role],
    )
    priors = authorization["identity"]["training_priors"]
    test = OpponentIntentDataset(test_dataset)
    if test.split != "test":
        raise OpponentIntentExperimentError("sealed O1 test path is not a test split")
    control_test, control_evidence = evaluate_model(
        control_model,
        test,
        priors,
    )
    selected_test, selected_evidence = evaluate_model(
        selected_model,
        test,
        priors,
    )
    comparison = paired_game_bootstrap(
        control_evidence,
        selected_evidence,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED + 1,
    )
    control_brier = control_test["disposition"]["multiclass_brier"]
    selected_brier = selected_test["disposition"]["multiclass_brier"]
    gates = {
        "relative_brier_gain": selected_brier
        <= control_brier * (1.0 - MIN_TEST_RELATIVE_BRIER_GAIN),
        "game_clustered_ci_excludes_zero": comparison["confidence_95"][1] < 0.0,
        "negative_log_likelihood_noninferior": selected_test["disposition"][
            "negative_log_likelihood"
        ]
        <= control_test["disposition"]["negative_log_likelihood"] + MAX_TEST_NLL_REGRESSION,
        "top_label_calibration_noninferior": selected_test["disposition"]["top_label_ece"]
        <= control_test["disposition"]["top_label_ece"] + MAX_TEST_ECE_REGRESSION,
        "survival_binary_brier_noninferior": selected_test["disposition"]["survival_binary"][
            "brier"
        ]
        <= control_test["disposition"]["survival_binary"]["brier"],
    }
    passed = all(gates.values())
    final_stress = OpponentIntentDataset(final_stress_dataset)
    if final_stress.split != "final":
        raise OpponentIntentExperimentError("O1 final-stress path is not a final split")
    control_stress, _control_stress_evidence = evaluate_model(
        control_model,
        final_stress,
        priors,
    )
    selected_stress, _selected_stress_evidence = evaluate_model(
        selected_model,
        final_stress,
        priors,
    )
    terminal = (
        "opponent_intent_policy_holdout_replication_passed"
        if passed
        else "opponent_intent_policy_holdout_replication_failed"
    )
    terminal_scientific = {
        "classification": terminal,
        "selected_arm": selected,
        "validation_classification_id": classification["classification_id"],
        "test": {
            "control": control_test,
            "selected": selected_test,
            "paired_game_bootstrap": comparison,
            "relative_brier_gain": (control_brier - selected_brier) / max(control_brier, 1e-12),
            "gates": gates,
        },
        "final_stress_descriptive": {
            "control": control_stress,
            "selected": selected_stress,
        },
        "claim_boundary": {
            "policy_held_out_offline_replication": passed,
            "high_regret_ranking_integration_authorized": passed,
            "paid_wipe_intent": False,
            "strategy_switch": False,
            "champion_generalization": False,
            "gameplay_strength_measured": False,
            "progress_to_100_claimed": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "classification_id": canonical_blake3(terminal_scientific),
        "scientific": terminal_scientific,
    }


def paired_game_bootstrap(
    control: dict[str, np.ndarray],
    treatment: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired primary-endpoint deltas at the game level."""
    for key in ("game_index", "focal_turn", "disposition_targets"):
        if not np.array_equal(control[key], treatment[key]):
            raise OpponentIntentExperimentError(f"paired O1 evidence disagrees on {key}")
    control_window = per_window_brier(
        control["disposition_probabilities"],
        control["disposition_targets"],
    )
    treatment_window = per_window_brier(
        treatment["disposition_probabilities"],
        treatment["disposition_targets"],
    )
    game_indices = control["game_index"]
    games = np.unique(game_indices)
    deltas = np.asarray(
        [
            np.mean(treatment_window[game_indices == game] - control_window[game_indices == game])
            for game in games
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(deltas),
        size=(samples, len(deltas)),
    )
    bootstrap = np.mean(deltas[draws], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "games": len(games),
        "samples": samples,
        "seed": seed,
        "mean_delta_treatment_minus_control": float(np.mean(deltas)),
        "confidence_95": [float(low), float(high)],
        "fraction_better_games": float(np.mean(deltas < 0.0)),
    }


def per_window_brier(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    one_hot = np.eye(4, dtype=np.float64)[np.asarray(targets, dtype=np.int64)]
    return np.mean(
        np.sum(
            np.square(np.asarray(probabilities, dtype=np.float64) - one_hot),
            axis=-1,
        ),
        axis=-1,
    )


def evidence_blake3(evidence: dict[str, np.ndarray]) -> str:
    digest = blake3.blake3()
    for name in sorted(evidence):
        array = np.ascontiguousarray(evidence[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_evidence(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise OpponentIntentExperimentError(
            f"cannot read O1 prediction evidence: {error}"
        ) from error
    required = {
        "schema_version",
        "game_index",
        "focal_turn",
        "disposition_probabilities",
        "disposition_targets",
    }
    if set(result) != required or not np.array_equal(
        result["schema_version"],
        np.asarray([1], dtype=np.int16),
    ):
        raise OpponentIntentExperimentError("O1 prediction evidence schema drifted")
    return result


def load_final_model(
    report: dict[str, Any],
    path: Path,
) -> OpponentIntentSurvivalModel:
    if _checksum(path) != report["model"]["final_model_file_blake3"]:
        raise OpponentIntentExperimentError("O1 final model file does not match its report")
    model = OpponentIntentSurvivalModel(
        OpponentIntentModelConfig.from_dict(report["model"]["config"])
    )
    model.load_weights(str(path))
    mx.eval(model.parameters())
    if parameter_tensor_blake3(model) != report["model"]["final_parameter_tensor_blake3"]:
        raise OpponentIntentExperimentError("loaded O1 final model tensor hash drifted")
    model.eval()
    return model


def dataset_identity(
    dataset: OpponentIntentDataset,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "split": dataset.split,
        "manifest_blake3": _checksum(dataset.root / "dataset.json"),
        "games": int(dataset.manifest["completed_games"]),
        "records": dataset.sample_count,
        "cohort": dataset.manifest["cohort"],
        "shard_set_blake3": canonical_blake3(
            [
                {
                    "file": item["file"],
                    "blake3": item["blake3"],
                    "byte_count": item["byte_count"],
                }
                for item in dataset.manifest["shards"]
            ]
        ),
    }


def report_scientific_identity(
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": report["experiment_id"],
        "adr": report["adr"],
        "protocol_id": report["protocol_id"],
        "mode": report["mode"],
        "role": report["role"],
        "arm": report["arm"],
        "authorization": report["authorization"],
        "protocol": report["protocol"],
        "corpus": report["corpus"],
        "datasets": report["datasets"],
        "training_priors_blake3": report["training_priors_blake3"],
        "model": report["model"],
        "optimization": {
            "global_step": report["optimization"]["global_step"],
            "training_examples": report["optimization"]["training_examples"],
        },
        "metrics": report["metrics"],
        "evidence": {
            "validation_array_blake3": report["evidence"]["validation_array_blake3"],
        },
        "integrity": report["integrity"],
        "claims": report["claims"],
    }


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def runtime_identity() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "device": str(mx.default_device()),
    }


def peak_process_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _run_manifest(
    config: OpponentIntentRunConfig,
    authorization: dict[str, Any],
    model_config: OpponentIntentModelConfig,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": EXPERIMENT_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "role": config.role,
        "arm": ROLE_ARMS[config.role],
        "target_steps": config.target_steps,
        "authorization_id": authorization["authorization_id"],
        "bundle_id": config.bundle_id,
        "protocol": config.protocol.to_dict(),
        "datasets": authorization["identity"]["datasets"],
        "model": model_config.to_dict(),
        "runtime_contract": {
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "device": str(mx.default_device()),
        },
    }


def _validate_corpus_classification(
    corpus: dict[str, Any],
) -> None:
    authorization = corpus.get("authorization", {})
    if (
        corpus.get("experiment_id") != CORPUS_EXPERIMENT_ID
        or corpus.get("classification") != CORPUS_CLASSIFICATION
        or authorization.get("public_state_control_training") is not True
        or authorization.get("recent_history_intent_training") is not True
        or authorization.get("next_draft_auxiliary_training") is not True
        or authorization.get("market_survival_training") is not True
        or authorization.get("policy_held_out_calibration") is not True
        or authorization.get("paid_wipe_intent_training") is not False
        or authorization.get("strategy_switch_training") is not False
        or authorization.get("gameplay_promotion") is not False
    ):
        raise OpponentIntentExperimentError(
            "O1 corpus classification does not authorize this factorial"
        )


def _role_neutral_scientific_identity(
    report: dict[str, Any],
) -> dict[str, Any]:
    identity = json.loads(json.dumps(report["scientific_identity"]))
    identity["role"] = report["arm"]
    return identity


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    return counts / np.sum(counts, axis=-1, keepdims=True)


def _ece(
    confidence: np.ndarray,
    accuracy: np.ndarray,
    *,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    value = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(np.sum(selected))
        if count:
            value += (
                count
                / total
                * abs(float(np.mean(confidence[selected])) - float(np.mean(accuracy[selected])))
            )
    return float(value)


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise OpponentIntentExperimentError(f"{label} is not a lowercase BLAKE3 digest")


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OpponentIntentExperimentError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise OpponentIntentExperimentError(f"{label} must be an object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return False


def _parse_role_paths(
    values: list[str],
    label: str,
) -> dict[str, Path]:
    result = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role not in ROLES or role in result:
            raise OpponentIntentExperimentError(f"invalid {label} role path: {value}")
        result[role] = Path(path)
    return result


def _add_open_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-dataset",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument(
        "--corpus-classification",
        type=Path,
        required=True,
    )
    parser.add_argument("--bundle-id", required=True)


def _train_tuple(values: list[Path]) -> tuple[Path, Path]:
    if len(values) != 2:
        raise OpponentIntentExperimentError("exactly two --train-dataset paths are required")
    return values[0], values[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    _add_open_data_arguments(authorize)
    authorize.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-authorization")
    _add_open_data_arguments(verify)
    verify.add_argument("--authorization", type=Path, required=True)
    verify.add_argument("--role", choices=ROLES, required=True)
    verify.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    _add_open_data_arguments(run)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--role", choices=ROLES, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke-steps", type=int)

    classify = subparsers.add_parser("classify")
    classify.add_argument(
        "--report",
        action="append",
        required=True,
    )
    classify.add_argument(
        "--evidence",
        action="append",
        required=True,
    )
    classify.add_argument(
        "--model",
        action="append",
        required=True,
    )
    classify.add_argument("--output", type=Path, required=True)

    selected = subparsers.add_parser("evaluate-selected")
    selected.add_argument("--classification", type=Path, required=True)
    selected.add_argument(
        "--report",
        action="append",
        required=True,
    )
    selected.add_argument(
        "--model",
        action="append",
        required=True,
    )
    selected.add_argument("--authorization", type=Path, required=True)
    selected.add_argument("--test-dataset", type=Path, required=True)
    selected.add_argument(
        "--final-stress-dataset",
        type=Path,
        required=True,
    )
    selected.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "authorize":
        result = build_authorization(
            train_datasets=_train_tuple(args.train_dataset),
            validation_dataset=args.validation_dataset,
            corpus_classification=args.corpus_classification,
            bundle_id=args.bundle_id,
        )
    elif args.command == "verify-authorization":
        result = verify_authorization(
            path=args.authorization,
            train_datasets=_train_tuple(args.train_dataset),
            validation_dataset=args.validation_dataset,
            corpus_classification=args.corpus_classification,
            bundle_id=args.bundle_id,
            role=args.role,
        )
    elif args.command == "run":
        result = run_experiment(
            OpponentIntentRunConfig(
                train_datasets=_train_tuple(args.train_dataset),
                validation_dataset=args.validation_dataset,
                corpus_classification=args.corpus_classification,
                authorization=args.authorization,
                run_dir=args.run_dir,
                output=args.output,
                bundle_id=args.bundle_id,
                role=args.role,
                resume=args.resume,
                smoke_steps=args.smoke_steps,
            )
        )
    elif args.command == "classify":
        result = classify_reports(
            _parse_role_paths(args.report, "report"),
            _parse_role_paths(args.evidence, "evidence"),
            _parse_role_paths(args.model, "model"),
        )
    else:
        reports = _parse_role_paths(args.report, "report")
        result = evaluate_selected(
            classification_path=args.classification,
            reports=reports,
            models=_parse_role_paths(args.model, "model"),
            authorization_path=args.authorization,
            test_dataset=args.test_dataset,
            final_stress_dataset=args.final_stress_dataset,
        )
    _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
