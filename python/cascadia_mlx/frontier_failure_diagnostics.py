"""Independent failure diagnostics for the rejected frontier-anchored ranker."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import socket
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GradedOracleBatch,
    GradedOracleDataset,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
    evaluate_frontier_anchored,
    frontier_anchored_loss_components,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

EXPERIMENT_ID = "complete-action-frontier-failure-diagnostics-v1"
EXPECTED_SOURCE_EXPERIMENT = "complete-action-frontier-anchored-set-ranker-v1"
EXPECTED_CHECKPOINT = "step-000003592-epoch-0008-batch-000000"
TRAIN_FIT_TARGET_RECALL_GATE = 0.80
TRAIN_FIT_EXACT_SET_GATE = 0.25
GENERALIZATION_GAP_GATE = 0.10
COLLISION_POSITIVE_MASS_GATE = 0.01
GRADIENT_DOMINATION_RATIO = 0.50
GRADIENT_CONFLICT_COSINE = -0.25
ERROR_CONCENTRATION_MISS_SHARE = 0.35
ERROR_CONCENTRATION_RECALL_GAP = 0.10
ERROR_CONCENTRATION_MINIMUM_POSITIVES = 50
OBJECTIVE_GRADIENT_GROUPS = 8
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
    "john4": "john4",
}
_PHASE_NAMES = {0: "early", 1: "middle", 2: "late"}
_WILDLIFE_NAMES = ("bear", "elk", "salmon", "hawk", "fox")
_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


@dataclass
class _BinarySlice:
    positives: int = 0
    recalled: int = 0

    def add(self, positive: bool, recalled: bool) -> None:
        if positive:
            self.positives += 1
            self.recalled += int(recalled)

    def report(self, total_misses: int) -> dict[str, float | int]:
        misses = self.positives - self.recalled
        return {
            "target_positives": self.positives,
            "recalled": self.recalled,
            "misses": misses,
            "recall": self.recalled / max(self.positives, 1),
            "miss_share": misses / max(total_misses, 1),
        }


def classify_train_fit(
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Classify optimization/capacity underfit versus generalization."""
    train_recall = float(train_metrics["target_positive_recall"])
    train_exact = float(train_metrics["target_set_exact_fraction"])
    validation_recall = float(validation_metrics["target_positive_recall"])
    underfit = (
        train_recall < TRAIN_FIT_TARGET_RECALL_GATE
        or train_exact < TRAIN_FIT_EXACT_SET_GATE
    )
    generalization_gap = train_recall - validation_recall
    generalization = not underfit and generalization_gap >= GENERALIZATION_GAP_GATE
    if underfit:
        primary = "optimization_or_capacity_underfit"
    elif generalization:
        primary = "generalization_failure"
    else:
        primary = "fit_not_primary"
    return {
        "primary": primary,
        "optimization_or_capacity_underfit": underfit,
        "generalization_failure": generalization,
        "train_target_positive_recall_gate": TRAIN_FIT_TARGET_RECALL_GATE,
        "train_target_set_exact_fraction_gate": TRAIN_FIT_EXACT_SET_GATE,
        "generalization_gap_gate": GENERALIZATION_GAP_GATE,
        "observed_generalization_gap": generalization_gap,
    }


def classify_collision(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify exact model-visible target contradictions."""
    positive_mass = float(metrics["conflicting_target_positive_fraction"])
    material = positive_mass >= COLLISION_POSITIVE_MASS_GATE
    return {
        "primary": (
            "exact_observable_collision_material"
            if material
            else "exact_observable_collision_not_material"
        ),
        "exact_observable_collision_material": material,
        "positive_mass_gate": COLLISION_POSITIVE_MASS_GATE,
        "observed_conflicting_target_positive_fraction": positive_mass,
    }


def classify_objective_gradient(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify auxiliary domination or direct gradient conflict."""
    target_norm = float(metrics["weighted_gradient_norms"]["target_set_cross_entropy"])
    auxiliary_norm = float(metrics["weighted_auxiliary_gradient_norm"])
    cosine = float(metrics["target_listwise_gradient_cosine"])
    dominated = target_norm < GRADIENT_DOMINATION_RATIO * auxiliary_norm
    conflict = (
        cosine <= GRADIENT_CONFLICT_COSINE
        and auxiliary_norm >= GRADIENT_DOMINATION_RATIO * target_norm
    )
    if conflict:
        primary = "objective_gradient_conflict"
    elif dominated:
        primary = "target_objective_gradient_dominated"
    else:
        primary = "objective_gradient_pressure_not_primary"
    return {
        "primary": primary,
        "objective_gradient_conflict": conflict,
        "target_objective_gradient_dominated": dominated,
        "domination_ratio_gate": GRADIENT_DOMINATION_RATIO,
        "conflict_cosine_gate": GRADIENT_CONFLICT_COSINE,
    }


def classify_error_anatomy(
    overall_recall: float,
    slices: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    """Identify whether misses concentrate in an actionable observable slice."""
    concentrated: list[dict[str, Any]] = []
    for dimension, values in slices.items():
        for name, metrics in values.items():
            positives = int(metrics["target_positives"])
            recall = float(metrics["recall"])
            miss_share = float(metrics["miss_share"])
            if (
                positives >= ERROR_CONCENTRATION_MINIMUM_POSITIVES
                and miss_share >= ERROR_CONCENTRATION_MISS_SHARE
                and recall <= overall_recall - ERROR_CONCENTRATION_RECALL_GAP
            ):
                concentrated.append(
                    {
                        "dimension": dimension,
                        "slice": name,
                        **metrics,
                        "recall_gap": overall_recall - recall,
                    }
                )
    concentrated.sort(
        key=lambda item: (
            -float(item["miss_share"]),
            -float(item["recall_gap"]),
            str(item["dimension"]),
            str(item["slice"]),
        )
    )
    return {
        "primary": (
            "error_concentration_material"
            if concentrated
            else "errors_broadly_distributed"
        ),
        "error_concentration_material": bool(concentrated),
        "concentrated_slices": concentrated,
        "miss_share_gate": ERROR_CONCENTRATION_MISS_SHARE,
        "recall_gap_gate": ERROR_CONCENTRATION_RECALL_GAP,
        "minimum_target_positives": ERROR_CONCENTRATION_MINIMUM_POSITIVES,
    }


def select_failure_mechanism(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered evidence priority to four diagnostics."""
    train = reports["train-fit"]["classification"]
    collision = reports["observable-collision"]["classification"]
    gradient = reports["objective-gradient"]["classification"]
    anatomy = reports["error-anatomy"]["classification"]
    if collision["exact_observable_collision_material"]:
        mechanism = "representation_collision"
        pilot = "add_the_smallest_feature_block_that_separates_conflicting_contexts"
    elif gradient["objective_gradient_conflict"]:
        mechanism = "objective_conflict"
        pilot = "target_only_set_objective_with_auxiliary_terms_removed"
    elif gradient["target_objective_gradient_dominated"]:
        mechanism = "objective_domination"
        pilot = "rebalance_the_set_objective_to_make_target_pressure_dominant"
    elif train["optimization_or_capacity_underfit"]:
        mechanism = "optimization_or_capacity_underfit"
        pilot = "single_host_target_set_curriculum_with_capacity_or_optimizer_change"
    elif train["generalization_failure"]:
        mechanism = "generalization"
        pilot = "single_host_regularized_or_augmented_generalization_treatment"
    elif anatomy["error_concentration_material"]:
        mechanism = "slice_specific_representation"
        pilot = "single_host_feature_treatment_for_the_top_concentrated_slice"
    else:
        mechanism = "diffuse_model_misspecification"
        pilot = "single_host_clean_larger_architecture_from_scratch"
    return {
        "selected_mechanism": mechanism,
        "authorized_pilot_family": pilot,
        "priority_order": [
            "representation_collision",
            "objective_conflict",
            "objective_domination",
            "optimization_or_capacity_underfit",
            "generalization",
            "slice_specific_representation",
            "diffuse_model_misspecification",
        ],
    }


def run_train_fit(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    """Evaluate the selected checkpoint on both open splits."""
    started = time.perf_counter()
    model, checkpoint, identities = _load_selected_model(
        run_dir,
        train_dataset,
        validation_dataset,
    )
    train = GradedOracleDataset(train_dataset)
    validation = GradedOracleDataset(validation_dataset)
    train_metrics = evaluate_frontier_anchored(model, train, group_batch_size=64)
    validation_metrics = evaluate_frontier_anchored(
        model,
        validation,
        group_batch_size=64,
    )
    scientific = {
        "identities": identities,
        "checkpoint": checkpoint.name,
        "train": train_metrics,
        "validation": validation_metrics,
        "classification": classify_train_fit(train_metrics, validation_metrics),
        "test_split_opened": False,
    }
    return _report("train-fit", scientific, started)


def run_observable_collision(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    """Measure exact contradictions under all model-visible inputs."""
    started = time.perf_counter()
    _model, checkpoint, identities = _load_selected_model(
        run_dir,
        train_dataset,
        validation_dataset,
    )
    datasets = (
        GradedOracleDataset(train_dataset),
        GradedOracleDataset(validation_dataset),
    )
    context_occurrences: dict[bytes, int] = defaultdict(int)
    groups = 0
    candidates = 0
    target_positives = 0
    conflicting_positive_occurrences = 0
    conflicting_occurrences = 0
    duplicate_candidate_occurrences = 0
    split_contexts: dict[str, set[bytes]] = defaultdict(set)
    within_context_conflicts: dict[bytes, tuple[int, int]] = defaultdict(
        lambda: (0, 0)
    )

    for dataset in datasets:
        for batch in dataset.batches(
            1,
            maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
            maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        ):
            candidate_bytes, target = _visible_candidate_rows_and_target(batch)
            parent = _visible_parent_digest(batch)
            rows = np.ascontiguousarray(candidate_bytes).view(
                np.dtype((np.void, candidate_bytes.shape[1]))
            ).reshape(-1)
            unique, inverse, counts = np.unique(
                rows,
                return_inverse=True,
                return_counts=True,
            )
            positives = np.bincount(
                inverse,
                weights=target.astype(np.int64),
                minlength=len(unique),
            ).astype(np.int64)
            negatives = counts.astype(np.int64) - positives
            conflicting = (positives > 0) & (negatives > 0)
            positive_conflicts = int(np.sum(positives[conflicting]))
            occurrence_conflicts = int(np.sum(counts[conflicting]))
            duplicate_candidate_occurrences += int(np.sum(counts[counts > 1]))

            context = blake3.blake3()
            context.update(parent)
            context.update(unique.tobytes())
            context.update(counts.astype("<u4", copy=False).tobytes())
            context_digest = context.digest()
            context_occurrences[context_digest] += 1
            previous_positive, previous_occurrences = within_context_conflicts[
                context_digest
            ]
            within_context_conflicts[context_digest] = (
                previous_positive + positive_conflicts,
                previous_occurrences + occurrence_conflicts,
            )
            split_contexts[dataset.split].add(context_digest)
            groups += 1
            candidates += len(rows)
            target_positives += int(np.sum(target))

    conflicting_positive_occurrences = sum(
        values[0] for values in within_context_conflicts.values()
    )
    conflicting_occurrences = sum(
        values[1] for values in within_context_conflicts.values()
    )
    repeated_contexts = {
        digest
        for digest, occurrences in context_occurrences.items()
        if occurrences > 1
    }
    if repeated_contexts:
        combined: dict[bytes, dict[bytes, list[int]]] = {
            digest: {} for digest in repeated_contexts
        }
        for dataset in datasets:
            for batch in dataset.batches(
                1,
                maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
                maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
            ):
                candidate_bytes, target = _visible_candidate_rows_and_target(batch)
                context_digest = _complete_context_digest(batch, candidate_bytes)
                if context_digest not in repeated_contexts:
                    continue
                row_counts = combined[context_digest]
                for row, positive in zip(candidate_bytes, target, strict=True):
                    key = row.tobytes()
                    counts = row_counts.setdefault(key, [0, 0])
                    counts[int(bool(positive))] += 1
        for context_digest, row_counts in combined.items():
            old_positive, old_occurrences = within_context_conflicts[context_digest]
            conflicting_positive_occurrences -= old_positive
            conflicting_occurrences -= old_occurrences
            for negatives, positives in row_counts.values():
                if negatives and positives:
                    conflicting_positive_occurrences += positives
                    conflicting_occurrences += negatives + positives

    repeated_context_groups = sum(
        occurrences for occurrences in context_occurrences.values() if occurrences > 1
    )
    metrics = {
        "groups": groups,
        "candidates": candidates,
        "target_positives": target_positives,
        "unique_complete_contexts": len(context_occurrences),
        "repeated_complete_context_groups": repeated_context_groups,
        "train_validation_context_overlap": len(
            split_contexts["train"] & split_contexts["validation"]
        ),
        "duplicate_candidate_observable_occurrences_within_context": (
            duplicate_candidate_occurrences
        ),
        "conflicting_target_occurrences": conflicting_occurrences,
        "conflicting_target_positive_occurrences": (
            conflicting_positive_occurrences
        ),
        "conflicting_target_positive_fraction": (
            conflicting_positive_occurrences / max(target_positives, 1)
        ),
    }
    scientific = {
        "identities": identities,
        "checkpoint": checkpoint.name,
        "metrics": metrics,
        "classification": classify_collision(metrics),
        "fingerprint_contract": (
            "exact parent tensors plus the permutation-invariant multiset of "
            "every candidate tensor consumed by the deployed model"
        ),
        "test_split_opened": False,
    }
    return _report("observable-collision", scientific, started)


def run_objective_gradient(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    """Measure component gradient magnitude and direction on widest groups."""
    started = time.perf_counter()
    model, checkpoint, identities = _load_selected_model(
        run_dir,
        train_dataset,
        validation_dataset,
    )
    train = GradedOracleDataset(train_dataset)
    batches = _widest_group_batches(train, OBJECTIVE_GRADIENT_GROUPS)
    component_names = (
        "target_set_cross_entropy",
        "r1200_listwise",
        "screen_only_regularization",
    )
    weights = {
        "target_set_cross_entropy": 1.0,
        "r1200_listwise": 0.5,
        "screen_only_regularization": 0.01,
    }
    value_sums = {name: 0.0 for name in component_names}
    norm_sums = {name: 0.0 for name in component_names}
    cosine_sum = 0.0
    target_margin_sum = 0.0
    cutoff_margin_sum = 0.0
    target_recall_sum = 0.0
    widths: list[int] = []

    gradient_functions = {
        name: nn.value_and_grad(
            model,
            lambda current, batch, component=name: frontier_anchored_loss_components(
                current,
                batch,
            )[component],
        )
        for name in component_names
    }
    for batch in batches:
        gradients: dict[str, dict[str, Any]] = {}
        for name, function in gradient_functions.items():
            value, gradient = function(model, batch)
            mx.eval(value, gradient)
            value_sums[name] += float(value.item())
            norm_sums[name] += _gradient_norm(gradient)
            gradients[name] = gradient
        cosine_sum += _gradient_cosine(
            gradients["target_set_cross_entropy"],
            gradients["r1200_listwise"],
        )
        margin = _target_score_margins(model, batch)
        target_margin_sum += margin["mean_margin"]
        cutoff_margin_sum += margin["cutoff_margin"]
        target_recall_sum += margin["target_recall"]
        widths.append(margin["candidates"])

    count = len(batches)
    mean_norms = {name: norm_sums[name] / count for name in component_names}
    weighted_norms = {
        name: mean_norms[name] * weights[name] for name in component_names
    }
    auxiliary_norm = float(
        np.hypot(
            weighted_norms["r1200_listwise"],
            weighted_norms["screen_only_regularization"],
        )
    )
    metrics = {
        "groups": count,
        "candidate_widths": widths,
        "mean_component_values": {
            name: value_sums[name] / count for name in component_names
        },
        "mean_gradient_norms": mean_norms,
        "weighted_gradient_norms": weighted_norms,
        "weighted_auxiliary_gradient_norm": auxiliary_norm,
        "target_listwise_gradient_cosine": cosine_sum / count,
        "mean_target_score_margin": target_margin_sum / count,
        "mean_target_cutoff_margin": cutoff_margin_sum / count,
        "mean_target_recall_at_quota": target_recall_sum / count,
    }
    scientific = {
        "identities": identities,
        "checkpoint": checkpoint.name,
        "metrics": metrics,
        "classification": classify_objective_gradient(metrics),
        "selection": (
            f"the {OBJECTIVE_GRADIENT_GROUPS} widest train groups, descending "
            "candidate count with deterministic shard/offset tie-breaks"
        ),
        "test_split_opened": False,
    }
    return _report("objective-gradient", scientific, started)


def run_error_anatomy(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    """Decompose nonfrontier target misses by observable decision slices."""
    started = time.perf_counter()
    model, checkpoint, identities = _load_selected_model(
        run_dir,
        train_dataset,
        validation_dataset,
    )
    validation = GradedOracleDataset(validation_dataset)
    slices: dict[str, dict[str, _BinarySlice]] = defaultdict(
        lambda: defaultdict(_BinarySlice)
    )
    total = _BinarySlice()
    groups = 0
    candidates = 0

    for batch in validation.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        mask = np.asarray(batch.candidate_mask)[0]
        count = int(np.sum(mask))
        scores = np.asarray(prediction.scores)[0, :count]
        flags = np.asarray(batch.source_flags)[0, :count]
        hashes = np.asarray(batch.action_hash)[0, :count]
        target = build_frontier_anchored_target_mask(
            r1200_mean=np.asarray(batch.r1200_mean),
            r1200_mask=np.asarray(batch.r1200_mask),
            source_flags=np.asarray(batch.source_flags),
            candidate_mask=np.asarray(batch.candidate_mask),
            action_hashes=np.asarray(batch.action_hash),
        )[0, :count]
        retained = frontier_anchored_retained_indices(
            scores=scores,
            source_flags=flags,
            action_hashes=hashes,
        )
        recalled = np.zeros(count, dtype=np.bool_)
        recalled[retained] = True
        frontier_count = int(
            np.sum((flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0)
        )
        phase = _PHASE_NAMES[int(np.asarray(batch.phase)[0])]
        width_bin = _candidate_count_bin(count)
        frontier_bin = _frontier_count_bin(frontier_count)
        draft_kind = np.asarray(batch.draft_kind)[0, :count]
        same_slot = np.asarray(batch.same_slot_independent)[0, :count]
        free_refresh = np.asarray(batch.replace_three_of_a_kind)[0, :count]
        wipe_count = np.asarray(batch.wipe_count)[0, :count]
        screen_rank = np.asarray(batch.screen_rank)[0, :count]
        action_features = np.asarray(batch.action_features)[0, :count]

        for index in np.flatnonzero(target):
            hit = bool(recalled[index])
            total.add(True, hit)
            categories = {
                "phase": phase,
                "candidate_count": width_bin,
                "frontier_count": frontier_bin,
                "action_family": _action_family(
                    int(draft_kind[index]),
                    int(same_slot[index]),
                    int(free_refresh[index]),
                    int(wipe_count[index]),
                ),
                "screen_rank": _screen_rank_bin(int(screen_rank[index])),
                "wildlife": _wildlife_name(action_features[index]),
                "immediate_delta_profile": _immediate_delta_profile(
                    action_features[index]
                ),
            }
            for dimension, name in categories.items():
                slices[dimension][name].add(True, hit)
        groups += 1
        candidates += count

    total_misses = total.positives - total.recalled
    reported_slices = {
        dimension: {
            name: accumulator.report(total_misses)
            for name, accumulator in sorted(values.items())
        }
        for dimension, values in sorted(slices.items())
    }
    overall = total.report(total_misses)
    scientific = {
        "identities": identities,
        "checkpoint": checkpoint.name,
        "metrics": {
            "groups": groups,
            "candidates": candidates,
            "overall": overall,
            "slices": reported_slices,
        },
        "classification": classify_error_anatomy(
            float(overall["recall"]),
            reported_slices,
        ),
        "test_split_opened": False,
    }
    return _report("error-anatomy", scientific, started)


def combine_reports(paths: list[Path]) -> dict[str, Any]:
    """Combine four host reports and select the next mechanism."""
    started = time.perf_counter()
    reports: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"unexpected diagnostic experiment in {path}")
        diagnostic = str(report["diagnostic"])
        if diagnostic in reports:
            raise ValueError(f"duplicate diagnostic report: {diagnostic}")
        if report["scientific"].get("test_split_opened"):
            raise ValueError(f"sealed test was opened by {path}")
        reports[diagnostic] = report["scientific"]
        identities.add(_canonical_digest(report["scientific"]["identities"]))
    required = {
        "train-fit",
        "observable-collision",
        "objective-gradient",
        "error-anatomy",
    }
    if set(reports) != required:
        raise ValueError(f"diagnostic set is incomplete: {sorted(reports)}")
    if len(identities) != 1:
        raise ValueError("diagnostic input identities differ across hosts")
    scientific = {
        "input_identity_blake3": next(iter(identities)),
        "diagnostics": {
            name: {
                "classification": report["classification"],
                "scientific_blake3": _canonical_digest(report),
            }
            for name, report in sorted(reports.items())
        },
        "selection": select_failure_mechanism(reports),
        "test_split_opened": False,
    }
    return _report("combined", scientific, started)


def _load_selected_model(
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> tuple[GradedOracleRanker, Path, dict[str, Any]]:
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("kind") != "graded-oracle-frontier-anchored-ranking":
        raise ValueError("source run is not a frontier-anchored ranking run")
    best = json.loads((run_dir / "best.json").read_text())
    if best.get("checkpoint") != EXPECTED_CHECKPOINT:
        raise ValueError("selected checkpoint drifted from ADR 0089")
    train = GradedOracleDataset(train_dataset)
    validation = GradedOracleDataset(validation_dataset)
    if train.split != "train" or validation.split != "validation":
        raise ValueError("diagnostics accept only open train and validation splits")
    if _checksum(train.root / "dataset.json") != run["datasets"]["train_manifest_blake3"]:
        raise ValueError("train manifest does not match the selected run")
    if (
        _checksum(validation.root / "dataset.json")
        != run["datasets"]["validation_manifest_blake3"]
    ):
        raise ValueError("validation manifest does not match the selected run")
    training = run["training"]
    if float(training["learning_rate"]) != 1e-4 or float(training["weight_decay"]) != 1e-4:
        raise ValueError("selected run optimizer identity drifted")
    checkpoint = run_dir / "checkpoints" / str(best["checkpoint"])
    checkpoint_manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    model_metadata = checkpoint_manifest["files"]["model.safetensors"]
    model_path = checkpoint / "model.safetensors"
    if (
        model_path.stat().st_size != int(model_metadata["bytes"])
        or _checksum(model_path) != model_metadata["blake3"]
    ):
        raise ValueError("selected checkpoint model failed integrity validation")
    model = GradedOracleRanker(
        GradedOracleModelConfig.from_dict(checkpoint_manifest["model_config"])
    )
    model.load_weights(str(model_path))
    mx.eval(model.parameters())
    if checkpoint.name != EXPECTED_CHECKPOINT:
        raise ValueError("checkpoint pointer drifted from ADR 0089")
    model.eval()
    identities = {
        "source_experiment": EXPECTED_SOURCE_EXPERIMENT,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(checkpoint / "model.safetensors"),
        "train_dataset_id": train.manifest["dataset_id"],
        "train_manifest_blake3": _checksum(train.root / "dataset.json"),
        "validation_dataset_id": validation.manifest["dataset_id"],
        "validation_manifest_blake3": _checksum(
            validation.root / "dataset.json"
        ),
    }
    return model, checkpoint, identities


def _visible_parent_digest(batch: GradedOracleBatch) -> bytes:
    digest = blake3.blake3()
    for value in (
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
    ):
        array = np.ascontiguousarray(np.asarray(value)[0])
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<u4").tobytes())
        digest.update(array.tobytes())
    return digest.digest()


def _visible_candidate_rows_and_target(
    batch: GradedOracleBatch,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(batch.candidate_mask)[0]
    count = int(np.sum(mask))
    pieces = []
    for value in (
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.screen_value,
    ):
        array = np.ascontiguousarray(np.asarray(value)[0, :count])
        pieces.append(array.view(np.uint8).reshape(count, -1))
    rows = np.ascontiguousarray(np.concatenate(pieces, axis=1))
    target = build_frontier_anchored_target_mask(
        r1200_mean=np.asarray(batch.r1200_mean),
        r1200_mask=np.asarray(batch.r1200_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=np.asarray(batch.candidate_mask),
        action_hashes=np.asarray(batch.action_hash),
    )[0, :count]
    return rows, target


def _complete_context_digest(
    batch: GradedOracleBatch,
    candidate_bytes: np.ndarray,
) -> bytes:
    rows = np.ascontiguousarray(candidate_bytes).view(
        np.dtype((np.void, candidate_bytes.shape[1]))
    ).reshape(-1)
    unique, counts = np.unique(rows, return_counts=True)
    context = blake3.blake3()
    context.update(_visible_parent_digest(batch))
    context.update(unique.tobytes())
    context.update(counts.astype("<u4", copy=False).tobytes())
    return context.digest()


def _widest_group_batches(
    dataset: GradedOracleDataset,
    count: int,
) -> list[GradedOracleBatch]:
    candidates = []
    for shard_index, shard in enumerate(dataset.shards):
        for ref_index, ref in enumerate(shard.groups):
            candidates.append(
                (-ref.candidate_count, shard_index, ref_index, ref)
            )
    selected = sorted(candidates)[:count]
    return [
        decode_graded_oracle_groups(
            dataset.shards[shard_index].bytes(),
            (ref,),
        )
        for _width, shard_index, _ref_index, ref in selected
    ]


def _gradient_norm(gradient: dict[str, Any]) -> float:
    total = 0.0
    for _name, value in tree_flatten(gradient):
        array = np.asarray(value, dtype=np.float64)
        total += float(np.sum(array * array))
    return float(np.sqrt(total))


def _gradient_cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_values = dict(tree_flatten(left))
    right_values = dict(tree_flatten(right))
    if left_values.keys() != right_values.keys():
        raise ValueError("gradient trees differ")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for name in left_values:
        left_array = np.asarray(left_values[name], dtype=np.float64)
        right_array = np.asarray(right_values[name], dtype=np.float64)
        dot += float(np.sum(left_array * right_array))
        left_norm += float(np.sum(left_array * left_array))
        right_norm += float(np.sum(right_array * right_array))
    return dot / max(float(np.sqrt(left_norm * right_norm)), 1e-30)


def _target_score_margins(
    model: GradedOracleRanker,
    batch: GradedOracleBatch,
) -> dict[str, float | int]:
    prediction = predict_graded_oracle_batch(model, batch)
    mx.eval(prediction.scores)
    mask = np.asarray(batch.candidate_mask)[0]
    count = int(np.sum(mask))
    scores = np.asarray(prediction.scores)[0, :count]
    flags = np.asarray(batch.source_flags)[0, :count]
    target = build_frontier_anchored_target_mask(
        r1200_mean=np.asarray(batch.r1200_mean),
        r1200_mask=np.asarray(batch.r1200_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=np.asarray(batch.candidate_mask),
        action_hashes=np.asarray(batch.action_hash),
    )[0, :count]
    eligible = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    non_target = eligible & ~target
    quota = int(np.sum(target))
    eligible_indices = np.flatnonzero(eligible)
    order = eligible_indices[
        np.argsort(-scores[eligible_indices], kind="stable")
    ]
    retained = order[:quota]
    return {
        "candidates": count,
        "mean_margin": float(np.mean(scores[target]) - np.mean(scores[non_target])),
        "cutoff_margin": float(np.min(scores[target]) - np.max(scores[non_target])),
        "target_recall": float(np.sum(target[retained]) / max(quota, 1)),
    }


def _candidate_count_bin(count: int) -> str:
    if count <= 2048:
        return "0001-2048"
    if count <= 4096:
        return "2049-4096"
    if count <= 8192:
        return "4097-8192"
    return "8193-16384"


def _frontier_count_bin(count: int) -> str:
    if count <= 16:
        return "00-16"
    if count <= 24:
        return "17-24"
    if count <= 31:
        return "25-31"
    return "32"


def _screen_rank_bin(rank: int) -> str:
    if rank <= 64:
        return "0001-0064"
    if rank <= 128:
        return "0065-0128"
    if rank <= 256:
        return "0129-0256"
    if rank <= 512:
        return "0257-0512"
    if rank <= 1024:
        return "0513-1024"
    return "1025-plus"


def _action_family(
    draft_kind: int,
    same_slot: int,
    free_refresh: int,
    wipe_count: int,
) -> str:
    if wipe_count > 0:
        return "paid-wipe"
    if free_refresh:
        return "free-refresh"
    if same_slot:
        return "same-slot-independent"
    if draft_kind == 1:
        return "independent"
    return "paired"


def _wildlife_name(action_features: np.ndarray) -> str:
    wildlife = action_features[29:34]
    return _WILDLIFE_NAMES[int(np.argmax(wildlife))]


def _immediate_delta_profile(action_features: np.ndarray) -> str:
    deltas = action_features[129:140]
    wildlife = float(np.sum(np.abs(deltas[:5])))
    habitat = float(np.sum(np.abs(deltas[5:10])))
    tokens = abs(float(deltas[10]))
    if wildlife >= habitat and wildlife >= tokens and wildlife > 0:
        return "wildlife-led"
    if habitat >= tokens and habitat > 0:
        return "habitat-led"
    if tokens > 0:
        return "token-led"
    return "no-immediate-delta"


def _report(
    diagnostic: str,
    scientific: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "diagnostic": diagnostic,
        "host": _host_alias(),
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "system_swap_used_bytes": _system_swap_used_bytes(),
        },
    }


def _host_alias() -> str:
    host = socket.gethostname().split(".")[0]
    return HOST_ALIASES.get(host, host)


def _system_swap_used_bytes() -> int | None:
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
    match = _SWAP_USED_RE.search(output)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="diagnostic", required=True)
    for name in (
        "train-fit",
        "observable-collision",
        "objective-gradient",
        "error-anatomy",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--train-dataset", type=Path, required=True)
        command.add_argument("--validation-dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--report", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.diagnostic == "combine":
        report = combine_reports(args.report)
    else:
        function = {
            "train-fit": run_train_fit,
            "observable-collision": run_observable_collision,
            "objective-gradient": run_objective_gradient,
            "error-anatomy": run_error_anatomy,
        }[args.diagnostic]
        report = function(
            run_dir=args.run_dir,
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
        )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
