#!/usr/bin/env python3
"""Qualify exact preverified vectorized R2 candidate materialization."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import socket
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.checkpoint import load_latest_checkpoint_with_factory
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
    CONTROL_MATERIALIZATION_VERIFIED,
    R3ActionEditMlxCache,
)
from cascadia_mlx.r3_action_edit_mlx_metrics import (
    CANDIDATE_CHUNK,
    _system_swap_used_bytes,
)
from cascadia_mlx.relational_substrate_mlx_cache import (
    CONTROL_ARM,
    RelationalSubstrateMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateModelConfig,
    RelationalSubstrateRanker,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

SCHEMA_VERSION = 1
EXPERIMENT_ID = "exact-r2-preverified-vectorized-materialization-v1"
PROTOCOL_ID = "exact-r2-vectorized-materialization-parity-v1"
ADR_ID = "0167"
LEGACY = "legacy-preverified"
VECTORIZED = "preverified-vectorized"
ORDERS = {
    "legacy-first": (LEGACY, VECTORIZED),
    "vectorized-first": (VECTORIZED, LEGACY),
}
EXPECTED_COUNTS = {
    "train": (560, 280_012),
    "validation": (240, 860_203),
}
PREDICTION_TOLERANCE = 1e-6
MINIMUM_P99_SPEEDUP = 10.0
MAXIMUM_VECTORIZED_P99_MILLISECONDS = 410.0
MAXIMUM_PEAK_RSS_BYTES = 4 * 1024**3
MATERIALIZATION_DIGEST_CONTRACT = (
    "blake3-over-label-dtype-shape-and-contiguous-bytes-v1"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R3_CACHE = (
    REPO_ROOT
    / "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache"
    / "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
DEFAULT_S1_CACHE = (
    REPO_ROOT
    / "artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache"
    / "2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15"
)
DEFAULT_RELATIONAL_CACHE = (
    REPO_ROOT
    / "artifacts/experiments/relational-substrate-mlx-tournament-v1/cache"
    / "d4f8e2eb83db237b136fd478b73802544938c36adf77db0bf40f2b3276181bef"
)
DEFAULT_TRAIN_DATASET = REPO_ROOT / "artifacts/datasets/complete-action-graded-oracle-v1-train"
DEFAULT_VALIDATION_DATASET = (
    REPO_ROOT / "artifacts/datasets/complete-action-graded-oracle-v1-validation"
)
DEFAULT_C0_RUN = (
    REPO_ROOT / "artifacts/experiments/relational-substrate-mlx-tournament-v1/runs" / "c0_exact_r2"
)
DEFAULT_PREREGISTRATION = (
    REPO_ROOT
    / "artifacts/experiments/exact-r2-preverified-vectorized-materialization-v1"
    / "preregistration.json"
)


class QualificationError(ValueError):
    """The materialization qualification is malformed or inexact."""


@dataclass(frozen=True)
class _Context:
    dataset: object
    open_data_identity: dict[str, Any]
    proof_id: str


def compare_materialization(
    *,
    split: str,
    order: str,
    rows: np.ndarray | None,
    r3_cache: Path,
    s1_cache: Path,
    relational_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    expected_proof_id: str | None,
) -> dict[str, Any]:
    """Compare complete legacy and vectorized tensors over one open split."""
    if split not in EXPECTED_COUNTS:
        raise QualificationError("split must be train or validation")
    if order not in ORDERS:
        raise QualificationError("unknown crossed benchmark order")
    context = _load_context(
        split=split,
        r3_cache=r3_cache,
        s1_cache=s1_cache,
        relational_cache=relational_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        expected_proof_id=expected_proof_id,
    )
    selected_rows = _normalize_rows(
        rows,
        group_count=int(context.dataset.group_count),
    )
    timings = {
        LEGACY: np.empty(len(selected_rows), dtype=np.float64),
        VECTORIZED: np.empty(len(selected_rows), dtype=np.float64),
    }
    action_counts = np.empty(len(selected_rows), dtype=np.int64)
    token_counts = np.empty(len(selected_rows), dtype=np.int64)
    maximum_tokens = np.empty(len(selected_rows), dtype=np.int64)
    parity_failures: list[dict[str, Any]] = []
    digests = {
        LEGACY: blake3.blake3(),
        VECTORIZED: blake3.blake3(),
    }
    swap_before = _system_swap_used_bytes()

    for sample_index, row in enumerate(selected_rows):
        row_field_digests: dict[str, dict[str, bytes]] = {}
        for mode in ORDERS[order]:
            started = time.perf_counter()
            batch = context.dataset.batch(
                [int(row)],
                arm=CONTROL_ARM,
                transform_ids=[0],
                verify_control_hashes=False,
                control_materialization=_materialization_mode(mode),
            )
            timings[mode][sample_index] = time.perf_counter() - started
            if mode == ORDERS[order][0]:
                action_counts[sample_index] = int(
                    np.asarray(
                        batch.base.candidate_mask,
                        dtype=np.bool_,
                    ).sum()
                )
                token_counts[sample_index] = int(
                    np.asarray(
                        batch.r3.candidate_token_mask,
                        dtype=np.bool_,
                    ).sum()
                )
                maximum_tokens[sample_index] = int(
                    np.asarray(
                        batch.r3.candidate_token_counts,
                        dtype=np.int64,
                    ).max()
                )
            field_digests = {}
            for label, values in _batch_arrays(batch):
                field_digest = _array_digest(label, np.asarray(values))
                field_digests[label] = field_digest
                _update_field_digest(
                    digests[mode],
                    label,
                    field_digest,
                )
            row_field_digests[mode] = field_digests
            del batch
            gc.collect()
            mx.clear_cache()

        if row_field_digests[LEGACY] != row_field_digests[VECTORIZED]:
            parity_failures.append(
                _diagnose_materialization_failure(
                    context=context,
                    split=split,
                    row=int(row),
                    legacy_digests=row_field_digests[LEGACY],
                    vectorized_digests=row_field_digests[VECTORIZED],
                )
            )
        if parity_failures:
            break

    swap_after = _system_swap_used_bytes()
    completed = len(selected_rows) if not parity_failures else sample_index + 1
    expected_groups, expected_actions = EXPECTED_COUNTS[split]
    complete_split = len(selected_rows) == expected_groups and np.array_equal(
        selected_rows,
        np.arange(expected_groups, dtype=np.int64),
    )
    observed_actions = int(action_counts[:completed].sum())
    if complete_split and observed_actions != expected_actions:
        raise QualificationError(f"{split} compared action count differs from the frozen cohort")

    performance = {
        mode: {
            "latency_seconds": timings[mode][:completed].tolist(),
            "latency_milliseconds": _latency_summary(timings[mode][:completed] * 1000.0),
            "elapsed_seconds": float(timings[mode][:completed].sum()),
            "actions_per_second": (
                observed_actions / max(float(timings[mode][:completed].sum()), 1e-12)
            ),
        }
        for mode in (LEGACY, VECTORIZED)
    }
    legacy_p99 = performance[LEGACY]["latency_milliseconds"]["p99"]
    vectorized_p99 = performance[VECTORIZED]["latency_milliseconds"]["p99"]
    performance["p99_speedup"] = legacy_p99 / max(vectorized_p99, 1e-12)
    performance["elapsed_speedup"] = performance[LEGACY]["elapsed_seconds"] / max(
        performance[VECTORIZED]["elapsed_seconds"], 1e-12
    )

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "complete-materialization-parity-and-performance",
        "split": split,
        "order": order,
        "rows": selected_rows.tolist(),
        "complete_split": complete_split,
        "groups_compared": completed,
        "actions_compared": observed_actions,
        "candidate_tokens_compared": int(token_counts[:completed].sum()),
        "maximum_candidate_tokens": int(maximum_tokens[:completed].max(initial=0)),
        "parity": {
            "pass": not parity_failures,
            "failures": parity_failures,
            "failure_count": len(parity_failures),
            "digest_contract": MATERIALIZATION_DIGEST_CONTRACT,
            "legacy_digest": digests[LEGACY].hexdigest(),
            "vectorized_digest": digests[VECTORIZED].hexdigest(),
            "digests_equal": (digests[LEGACY].digest() == digests[VECTORIZED].digest()),
        },
        "performance": performance,
        "memory": {
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "system_swap_before_bytes": swap_before,
            "system_swap_after_bytes": swap_after,
            "system_swap_delta_bytes": _swap_delta(swap_before, swap_after),
        },
        "open_data_verification": context.open_data_identity,
        "open_data_verification_id": context.proof_id,
        "source": _source_identity(),
        "runtime": _runtime_identity(),
    }
    return _report(identity)


def compare_predictions(
    *,
    rows: np.ndarray | None,
    run_dir: Path,
    r3_cache: Path,
    s1_cache: Path,
    relational_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    expected_proof_id: str | None,
    candidate_chunk: int,
) -> dict[str, Any]:
    """Replay the frozen C0 model over both exact materialization paths."""
    if candidate_chunk <= 0:
        raise QualificationError("candidate chunk must be positive")
    context = _load_context(
        split="validation",
        r3_cache=r3_cache,
        s1_cache=s1_cache,
        relational_cache=relational_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        expected_proof_id=expected_proof_id,
    )
    selected_rows = _normalize_rows(
        rows,
        group_count=int(context.dataset.group_count),
    )
    mx.set_default_device(mx.gpu)
    previous_cache_limit = int(mx.set_cache_limit(1024**3))
    model, _optimizer, state, checkpoint = load_latest_checkpoint_with_factory(
        run_dir,
        learning_rate=1e-4,
        weight_decay=1e-4,
        model_factory=lambda values: RelationalSubstrateRanker(
            RelationalSubstrateModelConfig.from_dict(values)
        ),
    )
    if model.config.arm != CONTROL_ARM:
        raise QualificationError("prediction parity requires the exact-R2 C0 checkpoint")
    model.eval()
    maximum_errors = {
        "scores": 0.0,
        "residuals": 0.0,
        "standard_errors": 0.0,
    }
    selected_rank_disagreements = 0
    parity_failures: list[dict[str, Any]] = []
    actions = 0
    digests = {
        LEGACY: blake3.blake3(),
        VECTORIZED: blake3.blake3(),
    }
    swap_before = _system_swap_used_bytes()
    started = time.perf_counter()

    for row in selected_rows:
        predictions: dict[str, dict[str, np.ndarray]] = {}
        selected_indices: dict[str, int] = {}
        for mode in (LEGACY, VECTORIZED):
            batch = context.dataset.batch(
                [int(row)],
                arm=CONTROL_ARM,
                transform_ids=[0],
                verify_control_hashes=False,
                control_materialization=_materialization_mode(mode),
            )
            selected_indices[mode] = int(np.asarray(batch.base.selected_index)[0])
            predictions[mode] = _predict_complete_decision(
                model,
                batch,
                candidate_chunk=candidate_chunk,
            )
            actions += int(np.asarray(batch.base.candidate_mask).sum()) if mode == LEGACY else 0
            for label, values in predictions[mode].items():
                _update_array_digest(digests[mode], label, values)
            del batch
            gc.collect()
            mx.clear_cache()

        for label in maximum_errors:
            expected = predictions[LEGACY][label]
            observed = predictions[VECTORIZED][label]
            error = float(np.max(np.abs(expected - observed), initial=0.0))
            maximum_errors[label] = max(maximum_errors[label], error)
            if error > PREDICTION_TOLERANCE:
                parity_failures.append(
                    _array_failure(
                        split="validation",
                        row=int(row),
                        label=label,
                        expected=expected,
                        observed=observed,
                    )
                )
                break
        legacy_rank = _stable_descending_rank(
            predictions[LEGACY]["scores"],
            selected_indices[LEGACY],
        )
        vectorized_rank = _stable_descending_rank(
            predictions[VECTORIZED]["scores"],
            selected_indices[VECTORIZED],
        )
        if legacy_rank != vectorized_rank:
            selected_rank_disagreements += 1
        if parity_failures:
            break

    elapsed = time.perf_counter() - started
    swap_after = _system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "frozen-c0-prediction-parity",
        "rows": selected_rows.tolist(),
        "complete_validation": (
            len(selected_rows) == EXPECTED_COUNTS["validation"][0]
            and np.array_equal(
                selected_rows,
                np.arange(EXPECTED_COUNTS["validation"][0], dtype=np.int64),
            )
        ),
        "actions_compared": actions,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "global_step": state.global_step,
            "manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
            "model_blake3": _checksum(checkpoint / "model.safetensors"),
        },
        "candidate_chunk": candidate_chunk,
        "parity": {
            "pass": (
                not parity_failures
                and selected_rank_disagreements == 0
                and max(maximum_errors.values()) <= PREDICTION_TOLERANCE
            ),
            "failures": parity_failures,
            "failure_count": len(parity_failures),
            "maximum_absolute_errors": maximum_errors,
            "selected_rank_disagreements": selected_rank_disagreements,
            "tolerance": PREDICTION_TOLERANCE,
            "legacy_digest": digests[LEGACY].hexdigest(),
            "vectorized_digest": digests[VECTORIZED].hexdigest(),
            "digests_equal": (digests[LEGACY].digest() == digests[VECTORIZED].digest()),
        },
        "elapsed_seconds": elapsed,
        "memory": {
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "system_swap_before_bytes": swap_before,
            "system_swap_after_bytes": swap_after,
            "system_swap_delta_bytes": _swap_delta(swap_before, swap_after),
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
        },
        "open_data_verification": context.open_data_identity,
        "open_data_verification_id": context.proof_id,
        "source": _source_identity(),
        "runtime": _runtime_identity(),
    }
    return _report(identity)


def classify_reports(
    comparisons: list[dict[str, Any]],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen ADR 0167 gates without report-order dependence."""
    if len(comparisons) < 3:
        raise QualificationError(
            "classification requires two crossed validation reports "
            "and one complete train report"
        )
    identities = [_validated_identity(report) for report in comparisons]
    prediction_identity = _validated_identity(prediction)
    if any(
        identity.get("kind") != "complete-materialization-parity-and-performance"
        for identity in identities
    ):
        raise QualificationError("comparison report kind is invalid")
    if prediction_identity.get("kind") != "frozen-c0-prediction-parity":
        raise QualificationError("prediction report kind is invalid")
    validation = [
        identity
        for identity in identities
        if identity.get("split") == "validation" and identity.get("complete_split") is True
    ]
    train = [
        identity
        for identity in identities
        if identity.get("split") == "train" and identity.get("complete_split") is True
    ]
    orders = {identity.get("order") for identity in validation}
    structural = (
        len(identities) == 3
        and len(validation) == 2
        and len(train) == 1
        and orders == {"legacy-first", "vectorized-first"}
        and all(
            int(identity.get("groups_compared", -1)) == EXPECTED_COUNTS["validation"][0]
            and int(identity.get("actions_compared", -1)) == EXPECTED_COUNTS["validation"][1]
            for identity in validation
        )
        and int(train[0].get("groups_compared", -1)) == EXPECTED_COUNTS["train"][0]
        and int(train[0].get("actions_compared", -1)) == EXPECTED_COUNTS["train"][1]
        and prediction_identity.get("complete_validation") is True
        and int(prediction_identity.get("actions_compared", -1))
        == EXPECTED_COUNTS["validation"][1]
    )
    all_identities = [*identities, prediction_identity]
    expected_source = _source_identity()
    runtime_hosts = [
        identity.get("runtime", {}).get("host")
        if isinstance(identity.get("runtime"), dict)
        else None
        for identity in all_identities
    ]
    proof_ids = [
        identity.get("open_data_verification_id")
        for identity in all_identities
    ]
    cross_host_consistent = (
        structural
        and all(identity.get("source") == expected_source for identity in all_identities)
        and all(isinstance(value, str) and value for value in proof_ids)
        and len(set(proof_ids)) == 1
        and all(isinstance(host, str) and host for host in runtime_hosts)
        and len(set(runtime_hosts)) == len(runtime_hosts)
    )
    parity_pass = (
        all(identity["parity"]["pass"] is True for identity in identities)
        and prediction_identity["parity"]["pass"] is True
    )
    p99_speedups = [float(identity["performance"]["p99_speedup"]) for identity in validation]
    vectorized_p99 = [
        float(identity["performance"][VECTORIZED]["latency_milliseconds"]["p99"])
        for identity in validation
    ]
    peak_rss = [
        int(identity["memory"]["peak_process_rss_bytes"])
        for identity in [*identities, prediction_identity]
    ]
    swap_deltas = [
        identity["memory"]["system_swap_delta_bytes"]
        for identity in [*identities, prediction_identity]
    ]
    speed_pass = (
        len(p99_speedups) >= 2
        and all(value >= MINIMUM_P99_SPEEDUP for value in p99_speedups)
        and all(value <= MAXIMUM_VECTORIZED_P99_MILLISECONDS for value in vectorized_p99)
    )
    memory_pass = all(value <= MAXIMUM_PEAK_RSS_BYTES for value in peak_rss)
    swap_pass = all(value is not None and int(value) <= 0 for value in swap_deltas)
    if not structural:
        classification = "exact_r2_vectorized_materialization_structurally_invalid"
    elif not cross_host_consistent:
        classification = "exact_r2_vectorized_materialization_cross_host_inconsistent"
    elif not parity_pass:
        classification = "exact_r2_vectorized_materialization_parity_failure"
    elif not memory_pass or not swap_pass:
        classification = "exact_r2_vectorized_materialization_memory_failure"
    elif not speed_pass:
        classification = "exact_r2_vectorized_materialization_speed_failure"
    else:
        classification = "exact_r2_vectorized_materialization_promoted"
    scientific = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "qualification-classification",
        "classification": classification,
        "promoted": (classification == "exact_r2_vectorized_materialization_promoted"),
        "gates": {
            "structural": structural,
            "cross_host_consistent": cross_host_consistent,
            "runtime_hosts": runtime_hosts,
            "source_identity_blake3": _canonical_blake3(expected_source),
            "feature_and_prediction_parity": parity_pass,
            "minimum_p99_speedup": min(p99_speedups, default=0.0),
            "minimum_p99_speedup_required": MINIMUM_P99_SPEEDUP,
            "maximum_vectorized_p99_milliseconds": max(
                vectorized_p99,
                default=float("inf"),
            ),
            "maximum_vectorized_p99_milliseconds_allowed": (MAXIMUM_VECTORIZED_P99_MILLISECONDS),
            "maximum_peak_process_rss_bytes": max(peak_rss, default=0),
            "maximum_peak_process_rss_bytes_allowed": MAXIMUM_PEAK_RSS_BYTES,
            "swap_pass": swap_pass,
        },
        "comparison_report_ids": sorted(report["report_id"] for report in comparisons),
        "prediction_report_id": prediction["report_id"],
        "source": _source_identity(),
    }
    return _report(scientific)


def _load_context(
    *,
    split: str,
    r3_cache: Path,
    s1_cache: Path,
    relational_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    expected_proof_id: str | None,
) -> _Context:
    r3 = R3ActionEditMlxCache(
        r3_cache,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    relational = RelationalSubstrateMlxCache(
        relational_cache,
        r3_cache=r3,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    s1 = S1ExactSupplyCache(
        s1_cache,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    identity = open_data_verification_identity(
        cache=relational,
        s1_cache=s1,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    proof_id = open_data_verification_id(identity)
    if expected_proof_id is not None and proof_id != expected_proof_id:
        raise QualificationError("open-data proof differs from the frozen protocol")
    dataset_root = train_dataset if split == "train" else validation_dataset
    dataset = relational.bind_dataset(
        dataset_root,
        s1_cache=s1,
        verify_dataset_checksums=False,
        preverified_open_data_proof_id=proof_id,
    )
    return _Context(
        dataset=dataset,
        open_data_identity=identity,
        proof_id=proof_id,
    )


def _materialization_mode(mode: str) -> str:
    if mode == LEGACY:
        return CONTROL_MATERIALIZATION_VERIFIED
    if mode == VECTORIZED:
        return CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED
    raise QualificationError(f"unknown materialization mode: {mode}")


def _parity_arrays(
    legacy: object,
    vectorized: object,
) -> tuple[tuple[str, object, object], ...]:
    left = dict(_batch_arrays(legacy))
    right = dict(_batch_arrays(vectorized))
    if left.keys() != right.keys():
        raise QualificationError("materialization parity field coverage differs")
    return tuple(
        (label, left[label], right[label])
        for label in left
    )


def _batch_arrays(
    batch: object,
) -> tuple[tuple[str, object], ...]:
    return (
        (
            "candidate_token_features",
            batch.r3.candidate_token_features,
        ),
        (
            "candidate_token_mask",
            batch.r3.candidate_token_mask,
        ),
        (
            "candidate_token_counts",
            batch.r3.candidate_token_counts,
        ),
        (
            "canonical_transform_ids",
            batch.r3.canonical_transform_ids,
        ),
        (
            "source_candidate_indices",
            batch.r3.source_candidate_indices,
        ),
        ("action_hash", batch.base.action_hash),
        (
            "candidate_mask",
            batch.base.candidate_mask,
        ),
        ("selected", batch.base.selected),
        ("champion", batch.base.champion),
        ("screen_value", batch.base.screen_value),
        (
            "parent_token_features",
            batch.parent.r2_token_features,
        ),
        (
            "supply_vector",
            batch.r3.supply_vector,
        ),
        (
            "staged_supply_vector",
            batch.r3.staged_supply_vector,
        ),
        (
            "frontier_features",
            batch.r3.frontier_features,
        ),
    )


def _diagnose_materialization_failure(
    *,
    context: _Context,
    split: str,
    row: int,
    legacy_digests: dict[str, bytes],
    vectorized_digests: dict[str, bytes],
) -> dict[str, Any]:
    legacy = context.dataset.batch(
        [row],
        arm=CONTROL_ARM,
        transform_ids=[0],
        verify_control_hashes=False,
        control_materialization=_materialization_mode(LEGACY),
    )
    vectorized = context.dataset.batch(
        [row],
        arm=CONTROL_ARM,
        transform_ids=[0],
        verify_control_hashes=False,
        control_materialization=_materialization_mode(VECTORIZED),
    )
    try:
        for label, expected, observed in _parity_arrays(
            legacy,
            vectorized,
        ):
            expected_array = np.asarray(expected)
            observed_array = np.asarray(observed)
            if not np.array_equal(expected_array, observed_array):
                return _array_failure(
                    split=split,
                    row=row,
                    label=label,
                    expected=expected_array,
                    observed=observed_array,
                )
        return {
            "split": split,
            "row": row,
            "label": "field-digest",
            "reason": "digest-mismatch-without-array-difference",
            "legacy_field_digests": {
                label: value.hex()
                for label, value in legacy_digests.items()
            },
            "vectorized_field_digests": {
                label: value.hex()
                for label, value in vectorized_digests.items()
            },
        }
    finally:
        del legacy, vectorized
        gc.collect()
        mx.clear_cache()


def _predict_complete_decision(
    model: RelationalSubstrateRanker,
    batch: object,
    *,
    candidate_chunk: int,
) -> dict[str, np.ndarray]:
    action_count = int(np.asarray(batch.base.candidate_mask, dtype=np.bool_).sum())
    parent = model.encode_parent(batch)
    values: dict[str, list[np.ndarray]] = {
        "scores": [],
        "residuals": [],
        "standard_errors": [],
    }
    for start in range(0, action_count, candidate_chunk):
        prediction = model.predict(
            batch,
            candidate_slice=slice(
                start,
                min(start + candidate_chunk, action_count),
            ),
            parent_state=parent,
        )
        mx.eval(
            prediction.scores,
            prediction.residuals,
            prediction.standard_errors,
        )
        values["scores"].append(np.asarray(prediction.scores)[0].copy())
        values["residuals"].append(np.asarray(prediction.residuals)[0].copy())
        values["standard_errors"].append(np.asarray(prediction.standard_errors)[0].copy())
    return {label: np.concatenate(chunks) for label, chunks in values.items()}


def _stable_descending_rank(scores: np.ndarray, selected_index: int) -> int:
    order = np.argsort(-np.asarray(scores), kind="stable")
    matches = np.flatnonzero(order == selected_index)
    if len(matches) != 1:
        raise QualificationError("selected action is absent from prediction order")
    return int(matches[0])


def _normalize_rows(
    rows: np.ndarray | None,
    *,
    group_count: int,
) -> np.ndarray:
    selected = (
        np.arange(group_count, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    )
    if (
        selected.ndim != 1
        or not len(selected)
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= group_count)
    ):
        raise QualificationError("rows must be unique, nonempty, and in range")
    return selected


def _parse_rows(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        return np.asarray(
            [int(value) for value in raw.split(",") if value],
            dtype=np.int64,
        )
    except ValueError as error:
        raise QualificationError("rows must be comma-separated integers") from error


def _array_failure(
    *,
    split: str,
    row: int,
    label: str,
    expected: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Any]:
    if expected.shape != observed.shape:
        return {
            "split": split,
            "row": row,
            "label": label,
            "reason": "shape",
            "expected_shape": list(expected.shape),
            "observed_shape": list(observed.shape),
        }
    differences = np.flatnonzero(expected.reshape(-1) != observed.reshape(-1))
    if not len(differences):
        return {
            "split": split,
            "row": row,
            "label": label,
            "reason": "non-equal-without-discrete-index",
        }
    flat = int(differences[0])
    index = np.unravel_index(flat, expected.shape)
    return {
        "split": split,
        "row": row,
        "label": label,
        "reason": "value",
        "index": [int(value) for value in index],
        "expected": _json_scalar(expected[index]),
        "observed": _json_scalar(observed[index]),
    }


def _update_array_digest(
    digest: blake3.blake3,
    label: str,
    values: np.ndarray,
) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(label.encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).view(np.uint8).tobytes())
    digest.update(memoryview(contiguous).cast("B"))


def _array_digest(label: str, values: np.ndarray) -> bytes:
    digest = blake3.blake3()
    _update_array_digest(digest, label, values)
    return digest.digest()


def _update_field_digest(
    digest: blake3.blake3,
    label: str,
    field_digest: bytes,
) -> None:
    digest.update(label.encode())
    digest.update(field_digest)


def _latency_summary(milliseconds: np.ndarray) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise QualificationError("latency samples must be finite and nonempty")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _source_identity() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "python/cascadia_mlx/r3_action_edit_mlx_cache.py",
        REPO_ROOT / "python/cascadia_mlx/relational_substrate_mlx_cache.py",
        REPO_ROOT / "docs/v2/decisions/0167-exact-r2-preverified-vectorized-materialization.md",
        REPO_ROOT
        / "docs/v2/reports/exact-r2-preverified-vectorized-materialization-v1-preregistration.md",
        REPO_ROOT
        / "docs/v2/decisions/0170-streaming-materialization-parity-verifier.md",
        REPO_ROOT
        / "docs/v2/decisions/0171-cross-host-materialization-classifier-completeness.md",
        DEFAULT_PREREGISTRATION,
    )
    return {str(path.relative_to(REPO_ROOT)): _checksum(path) for path in paths}


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


def _swap_delta(before: int | None, after: int | None) -> int | None:
    return None if before is None or after is None else after - before


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _report(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }


def _validated_identity(report: dict[str, Any]) -> dict[str, Any]:
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or not isinstance(identity, dict)
        or report.get("report_id") != _canonical_blake3(identity)
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("protocol_id") != PROTOCOL_ID
        or identity.get("adr") != ADR_ID
    ):
        raise QualificationError("qualification report identity is invalid")
    return identity


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


def _read_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read report {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"report is not an object: {path}")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--r3-cache", type=Path, default=DEFAULT_R3_CACHE)
    parser.add_argument("--s1-cache", type=Path, default=DEFAULT_S1_CACHE)
    parser.add_argument(
        "--relational-cache",
        type=Path,
        default=DEFAULT_RELATIONAL_CACHE,
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    parser.add_argument("--proof-id")
    parser.add_argument("--rows")
    parser.add_argument("--output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser(
        "compare",
        help="compare exact materialization tensors and latency",
    )
    _common_paths(compare)
    compare.add_argument("--split", choices=tuple(EXPECTED_COUNTS), required=True)
    compare.add_argument("--order", choices=tuple(ORDERS), required=True)

    prediction = subparsers.add_parser(
        "predict",
        help="replay frozen C0 predictions over both paths",
    )
    _common_paths(prediction)
    prediction.add_argument("--run-dir", type=Path, default=DEFAULT_C0_RUN)
    prediction.add_argument(
        "--candidate-chunk",
        type=int,
        default=CANDIDATE_CHUNK,
    )

    classify = subparsers.add_parser(
        "classify",
        help="apply the frozen qualification gates",
    )
    classify.add_argument(
        "--comparison",
        type=Path,
        action="append",
        required=True,
    )
    classify.add_argument("--prediction", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "compare":
        report = compare_materialization(
            split=args.split,
            order=args.order,
            rows=_parse_rows(args.rows),
            r3_cache=args.r3_cache.resolve(),
            s1_cache=args.s1_cache.resolve(),
            relational_cache=args.relational_cache.resolve(),
            train_dataset=args.train_dataset.resolve(),
            validation_dataset=args.validation_dataset.resolve(),
            expected_proof_id=args.proof_id,
        )
    elif args.command == "predict":
        report = compare_predictions(
            rows=_parse_rows(args.rows),
            run_dir=args.run_dir.resolve(),
            r3_cache=args.r3_cache.resolve(),
            s1_cache=args.s1_cache.resolve(),
            relational_cache=args.relational_cache.resolve(),
            train_dataset=args.train_dataset.resolve(),
            validation_dataset=args.validation_dataset.resolve(),
            expected_proof_id=args.proof_id,
            candidate_chunk=args.candidate_chunk,
        )
    else:
        report = classify_reports(
            [_read_report(path) for path in args.comparison],
            _read_report(args.prediction),
        )
    _write_report(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
