#!/usr/bin/env python3
"""Deterministically classify the four-run ADR 0146 R2 MLX tournament."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3
import numpy as np
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_OWNERSHIP_ENCODING,
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    EXPECTED_ACTIVE_TOKENS,
    EXPECTED_LAYER_MAXIMA,
    EXPECTED_SPLIT_RECORDS,
    EXPECTED_TYPE_TOKEN_TOTALS,
    EXPERIMENT_ID,
    FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS,
    FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS,
    GRAPH_MAX_DEGREE,
    TOKEN_CAPACITY,
    TOKEN_TYPE_NAMES,
)
from cascadia_mlx.r2_sparse_mlx_model import architecture_parameter_counts
from cascadia_mlx.r2_sparse_mlx_tournament import (
    ADR_ID,
    AUTHORIZED_RUNS,
    BATCH_SIZE,
    MAX_PARAMETER_SPREAD_FRACTION,
    PRIMARY_RUNS,
    PROTOCOL_ID,
    RUN_ARCHITECTURES,
    TRAINING_STEPS,
    VALIDATION_PROBE_ROWS,
    report_scientific_identity,
)

R0_BINDING_CONTRACT = "r2-sparse-mlx-r0-control-binding-v1"
R0_COMPLETE_CLASSIFICATION = "r0_spatial_mlx_tournament_complete"
R0_EXACT_CONTROL = "exact-entity-control"

MAX_R0_TOTAL_MAE_DELTA = 1.0
MAX_R0_TOTAL_RMSE_DELTA = 1.5
MAX_R0_MEAN_COMPONENT_MAE_DELTA = 0.25
MAX_REPLAY_TOTAL_MAE_DELTA = 0.10
MAX_REPLAY_TOTAL_RMSE_DELTA = 0.15
MAX_REPLAY_MEAN_COMPONENT_MAE_DELTA = 0.03
MAX_REPLAY_COMPONENT_PREDICTION_DELTA = 0.10

CLASSIFICATION_COMPLETE = "r2_sparse_mlx_tournament_complete"
CLASSIFICATION_SEMANTIC_FAILURE = "r2_sparse_mlx_tournament_semantic_failure"
CLASSIFICATION_INCOMPLETE = "r2_sparse_mlx_tournament_incomplete"
CLASSIFICATION_INSUFFICIENT_PERFORMANCE = (
    "r2_sparse_mlx_tournament_insufficient_performance_evidence"
)
CLASSIFICATION_REPLAY_FAILURE = "r2_sparse_mlx_tournament_replay_failure"
EXIT_CODES = {
    CLASSIFICATION_COMPLETE: 0,
    CLASSIFICATION_SEMANTIC_FAILURE: 2,
    CLASSIFICATION_INCOMPLETE: 3,
    CLASSIFICATION_INSUFFICIENT_PERFORMANCE: 4,
    CLASSIFICATION_REPLAY_FAILURE: 5,
}


def classify_reports(
    reports: list[dict[str, Any]],
    r0_control: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Normalize reports, apply fail-closed gates, and return a stable aggregate."""
    semantic_errors: list[str] = []
    structural_errors: list[str] = []
    performance_errors: list[str] = []
    replay_errors: list[str] = []

    r0_identity = _validate_r0_control_binding(r0_control, structural_errors)
    by_role: dict[str, dict[str, Any]] = {}
    for report in reports:
        role = report.get("run_role")
        if not isinstance(role, str):
            structural_errors.append("report lacks a string run role")
            continue
        if role in by_role:
            structural_errors.append(f"duplicate report for run role {role}")
            continue
        by_role[role] = report
    missing = sorted(set(AUTHORIZED_RUNS) - set(by_role))
    extra = sorted(set(by_role) - set(AUTHORIZED_RUNS))
    if missing:
        structural_errors.append(f"missing required run roles: {missing}")
    if extra:
        structural_errors.append(f"unexpected run roles: {extra}")

    normalized: list[dict[str, Any]] = []
    controls: dict[str, Any] | None = None
    semantic_reference: dict[str, Any] | None = None
    observed_primary_counts: dict[str, int] = {}
    for role in AUTHORIZED_RUNS:
        report = by_role.get(role)
        if report is None:
            continue
        _validate_report_envelope(report, role, structural_errors)
        _validate_report_semantics(report, role, semantic_errors)
        _validate_report_performance(report, role, performance_errors)
        report_controls = _controlled_identity(report)
        if controls is None:
            controls = report_controls
        elif report_controls != controls:
            structural_errors.append(f"controlled training identity drifted for {role}")
        report_semantics = _semantic_identity(report)
        if semantic_reference is None:
            semantic_reference = report_semantics
        elif report_semantics != semantic_reference:
            semantic_errors.append(f"cache semantic or target identity drifted for {role}")
        architecture = RUN_ARCHITECTURES[role]
        if role in PRIMARY_RUNS:
            observed_primary_counts[architecture] = report.get("model", {}).get(
                "parameter_count"
            )
        normalized.append(_normalized_run(report))

    _validate_parameter_match(observed_primary_counts, structural_errors)
    replay = _replay_comparison(by_role, replay_errors)
    comparisons: list[dict[str, Any]] = []
    selected_role: str | None = None
    if r0_identity is not None:
        for role in PRIMARY_RUNS:
            report = by_role.get(role)
            if report is None:
                continue
            try:
                comparisons.append(_r0_comparison(report, r0_identity))
            except (KeyError, TypeError):
                structural_errors.append(
                    f"{role} cannot produce the preregistered R0 comparison"
                )
        eligible = [
            comparison
            for comparison in comparisons
            if comparison["r0_value_noninferior"]
        ]
        if eligible and not replay_errors:
            selected_role = min(
                eligible,
                key=lambda value: (
                    value["validation_total_mae"],
                    value["validation_mean_component_mae"],
                    value["inference_latency_p50_ms"],
                    value["architecture"],
                ),
            )["run_role"]

    if semantic_errors:
        classification = CLASSIFICATION_SEMANTIC_FAILURE
    elif structural_errors:
        classification = CLASSIFICATION_INCOMPLETE
    elif performance_errors:
        classification = CLASSIFICATION_INSUFFICIENT_PERFORMANCE
    elif replay_errors:
        classification = CLASSIFICATION_REPLAY_FAILURE
    else:
        classification = CLASSIFICATION_COMPLETE

    selected_architecture = (
        RUN_ARCHITECTURES[selected_role] if selected_role is not None else None
    )
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "semantic_errors": sorted(set(semantic_errors)),
        "structural_errors": sorted(set(structural_errors)),
        "performance_errors": sorted(set(performance_errors)),
        "replay_errors": sorted(set(replay_errors)),
        "gates": {
            "max_parameter_spread_fraction": MAX_PARAMETER_SPREAD_FRACTION,
            "max_r0_total_mae_delta": MAX_R0_TOTAL_MAE_DELTA,
            "max_r0_total_rmse_delta": MAX_R0_TOTAL_RMSE_DELTA,
            "max_r0_mean_component_mae_delta": (
                MAX_R0_MEAN_COMPONENT_MAE_DELTA
            ),
            "max_replay_total_mae_delta": MAX_REPLAY_TOTAL_MAE_DELTA,
            "max_replay_total_rmse_delta": MAX_REPLAY_TOTAL_RMSE_DELTA,
            "max_replay_mean_component_mae_delta": (
                MAX_REPLAY_MEAN_COMPONENT_MAE_DELTA
            ),
            "max_replay_component_prediction_delta": (
                MAX_REPLAY_COMPONENT_PREDICTION_DELTA
            ),
        },
        "r0_control": r0_identity,
        "runs": normalized,
        "replay_comparison": replay,
        "r0_comparisons": comparisons,
        "selected_run_role": selected_role,
        "selected_architecture": selected_architecture,
        "claims": {
            "matched_architecture_screen_complete": (
                classification == CLASSIFICATION_COMPLETE
            ),
            "r0_value_reference_applied": r0_identity is not None,
            "independent_replay_passed": not replay_errors and replay is not None,
            "action_ranking_measured": False,
            "retained_regret_measured": False,
            "paired_gameplay_measured": False,
            "production_training_authorized": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    output = {**identity, "aggregate_id": canonical_blake3(identity)}
    return output, EXIT_CODES[classification]


def load_collection(path: Path, *, reverse: bool = False) -> list[dict[str, Any]]:
    collection = _read_json(path, "collection")
    entries = collection.get("reports")
    if (
        collection.get("schema_version") != 1
        or collection.get("experiment_id") != EXPERIMENT_ID
        or collection.get("adr") != ADR_ID
        or not isinstance(entries, list)
    ):
        raise ValueError("R2 MLX collection manifest is invalid")
    paths = [path.parent / str(entry.get("file", "")) for entry in entries]
    if reverse:
        paths.reverse()
    return [_read_json(report_path, "run report") for report_path in paths]


def _validate_r0_control_binding(
    binding: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    identity = binding.get("identity")
    if (
        binding.get("schema_version") != 1
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("adr") != ADR_ID
        or binding.get("contract_id") != R0_BINDING_CONTRACT
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != binding.get("binding_id")
    ):
        errors.append("R0 control binding is absent, malformed, or content-address drifted")
        return None
    if (
        identity.get("r0_classification") != R0_COMPLETE_CLASSIFICATION
        or identity.get("classification_order_byte_identical") is not True
        or not _is_digest(identity.get("r0_classification_aggregate_id"))
        or not _is_digest(identity.get("r0_classification_file_blake3"))
        or not _is_digest(identity.get("r0_order_proof_file_blake3"))
        or not _is_digest(identity.get("r0_control_report_id"))
        or not _is_digest(identity.get("r0_control_report_file_blake3"))
        or identity.get("selected_control_arm") not in {
            R0_EXACT_CONTROL,
            "hex-radius-6-127",
            "hex-radius-5-91",
            "hex-radius-4-61",
        }
        or not isinstance(identity.get("validation"), dict)
        or not _all_finite(identity["validation"])
    ):
        errors.append("R0 selected control is incomplete or scientifically invalid")
        return None
    selected = identity.get("r0_selected_stage2_candidate")
    expected_control = selected if selected is not None else R0_EXACT_CONTROL
    if identity.get("selected_control_arm") != expected_control:
        errors.append("R0 null-selection fallback did not bind exact-entity-control")
        return None
    return {
        "binding_id": binding["binding_id"],
        "r0_classification_aggregate_id": identity[
            "r0_classification_aggregate_id"
        ],
        "r0_selected_stage2_candidate": selected,
        "selected_control_arm": identity["selected_control_arm"],
        "r0_control_report_id": identity["r0_control_report_id"],
        "validation": identity["validation"],
    }


def _validate_report_envelope(
    report: dict[str, Any],
    role: str,
    errors: list[str],
) -> None:
    architecture = RUN_ARCHITECTURES[role]
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("adr") != ADR_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("run_role") != role
        or report.get("architecture") != architecture
    ):
        errors.append(f"report envelope drifted for {role}")
    scientific_identity = report.get("scientific_identity")
    try:
        expected_identity = report_scientific_identity(report)
    except (KeyError, TypeError):
        expected_identity = None
    if (
        not isinstance(scientific_identity, dict)
        or scientific_identity != expected_identity
        or canonical_blake3(scientific_identity) != report.get("report_id")
    ):
        errors.append(f"report content address drifted for {role}")
    model = report.get("model", {})
    expected_count = architecture_parameter_counts()[architecture]
    if (
        model.get("config", {}).get("architecture") != architecture
        or model.get("parameter_count") != expected_count
        or model.get("parameter_count_scope")
        != report.get("protocol", {}).get("parameter_count_scope")
        or model.get("state_encoder_invocations_per_prediction") != 1
    ):
        errors.append(f"model architecture, capacity, or encoder count drifted for {role}")
    optimization = report.get("optimization", {})
    if (
        optimization.get("global_step") != TRAINING_STEPS
        or optimization.get("training_examples") != TRAINING_STEPS * BATCH_SIZE
    ):
        errors.append(f"optimizer budget drifted for {role}")
    metrics = report.get("metrics", {})
    if metrics.get("train", {}).get("samples") != EXPECTED_SPLIT_RECORDS["train"]:
        errors.append(f"train evaluation coverage drifted for {role}")
    if (
        metrics.get("validation", {}).get("samples")
        != EXPECTED_SPLIT_RECORDS["validation"]
    ):
        errors.append(f"validation evaluation coverage drifted for {role}")
    ablations = metrics.get("validation_type_ablations")
    if (
        not isinstance(ablations, dict)
        or set(ablations) != set(TOKEN_TYPE_NAMES.values())
        or any(
            not isinstance(value, dict)
            or value.get("samples") != EXPECTED_SPLIT_RECORDS["validation"]
            or value.get("masked_token_type") != token_type
            for token_type, name in TOKEN_TYPE_NAMES.items()
            for value in [ablations.get(name)]
        )
    ):
        errors.append(f"validation type-ablation coverage drifted for {role}")
    probe = metrics.get("validation_probe", {})
    try:
        probe_values = np.asarray(probe.get("predictions"), dtype="<f4")
        probe_blake3 = blake3.blake3(
            probe_values.tobytes(order="C")
        ).hexdigest()
    except (TypeError, ValueError):
        probe_values = np.empty((0, 0), dtype="<f4")
        probe_blake3 = None
    if (
        probe.get("rows") != VALIDATION_PROBE_ROWS
        or probe.get("indices") != list(range(VALIDATION_PROBE_ROWS))
        or probe_values.shape != (VALIDATION_PROBE_ROWS, 11)
        or probe.get("prediction_blake3") != probe_blake3
    ):
        errors.append(f"validation replay probe drifted for {role}")
    claims = report.get("claims", {})
    if (
        claims.get("matched_architecture_screen_complete") is not True
        or claims.get("gameplay_strength_measured") is not False
        or claims.get("production_model_selected") is not False
        or claims.get("promotion_authorized") is not False
        or claims.get("progress_to_100_claimed") is not False
    ):
        errors.append(f"scientific claims drifted for {role}")
    if not _all_finite(report):
        errors.append(f"report contains a nonfinite numeric value for {role}")


def _validate_report_semantics(
    report: dict[str, Any],
    role: str,
    errors: list[str],
) -> None:
    integrity = report.get("integrity", {})
    for field in (
        "cache_verified",
        "exact_no_truncation_verified",
        "padding_zero_verified",
        "board_local_layout_verified",
        "graph_degree_bound_verified",
        "d6_regeneration_verified_by_exporter",
        "derived_relations_loaded_from_content_addressed_cache",
        "state_trunk_encoded_once",
        "type_balanced_pooling_verified",
        "typewise_ablation_reported",
        "all_metrics_finite",
    ):
        if integrity.get(field) is not True:
            errors.append(f"{role} failed semantic integrity field {field}")
    if integrity.get("test_or_final_data_opened") is not False:
        errors.append(f"{role} opened prohibited test or final data")
    cache = report.get("cache", {})
    if (
        cache.get("token_capacity") != TOKEN_CAPACITY
        or cache.get("board_slots") != BOARD_SLOTS
        or cache.get("board_token_capacity") != BOARD_TOKEN_CAPACITY
        or cache.get("graph_max_degree") != GRAPH_MAX_DEGREE
        or cache.get("board_ownership_encoding") != BOARD_OWNERSHIP_ENCODING
        or not isinstance(cache.get("active_token_statistics"), dict)
    ):
        errors.append(f"{role} changed the exact R2 tensor capacity")
    else:
        _validate_active_token_statistics(
            cache["active_token_statistics"],
            role,
            errors,
        )
    for field in (
        "cache_id",
        "corpus_lock_id",
        "identity_semantic_blake3",
        "d6_semantic_blake3",
        "target_blake3",
    ):
        if not _is_digest(cache.get(field)):
            errors.append(f"{role} has an invalid {field}")


def _validate_report_performance(
    report: dict[str, Any],
    role: str,
    errors: list[str],
) -> None:
    optimization = report.get("optimization", {})
    performance = report.get("performance", {})
    training_step = performance.get("training_step", {})
    positive = {
        "cumulative training seconds": optimization.get("training_seconds"),
        "cumulative training examples/s": optimization.get(
            "training_examples_per_second"
        ),
        "compile seconds": performance.get("compile_seconds"),
        "warmup examples/s": performance.get("warmup_examples_per_second"),
        "steady examples/s": performance.get("steady_examples_per_second"),
        "inference actions/s": performance.get("inference_actions_per_second"),
        "gradient examples/s": training_step.get("examples_per_second"),
    }
    for label, value in positive.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(f"{role} lacks positive {label}")
    latency = performance.get("latency_milliseconds", {})
    for percentile in ("p50", "p90", "p99"):
        value = latency.get(percentile)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(f"{role} lacks positive inference latency {percentile}")
    for field in (
        "inference_peak_active_memory_bytes",
        "training_peak_active_memory_bytes",
        "peak_process_rss_bytes",
    ):
        value = performance.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{role} has invalid memory evidence: {field}")


def _validate_parameter_match(
    counts: dict[str, int],
    errors: list[str],
) -> None:
    expected = architecture_parameter_counts()
    if counts != expected:
        errors.append("primary architecture parameter counts drifted")
        return
    values = list(counts.values())
    spread = (max(values) - min(values)) / min(values)
    if spread > MAX_PARAMETER_SPREAD_FRACTION:
        errors.append("primary architecture parameter spread exceeds three percent")


def _controlled_identity(report: dict[str, Any]) -> dict[str, Any]:
    runtime = report.get("runtime", {})
    return {
        "authorization_id": report.get("authorization", {}).get("authorization_id"),
        "r0_control_binding_id": report.get("authorization", {}).get(
            "r0_control_binding_id"
        ),
        "protocol": report.get("protocol"),
        "corpus_lock_id": report.get("cache", {}).get("corpus_lock_id"),
        "source_v2_blake3": report.get("source", {}).get("v2_source_blake3"),
        "runtime": {
            "mlx_version": runtime.get("mlx_version"),
            "python_version": runtime.get("python_version"),
            "machine": runtime.get("machine"),
            "device_kind": _device_kind(runtime.get("device")),
        },
    }


def _semantic_identity(report: dict[str, Any]) -> dict[str, Any]:
    cache = report.get("cache", {})
    return {
        "cache_id": cache.get("cache_id"),
        "identity_semantic_blake3": cache.get("identity_semantic_blake3"),
        "d6_semantic_blake3": cache.get("d6_semantic_blake3"),
        "target_blake3": cache.get("target_blake3"),
    }


def _normalized_run(report: dict[str, Any]) -> dict[str, Any]:
    validation = report.get("metrics", {}).get("validation", {})
    performance = report.get("performance", {})
    return {
        "run_role": report.get("run_role"),
        "architecture": report.get("architecture"),
        "report_id": report.get("report_id"),
        "cache_id": report.get("cache", {}).get("cache_id"),
        "parameter_count": report.get("model", {}).get("parameter_count"),
        "token_accounting": report.get("cache", {}).get("active_token_statistics"),
        "validation": {
            "loss": validation.get("loss"),
            "mean_component_mae": validation.get("mean_component_mae"),
            "total_mae": validation.get("total_mae"),
            "total_rmse": validation.get("total_rmse"),
            "total_bias": validation.get("total_bias"),
            "total_correlation": validation.get("total_correlation"),
            "calibration_slope": validation.get("calibration_slope"),
            "calibration_intercept": validation.get("calibration_intercept"),
        },
        "validation_type_ablations": report.get("metrics", {}).get(
            "validation_type_ablations"
        ),
        "observed_training_examples_per_second": report.get(
            "optimization",
            {},
        ).get("training_examples_per_second"),
        "observed_inference_actions_per_second": performance.get(
            "inference_actions_per_second"
        ),
        "inference_latency_p50_ms": performance.get(
            "latency_milliseconds",
            {},
        ).get("p50"),
        "gradient_examples_per_second": performance.get("training_step", {}).get(
            "examples_per_second"
        ),
        "inference_peak_active_memory_bytes": performance.get(
            "inference_peak_active_memory_bytes"
        ),
        "training_peak_active_memory_bytes": performance.get(
            "training_peak_active_memory_bytes"
        ),
        "independent_replay": report.get("run_role") == "set-replay",
    }


def _replay_comparison(
    by_role: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any] | None:
    primary = by_role.get("set-primary")
    replay = by_role.get("set-replay")
    if primary is None or replay is None:
        errors.append("set primary or independent replay is unavailable")
        return None
    try:
        primary_validation = primary["metrics"]["validation"]
        replay_validation = replay["metrics"]["validation"]
        primary_probe = np.asarray(
            primary["metrics"]["validation_probe"]["predictions"],
            dtype=np.float64,
        )
        replay_probe = np.asarray(
            replay["metrics"]["validation_probe"]["predictions"],
            dtype=np.float64,
        )
        if primary_probe.shape != (VALIDATION_PROBE_ROWS, 11):
            raise ValueError("primary probe shape drifted")
        if replay_probe.shape != primary_probe.shape:
            raise ValueError("replay probe shape drifted")
        max_prediction_delta = float(np.max(np.abs(primary_probe - replay_probe)))
        total_mae_delta = abs(
            replay_validation["total_mae"] - primary_validation["total_mae"]
        )
        total_rmse_delta = abs(
            replay_validation["total_rmse"] - primary_validation["total_rmse"]
        )
        component_delta = abs(
            replay_validation["mean_component_mae"]
            - primary_validation["mean_component_mae"]
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"independent replay evidence is malformed: {error}")
        return None
    passed = (
        total_mae_delta <= MAX_REPLAY_TOTAL_MAE_DELTA
        and total_rmse_delta <= MAX_REPLAY_TOTAL_RMSE_DELTA
        and component_delta <= MAX_REPLAY_MEAN_COMPONENT_MAE_DELTA
        and max_prediction_delta <= MAX_REPLAY_COMPONENT_PREDICTION_DELTA
    )
    if not passed:
        errors.append("independent Set Transformer replay exceeded frozen tolerances")
    return {
        "architecture": "padded-set-transformer",
        "primary_report_id": primary.get("report_id"),
        "replay_report_id": replay.get("report_id"),
        "validation_total_mae_absolute_delta": total_mae_delta,
        "validation_total_rmse_absolute_delta": total_rmse_delta,
        "validation_mean_component_mae_absolute_delta": component_delta,
        "validation_probe_max_component_absolute_delta": max_prediction_delta,
        "passed": passed,
    }


def _validate_active_token_statistics(
    statistics: dict[str, Any],
    role: str,
    errors: list[str],
) -> None:
    observed_active = 0
    observed_type_totals = {name: 0 for name in TOKEN_TYPE_NAMES.values()}
    observed_type_maxima = {name: 0 for name in TOKEN_TYPE_NAMES.values()}
    observed_position_maximum = 0
    observed_board_maximum = 0
    for split, records in EXPECTED_SPLIT_RECORDS.items():
        values = statistics.get(split)
        if not isinstance(values, dict):
            errors.append(f"{role} lacks {split} active-token accounting")
            continue
        active = values.get("active_tokens_total")
        padding = values.get("padding_tokens_total")
        mean = values.get("active_tokens_mean")
        maximum = values.get("active_tokens_max")
        board_maximum = values.get("active_tokens_max_per_board")
        type_tokens = values.get("type_tokens")
        if (
            values.get("records") != records
            or values.get("padded_capacity_per_position") != TOKEN_CAPACITY
            or values.get("board_slots") != BOARD_SLOTS
            or values.get("padded_capacity_per_board") != BOARD_TOKEN_CAPACITY
            or values.get("foundation_per_board_p99_active_tokens")
            != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
            or values.get("foundation_per_board_max_active_tokens")
            != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
            or not isinstance(active, int)
            or isinstance(active, bool)
            or active <= 0
            or not isinstance(padding, int)
            or isinstance(padding, bool)
            or padding < 0
            or active + padding != records * TOKEN_CAPACITY
            or not isinstance(mean, (int, float))
            or isinstance(mean, bool)
            or not math.isclose(mean, active / records, rel_tol=0.0, abs_tol=1e-12)
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not 0 < maximum <= 340
            or not isinstance(board_maximum, int)
            or isinstance(board_maximum, bool)
            or not 0 < board_maximum <= BOARD_TOKEN_CAPACITY
            or not isinstance(type_tokens, dict)
            or set(type_tokens) != set(TOKEN_TYPE_NAMES.values())
        ):
            errors.append(f"{role} has invalid {split} active-token accounting")
            continue
        type_total = 0
        for name in TOKEN_TYPE_NAMES.values():
            type_values = type_tokens[name]
            if not isinstance(type_values, dict):
                errors.append(f"{role} has invalid {split} {name} token accounting")
                continue
            total = type_values.get("total")
            type_mean = type_values.get("mean_per_position")
            fraction = type_values.get("fraction_of_active")
            type_maximum = type_values.get("maximum_per_position")
            if (
                not isinstance(total, int)
                or isinstance(total, bool)
                or total <= 0
                or not isinstance(type_mean, (int, float))
                or isinstance(type_mean, bool)
                or not math.isclose(
                    type_mean,
                    total / records,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not math.isclose(
                    fraction,
                    total / active,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not isinstance(type_maximum, int)
                or isinstance(type_maximum, bool)
                or type_maximum <= 0
            ):
                errors.append(f"{role} has invalid {split} {name} token accounting")
            type_total += total if isinstance(total, int) else 0
            if isinstance(total, int) and not isinstance(total, bool):
                observed_type_totals[name] += total
            if isinstance(type_maximum, int) and not isinstance(
                type_maximum,
                bool,
            ):
                observed_type_maxima[name] = max(
                    observed_type_maxima[name],
                    type_maximum,
                )
        if type_total != active:
            errors.append(f"{role} {split} type totals do not equal active tokens")
        if isinstance(active, int) and not isinstance(active, bool):
            observed_active += active
        if isinstance(maximum, int) and not isinstance(maximum, bool):
            observed_position_maximum = max(observed_position_maximum, maximum)
        if isinstance(board_maximum, int) and not isinstance(
            board_maximum,
            bool,
        ):
            observed_board_maximum = max(
                observed_board_maximum,
                board_maximum,
            )

    expected_totals = dict(
        zip(
            TOKEN_TYPE_NAMES.values(),
            EXPECTED_TYPE_TOKEN_TOTALS,
            strict=True,
        )
    )
    expected_maxima = dict(
        zip(
            TOKEN_TYPE_NAMES.values(),
            EXPECTED_LAYER_MAXIMA[:4],
            strict=True,
        )
    )
    if (
        observed_active != EXPECTED_ACTIVE_TOKENS
        or observed_type_totals != expected_totals
        or observed_type_maxima != expected_maxima
        or observed_position_maximum != EXPECTED_LAYER_MAXIMA[4]
        or observed_board_maximum != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
    ):
        errors.append(f"{role} active-token census differs from ADR 0145")


def _r0_comparison(
    report: dict[str, Any],
    r0_control: dict[str, Any],
) -> dict[str, Any]:
    validation = report["metrics"]["validation"]
    reference = r0_control["validation"]
    total_mae_delta = validation["total_mae"] - reference["total_mae"]
    total_rmse_delta = validation["total_rmse"] - reference["total_rmse"]
    component_delta = (
        validation["mean_component_mae"] - reference["mean_component_mae"]
    )
    return {
        "run_role": report["run_role"],
        "architecture": report["architecture"],
        "validation_total_mae": validation["total_mae"],
        "validation_total_rmse": validation["total_rmse"],
        "validation_mean_component_mae": validation["mean_component_mae"],
        "validation_total_mae_delta_vs_r0": total_mae_delta,
        "validation_total_rmse_delta_vs_r0": total_rmse_delta,
        "validation_mean_component_mae_delta_vs_r0": component_delta,
        "inference_latency_p50_ms": report["performance"]["latency_milliseconds"][
            "p50"
        ],
        "r0_value_noninferior": (
            total_mae_delta <= MAX_R0_TOTAL_MAE_DELTA
            and total_rmse_delta <= MAX_R0_TOTAL_RMSE_DELTA
            and component_delta <= MAX_R0_MEAN_COMPONENT_MAE_DELTA
        ),
        "promotion_authorized": False,
    }


def _device_kind(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    if "gpu" in lowered:
        return "gpu"
    if "cpu" in lowered:
        return "cpu"
    return lowered


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_blake3(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return blake3.blake3(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", type=Path)
    source.add_argument("--report", type=Path, action="append")
    parser.add_argument("--r0-control", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.collection is not None:
            reports = load_collection(
                args.collection,
                reverse=args.order == "reverse",
            )
        else:
            paths = list(args.report)
            if args.order == "reverse":
                paths.reverse()
            reports = [_read_json(path, "run report") for path in paths]
        r0_control = _read_json(args.r0_control, "R0 control binding")
        output, exit_code = classify_reports(reports, r0_control)
    except ValueError as error:
        identity = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "adr": ADR_ID,
            "protocol_id": PROTOCOL_ID,
            "classification": CLASSIFICATION_INCOMPLETE,
            "semantic_errors": [],
            "structural_errors": [str(error)],
            "performance_errors": [],
            "replay_errors": [],
            "gates": {},
            "r0_control": None,
            "runs": [],
            "replay_comparison": None,
            "r0_comparisons": [],
            "selected_run_role": None,
            "selected_architecture": None,
            "claims": {
                "matched_architecture_screen_complete": False,
                "r0_value_reference_applied": False,
                "independent_replay_passed": False,
                "action_ranking_measured": False,
                "retained_regret_measured": False,
                "paired_gameplay_measured": False,
                "production_training_authorized": False,
                "promotion_authorized": False,
                "progress_to_100_claimed": False,
            },
        }
        output = {**identity, "aggregate_id": canonical_blake3(identity)}
        exit_code = EXIT_CODES[CLASSIFICATION_INCOMPLETE]
    _write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
