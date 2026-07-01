#!/usr/bin/env python3
"""Deterministically classify the five-arm ADR 0142 R0 MLX tournament."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.r0_spatial_mlx_cache import ARM_TOKEN_CAPACITY, EXPERIMENT_ID
from cascadia_mlx.r0_spatial_mlx_tournament import (
    ADR_ID,
    BATCH_SIZE,
    PROTOCOL_ID,
    TRAINING_STEPS,
)

ARM_ORDER = tuple(ARM_TOKEN_CAPACITY)
CONTROL_ARM = "exact-entity-control"
DIAGNOSTIC_ARM = "historical-square-21x21-441"
COMPACT_ARMS = tuple(arm for arm in ARM_ORDER if arm not in {CONTROL_ARM, DIAGNOSTIC_ARM})
MAX_TOTAL_MAE_DELTA = 1.0
MAX_TOTAL_RMSE_DELTA = 1.5
MAX_MEAN_COMPONENT_MAE_DELTA = 0.25
MIN_INFERENCE_SPEEDUP = 1.5
MIN_TRAINING_SPEEDUP = 1.3

CLASSIFICATION_COMPLETE = "r0_spatial_mlx_tournament_complete"
CLASSIFICATION_SEMANTIC_FAILURE = "r0_spatial_mlx_tournament_semantic_failure"
CLASSIFICATION_INCOMPLETE = "r0_spatial_mlx_tournament_incomplete"
CLASSIFICATION_INSUFFICIENT_PERFORMANCE = (
    "r0_spatial_mlx_tournament_insufficient_performance_evidence"
)
EXIT_CODES = {
    CLASSIFICATION_COMPLETE: 0,
    CLASSIFICATION_SEMANTIC_FAILURE: 2,
    CLASSIFICATION_INCOMPLETE: 3,
    CLASSIFICATION_INSUFFICIENT_PERFORMANCE: 4,
}


def classify_reports(reports: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Normalize reports, apply fail-closed gates, and return a stable aggregate."""
    semantic_errors: list[str] = []
    structural_errors: list[str] = []
    performance_errors: list[str] = []

    by_arm: dict[str, dict[str, Any]] = {}
    for report in reports:
        arm = report.get("arm")
        if not isinstance(arm, str):
            structural_errors.append("report lacks a string arm ID")
            continue
        if arm in by_arm:
            structural_errors.append(f"duplicate report for arm {arm}")
            continue
        by_arm[arm] = report
    missing = sorted(set(ARM_ORDER) - set(by_arm))
    extra = sorted(set(by_arm) - set(ARM_ORDER))
    if missing:
        structural_errors.append(f"missing required arms: {missing}")
    if extra:
        structural_errors.append(f"unexpected arms: {extra}")

    normalized: list[dict[str, Any]] = []
    controls: dict[str, Any] | None = None
    semantic_reference: dict[str, Any] | None = None
    for arm in ARM_ORDER:
        report = by_arm.get(arm)
        if report is None:
            continue
        _validate_report_envelope(report, arm, structural_errors)
        _validate_report_semantics(report, arm, semantic_errors)
        _validate_report_performance(report, arm, performance_errors)
        report_controls = _controlled_identity(report)
        if controls is None:
            controls = report_controls
        elif report_controls != controls:
            structural_errors.append(f"controlled training identity drifted for {arm}")
        report_semantics = _semantic_identity(report)
        if semantic_reference is None:
            semantic_reference = report_semantics
        elif report_semantics != semantic_reference:
            semantic_errors.append(f"Rust semantic or target identity drifted for {arm}")
        normalized.append(_normalized_arm(report))

    comparisons: list[dict[str, Any]] = []
    selected_candidate: str | None = None
    if CONTROL_ARM in by_arm:
        control = by_arm[CONTROL_ARM]
        for arm in COMPACT_ARMS:
            report = by_arm.get(arm)
            if report is None:
                continue
            try:
                comparisons.append(_compact_comparison(report, control))
            except (KeyError, TypeError, ZeroDivisionError):
                performance_errors.append(
                    f"{arm} cannot produce the preregistered compact-arm comparison"
                )
        passing = [comparison for comparison in comparisons if comparison["stage2_candidate"]]
        if passing:
            selected_candidate = min(
                passing,
                key=lambda value: (
                    value["spatial_token_capacity"],
                    value["validation_total_mae"],
                    -value["same_host_inference_speedup"],
                    value["arm"],
                ),
            )["arm"]
    else:
        structural_errors.append("exact-entity-control is unavailable as the control")

    if semantic_errors:
        classification = CLASSIFICATION_SEMANTIC_FAILURE
    elif structural_errors:
        classification = CLASSIFICATION_INCOMPLETE
    elif performance_errors:
        classification = CLASSIFICATION_INSUFFICIENT_PERFORMANCE
    else:
        classification = CLASSIFICATION_COMPLETE

    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "semantic_errors": sorted(set(semantic_errors)),
        "structural_errors": sorted(set(structural_errors)),
        "performance_errors": sorted(set(performance_errors)),
        "gates": {
            "max_total_mae_delta": MAX_TOTAL_MAE_DELTA,
            "max_total_rmse_delta": MAX_TOTAL_RMSE_DELTA,
            "max_mean_component_mae_delta": MAX_MEAN_COMPONENT_MAE_DELTA,
            "min_same_host_inference_speedup": MIN_INFERENCE_SPEEDUP,
            "min_same_host_training_speedup": MIN_TRAINING_SPEEDUP,
        },
        "arms": normalized,
        "compact_comparisons": comparisons,
        "selected_stage2_candidate": selected_candidate,
        "historical441_diagnostic_only": True,
        "claims": {
            "iso_architecture_stage2_complete": classification == CLASSIFICATION_COMPLETE,
            "action_ranking_measured": False,
            "retained_regret_measured": False,
            "paired_gameplay_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    output = {
        **identity,
        "aggregate_id": canonical_blake3(identity),
    }
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
        raise ValueError("R0 MLX collection manifest is invalid")
    paths = [path.parent / str(entry.get("file", "")) for entry in entries]
    if reverse:
        paths.reverse()
    return [_read_json(report_path, "arm report") for report_path in paths]


def _validate_report_envelope(
    report: dict[str, Any],
    arm: str,
    errors: list[str],
) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("adr") != ADR_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("arm") != arm
    ):
        errors.append(f"report envelope drifted for {arm}")
    scientific_identity = report.get("scientific_identity")
    if not isinstance(scientific_identity, dict) or canonical_blake3(
        scientific_identity
    ) != report.get("report_id"):
        errors.append(f"report content address drifted for {arm}")
    cache = report.get("cache", {})
    if cache.get("spatial_token_capacity") != ARM_TOKEN_CAPACITY[arm]:
        errors.append(f"spatial token capacity drifted for {arm}")
    optimization = report.get("optimization", {})
    if (
        optimization.get("global_step") != TRAINING_STEPS
        or optimization.get("training_examples") != TRAINING_STEPS * BATCH_SIZE
    ):
        errors.append(f"optimizer budget drifted for {arm}")
    metrics = report.get("metrics", {})
    if metrics.get("train", {}).get("samples") != 50_000:
        errors.append(f"train evaluation coverage drifted for {arm}")
    if metrics.get("validation", {}).get("samples") != 10_000:
        errors.append(f"validation evaluation coverage drifted for {arm}")
    claims = report.get("claims", {})
    if (
        claims.get("learned_representation_screen_complete") is not True
        or claims.get("gameplay_strength_measured") is not False
        or claims.get("promotion_authorized") is not False
        or claims.get("progress_to_100_claimed") is not False
    ):
        errors.append(f"scientific claims drifted for {arm}")
    if not _all_finite(report):
        errors.append(f"report contains a nonfinite numeric value for {arm}")


def _validate_report_semantics(
    report: dict[str, Any],
    arm: str,
    errors: list[str],
) -> None:
    integrity = report.get("integrity", {})
    for field in (
        "cache_verified",
        "padding_verified",
        "semantic_round_trip_verified",
        "overflow_exact_entities_retained",
        "all_metrics_finite",
    ):
        if integrity.get(field) is not True:
            errors.append(f"{arm} failed semantic integrity field {field}")
    if integrity.get("test_or_final_data_opened") is not False:
        errors.append(f"{arm} opened prohibited test or final data")
    cache = report.get("cache", {})
    for field in ("source_semantic_blake3", "d6_semantic_blake3", "target_blake3"):
        if not _is_digest(cache.get(field)):
            errors.append(f"{arm} has an invalid {field}")


def _validate_report_performance(
    report: dict[str, Any],
    arm: str,
    errors: list[str],
) -> None:
    optimization = report.get("optimization", {})
    performance = report.get("performance", {})
    exact = performance.get("same_host_exact_shape_control", {})
    arm_training = performance.get("same_host_training_step", {})
    exact_training = performance.get("same_host_exact_shape_training_step", {})
    ratios = performance.get("same_host_shape_ratios", {})
    positive = {
        "cumulative training seconds": optimization.get("training_seconds"),
        "cumulative training examples/s": optimization.get("training_examples_per_second"),
        "compile seconds": performance.get("compile_seconds"),
        "warmup examples/s": performance.get("warmup_examples_per_second"),
        "steady examples/s": performance.get("steady_examples_per_second"),
        "inference actions/s": performance.get("inference_actions_per_second"),
        "exact-shape steady examples/s": exact.get("steady_examples_per_second"),
        "arm gradient examples/s": arm_training.get("examples_per_second"),
        "exact-shape gradient examples/s": exact_training.get("examples_per_second"),
        "same-host inference ratio": ratios.get("inference_examples_per_second"),
        "same-host training ratio": ratios.get("training_examples_per_second"),
    }
    for label, value in positive.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(f"{arm} lacks positive {label}")
    if performance.get("spatial_token_capacity") != ARM_TOKEN_CAPACITY[arm]:
        errors.append(f"{arm} performance shape does not match its cache")
    if exact.get("spatial_token_capacity") != ARM_TOKEN_CAPACITY[CONTROL_ARM]:
        errors.append(f"{arm} lacks the same-host 23-token exact-shape control")
    for field in (
        "inference_peak_active_memory_bytes",
        "training_peak_active_memory_bytes",
        "peak_process_rss_bytes",
    ):
        value = performance.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{arm} has invalid memory evidence: {field}")


def _controlled_identity(report: dict[str, Any]) -> dict[str, Any]:
    runtime = report.get("runtime", {})
    return {
        "authorization_id": report.get("authorization", {}).get("authorization_id"),
        "protocol": report.get("protocol"),
        "model": report.get("model"),
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
        "source_semantic_blake3": cache.get("source_semantic_blake3"),
        "d6_semantic_blake3": cache.get("d6_semantic_blake3"),
        "target_blake3": cache.get("target_blake3"),
    }


def _normalized_arm(report: dict[str, Any]) -> dict[str, Any]:
    validation = report.get("metrics", {}).get("validation", {})
    performance = report.get("performance", {})
    return {
        "arm": report.get("arm"),
        "report_id": report.get("report_id"),
        "cache_id": report.get("cache", {}).get("cache_id"),
        "spatial_token_capacity": report.get("cache", {}).get("spatial_token_capacity"),
        "parameter_count": report.get("model", {}).get("parameter_count"),
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
        "observed_training_examples_per_second": report.get("optimization", {}).get(
            "training_examples_per_second"
        ),
        "observed_inference_actions_per_second": performance.get("inference_actions_per_second"),
        "same_host_shape_ratios": performance.get("same_host_shape_ratios"),
        "inference_peak_active_memory_bytes": performance.get("inference_peak_active_memory_bytes"),
        "training_peak_active_memory_bytes": performance.get("training_peak_active_memory_bytes"),
        "diagnostic_only": report.get("arm") == DIAGNOSTIC_ARM,
    }


def _compact_comparison(
    report: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    arm_validation = report["metrics"]["validation"]
    control_validation = control["metrics"]["validation"]
    ratios = report["performance"]["same_host_shape_ratios"]
    total_mae_delta = arm_validation["total_mae"] - control_validation["total_mae"]
    total_rmse_delta = arm_validation["total_rmse"] - control_validation["total_rmse"]
    component_delta = (
        arm_validation["mean_component_mae"] - control_validation["mean_component_mae"]
    )
    value_noninferior = (
        total_mae_delta <= MAX_TOTAL_MAE_DELTA
        and total_rmse_delta <= MAX_TOTAL_RMSE_DELTA
        and component_delta <= MAX_MEAN_COMPONENT_MAE_DELTA
    )
    leverage = (
        ratios["inference_examples_per_second"] >= MIN_INFERENCE_SPEEDUP
        or ratios["training_examples_per_second"] >= MIN_TRAINING_SPEEDUP
    )
    return {
        "arm": report["arm"],
        "spatial_token_capacity": report["cache"]["spatial_token_capacity"],
        "validation_total_mae": arm_validation["total_mae"],
        "validation_total_mae_delta": total_mae_delta,
        "validation_total_rmse_delta": total_rmse_delta,
        "validation_mean_component_mae_delta": component_delta,
        "same_host_inference_speedup": ratios["inference_examples_per_second"],
        "same_host_training_speedup": ratios["training_examples_per_second"],
        "value_noninferior": value_noninferior,
        "throughput_gate_passed": leverage,
        "stage2_candidate": value_noninferior and leverage,
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
    if isinstance(value, list):
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
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", type=Path)
    source.add_argument("--report", type=Path, action="append")
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
            reports = [_read_json(path, "arm report") for path in paths]
        output, exit_code = classify_reports(reports)
    except ValueError as error:
        output = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "adr": ADR_ID,
            "protocol_id": PROTOCOL_ID,
            "classification": CLASSIFICATION_INCOMPLETE,
            "semantic_errors": [],
            "structural_errors": [str(error)],
            "performance_errors": [],
            "arms": [],
            "compact_comparisons": [],
            "selected_stage2_candidate": None,
            "historical441_diagnostic_only": True,
            "claims": {
                "iso_architecture_stage2_complete": False,
                "action_ranking_measured": False,
                "retained_regret_measured": False,
                "paired_gameplay_measured": False,
                "promotion_authorized": False,
                "progress_to_100_claimed": False,
            },
        }
        output["aggregate_id"] = canonical_blake3(output)
        exit_code = EXIT_CODES[CLASSIFICATION_INCOMPLETE]
    _write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
