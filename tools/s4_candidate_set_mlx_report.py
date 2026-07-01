#!/usr/bin/env python3
"""Validate and classify the ADR 0153 candidate-context comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "s4-candidate-context-mlx-comparison-v1"
PROTOCOL_ID = "s4-candidate-context-matched-comparison-v1"
ADR_ID = "0153"
ARMS = (
    "c0-independent",
    "t1-inducing-16",
    "t2-exact-relations",
    "t3-combined",
)
CONTROL = ARMS[0]
ARM_HOSTS = {
    ARMS[0]: "john1",
    ARMS[1]: "john2",
    ARMS[2]: "john3",
    ARMS[3]: "john4",
}
R3_EXPERIMENT_ID = "r3-action-edit-mlx-comparison-v1"
R3_PROTOCOL_ID = "r3-action-edit-mlx-matched-comparison-v1"
R3_ADR_ID = "0150"
R3_CONTROL = "c0-full-r2-afterstate"
R3_SUBSTRATE = "t3-r3-radius1-global"
R3_REQUIRED_CLASSIFICATION = "r3_action_edit_mlx_all_treatments_degraded"

INVALID = "s4_candidate_context_mlx_invalid_evidence"
CONTROL_FAILED = "s4_candidate_context_mlx_control_failed"
ALL_TREATMENTS_DEGRADED = (
    "s4_candidate_context_mlx_all_treatments_degraded"
)
CONTEXT_NULL = "s4_candidate_context_mlx_context_null"
CONTEXT_SIGNAL_ONLY = "s4_candidate_context_mlx_context_signal_only"
RESCUE_SELECTED = "s4_candidate_context_mlx_compact_rescue_selected"

MAX_MAE_DELTA = 0.05
MAX_RMSE_DELTA = 0.05
MIN_RECALL_DELTA = -0.005
MAX_REGRET_DELTA = 0.005
MIN_SLICE_RECALL_DELTA = -0.01
MIN_CONFIDENCE_COVERAGE = 0.99
MIN_MAE_REDUCTION = 0.05
MIN_RMSE_REDUCTION = 0.05
MIN_RECALL_GAIN = 0.02
MIN_REGRET_REDUCTION = 0.02
MIN_COVERAGE_GAIN = 0.01
MIN_R3_THROUGHPUT_FRACTION = 0.25
MAX_R3_P99_MULTIPLIER = 1.75
MAX_R3_ACTIVE_MEMORY_MULTIPLIER = 2.50
MAX_R3_RSS_MULTIPLIER = 1.50


class S4ReportError(ValueError):
    """S4 evidence is missing, inconsistent, or outside the frozen protocol."""


def classify_reports(
    paths: list[Path],
    *,
    r3_classification_path: Path,
    r3_control_path: Path,
    r3_substrate_path: Path,
) -> dict[str, Any]:
    """Apply matched-context, compact-rescue, and serving gates."""
    try:
        return _classify_reports(
            paths,
            r3_classification_path=r3_classification_path,
            r3_control_path=r3_control_path,
            r3_substrate_path=r3_substrate_path,
        )
    except S4ReportError:
        raise
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise S4ReportError(f"S4 report schema is incomplete: {error}") from error


def validate_r3_rescue_evidence(
    *,
    classification_path: Path,
    control_path: Path,
    substrate_path: Path,
) -> dict[str, Any]:
    """Validate and content-address the failed R3 substrate evidence."""
    classification = _read_r3_classification(classification_path)
    control = _read_r3_report(control_path, expected_arm=R3_CONTROL)
    substrate = _read_r3_report(
        substrate_path,
        expected_arm=R3_SUBSTRATE,
    )
    scientific = classification["scientific"]
    report_ids = scientific.get("arm_report_ids")
    if (
        not isinstance(report_ids, dict)
        or report_ids.get(R3_CONTROL) != control.get("report_id")
        or report_ids.get(R3_SUBSTRATE) != substrate.get("report_id")
        or scientific.get("cache_id") != control.get("cache_id")
        or scientific.get("cache_id") != substrate.get("cache_id")
        or scientific.get("s1_cache_id") != control.get("s1_cache_id")
        or scientific.get("s1_cache_id") != substrate.get("s1_cache_id")
    ):
        raise S4ReportError("R3 rescue evidence is internally inconsistent")
    identity = {
        "classification_id": classification["classification_id"],
        "scientific_blake3": classification["scientific_blake3"],
        "classification": classification["classification"],
        "cache_id": scientific["cache_id"],
        "s1_cache_id": scientific["s1_cache_id"],
        "control_report_id": control["report_id"],
        "substrate_report_id": substrate["report_id"],
        "control_checkpoint": control["checkpoint"],
        "substrate_checkpoint": substrate["checkpoint"],
    }
    return {
        "identity": identity,
        "evidence_id": canonical_blake3(identity),
        "classification": classification,
        "control": control,
        "substrate": substrate,
    }


def _classify_reports(
    paths: list[Path],
    *,
    r3_classification_path: Path,
    r3_control_path: Path,
    r3_substrate_path: Path,
) -> dict[str, Any]:
    if len(paths) != len(ARMS):
        raise S4ReportError("S4 classification requires exactly four arm reports")
    reports = [_read_arm_report(path) for path in paths]
    by_arm = {str(report["arm"]): report for report in reports}
    if set(by_arm) != set(ARMS) or len(by_arm) != len(ARMS):
        raise S4ReportError("S4 reports must contain each arm exactly once")
    _validate_shared_identity(by_arm)

    r3_evidence = validate_r3_rescue_evidence(
        classification_path=r3_classification_path,
        control_path=r3_control_path,
        substrate_path=r3_substrate_path,
    )
    r3_classification = r3_evidence["classification"]
    r3_control = r3_evidence["control"]
    r3_substrate = r3_evidence["substrate"]
    _validate_r3_binding(
        by_arm,
        classification=r3_classification,
        control=r3_control,
        substrate=r3_substrate,
    )

    gates: dict[str, dict[str, dict[str, bool]]] = {
        arm: {"absolute": _absolute_gates(report)}
        for arm, report in by_arm.items()
    }
    control_passed = all(gates[CONTROL]["absolute"].values())
    selected_arm: str | None = None
    if not control_passed:
        classification = CONTROL_FAILED
    else:
        noninferior: list[str] = []
        material: list[str] = []
        rescued: list[str] = []
        for arm in ARMS[1:]:
            report = by_arm[arm]
            quality = _quality_gates(report, by_arm[CONTROL])
            context = _material_context_gates(report, by_arm[CONTROL])
            external = _external_rescue_gates(report, r3_control)
            performance = _performance_rescue_gates(report, r3_control)
            gates[arm].update(
                {
                    "quality_noninferiority_to_s4_control": quality,
                    "material_context_effect": context,
                    "quality_rescue_to_r3_control": external,
                    "performance_vs_r3_control": performance,
                }
            )
            if all(gates[arm]["absolute"].values()) and all(quality.values()):
                noninferior.append(arm)
                if any(context.values()):
                    material.append(arm)
                    if all(external.values()) and all(performance.values()):
                        rescued.append(arm)
        if not noninferior:
            classification = ALL_TREATMENTS_DEGRADED
        elif not material:
            classification = CONTEXT_NULL
        elif not rescued:
            classification = CONTEXT_SIGNAL_ONLY
        else:
            classification = RESCUE_SELECTED
            selected_arm = _select_arm(rescued, by_arm)

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected_arm,
        "cache_id": by_arm[CONTROL]["cache_id"],
        "s1_cache_id": by_arm[CONTROL]["s1_cache_id"],
        "context_cache_id": by_arm[CONTROL]["context_cache_id"],
        "warm_start": by_arm[CONTROL]["warm_start"],
        "r3_classification_id": r3_classification["classification_id"],
        "r3_control_report_id": r3_control["report_id"],
        "r3_substrate_report_id": r3_substrate["report_id"],
        "arm_report_ids": {
            arm: by_arm[arm]["report_id"] for arm in ARMS
        },
        "thresholds": {
            "max_mae_delta": MAX_MAE_DELTA,
            "max_rmse_delta": MAX_RMSE_DELTA,
            "min_recall_delta": MIN_RECALL_DELTA,
            "max_regret_delta": MAX_REGRET_DELTA,
            "min_slice_recall_delta": MIN_SLICE_RECALL_DELTA,
            "min_confidence_coverage": MIN_CONFIDENCE_COVERAGE,
            "min_mae_reduction": MIN_MAE_REDUCTION,
            "min_rmse_reduction": MIN_RMSE_REDUCTION,
            "min_recall_gain": MIN_RECALL_GAIN,
            "min_regret_reduction": MIN_REGRET_REDUCTION,
            "min_coverage_gain": MIN_COVERAGE_GAIN,
            "min_r3_throughput_fraction": MIN_R3_THROUGHPUT_FRACTION,
            "max_r3_p99_multiplier": MAX_R3_P99_MULTIPLIER,
            "max_r3_active_memory_multiplier": (
                MAX_R3_ACTIVE_MEMORY_MULTIPLIER
            ),
            "max_r3_rss_multiplier": MAX_R3_RSS_MULTIPLIER,
        },
        "gates": {arm: gates[arm] for arm in ARMS},
    }
    output = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected_arm,
        "scientific": scientific,
        "scientific_blake3": canonical_blake3(scientific),
        "claims": {
            "offline_comparison_complete": True,
            "compact_representation_rescued": (
                classification == RESCUE_SELECTED
            ),
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    output["classification_id"] = canonical_blake3(
        {
            "schema_version": output["schema_version"],
            "scientific": output["scientific"],
            "scientific_blake3": output["scientific_blake3"],
            "claims": output["claims"],
        }
    )
    return output


def compare_classifications(forward: Path, reverse: Path) -> dict[str, Any]:
    """Prove order-invariant classification bytes."""
    forward_bytes = forward.read_bytes()
    reverse_bytes = reverse.read_bytes()
    if forward_bytes != reverse_bytes:
        raise S4ReportError(
            "forward and reverse S4 classifications are not byte-identical"
        )
    parsed = _read_json(forward, "S4 classification")
    if (
        parsed.get("classification") == INVALID
        or not _is_blake3(parsed.get("classification_id"))
    ):
        raise S4ReportError("S4 classification is malformed")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": parsed["classification"],
        "classification_id": parsed["classification_id"],
        "scientific_blake3": parsed["scientific_blake3"],
        "classification_file_blake3": blake3.blake3(
            forward_bytes
        ).hexdigest(),
        "forward_reverse_byte_identical": True,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "proof_id": canonical_blake3(identity),
        "scientific_identity": identity,
    }


def _read_arm_report(path: Path) -> dict[str, Any]:
    report = _read_json(path, f"S4 arm report {path}")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("mode") != "production"
        or report.get("arm") not in ARMS
        or report.get("host") != ARM_HOSTS.get(report.get("arm"))
        or not isinstance(identity, dict)
        or identity != _arm_scientific_identity(report)
        or canonical_blake3(identity) != report.get("report_id")
    ):
        raise S4ReportError(f"S4 arm report is malformed: {path}")
    boundary = report.get("information_boundary")
    claims = report.get("claims")
    parity = report.get("initial_prediction_parity")
    if (
        not isinstance(boundary, dict)
        or boundary.get("sealed_test_opened") is not False
        or boundary.get("gameplay_run") is not False
        or boundary.get("hidden_order_read") is not False
        or boundary.get("future_refill_read") is not False
        or not isinstance(claims, dict)
        or claims.get("offline_comparison_complete") is not True
        or claims.get("gameplay_strength_measured") is not False
        or claims.get("promotion_authorized") is not False
        or claims.get("progress_to_100_claimed") is not False
        or not isinstance(parity, dict)
        or parity.get("scores_byte_identical") is not True
        or parity.get("standard_errors_byte_identical") is not True
    ):
        raise S4ReportError(f"S4 claim or parity boundary failed: {path}")
    _validate_metrics(report["metrics"], label=str(report["arm"]))
    _validate_performance(report["performance"], label=str(report["arm"]))
    return report


def _validate_shared_identity(
    reports: dict[str, dict[str, Any]],
) -> None:
    control = reports[CONTROL]
    shared_fields = (
        "cache_id",
        "s1_cache_id",
        "context_cache_id",
        "protocol",
        "warm_start",
        "cross_arm_initialization",
    )
    control_source = control["source"].get("v2_source_blake3")
    control_authorization = control["controls"].get("authorization_id")
    schedule = _batch_schedule(control)
    if not _is_blake3(control_source) or not _is_blake3(
        control_authorization
    ):
        raise S4ReportError("S4 source or authorization identity is malformed")
    for arm, report in reports.items():
        if any(report.get(field) != control.get(field) for field in shared_fields):
            raise S4ReportError(f"S4 shared identity differs for {arm}")
        model = report["model"]
        controls = report["controls"]
        runtime = report["runtime"]
        if (
            report["source"].get("v2_source_blake3") != control_source
            or controls.get("authorization_id") != control_authorization
            or not _is_blake3(controls.get("preflight_id"))
            or runtime.get("machine") != "arm64"
            or "gpu" not in str(runtime.get("default_device", "")).lower()
            or report["optimization"].get("global_step") != 3000
            or model["config"].get("arm") != arm
            or model["config"].get("r3_arm") != R3_SUBSTRATE
            or model.get("parameter_count")
            != control["model"].get("parameter_count")
            or model.get("parameter_layout_blake3")
            != control["model"].get("parameter_layout_blake3")
            or model.get("initial_parameter_tensor_blake3")
            != control["model"].get("initial_parameter_tensor_blake3")
            or _batch_schedule(report) != schedule
        ):
            raise S4ReportError(f"S4 matched protocol differs for {arm}")


def _batch_schedule(report: dict[str, Any]) -> list[tuple[int, str, int]]:
    optimization = report.get("optimization")
    if not isinstance(optimization, dict):
        raise S4ReportError("S4 optimization evidence is absent")
    trace = optimization.get("loss_trace")
    if not isinstance(trace, list) or len(trace) != 3000:
        raise S4ReportError("S4 loss trace must contain exactly 3000 steps")
    schedule: list[tuple[int, str, int]] = []
    candidates = 0
    for expected_step, event in enumerate(trace, start=1):
        if (
            not isinstance(event, dict)
            or event.get("step") != expected_step
            or not _is_blake3(event.get("batch_blake3"))
            or _number(event.get("elapsed_seconds")) <= 0
        ):
            raise S4ReportError("S4 loss trace is malformed")
        count = int(_number(event.get("candidates")))
        _number(event.get("loss"))
        if count <= 0:
            raise S4ReportError("S4 loss trace candidate count is invalid")
        candidates += count
        schedule.append((expected_step, str(event["batch_blake3"]), count))
    if optimization.get("candidates") != candidates:
        raise S4ReportError("S4 candidate accounting differs from its trace")
    return schedule


def _validate_metrics(metrics: object, *, label: str) -> None:
    if not isinstance(metrics, dict):
        raise S4ReportError(f"S4 metrics are absent for {label}")
    if (
        metrics.get("groups") != 240
        or metrics.get("candidates") != 860_203
        or metrics.get("all_groups_scored_once") is not True
        or metrics.get("all_candidates_scored_once") is not True
        or metrics.get("all_scores_and_uncertainties_finite") is not True
        or metrics.get("parent_encodes") != 240
        or metrics.get("parent_encode_count_exact") is not True
    ):
        raise S4ReportError(f"S4 validation coverage failed for {label}")
    value = metrics.get("r4800_value")
    subsets = metrics.get("subsets")
    panel = metrics.get("prediction_panel")
    if (
        not isinstance(value, dict)
        or not isinstance(subsets, dict)
        or not isinstance(panel, dict)
        or panel.get("count") != 64
    ):
        raise S4ReportError(f"S4 required metrics are absent for {label}")
    for key in (
        "mae",
        "rmse",
        "bias",
        "correlation",
        "calibration_slope",
        "calibration_intercept",
    ):
        _number(value.get(key))
    for key in (
        "top64_r4800_winner_recall",
        "mean_top64_retained_r4800_regret",
        "top64_confidence_set_coverage_95",
    ):
        _number(metrics.get(key))
    for name in (
        "early",
        "middle",
        "late",
        "low_supply",
        "independent_draft_winner",
    ):
        subset = subsets.get(name)
        if not isinstance(subset, dict) or _number(subset.get("groups")) <= 0:
            raise S4ReportError(f"S4 subset {name} is absent for {label}")
        _number(subset.get("top64_r4800_winner_recall"))
        _number(subset.get("mean_top64_retained_r4800_regret"))
        _number(subset.get("top64_confidence_set_coverage_95"))


def _validate_performance(performance: object, *, label: str) -> None:
    if not isinstance(performance, dict):
        raise S4ReportError(f"S4 performance is absent for {label}")
    fixed = performance.get("fixed_chunk")
    decisions = performance.get("complete_decisions")
    memory = performance.get("memory")
    measurement = performance.get("measurement")
    if (
        not isinstance(fixed, dict)
        or not isinstance(decisions, dict)
        or not isinstance(memory, dict)
        or not isinstance(measurement, dict)
        or fixed.get("actions") != 256
        or fixed.get("warmup_iterations") != 5
        or fixed.get("steady_iterations") != 30
        or decisions.get("groups") != 20
        or decisions.get("parent_encodes") != 20
        or decisions.get("anchor_encodes") != 20
        or decisions.get("parent_encode_count_exact") is not True
        or decisions.get("anchor_encode_count_exact") is not True
        or measurement.get("isolated_process") is not True
        or measurement.get("verification_source") != "cluster-preflight"
        or not _is_blake3(measurement.get("request_id"))
        or not _is_blake3(measurement.get("result_id"))
        or not _is_blake3(measurement.get("checkpoint_model_blake3"))
        or not _is_blake3(measurement.get("open_data_verification_id"))
        or not _is_blake3(measurement.get("context_cache_id"))
    ):
        raise S4ReportError(f"S4 serving protocol differs for {label}")
    _number(fixed.get("action_scores_per_second"))
    _number(decisions.get("action_scores_per_second"))
    for section in (fixed, decisions):
        latency = section.get("latency_milliseconds")
        if not isinstance(latency, dict):
            raise S4ReportError(f"S4 latency is absent for {label}")
        for quantile in ("p50", "p95", "p99"):
            _number(latency.get(quantile))
    for key in (
        "peak_active_bytes",
        "peak_process_rss_bytes",
        "process_swaps",
    ):
        _number(memory.get(key))


def _read_r3_classification(path: Path) -> dict[str, Any]:
    value = _read_json(path, "R3 classification")
    scientific = value.get("scientific")
    claims = value.get("claims")
    if (
        value.get("schema_version") != 1
        or value.get("experiment_id") != R3_EXPERIMENT_ID
        or value.get("protocol_id") != R3_PROTOCOL_ID
        or value.get("adr") != R3_ADR_ID
        or value.get("classification") != R3_REQUIRED_CLASSIFICATION
        or value.get("selected_arm") is not None
        or not isinstance(scientific, dict)
        or canonical_blake3(scientific) != value.get("scientific_blake3")
        or not isinstance(claims, dict)
    ):
        raise S4ReportError("R3 classification cannot authorize S4 rescue")
    expected = canonical_blake3(
        {
            "schema_version": value["schema_version"],
            "scientific": scientific,
            "scientific_blake3": value["scientific_blake3"],
            "claims": claims,
        }
    )
    if expected != value.get("classification_id"):
        raise S4ReportError("R3 classification content address differs")
    return value


def _read_r3_report(path: Path, *, expected_arm: str) -> dict[str, Any]:
    report = _read_json(path, f"R3 {expected_arm} report")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != R3_EXPERIMENT_ID
        or report.get("protocol_id") != R3_PROTOCOL_ID
        or report.get("adr") != R3_ADR_ID
        or report.get("mode") != "production"
        or report.get("arm") != expected_arm
        or not isinstance(identity, dict)
        or identity != _r3_scientific_identity(report)
        or canonical_blake3(identity) != report.get("report_id")
    ):
        raise S4ReportError(f"R3 reference report is malformed: {expected_arm}")
    _validate_metrics(report.get("metrics"), label=expected_arm)
    _validate_r3_performance(
        report.get("performance"),
        label=expected_arm,
    )
    return report


def _validate_r3_performance(performance: object, *, label: str) -> None:
    if not isinstance(performance, dict):
        raise S4ReportError(f"R3 performance is absent for {label}")
    fixed = performance.get("fixed_chunk")
    decisions = performance.get("complete_decisions")
    memory = performance.get("memory")
    measurement = performance.get("measurement")
    if (
        not isinstance(fixed, dict)
        or not isinstance(decisions, dict)
        or not isinstance(memory, dict)
        or not isinstance(measurement, dict)
        or fixed.get("actions") != 256
        or fixed.get("warmup_iterations") != 5
        or fixed.get("steady_iterations") != 30
        or decisions.get("groups") != 20
        or decisions.get("parent_encodes") != 20
        or decisions.get("parent_encode_count_exact") is not True
        or measurement.get("isolated_process") is not True
        or measurement.get("verification_source") != "cluster-preflight"
        or not _is_blake3(measurement.get("request_id"))
        or not _is_blake3(measurement.get("result_id"))
        or not _is_blake3(measurement.get("checkpoint_model_blake3"))
        or not _is_blake3(measurement.get("open_data_verification_id"))
    ):
        raise S4ReportError(f"R3 serving protocol differs for {label}")
    _number(fixed.get("action_scores_per_second"))
    _number(decisions["latency_milliseconds"].get("p99"))
    _number(memory.get("peak_active_bytes"))
    _number(memory.get("peak_process_rss_bytes"))


def _validate_r3_binding(
    reports: dict[str, dict[str, Any]],
    *,
    classification: dict[str, Any],
    control: dict[str, Any],
    substrate: dict[str, Any],
) -> None:
    scientific = classification["scientific"]
    report_ids = scientific.get("arm_report_ids")
    warm_start = reports[CONTROL]["warm_start"]
    if (
        not isinstance(report_ids, dict)
        or report_ids.get(R3_CONTROL) != control.get("report_id")
        or report_ids.get(R3_SUBSTRATE) != substrate.get("report_id")
        or scientific.get("cache_id") != reports[CONTROL]["cache_id"]
        or scientific.get("s1_cache_id") != reports[CONTROL]["s1_cache_id"]
        or warm_start.get("global_step") != 3000
        or warm_start.get("model_blake3")
        != substrate["checkpoint"].get("model_blake3")
        or warm_start.get("manifest_blake3")
        != substrate["checkpoint"].get("manifest_blake3")
        or warm_start.get("model_config", {}).get("arm") != R3_SUBSTRATE
    ):
        raise S4ReportError("S4 warm start is not the frozen R3 radius-one arm")


def _absolute_gates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report["metrics"]
    performance = report["performance"]
    fixed = performance["fixed_chunk"]
    decision = performance["complete_decisions"]
    memory = performance["memory"]
    return {
        "complete_validation_coverage": (
            metrics["groups"] == 240
            and metrics["candidates"] == 860_203
            and metrics["all_groups_scored_once"] is True
            and metrics["all_candidates_scored_once"] is True
        ),
        "finite_scores_and_uncertainties": (
            metrics["all_scores_and_uncertainties_finite"] is True
        ),
        "one_parent_and_anchor_encode_per_decision": (
            metrics["parent_encodes"] == 240
            and metrics["parent_encode_count_exact"] is True
            and decision["parent_encode_count_exact"] is True
            and decision["anchor_encode_count_exact"] is True
        ),
        "process_swap_zero": memory["process_swaps"] == 0,
        "peak_active_memory_at_most_4_gib": (
            _number(memory["peak_active_bytes"]) <= 4 * 1024**3
        ),
        "peak_rss_at_most_4_gib": (
            _number(memory["peak_process_rss_bytes"]) <= 4 * 1024**3
        ),
        "p99_decision_latency_at_most_250_ms": (
            _number(decision["latency_milliseconds"]["p99"]) <= 250.0
        ),
        "action_throughput_at_least_20000": (
            _number(fixed["action_scores_per_second"]) >= 20_000.0
        ),
    }


def _quality_gates(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, bool]:
    return _noninferiority_gates(treatment, control)


def _external_rescue_gates(
    treatment: dict[str, Any],
    r3_control: dict[str, Any],
) -> dict[str, bool]:
    return _noninferiority_gates(treatment, r3_control)


def _noninferiority_gates(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, bool]:
    tm = treatment["metrics"]
    cm = control["metrics"]
    return {
        "r4800_mae_delta_at_most_0_05": (
            _number(tm["r4800_value"]["mae"])
            - _number(cm["r4800_value"]["mae"])
            <= MAX_MAE_DELTA
        ),
        "r4800_rmse_delta_at_most_0_05": (
            _number(tm["r4800_value"]["rmse"])
            - _number(cm["r4800_value"]["rmse"])
            <= MAX_RMSE_DELTA
        ),
        "top64_winner_recall_delta_at_least_minus_0_005": (
            _number(tm["top64_r4800_winner_recall"])
            - _number(cm["top64_r4800_winner_recall"])
            >= MIN_RECALL_DELTA
        ),
        "top64_retained_regret_delta_at_most_0_005": (
            _number(tm["mean_top64_retained_r4800_regret"])
            - _number(cm["mean_top64_retained_r4800_regret"])
            <= MAX_REGRET_DELTA
        ),
        "low_supply_top64_recall_delta_at_least_minus_0_01": (
            _number(
                tm["subsets"]["low_supply"]["top64_r4800_winner_recall"]
            )
            - _number(
                cm["subsets"]["low_supply"]["top64_r4800_winner_recall"]
            )
            >= MIN_SLICE_RECALL_DELTA
        ),
        "independent_winner_top64_recall_delta_at_least_minus_0_01": (
            _number(
                tm["subsets"]["independent_draft_winner"][
                    "top64_r4800_winner_recall"
                ]
            )
            - _number(
                cm["subsets"]["independent_draft_winner"][
                    "top64_r4800_winner_recall"
                ]
            )
            >= MIN_SLICE_RECALL_DELTA
        ),
        "top64_confidence_set_coverage_at_least_0_99": (
            _number(tm["top64_confidence_set_coverage_95"])
            >= MIN_CONFIDENCE_COVERAGE
        ),
    }


def _material_context_gates(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, bool]:
    tm = treatment["metrics"]
    cm = control["metrics"]
    return {
        "mae_reduction_at_least_0_05": (
            _number(cm["r4800_value"]["mae"])
            - _number(tm["r4800_value"]["mae"])
            >= MIN_MAE_REDUCTION
        ),
        "rmse_reduction_at_least_0_05": (
            _number(cm["r4800_value"]["rmse"])
            - _number(tm["r4800_value"]["rmse"])
            >= MIN_RMSE_REDUCTION
        ),
        "top64_recall_gain_at_least_0_02": (
            _number(tm["top64_r4800_winner_recall"])
            - _number(cm["top64_r4800_winner_recall"])
            >= MIN_RECALL_GAIN
        ),
        "top64_regret_reduction_at_least_0_02": (
            _number(cm["mean_top64_retained_r4800_regret"])
            - _number(tm["mean_top64_retained_r4800_regret"])
            >= MIN_REGRET_REDUCTION
        ),
        "confidence_coverage_gain_at_least_0_01": (
            _number(tm["top64_confidence_set_coverage_95"])
            - _number(cm["top64_confidence_set_coverage_95"])
            >= MIN_COVERAGE_GAIN
        ),
    }


def _performance_rescue_gates(
    treatment: dict[str, Any],
    r3_control: dict[str, Any],
) -> dict[str, bool]:
    tp = treatment["performance"]
    cp = r3_control["performance"]
    return {
        "throughput_at_least_0_25x_r3_control": (
            _number(tp["fixed_chunk"]["action_scores_per_second"])
            >= MIN_R3_THROUGHPUT_FRACTION
            * _number(cp["fixed_chunk"]["action_scores_per_second"])
        ),
        "p99_latency_at_most_1_75x_r3_control": (
            _number(tp["complete_decisions"]["latency_milliseconds"]["p99"])
            <= MAX_R3_P99_MULTIPLIER
            * _number(
                cp["complete_decisions"]["latency_milliseconds"]["p99"]
            )
        ),
        "active_memory_at_most_2_50x_r3_control": (
            _number(tp["memory"]["peak_active_bytes"])
            <= MAX_R3_ACTIVE_MEMORY_MULTIPLIER
            * _number(cp["memory"]["peak_active_bytes"])
        ),
        "rss_at_most_1_50x_r3_control": (
            _number(tp["memory"]["peak_process_rss_bytes"])
            <= MAX_R3_RSS_MULTIPLIER
            * _number(cp["memory"]["peak_process_rss_bytes"])
        ),
    }


def _select_arm(
    arms: list[str],
    reports: dict[str, dict[str, Any]],
) -> str:
    return min(
        arms,
        key=lambda arm: (
            -_number(
                reports[arm]["metrics"]["top64_r4800_winner_recall"]
            ),
            _number(
                reports[arm]["metrics"][
                    "mean_top64_retained_r4800_regret"
                ]
            ),
            _number(reports[arm]["metrics"]["r4800_value"]["mae"]),
            -_number(
                reports[arm]["performance"]["fixed_chunk"][
                    "action_scores_per_second"
                ]
            ),
            ARMS.index(arm),
        ),
    )


def _arm_scientific_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "arm",
            "host",
            "cache_id",
            "s1_cache_id",
            "context_cache_id",
            "protocol",
            "warm_start",
            "initial_prediction_parity",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
            "performance",
            "runtime",
            "source",
            "controls",
            "information_boundary",
            "claims",
        )
    }


def _r3_scientific_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "arm",
            "host",
            "cache_id",
            "s1_cache_id",
            "protocol",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
            "performance",
            "runtime",
            "source",
            "controls",
            "information_boundary",
            "claims",
        )
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S4ReportError("S4 report metric is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise S4ReportError("S4 report metric is non-finite")
    return result


def _is_blake3(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise S4ReportError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise S4ReportError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    classify = subparsers.add_parser("classify")
    classify.add_argument("--report", type=Path, action="append", required=True)
    classify.add_argument("--r3-classification", type=Path, required=True)
    classify.add_argument("--r3-control", type=Path, required=True)
    classify.add_argument("--r3-substrate", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--forward", type=Path, required=True)
    compare.add_argument("--reverse", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "classify":
            result = classify_reports(
                args.report,
                r3_classification_path=args.r3_classification,
                r3_control_path=args.r3_control,
                r3_substrate_path=args.r3_substrate,
            )
        else:
            result = compare_classifications(args.forward, args.reverse)
        _write_json_atomic(args.output, result)
        print(json.dumps(result, sort_keys=True))
    except S4ReportError as error:
        invalid = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "classification": INVALID,
            "error": str(error),
        }
        _write_json_atomic(args.output, invalid)
        print(json.dumps(invalid, sort_keys=True))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
