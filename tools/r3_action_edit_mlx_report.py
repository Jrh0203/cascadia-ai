#!/usr/bin/env python3
"""Validate four ADR 0150 arm reports and emit the frozen classification."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "r3-action-edit-mlx-comparison-v1"
PROTOCOL_ID = "r3-action-edit-mlx-matched-comparison-v1"
ADR_ID = "0150"
ARMS = (
    "c0-full-r2-afterstate",
    "t1-r3-radius3-global",
    "t2-r3-radius2-global",
    "t3-r3-radius1-global",
)
CONTROL = ARMS[0]
ARM_HOSTS = {
    ARMS[0]: "john1",
    ARMS[1]: "john2",
    ARMS[2]: "john3",
    ARMS[3]: "john4",
}
RADII = {
    ARMS[1]: 3,
    ARMS[2]: 2,
    ARMS[3]: 1,
}

INVALID = "r3_action_edit_mlx_invalid_evidence"
CONTROL_FAILED = "r3_action_edit_mlx_control_failed"
ALL_TREATMENTS_DEGRADED = "r3_action_edit_mlx_all_treatments_degraded"
QUALITY_ONLY_NULL = "r3_action_edit_mlx_quality_only_null"
SELECTED = "r3_action_edit_mlx_compact_representation_selected"


class R3ReportError(ValueError):
    """Arm evidence is missing, inconsistent, or outside the frozen protocol."""


def classify_reports(paths: list[Path]) -> dict[str, Any]:
    try:
        return _classify_reports(paths)
    except R3ReportError:
        raise
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise R3ReportError(f"R3 report schema is incomplete: {error}") from error


def _classify_reports(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(ARMS):
        raise R3ReportError("R3 classification requires exactly four arm reports")
    reports = [_read_report(path) for path in paths]
    by_arm = {str(report["arm"]): report for report in reports}
    if set(by_arm) != set(ARMS) or len(by_arm) != len(ARMS):
        raise R3ReportError("R3 reports must contain each arm exactly once")
    _validate_shared_identity(by_arm)

    gates = {
        arm: {
            "absolute": _absolute_gates(report),
        }
        for arm, report in by_arm.items()
    }
    control_passed = all(gates[CONTROL]["absolute"].values())
    if not control_passed:
        classification = CONTROL_FAILED
        selected_arm = None
    else:
        control = by_arm[CONTROL]
        eligible: list[str] = []
        efficient: list[str] = []
        for arm in ARMS[1:]:
            quality = _quality_gates(by_arm[arm], control)
            efficiency = _efficiency_gates(by_arm[arm], control)
            gates[arm]["quality_noninferiority"] = quality
            gates[arm]["material_efficiency"] = efficiency
            if all(gates[arm]["absolute"].values()) and all(quality.values()):
                eligible.append(arm)
                if any(efficiency.values()):
                    efficient.append(arm)
        if not eligible:
            classification = ALL_TREATMENTS_DEGRADED
            selected_arm = None
        elif not efficient:
            classification = QUALITY_ONLY_NULL
            selected_arm = None
        else:
            classification = SELECTED
            selected_arm = _select_arm(efficient, by_arm)

    scientific = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected_arm,
        "cache_id": by_arm[CONTROL]["cache_id"],
        "s1_cache_id": by_arm[CONTROL]["s1_cache_id"],
        "arm_report_ids": {arm: by_arm[arm]["report_id"] for arm in ARMS},
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
        "scientific_blake3": _canonical_blake3(scientific),
        "claims": {
            "offline_comparison_complete": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    output["classification_id"] = _canonical_blake3(
        {
            "schema_version": output["schema_version"],
            "scientific": output["scientific"],
            "scientific_blake3": output["scientific_blake3"],
            "claims": output["claims"],
        }
    )
    return output


def compare_classifications(forward: Path, reverse: Path) -> dict[str, Any]:
    forward_bytes = forward.read_bytes()
    reverse_bytes = reverse.read_bytes()
    if forward_bytes != reverse_bytes:
        raise R3ReportError("forward and reverse R3 classifications are not byte-identical")
    parsed = _validate_classification(_read_json(forward, "R3 classification"))
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": parsed.get("classification"),
        "classification_id": parsed.get("classification_id"),
        "scientific_blake3": parsed.get("scientific_blake3"),
        "classification_file_blake3": blake3.blake3(forward_bytes).hexdigest(),
        "forward_reverse_byte_identical": True,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "proof_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }


def _read_report(path: Path) -> dict[str, Any]:
    report = _read_json(path, f"R3 arm report {path}")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("mode") != "production"
        or report.get("arm") not in ARMS
        or not isinstance(identity, dict)
        or identity != _arm_scientific_identity(report)
        or _canonical_blake3(identity) != report.get("report_id")
    ):
        raise R3ReportError(f"R3 arm report is malformed: {path}")
    information = report.get("information_boundary")
    claims = report.get("claims")
    if (
        not isinstance(information, dict)
        or information.get("sealed_test_opened") is not False
        or information.get("gameplay_run") is not False
        or information.get("hidden_order_read") is not False
        or information.get("future_refill_read") is not False
        or not isinstance(claims, dict)
        or claims.get("offline_comparison_complete") is not True
        or claims.get("gameplay_strength_measured") is not False
        or claims.get("promotion_authorized") is not False
        or claims.get("progress_to_100_claimed") is not False
    ):
        raise R3ReportError(f"R3 information boundary is invalid: {path}")
    return report


def _validate_shared_identity(reports: dict[str, dict[str, Any]]) -> None:
    shared_fields = (
        "cache_id",
        "s1_cache_id",
        "protocol",
    )
    control = reports[CONTROL]
    control_source = control["source"].get("v2_source_blake3")
    control_authorization = control["controls"].get("authorization_id")
    control_schedule = _batch_schedule(control)
    if not _is_blake3(control_source) or not _is_blake3(control_authorization):
        raise R3ReportError("R3 source or authorization identity is malformed")
    for arm, report in reports.items():
        if any(report.get(field) != control.get(field) for field in shared_fields):
            raise R3ReportError(f"R3 shared identity differs for {arm}")
        optimization = report.get("optimization")
        model = report.get("model")
        metrics = report.get("metrics")
        source = report.get("source")
        runtime = report.get("runtime")
        controls = report.get("controls")
        if (
            report.get("host") != ARM_HOSTS[arm]
            or not isinstance(source, dict)
            or source.get("v2_source_blake3") != control_source
            or not isinstance(runtime, dict)
            or runtime.get("machine") != "arm64"
            or "gpu" not in str(runtime.get("default_device", "")).lower()
            or runtime.get("host") != ARM_HOSTS[arm]
            or not isinstance(controls, dict)
            or controls.get("authorization_id") != control_authorization
            or not _is_blake3(controls.get("preflight_id"))
            or not isinstance(optimization, dict)
            or optimization.get("global_step") != 3000
            or not isinstance(model, dict)
            or model.get("parameter_count") != control["model"].get("parameter_count")
            or model.get("parameter_layout_blake3")
            != control["model"].get("parameter_layout_blake3")
            or model.get("initial_parameter_tensor_blake3")
            != control["model"].get("initial_parameter_tensor_blake3")
            or not isinstance(metrics, dict)
            or metrics.get("groups") != 240
            or metrics.get("candidates") != 860_203
            or metrics.get("all_groups_scored_once") is not True
            or metrics.get("all_candidates_scored_once") is not True
        ):
            raise R3ReportError(f"R3 production coverage or model parity failed for {arm}")
        if _batch_schedule(report) != control_schedule:
            raise R3ReportError(f"R3 scientific training schedule differs for {arm}")
        _validate_complete_metrics(metrics, arm=arm)
        _validate_performance(report.get("performance"), arm=arm)


def _batch_schedule(report: dict[str, Any]) -> list[tuple[int, str, int]]:
    optimization = report.get("optimization")
    if not isinstance(optimization, dict):
        raise R3ReportError("R3 optimization evidence is absent")
    trace = optimization.get("loss_trace")
    if not isinstance(trace, list) or len(trace) != 3000:
        raise R3ReportError("R3 loss trace must contain exactly 3000 steps")
    schedule: list[tuple[int, str, int]] = []
    measured_candidates = 0
    for expected_step, event in enumerate(trace, start=1):
        if (
            not isinstance(event, dict)
            or event.get("step") != expected_step
            or not _is_blake3(event.get("batch_blake3"))
        ):
            raise R3ReportError("R3 loss trace order or batch identity is invalid")
        candidates = int(_number(event.get("candidates")))
        if candidates <= 0 or candidates != _number(event.get("candidates")):
            raise R3ReportError("R3 loss trace candidate count is invalid")
        if _number(event.get("elapsed_seconds")) <= 0:
            raise R3ReportError("R3 loss trace elapsed time is invalid")
        _number(event.get("loss"))
        measured_candidates += candidates
        schedule.append((expected_step, str(event["batch_blake3"]), candidates))
    if optimization.get("candidates") != measured_candidates:
        raise R3ReportError("R3 measured candidate accounting differs from the trace")
    return schedule


def _validate_complete_metrics(metrics: object, *, arm: str) -> None:
    if not isinstance(metrics, dict):
        raise R3ReportError(f"R3 metrics are absent for {arm}")
    r4800 = metrics.get("r4800_value")
    subsets = metrics.get("subsets")
    tokens = metrics.get("candidate_tokens")
    panel = metrics.get("prediction_panel")
    if (
        not isinstance(r4800, dict)
        or not isinstance(subsets, dict)
        or set(("early", "middle", "late", "low_supply", "independent_draft_winner")) - set(subsets)
        or not isinstance(tokens, dict)
        or not isinstance(panel, dict)
    ):
        raise R3ReportError(f"R3 required validation evidence is absent for {arm}")
    for key in (
        "mae",
        "rmse",
        "bias",
        "correlation",
        "calibration_slope",
        "calibration_intercept",
    ):
        _number(r4800.get(key))
    for width in (1, 8, 32, 64):
        _number(metrics.get(f"top{width}_r4800_winner_recall"))
        _number(metrics.get(f"mean_top{width}_retained_r4800_regret"))
    _number(metrics.get("top64_confidence_set_coverage_95"))
    for name in ("early", "middle", "late", "low_supply", "independent_draft_winner"):
        values = subsets.get(name)
        if not isinstance(values, dict):
            raise R3ReportError(f"R3 validation slice {name} is absent for {arm}")
        if int(_number(values.get("groups"))) <= 0:
            raise R3ReportError(f"R3 validation slice {name} is empty for {arm}")
        _number(values.get("top64_r4800_winner_recall"))
        _number(values.get("top64_confidence_set_coverage_95"))
        _number(values.get("mean_top64_retained_r4800_regret"))
    for key in ("count", "minimum", "mean", "p50", "p90", "p99", "maximum", "padding_tokens"):
        _number(tokens.get(key))
    _validate_prediction_panel(panel, arm=arm)


def _validate_prediction_panel(panel: dict[str, Any], *, arm: str) -> None:
    count = int(_number(panel.get("count")))
    hashes = panel.get("action_hashes")
    scores = panel.get("scores")
    uncertainties = panel.get("standard_errors")
    if (
        count != 64
        or not isinstance(hashes, list)
        or not isinstance(scores, list)
        or not isinstance(uncertainties, list)
        or len(hashes) != count
        or len(scores) != count
        or len(uncertainties) != count
    ):
        raise R3ReportError(f"R3 prediction panel has the wrong shape for {arm}")
    digest = blake3.blake3()
    try:
        if any(not _is_blake3(value) for value in hashes):
            raise ValueError("action hash is not a BLAKE3 digest")
        digest.update(b"".join(bytes.fromhex(str(value)) for value in hashes))
    except ValueError as error:
        raise R3ReportError(f"R3 prediction panel action hash is invalid for {arm}") from error
    for value in scores:
        digest.update(struct.pack("<f", _number(value)))
    for value in uncertainties:
        uncertainty = _number(value)
        if uncertainty <= 0:
            raise R3ReportError(f"R3 prediction uncertainty is invalid for {arm}")
        digest.update(struct.pack("<f", uncertainty))
    if digest.hexdigest() != panel.get("panel_blake3"):
        raise R3ReportError(f"R3 prediction panel digest differs for {arm}")


def _validate_performance(performance: object, *, arm: str) -> None:
    if not isinstance(performance, dict):
        raise R3ReportError(f"R3 performance evidence is absent for {arm}")
    for section in ("fixed_chunk", "complete_decisions", "memory"):
        if not isinstance(performance.get(section), dict):
            raise R3ReportError(f"R3 performance section {section} is absent for {arm}")
    fixed = performance["fixed_chunk"]
    decisions = performance["complete_decisions"]
    measurement = performance.get("measurement")
    if (
        fixed.get("actions") != 256
        or int(_number(fixed.get("warmup_iterations"))) <= 0
        or int(_number(fixed.get("steady_iterations"))) <= 0
        or decisions.get("groups") != 20
        or decisions.get("parent_encodes") != 20
        or decisions.get("parent_encode_count_exact") is not True
        or not isinstance(measurement, dict)
        or measurement.get("isolated_process") is not True
        or not _is_blake3(measurement.get("request_id"))
        or not _is_blake3(measurement.get("result_id"))
        or not _is_blake3(measurement.get("checkpoint_model_blake3"))
        or not _is_blake3(measurement.get("open_data_verification_id"))
        or measurement.get("verification_source") != "cluster-preflight"
    ):
        raise R3ReportError(f"R3 performance protocol differs for {arm}")
    for section in (fixed, decisions):
        _number(section.get("action_scores_per_second"))
        latency = section.get("latency_milliseconds")
        if not isinstance(latency, dict):
            raise R3ReportError(f"R3 latency evidence is absent for {arm}")
        for quantile in ("p50", "p95", "p99"):
            _number(latency.get(quantile))


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


def _validate_classification(value: dict[str, Any]) -> dict[str, Any]:
    scientific = value.get("scientific")
    claims = value.get("claims")
    if (
        value.get("schema_version") != 1
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("adr") != ADR_ID
        or value.get("classification") == INVALID
        or value.get("classification")
        not in (CONTROL_FAILED, ALL_TREATMENTS_DEGRADED, QUALITY_ONLY_NULL, SELECTED)
        or not isinstance(scientific, dict)
        or not isinstance(claims, dict)
        or scientific.get("experiment_id") != EXPERIMENT_ID
        or scientific.get("protocol_id") != PROTOCOL_ID
        or scientific.get("adr") != ADR_ID
        or scientific.get("classification") != value.get("classification")
        or scientific.get("selected_arm") != value.get("selected_arm")
    ):
        raise R3ReportError("R3 classification is malformed")
    if _canonical_blake3(scientific) != value.get("scientific_blake3"):
        raise R3ReportError("R3 scientific classification content address differs")
    expected_id = _canonical_blake3(
        {
            "schema_version": value["schema_version"],
            "scientific": scientific,
            "scientific_blake3": value["scientific_blake3"],
            "claims": claims,
        }
    )
    if expected_id != value.get("classification_id"):
        raise R3ReportError("R3 classification content address differs")
    return value


def _absolute_gates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report["metrics"]
    performance = report["performance"]
    memory = performance["memory"]
    fixed = performance["fixed_chunk"]
    decision = performance["complete_decisions"]
    return {
        "complete_validation_coverage": (
            metrics.get("groups") == 240
            and metrics.get("candidates") == 860_203
            and metrics.get("all_groups_scored_once") is True
            and metrics.get("all_candidates_scored_once") is True
        ),
        "finite_scores_and_uncertainties": (
            metrics.get("all_scores_and_uncertainties_finite") is True
        ),
        "one_parent_encode_per_decision": (
            metrics.get("parent_encodes") == 240
            and metrics.get("parent_encode_count_exact") is True
            and decision.get("parent_encode_count_exact") is True
        ),
        "process_swap_zero": memory.get("process_swaps") == 0,
        "peak_active_memory_at_most_4_gib": (
            _number(memory.get("peak_active_bytes")) <= 4 * 1024**3
        ),
        "peak_rss_at_most_4_gib": (_number(memory.get("peak_process_rss_bytes")) <= 4 * 1024**3),
        "p99_decision_latency_at_most_250_ms": (
            _number(decision["latency_milliseconds"].get("p99")) <= 250.0
        ),
        "action_throughput_at_least_20000": (
            _number(fixed.get("action_scores_per_second")) >= 20_000.0
        ),
    }


def _quality_gates(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, bool]:
    treatment_metrics = treatment["metrics"]
    control_metrics = control["metrics"]
    treatment_value = treatment_metrics["r4800_value"]
    control_value = control_metrics["r4800_value"]
    treatment_subsets = treatment_metrics["subsets"]
    control_subsets = control_metrics["subsets"]
    return {
        "r4800_mae_delta_at_most_0_05": (
            _number(treatment_value["mae"]) - _number(control_value["mae"]) <= 0.05
        ),
        "r4800_rmse_delta_at_most_0_05": (
            _number(treatment_value["rmse"]) - _number(control_value["rmse"]) <= 0.05
        ),
        "top64_winner_recall_delta_at_least_minus_0_005": (
            _number(treatment_metrics["top64_r4800_winner_recall"])
            - _number(control_metrics["top64_r4800_winner_recall"])
            >= -0.005
        ),
        "top64_retained_regret_delta_at_most_0_005": (
            _number(treatment_metrics["mean_top64_retained_r4800_regret"])
            - _number(control_metrics["mean_top64_retained_r4800_regret"])
            <= 0.005
        ),
        "low_supply_top64_recall_delta_at_least_minus_0_01": (
            _number(treatment_subsets["low_supply"]["top64_r4800_winner_recall"])
            - _number(control_subsets["low_supply"]["top64_r4800_winner_recall"])
            >= -0.01
        ),
        "independent_winner_top64_recall_delta_at_least_minus_0_01": (
            _number(treatment_subsets["independent_draft_winner"]["top64_r4800_winner_recall"])
            - _number(control_subsets["independent_draft_winner"]["top64_r4800_winner_recall"])
            >= -0.01
        ),
        "top64_confidence_set_coverage_at_least_0_99": (
            _number(treatment_metrics["top64_confidence_set_coverage_95"]) >= 0.99
        ),
    }


def _efficiency_gates(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, bool]:
    treatment_performance = treatment["performance"]
    control_performance = control["performance"]
    treatment_memory = treatment_performance["memory"]
    control_memory = control_performance["memory"]
    return {
        "throughput_at_least_1_35x_control": (
            _number(treatment_performance["fixed_chunk"]["action_scores_per_second"])
            >= 1.35 * _number(control_performance["fixed_chunk"]["action_scores_per_second"])
        ),
        "p99_latency_at_most_0_80x_control": (
            _number(treatment_performance["complete_decisions"]["latency_milliseconds"]["p99"])
            <= 0.80
            * _number(control_performance["complete_decisions"]["latency_milliseconds"]["p99"])
        ),
        "peak_active_memory_at_most_0_80x_control": (
            _number(treatment_memory["peak_active_bytes"])
            <= 0.80 * _number(control_memory["peak_active_bytes"])
        ),
        "peak_rss_at_most_0_80x_control": (
            _number(treatment_memory["peak_process_rss_bytes"])
            <= 0.80 * _number(control_memory["peak_process_rss_bytes"])
        ),
    }


def _select_arm(
    arms: list[str],
    reports: dict[str, dict[str, Any]],
) -> str:
    throughput = {
        arm: _number(reports[arm]["performance"]["fixed_chunk"]["action_scores_per_second"])
        for arm in arms
    }
    fastest = max(throughput.values())
    tied = [arm for arm in arms if throughput[arm] >= 0.99 * fastest]
    return min(
        tied,
        key=lambda arm: (
            RADII[arm],
            _number(
                reports[arm]["performance"]["complete_decisions"]["latency_milliseconds"]["p99"]
            ),
            _number(reports[arm]["performance"]["memory"]["peak_active_bytes"]),
            arm,
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R3ReportError("R3 report metric is not numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise R3ReportError("R3 report metric is non-finite")
    return result


def _is_blake3(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R3ReportError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise R3ReportError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    classify = subparsers.add_parser("classify")
    classify.add_argument("--report", type=Path, action="append", required=True)
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
            result = classify_reports(args.report)
        else:
            result = compare_classifications(args.forward, args.reverse)
        _write_json_atomic(args.output, result)
        print(json.dumps(result, sort_keys=True))
    except R3ReportError as error:
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
