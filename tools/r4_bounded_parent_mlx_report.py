#!/usr/bin/env python3
"""Classify ADR 0156 quality and host-paired serving evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.r4_bounded_parent_mlx_cache import (
    ADR_ID,
    ARMS,
    CONTROL_ARM,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)
from cascadia_mlx.r4_bounded_parent_mlx_train import ARM_HOSTS, TRAINING_STEPS

CLASSIFICATION_INVALID = "r4_bounded_parent_mlx_invalid"
CLASSIFICATION_CONTROL_FAILED = "r4_bounded_parent_mlx_control_failed"
CLASSIFICATION_ALL_DEGRADED = "r4_bounded_parent_mlx_all_treatments_degraded"
CLASSIFICATION_QUALITY_ONLY_NULL = "r4_bounded_parent_mlx_quality_only_null"
CLASSIFICATION_SELECTED = "r4_bounded_parent_mlx_representation_selected"

CONTROL_LIMITS = {
    "mae_max": 1.42,
    "rmse_max": 1.85,
    "top64_recall_min": 0.70,
    "top64_regret_max": 0.12,
    "low_supply_recall_min": 0.88,
    "independent_recall_min": 0.76,
    "coverage_min": 0.97,
}
QUALITY_LIMITS = {
    "mae_delta_max": 0.05,
    "rmse_delta_max": 0.05,
    "top64_recall_delta_min": -0.005,
    "top64_regret_delta_max": 0.005,
    "low_supply_recall_delta_min": -0.01,
    "independent_recall_delta_min": -0.01,
    "coverage_min": 0.99,
}
ABSOLUTE_SERVING = {
    "groups": 240,
    "actions": 860_203,
    "memory_max": 4 * 1024**3,
    "p99_ms_max": 250.0,
    "throughput_min": 20_000.0,
}
TIE_ORDER = {
    "q1-seat-marginal-parent": 0,
    "q3-affordance-parent": 1,
    "q2-directional-parent": 2,
}


class R4ParentReportError(ValueError):
    """ADR 0156 evidence is incomplete or internally inconsistent."""


def aggregate_with_order_proof(
    reports: list[dict[str, Any]],
    paired_controls: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    forward = classify_reports(reports, paired_controls)
    reverse = classify_reports(list(reversed(reports)), list(reversed(paired_controls)))
    forward_bytes = _canonical_bytes(forward["scientific_identity"])
    reverse_bytes = _canonical_bytes(reverse["scientific_identity"])
    proof_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "forward_aggregate_id": forward["aggregate_id"],
        "reverse_aggregate_id": reverse["aggregate_id"],
        "byte_identical": forward_bytes == reverse_bytes,
        "forward_blake3": blake3.blake3(forward_bytes).hexdigest(),
        "reverse_blake3": blake3.blake3(reverse_bytes).hexdigest(),
    }
    proof = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "proof_id": _canonical_blake3(proof_identity),
        "scientific_identity": proof_identity,
    }
    if not proof_identity["byte_identical"]:
        raise R4ParentReportError("ADR 0156 classification depends on input order")
    return forward, reverse, proof


def invalid_outputs(
    error: Exception,
    report_paths: list[Path],
    paired_control_paths: list[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Emit a content-addressed invalid classification for structural failures."""
    inputs = {
        "reports": sorted(_path_identity(path) for path in report_paths),
        "paired_controls": sorted(_path_identity(path) for path in paired_control_paths),
    }
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": CLASSIFICATION_INVALID,
        "selected_arm": None,
        "structural_error_type": type(error).__name__,
        "structural_error": str(error),
        "inputs": inputs,
        "claim_boundary": {
            "evidence_valid": False,
            "compact_parent_substrate_may_be_selected": False,
            "gameplay_strength_established": False,
            "progress_to_100_established": False,
        },
    }
    aggregate = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": CLASSIFICATION_INVALID,
        "selected_arm": None,
        "aggregate_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }
    proof_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "forward_aggregate_id": aggregate["aggregate_id"],
        "reverse_aggregate_id": aggregate["aggregate_id"],
        "byte_identical": True,
        "invalid_evidence": True,
        "forward_blake3": blake3.blake3(_canonical_bytes(identity)).hexdigest(),
        "reverse_blake3": blake3.blake3(_canonical_bytes(identity)).hexdigest(),
    }
    proof = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "proof_id": _canonical_blake3(proof_identity),
        "scientific_identity": proof_identity,
    }
    return aggregate, dict(aggregate), proof


def classify_reports(
    reports: list[dict[str, Any]],
    paired_controls: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm = _validate_reports(reports)
    paired = _validate_paired_controls(paired_controls, by_arm)
    control = by_arm[CONTROL_ARM]
    control_quality = _quality(control)
    control_serving = _absolute_serving(control)
    control_sanity = {
        "mae": control_quality["mae"] <= CONTROL_LIMITS["mae_max"],
        "rmse": control_quality["rmse"] <= CONTROL_LIMITS["rmse_max"],
        "top64_recall": control_quality["top64_recall"] >= CONTROL_LIMITS["top64_recall_min"],
        "top64_regret": control_quality["top64_regret"] <= CONTROL_LIMITS["top64_regret_max"],
        "low_supply_recall": control_quality["low_supply_recall"]
        >= CONTROL_LIMITS["low_supply_recall_min"],
        "independent_recall": control_quality["independent_recall"]
        >= CONTROL_LIMITS["independent_recall_min"],
        "coverage": control_quality["coverage"] >= CONTROL_LIMITS["coverage_min"],
        "absolute_serving": control_serving["passed"],
    }

    assessments: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    quality_passing = 0
    for arm in ARMS[1:]:
        treatment = by_arm[arm]
        quality = _quality(treatment)
        serving = _absolute_serving(treatment)
        quality_delta = {
            "mae": quality["mae"] - control_quality["mae"],
            "rmse": quality["rmse"] - control_quality["rmse"],
            "top64_recall": quality["top64_recall"] - control_quality["top64_recall"],
            "top64_regret": quality["top64_regret"] - control_quality["top64_regret"],
            "low_supply_recall": quality["low_supply_recall"]
            - control_quality["low_supply_recall"],
            "independent_recall": quality["independent_recall"]
            - control_quality["independent_recall"],
        }
        quality_checks = {
            "mae": quality_delta["mae"] <= QUALITY_LIMITS["mae_delta_max"],
            "rmse": quality_delta["rmse"] <= QUALITY_LIMITS["rmse_delta_max"],
            "top64_recall": quality_delta["top64_recall"]
            >= QUALITY_LIMITS["top64_recall_delta_min"],
            "top64_regret": quality_delta["top64_regret"]
            <= QUALITY_LIMITS["top64_regret_delta_max"],
            "low_supply_recall": quality_delta["low_supply_recall"]
            >= QUALITY_LIMITS["low_supply_recall_delta_min"],
            "independent_recall": quality_delta["independent_recall"]
            >= QUALITY_LIMITS["independent_recall_delta_min"],
            "coverage": quality["coverage"] >= QUALITY_LIMITS["coverage_min"],
        }
        quality_passed = all(quality_checks.values())
        quality_passing += int(quality_passed)
        host_control = paired[arm]["performance"]
        treatment_performance = treatment["performance"]
        efficiency = _efficiency(treatment_performance, host_control)
        eligible_arm = quality_passed and serving["passed"] and efficiency["passed"]
        if eligible_arm:
            eligible.append(arm)
        assessments[arm] = {
            "host": treatment["host"],
            "quality": quality,
            "quality_delta": quality_delta,
            "quality_checks": quality_checks,
            "quality_passed": quality_passed,
            "absolute_serving": serving,
            "host_paired_control_replay_id": paired[arm]["replay_id"],
            "efficiency": efficiency,
            "eligible": eligible_arm,
        }

    if not all(control_sanity.values()):
        classification = CLASSIFICATION_CONTROL_FAILED
        selected = None
    elif not quality_passing:
        classification = CLASSIFICATION_ALL_DEGRADED
        selected = None
    elif not eligible:
        classification = CLASSIFICATION_QUALITY_ONLY_NULL
        selected = None
    else:
        selected = _select_arm(eligible, assessments)
        classification = CLASSIFICATION_SELECTED

    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected,
        "report_ids": {arm: by_arm[arm]["report_id"] for arm in ARMS},
        "paired_control_replay_ids": {arm: paired[arm]["replay_id"] for arm in ARMS[1:]},
        "common_identity": _common_identity(by_arm),
        "control": {
            "quality": control_quality,
            "sanity_checks": control_sanity,
            "absolute_serving": control_serving,
        },
        "treatments": assessments,
        "limits": {
            "control": CONTROL_LIMITS,
            "quality": QUALITY_LIMITS,
            "absolute_serving": ABSOLUTE_SERVING,
            "parent_p50_ratio_max": 0.80,
            "material_end_to_end": {
                "throughput_ratio_min": 1.05,
                "p99_ratio_max": 0.95,
                "active_memory_ratio_max": 0.85,
                "rss_ratio_max": 0.85,
            },
        },
        "claim_boundary": {
            "compact_parent_substrate_may_be_selected": classification == CLASSIFICATION_SELECTED,
            "candidate_afterstate_compression_established": False,
            "gameplay_strength_established": False,
            "progress_to_100_established": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected,
        "aggregate_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }


def _validate_reports(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(reports) != len(ARMS):
        raise R4ParentReportError("ADR 0156 requires exactly four arm reports")
    by_arm: dict[str, dict[str, Any]] = {}
    batch_traces: dict[str, list[tuple[int, str, int]]] = {}
    for report in reports:
        identity = report.get("scientific_identity")
        arm = report.get("arm")
        if (
            report.get("schema_version") != 1
            or report.get("experiment_id") != EXPERIMENT_ID
            or report.get("protocol_id") != PROTOCOL_ID
            or report.get("adr") != ADR_ID
            or report.get("mode") != "production"
            or arm not in ARMS
            or report.get("host") != ARM_HOSTS[arm]
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != report.get("report_id")
            or arm in by_arm
        ):
            raise R4ParentReportError("ADR 0156 arm report is malformed or duplicated")
        optimization = report.get("optimization")
        trace = optimization.get("loss_trace") if isinstance(optimization, dict) else None
        if (
            not isinstance(trace, list)
            or len(trace) != TRAINING_STEPS
            or optimization.get("global_step") != TRAINING_STEPS
        ):
            raise R4ParentReportError(f"{arm} training trace is incomplete")
        normalized = []
        for step, event in enumerate(trace, start=1):
            if (
                not isinstance(event, dict)
                or event.get("step") != step
                or not isinstance(event.get("batch_blake3"), str)
                or not isinstance(event.get("candidates"), int)
                or not _finite(event.get("loss"))
                or not _finite(event.get("elapsed_seconds"))
            ):
                raise R4ParentReportError(f"{arm} training trace is malformed")
            normalized.append((step, event["batch_blake3"], event["candidates"]))
        batch_traces[arm] = normalized
        by_arm[arm] = report
    if set(by_arm) != set(ARMS) or len({tuple(value) for value in batch_traces.values()}) != 1:
        raise R4ParentReportError("ADR 0156 arms did not consume identical scientific batches")
    _common_identity(by_arm)
    return by_arm


def _common_identity(by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "r3_cache_id",
        "parent_cache_id",
        "s1_cache_id",
        "protocol",
    )
    common = {field: by_arm[CONTROL_ARM].get(field) for field in fields}
    model = by_arm[CONTROL_ARM].get("model", {})
    common["parameter_count"] = model.get("parameter_count")
    common["parameter_layout_blake3"] = model.get("parameter_layout_blake3")
    common["initial_parameter_tensor_blake3"] = model.get("initial_parameter_tensor_blake3")
    for arm, report in by_arm.items():
        candidate = {field: report.get(field) for field in fields}
        candidate_model = report.get("model", {})
        candidate["parameter_count"] = candidate_model.get("parameter_count")
        candidate["parameter_layout_blake3"] = candidate_model.get("parameter_layout_blake3")
        candidate["initial_parameter_tensor_blake3"] = candidate_model.get(
            "initial_parameter_tensor_blake3"
        )
        if candidate != common:
            raise R4ParentReportError(f"{arm} common scientific identity drifted")
    return common


def _validate_paired_controls(
    controls: list[dict[str, Any]],
    by_arm: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(controls) != 3:
        raise R4ParentReportError("ADR 0156 requires three host-paired C0 replays")
    output: dict[str, dict[str, Any]] = {}
    control_report = by_arm[CONTROL_ARM]
    for replay in controls:
        identity = replay.get("scientific_identity")
        arm = replay.get("treatment_arm")
        if (
            replay.get("schema_version") != 1
            or replay.get("experiment_id") != EXPERIMENT_ID
            or replay.get("protocol_id") != PROTOCOL_ID
            or replay.get("adr") != ADR_ID
            or arm not in ARMS[1:]
            or replay.get("host") != ARM_HOSTS[arm]
            or replay.get("control_arm") != CONTROL_ARM
            or replay.get("control_report_id") != control_report["report_id"]
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != replay.get("replay_id")
            or not isinstance(replay.get("performance"), dict)
            or arm in output
        ):
            raise R4ParentReportError("host-paired C0 replay is malformed or duplicated")
        absolute = _absolute_performance(replay["performance"])
        if not absolute["passed"]:
            raise R4ParentReportError(f"host-paired C0 replay failed absolute serving on {arm}")
        output[arm] = replay
    if set(output) != set(ARMS[1:]):
        raise R4ParentReportError("host-paired C0 replay coverage is incomplete")
    return output


def _quality(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "mae": float(metrics["r4800_value"]["mae"]),
        "rmse": float(metrics["r4800_value"]["rmse"]),
        "top64_recall": float(metrics["top64_r4800_winner_recall"]),
        "top64_regret": float(metrics["mean_top64_retained_r4800_regret"]),
        "low_supply_recall": float(metrics["subsets"]["low_supply"]["top64_r4800_winner_recall"]),
        "independent_recall": float(
            metrics["subsets"]["independent_draft_winner"]["top64_r4800_winner_recall"]
        ),
        "coverage": float(metrics["top64_confidence_set_coverage_95"]),
    }


def _absolute_serving(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    checks = _absolute_performance(report["performance"])
    checks["checks"]["metrics_complete_groups"] = metrics.get("groups") == 240
    checks["checks"]["metrics_complete_actions"] = metrics.get("candidates") == 860_203
    checks["checks"]["all_groups_scored_once"] = metrics.get("all_groups_scored_once") is True
    checks["checks"]["all_candidates_scored_once"] = (
        metrics.get("all_candidates_scored_once") is True
    )
    checks["checks"]["finite"] = metrics.get("all_scores_and_uncertainties_finite") is True
    checks["checks"]["parent_encode_count"] = metrics.get("parent_encodes") == 240
    checks["passed"] = all(checks["checks"].values())
    return checks


def _absolute_performance(performance: dict[str, Any]) -> dict[str, Any]:
    complete = performance["complete_decisions"]
    fixed = performance["fixed_chunk"]
    memory = performance["memory"]
    checks = {
        "complete_groups": complete.get("groups") == ABSOLUTE_SERVING["groups"],
        "complete_actions": complete.get("actions") == ABSOLUTE_SERVING["actions"],
        "parent_encodes": complete.get("parent_encodes") == ABSOLUTE_SERVING["groups"],
        "parent_encode_count_exact": complete.get("parent_encode_count_exact") is True,
        "process_swap": memory.get("process_swaps") == 0
        and (memory.get("system_swap_delta_bytes") in (None, 0)),
        "active_memory": memory.get("peak_active_bytes", float("inf"))
        <= ABSOLUTE_SERVING["memory_max"],
        "rss": memory.get("peak_process_rss_bytes", float("inf")) <= ABSOLUTE_SERVING["memory_max"],
        "p99_latency": complete["latency_milliseconds"]["p99"] <= ABSOLUTE_SERVING["p99_ms_max"],
        "fixed_throughput": fixed.get("action_scores_per_second", 0.0)
        >= ABSOLUTE_SERVING["throughput_min"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def _efficiency(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    ratios = {
        "parent_p50": treatment["parent_encode"]["latency_milliseconds"]["p50"]
        / control["parent_encode"]["latency_milliseconds"]["p50"],
        "fixed_throughput": treatment["fixed_chunk"]["action_scores_per_second"]
        / control["fixed_chunk"]["action_scores_per_second"],
        "complete_p99": treatment["complete_decisions"]["latency_milliseconds"]["p99"]
        / control["complete_decisions"]["latency_milliseconds"]["p99"],
        "active_memory": treatment["memory"]["peak_active_bytes"]
        / control["memory"]["peak_active_bytes"],
        "rss": treatment["memory"]["peak_process_rss_bytes"]
        / control["memory"]["peak_process_rss_bytes"],
    }
    parent_passed = ratios["parent_p50"] <= 0.80
    end_to_end = {
        "fixed_throughput": ratios["fixed_throughput"] >= 1.05,
        "complete_p99": ratios["complete_p99"] <= 0.95,
        "active_memory": ratios["active_memory"] <= 0.85,
        "rss": ratios["rss"] <= 0.85,
    }
    return {
        "ratios": ratios,
        "parent_p50_passed": parent_passed,
        "end_to_end_checks": end_to_end,
        "end_to_end_passed": any(end_to_end.values()),
        "passed": parent_passed and any(end_to_end.values()),
    }


def _select_arm(
    eligible: list[str],
    assessments: dict[str, dict[str, Any]],
) -> str:
    best_throughput = max(
        assessments[arm]["efficiency"]["ratios"]["fixed_throughput"] for arm in eligible
    )
    tied = [
        arm
        for arm in eligible
        if assessments[arm]["efficiency"]["ratios"]["fixed_throughput"] >= best_throughput * 0.99
    ]

    def key(arm: str) -> tuple[float, float, float, int]:
        ratios = assessments[arm]["efficiency"]["ratios"]
        return (
            ratios["complete_p99"],
            ratios["active_memory"],
            ratios["rss"],
            TIE_ORDER[arm],
        )

    return min(tied, key=key)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R4ParentReportError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise R4ParentReportError(f"{path} must contain a JSON object")
    return value


def _path_identity(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        return f"{path.name}:unreadable:{type(error).__name__}"
    return f"{path.name}:{blake3.blake3(payload).hexdigest()}"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify the ADR 0156 comparison")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--paired-control", type=Path, action="append", required=True)
    parser.add_argument("--forward-output", type=Path, required=True)
    parser.add_argument("--reverse-output", type=Path, required=True)
    parser.add_argument("--order-proof-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    exit_code = 0
    try:
        reports = [_read_json(path) for path in args.report]
        controls = [_read_json(path) for path in args.paired_control]
        forward, reverse, proof = aggregate_with_order_proof(reports, controls)
    except (R4ParentReportError, KeyError, TypeError, ValueError) as error:
        forward, reverse, proof = invalid_outputs(
            error,
            args.report,
            args.paired_control,
        )
        exit_code = 2
    _write_json(args.forward_output, forward)
    _write_json(args.reverse_output, reverse)
    _write_json(args.order_proof_output, proof)
    print(
        json.dumps(
            {
                "classification": forward["classification"],
                "selected_arm": forward["selected_arm"],
                "aggregate_id": forward["aggregate_id"],
                "order_proof_id": proof["proof_id"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
