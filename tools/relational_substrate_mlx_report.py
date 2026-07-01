#!/usr/bin/env python3
"""Classify the frozen ADR 0161 relational-substrate MLX tournament."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.relational_substrate_mlx_cache import (
    ADR_ID,
    ARMS,
    CONTROL_ARM,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)
from cascadia_mlx.relational_substrate_mlx_train import (
    ARM_HOSTS,
    TRAINING_STEPS,
)

CLASSIFICATION_INVALID = "relational_substrate_mlx_invalid"
CLASSIFICATION_CONTROL_FAILED = "relational_substrate_mlx_control_failed"
CLASSIFICATION_ALL_DEGRADED = (
    "relational_substrate_mlx_all_treatments_degraded"
)
CLASSIFICATION_QUALITY_ONLY_NULL = (
    "relational_substrate_mlx_quality_only_null"
)
CLASSIFICATION_SELECTED = "relational_substrate_mlx_selected"

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
    "strategic_mean_delta_min": 0.015,
    "strategic_family_delta_min": -0.01,
}
ABSOLUTE_SERVING = {
    "groups": 240,
    "actions": 860_203,
    "memory_max": 4 * 1024**3,
    "p99_ms_max": 250.0,
    "throughput_min": 20_000.0,
}
MATERIAL_EFFICIENCY = {
    "throughput_ratio_min": 1.10,
    "p99_ratio_max": 0.90,
}
TIE_ORDER = {
    "q1-r5-quotient-local": 0,
    "g2-r5-s3": 1,
    "d3-r5-s3-s5": 2,
}


class RelationalSubstrateReportError(ValueError):
    """ADR 0161 evidence is incomplete or internally inconsistent."""


def aggregate_with_order_proof(
    reports: list[dict[str, Any]],
    paired_controls: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    forward = classify_reports(reports, paired_controls)
    reverse = classify_reports(
        list(reversed(reports)),
        list(reversed(paired_controls)),
    )
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
        raise RelationalSubstrateReportError(
            "ADR 0161 classification depends on input order"
        )
    return forward, reverse, proof


def invalid_outputs(
    error: Exception,
    report_paths: list[Path],
    paired_control_paths: list[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs = {
        "reports": sorted(_path_identity(path) for path in report_paths),
        "paired_controls": sorted(
            _path_identity(path) for path in paired_control_paths
        ),
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
            "relational_substrate_may_be_selected": False,
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
        "forward_blake3": blake3.blake3(
            _canonical_bytes(identity)
        ).hexdigest(),
        "reverse_blake3": blake3.blake3(
            _canonical_bytes(identity)
        ).hexdigest(),
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
    control_serving_integrity = _baseline_performance_integrity(
        control["performance"]
    )
    control_sanity = {
        "mae": control_quality["mae"] <= CONTROL_LIMITS["mae_max"],
        "rmse": control_quality["rmse"] <= CONTROL_LIMITS["rmse_max"],
        "top64_recall": (
            control_quality["top64_recall"]
            >= CONTROL_LIMITS["top64_recall_min"]
        ),
        "top64_regret": (
            control_quality["top64_regret"]
            <= CONTROL_LIMITS["top64_regret_max"]
        ),
        "low_supply_recall": (
            control_quality["low_supply_recall"]
            >= CONTROL_LIMITS["low_supply_recall_min"]
        ),
        "independent_recall": (
            control_quality["independent_recall"]
            >= CONTROL_LIMITS["independent_recall_min"]
        ),
        "coverage": (
            control_quality["coverage"] >= CONTROL_LIMITS["coverage_min"]
        ),
        "serving_integrity": control_serving_integrity["passed"],
    }

    assessments: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    quality_passing = 0
    for arm in ARMS[1:]:
        treatment = by_arm[arm]
        quality = _quality(treatment)
        serving = _absolute_serving(treatment)
        quality_delta = {
            key: quality[key] - control_quality[key]
            for key in (
                "mae",
                "rmse",
                "top64_recall",
                "top64_regret",
                "low_supply_recall",
                "independent_recall",
                "strategic_mean",
                "elk_recall",
                "salmon_recall",
                "hawk_recall",
                "bear_recall",
            )
        }
        quality_checks = {
            "mae": (
                quality_delta["mae"]
                <= QUALITY_LIMITS["mae_delta_max"]
            ),
            "rmse": (
                quality_delta["rmse"]
                <= QUALITY_LIMITS["rmse_delta_max"]
            ),
            "top64_recall": (
                quality_delta["top64_recall"]
                >= QUALITY_LIMITS["top64_recall_delta_min"]
            ),
            "top64_regret": (
                quality_delta["top64_regret"]
                <= QUALITY_LIMITS["top64_regret_delta_max"]
            ),
            "low_supply_recall": (
                quality_delta["low_supply_recall"]
                >= QUALITY_LIMITS["low_supply_recall_delta_min"]
            ),
            "independent_recall": (
                quality_delta["independent_recall"]
                >= QUALITY_LIMITS["independent_recall_delta_min"]
            ),
            "coverage": quality["coverage"] >= QUALITY_LIMITS["coverage_min"],
            "strategic_mean": (
                quality_delta["strategic_mean"]
                >= QUALITY_LIMITS["strategic_mean_delta_min"]
            ),
            "elk_recall": (
                quality_delta["elk_recall"]
                >= QUALITY_LIMITS["strategic_family_delta_min"]
            ),
            "salmon_recall": (
                quality_delta["salmon_recall"]
                >= QUALITY_LIMITS["strategic_family_delta_min"]
            ),
            "hawk_recall": (
                quality_delta["hawk_recall"]
                >= QUALITY_LIMITS["strategic_family_delta_min"]
            ),
        }
        quality_passed = all(quality_checks.values())
        quality_passing += int(quality_passed)
        host_control = paired[arm]["performance"]
        efficiency = _efficiency(treatment["performance"], host_control)
        eligible_arm = quality_passed and serving["passed"] and efficiency[
            "passed"
        ]
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
        "report_ids": {
            arm: by_arm[arm]["report_id"] for arm in ARMS
        },
        "paired_control_replay_ids": {
            arm: paired[arm]["replay_id"] for arm in ARMS[1:]
        },
        "common_identity": _common_identity(by_arm),
        "control": {
            "quality": control_quality,
            "sanity_checks": control_sanity,
            "absolute_serving": control_serving,
            "serving_integrity": control_serving_integrity,
        },
        "treatments": assessments,
        "limits": {
            "control": CONTROL_LIMITS,
            "quality": QUALITY_LIMITS,
            "absolute_serving": ABSOLUTE_SERVING,
            "material_efficiency": MATERIAL_EFFICIENCY,
        },
        "claim_boundary": {
            "relational_substrate_may_be_selected": (
                classification == CLASSIFICATION_SELECTED
            ),
            "paired_gameplay_qualification_authorized": (
                classification == CLASSIFICATION_SELECTED
            ),
            "champion_changed": False,
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


def _validate_reports(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(reports) != len(ARMS):
        raise RelationalSubstrateReportError(
            "ADR 0161 requires exactly four arm reports"
        )
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
            or report.get("information_boundary", {}).get(
                "sealed_test_opened"
            )
            is not False
            or report.get("information_boundary", {}).get("gameplay_run")
            is not False
            or report.get("claims", {}).get(
                "offline_comparison_complete"
            )
            is not True
            or report.get("claims", {}).get("promotion_authorized")
            is not False
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != report.get("report_id")
            or arm in by_arm
        ):
            raise RelationalSubstrateReportError(
                "ADR 0161 arm report is malformed or duplicated"
            )
        optimization = report.get("optimization")
        trace = (
            optimization.get("loss_trace")
            if isinstance(optimization, dict)
            else None
        )
        if (
            not isinstance(trace, list)
            or len(trace) != TRAINING_STEPS
            or optimization.get("global_step") != TRAINING_STEPS
        ):
            raise RelationalSubstrateReportError(
                f"{arm} training trace is incomplete"
            )
        normalized = []
        for step, event in enumerate(trace, start=1):
            if (
                not isinstance(event, dict)
                or event.get("step") != step
                or not isinstance(event.get("batch_blake3"), str)
                or len(event["batch_blake3"]) != 64
                or not isinstance(event.get("candidates"), int)
                or not _finite(event.get("loss"))
                or not _finite(event.get("elapsed_seconds"))
            ):
                raise RelationalSubstrateReportError(
                    f"{arm} training trace is malformed"
                )
            normalized.append(
                (step, event["batch_blake3"], event["candidates"])
            )
        batch_traces[arm] = normalized
        by_arm[arm] = report
    if set(by_arm) != set(ARMS):
        raise RelationalSubstrateReportError(
            "ADR 0161 arm report coverage is incomplete"
        )
    if len({tuple(value) for value in batch_traces.values()}) != 1:
        raise RelationalSubstrateReportError(
            "ADR 0161 arms did not consume identical scientific batches"
        )
    _common_identity(by_arm)
    return by_arm


def _common_identity(
    by_arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "r3_cache_id",
        "relational_cache_id",
        "s1_cache_id",
        "protocol",
    )
    common = {
        field: by_arm[CONTROL_ARM].get(field) for field in fields
    }
    model = by_arm[CONTROL_ARM].get("model", {})
    common.update(
        {
            "parameter_count": model.get("parameter_count"),
            "parameter_layout_blake3": model.get(
                "parameter_layout_blake3"
            ),
            "initial_parameter_tensor_blake3": model.get(
                "initial_parameter_tensor_blake3"
            ),
        }
    )
    common.update(_shared_report_provenance(by_arm[CONTROL_ARM]))
    common["serving_binary_contract"] = "host-paired-c0-replay-v1"
    for arm, report in by_arm.items():
        candidate = {field: report.get(field) for field in fields}
        candidate_model = report.get("model", {})
        candidate.update(
            {
                "parameter_count": candidate_model.get(
                    "parameter_count"
                ),
                "parameter_layout_blake3": candidate_model.get(
                    "parameter_layout_blake3"
                ),
                "initial_parameter_tensor_blake3": candidate_model.get(
                    "initial_parameter_tensor_blake3"
                ),
            }
        )
        candidate.update(_shared_report_provenance(report))
        candidate["serving_binary_contract"] = (
            "host-paired-c0-replay-v1"
        )
        if candidate != common:
            raise RelationalSubstrateReportError(
                f"{arm} common scientific identity drifted"
            )
    return common


def _shared_report_provenance(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("source")
    controls = report.get("controls")
    if not isinstance(source, dict) or not isinstance(controls, dict):
        raise RelationalSubstrateReportError(
            "ADR 0161 report provenance or launch controls are absent"
        )
    return {
        "source_blake3": source.get("v2_source_blake3"),
        "authorization_id": controls.get("authorization_id"),
        "open_data_verification_id": controls.get(
            "open_data_verification_id"
        ),
    }


def _validate_paired_controls(
    controls: list[dict[str, Any]],
    by_arm: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(controls) != len(ARMS) - 1:
        raise RelationalSubstrateReportError(
            "ADR 0161 requires three host-paired C0 replays"
        )
    output: dict[str, dict[str, Any]] = {}
    control_report = by_arm[CONTROL_ARM]
    control_checkpoint = control_report.get("checkpoint")
    control_controls = control_report.get("controls")
    if not isinstance(control_checkpoint, dict) or not isinstance(
        control_controls,
        dict,
    ):
        raise RelationalSubstrateReportError(
            "ADR 0161 control checkpoint or launch controls are absent"
        )
    required_assertions = {
        "control_checkpoint_manifest_identical",
        "control_checkpoint_model_identical",
        "r6_binary_identical",
        "replay_host_is_treatment_host",
        "replay_host_differs_from_control_host",
        "isolated_process",
        "open_data_reverified",
        "all_validation_decisions_measured",
        "all_validation_actions_measured",
        "r6_apply_undo_exact",
    }
    for replay in controls:
        identity = replay.get("scientific_identity")
        arm = replay.get("treatment_arm")
        treatment_report = by_arm.get(arm)
        treatment_r6 = (
            treatment_report.get("r6_binary")
            if isinstance(treatment_report, dict)
            else None
        )
        replay_checkpoint = (
            identity.get("checkpoint")
            if isinstance(identity, dict)
            else None
        )
        assertions = (
            identity.get("assertions")
            if isinstance(identity, dict)
            else None
        )
        if (
            replay.get("schema_version") != 1
            or replay.get("experiment_id") != EXPERIMENT_ID
            or replay.get("protocol_id") != PROTOCOL_ID
            or replay.get("adr") != ADR_ID
            or arm not in ARMS[1:]
            or replay.get("host") != ARM_HOSTS[arm]
            or replay.get("control_arm") != CONTROL_ARM
            or replay.get("control_report_id")
            != control_report["report_id"]
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != replay.get("replay_id")
            or not isinstance(replay.get("performance"), dict)
            or not isinstance(treatment_r6, dict)
            or identity.get("experiment_id") != EXPERIMENT_ID
            or identity.get("protocol_id") != PROTOCOL_ID
            or identity.get("adr") != ADR_ID
            or identity.get("treatment_arm") != arm
            or identity.get("host") != ARM_HOSTS[arm]
            or identity.get("control_arm") != CONTROL_ARM
            or identity.get("control_report_id")
            != control_report["report_id"]
            or identity.get("r3_cache_id")
            != control_report.get("r3_cache_id")
            or identity.get("relational_cache_id")
            != control_report.get("relational_cache_id")
            or identity.get("s1_cache_id")
            != control_report.get("s1_cache_id")
            or identity.get("authorization_id")
            != control_controls.get("authorization_id")
            or identity.get("open_data_verification_id")
            != control_controls.get("open_data_verification_id")
            or not isinstance(replay_checkpoint, dict)
            or replay_checkpoint.get("manifest_blake3")
            != control_checkpoint.get("manifest_blake3")
            or replay_checkpoint.get("model_blake3")
            != control_checkpoint.get("model_blake3")
            or replay_checkpoint.get("global_step") != TRAINING_STEPS
            or identity.get("r6_binary_blake3")
            != treatment_r6.get("blake3")
            or not isinstance(assertions, dict)
            or not required_assertions.issubset(assertions)
            or any(assertions[key] is not True for key in required_assertions)
            or arm in output
        ):
            raise RelationalSubstrateReportError(
                "host-paired C0 replay is malformed or duplicated"
            )
        integrity = _baseline_performance_integrity(
            replay["performance"]
        )
        if not integrity["passed"]:
            raise RelationalSubstrateReportError(
                f"host-paired C0 replay failed integrity gates on {arm}"
            )
        output[arm] = replay
    if set(output) != set(ARMS[1:]):
        raise RelationalSubstrateReportError(
            "host-paired C0 replay coverage is incomplete"
        )
    return output


def _quality(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["metrics"]
    strategic = metrics["strategic_opportunity_recall"]
    return {
        "mae": float(metrics["r4800_value"]["mae"]),
        "rmse": float(metrics["r4800_value"]["rmse"]),
        "top64_recall": float(metrics["top64_r4800_winner_recall"]),
        "top64_regret": float(
            metrics["mean_top64_retained_r4800_regret"]
        ),
        "low_supply_recall": float(
            metrics["subsets"]["low_supply"][
                "top64_r4800_winner_recall"
            ]
        ),
        "independent_recall": float(
            metrics["subsets"]["independent_draft_winner"][
                "top64_r4800_winner_recall"
            ]
        ),
        "coverage": float(metrics["top64_confidence_set_coverage_95"]),
        "strategic_mean": float(strategic["primary_mean"]),
        "elk_recall": float(strategic["elk"]),
        "salmon_recall": float(strategic["salmon"]),
        "hawk_recall": float(strategic["hawk"]),
        "bear_recall": float(strategic["bear_diagnostic"]),
    }


def _absolute_serving(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    result = _absolute_performance(report["performance"])
    result["checks"].update(
        {
            "metrics_complete_groups": (
                metrics.get("groups") == ABSOLUTE_SERVING["groups"]
            ),
            "metrics_complete_actions": (
                metrics.get("candidates") == ABSOLUTE_SERVING["actions"]
            ),
            "all_groups_scored_once": (
                metrics.get("all_groups_scored_once") is True
            ),
            "all_candidates_scored_once": (
                metrics.get("all_candidates_scored_once") is True
            ),
            "finite": (
                metrics.get("all_scores_and_uncertainties_finite") is True
            ),
            "parent_encode_count": (
                metrics.get("parent_encodes")
                == ABSOLUTE_SERVING["groups"]
            ),
        }
    )
    result["passed"] = all(result["checks"].values())
    return result


def _absolute_performance(
    performance: dict[str, Any],
) -> dict[str, Any]:
    combined = performance["combined_with_r6"]
    fixed = performance["fixed_chunk"]
    memory = performance["memory"]
    r6 = performance["r6_apply_undo"]
    checks = {
        "complete_groups": (
            combined.get("groups") == ABSOLUTE_SERVING["groups"]
        ),
        "complete_actions": (
            combined.get("actions") == ABSOLUTE_SERVING["actions"]
        ),
        "r6_exact_parity": (
            combined.get("r6_exact_parity_pass") is True
            and r6.get("exact_parity_pass") is True
        ),
        "r6_apply_failures": r6.get("apply_failures") == 0,
        "r6_undo_failures": r6.get("undo_failures") == 0,
        "process_swap": (
            memory.get("process_swaps") == 0
            and _swap_did_not_grow(
                memory.get("system_swap_delta_bytes")
            )
        ),
        "active_memory": (
            memory.get("peak_active_bytes", float("inf"))
            <= ABSOLUTE_SERVING["memory_max"]
        ),
        "rss": (
            memory.get("peak_process_rss_bytes", float("inf"))
            <= ABSOLUTE_SERVING["memory_max"]
        ),
        "p99_latency": (
            combined["latency_milliseconds"]["p99"]
            <= ABSOLUTE_SERVING["p99_ms_max"]
        ),
        "fixed_throughput": (
            fixed.get("action_scores_per_second", 0.0)
            >= ABSOLUTE_SERVING["throughput_min"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "measurements": {
            "fixed_chunk_actions_per_second": fixed[
                "action_scores_per_second"
            ],
            "combined_actions_per_second": combined[
                "action_scores_per_second"
            ],
            "combined_p99_milliseconds": combined[
                "latency_milliseconds"
            ]["p99"],
            "peak_active_bytes": memory["peak_active_bytes"],
            "peak_process_rss_bytes": memory[
                "peak_process_rss_bytes"
            ],
        },
    }


def _baseline_performance_integrity(
    performance: dict[str, Any],
) -> dict[str, Any]:
    combined = performance["combined_with_r6"]
    fixed = performance["fixed_chunk"]
    memory = performance["memory"]
    r6 = performance["r6_apply_undo"]
    fixed_throughput = fixed.get("action_scores_per_second")
    combined_throughput = combined.get("action_scores_per_second")
    p99 = combined.get("latency_milliseconds", {}).get("p99")
    peak_active = memory.get("peak_active_bytes")
    peak_rss = memory.get("peak_process_rss_bytes")
    checks = {
        "complete_groups": (
            combined.get("groups") == ABSOLUTE_SERVING["groups"]
        ),
        "complete_actions": (
            combined.get("actions") == ABSOLUTE_SERVING["actions"]
        ),
        "r6_exact_parity": (
            combined.get("r6_exact_parity_pass") is True
            and r6.get("exact_parity_pass") is True
        ),
        "r6_apply_failures": r6.get("apply_failures") == 0,
        "r6_undo_failures": r6.get("undo_failures") == 0,
        "process_swap": (
            memory.get("process_swaps") == 0
            and _swap_did_not_grow(
                memory.get("system_swap_delta_bytes")
            )
        ),
        "fixed_throughput_finite_positive": (
            _finite(fixed_throughput) and fixed_throughput > 0
        ),
        "combined_throughput_finite_positive": (
            _finite(combined_throughput) and combined_throughput > 0
        ),
        "p99_latency_finite_nonnegative": (
            _finite(p99) and p99 >= 0
        ),
        "active_memory_finite_nonnegative": (
            _finite(peak_active) and peak_active >= 0
        ),
        "rss_finite_nonnegative": (
            _finite(peak_rss) and peak_rss >= 0
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "measurements": {
            "fixed_chunk_actions_per_second": fixed_throughput,
            "combined_actions_per_second": combined_throughput,
            "combined_p99_milliseconds": p99,
            "peak_active_bytes": peak_active,
            "peak_process_rss_bytes": peak_rss,
        },
    }


def _swap_did_not_grow(value: object) -> bool:
    return value is None or (_finite(value) and value <= 0)


def _efficiency(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    treatment_combined = treatment["combined_with_r6"]
    control_combined = control["combined_with_r6"]
    throughput_ratio = (
        treatment_combined["action_scores_per_second"]
        / control_combined["action_scores_per_second"]
    )
    p99_ratio = (
        treatment_combined["latency_milliseconds"]["p99"]
        / control_combined["latency_milliseconds"]["p99"]
    )
    checks = {
        "combined_throughput": (
            throughput_ratio
            >= MATERIAL_EFFICIENCY["throughput_ratio_min"]
        ),
        "combined_p99": (
            p99_ratio <= MATERIAL_EFFICIENCY["p99_ratio_max"]
        ),
    }
    return {
        "ratios": {
            "combined_throughput": throughput_ratio,
            "combined_p99": p99_ratio,
            "peak_active_memory": (
                treatment["memory"]["peak_active_bytes"]
                / control["memory"]["peak_active_bytes"]
            ),
            "peak_process_rss": (
                treatment["memory"]["peak_process_rss_bytes"]
                / control["memory"]["peak_process_rss_bytes"]
            ),
        },
        "checks": checks,
        "passed": any(checks.values()),
    }


def _select_arm(
    eligible: list[str],
    assessments: dict[str, dict[str, Any]],
) -> str:
    def key(arm: str) -> tuple[float, float, float, float, int]:
        assessment = assessments[arm]
        quality = assessment["quality"]
        efficiency = assessment["efficiency"]["ratios"]
        serving = assessment["absolute_serving"]
        return (
            -assessment["quality_delta"]["strategic_mean"],
            quality["top64_regret"],
            -efficiency["combined_throughput"],
            serving["measurements"]["peak_process_rss_bytes"],
            TIE_ORDER[arm],
        )

    return min(eligible, key=key)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


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
        raise RelationalSubstrateReportError(
            f"cannot read {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RelationalSubstrateReportError(
            f"{path} must contain a JSON object"
        )
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
    parser = argparse.ArgumentParser(
        description="Classify the ADR 0161 relational tournament"
    )
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--paired-control",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--forward-output", type=Path, required=True)
    parser.add_argument("--reverse-output", type=Path, required=True)
    parser.add_argument(
        "--order-proof-output",
        type=Path,
        required=True,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    exit_code = 0
    try:
        reports = [_read_json(path) for path in args.report]
        controls = [_read_json(path) for path in args.paired_control]
        forward, reverse, proof = aggregate_with_order_proof(
            reports,
            controls,
        )
    except (
        RelationalSubstrateReportError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
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
