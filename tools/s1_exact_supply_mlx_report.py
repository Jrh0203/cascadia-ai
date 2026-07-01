#!/usr/bin/env python3
"""Deterministically classify the ADR 0147 learned exact-supply comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ADR_ID,
    ARM_INPUT_CONTRACTS,
    ARMS,
    EXPERIMENT_ID,
    NORMALIZATION_CONTRACT,
    PROTOCOL_ID,
    S1_D6_CONTRACT,
)
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
)
from cascadia_mlx.s1_exact_supply_mlx_train import (
    S1ExactSupplyTrainingProtocol,
)

CONTROL = ARMS[0]
EXACT = ARMS[1]
RELATIONAL = ARMS[2]
ARM_HOSTS = {
    CONTROL: "john1",
    EXACT: "john2",
    RELATIONAL: "john3",
}
REPLAY_CHECK_NAMES = frozenset(
    {
        "full_train_group_coverage",
        "full_train_candidate_coverage",
        "full_validation_group_coverage",
        "full_validation_candidate_coverage",
        "all_12_d6_inverse_round_trips",
        "supply_and_frontier_features_d6_invariant",
        "cross_arm_parameter_counts_equal",
        "cross_arm_parameter_count_frozen",
        "cross_arm_parameter_layouts_identical",
        "cross_arm_initial_weights_identical",
        "cache_hidden_information_boundary_clean",
    }
)

MAX_VALUE_MAE_DELTA = 0.05
MAX_VALUE_RMSE_DELTA = 0.05
MAX_CALIBRATION_SLOPE_ERROR_DELTA = 0.05
MAX_CALIBRATION_INTERCEPT_ABS_DELTA = 0.25
MIN_TOP64_RECALL_DELTA = 0.02
MIN_TOP64_REGRET_REDUCTION = 0.01
MIN_CONFIDENCE_COVERAGE = 0.995
MIN_SLICE_RECALL_DELTA = 0.02
MIN_REFILL_FIDELITY = 0.9999
MIN_T2_THROUGHPUT_FRACTION = 0.60
MAX_T2_MEMORY_MULTIPLIER = 1.50

CLASSIFICATION_INVALID = "exact_supply_learned_comparison_invalid_evidence"
CLASSIFICATION_CONTROL_FAILED = "exact_supply_learned_comparison_control_failed"
CLASSIFICATION_EXACT_FAILED = "exact_supply_learned_comparison_exact_representation_failed"
CLASSIFICATION_RELATIONAL_SUCCESS = "exact_supply_learned_comparison_relational_success"
CLASSIFICATION_RELATIONAL_NULL = "exact_supply_learned_comparison_relational_null"
EXIT_CODES = {
    CLASSIFICATION_RELATIONAL_SUCCESS: 0,
    CLASSIFICATION_RELATIONAL_NULL: 0,
    CLASSIFICATION_CONTROL_FAILED: 2,
    CLASSIFICATION_EXACT_FAILED: 3,
    CLASSIFICATION_INVALID: 4,
}


def classify_reports(
    reports: list[dict[str, Any]],
    replay: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Apply fail-closed identity, quality, refill, and performance gates."""
    errors: list[str] = []
    by_arm: dict[str, dict[str, Any]] = {}
    for report in reports:
        arm = report.get("arm")
        if not isinstance(arm, str):
            errors.append("arm report lacks a string arm")
        elif arm in by_arm:
            errors.append(f"duplicate arm report: {arm}")
        else:
            by_arm[arm] = report
    missing = sorted(set(ARMS) - set(by_arm))
    extra = sorted(set(by_arm) - set(ARMS))
    if missing:
        errors.append(f"missing arm reports: {missing}")
    if extra:
        errors.append(f"unexpected arm reports: {extra}")

    controlled_identity: dict[str, Any] | None = None
    normalized = []
    for arm in ARMS:
        report = by_arm.get(arm)
        if report is None:
            continue
        _validate_arm_report(report, arm, errors)
        identity = _controlled_identity(report)
        if controlled_identity is None:
            controlled_identity = identity
        elif identity != controlled_identity:
            errors.append(f"controlled scientific identity drifted for {arm}")
        normalized.append(_normalize_arm(report))
    _validate_replay(replay, controlled_identity, errors)

    gates: dict[str, bool] = {}
    comparisons: dict[str, Any] = {}
    if not errors and all(arm in by_arm for arm in ARMS):
        control = by_arm[CONTROL]
        exact = by_arm[EXACT]
        relational = by_arm[RELATIONAL]
        gates.update(_absolute_gates(control, prefix="c0"))
        gates.update(_absolute_gates(exact, prefix="t1"))
        gates.update(_absolute_gates(relational, prefix="t2"))
        comparisons["t1_vs_c0"] = _value_comparison(exact, control)
        comparisons["t2_vs_c0"] = _value_comparison(relational, control)
        comparisons["t2_ranking_vs_c0"] = _ranking_comparison(relational, control)
        comparisons["t2_performance_vs_c0"] = _performance_comparison(
            relational,
            control,
        )
        gates.update(
            {
                "t1_value_noninferior": comparisons["t1_vs_c0"]["value_noninferior"],
                "t2_value_noninferior": comparisons["t2_vs_c0"]["value_noninferior"],
                "t1_refill_fidelity_at_least_0_9999": (
                    exact["metrics"]["refill"]["mean_fidelity"]
                    >= MIN_REFILL_FIDELITY
                ),
                "t2_refill_fidelity_at_least_0_9999": (
                    relational["metrics"]["refill"]["mean_fidelity"]
                    >= MIN_REFILL_FIDELITY
                ),
                "t2_top64_recall_improves_by_0_02": (
                    comparisons["t2_ranking_vs_c0"]["top64_recall_delta"]
                    >= MIN_TOP64_RECALL_DELTA
                ),
                "t2_top64_regret_reduces_by_0_01": (
                    comparisons["t2_ranking_vs_c0"]["top64_regret_reduction"]
                    >= MIN_TOP64_REGRET_REDUCTION
                ),
                "t2_confidence_coverage_at_least_0_995": (
                    relational["metrics"]["top64_confidence_set_coverage_95"]
                    >= MIN_CONFIDENCE_COVERAGE
                ),
                "t2_low_supply_recall_improves_by_0_02": (
                    comparisons["t2_ranking_vs_c0"]["low_supply_recall_delta"]
                    >= MIN_SLICE_RECALL_DELTA
                ),
                "t2_independent_draft_recall_improves_by_0_02": (
                    comparisons["t2_ranking_vs_c0"][
                        "independent_draft_recall_delta"
                    ]
                    >= MIN_SLICE_RECALL_DELTA
                ),
                "t2_throughput_at_least_0_60_c0": (
                    comparisons["t2_performance_vs_c0"]["throughput_fraction"]
                    >= MIN_T2_THROUGHPUT_FRACTION
                ),
                "t2_active_memory_at_most_1_50_c0": (
                    comparisons["t2_performance_vs_c0"][
                        "active_memory_multiplier"
                    ]
                    <= MAX_T2_MEMORY_MULTIPLIER
                ),
                "t2_process_rss_at_most_1_50_c0": (
                    comparisons["t2_performance_vs_c0"]["rss_multiplier"]
                    <= MAX_T2_MEMORY_MULTIPLIER
                ),
                "independent_replay_passed": replay.get("passed") is True,
            }
        )

    if errors:
        classification = CLASSIFICATION_INVALID
    elif not all(value for name, value in gates.items() if name.startswith("c0_")):
        classification = CLASSIFICATION_CONTROL_FAILED
    elif not all(
        gates[name]
        for name in (
            "t1_value_noninferior",
            "t2_value_noninferior",
            "t1_refill_fidelity_at_least_0_9999",
            "t2_refill_fidelity_at_least_0_9999",
        )
    ) or not all(
        all(_absolute_gates(by_arm[arm], prefix=prefix).values())
        for arm, prefix in (
            (EXACT, "t1"),
            (RELATIONAL, "t2"),
        )
    ):
        classification = CLASSIFICATION_EXACT_FAILED
    elif all(gates.values()):
        classification = CLASSIFICATION_RELATIONAL_SUCCESS
    else:
        classification = CLASSIFICATION_RELATIONAL_NULL

    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "errors": sorted(set(errors)),
        "thresholds": {
            "max_value_mae_delta": MAX_VALUE_MAE_DELTA,
            "max_value_rmse_delta": MAX_VALUE_RMSE_DELTA,
            "max_calibration_slope_error_delta": (
                MAX_CALIBRATION_SLOPE_ERROR_DELTA
            ),
            "max_calibration_intercept_abs_delta": (
                MAX_CALIBRATION_INTERCEPT_ABS_DELTA
            ),
            "min_top64_recall_delta": MIN_TOP64_RECALL_DELTA,
            "min_top64_regret_reduction": MIN_TOP64_REGRET_REDUCTION,
            "min_confidence_coverage": MIN_CONFIDENCE_COVERAGE,
            "min_slice_recall_delta": MIN_SLICE_RECALL_DELTA,
            "min_refill_fidelity": MIN_REFILL_FIDELITY,
            "min_t2_throughput_fraction": MIN_T2_THROUGHPUT_FRACTION,
            "max_t2_memory_multiplier": MAX_T2_MEMORY_MULTIPLIER,
        },
        "gates": dict(sorted(gates.items())),
        "comparisons": comparisons,
        "arms": normalized,
        "replay_id": replay.get("replay_id"),
        "claims": {
            "offline_learned_comparison_complete": classification
            in {
                CLASSIFICATION_RELATIONAL_SUCCESS,
                CLASSIFICATION_RELATIONAL_NULL,
            },
            "gameplay_strength_measured": False,
            "production_gameplay_authorized": False,
            "model_promotion_authorized": False,
            "research_queue_mutated": False,
            "progress_to_100_claimed": False,
        },
    }
    result = {
        **identity,
        "aggregate_id": canonical_blake3(identity),
    }
    return result, EXIT_CODES[classification]


def _validate_arm_report(
    report: dict[str, Any],
    arm: str,
    errors: list[str],
) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("arm") != arm
        or report.get("host") != ARM_HOSTS[arm]
    ):
        errors.append(f"arm report envelope drifted for {arm}")
    if (
        not _is_blake3(report.get("cache_id"))
        or not _is_blake3(report.get("authorization_id"))
        or not _is_blake3(report.get("preflight_id"))
        or report.get("protocol") != S1ExactSupplyTrainingProtocol().to_dict()
    ):
        errors.append(f"arm report control identity drifted for {arm}")
    identity = report.get("scientific_identity")
    expected_identity = _arm_scientific_identity(report)
    if (
        not isinstance(identity, dict)
        or identity != expected_identity
        or canonical_blake3(expected_identity) != report.get("report_id")
    ):
        errors.append(f"arm report content address drifted for {arm}")
    model = report.get("model", {})
    counts = model.get("cross_arm_parameter_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(ARMS)
        or len(set(counts.values())) != 1
        or set(counts.values()) != {FROZEN_PARAMETER_COUNT}
        or model.get("parameter_count") != counts.get(arm)
        or model.get("config") != S1ExactSupplyModelConfig(arm=arm).to_dict()
        or model.get("parameter_count_scope") != "all-trainable-scalars"
    ):
        errors.append(f"parameter-budget equality failed for {arm}")
    layouts = model.get("cross_arm_parameter_layout_blake3")
    if (
        not isinstance(layouts, dict)
        or set(layouts) != set(ARMS)
        or len(set(layouts.values())) != 1
        or not all(_is_blake3(value) for value in layouts.values())
        or model.get("parameter_layout_blake3") != layouts.get(arm)
    ):
        errors.append(f"parameter-layout equality failed for {arm}")
    if (
        report.get("normalization") != NORMALIZATION_CONTRACT
        or report.get("input_contract") != ARM_INPUT_CONTRACTS[arm]
        or not _is_blake3(report.get("collision_witness_id"))
    ):
        errors.append(f"input or normalization contract drifted for {arm}")
    metrics = report.get("metrics", {})
    if (
        metrics.get("all_groups_scored_once") is not True
        or metrics.get("all_candidates_scored_once") is not True
        or metrics.get("all_scores_finite") is not True
        or metrics.get("groups") != 240
        or metrics.get("candidates") != 860_203
        or metrics.get("expected_groups") != 240
        or metrics.get("expected_candidates") != 860_203
    ):
        errors.append(f"validation coverage or score finiteness failed for {arm}")
    refill = metrics.get("refill", {})
    if refill.get("all_probabilities_finite") is not True:
        errors.append(f"refill probabilities are not finite for {arm}")
    required_numeric_paths = (
        ("metrics", "r4800_value", "mae"),
        ("metrics", "r4800_value", "rmse"),
        ("metrics", "r4800_value", "bias"),
        ("metrics", "r4800_value", "correlation"),
        ("metrics", "r4800_value", "calibration_slope"),
        ("metrics", "r4800_value", "calibration_intercept"),
        ("metrics", "training_objective"),
        ("metrics", "top1_r4800_winner_recall"),
        ("metrics", "top8_r4800_winner_recall"),
        ("metrics", "top32_r4800_winner_recall"),
        ("metrics", "top64_r4800_winner_recall"),
        ("metrics", "mean_top1_retained_r4800_regret"),
        ("metrics", "mean_top8_retained_r4800_regret"),
        ("metrics", "mean_top32_retained_r4800_regret"),
        ("metrics", "mean_top64_retained_r4800_regret"),
        ("metrics", "top64_confidence_set_coverage_95"),
        ("metrics", "refill", "mean_total_variation"),
        ("metrics", "refill", "p99_total_variation"),
        ("metrics", "refill", "mean_cross_entropy"),
        ("metrics", "refill", "mean_probability_mae"),
        ("metrics", "refill", "top1_mode_accuracy"),
        ("metrics", "refill", "mean_fidelity"),
        ("metrics", "subsets", "low_supply", "groups"),
        ("metrics", "subsets", "low_supply", "top64_r4800_winner_recall"),
        (
            "metrics",
            "subsets",
            "low_supply",
            "top64_confidence_set_coverage_95",
        ),
        (
            "metrics",
            "subsets",
            "low_supply",
            "mean_top64_retained_r4800_regret",
        ),
        ("metrics", "subsets", "independent_draft_winner", "groups"),
        (
            "metrics",
            "subsets",
            "independent_draft_winner",
            "top64_r4800_winner_recall",
        ),
        (
            "metrics",
            "subsets",
            "independent_draft_winner",
            "top64_confidence_set_coverage_95",
        ),
        (
            "metrics",
            "subsets",
            "independent_draft_winner",
            "mean_top64_retained_r4800_regret",
        ),
        ("performance", "groups"),
        ("performance", "actions"),
        ("performance", "elapsed_seconds"),
        ("performance", "action_scores_per_second"),
        ("performance", "mean_decision_milliseconds"),
        ("performance", "p99_decision_milliseconds"),
        ("performance", "peak_active_memory_bytes"),
        ("performance", "peak_process_rss_bytes"),
        ("performance", "process_swaps"),
        ("performance", "system_swap_delta_bytes"),
    )
    if any(not _finite_number(_nested(report, path)) for path in required_numeric_paths):
        errors.append(f"required metric or performance evidence is missing for {arm}")
    probability_paths = (
        ("metrics", "top1_r4800_winner_recall"),
        ("metrics", "top8_r4800_winner_recall"),
        ("metrics", "top32_r4800_winner_recall"),
        ("metrics", "top64_r4800_winner_recall"),
        ("metrics", "top64_confidence_set_coverage_95"),
        ("metrics", "refill", "top1_mode_accuracy"),
        ("metrics", "refill", "mean_fidelity"),
        ("metrics", "subsets", "low_supply", "top64_r4800_winner_recall"),
        (
            "metrics",
            "subsets",
            "low_supply",
            "top64_confidence_set_coverage_95",
        ),
        (
            "metrics",
            "subsets",
            "independent_draft_winner",
            "top64_r4800_winner_recall",
        ),
        (
            "metrics",
            "subsets",
            "independent_draft_winner",
            "top64_confidence_set_coverage_95",
        ),
    )
    if any(
        not 0.0 <= float(_nested(report, path)) <= 1.0
        for path in probability_paths
        if _finite_number(_nested(report, path))
    ):
        errors.append(f"probability metric is out of range for {arm}")
    performance = report.get("performance", {})
    if (
        performance.get("groups") != 240
        or performance.get("actions") != 860_203
        or not _positive_number(performance.get("action_scores_per_second"))
        or not _nonnegative_number(performance.get("elapsed_seconds"))
        or not _nonnegative_number(
            performance.get("mean_decision_milliseconds")
        )
        or not _nonnegative_number(
            performance.get("p99_decision_milliseconds")
        )
        or not _nonnegative_number(performance.get("peak_active_memory_bytes"))
        or not _nonnegative_number(performance.get("peak_process_rss_bytes"))
        or not _nonnegative_number(performance.get("process_swaps"))
    ):
        errors.append(f"performance coverage or range failed for {arm}")
    checkpoint = report.get("checkpoint", {})
    if (
        not isinstance(checkpoint.get("path"), str)
        or not _is_blake3(checkpoint.get("manifest_blake3"))
        or not _is_blake3(checkpoint.get("model_blake3"))
    ):
        errors.append(f"checkpoint identity failed for {arm}")
    boundary = report.get("information_boundary", {})
    if (
        boundary.get("open_train_used") is not True
        or boundary.get("open_validation_used") is not True
        or boundary.get("sealed_test_opened") is not False
        or boundary.get("gameplay_run") is not False
        or boundary.get("hidden_order_read") is not False
    ):
        errors.append(f"information boundary failed for {arm}")
    claims = report.get("claims", {})
    if (
        claims.get("offline_comparison_complete") is not True
        or claims.get("gameplay_strength_measured") is not False
        or claims.get("promotion_authorized") is not False
        or claims.get("progress_to_100_claimed") is not False
    ):
        errors.append(f"claim boundary failed for {arm}")
    if not _all_finite(report):
        errors.append(f"arm report contains nonfinite values: {arm}")


def _validate_replay(
    replay: dict[str, Any],
    controlled: dict[str, Any] | None,
    errors: list[str],
) -> None:
    identity = replay.get("scientific_identity")
    if (
        replay.get("schema_version") != 1
        or replay.get("experiment_id") != EXPERIMENT_ID
        or replay.get("protocol_id") != PROTOCOL_ID
        or replay.get("adr") != ADR_ID
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != replay.get("replay_id")
        or replay.get("passed") is not True
        or identity.get("host") != "john4"
        or identity.get("role") != "independent-replay-control"
        or identity.get("cache_id") != replay.get("cache_id")
        or identity.get("authorization_id") != replay.get("authorization_id")
        or identity.get("checks") != replay.get("checks")
        or identity.get("sealed_test_opened") is not False
        or identity.get("gameplay_run") is not False
        or identity.get("training_run") is not False
    ):
        errors.append("independent replay/control report is invalid")
        return
    checks = replay.get("checks", {})
    if (
        not isinstance(checks, dict)
        or set(checks) != REPLAY_CHECK_NAMES
        or any(value is not True for value in checks.values())
    ):
        errors.append("independent replay/control checks did not all pass")
    coverage = identity.get("coverage", {}) if isinstance(identity, dict) else {}
    counts = identity.get("parameter_counts", {}) if isinstance(identity, dict) else {}
    layouts = (
        identity.get("parameter_layout_blake3", {})
        if isinstance(identity, dict)
        else {}
    )
    fingerprints = (
        identity.get("initial_weight_fingerprints", {})
        if isinstance(identity, dict)
        else {}
    )
    if (
        coverage
        != {
            "train": {"groups": 560, "candidates": 2_135_111},
            "validation": {"groups": 240, "candidates": 860_203},
        }
        or identity.get("d6_contract") != S1_D6_CONTRACT
        or identity.get("d6_round_trips") != 12
        or not isinstance(counts, dict)
        or set(counts) != set(ARMS)
        or set(counts.values()) != {FROZEN_PARAMETER_COUNT}
        or not isinstance(layouts, dict)
        or set(layouts) != set(ARMS)
        or len(set(layouts.values())) != 1
        or not all(_is_blake3(value) for value in layouts.values())
        or not isinstance(fingerprints, dict)
        or set(fingerprints) != set(ARMS)
        or len(set(fingerprints.values())) != 1
        or not all(_is_blake3(value) for value in fingerprints.values())
    ):
        errors.append("independent replay/control scientific structure drifted")
    if controlled is not None and (
        replay.get("cache_id") != controlled["cache_id"]
        or replay.get("authorization_id") != controlled["authorization_id"]
    ):
        errors.append("independent replay/control identity disagrees with arm reports")


def _controlled_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_id": report.get("cache_id"),
        "authorization_id": report.get("authorization_id"),
        "protocol": report.get("protocol"),
        "normalization": report.get("normalization"),
        "collision_witness_id": report.get("collision_witness_id"),
        "parameter_counts": report.get("model", {}).get(
            "cross_arm_parameter_counts"
        ),
        "parameter_layouts": report.get("model", {}).get(
            "cross_arm_parameter_layout_blake3"
        ),
    }


def _arm_scientific_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "arm",
            "host",
            "cache_id",
            "authorization_id",
            "preflight_id",
            "protocol",
            "normalization",
            "input_contract",
            "collision_witness_id",
            "model",
            "checkpoint",
            "metrics",
            "performance",
            "information_boundary",
        )
    }


def _absolute_gates(report: dict[str, Any], *, prefix: str) -> dict[str, bool]:
    performance = report["performance"]
    return {
        f"{prefix}_actions_per_second_at_least_20000": (
            performance["action_scores_per_second"] >= 20_000.0
        ),
        f"{prefix}_p99_at_most_250ms": (
            performance["p99_decision_milliseconds"] <= 250.0
        ),
        f"{prefix}_rss_at_most_4gib": (
            performance["peak_process_rss_bytes"] <= 4 * 1024**3
        ),
        f"{prefix}_active_memory_at_most_4gib": (
            performance["peak_active_memory_bytes"] <= 4 * 1024**3
        ),
        f"{prefix}_process_swaps_zero": performance["process_swaps"] == 0,
        f"{prefix}_system_swap_not_consumed": (
            performance["system_swap_delta_bytes"] is not None
            and performance["system_swap_delta_bytes"] <= 0
        ),
    }


def _value_comparison(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    treatment_value = treatment["metrics"]["r4800_value"]
    control_value = control["metrics"]["r4800_value"]
    mae_delta = treatment_value["mae"] - control_value["mae"]
    rmse_delta = treatment_value["rmse"] - control_value["rmse"]
    slope_error_delta = abs(treatment_value["calibration_slope"] - 1.0) - abs(
        control_value["calibration_slope"] - 1.0
    )
    intercept_delta = abs(treatment_value["calibration_intercept"]) - abs(
        control_value["calibration_intercept"]
    )
    return {
        "mae_delta": mae_delta,
        "rmse_delta": rmse_delta,
        "calibration_slope_error_delta": slope_error_delta,
        "calibration_intercept_abs_delta": intercept_delta,
        "value_noninferior": (
            mae_delta <= MAX_VALUE_MAE_DELTA
            and rmse_delta <= MAX_VALUE_RMSE_DELTA
            and slope_error_delta <= MAX_CALIBRATION_SLOPE_ERROR_DELTA
            and intercept_delta <= MAX_CALIBRATION_INTERCEPT_ABS_DELTA
        ),
    }


def _ranking_comparison(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, float]:
    treatment_metrics = treatment["metrics"]
    control_metrics = control["metrics"]
    return {
        "top64_recall_delta": (
            treatment_metrics["top64_r4800_winner_recall"]
            - control_metrics["top64_r4800_winner_recall"]
        ),
        "top64_regret_reduction": (
            control_metrics["mean_top64_retained_r4800_regret"]
            - treatment_metrics["mean_top64_retained_r4800_regret"]
        ),
        "confidence_coverage_delta": (
            treatment_metrics["top64_confidence_set_coverage_95"]
            - control_metrics["top64_confidence_set_coverage_95"]
        ),
        "low_supply_recall_delta": (
            treatment_metrics["subsets"]["low_supply"][
                "top64_r4800_winner_recall"
            ]
            - control_metrics["subsets"]["low_supply"][
                "top64_r4800_winner_recall"
            ]
        ),
        "independent_draft_recall_delta": (
            treatment_metrics["subsets"]["independent_draft_winner"][
                "top64_r4800_winner_recall"
            ]
            - control_metrics["subsets"]["independent_draft_winner"][
                "top64_r4800_winner_recall"
            ]
        ),
    }


def _performance_comparison(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, float]:
    treatment_performance = treatment["performance"]
    control_performance = control["performance"]
    return {
        "throughput_fraction": (
            treatment_performance["action_scores_per_second"]
            / control_performance["action_scores_per_second"]
        ),
        "active_memory_multiplier": (
            treatment_performance["peak_active_memory_bytes"]
            / max(control_performance["peak_active_memory_bytes"], 1)
        ),
        "rss_multiplier": (
            treatment_performance["peak_process_rss_bytes"]
            / max(control_performance["peak_process_rss_bytes"], 1)
        ),
    }


def _normalize_arm(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": report.get("arm"),
        "host": report.get("host"),
        "report_id": report.get("report_id"),
        "cache_id": report.get("cache_id"),
        "parameter_count": report.get("model", {}).get("parameter_count"),
        "metrics": report.get("metrics"),
        "performance": report.get("performance"),
    }


def canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def compare_outputs(forward: Path, reverse: Path) -> dict[str, Any]:
    forward_bytes = forward.read_bytes()
    reverse_bytes = reverse.read_bytes()
    if forward_bytes != reverse_bytes:
        raise ValueError("S1 forward and reverse classifications are not byte-identical")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "byte_identical": True,
        "classification_blake3": blake3.blake3(forward_bytes).hexdigest(),
        "promotion_authorized": False,
    }


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


def _is_blake3(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nested(value: object, path: tuple[str, ...]) -> object:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _nonnegative_number(value: object) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    classify = subparsers.add_parser("classify")
    classify.add_argument("--report", type=Path, action="append", required=True)
    classify.add_argument("--replay", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--forward", type=Path, required=True)
    compare.add_argument("--reverse", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "compare":
            result = compare_outputs(args.forward, args.reverse)
            _write_json_atomic(args.output, result)
            print(json.dumps(result, sort_keys=True))
            return 0
        reports = [_read_json(path, "S1 arm report") for path in args.report]
        replay = _read_json(args.replay, "S1 replay report")
        result, exit_code = classify_reports(reports, replay)
        _write_json_atomic(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return command_exit_code(exit_code)
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_CODES[CLASSIFICATION_INVALID]


def command_exit_code(classification_exit_code: int) -> int:
    """Reserve command failure for malformed evidence, not a valid rejection."""
    if classification_exit_code == EXIT_CODES[CLASSIFICATION_INVALID]:
        return classification_exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
