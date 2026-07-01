from __future__ import annotations

import json
import random

import s1_exact_supply_mlx_report as report_tool
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ARM_INPUT_CONTRACTS,
    ARMS,
    NORMALIZATION_CONTRACT,
    S1_D6_CONTRACT,
)
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
)
from cascadia_mlx.s1_exact_supply_mlx_train import S1ExactSupplyTrainingProtocol


def _arm_report(
    arm: str,
    *,
    recall: float,
    regret: float,
    low_supply_recall: float,
    independent_recall: float,
) -> dict:
    host = report_tool.ARM_HOSTS[arm]
    layouts = dict.fromkeys(ARMS, "b" * 64)
    counts = dict.fromkeys(ARMS, FROZEN_PARAMETER_COUNT)
    report = {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "arm": arm,
        "host": host,
        "cache_id": "c" * 64,
        "authorization_id": "d" * 64,
        "preflight_id": "e" * 64,
        "protocol": S1ExactSupplyTrainingProtocol().to_dict(),
        "normalization": NORMALIZATION_CONTRACT,
        "input_contract": ARM_INPUT_CONTRACTS[arm],
        "collision_witness_id": "f" * 64,
        "model": {
            "config": S1ExactSupplyModelConfig(arm=arm).to_dict(),
            "parameter_count": FROZEN_PARAMETER_COUNT,
            "cross_arm_parameter_counts": counts,
            "parameter_layout_blake3": layouts[arm],
            "cross_arm_parameter_layout_blake3": layouts,
            "parameter_count_scope": "all-trainable-scalars",
        },
        "checkpoint": {
            "path": "/immutable/checkpoint",
            "manifest_blake3": "1" * 64,
            "model_blake3": "2" * 64,
        },
        "metrics": {
            "groups": 240,
            "candidates": 860_203,
            "expected_groups": 240,
            "expected_candidates": 860_203,
            "all_groups_scored_once": True,
            "all_candidates_scored_once": True,
            "all_scores_finite": True,
            "training_objective": 1.0,
            "r4800_value": {
                "mae": 1.0,
                "rmse": 1.2,
                "bias": 0.0,
                "correlation": 0.8,
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            },
            "top1_r4800_winner_recall": 0.2,
            "top8_r4800_winner_recall": 0.5,
            "top32_r4800_winner_recall": 0.7,
            "top64_r4800_winner_recall": recall,
            "mean_top1_retained_r4800_regret": 1.0,
            "mean_top8_retained_r4800_regret": 0.5,
            "mean_top32_retained_r4800_regret": 0.3,
            "mean_top64_retained_r4800_regret": regret,
            "top64_confidence_set_coverage_95": 0.999,
            "refill": {
                "groups": 240,
                "mean_total_variation": 0.0,
                "p99_total_variation": 0.0,
                "mean_cross_entropy": 1.0,
                "mean_probability_mae": 0.0,
                "top1_mode_accuracy": 1.0,
                "mean_fidelity": 1.0 if arm != ARMS[0] else 0.8,
                "all_probabilities_finite": True,
            },
            "subsets": {
                "low_supply": {
                    "groups": 20,
                    "top64_r4800_winner_recall": low_supply_recall,
                    "top64_confidence_set_coverage_95": 1.0,
                    "mean_top64_retained_r4800_regret": regret,
                },
                "independent_draft_winner": {
                    "groups": 20,
                    "top64_r4800_winner_recall": independent_recall,
                    "top64_confidence_set_coverage_95": 1.0,
                    "mean_top64_retained_r4800_regret": regret,
                },
            },
        },
        "performance": {
            "groups": 240,
            "actions": 860_203,
            "elapsed_seconds": 30.0,
            "action_scores_per_second": 30_000.0,
            "mean_decision_milliseconds": 10.0,
            "p99_decision_milliseconds": 100.0,
            "peak_active_memory_bytes": 1_000_000,
            "peak_process_rss_bytes": 2_000_000,
            "process_swaps": 0,
            "system_swap_delta_bytes": 0,
        },
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_read": False,
        },
        "claims": {
            "offline_comparison_complete": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = report_tool._arm_scientific_identity(report)
    report["report_id"] = report_tool.canonical_blake3(
        report["scientific_identity"]
    )
    return report


def _reports() -> list[dict]:
    return [
        _arm_report(
            ARMS[0],
            recall=0.80,
            regret=0.20,
            low_supply_recall=0.70,
            independent_recall=0.72,
        ),
        _arm_report(
            ARMS[1],
            recall=0.81,
            regret=0.19,
            low_supply_recall=0.71,
            independent_recall=0.73,
        ),
        _arm_report(
            ARMS[2],
            recall=0.83,
            regret=0.18,
            low_supply_recall=0.73,
            independent_recall=0.75,
        ),
    ]


def _replay() -> dict:
    checks = {
        "full_train_group_coverage": True,
        "full_train_candidate_coverage": True,
        "full_validation_group_coverage": True,
        "full_validation_candidate_coverage": True,
        "all_12_d6_inverse_round_trips": True,
        "supply_and_frontier_features_d6_invariant": True,
        "cross_arm_parameter_counts_equal": True,
        "cross_arm_parameter_count_frozen": True,
        "cross_arm_parameter_layouts_identical": True,
        "cross_arm_initial_weights_identical": True,
        "cache_hidden_information_boundary_clean": True,
    }
    identity = {
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "host": "john4",
        "role": "independent-replay-control",
        "bundle_id": "a" * 64,
        "authorization_id": "d" * 64,
        "cache_id": "c" * 64,
        "coverage": {
            "train": {"groups": 560, "candidates": 2_135_111},
            "validation": {"groups": 240, "candidates": 860_203},
        },
        "d6_contract": S1_D6_CONTRACT,
        "d6_round_trips": 12,
        "parameter_counts": dict.fromkeys(ARMS, FROZEN_PARAMETER_COUNT),
        "parameter_layout_blake3": dict.fromkeys(ARMS, "b" * 64),
        "initial_weight_fingerprints": dict.fromkeys(ARMS, "9" * 64),
        "checks": checks,
        "sealed_test_opened": False,
        "gameplay_run": False,
        "training_run": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "cache_id": "c" * 64,
        "authorization_id": "d" * 64,
        "scientific_identity": identity,
        "replay_id": report_tool.canonical_blake3(identity),
        "checks": checks,
        "passed": True,
    }


def test_success_classifier_is_order_invariant_and_never_promotes() -> None:
    reports = _reports()
    forward, exit_code = report_tool.classify_reports(reports, _replay())
    random.Random(17).shuffle(reports)
    reverse, reverse_code = report_tool.classify_reports(reports, _replay())
    assert exit_code == reverse_code == 0
    assert forward["classification"] == report_tool.CLASSIFICATION_RELATIONAL_SUCCESS
    assert json.dumps(forward, sort_keys=True) == json.dumps(reverse, sort_keys=True)
    assert forward["claims"]["model_promotion_authorized"] is False
    assert forward["claims"]["gameplay_strength_measured"] is False


def test_report_mutation_or_wrong_host_fails_closed() -> None:
    reports = _reports()
    reports[0]["metrics"]["r4800_value"]["mae"] = 9.0
    result, exit_code = report_tool.classify_reports(reports, _replay())
    assert exit_code == 4
    assert result["classification"] == report_tool.CLASSIFICATION_INVALID
    assert any("content address" in error for error in result["errors"])

    reports = _reports()
    reports[1]["host"] = "john3"
    reports[1]["scientific_identity"] = report_tool._arm_scientific_identity(
        reports[1]
    )
    reports[1]["report_id"] = report_tool.canonical_blake3(
        reports[1]["scientific_identity"]
    )
    result, exit_code = report_tool.classify_reports(reports, _replay())
    assert exit_code == 4
    assert result["classification"] == report_tool.CLASSIFICATION_INVALID


def test_exact_arm_absolute_performance_failure_is_not_relational_null() -> None:
    reports = _reports()
    reports[1]["performance"]["action_scores_per_second"] = 10_000.0
    reports[1]["scientific_identity"] = report_tool._arm_scientific_identity(
        reports[1]
    )
    reports[1]["report_id"] = report_tool.canonical_blake3(
        reports[1]["scientific_identity"]
    )
    result, exit_code = report_tool.classify_reports(reports, _replay())
    assert exit_code == 3
    assert result["classification"] == report_tool.CLASSIFICATION_EXACT_FAILED


def test_command_exit_code_distinguishes_rejection_from_malformed_evidence() -> None:
    assert report_tool.command_exit_code(0) == 0
    assert report_tool.command_exit_code(2) == 0
    assert report_tool.command_exit_code(3) == 0
    assert report_tool.command_exit_code(4) == 4
