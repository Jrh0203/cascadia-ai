from __future__ import annotations

from pathlib import Path

import blake3
import pytest
import relational_substrate_mlx_report as report_tool


def _trace() -> list[dict]:
    return [
        {
            "step": step,
            "batch_blake3": blake3.blake3(
                step.to_bytes(8, "little")
            ).hexdigest(),
            "loss": 1.0 / step,
            "candidates": 1500 + step % 17,
            "elapsed_seconds": 0.01,
        }
        for step in range(1, 3001)
    ]


def _performance(
    *,
    throughput: float,
    p99: float,
    fixed_throughput: float = 30_000.0,
    active: int = 100_000_000,
    rss: int = 200_000_000,
) -> dict:
    return {
        "fixed_chunk": {
            "action_scores_per_second": fixed_throughput
        },
        "combined_with_r6": {
            "groups": 240,
            "actions": 860_203,
            "action_scores_per_second": throughput,
            "latency_milliseconds": {"p99": p99},
            "r6_exact_parity_pass": True,
        },
        "r6_apply_undo": {
            "exact_parity_pass": True,
            "apply_failures": 0,
            "undo_failures": 0,
        },
        "memory": {
            "peak_active_bytes": active,
            "peak_process_rss_bytes": rss,
            "process_swaps": 0,
            "system_swap_delta_bytes": None,
        },
    }


def _report(
    arm: str,
    *,
    mae: float = 1.30,
    rmse: float = 1.70,
    recall: float = 0.75,
    regret: float = 0.10,
    low_supply: float = 0.90,
    independent: float = 0.80,
    coverage: float = 0.995,
    strategic: tuple[float, float, float] = (0.72, 0.72, 0.72),
    performance: dict | None = None,
) -> dict:
    report = {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "mode": "production",
        "arm": arm,
        "host": report_tool.ARM_HOSTS[arm],
        "r3_cache_id": "a" * 64,
        "relational_cache_id": "b" * 64,
        "s1_cache_id": "c" * 64,
        "r6_binary": {
            "path": f"/tmp/{arm}/r6",
            "blake3": "d" * 64,
        },
        "protocol": {"seed": 2026061716, "training_steps": 3000},
        "model": {
            "parameter_count": 631_170,
            "parameter_layout_blake3": "e" * 64,
            "initial_parameter_tensor_blake3": "f" * 64,
        },
        "checkpoint": {
            "path": f"/tmp/{arm}/checkpoint",
            "manifest_blake3": "4" * 64,
            "model_blake3": "5" * 64,
        },
        "optimization": {
            "global_step": 3000,
            "loss_trace": _trace(),
        },
        "metrics": {
            "groups": 240,
            "candidates": 860_203,
            "all_groups_scored_once": True,
            "all_candidates_scored_once": True,
            "all_scores_and_uncertainties_finite": True,
            "parent_encodes": 240,
            "r4800_value": {"mae": mae, "rmse": rmse},
            "top64_r4800_winner_recall": recall,
            "mean_top64_retained_r4800_regret": regret,
            "top64_confidence_set_coverage_95": coverage,
            "subsets": {
                "low_supply": {
                    "top64_r4800_winner_recall": low_supply
                },
                "independent_draft_winner": {
                    "top64_r4800_winner_recall": independent
                },
            },
            "strategic_opportunity_recall": {
                "elk": strategic[0],
                "salmon": strategic[1],
                "hawk": strategic[2],
                "bear_diagnostic": 0.70,
                "primary_mean": sum(strategic) / 3,
            },
        },
        "performance": performance
        or _performance(throughput=1000.0, p99=100.0),
        "source": {"v2_source_blake3": "1" * 64},
        "controls": {
            "authorization_id": "2" * 64,
            "preflight_id": f"{arm}-preflight",
            "open_data_verification_id": "3" * 64,
        },
        "information_boundary": {
            "sealed_test_opened": False,
            "gameplay_run": False,
        },
        "claims": {
            "offline_comparison_complete": True,
            "promotion_authorized": False,
        },
    }
    report["scientific_identity"] = {
        key: value
        for key, value in report.items()
        if key != "schema_version"
    }
    report["report_id"] = report_tool._canonical_blake3(
        report["scientific_identity"]
    )
    return report


def _paired_control(
    treatment_report: dict,
    control_report: dict,
    performance: dict,
) -> dict:
    treatment_arm = treatment_report["arm"]
    identity = {
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "replay_kind": "same-host-exact-c0-serving-control-with-r6",
        "treatment_arm": treatment_arm,
        "host": report_tool.ARM_HOSTS[treatment_arm],
        "control_arm": report_tool.CONTROL_ARM,
        "control_report_id": control_report["report_id"],
        "authorization_id": control_report["controls"][
            "authorization_id"
        ],
        "r3_cache_id": control_report["r3_cache_id"],
        "relational_cache_id": control_report["relational_cache_id"],
        "s1_cache_id": control_report["s1_cache_id"],
        "checkpoint": {
            "manifest_blake3": control_report["checkpoint"][
                "manifest_blake3"
            ],
            "model_blake3": control_report["checkpoint"][
                "model_blake3"
            ],
            "global_step": 3000,
        },
        "r6_binary_blake3": treatment_report["r6_binary"]["blake3"],
        "open_data_verification_id": control_report["controls"][
            "open_data_verification_id"
        ],
        "benchmark_request_id": "6" * 64,
        "benchmark_result_id": "7" * 64,
        "assertions": {
            "control_checkpoint_manifest_identical": True,
            "control_checkpoint_model_identical": True,
            "r6_binary_identical": True,
            "replay_host_is_treatment_host": True,
            "replay_host_differs_from_control_host": True,
            "isolated_process": True,
            "open_data_reverified": True,
            "all_validation_decisions_measured": True,
            "all_validation_actions_measured": True,
            "r6_apply_undo_exact": True,
        },
    }
    replay = {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "treatment_arm": treatment_arm,
        "host": report_tool.ARM_HOSTS[treatment_arm],
        "control_arm": report_tool.CONTROL_ARM,
        "control_report_id": control_report["report_id"],
        "scientific_identity": identity,
        "performance": performance,
    }
    replay["replay_id"] = report_tool._canonical_blake3(identity)
    return replay


def _evidence() -> tuple[list[dict], list[dict]]:
    reports = [
        _report(
            report_tool.CONTROL_ARM,
            strategic=(0.70, 0.70, 0.70),
        ),
        _report(
            "q1-r5-quotient-local",
            strategic=(0.72, 0.72, 0.72),
            performance=_performance(throughput=1200.0, p99=82.0),
        ),
        _report(
            "g2-r5-s3",
            strategic=(0.72, 0.72, 0.72),
            performance=_performance(throughput=1150.0, p99=88.0),
        ),
        _report(
            "d3-r5-s3-s5",
            strategic=(0.72, 0.72, 0.72),
            performance=_performance(throughput=1120.0, p99=89.0),
        ),
    ]
    controls = []
    by_arm = {report["arm"]: report for report in reports}
    for arm in report_tool.ARMS[1:]:
        replay = _paired_control(
            by_arm[arm],
            reports[0],
            _performance(throughput=1000.0, p99=100.0),
        )
        controls.append(replay)
    return reports, controls


def _rehash(report: dict) -> None:
    report["scientific_identity"] = {
        key: value
        for key, value in report.items()
        if key not in ("schema_version", "scientific_identity", "report_id")
    }
    report["report_id"] = report_tool._canonical_blake3(
        report["scientific_identity"]
    )


def test_q1_is_selected_by_frozen_tie_break() -> None:
    reports, controls = _evidence()
    aggregate = report_tool.classify_reports(reports, controls)
    assert aggregate["classification"] == report_tool.CLASSIFICATION_SELECTED
    assert aggregate["selected_arm"] == "q1-r5-quotient-local"


def test_quality_only_null_when_material_efficiency_does_not_improve() -> None:
    reports, controls = _evidence()
    for report in reports[1:]:
        report["performance"] = _performance(
            throughput=1050.0,
            p99=95.0,
        )
        _rehash(report)
    aggregate = report_tool.classify_reports(reports, controls)
    assert (
        aggregate["classification"]
        == report_tool.CLASSIFICATION_QUALITY_ONLY_NULL
    )
    assert aggregate["selected_arm"] is None


def test_forward_reverse_order_proof_is_byte_identical() -> None:
    reports, controls = _evidence()
    forward, reverse, proof = report_tool.aggregate_with_order_proof(
        reports,
        controls,
    )
    assert forward == reverse
    assert proof["scientific_identity"]["byte_identical"] is True


def test_strategic_regression_classifies_all_treatments_degraded() -> None:
    reports, controls = _evidence()
    for report in reports[1:]:
        report["metrics"]["strategic_opportunity_recall"].update(
            {
                "elk": 0.70,
                "salmon": 0.70,
                "hawk": 0.70,
                "primary_mean": 0.70,
            }
        )
        _rehash(report)
    aggregate = report_tool.classify_reports(reports, controls)
    assert (
        aggregate["classification"]
        == report_tool.CLASSIFICATION_ALL_DEGRADED
    )


def test_control_failure_precedes_treatment_selection() -> None:
    reports, controls = _evidence()
    reports[0]["metrics"]["r4800_value"]["mae"] = 1.50
    _rehash(reports[0])
    for replay in controls:
        replay["control_report_id"] = reports[0]["report_id"]
        replay["scientific_identity"]["control_report_id"] = reports[0][
            "report_id"
        ]
        replay["replay_id"] = report_tool._canonical_blake3(
            replay["scientific_identity"]
        )
    aggregate = report_tool.classify_reports(reports, controls)
    assert (
        aggregate["classification"]
        == report_tool.CLASSIFICATION_CONTROL_FAILED
    )


def test_structural_failure_emits_content_addressed_invalid_artifacts(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    forward, reverse, proof = report_tool.invalid_outputs(
        report_tool.RelationalSubstrateReportError(
            "malformed evidence"
        ),
        [malformed],
        [],
    )
    assert forward == reverse
    assert forward["classification"] == report_tool.CLASSIFICATION_INVALID
    assert proof["scientific_identity"]["byte_identical"] is True
    assert proof["scientific_identity"]["invalid_evidence"] is True


def test_batch_identity_drift_is_invalid() -> None:
    reports, controls = _evidence()
    reports[3]["optimization"]["loss_trace"][7][
        "batch_blake3"
    ] = "0" * 64
    _rehash(reports[3])
    with pytest.raises(
        report_tool.RelationalSubstrateReportError,
        match="identical scientific batches",
    ):
        report_tool.classify_reports(reports, controls)


def test_global_r6_difference_is_valid_with_exact_host_paired_controls() -> None:
    reports, controls = _evidence()
    for index, report in enumerate(reports[1:], start=1):
        report["r6_binary"]["blake3"] = f"{index:064x}"
        _rehash(report)
        controls[index - 1]["scientific_identity"][
            "r6_binary_blake3"
        ] = report["r6_binary"]["blake3"]
        controls[index - 1]["replay_id"] = report_tool._canonical_blake3(
            controls[index - 1]["scientific_identity"]
        )

    aggregate = report_tool.classify_reports(reports, controls)

    assert aggregate["classification"] == report_tool.CLASSIFICATION_SELECTED
    assert (
        aggregate["scientific_identity"]["common_identity"][
            "serving_binary_contract"
        ]
        == "host-paired-c0-replay-v1"
    )


def test_paired_control_r6_mismatch_is_invalid() -> None:
    reports, controls = _evidence()
    controls[0]["scientific_identity"]["r6_binary_blake3"] = "0" * 64
    controls[0]["replay_id"] = report_tool._canonical_blake3(
        controls[0]["scientific_identity"]
    )

    with pytest.raises(
        report_tool.RelationalSubstrateReportError,
        match="malformed or duplicated",
    ):
        report_tool.classify_reports(reports, controls)


def test_paired_control_checkpoint_mismatch_is_invalid() -> None:
    reports, controls = _evidence()
    controls[0]["scientific_identity"]["checkpoint"][
        "model_blake3"
    ] = "0" * 64
    controls[0]["replay_id"] = report_tool._canonical_blake3(
        controls[0]["scientific_identity"]
    )

    with pytest.raises(
        report_tool.RelationalSubstrateReportError,
        match="malformed or duplicated",
    ):
        report_tool.classify_reports(reports, controls)


def test_slow_control_is_a_valid_baseline_not_a_treatment_gate() -> None:
    reports, controls = _evidence()
    reports[0]["performance"] = _performance(
        throughput=100.0,
        p99=5_000.0,
        fixed_throughput=100.0,
    )
    _rehash(reports[0])
    for control in controls:
        control["control_report_id"] = reports[0]["report_id"]
        control["scientific_identity"]["control_report_id"] = reports[0][
            "report_id"
        ]
        control["performance"] = _performance(
            throughput=100.0,
            p99=5_000.0,
            fixed_throughput=100.0,
        )
        control["replay_id"] = report_tool._canonical_blake3(
            control["scientific_identity"]
        )

    aggregate = report_tool.classify_reports(reports, controls)

    assert aggregate["classification"] == report_tool.CLASSIFICATION_SELECTED
    control = aggregate["scientific_identity"]["control"]
    assert control["absolute_serving"]["passed"] is False
    assert control["serving_integrity"]["passed"] is True


def test_negative_system_swap_delta_counts_as_no_swap_growth() -> None:
    performance = _performance(throughput=1_000.0, p99=100.0)
    performance["memory"]["system_swap_delta_bytes"] = -(8 * 1024**2)

    assert report_tool._baseline_performance_integrity(
        performance
    )["checks"]["process_swap"] is True
    assert report_tool._absolute_performance(
        performance
    )["checks"]["process_swap"] is True
