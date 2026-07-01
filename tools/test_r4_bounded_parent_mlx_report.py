from __future__ import annotations

import copy
from pathlib import Path

import blake3
import r4_bounded_parent_mlx_report as report_tool


def _trace() -> list[dict]:
    return [
        {
            "step": step,
            "batch_blake3": blake3.blake3(step.to_bytes(8, "little")).hexdigest(),
            "loss": 1.0 / step,
            "candidates": 1500 + step % 17,
            "elapsed_seconds": 0.01,
        }
        for step in range(1, 3001)
    ]


def _performance(
    *,
    parent_p50: float,
    throughput: float,
    p99: float,
    active: int = 100_000_000,
    rss: int = 200_000_000,
) -> dict:
    return {
        "parent_encode": {"latency_milliseconds": {"p50": parent_p50}},
        "fixed_chunk": {"action_scores_per_second": throughput},
        "complete_decisions": {
            "groups": 240,
            "actions": 860_203,
            "parent_encodes": 240,
            "parent_encode_count_exact": True,
            "latency_milliseconds": {"p99": p99},
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
        "parent_cache_id": "b" * 64,
        "s1_cache_id": "c" * 64,
        "protocol": {"seed": 2026061710, "training_steps": 3000},
        "model": {
            "parameter_count": 600_000,
            "parameter_layout_blake3": "d" * 64,
            "initial_parameter_tensor_blake3": "e" * 64,
        },
        "optimization": {"global_step": 3000, "loss_trace": _trace()},
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
                "low_supply": {"top64_r4800_winner_recall": low_supply},
                "independent_draft_winner": {"top64_r4800_winner_recall": independent},
            },
        },
        "performance": performance or _performance(parent_p50=1.0, throughput=30_000.0, p99=100.0),
    }
    report["scientific_identity"] = {
        key: value for key, value in report.items() if key != "schema_version"
    }
    report["report_id"] = report_tool._canonical_blake3(report["scientific_identity"])
    return report


def _paired_control(treatment_arm: str, performance: dict) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "treatment_arm": treatment_arm,
        "host": report_tool.ARM_HOSTS[treatment_arm],
        "control_arm": report_tool.CONTROL_ARM,
        "control_report_id": "",
        "performance": performance,
    }


def _evidence() -> tuple[list[dict], list[dict]]:
    reports = [
        _report(report_tool.CONTROL_ARM),
        _report(
            "q1-seat-marginal-parent",
            performance=_performance(
                parent_p50=0.70,
                throughput=33_500.0,
                p99=94.0,
            ),
        ),
        _report(
            "q2-directional-parent",
            performance=_performance(
                parent_p50=0.90,
                throughput=34_000.0,
                p99=93.0,
            ),
        ),
        _report(
            "q3-affordance-parent",
            performance=_performance(
                parent_p50=0.70,
                throughput=30_500.0,
                p99=99.0,
            ),
        ),
    ]
    controls = []
    for arm in report_tool.ARMS[1:]:
        replay = _paired_control(
            arm,
            _performance(parent_p50=1.0, throughput=30_000.0, p99=100.0),
        )
        replay["control_report_id"] = reports[0]["report_id"]
        replay["scientific_identity"] = {
            key: value for key, value in replay.items() if key != "schema_version"
        }
        replay["replay_id"] = report_tool._canonical_blake3(replay["scientific_identity"])
        controls.append(replay)
    return reports, controls


def test_q1_is_selected_when_quality_and_material_efficiency_pass() -> None:
    reports, controls = _evidence()
    aggregate = report_tool.classify_reports(reports, controls)
    assert aggregate["classification"] == report_tool.CLASSIFICATION_SELECTED
    assert aggregate["selected_arm"] == "q1-seat-marginal-parent"


def test_quality_only_null_when_parent_latency_does_not_improve() -> None:
    reports, controls = _evidence()
    for report in reports[1:]:
        report["performance"]["parent_encode"]["latency_milliseconds"]["p50"] = 0.9
        report["scientific_identity"]["performance"] = report["performance"]
        report["report_id"] = report_tool._canonical_blake3(report["scientific_identity"])
    aggregate = report_tool.classify_reports(reports, controls)
    assert aggregate["classification"] == report_tool.CLASSIFICATION_QUALITY_ONLY_NULL
    assert aggregate["selected_arm"] is None


def test_forward_reverse_order_proof_is_byte_identical() -> None:
    reports, controls = _evidence()
    forward, reverse, proof = report_tool.aggregate_with_order_proof(
        reports,
        controls,
    )
    assert forward == reverse
    assert proof["scientific_identity"]["byte_identical"] is True


def test_quality_regression_classifies_all_treatments_degraded() -> None:
    reports, controls = _evidence()
    for report in reports[1:]:
        report["metrics"]["r4800_value"]["mae"] = 1.50
        report["scientific_identity"]["metrics"] = copy.deepcopy(report["metrics"])
        report["report_id"] = report_tool._canonical_blake3(report["scientific_identity"])
    aggregate = report_tool.classify_reports(reports, controls)
    assert aggregate["classification"] == report_tool.CLASSIFICATION_ALL_DEGRADED


def test_structural_failure_emits_content_addressed_invalid_artifacts(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    forward, reverse, proof = report_tool.invalid_outputs(
        report_tool.R4ParentReportError("malformed evidence"),
        [malformed],
        [],
    )
    assert forward == reverse
    assert forward["classification"] == report_tool.CLASSIFICATION_INVALID
    assert proof["scientific_identity"]["byte_identical"] is True
    assert proof["scientific_identity"]["invalid_evidence"] is True
