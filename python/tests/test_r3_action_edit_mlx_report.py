from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import blake3
import pytest
import r3_action_edit_mlx_report as report_tool


def _prediction_panel() -> dict:
    hashes = [blake3.blake3(index.to_bytes(4, "little")).hexdigest() for index in range(64)]
    scores = [float(index) / 10.0 for index in range(64)]
    standard_errors = [0.25 + float(index) / 1000.0 for index in range(64)]
    digest = blake3.blake3()
    digest.update(b"".join(bytes.fromhex(value) for value in hashes))
    for value in scores:
        digest.update(struct.pack("<f", value))
    for value in standard_errors:
        digest.update(struct.pack("<f", value))
    return {
        "count": 64,
        "action_hashes": hashes,
        "scores": scores,
        "standard_errors": standard_errors,
        "panel_blake3": digest.hexdigest(),
    }


def _loss_trace() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "step": step,
            "batch_blake3": blake3.blake3(step.to_bytes(8, "little")).hexdigest(),
            "loss": 1.0 / step,
            "candidates": 1_500 + step % 400,
            "elapsed_seconds": 0.01,
        }
        for step in range(1, 3001)
    ]


def _slice(*, recall: float, regret: float, groups: int = 80) -> dict:
    return {
        "groups": groups,
        "top64_r4800_winner_recall": recall,
        "top64_confidence_set_coverage_95": 1.0,
        "mean_top64_retained_r4800_regret": regret,
    }


def _report(
    arm: str,
    *,
    recall: float = 0.80,
    regret: float = 0.20,
    low_supply_recall: float = 0.75,
    independent_recall: float = 0.76,
    throughput: float = 42_000.0,
    p99_latency: float = 75.0,
    active_memory: int = 75_000_000,
    rss: int = 150_000_000,
) -> dict:
    host = report_tool.ARM_HOSTS[arm]
    trace = _loss_trace()
    report = {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "adr": report_tool.ADR_ID,
        "mode": "production",
        "arm": arm,
        "host": host,
        "cache_id": "c" * 64,
        "s1_cache_id": "d" * 64,
        "protocol": {"seed": 2026061708, "training_steps": 3000},
        "model": {
            "config": {"arm": arm, "hidden_size": 64},
            "parameter_count": 552_770,
            "parameter_layout_blake3": "1" * 64,
            "initial_parameter_tensor_blake3": "2" * 64,
            "final_parameter_tensor_blake3": blake3.blake3(arm.encode()).hexdigest(),
        },
        "optimization": {
            "global_step": 3000,
            "candidates": sum(int(event["candidates"]) for event in trace),
            "training_seconds": 30.0,
            "training_wall_seconds": 35.0,
            "candidates_per_second": 150_000.0,
            "training_peak_active_memory_bytes": active_memory,
            "loss_trace": trace,
        },
        "checkpoint": {
            "path": f"/immutable/{arm}",
            "manifest_blake3": "3" * 64,
            "model_blake3": "4" * 64,
        },
        "metrics": {
            "groups": 240,
            "candidates": 860_203,
            "expected_groups": 240,
            "expected_candidates": 860_203,
            "all_groups_scored_once": True,
            "all_candidates_scored_once": True,
            "parent_encodes": 240,
            "parent_encode_count_exact": True,
            "nonfinite_scores": 0,
            "nonfinite_uncertainties": 0,
            "all_scores_and_uncertainties_finite": True,
            "r4800_value": {
                "count": 860_203,
                "mae": 1.0,
                "rmse": 1.2,
                "bias": 0.0,
                "correlation": 0.8,
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            },
            "top1_r4800_winner_recall": 0.20,
            "top8_r4800_winner_recall": 0.45,
            "top32_r4800_winner_recall": 0.70,
            "top64_r4800_winner_recall": recall,
            "mean_top1_retained_r4800_regret": 1.0,
            "mean_top8_retained_r4800_regret": 0.5,
            "mean_top32_retained_r4800_regret": 0.3,
            "mean_top64_retained_r4800_regret": regret,
            "top64_confidence_set_coverage_95": 0.999,
            "subsets": {
                "early": _slice(recall=recall, regret=regret),
                "middle": _slice(recall=recall, regret=regret),
                "late": _slice(recall=recall, regret=regret),
                "low_supply": _slice(
                    recall=low_supply_recall,
                    regret=regret,
                    groups=40,
                ),
                "independent_draft_winner": _slice(
                    recall=independent_recall,
                    regret=regret,
                    groups=40,
                ),
            },
            "candidate_tokens": {
                "count": 860_203,
                "minimum": 20,
                "mean": 50.0,
                "p50": 50.0,
                "p90": 80.0,
                "p99": 95.0,
                "maximum": 100,
                "padding_tokens": 10_000,
            },
            "prediction_panel": _prediction_panel(),
        },
        "performance": {
            "fixed_chunk": {
                "actions": 256,
                "compile_seconds": 0.2,
                "warmup_iterations": 5,
                "warmup_seconds": 0.1,
                "steady_iterations": 30,
                "steady_seconds": 0.2,
                "action_scores_per_second": throughput,
                "latency_milliseconds": {"p50": 5.0, "p95": 7.0, "p99": 8.0},
            },
            "complete_decisions": {
                "groups": 20,
                "actions": 70_000,
                "parent_encodes": 20,
                "parent_encode_count_exact": True,
                "elapsed_seconds": 2.0,
                "action_scores_per_second": throughput,
                "latency_milliseconds": {
                    "p50": p99_latency * 0.5,
                    "p95": p99_latency * 0.9,
                    "p99": p99_latency,
                },
            },
            "memory": {
                "active_bytes": active_memory // 2,
                "cache_bytes": 1_000_000,
                "peak_active_bytes": active_memory,
                "peak_process_rss_bytes": rss,
                "process_swaps": 0,
                "system_swap_before_bytes": 0,
                "system_swap_after_bytes": 0,
                "system_swap_delta_bytes": 0,
            },
            "measurement": {
                "isolated_process": True,
                "request_id": "8" * 64,
                "result_id": "9" * 64,
                "checkpoint_model_blake3": "4" * 64,
                "open_data_verification_id": "a" * 64,
                "verification_source": "cluster-preflight",
                "worker_runtime": {
                    "machine": "arm64",
                    "default_device": "Device(gpu, 0)",
                },
            },
        },
        "runtime": {
            "python": "3.12.10",
            "platform": "macOS",
            "machine": "arm64",
            "mlx": "0.31.2",
            "numpy": "2.2.0",
            "default_device": "Device(gpu, 0)",
            "host": host,
            "mlx_cache_limit_bytes": 1_073_741_824,
            "previous_mlx_cache_limit_bytes": 0,
        },
        "source": {
            "git_revision": "5" * 40,
            "git_dirty": False,
            "v2_source_blake3": "6" * 64,
        },
        "controls": {
            "authorization_id": "7" * 64,
            "preflight_id": blake3.blake3(f"preflight:{arm}".encode()).hexdigest(),
        },
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_read": False,
            "future_refill_read": False,
        },
        "claims": {
            "offline_comparison_complete": True,
            "bounded_smoke_complete": False,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    return _reseal(report)


def _reseal(report: dict) -> dict:
    report["scientific_identity"] = report_tool._arm_scientific_identity(report)
    report["report_id"] = report_tool._canonical_blake3(report["scientific_identity"])
    return report


def _reports() -> list[dict]:
    reports = [
        _report(
            report_tool.CONTROL,
            throughput=30_000.0,
            p99_latency=100.0,
            active_memory=100_000_000,
            rss=200_000_000,
        )
    ]
    reports.extend(_report(arm) for arm in report_tool.ARMS[1:])
    return reports


def _write_reports(tmp_path: Path, reports: list[dict]) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"{index}-{report['arm']}.json"
        path.write_text(json.dumps(report, sort_keys=True) + "\n")
        paths.append(path)
    return paths


def test_compact_selection_is_order_invariant_and_prefers_smallest_tied_radius(
    tmp_path: Path,
) -> None:
    reports = _reports()
    forward = report_tool.classify_reports(_write_reports(tmp_path / "forward", reports))
    reverse = report_tool.classify_reports(
        _write_reports(tmp_path / "reverse", list(reversed(reports)))
    )
    assert forward == reverse
    assert forward["classification"] == report_tool.SELECTED
    assert forward["selected_arm"] == "t3-r3-radius1-global"
    assert forward["claims"]["promotion_authorized"] is False
    assert forward["claims"]["gameplay_strength_measured"] is False


def test_quality_only_all_degraded_and_control_failed_are_distinct(tmp_path: Path) -> None:
    quality_only = _reports()
    for report in quality_only[1:]:
        report["performance"]["fixed_chunk"]["action_scores_per_second"] = 30_000.0
        report["performance"]["complete_decisions"]["latency_milliseconds"]["p99"] = 100.0
        report["performance"]["memory"]["peak_active_bytes"] = 100_000_000
        report["performance"]["memory"]["peak_process_rss_bytes"] = 200_000_000
        _reseal(report)
    result = report_tool.classify_reports(_write_reports(tmp_path / "quality", quality_only))
    assert result["classification"] == report_tool.QUALITY_ONLY_NULL

    degraded = _reports()
    for report in degraded[1:]:
        report["metrics"]["top64_r4800_winner_recall"] = 0.70
        report["metrics"]["subsets"]["low_supply"]["top64_r4800_winner_recall"] = 0.60
        report["metrics"]["subsets"]["independent_draft_winner"]["top64_r4800_winner_recall"] = 0.60
        _reseal(report)
    result = report_tool.classify_reports(_write_reports(tmp_path / "degraded", degraded))
    assert result["classification"] == report_tool.ALL_TREATMENTS_DEGRADED

    control_failed = _reports()
    control_failed[0]["performance"]["fixed_chunk"]["action_scores_per_second"] = 10_000.0
    _reseal(control_failed[0])
    result = report_tool.classify_reports(_write_reports(tmp_path / "control", control_failed))
    assert result["classification"] == report_tool.CONTROL_FAILED


def test_mutation_wrong_host_and_source_drift_fail_closed(tmp_path: Path) -> None:
    reports = _reports()
    reports[0]["host"] = "john4"
    with pytest.raises(report_tool.R3ReportError, match="malformed"):
        report_tool.classify_reports(_write_reports(tmp_path / "mutation", reports))

    reports = _reports()
    reports[1]["host"] = "john3"
    reports[1]["runtime"]["host"] = "john3"
    _reseal(reports[1])
    with pytest.raises(report_tool.R3ReportError, match="model parity"):
        report_tool.classify_reports(_write_reports(tmp_path / "host", reports))

    reports = _reports()
    reports[2]["source"]["v2_source_blake3"] = "8" * 64
    _reseal(reports[2])
    with pytest.raises(report_tool.R3ReportError, match="model parity"):
        report_tool.classify_reports(_write_reports(tmp_path / "source", reports))


def test_malformed_nested_evidence_becomes_report_error(tmp_path: Path) -> None:
    reports = _reports()
    del reports[3]["performance"]["memory"]
    _reseal(reports[3])
    with pytest.raises(report_tool.R3ReportError):
        report_tool.classify_reports(_write_reports(tmp_path, reports))


def test_order_proof_validates_classification_content_address(tmp_path: Path) -> None:
    classification = report_tool.classify_reports(_write_reports(tmp_path / "reports", _reports()))
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    encoded = json.dumps(classification, indent=2, sort_keys=True) + "\n"
    forward.write_text(encoded)
    reverse.write_text(encoded)
    proof = report_tool.compare_classifications(forward, reverse)
    assert proof["scientific_identity"]["forward_reverse_byte_identical"] is True
    assert proof["scientific_identity"]["scientific_blake3"] == classification["scientific_blake3"]

    tampered = copy.deepcopy(classification)
    tampered["selected_arm"] = report_tool.ARMS[1]
    tampered["scientific"]["selected_arm"] = report_tool.ARMS[1]
    tampered_encoded = json.dumps(tampered, indent=2, sort_keys=True) + "\n"
    forward.write_text(tampered_encoded)
    reverse.write_text(tampered_encoded)
    with pytest.raises(report_tool.R3ReportError, match="content address"):
        report_tool.compare_classifications(forward, reverse)
