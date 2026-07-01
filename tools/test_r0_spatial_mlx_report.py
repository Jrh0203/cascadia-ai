from __future__ import annotations

import json
import random
from pathlib import Path

import r0_spatial_mlx_report as report_tool
from cascadia_mlx.r0_spatial_mlx_tournament import R0SpatialMlxTournamentProtocol


def _report(
    arm: str,
    *,
    total_mae: float,
    total_rmse: float,
    component_mae: float,
    inference_ratio: float,
    training_ratio: float,
) -> dict:
    capacity = report_tool.ARM_TOKEN_CAPACITY[arm]
    scientific_identity = {"arm": arm, "result": "synthetic-complete"}
    return {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "adr": report_tool.ADR_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "arm": arm,
        "report_id": report_tool.canonical_blake3(scientific_identity),
        "scientific_identity": scientific_identity,
        "cache": {
            "cache_id": report_tool.canonical_blake3({"cache": arm}),
            "corpus_lock_id": "1" * 64,
            "source_semantic_blake3": "2" * 64,
            "d6_semantic_blake3": "3" * 64,
            "target_blake3": "4" * 64,
            "spatial_token_capacity": capacity,
            "local_capacity": max(capacity - 23, 0),
        },
        "authorization": {
            "authorization_id": "5" * 64,
            "approved_by": "parent",
            "approved_unix_ms": 1,
        },
        "protocol": R0SpatialMlxTournamentProtocol().to_dict(),
        "model": {
            "config": R0SpatialMlxTournamentProtocol().model.to_dict(),
            "parameter_count": 74_635,
        },
        "optimization": {
            "global_step": 500,
            "training_examples": 16_000,
            "training_seconds": 10.0,
            "training_examples_per_second": 1_600.0,
            "invocation_training_examples": 16_000,
            "invocation_training_seconds": 10.0,
            "invocation_wall_seconds": 11.0,
            "last_checkpoint": "/host-specific/checkpoint",
            "last_checkpoint_manifest_blake3": "6" * 64,
        },
        "metrics": {
            "train": {
                "samples": 50_000,
                "loss": 0.1,
                "component_mae": [component_mae] * 11,
                "component_rmse": [component_mae + 0.5] * 11,
                "component_bias": [0.0] * 11,
                "mean_component_mae": component_mae,
                "total_mae": total_mae,
                "total_rmse": total_rmse,
                "predicted_total_mean": 80.0,
                "target_total_mean": 80.0,
                "total_bias": 0.0,
                "total_correlation": 0.8,
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            },
            "validation": {
                "samples": 10_000,
                "loss": 0.2,
                "component_mae": [component_mae] * 11,
                "component_rmse": [component_mae + 0.5] * 11,
                "component_bias": [0.0] * 11,
                "mean_component_mae": component_mae,
                "total_mae": total_mae,
                "total_rmse": total_rmse,
                "predicted_total_mean": 80.0,
                "target_total_mean": 80.0,
                "total_bias": 0.0,
                "total_correlation": 0.8,
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            },
        },
        "performance": {
            "definition": "synthetic",
            "spatial_token_capacity": capacity,
            "compile_seconds": 0.1,
            "compile_batch_examples": 64,
            "warmup_iterations": 5,
            "warmup_seconds": 0.1,
            "warmup_examples_per_second": 3_200.0,
            "steady_iterations": 30,
            "steady_seconds": 0.5,
            "steady_examples_per_second": 3_840.0,
            "inference_actions_per_second": 3_840.0,
            "latency_milliseconds": {"p50": 1.0, "p90": 1.2, "p99": 1.5},
            "inference_peak_active_memory_bytes": 1_000,
            "inference_active_memory_bytes": 900,
            "inference_cache_memory_bytes": 100,
            "same_host_exact_shape_control": {
                "spatial_token_capacity": 23,
                "steady_examples_per_second": 2_400.0,
            },
            "same_host_training_step": {
                "spatial_token_capacity": capacity,
                "examples_per_second": 1_300.0,
                "peak_active_memory_bytes": 1_200,
            },
            "same_host_exact_shape_training_step": {
                "spatial_token_capacity": 23,
                "examples_per_second": 1_000.0,
                "peak_active_memory_bytes": 800,
            },
            "same_host_shape_ratios": {
                "inference_examples_per_second": inference_ratio,
                "training_examples_per_second": training_ratio,
                "inference_peak_memory_fraction": 1.25,
                "training_peak_memory_fraction": 1.5,
            },
            "training_peak_active_memory_bytes": 1_500,
            "peak_process_rss_bytes": 2_000,
        },
        "runtime": {
            "mlx_version": "0.29.3",
            "python_version": "3.12.10",
            "machine": "arm64",
            "platform": "host-specific",
            "device": "Device(gpu, 0)",
            "mlx_cache_limit_bytes": 1_073_741_824,
            "previous_mlx_cache_limit_bytes": 0,
        },
        "source": {
            "git_revision": "unavailable",
            "git_dirty": True,
            "v2_source_blake3": "7" * 64,
        },
        "integrity": {
            "cache_verified": True,
            "padding_verified": True,
            "semantic_round_trip_verified": True,
            "overflow_exact_entities_retained": True,
            "all_metrics_finite": True,
            "test_or_final_data_opened": False,
        },
        "claims": {
            "learned_representation_screen_complete": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }


def _reports() -> list[dict]:
    return [
        _report(
            "exact-entity-control",
            total_mae=5.0,
            total_rmse=6.0,
            component_mae=1.0,
            inference_ratio=1.0,
            training_ratio=1.0,
        ),
        _report(
            "hex-radius-6-127",
            total_mae=5.5,
            total_rmse=6.5,
            component_mae=1.1,
            inference_ratio=1.6,
            training_ratio=1.2,
        ),
        _report(
            "hex-radius-5-91",
            total_mae=5.4,
            total_rmse=6.4,
            component_mae=1.1,
            inference_ratio=1.4,
            training_ratio=1.2,
        ),
        _report(
            "hex-radius-4-61",
            total_mae=5.8,
            total_rmse=7.0,
            component_mae=1.2,
            inference_ratio=1.7,
            training_ratio=1.4,
        ),
        _report(
            "historical-square-21x21-441",
            total_mae=5.1,
            total_rmse=6.1,
            component_mae=1.0,
            inference_ratio=0.2,
            training_ratio=0.3,
        ),
    ]


def test_complete_tournament_selects_smallest_passing_compact_arm() -> None:
    output, exit_code = report_tool.classify_reports(_reports())
    assert exit_code == 0
    assert output["classification"] == report_tool.CLASSIFICATION_COMPLETE
    assert output["selected_stage2_candidate"] == "hex-radius-4-61"
    assert output["claims"]["promotion_authorized"] is False
    assert output["historical441_diagnostic_only"] is True


def test_classifier_is_byte_deterministic_under_input_order() -> None:
    reports = _reports()
    forward, _ = report_tool.classify_reports(reports)
    random.Random(17).shuffle(reports)
    shuffled, _ = report_tool.classify_reports(reports)
    assert json.dumps(forward, sort_keys=True) == json.dumps(shuffled, sort_keys=True)


def test_semantic_failure_has_precedence() -> None:
    reports = _reports()
    reports[2]["integrity"]["padding_verified"] = False
    output, exit_code = report_tool.classify_reports(reports)
    assert exit_code == 2
    assert output["classification"] == report_tool.CLASSIFICATION_SEMANTIC_FAILURE


def test_missing_arm_is_structurally_incomplete() -> None:
    output, exit_code = report_tool.classify_reports(_reports()[:-1])
    assert exit_code == 3
    assert output["classification"] == report_tool.CLASSIFICATION_INCOMPLETE
    assert output["selected_stage2_candidate"] == "hex-radius-4-61"


def test_runtime_or_model_drift_is_structurally_incomplete() -> None:
    reports = _reports()
    reports[1]["model"]["parameter_count"] += 1
    output, exit_code = report_tool.classify_reports(reports)
    assert exit_code == 3
    assert any("controlled training identity" in error for error in output["structural_errors"])


def test_zero_timing_is_insufficient_performance_evidence() -> None:
    reports = _reports()
    reports[3]["performance"]["compile_seconds"] = 0.0
    output, exit_code = report_tool.classify_reports(reports)
    assert exit_code == 4
    assert output["classification"] == report_tool.CLASSIFICATION_INSUFFICIENT_PERFORMANCE


def test_cross_arm_semantic_digest_drift_is_a_semantic_failure() -> None:
    reports = _reports()
    reports[4]["cache"]["target_blake3"] = "9" * 64
    output, exit_code = report_tool.classify_reports(reports)
    assert exit_code == 2
    assert any("semantic or target identity" in error for error in output["semantic_errors"])


def test_collection_loader_accepts_absolute_collected_paths(tmp_path: Path) -> None:
    entries = []
    for index, report in enumerate(_reports()):
        path = tmp_path / f"arm-{index}.json"
        path.write_text(json.dumps(report))
        entries.append({"file": str(path)})
    collection = tmp_path / "collection.json"
    collection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": report_tool.EXPERIMENT_ID,
                "adr": report_tool.ADR_ID,
                "reports": entries,
            }
        )
    )
    forward = report_tool.load_collection(collection)
    reverse = report_tool.load_collection(collection, reverse=True)
    assert [report["arm"] for report in reverse] == [report["arm"] for report in reversed(forward)]
