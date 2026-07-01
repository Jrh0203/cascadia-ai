from __future__ import annotations

import json
import random

import blake3
import numpy as np
import r2_sparse_mlx_report as report_tool
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    TOKEN_CAPACITY,
    TOKEN_TYPE_NAMES,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    R2SparseMlxModelConfig,
    architecture_parameter_counts,
)
from cascadia_mlx.r2_sparse_mlx_tournament import (
    R2SparseMlxTournamentProtocol,
)


def _metrics(total_mae: float, total_rmse: float, component_mae: float) -> dict:
    return {
        "samples": 10_000,
        "loss": 0.02,
        "component_mae": [component_mae] * 11,
        "component_rmse": [component_mae + 0.5] * 11,
        "component_bias": [0.0] * 11,
        "mean_component_mae": component_mae,
        "total_mae": total_mae,
        "total_rmse": total_rmse,
        "predicted_total_mean": 91.0,
        "target_total_mean": 91.7,
        "total_bias": -0.7,
        "total_correlation": 0.2,
        "calibration_slope": 0.5,
        "calibration_intercept": 45.0,
    }


def _active_statistics() -> dict:
    train_type_totals = {
        "occupied": 2_575_000,
        "frontier": 3_463_262,
        "habitat_component": 1_881_333,
        "wildlife_motif": 1_971_617,
    }
    validation_type_totals = {
        "occupied": 515_000,
        "frontier": 692_652,
        "habitat_component": 376_267,
        "wildlife_motif": 394_323,
    }
    type_maxima = {
        "occupied": 91,
        "frontier": 107,
        "habitat_component": 69,
        "wildlife_motif": 79,
    }

    def split_statistics(records: int, totals: dict[str, int]) -> dict:
        active = sum(totals.values())
        return {
            "records": records,
            "padded_capacity_per_position": TOKEN_CAPACITY,
            "board_slots": BOARD_SLOTS,
            "padded_capacity_per_board": BOARD_TOKEN_CAPACITY,
            "active_tokens_total": active,
            "active_tokens_mean": active / records,
            "active_tokens_max": 340,
            "active_tokens_max_per_board": 92,
            "padding_tokens_total": records * TOKEN_CAPACITY - active,
            "type_tokens": {
                name: {
                    "total": total,
                    "mean_per_position": total / records,
                    "fraction_of_active": total / active,
                    "maximum_per_position": type_maxima[name],
                }
                for name, total in totals.items()
            },
            "foundation_per_board_p99_active_tokens": 83,
            "foundation_per_board_max_active_tokens": 92,
        }

    return {
        "train": split_statistics(50_000, train_type_totals),
        "validation": split_statistics(10_000, validation_type_totals),
    }


def _report(
    role: str,
    *,
    total_mae: float,
    total_rmse: float,
    component_mae: float,
    probe_offset: float = 0.0,
) -> dict:
    architecture = report_tool.RUN_ARCHITECTURES[role]
    protocol = R2SparseMlxTournamentProtocol().to_dict()
    validation = _metrics(total_mae, total_rmse, component_mae)
    train = dict(validation)
    train["samples"] = 50_000
    validation_type_ablations = {
        name: {
            "masked_token_type": token_type,
            "samples": 10_000,
            "mean_component_mae": component_mae + 0.05,
            "total_mae": total_mae + 0.1,
            "total_rmse": total_rmse + 0.1,
            "total_bias": -0.6,
            "total_correlation": 0.15,
            "delta_mean_component_mae": 0.05,
            "delta_total_mae": 0.1,
            "delta_total_rmse": 0.1,
        }
        for token_type, name in TOKEN_TYPE_NAMES.items()
    }
    probe = [[probe_offset] * 11 for _ in range(256)]
    value = {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "adr": report_tool.ADR_ID,
        "protocol_id": report_tool.PROTOCOL_ID,
        "run_role": role,
        "architecture": architecture,
        "cache": {
            "cache_id": "1" * 64,
            "manifest_blake3": "0" * 64,
            "corpus_lock_id": "2" * 64,
            "identity_semantic_blake3": "3" * 64,
            "d6_semantic_blake3": "4" * 64,
            "target_blake3": "5" * 64,
            "token_capacity": TOKEN_CAPACITY,
            "board_slots": BOARD_SLOTS,
            "board_token_capacity": BOARD_TOKEN_CAPACITY,
            "graph_max_degree": 24,
            "board_ownership_encoding": "relative-seat-one-hot-4",
            "active_token_statistics": _active_statistics(),
        },
        "authorization": {
            "authorization_id": "6" * 64,
            "r0_control_binding_id": "7" * 64,
            "approved_by": "parent",
            "approved_unix_ms": 1,
        },
        "protocol": protocol,
        "model": {
            "config": R2SparseMlxModelConfig(architecture=architecture).to_dict(),
            "parameter_count": architecture_parameter_counts()[architecture],
            "parameter_count_scope": protocol["parameter_count_scope"],
            "state_encoder_invocations_per_prediction": 1,
        },
        "optimization": {
            "global_step": 500,
            "training_examples": 16_000,
            "training_seconds": 10.0,
            "training_examples_per_second": 1_600.0,
            "last_checkpoint_manifest_blake3": "8" * 64,
        },
        "metrics": {
            "train": train,
            "validation": validation,
            "validation_type_ablations": validation_type_ablations,
            "validation_probe": {
                "rows": 256,
                "indices": list(range(256)),
                "predictions": probe,
                "prediction_blake3": "",
            },
        },
        "performance": {
            "compile_seconds": 0.2,
            "warmup_examples_per_second": 1_000.0,
            "steady_examples_per_second": 900.0,
            "inference_actions_per_second": 900.0,
            "latency_milliseconds": {"p50": 1.0, "p90": 1.2, "p99": 1.5},
            "inference_peak_active_memory_bytes": 1_000,
            "training_peak_active_memory_bytes": 2_000,
            "peak_process_rss_bytes": 3_000,
            "training_step": {"examples_per_second": 500.0},
        },
        "runtime": {
            "mlx_version": "0.31.2",
            "python_version": "3.12.10",
            "machine": "arm64",
            "platform": "host-specific",
            "device": "Device(gpu, 0)",
        },
        "source": {"v2_source_blake3": "a" * 64},
        "integrity": {
            "cache_verified": True,
            "exact_no_truncation_verified": True,
            "padding_zero_verified": True,
            "board_local_layout_verified": True,
            "graph_degree_bound_verified": True,
            "d6_regeneration_verified_by_exporter": True,
            "derived_relations_loaded_from_content_addressed_cache": True,
            "state_trunk_encoded_once": True,
            "type_balanced_pooling_verified": True,
            "typewise_ablation_reported": True,
            "all_metrics_finite": True,
            "test_or_final_data_opened": False,
        },
        "claims": {
            "matched_architecture_screen_complete": True,
            "gameplay_strength_measured": False,
            "production_model_selected": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    return _reseal(value)


def _reseal(value: dict) -> dict:
    probe = np.asarray(
        value["metrics"]["validation_probe"]["predictions"],
        dtype="<f4",
    )
    value["metrics"]["validation_probe"]["prediction_blake3"] = (
        blake3.blake3(probe.tobytes(order="C")).hexdigest()
    )
    scientific_identity = report_tool.report_scientific_identity(value)
    value["scientific_identity"] = scientific_identity
    value["report_id"] = report_tool.canonical_blake3(scientific_identity)
    return value


def _reports() -> list[dict]:
    return [
        _report(
            "set-primary",
            total_mae=2.5,
            total_rmse=3.3,
            component_mae=2.65,
        ),
        _report(
            "graph-primary",
            total_mae=2.4,
            total_rmse=3.2,
            component_mae=2.60,
        ),
        _report(
            "perceiver-primary",
            total_mae=2.7,
            total_rmse=3.4,
            component_mae=2.70,
        ),
        _report(
            "set-replay",
            total_mae=2.55,
            total_rmse=3.35,
            component_mae=2.67,
            probe_offset=0.05,
        ),
    ]


def _r0_binding() -> dict:
    identity = {
        "r0_classification": report_tool.R0_COMPLETE_CLASSIFICATION,
        "classification_order_byte_identical": True,
        "r0_classification_aggregate_id": "b" * 64,
        "r0_classification_file_blake3": "c" * 64,
        "r0_order_proof_file_blake3": "d" * 64,
        "r0_selected_stage2_candidate": None,
        "selected_control_arm": report_tool.R0_EXACT_CONTROL,
        "r0_control_report_id": "e" * 64,
        "r0_control_report_file_blake3": "f" * 64,
        "validation": _metrics(2.65, 3.37, 2.70),
    }
    return {
        "schema_version": 1,
        "experiment_id": report_tool.EXPERIMENT_ID,
        "adr": report_tool.ADR_ID,
        "contract_id": report_tool.R0_BINDING_CONTRACT,
        "binding_id": report_tool.canonical_blake3(identity),
        "identity": identity,
    }


def test_complete_tournament_selects_best_r0_noninferior_architecture() -> None:
    output, exit_code = report_tool.classify_reports(_reports(), _r0_binding())
    assert exit_code == 0
    assert output["classification"] == report_tool.CLASSIFICATION_COMPLETE
    assert output["selected_run_role"] == "graph-primary"
    assert output["selected_architecture"] == "directional-graph-attention"
    assert output["claims"]["independent_replay_passed"] is True
    assert output["claims"]["promotion_authorized"] is False


def test_classifier_is_order_invariant() -> None:
    reports = _reports()
    forward, _ = report_tool.classify_reports(reports, _r0_binding())
    random.Random(17).shuffle(reports)
    shuffled, _ = report_tool.classify_reports(reports, _r0_binding())
    assert json.dumps(forward, sort_keys=True) == json.dumps(shuffled, sort_keys=True)


def test_replay_failure_has_fail_closed_classification() -> None:
    reports = _reports()
    reports[-1]["metrics"]["validation_probe"]["predictions"] = [
        [1.0] * 11 for _ in range(256)
    ]
    _reseal(reports[-1])
    output, exit_code = report_tool.classify_reports(reports, _r0_binding())
    assert exit_code == 5
    assert output["classification"] == report_tool.CLASSIFICATION_REPLAY_FAILURE
    assert output["selected_run_role"] is None


def test_semantic_failure_precedes_replay_failure() -> None:
    reports = _reports()
    reports[1]["integrity"]["padding_zero_verified"] = False
    reports[-1]["metrics"]["validation_probe"]["predictions"] = [
        [1.0] * 11 for _ in range(256)
    ]
    output, exit_code = report_tool.classify_reports(reports, _r0_binding())
    assert exit_code == 2
    assert output["classification"] == report_tool.CLASSIFICATION_SEMANTIC_FAILURE


def test_report_mutation_without_resealing_is_structurally_incomplete() -> None:
    reports = _reports()
    reports[0]["metrics"]["validation"]["total_mae"] = 1.0
    output, exit_code = report_tool.classify_reports(reports, _r0_binding())
    assert exit_code == 3
    assert output["classification"] == report_tool.CLASSIFICATION_INCOMPLETE


def test_exact_type_census_drift_is_a_semantic_failure() -> None:
    reports = _reports()
    train = reports[0]["cache"]["active_token_statistics"]["train"]
    train["type_tokens"]["frontier"]["total"] += 1
    train["active_tokens_total"] += 1
    train["active_tokens_mean"] = train["active_tokens_total"] / train["records"]
    train["padding_tokens_total"] -= 1
    for values in train["type_tokens"].values():
        values["fraction_of_active"] = (
            values["total"] / train["active_tokens_total"]
        )
    train["type_tokens"]["frontier"]["mean_per_position"] = (
        train["type_tokens"]["frontier"]["total"] / train["records"]
    )
    _reseal(reports[0])

    output, exit_code = report_tool.classify_reports(reports, _r0_binding())
    assert exit_code == 2
    assert output["classification"] == report_tool.CLASSIFICATION_SEMANTIC_FAILURE


def test_missing_r0_binding_is_structurally_incomplete() -> None:
    binding = _r0_binding()
    binding["identity"]["r0_classification"] = "incomplete"
    binding["binding_id"] = report_tool.canonical_blake3(binding["identity"])
    output, exit_code = report_tool.classify_reports(_reports(), binding)
    assert exit_code == 3
    assert output["classification"] == report_tool.CLASSIFICATION_INCOMPLETE
    assert output["selected_run_role"] is None
