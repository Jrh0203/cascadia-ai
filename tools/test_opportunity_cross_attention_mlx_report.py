from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import opportunity_cross_attention_mlx_report as report_module
import pytest
from cascadia_mlx.opportunity_cross_attention_mlx_model import ARMS
from cascadia_mlx.opportunity_cross_attention_mlx_pairwise import (
    panel_identity,
)
from opportunity_cross_attention_mlx_report import (
    CLASSIFICATION_ADVANCE,
    CLASSIFICATION_QUALITY,
    PARENT_ARM,
    _canonical_blake3,
    _select_arm,
    aggregate_with_order_proof,
    classify_reports,
    invalid_outputs,
    validate_collected_checkpoints,
)


def _panel(recalls: list[bool], *, squared_error: float) -> list[dict]:
    records = []
    for row, recalled in enumerate(recalls):
        records.append(
            {
                "row": row,
                "group_id": 1000 + row,
                "turn": row,
                "candidates": 80,
                "labeled_candidates": 64,
                "teacher_winner_index": 0,
                "teacher_winner_action_hash": f"{row + 1:064x}",
                "winner_rank": 1 if recalled else 65,
                "top64_recalled": recalled,
                "top64_regret": 0.0 if recalled else 0.2,
                "absolute_error_sum": 1.0,
                "squared_error_sum": squared_error,
                "bias_sum": 0.0,
                "low_supply": row >= 200,
                "independent_draft_winner": row % 5 == 0,
                "phase": (
                    "early" if row < 80 else "middle" if row < 160 else "late"
                ),
                "opportunities": {
                    "elk": row % 3 == 0,
                    "salmon": row % 3 == 1,
                    "hawk": row % 3 == 2,
                    "bear": row % 4 == 0,
                },
                "prediction_blake3": f"{10_000 + row:064x}",
            }
        )
    return records


def _performance() -> dict:
    return {
        "combined_with_r6": {
            "groups": 240,
            "actions": 860_203,
            "r6_exact_parity_pass": True,
            "action_scores_per_second": 50_000.0,
            "latency_milliseconds": {"p99": 100.0},
        },
        "r6_apply_undo": {
            "exact_parity_pass": True,
            "apply_failures": 0,
            "undo_failures": 0,
        },
        "memory": {
            "process_swaps": 0,
            "system_swap_delta_bytes": 0,
            "peak_process_rss_bytes": 1024**3,
            "peak_active_bytes": 512 * 1024**2,
        },
    }


def _metrics() -> dict:
    return {
        "groups": 240,
        "candidates": 860_203,
        "all_groups_scored_once": True,
        "all_candidates_scored_once": True,
        "all_scores_and_uncertainties_finite": True,
    }


def _report(arm: str, panel: list[dict]) -> dict:
    common_model = {
        "total_parameter_count": 10,
        "total_parameter_layout_blake3": "1" * 64,
        "adapter_parameter_count": 2,
        "adapter_parameter_layout_blake3": "2" * 64,
        "initial_all_parameter_tensor_blake3": "3" * 64,
        "initial_adapter_parameter_tensor_blake3": "4" * 64,
        "base_parameter_tensor_blake3": "5" * 64,
    }
    trace = [
        {
            "step": step,
            "batch_blake3": f"{step:064x}"[-64:],
            "candidates": 100,
            "loss": 1.0,
            "elapsed_seconds": 0.1,
        }
        for step in range(1, 2001)
    ]
    report = {
        "schema_version": 1,
        "experiment_id": "opportunity-cross-attention-mlx-tournament-v1",
        "protocol_id": "exact-r2-opportunity-query-factorial-v1",
        "adr": "0166",
        "mode": "production",
        "arm": arm,
        "host": {
            ARMS[0]: "john1",
            ARMS[1]: "john2",
            ARMS[2]: "john3",
            ARMS[3]: "john4",
        }[arm],
        "data_arm": "c0-exact-r2",
        "r3_cache_id": "6" * 64,
        "relational_cache_id": "7" * 64,
        "s1_cache_id": "8" * 64,
        "r6_binary": {"blake3": "9" * 64},
        "protocol": {"frozen": True},
        "warm_start": {"warm_start_id": "a" * 64},
        "model": common_model,
        "optimization": {
            "global_step": 2000,
            "loss_trace": trace,
        },
        "metrics": _metrics(),
        "paired_panel": panel,
        "paired_panel_id": panel_identity(panel),
        "performance": _performance(),
        "source": {"v2_source_blake3": "b" * 64},
        "controls": {
            "authorization_id": "c" * 64,
            "open_data_verification_id": "d" * 64,
        },
        "information_boundary": {
            "sealed_test_opened": False,
            "gameplay_run": False,
        },
        "claims": {
            "offline_comparison_complete": True,
            "base_parameters_frozen": True,
        },
    }
    report["scientific_identity"] = {
        key: report[key]
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "arm",
            "host",
            "data_arm",
            "r3_cache_id",
            "relational_cache_id",
            "s1_cache_id",
            "r6_binary",
            "protocol",
            "warm_start",
            "model",
            "optimization",
            "metrics",
            "paired_panel",
            "paired_panel_id",
            "performance",
            "source",
            "controls",
            "information_boundary",
            "claims",
        )
    }
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    return report


def _control(panel: list[dict], parent: dict) -> dict:
    identity = {
        "experiment_id": parent["experiment_id"],
        "protocol_id": parent["protocol_id"],
        "adr": parent["adr"],
        "control_kind": "untouched-c0-exact-r2",
        "warm_start": parent["warm_start"],
        "source_report_id": "e" * 64,
        "checkpoint": {"model_blake3": "f" * 64},
        "r3_cache_id": parent["r3_cache_id"],
        "relational_cache_id": parent["relational_cache_id"],
        "s1_cache_id": parent["s1_cache_id"],
        "validation_dataset_id": "0" * 64,
        "metrics": _metrics(),
        "performance": _performance(),
        "paired_panel": panel,
        "paired_panel_id": panel_identity(panel),
        "information_boundary": {
            "sealed_test_opened": False,
            "gameplay_run": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": identity["experiment_id"],
        "protocol_id": identity["protocol_id"],
        "adr": identity["adr"],
        "control_kind": identity["control_kind"],
        "control_id": _canonical_blake3(identity),
        "scientific_identity": identity,
        **{
            key: identity[key]
            for key in identity
            if key
            not in ("experiment_id", "protocol_id", "adr", "control_kind")
        },
    }


def _comparison(
    *,
    global_delta: float = 0.0,
    global_probability: float = 0.5,
    strategic_delta: float = 0.0,
    strategic_probability: float = 0.5,
    low_supply_delta: float = 0.0,
    independent_draft_delta: float = 0.0,
    rmse_delta: float = 0.0,
    regret_delta: float = 0.0,
) -> dict:
    return {
        "global_top64_recall": {
            "delta": global_delta,
            "probability_favorable": global_probability,
        },
        "strategic_top64_recall": {
            "delta": strategic_delta,
            "probability_favorable": strategic_probability,
        },
        "protected": {
            "low_supply": {"delta": low_supply_delta},
            "independent_draft_winner": {
                "delta": independent_draft_delta
            },
        },
        "r4800_rmse": {"delta": rmse_delta},
        "top64_regret": {"delta": regret_delta},
    }


def _checkpoint_report(
    arm: str,
    source_path: Path,
    manifest_blake3: str,
    model_blake3: str,
) -> dict:
    return {
        "arm": arm,
        "checkpoint": {
            "path": str(source_path),
            "manifest_blake3": manifest_blake3,
            "model_blake3": model_blake3,
        },
    }


def test_collected_checkpoint_transport_directory_need_not_equal_checkpoint_id(
    tmp_path: Path,
) -> None:
    reports = []
    checkpoint_dirs = {}
    for index, arm in enumerate(ARMS):
        checkpoint_id = f"step-{index:09d}-epoch-0000-batch-{index:06d}"
        transport_dir = tmp_path / arm.replace("-", "_")
        transport_dir.mkdir()
        manifest = {
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "model_config": {"arm": arm},
        }
        manifest_path = transport_dir / "checkpoint.json"
        model_path = transport_dir / "model.safetensors"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        model_path.write_bytes(f"model-{arm}".encode())
        reports.append(
            _checkpoint_report(
                arm,
                Path("/remote/checkpoints") / checkpoint_id,
                report_module._file_blake3(manifest_path),
                report_module._file_blake3(model_path),
            )
        )
        checkpoint_dirs[arm] = transport_dir

    validate_collected_checkpoints(reports, checkpoint_dirs)


def test_collected_checkpoint_rejects_intrinsic_checkpoint_id_mismatch(
    tmp_path: Path,
) -> None:
    reports = []
    checkpoint_dirs = {}
    for arm in ARMS:
        transport_dir = tmp_path / arm.replace("-", "_")
        transport_dir.mkdir()
        manifest_path = transport_dir / "checkpoint.json"
        model_path = transport_dir / "model.safetensors"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_id": "actual-checkpoint",
                    "model_config": {"arm": arm},
                },
                sort_keys=True,
            )
        )
        model_path.write_bytes(f"model-{arm}".encode())
        reports.append(
            _checkpoint_report(
                arm,
                Path("/remote/checkpoints/reported-checkpoint"),
                report_module._file_blake3(manifest_path),
                report_module._file_blake3(model_path),
            )
        )
        checkpoint_dirs[arm] = transport_dir

    with pytest.raises(
        report_module.OpportunityReportError,
        match="collected checkpoint differs",
    ):
        validate_collected_checkpoints(reports, checkpoint_dirs)


def test_invalid_outputs_sort_report_evidence_by_path(tmp_path: Path) -> None:
    later = tmp_path / "z-report.json"
    earlier = tmp_path / "a-report.json"
    control = tmp_path / "control.json"
    for path in (later, earlier, control):
        path.write_text("{}")

    forward, reverse, proof = invalid_outputs(
        ValueError("broken evidence"),
        [later, earlier],
        control,
    )

    paths = [
        item["path"]
        for item in forward["scientific_identity"]["inputs"]["reports"]
    ]
    assert paths == sorted(paths)
    assert reverse == forward
    assert proof["scientific_identity"]["invalid_evidence"] is True


def test_candidate_query_advances_with_order_proof() -> None:
    base = _panel([row % 4 != 0 for row in range(240)], squared_error=64.0)
    parent_panel = deepcopy(base)
    for row in range(0, 240, 20):
        parent_panel[row]["top64_recalled"] = True
    t1_panel = deepcopy(parent_panel)
    for row in range(4, 240, 12):
        t1_panel[row]["top64_recalled"] = True
        t1_panel[row]["top64_regret"] = 0.0
    reports = [
        _report(PARENT_ARM, parent_panel),
        _report(ARMS[1], t1_panel),
        _report(ARMS[2], parent_panel),
        _report(ARMS[3], parent_panel),
    ]
    control = _control(base, reports[0])

    forward, reverse, proof = aggregate_with_order_proof(
        reports,
        control,
        bootstrap_replicates=2_000,
    )

    assert forward["classification"] == CLASSIFICATION_ADVANCE
    assert forward["selected_arm"] == ARMS[1]
    assert forward["aggregate_id"] == reverse["aggregate_id"]
    assert proof["scientific_identity"]["byte_identical"] is True


def test_protected_failure_is_scoped_to_same_utility_positive_arm(
    monkeypatch,
) -> None:
    panel = _panel(
        [row % 4 != 0 for row in range(240)],
        squared_error=64.0,
    )
    reports = [_report(arm, deepcopy(panel)) for arm in ARMS]
    control = _control(deepcopy(panel), reports[0])
    responses = iter(
        [
            _comparison(),
            _comparison(
                global_delta=0.05,
                global_probability=0.99,
                strategic_delta=0.05,
                strategic_probability=0.95,
                rmse_delta=0.10,
            ),
            _comparison(global_delta=0.05),
            _comparison(low_supply_delta=-0.05),
            _comparison(),
            _comparison(),
            _comparison(),
        ]
    )
    monkeypatch.setattr(
        report_module,
        "compare_decision_panels",
        lambda *_args, **_kwargs: deepcopy(next(responses)),
    )
    monkeypatch.setattr(
        report_module,
        "factorial_effects",
        lambda _panels: {},
    )

    result = classify_reports(
        reports,
        control,
        bootstrap_replicates=10,
    )

    assert result["classification"] == CLASSIFICATION_QUALITY


def test_selection_uses_rmse_before_latency_and_memory() -> None:
    assessments = {
        ARMS[1]: {
            "versus_parent": _comparison(
                global_delta=0.05,
                strategic_delta=0.04,
                regret_delta=-0.01,
                rmse_delta=-0.02,
            ),
            "serving": {
                "measurements": {
                    "p99_milliseconds": 200.0,
                    "peak_process_rss_bytes": 3 * 1024**3,
                }
            },
        },
        ARMS[2]: {
            "versus_parent": _comparison(
                global_delta=0.05,
                strategic_delta=0.04,
                regret_delta=-0.01,
                rmse_delta=-0.01,
            ),
            "serving": {
                "measurements": {
                    "p99_milliseconds": 100.0,
                    "peak_process_rss_bytes": 1024**3,
                }
            },
        },
    }

    assert _select_arm(list(assessments), assessments) == ARMS[1]


def test_selection_uses_memory_after_latency() -> None:
    assessments = {
        arm: {
            "versus_parent": _comparison(
                global_delta=0.05,
                strategic_delta=0.04,
                regret_delta=-0.01,
                rmse_delta=-0.02,
            ),
            "serving": {
                "measurements": {
                    "p99_milliseconds": 100.0,
                    "peak_process_rss_bytes": rss,
                }
            },
        }
        for arm, rss in (
            (ARMS[1], 2 * 1024**3),
            (ARMS[2], 1024**3),
        )
    }

    assert _select_arm(list(assessments), assessments) == ARMS[2]
