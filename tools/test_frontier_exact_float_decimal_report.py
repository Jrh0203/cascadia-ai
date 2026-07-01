from __future__ import annotations

from frontier_exact_float_decimal_report import render_markdown


def test_report_authorizes_optimizer_only_after_pass() -> None:
    combined = {
        "experiment_id": (
            "complete-action-frontier-exact-float-decimal-control-v1"
        ),
        "scientific": {
            "classification": "frozen_optimizer_hyperparameters_insufficient",
            "aggregate": {
                "target_positive_recall": 1.0,
                "target_set_exact_fraction": 1.0,
            },
            "maximum_normalization_residual": "1e-90",
            "maximum_kkt_violation": "1e-90",
            "maximum_objective_difference": "1e-14",
            "maximum_offset_difference": "1e-14",
            "groups": [],
            "gates": {"control_pipeline_passed": True},
        },
    }
    campaign = {
        "origin_makespan_seconds": 1.0,
        "campaign_wall_seconds": 2.0,
        "scheduled_process_seconds": 4.0,
        "confirmation_compute_fraction": 0.5,
        "mean_active_group_processes": 2.0,
        "maximum_active_group_processes": 4,
        "group_process_core_occupancy": 0.05,
        "idle_slot_seconds_with_compatible_work": 0.0,
        "host_tasks": {f"john{i}": 12 for i in range(1, 5)},
        "host_seconds": {f"john{i}": 1.0 for i in range(1, 5)},
        "final_capacities": {f"john{i}": 4 for i in range(1, 5)},
    }
    text = render_markdown(
        combined_report=combined,
        source_identity={
            "files": 112,
            "bundle_sha256": "a" * 64,
            "hosts": ["john1", "john2", "john3", "john4"],
        },
        replay_summary={"reports": []},
        campaign_summary=campaign,
    )
    assert "authorizes exactly one calibrated local optimizer" in text
