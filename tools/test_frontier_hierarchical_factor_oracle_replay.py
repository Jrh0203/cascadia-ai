from __future__ import annotations

from frontier_hierarchical_factor_oracle_replay import compare_reports


def test_identical_arm_payloads_pass() -> None:
    report = {
        "experiment_id": "full-legal-hierarchical-factor-oracle-v1",
        "scientific": {
            "arm": "conditional-wide",
            "arm_index": 2,
        },
        "telemetry": {"host": "john1"},
    }
    replay = {**report, "telemetry": {"host": "john4"}}
    assert compare_reports(report, replay)["scientific_payload_identical"]
