from __future__ import annotations

from frontier_local_geometry_balanced_replay import compare_reports


def test_identical_payloads_pass() -> None:
    scientific = {
        "arm": "local-geometry-balanced-target-group",
        "group_index": 0,
    }
    origin = {
        "experiment_id": (
            "complete-action-frontier-local-geometry-balanced-target-control-v1"
        ),
        "scientific": scientific,
        "telemetry": {"host": "john1"},
    }
    replay = {
        **origin,
        "telemetry": {"host": "john2"},
    }
    assert compare_reports(origin, replay)["scientific_payload_identical"]
