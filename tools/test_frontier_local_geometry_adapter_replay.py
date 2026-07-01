from __future__ import annotations

import copy

from frontier_local_geometry_adapter_replay import compare_reports


def _report(host: str) -> dict:
    return {
        "experiment_id": (
            "complete-action-frontier-calibrated-local-geometry-adapter-v1"
        ),
        "scientific": {
            "arm": "calibrated-local-geometry-adapter-group",
            "group_index": 2,
            "value": 7,
        },
        "telemetry": {"host": host, "elapsed_seconds": 1.0},
    }


def test_identical_scientific_payloads_pass() -> None:
    comparison = compare_reports(
        _report("john1"),
        _report("john3"),
    )
    assert comparison["scientific_payload_identical"]
    assert comparison["group_index"] == 2


def test_scientific_drift_is_detected() -> None:
    origin = _report("john1")
    replay = copy.deepcopy(origin)
    replay["telemetry"]["host"] = "john2"
    replay["scientific"]["value"] = 8
    comparison = compare_reports(origin, replay)
    assert not comparison["scientific_payload_identical"]
