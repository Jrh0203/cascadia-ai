from __future__ import annotations

import copy

from frontier_local_geometry_feasibility_replay import compare_reports


def _report(host: str) -> dict:
    return {
        "experiment_id": (
            "complete-action-frontier-local-geometry-feasibility-forensic-v1"
        ),
        "scientific": {
            "arm": "local-geometry-feasibility-group",
            "group_index": 1,
            "value": 3,
        },
        "telemetry": {"host": host},
    }


def test_identical_payloads_pass() -> None:
    assert compare_reports(
        _report("john1"),
        _report("john4"),
    )["scientific_payload_identical"]


def test_payload_drift_fails() -> None:
    origin = _report("john1")
    replay = copy.deepcopy(origin)
    replay["telemetry"]["host"] = "john2"
    replay["scientific"]["value"] = 4
    assert not compare_reports(origin, replay)[
        "scientific_payload_identical"
    ]
