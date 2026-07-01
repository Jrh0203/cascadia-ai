from __future__ import annotations

from frontier_arbitrary_precision_replay import compare_reports


def _report(host: str, objective: str = "1.25") -> dict[str, object]:
    return {
        "experiment_id": (
            "complete-action-frontier-arbitrary-precision-control-v1"
        ),
        "scientific": {
            "arm": "arbitrary-precision-control-group",
            "group_index": 3,
            "objective": objective,
        },
        "telemetry": {"host": host, "elapsed_seconds": 1.0},
    }


def test_replay_ignores_host_and_timing() -> None:
    report = compare_reports(_report("john1"), _report("john3"))
    assert report["scientific_payload_identical"]
    assert report["origin_scientific_blake3"] == report[
        "replay_scientific_blake3"
    ]


def test_replay_detects_scientific_difference() -> None:
    report = compare_reports(
        _report("john1"),
        _report("john3", objective="1.2500000001"),
    )
    assert not report["scientific_payload_identical"]
