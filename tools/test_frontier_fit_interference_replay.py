from __future__ import annotations

from typing import Any

from frontier_fit_interference_replay import compare_reports


def _report(host: str, elapsed: float, metric: float = 0.5) -> dict[str, Any]:
    return {
        "experiment_id": "complete-action-frontier-fit-interference-audit-v1",
        "scientific": {
            "arm": "nested-subset",
            "trajectory": [
                {
                    "elapsed_seconds": elapsed,
                    "metrics": {"recall": metric},
                }
            ],
        },
        "telemetry": {
            "host": host,
            "elapsed_seconds": elapsed,
        },
    }


def test_replay_ignores_only_elapsed_seconds() -> None:
    report = compare_reports(_report("john1", 10.0), _report("john2", 20.0))
    assert report["scientific_payload_identical"]
    assert report["origin_host"] == "john1"
    assert report["replay_host"] == "john2"


def test_replay_detects_scientific_metric_drift() -> None:
    report = compare_reports(
        _report("john1", 10.0),
        _report("john2", 20.0, metric=0.6),
    )
    assert not report["scientific_payload_identical"]
