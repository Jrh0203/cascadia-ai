from __future__ import annotations

from typing import Any

from frontier_projected_repair_replay import compare_reports


def _report(host: str, metric: float) -> dict[str, Any]:
    return {
        "experiment_id": "complete-action-frontier-projected-control-repair-v1",
        "scientific": {
            "arm": "projected-control-repair-shard",
            "shard_index": 0,
            "metric": metric,
        },
        "telemetry": {"host": host},
    }


def test_repair_replay_detects_scientific_drift() -> None:
    assert compare_reports(
        _report("john1", 1.0),
        _report("john2", 1.0),
    )["scientific_payload_identical"]
    assert not compare_reports(
        _report("john1", 1.0),
        _report("john2", 2.0),
    )["scientific_payload_identical"]
