from __future__ import annotations

from typing import Any

from frontier_free_residual_replay import compare_reports


def _report(
    host: str,
    elapsed: float,
    *,
    metric: float = 0.5,
    group_index: int | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": "complete-action-frontier-free-residual-audit-v1",
        "scientific": {
            "arm": (
                "analytic-optimum"
                if group_index is None
                else "neural-continuation-shard"
            ),
            "group_index": group_index,
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


def test_replay_ignores_elapsed_time() -> None:
    report = compare_reports(_report("john1", 1.0), _report("john2", 2.0))
    assert report["scientific_payload_identical"]


def test_replay_detects_metric_or_shard_drift() -> None:
    assert not compare_reports(
        _report("john1", 1.0),
        _report("john2", 2.0, metric=0.6),
    )["scientific_payload_identical"]
