from __future__ import annotations

from cluster_campaign_telemetry import summarize


def test_summarize_uses_reachable_core_weighted_samples() -> None:
    samples = [
        {
            "timestamp_unix_ms": 1000,
            "nodes": [
                {
                    "node_id": "a",
                    "reachable": True,
                    "cpu_percent": 100.0,
                    "memory_percent": 50.0,
                },
                {
                    "node_id": "b",
                    "reachable": True,
                    "cpu_percent": 0.0,
                    "memory_percent": 25.0,
                },
            ],
        },
        {
            "timestamp_unix_ms": 2000,
            "nodes": [
                {
                    "node_id": "a",
                    "reachable": False,
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                },
                {
                    "node_id": "b",
                    "reachable": True,
                    "cpu_percent": 50.0,
                    "memory_percent": 75.0,
                },
            ],
        },
    ]
    report = summarize(
        samples,
        start_unix_ms=1000,
        end_unix_ms=2000,
        cores={"a": 2, "b": 6},
    )
    assert report["observed_samples"] == 2
    assert report["mean_core_weighted_cpu_percent"] == 37.5
    assert report["peak_core_weighted_cpu_percent"] == 50.0
    assert report["nodes"]["a"]["reachable_fraction"] == 0.5
    assert report["nodes"]["a"]["mean_cpu_percent"] == 100.0
    assert report["nodes"]["b"]["mean_cpu_percent"] == 25.0
