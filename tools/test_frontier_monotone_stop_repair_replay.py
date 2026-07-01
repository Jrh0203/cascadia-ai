from __future__ import annotations

from frontier_monotone_stop_repair_replay import compare_reports


def _report(host: str) -> dict[str, object]:
    return {
        "experiment_id": (
            "complete-action-frontier-monotone-adamw-stop-repair-v1"
        ),
        "scientific": {
            "arm": "monotone-adamw-stop-repair-group",
            "group_index": 8,
            "numerical_convergence": {
                "smallest_attempted_rate": 1e-8,
            },
        },
        "telemetry": {"host": host},
    }


def test_stop_repair_replay_ignores_host() -> None:
    report = compare_reports(_report("john1"), _report("john4"))
    assert report["scientific_payload_identical"]
