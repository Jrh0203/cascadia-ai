from __future__ import annotations

from frontier_calibrated_adamw_replay import compare_reports


def _report(host: str) -> dict[str, object]:
    return {
        "experiment_id": (
            "complete-action-frontier-calibrated-monotone-adamw-v1"
        ),
        "scientific": {
            "arm": "calibrated-free-residual-group",
            "group_index": 2,
            "final": {"target_positive_recall": 1.0},
        },
        "telemetry": {"host": host},
    }


def test_optimizer_replay_ignores_host() -> None:
    report = compare_reports(_report("john1"), _report("john3"))
    assert report["scientific_payload_identical"]
