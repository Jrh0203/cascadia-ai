from __future__ import annotations

from frontier_calibrated_neural_replay import compare_reports


def _report(host: str) -> dict[str, object]:
    return {
        "experiment_id": (
            "complete-action-frontier-calibrated-neural-stage-v1"
        ),
        "scientific": {
            "arm": "calibrated-neural-local-fit-group",
            "group_index": 2,
            "final": {"target_positive_recall": 1.0},
        },
        "telemetry": {"host": host},
    }


def test_neural_replay_ignores_host() -> None:
    report = compare_reports(_report("john2"), _report("john4"))
    assert report["scientific_payload_identical"]
