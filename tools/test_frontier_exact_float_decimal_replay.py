from __future__ import annotations

from frontier_arbitrary_precision_replay import compare_reports

EXPERIMENT_ID = "complete-action-frontier-exact-float-decimal-control-v1"
SCIENTIFIC_ARM = "exact-float-decimal-control-group"


def _report(host: str) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "arm": SCIENTIFIC_ARM,
            "group_index": 4,
            "objective": "2.5",
        },
        "telemetry": {"host": host},
    }


def test_exact_float_replay_uses_new_identity() -> None:
    report = compare_reports(
        _report("john1"),
        _report("john4"),
        experiment_id=EXPERIMENT_ID,
        scientific_arm=SCIENTIFIC_ARM,
    )
    assert report["experiment_id"] == EXPERIMENT_ID
    assert report["scientific_payload_identical"]
