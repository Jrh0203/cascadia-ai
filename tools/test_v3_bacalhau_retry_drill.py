from __future__ import annotations

from v3_bacalhau_retry_drill import (
    DRILL_SLEEP_SECONDS,
    RETRY_COMPLETION_TIMEOUT_SECONDS,
    _active,
    _payload,
)


class _API:
    def __init__(self, executions: list[dict[str, object]]) -> None:
        self._executions = executions

    def executions(self, _job_id: str) -> list[dict[str, object]]:
        return self._executions


def test_active_accepts_native_and_bacalhau_1_9_live_encodings() -> None:
    native = {"ID": "native", "ComputeState": {"StateType": "Running"}}
    observed = {
        "ID": "observed",
        "ComputeState": {"StateType": "BidAccepted", "Message": "Running"},
    }
    accepted_only = {
        "ID": "accepted-only",
        "ComputeState": {"StateType": "BidAccepted", "Message": "accepted"},
    }
    completed = {"ID": "completed", "ComputeState": {"StateType": "Completed"}}

    assert [item["ID"] for item in _active(
        _API([native, observed, accepted_only, completed]), "job"
    )] == ["native", "observed"]


def test_payload_reserves_a_whole_worker_without_scientific_eligibility() -> None:
    payload = _payload("drill", "registry/image@sha256:abc", DRILL_SLEEP_SECONDS)

    assert payload["Meta"] == {"cascadia.v3.scientific_eligible": "false"}
    task = payload["Tasks"][0]
    assert task["Resources"]["CPU"] == "10"
    assert task["Engine"]["Params"]["Parameters"] == [
        "-c",
        f"sleep {DRILL_SLEEP_SECONDS}",
    ]
    assert RETRY_COMPLETION_TIMEOUT_SECONDS > DRILL_SLEEP_SECONDS
