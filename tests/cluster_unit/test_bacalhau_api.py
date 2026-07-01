from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest
from cascadia_cluster import BacalhauAPIError
from cascadia_cluster.bacalhau_api import BacalhauAPI


class _Response:
    def __init__(self, value: dict) -> None:
        self.value = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.value


def test_transient_rest_failure_is_retried_with_bounded_backoff(monkeypatch) -> None:
    calls: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: float):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "unavailable",
                {},
                io.BytesIO(b"temporary"),
            )
        return _Response({"Status": "OK"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("cascadia_cluster.bacalhau_api.time.sleep", lambda _value: None)
    api = BacalhauAPI("http://scheduler", maximum_attempts=3)
    assert api.alive()
    assert len(calls) == 3


def test_job_listing_uses_recovery_timeout_without_relaxing_health_calls(monkeypatch) -> None:
    observed: list[tuple[str, float]] = []

    def urlopen(request: urllib.request.Request, *, timeout: float):
        observed.append((request.full_url, timeout))
        if request.full_url.endswith("/api/v1/agent/alive"):
            return _Response({"Status": "OK"})
        return _Response({"Items": []})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    api = BacalhauAPI(
        "http://scheduler",
        request_timeout_seconds=3.0,
        list_jobs_timeout_seconds=45.0,
    )
    assert api.alive()
    assert api.list_jobs(labels={"cascadia.request_id": "request"}) == []
    assert observed == [
        ("http://scheduler/api/v1/agent/alive", 3.0),
        (
            "http://scheduler/api/v1/orchestrator/jobs?"
            "labels=cascadia.request_id%3Drequest",
            45.0,
        ),
    ]


@pytest.mark.parametrize(
    "field",
    ["request_timeout_seconds", "list_jobs_timeout_seconds"],
)
def test_request_timeouts_must_be_positive(field: str) -> None:
    with pytest.raises(BacalhauAPIError, match="retry bounds"):
        BacalhauAPI("http://scheduler", **{field: 0.0})


def test_nonretryable_rest_failure_is_immediate(monkeypatch) -> None:
    calls = 0

    def urlopen(request: urllib.request.Request, *, timeout: float):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            {},
            io.BytesIO(b"invalid"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    api = BacalhauAPI("http://scheduler")
    with pytest.raises(BacalhauAPIError, match="invalid"):
        api.alive()
    assert calls == 1


def test_unchanged_job_submission_recovers_existing_job_id(monkeypatch) -> None:
    calls = 0

    def urlopen(request: urllib.request.Request, *, timeout: float):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "internal server error",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "Status": 500,
                        "Message": (
                            "no changes detected for new job spec. "
                            "Job Name: 'worker', Job Id: 'j-existing'"
                        ),
                    }
                ).encode()
            ),
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    response = BacalhauAPI("http://scheduler").submit(
        {"Name": "worker"}, idempotency_token="stable-token"
    )
    assert response == {"JobID": "j-existing", "Recovered": True}
    assert calls == 1


def test_unrelated_internal_server_error_is_still_retried(monkeypatch) -> None:
    calls = 0

    def urlopen(request: urllib.request.Request, *, timeout: float):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "internal server error",
            {},
            io.BytesIO(b'{"Status":500,"Message":"database unavailable"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("cascadia_cluster.bacalhau_api.time.sleep", lambda _value: None)
    with pytest.raises(BacalhauAPIError, match="database unavailable"):
        BacalhauAPI("http://scheduler", maximum_attempts=3).submit(
            {"Name": "worker"}, idempotency_token="stable-token"
        )
    assert calls == 3


def test_nodes_uses_v19_nodes_response_key(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, *, timeout: _Response({"Nodes": [{"Info": {"NodeID": "john2"}}]}),
    )
    assert BacalhauAPI("http://scheduler").nodes()[0]["Info"]["NodeID"] == "john2"


def test_executions_normalize_v19_numeric_state_enums(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, *, timeout: _Response(
            {
                "Items": [
                    {"ID": "rejected", "ComputeState": {"StateType": 4}},
                    {"ID": "accepted", "ComputeState": {"StateType": 9}},
                ]
            }
        ),
    )
    assert [
        item["ComputeState"]["StateType"]
        for item in BacalhauAPI("http://scheduler").executions("job")
    ] == ["AskForBidRejected", "Completed"]
