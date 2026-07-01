from __future__ import annotations

import urllib.request

import pytest
from cascadia_cluster.bacalhau_api import BacalhauAPI
from cascadia_cluster.errors import BacalhauAPIError


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_invalid_json_response_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter((_Response(b""), _Response(b'{"Status":"OK"}')))
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("cascadia_cluster.bacalhau_api.time.sleep", sleeps.append)

    api = BacalhauAPI(
        "http://bacalhau.test",
        maximum_attempts=3,
        initial_backoff_seconds=0.01,
    )

    assert api.alive()
    assert calls == 2
    assert sleeps == [0.01]


def test_invalid_json_exhausts_bounded_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(b"not-json")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("cascadia_cluster.bacalhau_api.time.sleep", sleeps.append)

    api = BacalhauAPI(
        "http://bacalhau.test",
        maximum_attempts=3,
        initial_backoff_seconds=0.01,
    )

    with pytest.raises(
        BacalhauAPIError,
        match=r"returned invalid JSON after 3 attempts \(8 bytes in final response\)",
    ):
        api.alive()

    assert calls == 3
    assert sleeps == [0.01, 0.02]
