from __future__ import annotations

import json

import v3_part1_infrastructure as infrastructure


class _Response:
    def __init__(self, value: dict[str, object]) -> None:
        self._payload = json.dumps(value)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload.encode()


def test_dashboard_hosts_accepts_live_api_status_envelope(monkeypatch) -> None:
    response = _Response(
        {"status": {"hosts": {name: {} for name in ("john4", "john2", "john1", "john3")}}}
    )
    monkeypatch.setattr(infrastructure.urllib.request, "urlopen", lambda *_a, **_k: response)

    assert infrastructure._dashboard_hosts("http://dashboard") == [
        "john1",
        "john2",
        "john3",
        "john4",
    ]


def test_dashboard_hosts_accepts_live_cluster_nodes_array(monkeypatch) -> None:
    response = _Response(
        {"nodes": [{"id": name} for name in ("john4", "john2", "john1", "john3")]}
    )
    monkeypatch.setattr(infrastructure.urllib.request, "urlopen", lambda *_a, **_k: response)

    assert infrastructure._dashboard_hosts("http://dashboard") == [
        "john1",
        "john2",
        "john3",
        "john4",
    ]
