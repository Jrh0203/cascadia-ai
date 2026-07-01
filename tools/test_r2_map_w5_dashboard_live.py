from __future__ import annotations

import copy

import pytest
from r2_map_w5_dashboard_live import (
    DEFAULT_ENDPOINTS,
    DashboardValidationError,
    canonical_json,
    document_sha256,
    validate_live_dashboard,
    validate_response,
)


def _response(updated: int, *, endpoint: str = DEFAULT_ENDPOINTS[0]) -> dict[str, object]:
    status = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.dashboard-status.v1",
        "campaign_id": "r2-map-expert-iteration-v1",
        "updated_unix_ms": updated,
        "stale_after_seconds": 30,
        "phase": "contracts-ready",
        "legal_next_transitions": ["bootstrap-generating"],
        "round_index": None,
        "models": {"incumbent": None, "candidate": None, "opponent_pool": []},
        "hosts": {"john1": {}, "john2": {}, "john3": {}},
        "training": {"active": False},
        "benchmark": {"active": False, "classification": "pending"},
    }
    return {
        "schema_version": 1,
        "configured": True,
        "condition": "fresh",
        "source_path": "artifacts/cluster/r2-map-dashboard-serving-projection-v2.json",
        "observed_unix_ms": updated + 1_000,
        "updated_unix_ms": updated,
        "age_seconds": 1.0,
        "stale_after_seconds": 30,
        "status": status,
        "error": None,
        "endpoint": endpoint,
    }


def test_valid_live_samples_are_hash_bound_and_advance() -> None:
    calls = 0
    sleeps: list[float] = []

    def fetcher(endpoint: str, *, timeout_seconds: float) -> dict[str, object]:
        nonlocal calls
        assert timeout_seconds == 5.0
        interval = calls // 2
        calls += 1
        value = _response(1_000 + interval * 10, endpoint=endpoint)
        return validate_response(value, endpoint=endpoint)

    result = validate_live_dashboard(
        sample_count=3,
        interval_seconds=10.0,
        fetcher=fetcher,
        sleeper=sleeps.append,
    )
    assert calls == 6
    assert sleeps == [10.0, 10.0]
    assert result["sample_count"] == 3
    assert result["summary_sha256"] == document_sha256(result)


def test_stale_or_non_revision_zero_payload_fails_closed() -> None:
    stale = _response(1_000)
    stale["condition"] = "stale"
    with pytest.raises(DashboardValidationError, match="fresh v2"):
        validate_response(stale, endpoint=DEFAULT_ENDPOINTS[0])

    advanced = _response(1_000)
    advanced["status"]["phase"] = "bootstrap-generating"  # type: ignore[index]
    with pytest.raises(DashboardValidationError, match="revision-0"):
        validate_response(advanced, endpoint=DEFAULT_ENDPOINTS[0])


def test_endpoint_divergence_or_nonadvancement_fails_closed() -> None:
    calls = 0

    def divergent(endpoint: str, *, timeout_seconds: float) -> dict[str, object]:
        nonlocal calls
        calls += 1
        value = _response(1_000, endpoint=endpoint)
        if calls == 2:
            value["status"]["hosts"]["john2"] = {"detail": "drift"}  # type: ignore[index]
        return validate_response(value, endpoint=endpoint)

    with pytest.raises(DashboardValidationError, match="payloads differ"):
        validate_live_dashboard(
            sample_count=2,
            interval_seconds=10.0,
            fetcher=divergent,
            sleeper=lambda _: None,
        )

    def static(endpoint: str, *, timeout_seconds: float) -> dict[str, object]:
        return validate_response(_response(1_000, endpoint=endpoint), endpoint=endpoint)

    with pytest.raises(DashboardValidationError, match="did not advance"):
        validate_live_dashboard(
            sample_count=2,
            interval_seconds=10.0,
            fetcher=static,
            sleeper=lambda _: None,
        )


def test_response_shape_and_canonical_json_are_strict() -> None:
    malformed = _response(1_000)
    malformed["age_seconds"] = float("nan")
    with pytest.raises(DashboardValidationError, match="finite"):
        validate_response(malformed, endpoint=DEFAULT_ENDPOINTS[0])
    with pytest.raises(ValueError):
        canonical_json({"nan": float("nan")})

    wrong_hosts = copy.deepcopy(_response(1_000))
    wrong_hosts["status"]["hosts"]["john4"] = {}  # type: ignore[index]
    with pytest.raises(DashboardValidationError, match="revision-0"):
        validate_response(wrong_hosts, endpoint=DEFAULT_ENDPOINTS[0])
