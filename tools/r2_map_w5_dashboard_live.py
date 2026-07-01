#!/usr/bin/env python3
"""Validate the single-writer R2-MAP dashboard across live publisher intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

SCHEMA_ID = "cascadia.r2-map.w5-live-dashboard-validation.v1"
STATUS_SCHEMA_ID = "cascadia.r2-map.dashboard-status.v1"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
SERVING_SOURCE = "artifacts/cluster/r2-map-dashboard-serving-projection-v2.json"
DEFAULT_ENDPOINTS = (
    "http://127.0.0.1:5187/api/v1/cluster/r2-map",
    "http://100.110.109.6:5187/api/v1/cluster/r2-map",
)
MAX_RESPONSE_BYTES = 1 << 20


class DashboardValidationError(RuntimeError):
    """The live dashboard is stale, inconsistent, or outside revision zero."""


class Response(Protocol):
    status: int

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> Response: ...

    def __exit__(self, *arguments: object) -> None: ...


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def document_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("summary_sha256", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DashboardValidationError(f"{label} is not a nonnegative integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DashboardValidationError(f"{label} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise DashboardValidationError(f"{label} is not finite and nonnegative")
    return numeric


def validate_response(value: Any, *, endpoint: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DashboardValidationError("dashboard response is not an object")
    if (
        value.get("schema_version") != 1
        or value.get("configured") is not True
        or value.get("condition") != "fresh"
        or value.get("error") is not None
        or value.get("source_path") != SERVING_SOURCE
    ):
        raise DashboardValidationError("dashboard response is not a fresh v2 projection")
    observed = _integer(value.get("observed_unix_ms"), "observed_unix_ms")
    updated = _integer(value.get("updated_unix_ms"), "updated_unix_ms")
    age = _finite_number(value.get("age_seconds"), "age_seconds")
    stale_after = _integer(value.get("stale_after_seconds"), "stale_after_seconds")
    if stale_after == 0 or age >= stale_after or observed < updated:
        raise DashboardValidationError("dashboard freshness arithmetic is invalid")

    status = value.get("status")
    if not isinstance(status, dict):
        raise DashboardValidationError("dashboard status payload is absent")
    models = status.get("models")
    training = status.get("training")
    benchmark = status.get("benchmark")
    hosts = status.get("hosts")
    if (
        status.get("schema_version") != 1
        or status.get("schema_id") != STATUS_SCHEMA_ID
        or status.get("campaign_id") != CAMPAIGN_ID
        or status.get("updated_unix_ms") != updated
        or status.get("phase") != "contracts-ready"
        or status.get("legal_next_transitions") != ["bootstrap-generating"]
        or status.get("round_index") is not None
        or not isinstance(models, dict)
        or models.get("incumbent") is not None
        or models.get("candidate") is not None
        or models.get("opponent_pool") != []
        or not isinstance(training, dict)
        or training.get("active") is not False
        or not isinstance(benchmark, dict)
        or benchmark.get("active") is not False
        or benchmark.get("classification") != "pending"
        or not isinstance(hosts, dict)
        or set(hosts) != {"john1", "john2", "john3"}
    ):
        raise DashboardValidationError("dashboard status is not the revision-0 boundary")
    status_sha256 = hashlib.sha256(canonical_json(status)).hexdigest()
    return {
        "endpoint": endpoint,
        "observed_unix_ms": observed,
        "updated_unix_ms": updated,
        "age_seconds": age,
        "stale_after_seconds": stale_after,
        "status_sha256": status_sha256,
    }


def fetch_response(
    endpoint: str,
    *,
    timeout_seconds: float,
    opener: Callable[..., Response] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        headers={"Accept": "application/json", "Cache-Control": "no-cache"},
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise DashboardValidationError(
                    f"dashboard endpoint returned HTTP {response.status}"
                )
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise DashboardValidationError(f"dashboard endpoint failed: {error}") from error
    if len(payload) > MAX_RESPONSE_BYTES:
        raise DashboardValidationError("dashboard response exceeds the byte ceiling")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DashboardValidationError("dashboard response is not valid JSON") from error
    return validate_response(value, endpoint=endpoint)


def validate_live_dashboard(
    *,
    endpoints: Sequence[str] = DEFAULT_ENDPOINTS,
    sample_count: int = 3,
    interval_seconds: float = 11.0,
    timeout_seconds: float = 5.0,
    fetcher: Callable[..., dict[str, Any]] = fetch_response,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if tuple(endpoints) != DEFAULT_ENDPOINTS:
        raise DashboardValidationError("W5 requires the exact local and Tailscale endpoints")
    if sample_count < 2 or sample_count > 4:
        raise DashboardValidationError("W5 requires two to four dashboard samples")
    if not math.isfinite(interval_seconds) or interval_seconds < 10.0:
        raise DashboardValidationError("samples must span complete publisher intervals")
    if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 10.0:
        raise DashboardValidationError("dashboard timeout is outside the safe bound")

    intervals: list[dict[str, Any]] = []
    previous_updated: int | None = None
    for sample_index in range(sample_count):
        samples = [
            fetcher(endpoint, timeout_seconds=timeout_seconds) for endpoint in endpoints
        ]
        identities = {(item["updated_unix_ms"], item["status_sha256"]) for item in samples}
        if len(identities) != 1:
            raise DashboardValidationError("local and Tailscale status payloads differ")
        updated = samples[0]["updated_unix_ms"]
        if previous_updated is not None and updated <= previous_updated:
            raise DashboardValidationError("dashboard publisher did not advance")
        previous_updated = updated
        intervals.append({"sample_index": sample_index, "endpoints": samples})
        if sample_index + 1 < sample_count:
            sleeper(interval_seconds)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "endpoints": list(endpoints),
        "sample_count": sample_count,
        "interval_seconds": interval_seconds,
        "phase": "contracts-ready",
        "revision_zero_observed": True,
        "single_projection_observed": True,
        "writer_process_count_verified_by_this_tool": False,
        "john4_used": False,
        "protected_seed_values_opened": False,
        "samples": intervals,
    }
    summary["summary_sha256"] = document_sha256(summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--samples", type=int, default=3)
    result.add_argument("--interval-seconds", type=float, default=11.0)
    result.add_argument("--timeout-seconds", type=float, default=5.0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        summary = validate_live_dashboard(
            sample_count=arguments.samples,
            interval_seconds=arguments.interval_seconds,
            timeout_seconds=arguments.timeout_seconds,
        )
    except DashboardValidationError as error:
        print(f"R2-MAP W5 live dashboard validation refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
