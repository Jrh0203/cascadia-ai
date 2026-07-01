"""Small v1.9 Bacalhau REST adapter isolated from the public Cascadia API."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import BacalhauAPIError

# Bacalhau v1.9's REST JSON encoder emits execution state enums as integers,
# while the job state endpoint emits names. Normalize that transport detail at
# the adapter boundary so callers never have to mix the two representations.
_EXECUTION_STATE_NAMES = {
    0: "Undefined",
    1: "New",
    2: "AskForBid",
    3: "AskForBidAccepted",
    4: "AskForBidRejected",
    5: "BidAccepted",
    6: "Running",
    7: "Publishing",
    8: "BidRejected",
    9: "Completed",
    10: "Failed",
    11: "Cancelled",
}

_UNCHANGED_JOB_ID = re.compile(r"\bJob Id: '([^']+)'")


def _normalize_execution(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    state = value.get("ComputeState")
    if isinstance(state, Mapping):
        normalized_state = dict(state)
        raw = normalized_state.get("StateType")
        if isinstance(raw, int) and not isinstance(raw, bool):
            try:
                normalized_state["StateType"] = _EXECUTION_STATE_NAMES[raw]
            except KeyError as error:
                raise BacalhauAPIError(
                    f"Bacalhau execution has unknown v1.9 state enum {raw}"
                ) from error
        normalized["ComputeState"] = normalized_state
    return normalized


@dataclass(frozen=True)
class BacalhauAPI:
    endpoint: str
    request_timeout_seconds: float = 15.0
    list_jobs_timeout_seconds: float = 60.0
    maximum_attempts: int = 4
    initial_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        endpoint = self.endpoint.rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise BacalhauAPIError("Bacalhau endpoint must be HTTP(S)")
        if (
            self.request_timeout_seconds <= 0
            or self.list_jobs_timeout_seconds <= 0
            or self.maximum_attempts < 1
            or self.initial_backoff_seconds <= 0
        ):
            raise BacalhauAPIError("Bacalhau retry bounds are invalid")
        object.__setattr__(self, "endpoint", endpoint)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        query_string = urllib.parse.urlencode(query or (), doseq=True)
        url = f"{self.endpoint}{path}" + (f"?{query_string}" if query_string else "")
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        raw = b""
        last_error: Exception | None = None
        last_error_detail = ""
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=(
                        self.request_timeout_seconds if timeout_seconds is None else timeout_seconds
                    ),
                ) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                last_error = error
                last_error_detail = error.read().decode(errors="replace")
                if method == "PUT" and path == "/api/v1/orchestrator/jobs" and error.code == 500:
                    # Bacalhau v1.9 reports an idempotent replay as HTTP 500
                    # "no changes detected" but includes the already-created
                    # Job ID. Normalize that transport quirk into the same
                    # contract as a successful idempotent submission.
                    try:
                        error_value = json.loads(last_error_detail)
                    except json.JSONDecodeError:
                        error_value = {}
                    message = error_value.get("Message")
                    unchanged = (
                        _UNCHANGED_JOB_ID.search(message)
                        if isinstance(message, str)
                        and message.startswith("no changes detected for new job spec.")
                        else None
                    )
                    if unchanged is not None:
                        return {"JobID": unchanged.group(1), "Recovered": True}
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise BacalhauAPIError(
                        f"Bacalhau {method} {path} failed: {error}; {last_error_detail}"
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                last_error_detail = ""
            else:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as error:
                    # A successful HTTP response can still be truncated or empty
                    # while the orchestrator is under load. Treat that transport
                    # failure exactly like the retryable network failures above.
                    last_error = error
                else:
                    if not isinstance(value, dict):
                        raise BacalhauAPIError(
                            f"Bacalhau {method} {path} response is not an object"
                        )
                    return value
            if attempt < self.maximum_attempts:
                time.sleep(self.initial_backoff_seconds * (2 ** (attempt - 1)))
        if isinstance(last_error, json.JSONDecodeError):
            raise BacalhauAPIError(
                f"Bacalhau {method} {path} returned invalid JSON after "
                f"{self.maximum_attempts} attempts ({len(raw)} bytes in final response)"
            ) from last_error
        if last_error is not None:
            raise BacalhauAPIError(
                f"Bacalhau {method} {path} failed after {self.maximum_attempts} attempts: "
                f"{last_error}; {last_error_detail}"
            ) from last_error
        raise AssertionError("Bacalhau request loop ended without a response or error")

    def alive(self) -> bool:
        return self._request("GET", "/api/v1/agent/alive").get("Status") == "OK"

    def submit(self, job: Mapping[str, Any], *, idempotency_token: str) -> dict[str, Any]:
        return self._request(
            "PUT",
            "/api/v1/orchestrator/jobs",
            payload={"Job": dict(job), "Force": False, "idempotencyToken": idempotency_token},
        )

    def list_jobs(self, *, labels: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        query = [("labels", f"{key}={value}") for key, value in sorted((labels or {}).items())]
        # Bacalhau's v1.9 list endpoint can legitimately take tens of seconds
        # once the campaign ledger contains thousands of completed jobs, even
        # for a label-filtered query. Recovery must not mistake that scheduler
        # latency for a failed request; latency-sensitive health and per-job
        # calls retain the tighter default timeout.
        value = self._request(
            "GET",
            "/api/v1/orchestrator/jobs",
            query=query,
            timeout_seconds=self.list_jobs_timeout_seconds,
        )
        items = value.get("Items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise BacalhauAPIError("Bacalhau jobs response has invalid Items")
        return items

    def get_job(self, job_id: str, *, include: str = "executions") -> dict[str, Any]:
        value = self._request(
            "GET",
            f"/api/v1/orchestrator/jobs/{urllib.parse.quote(job_id, safe='')}",
            query={"include": include, "limit": 100},
        )
        if not isinstance(value.get("Job"), dict):
            raise BacalhauAPIError("Bacalhau get-job response omits Job")
        return value

    def executions(self, job_id: str) -> list[dict[str, Any]]:
        value = self._request(
            "GET",
            f"/api/v1/orchestrator/jobs/{urllib.parse.quote(job_id, safe='')}/executions",
            query={"limit": 100},
        )
        items = value.get("Items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise BacalhauAPIError("Bacalhau executions response has invalid Items")
        return [_normalize_execution(item) for item in items]

    def results(self, job_id: str) -> list[dict[str, Any]]:
        value = self._request(
            "GET", f"/api/v1/orchestrator/jobs/{urllib.parse.quote(job_id, safe='')}/results"
        )
        items = value.get("Items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise BacalhauAPIError("Bacalhau results response has invalid Items")
        return items

    def stop(self, job_id: str, *, reason: str) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/api/v1/orchestrator/jobs/{urllib.parse.quote(job_id, safe='')}",
            query={"reason": reason},
        )

    def nodes(self) -> list[dict[str, Any]]:
        value = self._request(
            "GET",
            "/api/v1/orchestrator/nodes",
            query={"limit": 100},
        )
        items = value.get("Nodes")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise BacalhauAPIError("Bacalhau nodes response has invalid Nodes")
        return items
