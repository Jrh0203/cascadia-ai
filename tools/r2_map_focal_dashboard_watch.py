#!/usr/bin/env python3
"""Publish topology-free paired-gate progress without reading blinded scores."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CAMPAIGN_ROOT = Path("/Users/johnherrick/cascadia-bench/r2-map-v1")
STATUS_PATH = CAMPAIGN_ROOT / "control/dashboard-status.json"
LOCK_PATH = CAMPAIGN_ROOT / "control/.dashboard-status.lock"
MAX_RSS_BYTES = 4 * 1024 * 1024 * 1024
TERMINAL_STALE_AFTER_SECONDS = 60 * 60
QUALIFIED_EXACT_NNUE_WEIGHTS_BLAKE3 = (
    "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"
)
STAGES = {
    "smoke": ("strength-blinded-smoke", 20, "r2-map-strength-blinded-smoke"),
    "development": ("development", 250, "r2-map-fixed-250-comparison"),
}


class FocalDashboardError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FocalDashboardError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise FocalDashboardError(f"JSON artifact is not an object: {path}")
    return value


def validate_contract(root: Path, stage: str) -> dict[str, Any]:
    contract = _read_json(root / "contract.json")
    expected_stage, expected_pairs, _ = STAGES[stage]
    if (
        contract.get("schema_id") != "cascadia.r2-map.focal-contract.v4"
        or contract.get("stage") != expected_stage
        or contract.get("pair_count") != expected_pairs
        or contract.get("execution_partition") != {"kind": "scheduler-managed-pairs"}
    ):
        raise FocalDashboardError("focal contract differs from the registered stage")
    return contract


def expected_work_items(stage: str) -> tuple[str, ...]:
    return tuple(f"pair-{pair_index:04}" for pair_index in range(STAGES[stage][1]))


def local_work_item_completed(root: Path, work_item: str) -> int:
    """Count only an atomic pair already imported into John1 authority."""

    receipts = list((root / "receipts" / work_item).glob("pair-*.json"))
    return int(len(receipts) == 1)


def request_observation(
    request_id: str | None,
    *,
    state_directory: Path,
    endpoint: str,
) -> dict[str, dict[str, Any]]:
    """Project Bacalhau lifecycle state without assigning work to a machine."""

    if request_id is None:
        return {}
    request_path = state_directory / "requests" / f"{request_id}.json"
    state = _read_json(request_path)
    if state.get("schema_id") not in {
        "cascadia.cluster.request-state.v1",
        "cascadia.cluster.managed-request-state.v2",
    }:
        raise FocalDashboardError("cluster request state has an unsupported schema")
    try:
        from cascadia_cluster.bacalhau_api import BacalhauAPI
        from cascadia_cluster.models import canonical_sha256
    except ImportError as error:
        raise FocalDashboardError(f"cannot load Bacalhau client: {error}") from error
    state_payload = dict(state)
    claimed_hash = state_payload.pop("state_sha256", None)
    if claimed_hash != canonical_sha256(state_payload):
        raise FocalDashboardError("cluster request state checksum differs")
    api = BacalhauAPI(endpoint)
    observations: dict[str, dict[str, Any]] = {}
    expected = set(expected_work_items(_stage_from_request_size(state)))
    for item in state.get("items", []):
        if not isinstance(item, dict) or item.get("key") not in expected:
            raise FocalDashboardError("focal request contains an unregistered pair work item")
        work_item = item["key"]
        job_id = item.get("bacalhau_job_id")
        if job_id is None and state.get("schema_id") == "cascadia.cluster.managed-request-state.v2":
            observations[work_item] = {
                "job_id": None,
                "state": "pending_admission",
                "message": "waiting for scheduler-capacity admission",
                "attempts": 0,
            }
            continue
        if not isinstance(job_id, str) or not job_id:
            raise FocalDashboardError(f"cluster request omitted job id for {work_item}")
        job = api.get_job(job_id)["Job"]
        job_state = job.get("State") if isinstance(job.get("State"), dict) else {}
        executions = api.executions(job_id)
        attempts = sum(
            str((execution.get("ComputeState") or {}).get("StateType"))
            in {"Completed", "Failed", "Cancelled"}
            for execution in executions
        )
        observations[work_item] = {
            "job_id": job_id,
            "state": str(job_state.get("StateType", "Unknown")).lower(),
            "message": str(job_state.get("Message", "")),
            "attempts": attempts,
        }
    if set(observations) != expected:
        raise FocalDashboardError("focal request must cover every registered pair work item")
    return observations


def scheduler_utilization_observation(endpoint: str) -> dict[str, float | int]:
    """Return a one-sample topology-free utilization projection for the UI."""

    try:
        from cascadia_cluster.bacalhau_api import BacalhauAPI
    except ImportError as error:
        raise FocalDashboardError(f"cannot load Bacalhau client: {error}") from error
    nodes = BacalhauAPI(endpoint).nodes()
    observed: dict[str, tuple[float, float]] = {}
    for node in nodes:
        info = node.get("Info") if isinstance(node, dict) else None
        labels = info.get("Labels") if isinstance(info, dict) else None
        compute = info.get("ComputeNodeInfo") if isinstance(info, dict) else None
        maximum = compute.get("MaxCapacity") if isinstance(compute, dict) else None
        available = compute.get("AvailableCapacity") if isinstance(compute, dict) else None
        name = labels.get("cascadia_internal_node") if isinstance(labels, dict) else None
        capacity = maximum.get("CPU") if isinstance(maximum, dict) else None
        free = available.get("CPU", 0) if isinstance(available, dict) else None
        if name not in {"john1", "john2", "john3"}:
            continue
        if (
            node.get("Connection") != "CONNECTED"
            or not isinstance(capacity, (int, float))
            or isinstance(capacity, bool)
            or not isinstance(free, (int, float))
            or isinstance(free, bool)
            or not 0 <= free <= capacity
        ):
            raise FocalDashboardError("Bacalhau scheduler utilization is malformed")
        observed[name] = (float(capacity), float(capacity - free))
    if set(observed) != {"john1", "john2", "john3"}:
        raise FocalDashboardError("Bacalhau scheduler utilization lacks the active fabric")
    capacity = sum(value[0] for value in observed.values())
    allocated = sum(value[1] for value in observed.values())
    utilization = allocated / capacity if capacity else 0.0
    return {
        "sample_count": 1,
        "observed_seconds": 0.0,
        "cpu_capacity_min": capacity,
        "cpu_capacity_max": capacity,
        "cpu_allocated_mean": allocated,
        "cpu_allocated_peak": allocated,
        "cpu_utilization_mean": utilization,
        "cpu_utilization_peak": utilization,
    }


def _stage_from_request_size(state: dict[str, Any]) -> str:
    items = state.get("items")
    if not isinstance(items, list):
        raise FocalDashboardError("cluster request state omitted its work items")
    matches = [stage for stage in STAGES if len(items) == STAGES[stage][1]]
    if len(matches) != 1:
        raise FocalDashboardError("cluster request size does not match a registered stage")
    return matches[0]


def _completed_report(
    root: Path, stage: str
) -> tuple[dict[str, Any] | None, bool | None, dict[str, Any] | None]:
    projection_path = root / "projections/dashboard-benchmark.json"
    report_path = root / "reports/focal-benchmark.json"
    if not projection_path.is_file() or not report_path.is_file():
        return None, None, None
    projection = _read_json(projection_path)
    report = _read_json(report_path)
    if stage == "smoke":
        result = report.get("result")
        statistics = result.get("statistics") if isinstance(result, dict) else None
        if (
            not isinstance(statistics, dict)
            or result.get("kind") != "strength-blinded-smoke"
            or statistics.get("strength_outputs_blinded") is not True
        ):
            raise FocalDashboardError("smoke report does not preserve strength blinding")
        swap_delta = statistics.get("maximum_swap_delta_bytes")
        passed = all(
            (
                statistics.get("pairs") == 20,
                statistics.get("physical_games") == 40,
                statistics.get("all_clean_shutdowns") is True,
                statistics.get("all_pinecone_conservation_checks_passed") is True,
                isinstance(statistics.get("peak_rss_bytes"), int)
                and statistics["peak_rss_bytes"] <= MAX_RSS_BYTES,
                isinstance(swap_delta, int) and swap_delta <= 0,
            )
        )
        if projection.get("focal") is not None or projection.get("paired_delta") is not None:
            raise FocalDashboardError("smoke dashboard projection exposed strength")
        return projection, passed, _completed_scheduler_status(root, stage)
    result = report.get("result")
    if not isinstance(result, dict) or result.get("kind") != "development":
        raise FocalDashboardError("fixed-250 report has the wrong result kind")
    return projection, True, _completed_scheduler_status(root, stage)


def _completed_scheduler_status(root: Path, stage: str) -> dict[str, Any]:
    path = root / "reports/scheduler-provenance.json"
    report = _read_json(path)
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    observed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    total = STAGES[stage][1]
    work_items = report.get("work_items")
    utilization = report.get("scheduler_utilization")
    required_utilization = {
        "sample_count",
        "observed_seconds",
        "cpu_capacity_min",
        "cpu_capacity_max",
        "cpu_allocated_mean",
        "cpu_allocated_peak",
        "cpu_utilization_mean",
        "cpu_utilization_peak",
        "nodes",
    }
    if (
        report.get("schema_id") != "cascadia.r2-map.scheduler-provenance.v1"
        or report.get("stage") != stage
        or claimed != observed
        or not isinstance(work_items, list)
        or len(work_items) != total
        or not isinstance(report.get("retry_count"), int)
        or report["retry_count"] < 0
        or not isinstance(utilization, dict)
        or set(utilization) != required_utilization
    ):
        raise FocalDashboardError("completed scheduler provenance is malformed")
    numeric = [
        utilization["observed_seconds"],
        utilization["cpu_capacity_min"],
        utilization["cpu_capacity_max"],
        utilization["cpu_allocated_mean"],
        utilization["cpu_allocated_peak"],
        utilization["cpu_utilization_mean"],
        utilization["cpu_utilization_peak"],
    ]
    if (
        not isinstance(utilization["sample_count"], int)
        or utilization["sample_count"] <= 0
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) < float("inf")
            for value in numeric
        )
        or utilization["cpu_capacity_min"] > utilization["cpu_capacity_max"]
        or utilization["cpu_allocated_mean"] > utilization["cpu_allocated_peak"]
        or utilization["cpu_allocated_peak"] > utilization["cpu_capacity_max"]
        or utilization["cpu_utilization_mean"] > utilization["cpu_utilization_peak"]
        or utilization["cpu_utilization_peak"] > 1.0
    ):
        raise FocalDashboardError("completed scheduler utilization is malformed")
    return {
        "completed": total,
        "total": total,
        "states": {"succeeded": total},
        "retry_attempts": report["retry_count"],
        "utilization": {
            key: utilization[key]
            for key in (
                "sample_count",
                "observed_seconds",
                "cpu_capacity_min",
                "cpu_capacity_max",
                "cpu_allocated_mean",
                "cpu_allocated_peak",
                "cpu_utilization_mean",
                "cpu_utilization_peak",
            )
        },
    }


def build_status(
    status: dict[str, Any],
    *,
    root: Path,
    stage: str,
    counts: dict[str, int],
    cluster_observation: dict[str, dict[str, Any]] | None = None,
    scheduler_utilization: dict[str, float | int] | None = None,
    now_ms: int,
    started_ms: int | None = None,
) -> dict[str, Any]:
    contract = validate_contract(root, stage)
    expected_stage, total_pairs, phase = STAGES[stage]
    work_items = expected_work_items(stage)
    if set(counts) != set(work_items) or any(
        not isinstance(counts[item], int) or counts[item] not in (0, 1)
        for item in work_items
    ):
        raise FocalDashboardError("observed pair work-item coverage is malformed")
    completed = sum(counts.values())
    elapsed_seconds = (
        max(0.001, (now_ms - started_ms) / 1000.0)
        if started_ms is not None and completed > 0
        else None
    )
    throughput = 2.0 * completed / elapsed_seconds if elapsed_seconds is not None else None
    eta_seconds = (
        round(2.0 * (total_pairs - completed) / throughput)
        if throughput is not None and throughput > 0.0
        else None
    )
    projection, report_passed, completed_scheduler = _completed_report(root, stage)
    report_complete = projection is not None
    if report_complete and completed != total_pairs:
        raise FocalDashboardError("complete focal report exists before all receipts")

    benchmark = dict(status.get("benchmark") or {})
    if projection is not None:
        benchmark.update(projection)
    else:
        benchmark.update(
            {
                "active": completed < total_pairs,
                "stage": expected_stage,
                "pairs_completed": completed,
                "pairs_total": total_pairs,
                "eta_seconds": eta_seconds,
                "throughput_games_per_second": throughput,
                "peak_rss_bytes": None,
                "swap_delta_bytes": None,
                "focal": None,
                "paired_delta": None,
                "classification": "pending",
            }
        )
    status["benchmark"] = benchmark
    observations = cluster_observation or {}
    state_counts: dict[str, int] = {}
    for observation in observations.values():
        state_name = str(observation.get("state", "unknown"))
        state_counts[state_name] = state_counts.get(state_name, 0) + 1
    scheduler_completed = sum(
        state_counts.get(state, 0) for state in ("completed", "succeeded")
    )
    progress_completed = max(completed, scheduler_completed)
    if projection is None and progress_completed != completed:
        elapsed_seconds = (
            max(0.001, (now_ms - started_ms) / 1000.0)
            if started_ms is not None and progress_completed > 0
            else None
        )
        throughput = (
            2.0 * progress_completed / elapsed_seconds
            if elapsed_seconds is not None
            else None
        )
        eta_seconds = (
            round(2.0 * (total_pairs - progress_completed) / throughput)
            if throughput is not None and throughput > 0.0
            else None
        )
        benchmark.update(
            {
                "pairs_completed": progress_completed,
                "eta_seconds": eta_seconds,
                "throughput_games_per_second": throughput,
            }
        )
    benchmark["scheduler_work_items"] = completed_scheduler or {
        "completed": progress_completed,
        "total": total_pairs,
        "states": dict(sorted(state_counts.items())),
        "retry_attempts": sum(
            max(0, int(observation.get("attempts", 0)) - 1)
            for observation in observations.values()
        ),
        "utilization": scheduler_utilization,
    }
    status["phase"] = f"{phase}-complete" if report_complete else phase
    latest = status.get("training", {}).get("latest_verified_checkpoint")
    candidate_blake3 = (
        latest.get("blake3")
        if isinstance(latest, dict)
        and latest.get("id") == contract["candidate_checkpoint_id"]
        else None
    )
    status["models"]["candidate"] = {
        "id": contract["candidate_checkpoint_id"],
        "blake3": candidate_blake3,
    }
    status["models"]["incumbent"] = {
        "id": contract["control_checkpoint_id"],
        "blake3": QUALIFIED_EXACT_NNUE_WEIGHTS_BLAKE3,
    }
    status["training"]["active"] = False
    for name in ("john1", "john2", "john3"):
        host = status["hosts"][name]
        if report_complete:
            host["intent"] = "control" if name == "john1" else "idle"
            host["detail"] = (
                f"role={'control' if name == 'john1' else 'execution'}; "
                f"phase={phase}-complete; placement=bacalhau-managed"
            )
        else:
            host["intent"] = "benchmark"
            host["detail"] = (
                f"role=scheduler-worker; phase={phase}; "
                "placement=bacalhau-managed; no-pair-affinity"
            )
        # Legacy per-host counters are required integers by the serving schema.
        # Keep them at zero with no total so they cannot imply pair ownership;
        # aggregate scheduler progress is reported above.
        host["benchmark_pairs_completed"] = 0
        host["benchmark_pairs_total"] = None
        host["eta_seconds"] = None
        host["throughput_games_per_second"] = None
        host["rss_bytes"] = None
        host["swap_delta_bytes"] = None
    if not report_complete:
        status["legal_next_transitions"] = [f"monitor-{expected_stage}"]
    elif stage == "smoke" and report_passed:
        status["legal_next_transitions"] = ["materialize-fixed-250-protected-domain"]
    elif stage == "smoke":
        status["legal_next_transitions"] = ["stop-invalid-cross-architecture-smoke"]
        status["benchmark"]["classification"] = "invalid"
    else:
        # The fixed-250 projection is itself the terminal published result.
        # No expert iteration is authorized by this campaign.
        status["legal_next_transitions"] = []
    status["updated_unix_ms"] = now_ms
    # A completed report is immutable campaign evidence, not a live heartbeat.
    # Keep the final projection durable; fleet reachability and scheduler health
    # continue to refresh independently through /api/v1/cluster.
    status["stale_after_seconds"] = (
        TERMINAL_STALE_AFTER_SECONDS if report_complete else 30
    )
    return status


def _write_status(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o600)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _watch_started_ms(root: Path, stage: str, now_ms: int) -> int:
    path = root / ".dashboard-watch-start.json"
    if path.is_file():
        state = _read_json(path)
        started_ms = state.get("started_unix_ms")
        if state.get("stage") != stage or not isinstance(started_ms, int) or started_ms > now_ms:
            raise FocalDashboardError("dashboard watch start state is invalid")
        return started_ms
    value = {
        "schema_id": "cascadia.r2-map.dashboard-watch-start.v1",
        "stage": stage,
        "started_unix_ms": now_ms,
    }
    _write_status(path, value)
    return now_ms


def publish_once(
    benchmark_root: Path,
    stage: str,
    *,
    request_id: str | None = None,
    state_directory: Path = Path("artifacts/cluster/fabric-state"),
    endpoint: str = "http://100.110.109.6:1234",
) -> dict[str, Any]:
    with LOCK_PATH.open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = _read_json(STATUS_PATH)
        counts = {
            work_item: local_work_item_completed(benchmark_root, work_item)
            for work_item in expected_work_items(stage)
        }
        now_ms = int(time.time() * 1000)
        report_complete = (
            (benchmark_root / "projections/dashboard-benchmark.json").is_file()
            and (benchmark_root / "reports/focal-benchmark.json").is_file()
        )
        updated = build_status(
            status,
            root=benchmark_root,
            stage=stage,
            counts=counts,
            cluster_observation=(
                None
                if report_complete
                else request_observation(
                    request_id, state_directory=state_directory, endpoint=endpoint
                )
            ),
            scheduler_utilization=(
                None if report_complete else scheduler_utilization_observation(endpoint)
            ),
            now_ms=now_ms,
            started_ms=_watch_started_ms(benchmark_root, stage, now_ms),
        )
        _write_status(STATUS_PATH, updated)
        return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--request-id")
    parser.add_argument(
        "--cluster-state-directory",
        type=Path,
        default=Path("artifacts/cluster/fabric-state"),
    )
    parser.add_argument("--bacalhau-endpoint", default="http://100.110.109.6:1234")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=10)
    arguments = parser.parse_args()
    while True:
        try:
            status = publish_once(
                arguments.benchmark_root,
                arguments.stage,
                request_id=arguments.request_id,
                state_directory=arguments.cluster_state_directory,
                endpoint=arguments.bacalhau_endpoint,
            )
            print(json.dumps({"phase": status["phase"], "benchmark": status["benchmark"]}))
        except FocalDashboardError as error:
            print(f"R2-MAP focal dashboard refused: {error}", file=sys.stderr)
            if not arguments.watch:
                return 2
        if not arguments.watch:
            return 0
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
