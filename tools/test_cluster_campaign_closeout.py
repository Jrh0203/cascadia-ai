from __future__ import annotations

import cluster_campaign_closeout as closeout
import cluster_research_queue as queue


def task_spec(
    task_id: str,
    *,
    host: str,
    dependencies: list[str] | None = None,
    decision_terminal: bool = False,
) -> dict[str, object]:
    return {
        "id": task_id,
        "title": task_id,
        "experiment_id": "experiment-v1",
        "decision": task_id,
        "workload_class": "independent-experiment",
        "priority": 0,
        "decision_value": 1.0,
        "expected_runtime_seconds": 10,
        "critical_path": True,
        "decision_terminal": decision_terminal,
        "compatible_hosts": [host],
        "dependencies": dependencies or [],
        "command": ["true"],
        "artifact_path": f"{task_id}.json",
        "stop_rule": "complete",
        "resources": {"cpu_cores": 1, "memory_gib": 1.0, "uses_mlx": False},
    }


def telemetry_sample(timestamp: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp_unix_ms": timestamp,
        "nodes": [
            {
                "node_id": host,
                "reachable": True,
                "cpu_percent": 50.0,
                "memory_percent": 25.0,
            }
            for host in ("john1", "john2", "john3", "john4")
        ],
    }


def completed_campaign() -> dict[str, object]:
    state = queue.empty_queue("campaign", now_ms=0)
    queue.add_task(
        state,
        task_spec("origin", host="john1"),
        now_ms=0,
    )
    queue.add_task(
        state,
        task_spec(
            "classification",
            host="john2",
            dependencies=["origin"],
            decision_terminal=True,
        ),
        now_ms=0,
    )
    origin = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=1_000)
    queue.finish_task(
        state,
        task_id="origin",
        host="john1",
        token=origin["claim"]["token"],
        outcome="completed",
        now_ms=5_000,
    )
    classification = queue.claim_next(
        state,
        host="john2",
        lease_seconds=30,
        now_ms=6_000,
    )
    queue.finish_task(
        state,
        task_id="classification",
        host="john2",
        token=classification["claim"]["token"],
        outcome="completed",
        now_ms=8_000,
    )
    return state


def test_interval_helpers_merge_and_subtract_exactly() -> None:
    assert closeout.merge_intervals([(0, 5), (5, 8), (10, 12)]) == [
        (0, 8),
        (10, 12),
    ]
    assert closeout.subtract_intervals([(0, 10)], [(2, 4), (6, 12)]) == [
        (0, 2),
        (4, 6),
    ]


def test_ready_intervals_respect_dependencies_and_claims() -> None:
    state = completed_campaign()
    tasks = {task["id"]: task for task in state["tasks"]}
    assert closeout.task_ready_intervals(
        tasks["origin"],
        tasks=tasks,
        start_unix_ms=0,
        end_unix_ms=10_000,
    ) == [(0, 1_000)]
    assert closeout.task_ready_intervals(
        tasks["classification"],
        tasks=tasks,
        start_unix_ms=0,
        end_unix_ms=10_000,
    ) == [(5_000, 6_000)]


def test_campaign_audit_reports_idle_duplicate_and_decision_rates() -> None:
    state = completed_campaign()
    samples = [telemetry_sample(timestamp) for timestamp in (0, 1_000, 5_000, 6_000)]
    report = closeout.audit_campaign(
        state,
        samples,
        start_unix_ms=0,
        end_unix_ms=10_000,
        cores={host: 10 for host in ("john1", "john2", "john3", "john4")},
    )
    assert report["decisions_completed"] == 1
    assert report["scheduled_process_seconds"] == 6.0
    assert report["duplicate_compute_fraction"] == 0.0
    assert report["hosts"]["john1"]["productive_seconds"] == 4.0
    assert report["hosts"]["john2"]["productive_seconds"] == 2.0
    assert report["hosts"]["john2"]["idle_with_compatible_work_queued_seconds"] == 1.0
    assert report["healthy_idle_with_compatible_work_queued_seconds"] >= 1.0
    assert report["telemetry"]["mean_core_weighted_cpu_percent"] == 50.0
