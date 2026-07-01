from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cluster_research_queue as queue
import pytest


def task_spec(
    task_id: str,
    *,
    dependencies: list[str] | None = None,
    compatible_hosts: list[str] | None = None,
    priority: int = 100,
    critical_path: bool = False,
    decision_value: float = 1.0,
    runtime: float = 100.0,
) -> dict[str, object]:
    return {
        "id": task_id,
        "title": task_id.replace("-", " ").title(),
        "experiment_id": "experiment-v1",
        "decision": f"Decide {task_id}",
        "workload_class": "independent-experiment",
        "priority": priority,
        "decision_value": decision_value,
        "expected_runtime_seconds": runtime,
        "critical_path": critical_path,
        "decision_terminal": False,
        "compatible_hosts": compatible_hosts or ["john1", "john2"],
        "dependencies": dependencies or [],
        "command": ["python3", "-c", "print('ok')"],
        "artifact_path": f"artifacts/{task_id}.json",
        "stop_rule": "Exit after the frozen decision is written.",
        "resources": {"cpu_cores": 1, "memory_gib": 1.0, "uses_mlx": False},
    }


def test_dependencies_claims_and_completion_are_atomic() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("origin"), now_ms=2)
    queue.add_task(state, task_spec("replay", dependencies=["origin"]), now_ms=3)
    assert [task["status"] for task in state["tasks"]] == ["ready", "blocked"]

    claimed = queue.claim_next(state, host="john2", lease_seconds=30, now_ms=10)
    assert claimed is not None
    assert claimed["id"] == "origin"
    assert state["hosts"]["john2"]["intent"] == "working"

    queue.finish_task(
        state,
        task_id="origin",
        host="john2",
        token=claimed["claim"]["token"],
        outcome="completed",
        artifact="artifacts/origin.json",
        now_ms=20,
    )
    assert [task["status"] for task in state["tasks"]] == ["completed", "ready"]
    assert state["hosts"]["john2"]["intent"] == "available"
    queue.validate_queue(state)


def test_add_tasks_installs_forward_dependency_graph_atomically() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    installed = queue.add_tasks(
        state,
        [
            task_spec("consumer", dependencies=["origin"]),
            task_spec("origin"),
        ],
        now_ms=2,
    )
    assert [task["id"] for task in installed] == ["consumer", "origin"]
    assert [task["status"] for task in state["tasks"]] == ["blocked", "ready"]
    assert [event["task_id"] for event in state["events"]] == ["consumer", "origin"]


def test_add_tasks_rejects_duplicate_without_partial_mutation() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("existing"), now_ms=2)
    before = json.loads(json.dumps(state))
    with pytest.raises(queue.QueueError, match="already contains"):
        queue.add_tasks(
            state,
            [task_spec("new"), task_spec("existing")],
            now_ms=3,
        )
    assert state == before


def test_add_tasks_rejects_invalid_graph_without_partial_mutation() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    before = json.loads(json.dumps(state))
    with pytest.raises(queue.QueueError, match="unknown dependencies"):
        queue.add_tasks(
            state,
            [task_spec("consumer", dependencies=["absent"])],
            now_ms=2,
        )
    assert state == before


def test_campaign_task_specifications_checks_envelope_identity() -> None:
    specification = task_spec("task")
    with pytest.raises(queue.QueueError, match="another experiment"):
        queue.campaign_task_specifications(
            {
                "experiment_id": "different-experiment",
                "task_count": 1,
                "tasks": [specification],
            }
        )
    with pytest.raises(queue.QueueError, match="task_count"):
        queue.campaign_task_specifications(
            {
                "experiment_id": "experiment-v1",
                "task_count": 2,
                "tasks": [specification],
            }
        )


def test_selection_prefers_priority_then_critical_path_then_decision_rate() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("slow", decision_value=10, runtime=100), now_ms=2)
    queue.add_task(
        state,
        task_spec("critical", critical_path=True, decision_value=1, runtime=100),
        now_ms=3,
    )
    queue.add_task(
        state,
        task_spec("priority", priority=10, decision_value=0.1, runtime=100),
        now_ms=4,
    )
    assert queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)["id"] == "priority"

    priority = state["tasks"][2]
    queue.finish_task(
        state,
        task_id="priority",
        host="john1",
        token=priority["claim"]["token"],
        outcome="completed",
        now_ms=11,
    )
    assert queue.claim_next(state, host="john1", lease_seconds=30, now_ms=12)["id"] == "critical"


def test_host_compatibility_and_lease_expiry_enable_work_stealing() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(
        state,
        task_spec("remote-only", compatible_hosts=["john3", "john4"]),
        now_ms=2,
    )
    assert queue.claim_next(state, host="john1", lease_seconds=1, now_ms=10) is None
    first = queue.claim_next(state, host="john3", lease_seconds=1, now_ms=10)
    assert first is not None
    assert first["claim"]["host"] == "john3"

    second = queue.claim_next(state, host="john4", lease_seconds=1, now_ms=1_011)
    assert second is not None
    assert second["id"] == "remote-only"
    assert second["claim"]["host"] == "john4"
    assert state["tasks"][0]["attempts"][0]["outcome"] == "lease-expired"
    assert state["hosts"]["john3"]["intent"] == "available"


def test_one_host_cannot_hold_two_running_claims() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("first"), now_ms=2)
    queue.add_task(state, task_spec("second"), now_ms=3)
    first = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)
    assert first is not None
    assert queue.claim_next(state, host="john1", lease_seconds=30, now_ms=11) is None
    assert state["tasks"][1]["status"] == "ready"


def test_stale_claim_token_is_rejected() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("task"), now_ms=2)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)
    with pytest.raises(queue.QueueError, match="does not match"):
        queue.finish_task(
            state,
            task_id="task",
            host="john1",
            token="wrong",
            outcome="completed",
            now_ms=20,
        )
    assert claimed is not None
    assert state["tasks"][0]["status"] == "running"


def test_validation_rejects_multiple_running_claims_on_one_host() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("first"), now_ms=2)
    queue.add_task(state, task_spec("second"), now_ms=3)
    first = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)
    second = state["tasks"][1]
    second["status"] = "running"
    second["claim"] = {
        **first["claim"],
        "token": "different",
        "claimed_unix_ms": 11,
        "heartbeat_unix_ms": 11,
        "lease_expires_unix_ms": 30_011,
    }
    with pytest.raises(queue.QueueError, match="multiple running claims"):
        queue.validate_queue(state)


def test_failed_task_can_retry_without_losing_attempt_history() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("task"), now_ms=2)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)
    queue.finish_task(
        state,
        task_id="task",
        host="john1",
        token=claimed["claim"]["token"],
        outcome="failed",
        error="transient",
        retry=True,
        now_ms=20,
    )
    assert state["tasks"][0]["status"] == "ready"
    assert state["tasks"][0]["attempts"][0]["error"] == "transient"
    assert state["tasks"][0]["result"] is None


def test_manually_retry_failed_task_preserves_failed_attempt() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("task"), now_ms=2)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=3)
    queue.finish_task(
        state,
        task_id="task",
        host="john1",
        token=claimed["claim"]["token"],
        outcome="failed",
        error="artifact was not collected",
        now_ms=4,
    )
    retried = queue.retry_failed_task(state, task_id="task", now_ms=5)
    assert retried["status"] == "ready"
    assert retried["result"] is None
    assert retried["attempts"][0]["outcome"] == "failed"


def test_administrative_cancellation_is_atomic_and_preserves_evidence() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("origin", priority=0), now_ms=2)
    queue.add_task(
        state,
        task_spec("dependent", dependencies=["origin"]),
        now_ms=3,
    )
    queue.add_task(state, task_spec("failed"), now_ms=4)
    failed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=5)
    assert failed is not None
    assert failed["id"] == "origin"
    queue.finish_task(
        state,
        task_id="origin",
        host="john1",
        token=failed["claim"]["token"],
        outcome="failed",
        error="superseded source tree",
        now_ms=6,
    )

    cancelled = queue.cancel_pending_tasks(
        state,
        task_ids=["origin", "dependent", "failed"],
        actor="research-coordinator",
        reason="Superseded by the immutable source-frozen campaign.",
        now_ms=7,
    )

    assert [task["status"] for task in cancelled] == [
        "cancelled",
        "cancelled",
        "cancelled",
    ]
    assert state["tasks"][0]["attempts"][0]["outcome"] == "failed"
    assert state["tasks"][0]["result"]["error"] == "superseded source tree"
    assert state["tasks"][0]["administrative_cancellation"] == {
        "actor": "research-coordinator",
        "reason": "Superseded by the immutable source-frozen campaign.",
        "cancelled_unix_ms": 7,
        "previous_status": "failed",
    }
    assert state["tasks"][1]["administrative_cancellation"]["previous_status"] == ("blocked")
    assert state["tasks"][2]["administrative_cancellation"]["previous_status"] == ("ready")
    assert [event["event"] for event in state["events"][-3:]] == [
        "administratively-cancelled",
        "administratively-cancelled",
        "administratively-cancelled",
    ]
    queue.refresh_dependencies(state)
    assert all(task["status"] == "cancelled" for task in state["tasks"])
    queue.validate_queue(state)


def test_administrative_cancellation_rejects_mixed_terminal_batch_atomically() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("completed"), now_ms=2)
    queue.add_task(state, task_spec("ready"), now_ms=3)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=4)
    assert claimed is not None
    queue.finish_task(
        state,
        task_id="completed",
        host="john1",
        token=claimed["claim"]["token"],
        outcome="completed",
        now_ms=5,
    )

    with pytest.raises(queue.QueueError, match="only unclaimed"):
        queue.cancel_pending_tasks(
            state,
            task_ids=["ready", "completed"],
            actor="research-coordinator",
            reason="Superseded.",
            now_ms=6,
        )

    assert state["tasks"][0]["status"] == "completed"
    assert state["tasks"][1]["status"] == "ready"
    assert state["tasks"][1].get("administrative_cancellation") is None


def test_completed_external_task_time_can_be_corrected_with_evidence() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("task"), now_ms=2)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)
    queue.heartbeat(
        state,
        task_id="task",
        host="john1",
        token=claimed["claim"]["token"],
        lease_seconds=30,
        now_ms=15,
    )
    queue.finish_task(
        state,
        task_id="task",
        host="john1",
        token=claimed["claim"]["token"],
        outcome="completed",
        now_ms=30,
    )
    corrected = queue.correct_completion_time(
        state,
        task_id="task",
        completed_unix_ms=20,
        evidence="artifact elapsed time",
        now_ms=40,
    )
    assert corrected["attempts"][0]["ended_unix_ms"] == 20
    assert corrected["result"]["completed_unix_ms"] == 20
    assert state["events"][-1]["detail"]["old_completed_unix_ms"] == 30
    queue.validate_queue(state)


def test_completion_time_correction_is_strictly_bounded() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("task"), now_ms=2)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=10)
    queue.finish_task(
        state,
        task_id="task",
        host="john1",
        token=claimed["claim"]["token"],
        outcome="completed",
        now_ms=20,
    )
    with pytest.raises(queue.QueueError, match="between the last heartbeat"):
        queue.correct_completion_time(
            state,
            task_id="task",
            completed_unix_ms=9,
            evidence="invalid",
            now_ms=30,
        )
    with pytest.raises(queue.QueueError, match="requires evidence"):
        queue.correct_completion_time(
            state,
            task_id="task",
            completed_unix_ms=15,
            evidence=" ",
            now_ms=30,
        )


def test_intentionally_idle_host_requires_a_reason() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    with pytest.raises(queue.QueueError, match="requires a reason"):
        queue.set_host_intent(
            state,
            host="john4",
            intent="intentionally-idle",
            reason=None,
            now_ms=2,
        )
    queue.set_host_intent(
        state,
        host="john4",
        intent="intentionally-idle",
        reason="No compatible work before the checkpoint dependency completes.",
        now_ms=3,
    )
    assert state["hosts"]["john4"]["intent"] == "intentionally-idle"


def test_dependencies_can_be_rewired_before_launch() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("origin"), now_ms=2)
    queue.add_task(state, task_spec("retrieve", dependencies=["origin"]), now_ms=3)
    queue.add_task(state, task_spec("replay", dependencies=["origin"]), now_ms=4)
    updated = queue.set_task_dependencies(
        state,
        task_id="replay",
        dependencies=["retrieve"],
        now_ms=5,
    )
    assert updated["dependencies"] == ["retrieve"]
    assert updated["status"] == "blocked"
    queue.validate_queue(state)


def test_command_vectors_may_repeat_flags() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    specification = task_spec("fanout")
    specification["command"] = [
        "python3",
        "fanout.py",
        "--destination",
        "john3:/tmp/run/",
        "--destination",
        "john4:/tmp/run/",
    ]
    queue.add_task(state, specification, now_ms=2)
    queue.validate_queue(state)


def test_dependencies_cannot_change_after_claim() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("task"), now_ms=2)
    queue.claim_next(state, host="john1", lease_seconds=30, now_ms=3)
    with pytest.raises(queue.QueueError, match="before a task starts"):
        queue.set_task_dependencies(
            state,
            task_id="task",
            dependencies=[],
            now_ms=4,
        )


def test_dispatchable_hosts_exclude_busy_and_incompatible_hosts() -> None:
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(
        state,
        task_spec("john1-only", compatible_hosts=["john1"]),
        now_ms=2,
    )
    queue.add_task(
        state,
        task_spec("john3-only", compatible_hosts=["john3"]),
        now_ms=3,
    )
    queue.claim_next(state, host="john1", lease_seconds=30, now_ms=4)
    assert queue.dispatchable_hosts(
        state,
        hosts=["john1", "john2", "john3", "john4"],
        now_ms=5,
    ) == ["john3"]


def test_coordinator_dispatches_until_queue_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "queue.json"
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(
        state,
        task_spec("first", compatible_hosts=["john1"]),
        now_ms=2,
    )
    queue.add_task(
        state,
        task_spec(
            "second",
            compatible_hosts=["john1"],
            dependencies=["first"],
        ),
        now_ms=3,
    )
    queue._atomic_write(path, state)

    def complete_one(
        queue_path: Path,
        *,
        host: str,
        lease_seconds: float,
        dry_run: bool = False,
    ) -> int:
        del lease_seconds, dry_run
        with queue.locked_queue(queue_path) as working:
            task = queue.claim_next(
                working,
                host=host,
                lease_seconds=30,
                now_ms=10 + len(working["events"]),
            )
            assert task is not None
            queue.finish_task(
                working,
                task_id=task["id"],
                host=host,
                token=task["claim"]["token"],
                outcome="completed",
                artifact=task["artifact_path"],
                now_ms=20 + len(working["events"]),
            )
        return 0

    monkeypatch.setattr(queue, "dispatch_one", complete_one)
    assert (
        queue.run_coordinator(
            path,
            hosts=["john1"],
            lease_seconds=30,
            poll_seconds=0.001,
            idle_timeout_seconds=0,
        )
        == 0
    )
    assert [task["status"] for task in queue.load_queue(path)["tasks"]] == [
        "completed",
        "completed",
    ]


def test_coordinator_idle_timeout_ignores_owned_active_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "queue.json"
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(
        state,
        task_spec("origin", compatible_hosts=["john1"]),
        now_ms=2,
    )
    queue.add_task(
        state,
        task_spec(
            "dependent",
            compatible_hosts=["john1"],
            dependencies=["origin"],
        ),
        now_ms=3,
    )
    queue._atomic_write(path, state)

    def complete_one(
        queue_path: Path,
        *,
        host: str,
        lease_seconds: float,
        dry_run: bool = False,
    ) -> int:
        del lease_seconds, dry_run
        with queue.locked_queue(queue_path) as working:
            task = queue.claim_next(
                working,
                host=host,
                lease_seconds=30,
            )
        assert task is not None
        if task["id"] == "origin":
            time.sleep(0.03)
        with queue.locked_queue(queue_path) as working:
            queue.finish_task(
                working,
                task_id=task["id"],
                host=host,
                token=task["claim"]["token"],
                outcome="completed",
                artifact=task["artifact_path"],
            )
        return 0

    monkeypatch.setattr(queue, "dispatch_one", complete_one)
    assert (
        queue.run_coordinator(
            path,
            hosts=["john1"],
            lease_seconds=30,
            poll_seconds=0.001,
            idle_timeout_seconds=0.005,
        )
        == 0
    )
    assert [task["status"] for task in queue.load_queue(path)["tasks"]] == [
        "completed",
        "completed",
    ]


def test_coordinator_polls_for_new_work_while_long_task_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "queue.json"
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(
        state,
        task_spec("origin", compatible_hosts=["john1"]),
        now_ms=2,
    )
    queue._atomic_write(path, state)
    origin_started = threading.Event()
    release_origin = threading.Event()
    independent_started = threading.Event()
    started_before_release: list[bool] = []

    def dispatch(
        queue_path: Path,
        *,
        host: str,
        lease_seconds: float,
        dry_run: bool = False,
    ) -> int:
        del lease_seconds, dry_run
        with queue.locked_queue(queue_path) as working:
            task = queue.claim_next(
                working,
                host=host,
                lease_seconds=30,
            )
        assert task is not None
        if task["id"] == "origin":
            origin_started.set()
            assert release_origin.wait(2)
        else:
            independent_started.set()
        with queue.locked_queue(queue_path) as working:
            queue.finish_task(
                working,
                task_id=task["id"],
                host=host,
                token=task["claim"]["token"],
                outcome="completed",
                artifact=task["artifact_path"],
            )
        return 0

    def add_work() -> None:
        assert origin_started.wait(2)
        with queue.locked_queue(path) as working:
            queue.add_task(
                working,
                task_spec("independent", compatible_hosts=["john2"]),
            )
        started_before_release.append(independent_started.wait(0.5))
        release_origin.set()

    monkeypatch.setattr(queue, "dispatch_one", dispatch)
    adder = threading.Thread(target=add_work)
    adder.start()
    result = queue.run_coordinator(
        path,
        hosts=["john1", "john2"],
        lease_seconds=30,
        poll_seconds=0.001,
        idle_timeout_seconds=0,
    )
    adder.join()
    assert result == 0
    assert started_before_release == [True]


def test_coordinator_reports_dependency_deadlock(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    state = queue.empty_queue("campaign", now_ms=1)
    queue.add_task(state, task_spec("origin"), now_ms=2)
    queue.add_task(
        state,
        task_spec("blocked", dependencies=["origin"]),
        now_ms=3,
    )
    state["tasks"][0]["status"] = "failed"
    state["tasks"][0]["result"] = {
        "artifact": None,
        "error": "failed",
        "completed_unix_ms": 4,
        "host": "john1",
    }
    queue.refresh_dependencies(state)
    queue.validate_queue(state)
    queue._atomic_write(path, state)
    assert (
        queue.run_coordinator(
            path,
            hosts=["john1", "john2"],
            lease_seconds=30,
            poll_seconds=0.001,
            idle_timeout_seconds=0,
        )
        == 2
    )


def test_locked_queue_persists_valid_atomic_updates(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    queue._atomic_write(path, queue.empty_queue("campaign", now_ms=1))
    with queue.locked_queue(path) as state:
        queue.add_task(state, task_spec("task"), now_ms=2)
    loaded = queue.load_queue(path)
    assert loaded["tasks"][0]["id"] == "task"
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_init_add_claim_and_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "queue.json"
    spec = tmp_path / "task.json"
    spec.write_text(json.dumps(task_spec("task")))
    assert queue.main(["--queue", str(path), "init", "--campaign-id", "campaign"]) == 0
    assert queue.main(["--queue", str(path), "add", "--spec", str(spec)]) == 0
    assert (
        queue.main(
            [
                "--queue",
                str(path),
                "claim",
                "--host",
                "john1",
                "--lease-seconds",
                "30",
            ]
        )
        == 0
    )
    assert queue.main(["--queue", str(path), "validate"]) == 0
    assert '"valid": true' in capsys.readouterr().out


def test_cli_install_spec_adds_complete_campaign(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "queue.json"
    spec = tmp_path / "campaign.json"
    spec.write_text(
        json.dumps(
            {
                "experiment_id": "experiment-v1",
                "task_count": 2,
                "tasks": [
                    task_spec("consumer", dependencies=["origin"]),
                    task_spec("origin"),
                ],
            }
        )
    )
    queue._atomic_write(path, queue.empty_queue("campaign", now_ms=1))
    assert (
        queue.main(
            [
                "--queue",
                str(path),
                "install-spec",
                "--spec",
                str(spec),
            ]
        )
        == 0
    )
    installed = queue.load_queue(path)
    assert [task["id"] for task in installed["tasks"]] == ["consumer", "origin"]
    assert '"task_count": 2' in capsys.readouterr().out


def test_cli_cancel_pending_records_actor_and_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "queue.json"
    spec = tmp_path / "task.json"
    spec.write_text(json.dumps(task_spec("task")))
    assert queue.main(["--queue", str(path), "init", "--campaign-id", "campaign"]) == 0
    assert queue.main(["--queue", str(path), "add", "--spec", str(spec)]) == 0
    assert (
        queue.main(
            [
                "--queue",
                str(path),
                "cancel-pending",
                "--task-id",
                "task",
                "--actor",
                "test-suite",
                "--reason",
                "Superseded by a corrected campaign.",
            ]
        )
        == 0
    )
    task = queue.load_queue(path)["tasks"][0]
    assert task["status"] == "cancelled"
    assert task["administrative_cancellation"]["actor"] == "test-suite"
    assert "corrected campaign" in task["administrative_cancellation"]["reason"]
    assert '"status": "cancelled"' in capsys.readouterr().out
