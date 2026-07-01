from __future__ import annotations

import json
from pathlib import Path

import pytest
from cascadia_cluster import (
    ArtifactFile,
    ArtifactManifest,
    ClusterClient,
    ContainerInput,
    JobResult,
    JobStatus,
    MapError,
    RequestConflictError,
    Resources,
)

IMAGE = "registry.cascadia/worker@sha256:" + "c" * 64


class FakeAPI:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.executions_by_job: dict[str, list[dict]] = {}
        self.submissions = 0
        self.list_calls = 0
        self.node_capacities = ((9, 10, 80), (10, 10, 80), (10, 10, 80))

    def list_jobs(self, *, labels: dict[str, str] | None = None) -> list[dict]:
        self.list_calls += 1
        return [
            job
            for job in self.jobs.values()
            if all(job.get("Labels", {}).get(key) == value for key, value in (labels or {}).items())
        ]

    def submit(self, job: dict, *, idempotency_token: str) -> dict:
        self.submissions += 1
        job_id = f"job-{self.submissions}"
        stored = dict(job)
        stored.update(
            {
                "ID": job_id,
                "State": {"StateType": "Pending", "Message": ""},
                "CreateTime": self.submissions,
                "ModifyTime": self.submissions,
            }
        )
        self.jobs[job_id] = stored
        self.executions_by_job[job_id] = []
        return {"JobID": job_id, "EvaluationID": f"eval-{self.submissions}"}

    def get_job(self, job_id: str, *, include: str = "executions") -> dict:
        return {"Job": self.jobs[job_id]}

    def executions(self, job_id: str) -> list[dict]:
        return self.executions_by_job[job_id]

    def stop(self, job_id: str, *, reason: str) -> dict:
        self.jobs[job_id]["State"] = {"StateType": "Stopped", "Message": reason}
        return {"JobID": job_id}

    def nodes(self) -> list[dict]:
        return [
            {
                "Connection": "CONNECTED",
                "Info": {
                    "ComputeNodeInfo": {
                        "ExecutionEngines": ["docker"],
                        "MaxCapacity": {
                            "CPU": cpu,
                            "Memory": memory_gib * 1024**3,
                            "Disk": disk_gib * 1024**3,
                        },
                    }
                },
            }
            for cpu, memory_gib, disk_gib in self.node_capacities
        ]


def _client(tmp_path: Path) -> tuple[ClusterClient, FakeAPI]:
    client = ClusterClient("http://bacalhau.invalid", state_directory=tmp_path)
    api = FakeAPI()
    client.api = api  # type: ignore[assignment]
    return client, api


def _submit(client: ClusterClient, request_id: str = "req-fixed"):
    return client.submit_map(
        image=IMAGE,
        jobs=[ContainerInput("first", args=("1",)), ContainerInput("second", args=("2",))],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id=request_id,
        experiment_id="unit-test",
    )


def test_results_preserve_input_order_under_out_of_order_completion(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    handle = _submit(client)
    ids = list(api.jobs)
    api.jobs[ids[1]]["State"] = {"StateType": "Completed", "Message": ""}
    api.executions_by_job[ids[1]] = [
        {
            "ID": "exec-second",
            "ComputeState": {"StateType": "Completed"},
            "RunOutput": {"ExitCode": 0},
        }
    ]
    api.jobs[ids[0]]["State"] = {"StateType": "Completed", "Message": ""}
    api.executions_by_job[ids[0]] = [
        {
            "ID": "exec-first",
            "ComputeState": {"StateType": "Completed"},
            "RunOutput": {"ExitCode": 0},
        }
    ]
    result = handle.results()
    assert [item.item_key for item in result.results] == ["first", "second"]
    assert [item.accepted_execution_id for item in result.results] == ["exec-first", "exec-second"]


def test_submission_is_idempotent_and_reconnectable(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    first = _submit(client)
    second = _submit(client)
    reconnected = client.reconnect("req-fixed")
    assert api.submissions == 2
    assert first.status() == second.status() == reconnected.status()


def test_bacalhau_job_name_binds_full_request_not_colliding_suffix(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    for request_id in ("r2-smoke-v8-aggregate", "r2-development-v8-aggregate"):
        client.submit_map(
            image=IMAGE,
            jobs=[ContainerInput("aggregate")],
            resources=Resources(1, 1, 1),
            outputs=["/outputs"],
            timeout_seconds=20,
            request_id=request_id,
            experiment_id="unit-test",
        )
    names = [job["Name"] for job in api.jobs.values()]
    assert len(names) == len(set(names)) == 2
    assert all(name.endswith("-0000-aggregate") for name in names)


def test_job_name_binds_full_request_id_not_a_colliding_suffix(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    suffix = "same-tail-12"
    for prefix in ("smoke", "development"):
        client.submit_map(
            image=IMAGE,
            jobs=[ContainerInput("aggregate")],
            resources=Resources(1, 1, 1),
            outputs=["/outputs"],
            timeout_seconds=20,
            request_id=f"{prefix}-{suffix}",
            experiment_id="unit-test",
        )
    names = [job["Name"] for job in api.jobs.values()]
    assert len(names) == len(set(names)) == 2
    assert all(suffix not in name for name in names)


def test_accepted_artifact_persists_scheduler_provenance(tmp_path: Path) -> None:
    client = ClusterClient(
        "http://bacalhau.invalid",
        state_directory=tmp_path / "state",
        object_store=object(),  # type: ignore[arg-type]
        artifact_directory=tmp_path / "artifacts",
    )
    result = JobResult(
        item_key="pair-0007",
        request_id="req-fixed",
        bacalhau_job_id="job-7",
        accepted_execution_id="exec-9",
        image_digest=IMAGE,
        spec_sha256="a" * 64,
        status=JobStatus.SUCCEEDED,
        artifact_manifest=ArtifactManifest(
            protocol_version="cascadia-cluster-map-v1",
            command=("worker", "pair-0007"),
            files=(ArtifactFile("result.json", 3, "b" * 64),),
            application_metadata={"gate_archive_sha256": "c" * 64},
        ),
        attempts=2,
        created_unix_ns=11,
        modified_unix_ns=22,
    )
    client._persist_result_receipt(result)
    client._persist_result_receipt(result)
    receipt = json.loads((tmp_path / "artifacts/req-fixed/.receipts/pair-0007.json").read_text())
    assert receipt["request_id"] == "req-fixed"
    assert receipt["item_id"] == "pair-0007"
    assert receipt["bacalhau_job_id"] == "job-7"
    assert receipt["accepted_execution_id"] == "exec-9"
    assert receipt["image_digest"] == IMAGE
    assert receipt["spec_sha256"] == "a" * 64
    assert receipt["application_metadata"] == {"gate_archive_sha256": "c" * 64}
    assert len(receipt["output_manifest_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64


def test_conflicting_item_reuse_is_rejected(tmp_path: Path) -> None:
    client, _api = _client(tmp_path)
    _submit(client)
    with pytest.raises(RequestConflictError, match="conflicts"):
        client.submit_map(
            image=IMAGE,
            jobs=[ContainerInput("first", args=("changed",))],
            resources=Resources(1, 1, 1),
            outputs=["/outputs"],
            timeout_seconds=20,
            request_id="req-fixed",
            experiment_id="unit-test",
        )


def test_partial_failure_preserves_all_results(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    handle = _submit(client, "req-partial")
    ids = list(api.jobs)
    api.jobs[ids[0]]["State"] = {"StateType": "Completed", "Message": ""}
    api.jobs[ids[1]]["State"] = {"StateType": "Failed", "Message": "exit 2"}
    api.executions_by_job[ids[0]] = [
        {"ID": "ok", "ComputeState": {"StateType": "Completed"}, "RunOutput": {"ExitCode": 0}}
    ]
    api.executions_by_job[ids[1]] = [
        {"ID": "bad", "ComputeState": {"StateType": "Failed"}, "RunOutput": {"ExitCode": 2}}
    ]
    result = handle.results()
    error = MapError(result.request_id, result.results)
    assert result.failure_count == 1
    assert error.results[0].status is JobStatus.SUCCEEDED
    assert error.results[1].status is JobStatus.FAILED


def test_attempt_count_excludes_scheduler_bid_rejections(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    handle = client.submit_map(
        image=IMAGE,
        jobs=[ContainerInput("first", args=("1",))],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-attempts",
        experiment_id="unit-test",
    )
    job_id = next(iter(api.jobs))
    api.jobs[job_id]["State"] = {"StateType": "Completed", "Message": ""}
    api.executions_by_job[job_id] = [
        {"ID": "bid-1", "ComputeState": {"StateType": "AskForBidRejected"}},
        {"ID": "bid-2", "ComputeState": {"StateType": "BidRejected"}},
        {
            "ID": "run-1",
            "ComputeState": {"StateType": "Completed"},
            "RunOutput": {"ExitCode": 0},
        },
    ]
    assert handle.results().results[0].attempts == 1


def test_completed_job_with_nonzero_worker_exit_is_failed(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    handle = client.submit_map(
        image=IMAGE,
        jobs=[ContainerInput("first", args=("1",))],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-worker-failure",
        experiment_id="unit-test",
    )
    job_id = next(iter(api.jobs))
    api.jobs[job_id]["State"] = {"StateType": "Completed", "Message": ""}
    api.executions_by_job[job_id] = [
        {
            "ID": "run-1",
            "ComputeState": {"StateType": "Completed"},
            "RunOutput": {"ExitCode": 1},
        }
    ]
    result = handle.results().results[0]
    assert result.status is JobStatus.FAILED
    assert result.exit_code == 1
    assert result.failure_reason == "worker execution exit 1"


def test_cancel_preserves_terminal_jobs_and_stops_only_nonterminal(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    handle = _submit(client, "req-cancel")
    first, second = list(api.jobs)
    api.jobs[first]["State"] = {"StateType": "Completed", "Message": ""}
    handle.cancel("unit cancellation")
    assert api.jobs[first]["State"]["StateType"] == "Completed"
    assert api.jobs[second]["State"]["StateType"] == "Stopped"


def test_s3_input_payload_binds_application_checksum_without_minio_head_metadata(
    tmp_path: Path,
) -> None:
    from cascadia_cluster import InputReference

    client, api = _client(tmp_path)
    client.submit_map(
        image=IMAGE,
        jobs=[
            ContainerInput(
                "input",
                inputs=(
                    InputReference(
                        "cascadia-inputs",
                        "sha256/ab/input.bin",
                        "a" * 64,
                        "/inputs/input",
                        endpoint="http://object-store",
                    ),
                ),
            )
        ],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-input",
        experiment_id="unit-test",
    )
    task = next(iter(api.jobs.values()))["Tasks"][0]
    source = task["InputSources"][0]["Source"]
    assert "ChecksumSHA256" not in source["Params"]
    assert json.loads(task["Env"]["CASCADIA_INPUT_SHA256_JSON"]) == {
        "/inputs/input/input.bin": "a" * 64
    }


def test_scheduler_backpressure_adds_bounded_multinode_placement_slack(
    tmp_path: Path,
) -> None:
    client, api = _client(tmp_path)
    handle = client.submit_map(
        image=IMAGE,
        jobs=[ContainerInput(f"pair-{index:04}") for index in range(10)],
        resources=Resources(2, 4, 4),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-managed",
        experiment_id="unit-test",
        scheduler_backpressure=True,
    )
    # Exact packing is six jobs. One largest-node window (two jobs) prevents
    # Bacalhau placement skew from idling another node without binding hosts.
    assert api.submissions == 8
    state = json.loads((tmp_path / "requests/req-managed.json").read_text())
    assert state["schema_id"] == "cascadia.cluster.managed-request-state.v2"
    assert state["admission"] == {
        "kind": "scheduler-capacity-backpressure",
        "maximum_outstanding": 8,
        "closed": False,
    }
    assert sum(item["bacalhau_job_id"] is not None for item in state["items"]) == 8
    assert handle.status().count(JobStatus.QUEUED) == 10


def test_managed_map_admits_next_item_after_terminal_and_reconnects_exactly_once(
    tmp_path: Path,
) -> None:
    client, api = _client(tmp_path)
    api.node_capacities = ((2, 2, 2),)
    jobs = [ContainerInput(f"item-{index}") for index in range(5)]
    handle = client.submit_map(
        image=IMAGE,
        jobs=jobs,
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-window",
        experiment_id="unit-test",
        scheduler_backpressure=True,
    )
    assert api.submissions == 2
    first = next(iter(api.jobs))
    api.jobs[first]["State"] = {"StateType": "Completed", "Message": ""}
    handle.status()
    assert api.submissions == 3

    reconnected = client.reconnect("req-window")
    reconnected.status()
    assert api.submissions == 3
    submitted = [item["Labels"]["cascadia.item_id"] for item in api.jobs.values()]
    assert submitted == ["item-0", "item-1", "item-2"]


def test_fresh_managed_admission_does_not_scan_scheduler_history(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    api.node_capacities = ((2, 2, 2),)
    handle = client.submit_map(
        image=IMAGE,
        jobs=[ContainerInput(f"item-{index}") for index in range(5)],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-fresh-managed",
        experiment_id="unit-test",
        scheduler_backpressure=True,
    )
    assert api.submissions == 2
    assert api.list_calls == 0

    first = next(iter(api.jobs))
    api.jobs[first]["State"] = {"StateType": "Completed", "Message": ""}
    handle.status()
    assert api.submissions == 3
    assert api.list_calls == 0


def test_managed_map_keeps_pending_unschedulable_job_inside_window(tmp_path: Path) -> None:
    client, api = _client(tmp_path)
    api.node_capacities = ((1, 1, 1),)
    handle = client.submit_map(
        image=IMAGE,
        jobs=[ContainerInput("item-0"), ContainerInput("item-1")],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-pending",
        experiment_id="unit-test",
        scheduler_backpressure=True,
    )
    first = next(iter(api.jobs))
    api.jobs[first]["State"] = {
        "StateType": "Pending",
        "Message": "not enough nodes to run job",
    }
    assert handle.status() == (JobStatus.QUEUED, JobStatus.QUEUED)
    assert api.submissions == 1


def test_managed_map_recovers_submit_before_state_write_without_duplicate(
    tmp_path: Path,
) -> None:
    client, api = _client(tmp_path)
    api.node_capacities = ((1, 1, 1),)
    client.submit_map(
        image=IMAGE,
        jobs=[
            ContainerInput("item-0"),
            ContainerInput("item-1"),
            ContainerInput("item-2"),
        ],
        resources=Resources(1, 1, 1),
        outputs=["/outputs"],
        timeout_seconds=20,
        request_id="req-recover",
        experiment_id="unit-test",
        scheduler_backpressure=True,
    )
    first = next(iter(api.jobs))
    api.jobs[first]["State"] = {"StateType": "Completed", "Message": ""}
    state_path = tmp_path / "requests/req-recover.json"
    state = json.loads(state_path.read_text())
    second_payload = state["items"][1]["job_payload"]
    client.api.submit(second_payload, idempotency_token="simulated-crash")
    # State still says item-1 was not submitted. Label/spec recovery must bind
    # the already-created scheduler job instead of creating a duplicate.
    reconnected = client.reconnect("req-recover")
    reconnected.status()
    assert api.submissions == 2
    assert api.list_calls == 1
    state = json.loads(state_path.read_text())
    assert state["items"][1]["bacalhau_job_id"] == "job-2"

    api.jobs["job-2"]["State"] = {"StateType": "Completed", "Message": ""}
    reconnected.status()
    assert api.submissions == 3
    assert api.list_calls == 1
    state = json.loads(state_path.read_text())
    assert state["items"][2]["bacalhau_job_id"] == "job-3"
