from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from cascadia_cluster import (
    ClusterClient,
    ContainerInput,
    JobStatus,
    ObjectStoreClient,
    ObjectStoreConfig,
    Resources,
)
from cascadia_cluster.bacalhau_api import BacalhauAPI

pytestmark = pytest.mark.integration
ENDPOINT = "http://100.110.109.6:1234"


def _enabled() -> None:
    if os.environ.get("CASCADIA_CLUSTER_INTEGRATION") != "1":
        pytest.skip("set CASCADIA_CLUSTER_INTEGRATION=1 for live fabric tests")


def _image() -> str:
    value = os.environ.get("CASCADIA_CLUSTER_TEST_IMAGE", "")
    if "@sha256:" not in value:
        pytest.skip("CASCADIA_CLUSTER_TEST_IMAGE must be an immutable published image")
    return value


def _store() -> ObjectStoreClient:
    try:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    except KeyError:
        pytest.skip("MinIO credentials are required for live artifact tests")
    return ObjectStoreClient(
        ObjectStoreConfig(
            endpoint=os.environ.get("AWS_ENDPOINT_URL_S3", "http://100.110.109.6:9000"),
            access_key=access_key,
            secret_key=secret_key,
        )
    )


def _client(tmp_path: Path) -> ClusterClient:
    store = _store()
    store.ensure_bucket(store.config.input_bucket)
    store.ensure_bucket(store.config.result_bucket)
    return ClusterClient(
        ENDPOINT,
        state_directory=tmp_path / "state",
        object_store=store,
        artifact_directory=tmp_path / "accepted",
    )


def _job(index: int, *, exit_code: int = 0, sleep_seconds: float = 0) -> ContainerInput:
    script = (
        f"sleep {sleep_seconds}; "
        f"printf '{{\"index\":{index},\"seed\":{10000 + index}}}\\n' > /outputs/result.json; "
        f"exit {exit_code}"
    )
    return ContainerInput(
        f"seed-{index:04d}",
        args=("/bin/sh", "-c", script),
        application_metadata={"index": str(index), "seed": str(10000 + index)},
    )


def _map(client: ClusterClient, jobs: list[ContainerInput], request_id: str):
    return client.map(
        image=_image(),
        jobs=jobs,
        resources=Resources(cpu=0.25, memory_gib=0.25, disk_gib=0.25),
        outputs=("/outputs",),
        timeout_seconds=300,
        entrypoint=("/usr/local/bin/cascadia-cluster-job",),
        experiment_id="cluster-acceptance-v1",
        request_id=request_id,
    )


def test_live_membership_is_exact_and_topology_free() -> None:
    _enabled()
    nodes = BacalhauAPI(ENDPOINT).nodes()
    names = {
        node["Info"]["Labels"]["cascadia_internal_node"]
        for node in nodes
        if node.get("Connection") == "CONNECTED"
    }
    assert names == {"john1", "john2", "john3"}
    assert all("docker" in node["Info"]["ComputeNodeInfo"]["ExecutionEngines"] for node in nodes)
    capacities = {
        node["Info"]["Labels"]["cascadia_internal_node"]: node["Info"]["ComputeNodeInfo"][
            "MaxCapacity"
        ]["CPU"]
        for node in nodes
    }
    assert capacities == {"john1": 9, "john2": 10, "john3": 10}


def test_live_ordered_artifacts_idempotency_and_reconnect(tmp_path: Path) -> None:
    _enabled()
    client = _client(tmp_path)
    request_id = f"accept-order-{time.time_ns()}"
    jobs = [_job(index, sleep_seconds=(5 - index) * 0.05) for index in range(6)]
    results = _map(client, jobs, request_id)
    assert [result.item_key for result in results] == [job.key for job in jobs]
    assert all(result.status is JobStatus.SUCCEEDED for result in results)
    hashes = []
    for index, result in enumerate(results):
        payload = tmp_path / "accepted" / request_id / result.item_key / "result.json"
        assert json.loads(payload.read_text())["index"] == index
        hashes.append(hashlib.sha256(payload.read_bytes()).hexdigest())
    reconnected = client.reconnect(request_id).results()
    assert [result.accepted_execution_id for result in reconnected.results] == [
        result.accepted_execution_id for result in results
    ]
    assert len(set(hashes)) == len(jobs)


def test_live_deterministic_failure_is_not_retried(tmp_path: Path) -> None:
    _enabled()
    client = _client(tmp_path)
    request_id = f"accept-failure-{time.time_ns()}"
    handle = client.submit_map(
        image=_image(),
        jobs=[_job(0, exit_code=2)],
        resources=Resources(cpu=0.25, memory_gib=0.25, disk_gib=0.25),
        outputs=("/outputs",),
        timeout_seconds=120,
        entrypoint=("/usr/local/bin/cascadia-cluster-job",),
        experiment_id="cluster-acceptance-v1",
        request_id=request_id,
    )
    result = handle.wait(timeout_seconds=120).results().results[0]
    assert result.status is JobStatus.FAILED
    assert result.exit_code == 2
    assert result.attempts == 1


def test_live_content_addressed_input_is_verified_before_execution(tmp_path: Path) -> None:
    _enabled()
    client = _client(tmp_path)
    payload = tmp_path / "input.txt"
    payload.write_text("content-addressed input\n")
    reference = client.object_store.stage_file(payload, target="/inputs/input")
    request_id = f"accept-input-{time.time_ns()}"
    result = client.map(
        image=_image(),
        jobs=[
            ContainerInput(
                "input",
                args=(
                    "/bin/sh",
                    "-c",
                    "cp /inputs/input/input.txt /outputs/result.txt",
                ),
                inputs=(reference,),
            )
        ],
        resources=Resources(cpu=0.25, memory_gib=0.25, disk_gib=0.25),
        outputs=("/outputs",),
        timeout_seconds=120,
        entrypoint=("/usr/local/bin/cascadia-cluster-job",),
        experiment_id="cluster-acceptance-v1",
        request_id=request_id,
    )[0]
    accepted = tmp_path / "accepted" / request_id / "input" / "result.txt"
    assert result.status is JobStatus.SUCCEEDED
    assert accepted.read_bytes() == payload.read_bytes()


def test_live_scale_uses_all_workers(tmp_path: Path) -> None:
    _enabled()
    count = int(os.environ.get("CASCADIA_CLUSTER_SCALE_JOBS", "100"))
    client = _client(tmp_path)
    request_id = f"accept-scale-{count}-{time.time_ns()}"
    results = _map(client, [_job(index, sleep_seconds=0.5) for index in range(count)], request_id)
    assert len(results) == count
    job_ids = [result.bacalhau_job_id for result in results]
    api = BacalhauAPI(ENDPOINT)
    node_ids = {
        execution.get("NodeID")
        for job_id in job_ids
        for execution in api.executions(job_id or "")
        if execution.get("ComputeState", {}).get("StateType") == "Completed"
    }
    assert {"Johns-Mac-mini-local", "john2", "john3"} <= node_ids
