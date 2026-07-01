from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import blake3
import v3_phase2_pipeline as pipeline
from cascadia_cluster import BacalhauAPIError, ContainerInput, JobStatus


class FakeStore:
    def stage_file(self, path: Path, *, target: str):
        from cascadia_cluster import InputReference

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return InputReference(
            bucket="inputs",
            key=f"sha256/{digest}/{path.name}",
            sha256=digest,
            target=target,
            endpoint="http://store",
        )


def test_monitor_survives_transient_scheduler_reads(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeHandle:
        status_calls = 0
        result_calls = 0

        def status(self):
            self.status_calls += 1
            if self.status_calls == 1:
                raise BacalhauAPIError("transient status failure")
            return (JobStatus.SUCCEEDED,)

        def results(self):
            self.result_calls += 1
            if self.result_calls == 1:
                raise BacalhauAPIError("transient result failure")
            return SimpleNamespace(
                failure_count=0,
                results=(SimpleNamespace(status=JobStatus.SUCCEEDED),),
                elapsed_seconds=12.5,
            )

    handle = FakeHandle()
    client = SimpleNamespace(submit_map=lambda **_kwargs: handle)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)
    progress = tmp_path / "progress.json"
    result = pipeline._monitor(
        client=client,
        image="registry/image@sha256:" + "a" * 64,
        jobs=[ContainerInput(key="item")],
        resources=SimpleNamespace(),
        request_id="request",
        experiment_id="experiment",
        artifact_directory=tmp_path / "artifacts",
        progress=progress,
        timeout_seconds=60,
        validate=lambda _directory, _job: {"records": 1},
    )
    final_progress = json.loads(progress.read_text())
    assert handle.status_calls == 2
    assert handle.result_calls == 2
    assert final_progress["scheduler_status"] == "healthy"
    assert final_progress["scheduler_transient_errors"] == 2
    assert final_progress["fraction_complete"] == 1.0
    assert result["passed"] is True
    assert result["totals"] == {"records": 1}


def test_verify_jobs_are_topology_free_and_content_addressed(tmp_path: Path) -> None:
    shards = []
    for index in range(2):
        path = tmp_path / f"shard-{index}.v3g"
        path.write_bytes(f"shard-{index}".encode())
        shards.append(path)
    jobs, sources = pipeline.build_verify_jobs(shards, FakeStore())
    assert [job.key for job in jobs] == ["verify-00000", "verify-00001"]
    assert set(sources) == {job.key for job in jobs}
    assert all(job.inputs[0].target == "/inputs/shard" for job in jobs)
    assert all("host" not in json.dumps(dict(job.application_metadata)) for job in jobs)
    assert all(len(job.application_metadata["source_sha256"]) == 64 for job in jobs)


def test_label_jobs_pin_exact_r600_and_shared_authority(tmp_path: Path) -> None:
    state = tmp_path / "campaign-state.json"
    weights = tmp_path / "qualified-v1.bin"
    root = tmp_path / "roots-00000.v3r"
    state.write_text("state")
    weights.write_bytes(b"weights")
    root.write_bytes(b"roots")
    jobs, _ = pipeline.build_label_jobs(
        [root],
        FakeStore(),
        campaign_state=state,
        v1_weights=weights,
        approved_readiness_sha256="a" * 64,
        cycle=None,
    )
    job = jobs[0]
    assert "--rollouts" in job.args
    assert job.args[job.args.index("--rollouts") + 1] == "600"
    assert len({reference.target for reference in job.inputs}) == 3
    assert job.environment == {"RAYON_NUM_THREADS": "1"}


def test_validation_cache_jobs_are_one_shard_per_container(tmp_path: Path) -> None:
    labels = []
    for index in range(2):
        path = tmp_path / f"validation-{index:05d}.v3l"
        path.write_bytes(f"labels-{index}".encode())
        path.with_suffix(".receipt.json").write_text(
            json.dumps(
                {
                    "schema_id": "cascadia-v3-teacher-label-shard-receipt-v1",
                    "passed": True,
                    "roots": 1_000,
                    "candidate_estimates": 26_000 + index,
                }
            )
        )
        labels.append(path)
    jobs, sources = pipeline.build_validation_cache_jobs(labels, FakeStore())
    assert [job.key for job in jobs] == [
        "validation-cache-00000",
        "validation-cache-00001",
    ]
    assert sources[jobs[0].key] == labels[0]
    assert jobs[0].args[0] == "teacher_labels_to_training"
    assert jobs[0].args[jobs[0].args.index("--output") + 1].endswith(".v3t")
    assert jobs[0].environment == {"RAYON_NUM_THREADS": "1"}
    assert jobs[0].application_metadata["expected_rows"] == "26000"


def test_validation_cache_artifact_requires_source_pinned_candidate_count(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "validation-00000.v3t"
    shard.write_bytes(b"packed-validation")
    receipt = {
        "schema_id": "cascadia-v3-teacher-training-expansion-v1",
        "passed": True,
        "scientific_eligible": True,
        "roots": 1_000,
        "rows": 26_638,
        "realized_rows": 525,
        "counterfactual_rows": 26_113,
        "output_bytes": shard.stat().st_size,
        "output_blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
    }
    (tmp_path / "validation-00000.receipt.json").write_text(json.dumps(receipt))
    source = tmp_path / "source"
    source.mkdir()
    labels = source / "validation-00000.v3l"
    labels.write_bytes(b"labels")
    labels.with_suffix(".receipt.json").write_text(
        json.dumps(
            {
                "schema_id": "cascadia-v3-teacher-label-shard-receipt-v1",
                "passed": True,
                "roots": 1_000,
                "candidate_estimates": 26_638,
            }
        )
    )
    job = pipeline.build_validation_cache_jobs([labels], FakeStore())[0][0]
    assert pipeline._validate_validation_cache(tmp_path, job) == {
        "roots": 1_000,
        "rows": 26_638,
        "realized_rows": 525,
        "counterfactual_rows": 26_113,
        "output_bytes": len(b"packed-validation"),
    }


def test_validation_cache_reconciliation_reuses_completed_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    labels = source / "validation-00000.v3l"
    labels.write_bytes(b"labels")
    labels.with_suffix(".receipt.json").write_text(
        json.dumps(
            {
                "schema_id": "cascadia-v3-teacher-label-shard-receipt-v1",
                "passed": True,
                "roots": 1_000,
                "candidate_estimates": 1_000,
            }
        )
    )
    job = pipeline.build_validation_cache_jobs([labels], FakeStore())[0][0]
    artifact = tmp_path / "artifacts" / "request" / job.key
    artifact.mkdir(parents=True)
    shard = artifact / "validation-00000.v3t"
    shard.write_bytes(b"packed")
    (artifact / "validation-00000.receipt.json").write_text(
        json.dumps(
            {
                "schema_id": "cascadia-v3-teacher-training-expansion-v1",
                "passed": True,
                "scientific_eligible": True,
                "roots": 1_000,
                "rows": 1_000,
                "realized_rows": 1_000,
                "counterfactual_rows": 0,
                "output_bytes": shard.stat().st_size,
                "output_blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        )
    )
    completion = pipeline._reconcile_validation_cache(
        jobs=[job],
        artifact_directory=tmp_path / "artifacts",
        request_id="request",
        image="registry/image@sha256:" + "a" * 64,
    )
    assert completion["passed"] is True
    assert completion["reconciled_existing_results"] is True
    assert completion["totals"]["rows"] == 1_000


def test_authorization_requires_exact_labeling_phase(tmp_path: Path) -> None:
    state = {
        "schema_id": "cascadia-v3-campaign-state-v1",
        "part": 2,
        "phase2_authorized": True,
        "phase": "bootstrap_labeling",
        "protected_seed_values_opened": False,
        "readiness_sha256": "b" * 64,
        "approved_readiness_sha256": "b" * 64,
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    assert pipeline._read_authorized_state(path, "bootstrap_labeling") == state
    try:
        pipeline._read_authorized_state(path, "bootstrap_training")
    except pipeline.PipelineError:
        pass
    else:
        raise AssertionError("wrong phase was accepted")


def test_fabric_is_exactly_john1_through_john3() -> None:
    nodes = [
        {
            "Connection": "CONNECTED",
            "Info": {
                "Labels": {"cascadia_internal_node": name},
                "ComputeNodeInfo": {
                    "MaxCapacity": {"CPU": cpu},
                    "ExecutionEngines": ["docker"],
                },
            },
        }
        for name, cpu in (("john1", 9), ("john2", 10), ("john3", 10))
    ]
    pipeline._validate_fabric(nodes)
    nodes.append(
        {
            "Connection": "CONNECTED",
            "Info": {
                "Labels": {"cascadia_internal_node": "john4"},
                "ComputeNodeInfo": {
                    "MaxCapacity": {"CPU": 10},
                    "ExecutionEngines": ["docker"],
                },
            },
        }
    )
    try:
        pipeline._validate_fabric(nodes)
    except pipeline.PipelineError:
        pass
    else:
        raise AssertionError("john4 was accepted into the V3 compute fabric")
