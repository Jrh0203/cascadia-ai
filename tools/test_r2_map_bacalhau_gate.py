from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import r2_map_bacalhau_gate as subject
from cascadia_cluster import InputReference, JobStatus

IMAGE = "registry/r2@sha256:" + "a" * 64


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_bytes(bytes(value))


def _fixture(root: Path, *, stage: str = "smoke") -> tuple[Path, Path, Path]:
    stage_name, count = subject.STAGES[stage]
    gate = root / "gate"
    checkpoint = "r2-terminal-step-000007235"
    field_id = f"field-{stage}"
    _write(
        gate / "contract.json",
        {
            "schema_id": subject.CONTRACT_SCHEMA,
            "stage": stage_name,
            "pair_count": count,
            "execution_partition": {"kind": "scheduler-managed-pairs"},
            "candidate_checkpoint_id": checkpoint,
            "opponent_field_manifest_id": field_id,
        },
    )
    _write(
        gate / "opponent-field.json",
        {
            "schema_id": subject.FIELD_SCHEMA,
            "manifest_id": field_id,
            "assignments": [{"pair_index": index} for index in range(count)],
        },
    )
    freeze = root / "freeze"
    _write(freeze / "freeze-receipt.json", {"checkpoint_id": checkpoint})
    _write(freeze / "r2-run/bundle.json", {"checkpoint_id": checkpoint})
    _write(freeze / "r2-run/checkpoints/model.safetensors", b"model")
    _write(freeze / "r2-backend-parity.json", {"passed": True})
    weights = root / "exact.bin"
    _write(weights, b"exact-nnue")
    contract = json.loads((gate / "contract.json").read_text())
    contract["execution_binding"] = {
        "image_digest": IMAGE,
        "candidate_freeze_receipt_sha256": subject._sha256_file(
            freeze / "freeze-receipt.json"
        ),
        "exact_weights_sha256": subject._sha256_file(weights),
        "opponent_field_sha256": subject._sha256_file(gate / "opponent-field.json"),
    }
    _write(gate / "contract.json", contract)
    return gate, freeze, weights


def _input_receipt(archive: Path, *, stage: str = "smoke") -> dict[str, object]:
    return {
        "stage": stage,
        "image_digest": IMAGE,
        "archive_sha256": subject._sha256_file(archive),
        "contract_sha256": "b" * 64,
        "field_sha256": "c" * 64,
        "exact_weights_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
    }


def test_gate_archive_is_deterministic_and_uses_pair_work_items(tmp_path: Path) -> None:
    gate, freeze, weights = _fixture(tmp_path)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_receipt = subject.build_gate_archive(
        gate_directory=gate,
        candidate_freeze=freeze,
        exact_weights=weights,
        stage="smoke",
        destination=first,
    )
    second_receipt = subject.build_gate_archive(
        gate_directory=gate,
        candidate_freeze=freeze,
        exact_weights=weights,
        stage="smoke",
        destination=second,
    )
    assert first_receipt["archive_sha256"] == second_receipt["archive_sha256"]
    with tarfile.open(first) as archive:
        names = set(archive.getnames())
    assert "campaign/receipts/pair-0000" in names
    assert "campaign/receipts/pair-0019" in names
    assert "r2-run/bundle.json" in names
    assert "exact-weights.bin" in names


def test_gate_input_rejects_topology_bearing_assignment(tmp_path: Path) -> None:
    gate, _, _ = _fixture(tmp_path)
    field = json.loads((gate / "opponent-field.json").read_text())
    field["assignments"][0]["compatible_hosts"] = ["john2"]
    _write(gate / "opponent-field.json", field)
    try:
        subject._validate_gate_inputs(gate, "smoke")
    except subject.GateFabricError as error:
        assert "topology-bearing" in str(error)
    else:
        raise AssertionError("physical host assignment was accepted")


def test_submit_gate_creates_one_pair_item_and_lets_bacalhau_pack_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "gate.tar"
    archive.write_bytes(b"archive")
    captured: dict[str, object] = {}

    class FakeStore:
        def stage_file(self, path: Path, *, target: str) -> InputReference:
            assert path == archive
            assert target == "/inputs/r2-gate"
            return InputReference(
                bucket="inputs",
                key="sha256/archive",
                sha256="a" * 64,
                target=target,
            )

    class FakeClient:
        class API:
            @staticmethod
            def nodes() -> list[dict]:
                return [
                    {
                        "Connection": "CONNECTED",
                        "Info": {
                            "Labels": {"cascadia_internal_node": name},
                            "ComputeNodeInfo": {
                                "ExecutionEngines": ["docker"],
                                "MaxCapacity": {"CPU": cpu},
                            },
                        },
                    }
                    for name, cpu in (("john1", 9), ("john2", 10), ("john3", 10))
                ]

        api = API()

        def submit_map(self, **kwargs: object) -> str:
            captured.update(kwargs)
            return "submitted"

    monkeypatch.setattr(subject, "_store", lambda: FakeStore())
    monkeypatch.setattr(subject, "_client", lambda *_args: FakeClient())
    result = subject.submit_gate(
        image=IMAGE,
        archive=archive,
        input_receipt=_input_receipt(archive),
        stage="smoke",
        request_id="request",
        state_directory=tmp_path / "state",
        artifact_directory=tmp_path / "artifacts",
    )
    assert result == "submitted"
    assert captured["scheduler_backpressure"] is True
    jobs = captured["jobs"]
    assert [job.key for job in jobs] == list(subject.expected_work_items("smoke"))
    assert all("--work-item" in subject.WORK_ITEM_SCRIPT for _job in jobs)
    unset = subject.WORK_ITEM_SCRIPT.index(
        "unset CASCADIA_APPLICATION_METADATA_JSON CASCADIA_OUTPUT_ROOT"
    )
    launch = subject.WORK_ITEM_SCRIPT.index("r2-map-cross-arch-focal")
    assert unset < launch
    assert "CASCADIA_PROTOCOL_VERSION CASCADIA_RETRYABLE_EXIT_CODES" in (
        subject.WORK_ITEM_SCRIPT
    )
    assert all("host" not in job.application_metadata for job in jobs)
    assert all(
        job.environment
        == {
            "MCE_LMR": "1",
            "MCE_DIVERSE_PREFILTER": "1",
            "RAYON_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
        for job in jobs
    )
    assert all(
        job.application_metadata["gate_archive_sha256"]
        == subject._sha256_file(archive)
        for job in jobs
    )
    resources = captured["resources"]
    assert resources.cpu == 2
    assert resources.memory_gib == 4


def test_live_gate_fabric_rejects_john4_or_capacity_drift() -> None:
    def node(name: str, cpu: int) -> dict:
        return {
            "Connection": "CONNECTED",
            "Info": {
                "Labels": {"cascadia_internal_node": name},
                "ComputeNodeInfo": {
                    "ExecutionEngines": ["docker"],
                    "MaxCapacity": {"CPU": cpu},
                },
            },
        }

    subject._validate_live_gate_fabric(
        [node("john1", 9), node("john2", 10), node("john3", 10)]
    )
    with pytest.raises(subject.GateFabricError, match="active fabric"):
        subject._validate_live_gate_fabric(
            [node("john1", 9), node("john2", 10), node("john3", 10), node("john4", 10)]
        )
    with pytest.raises(subject.GateFabricError, match="9/10/10"):
        subject._validate_live_gate_fabric(
            [node("john1", 8), node("john2", 10), node("john3", 10)]
        )


def test_aggregate_request_id_binds_campaign_archive() -> None:
    first = subject._aggregate_request_id("request", "a" * 64)
    second = subject._aggregate_request_id("request", "b" * 64)
    assert first == "request-aggregate-aaaaaaaaaaaa"
    assert second == "request-aggregate-bbbbbbbbbbbb"
    assert first != second
    with pytest.raises(subject.GateFabricError, match="malformed"):
        subject._aggregate_request_id("request", "not-a-sha")
def test_merge_work_items_is_idempotent_and_refuses_input_drift(tmp_path: Path) -> None:
    gate, _, _ = _fixture(tmp_path)
    result_root = tmp_path / "results"
    request_id = "req-r2"
    image = IMAGE
    archive = tmp_path / "gate.tar"
    archive.write_bytes(b"gate")
    input_receipt = _input_receipt(archive)
    for pair_index in range(subject.STAGES["smoke"][1]):
        work_item = f"pair-{pair_index:04}"
        campaign = result_root / request_id / work_item / "campaign"
        _write(campaign / "contract.json", json.loads((gate / "contract.json").read_text()))
        _write(
            campaign / "opponent-field.json",
            json.loads((gate / "opponent-field.json").read_text()),
        )
        _write(
            campaign / f"work-item-summaries/{work_item}.json",
            {"work_item_id": work_item},
        )
        _write(
            campaign / f"receipts/{work_item}/{work_item}.json",
            {"pair_index": pair_index},
        )
        scheduler = {
            "schema_id": "cascadia.cluster.accepted-result.v1",
            "request_id": request_id,
            "item_id": work_item,
            "bacalhau_job_id": f"job-{pair_index}",
            "accepted_execution_id": f"execution-{pair_index}",
            "image_digest": image,
            "spec_sha256": "f" * 64,
            "output_manifest_sha256": "1" * 64,
            "application_metadata": {
                "stage": "smoke",
                "work_item": work_item,
                "pair_index": f"{pair_index:04}",
                "image_digest": image,
                "gate_archive_sha256": input_receipt["archive_sha256"],
                "contract_sha256": input_receipt["contract_sha256"],
                "opponent_field_sha256": input_receipt["field_sha256"],
                "exact_weights_sha256": input_receipt["exact_weights_sha256"],
                "freeze_receipt_sha256": input_receipt["freeze_receipt_sha256"],
            },
            "attempts": 1,
            "created_unix_ns": 1,
            "modified_unix_ns": 2,
        }
        scheduler["receipt_sha256"] = subject._canonical_sha256(scheduler)
        _write(
            result_root / request_id / ".receipts" / f"{work_item}.json",
            scheduler,
        )
    destination = tmp_path / "campaign"
    subject._merge_work_items(
        gate_directory=gate,
        result_root=result_root,
        request_id=request_id,
        image=image,
        input_receipt=input_receipt,
        destination=destination,
    )
    subject._merge_work_items(
        gate_directory=gate,
        result_root=result_root,
        request_id=request_id,
        image=image,
        input_receipt=input_receipt,
        destination=destination,
    )
    _write(
        result_root / request_id / "pair-0001/campaign/contract.json",
        {"drift": True},
    )
    try:
        subject._merge_work_items(
            gate_directory=gate,
            result_root=result_root,
            request_id=request_id,
            image=image,
            input_receipt=input_receipt,
            destination=destination,
        )
    except subject.GateFabricError as error:
        assert "immutable campaign inputs" in str(error)
    else:
        raise AssertionError("drifted work-item input was accepted")


def test_scheduler_wall_time_is_reconnect_stable_and_order_independent() -> None:
    first = SimpleNamespace(created_unix_ns=1_000_000_000, modified_unix_ns=4_000_000_000)
    second = SimpleNamespace(created_unix_ns=2_000_000_000, modified_unix_ns=8_000_000_000)
    assert subject._scheduler_wall_seconds([first, second]) == 7.0
    assert subject._scheduler_wall_seconds([second, first]) == 7.0


def test_scheduler_wall_time_rejects_missing_or_nonmonotonic_provenance() -> None:
    with pytest.raises(subject.GateFabricError, match="timestamps"):
        subject._scheduler_wall_seconds(
            [SimpleNamespace(created_unix_ns=2, modified_unix_ns=1)]
        )


def test_aggregate_archive_excludes_prior_aggregate_outputs(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write(campaign / "contract.json", {"contract": 1})
    _write(campaign / "opponent-field.json", {"field": 1})
    _write(campaign / "work-item-summaries/pair-0000.json", {"pair": 0})
    _write(campaign / "receipts/pair-0000/pair-0000.json", {"pair": 0})
    _write(campaign / "scheduler-provenance/pair-0000.json", {"pair": 0})
    _write(campaign / "scheduler-provenance/aggregate.json", {"old": 1})
    _write(campaign / "reports/focal-benchmark.json", {"old": 1})
    _write(campaign / "projections/dashboard.json", {"old": 1})
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    subject._campaign_archive(campaign, first)
    _write(campaign / "scheduler-provenance/aggregate.json", {"old": 2})
    _write(campaign / "reports/focal-benchmark.json", {"old": 2})
    _write(campaign / "projections/dashboard.json", {"old": 2})
    subject._campaign_archive(campaign, second)
    assert subject._sha256_file(first) == subject._sha256_file(second)
    with tarfile.open(first) as archive:
        names = set(archive.getnames())
    assert "campaign/scheduler-provenance/pair-0000.json" in names
    assert "campaign/scheduler-provenance/aggregate.json" not in names
    assert "campaign/reports/focal-benchmark.json" not in names


def _scheduler_nodes(allocated: tuple[int, int, int]) -> list[dict[str, object]]:
    return [
        {
            "Connection": "CONNECTED",
            "Info": {
                "Labels": {"cascadia_internal_node": name},
                "ComputeNodeInfo": {
                    "MaxCapacity": {"CPU": capacity},
                    # Bacalhau's protobuf JSON projection elides scalar zeroes.
                    "AvailableCapacity": (
                        {"CPU": capacity - used} if capacity != used else {}
                    ),
                    "RunningExecutions": used // 2,
                },
            },
        }
        for name, capacity, used in zip(
            ("john1", "john2", "john3"), (9, 10, 10), allocated, strict=True
        )
    ]


def test_scheduler_observations_persist_terminal_state_and_summarize(tmp_path: Path) -> None:
    class Handle:
        def __init__(self) -> None:
            self.calls = 0

        def status(self):
            self.calls += 1
            return (
                (JobStatus.RUNNING, JobStatus.QUEUED)
                if self.calls == 1
                else (JobStatus.SUCCEEDED, JobStatus.SUCCEEDED)
            )

    class API:
        def __init__(self) -> None:
            self.calls = 0

        def nodes(self):
            self.calls += 1
            return _scheduler_nodes((8, 10, 10) if self.calls == 1 else (0, 0, 0))

    handle = Handle()
    api = API()
    client = SimpleNamespace(api=api)
    samples, path = subject._monitor_scheduler_request(
        handle=handle,
        client=client,
        state_directory=tmp_path,
        request_id="request",
        poll_seconds=0.0,
    )
    assert path.is_file()
    assert len(samples) == 2
    summary = subject._scheduler_utilization(samples)
    assert summary["cpu_capacity_min"] == 29
    assert summary["cpu_allocated_peak"] == 28
    assert summary["nodes"]["john2"]["cpu_allocated_peak"] == 10
    before = (handle.calls, api.calls)
    resumed, resumed_path = subject._monitor_scheduler_request(
        handle=handle,
        client=client,
        state_directory=tmp_path,
        request_id="request",
        poll_seconds=0.0,
    )
    assert resumed == samples
    assert resumed_path == path
    assert (handle.calls, api.calls) == before


def test_completion_packet_joins_focal_and_scheduler_evidence(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write(
        campaign / "reports/focal-benchmark.json",
        {
            "result": {
                "kind": "development",
                "statistics": {
                    "pairs": 250,
                    "physical_games": 500,
                    "classification": "promote",
                },
            }
        },
    )
    (campaign / "reports/focal-benchmark.md").write_text("# Focal report\n")
    utilization = {
        "sample_count": 2,
        "observed_seconds": 10.0,
        "cpu_capacity_min": 29.0,
        "cpu_capacity_max": 29.0,
        "cpu_allocated_mean": 27.5,
        "cpu_allocated_peak": 28.0,
        "cpu_utilization_mean": 27.5 / 29.0,
        "cpu_utilization_peak": 28.0 / 29.0,
        "nodes": {
            name: {
                "cpu_capacity_min": capacity,
                "cpu_capacity_max": capacity,
                "cpu_allocated_mean": capacity - 0.5,
                "cpu_allocated_peak": capacity,
            }
            for name, capacity in (("john1", 9.0), ("john2", 10.0), ("john3", 10.0))
        },
    }
    scheduler = {
        "schema_id": "cascadia.r2-map.scheduler-provenance.v1",
        "stage": "development",
        "request_id": "request",
        "aggregate_request_id": "request-aggregate",
        "image_digest": IMAGE,
        "gate_input": {"archive_sha256": "a" * 64},
        "scheduler_observations_sha256": "b" * 64,
        "scheduler_utilization": utilization,
        "work_items": [{"item_id": f"pair-{index:04}"} for index in range(250)],
        "retry_count": 3,
        "report_sha256": "c" * 64,
    }
    _write(campaign / "reports/scheduler-provenance.json", scheduler)
    subject._write_completion_artifacts(
        campaign_directory=campaign,
        scheduler_report=scheduler,
    )
    subject._write_completion_artifacts(
        campaign_directory=campaign,
        scheduler_report=scheduler,
    )
    completion = json.loads((campaign / "reports/campaign-completion.json").read_text())
    combined = (campaign / "reports/focal-benchmark-complete.md").read_text()
    assert completion["pairs"] == 250
    assert completion["physical_games"] == 500
    assert completion["classification"] == "promote"
    assert completion["retry_count"] == 3
    assert completion["scheduler_utilization"]["cpu_allocated_peak"] == 28.0
    assert "Scheduler utilization mean/peak" in combined
    assert "john1" in combined and "john2" in combined and "john3" in combined
