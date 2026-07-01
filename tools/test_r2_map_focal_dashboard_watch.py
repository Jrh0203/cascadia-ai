from __future__ import annotations

import json
from pathlib import Path

import r2_map_focal_dashboard_watch as subject
from cascadia_cluster.models import canonical_sha256


def _status() -> dict:
    host = {
        "intent": "idle",
        "detail": None,
        "benchmark_pairs_completed": 0,
        "benchmark_pairs_total": None,
        "eta_seconds": None,
    }
    return {
        "phase": "bootstrap-training-complete",
        "models": {"candidate": None, "incumbent": None, "opponent_pool": []},
        "training": {"active": False},
        "benchmark": {},
        "hosts": {"john1": dict(host), "john2": dict(host), "john3": dict(host)},
        "legal_next_transitions": [],
    }


def _contract(root: Path, stage: str) -> None:
    stage_name, pairs, _ = subject.STAGES[stage]
    root.mkdir()
    (root / "contract.json").write_text(
        json.dumps(
            {
                "schema_id": "cascadia.r2-map.focal-contract.v4",
                "stage": stage_name,
                "pair_count": pairs,
                "execution_partition": {"kind": "scheduler-managed-pairs"},
                "candidate_checkpoint_id": "r2-final",
                "control_checkpoint_id": "qualified-nnue",
            }
        )
    )


def _counts(stage: str, completed: int) -> dict[str, int]:
    return {
        work_item: int(index < completed)
        for index, work_item in enumerate(subject.expected_work_items(stage))
    }


def _scheduler_report(root: Path, stage: str) -> None:
    total = subject.STAGES[stage][1]
    capacities = {"john1": 9.0, "john2": 10.0, "john3": 10.0}
    report = {
        "schema_id": "cascadia.r2-map.scheduler-provenance.v1",
        "stage": stage,
        "work_items": [{"item_id": f"pair-{index:04}"} for index in range(total)],
        "retry_count": 2,
        "scheduler_utilization": {
            "sample_count": 4,
            "observed_seconds": 60.0,
            "cpu_capacity_min": 29.0,
            "cpu_capacity_max": 29.0,
            "cpu_allocated_mean": 27.0,
            "cpu_allocated_peak": 28.0,
            "cpu_utilization_mean": 27.0 / 29.0,
            "cpu_utilization_peak": 28.0 / 29.0,
            "nodes": {
                name: {
                    "cpu_capacity_min": capacity,
                    "cpu_capacity_max": capacity,
                    "cpu_allocated_mean": capacity - 1.0,
                    "cpu_allocated_peak": capacity,
                }
                for name, capacity in capacities.items()
            },
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    (root / "reports/scheduler-provenance.json").write_text(json.dumps(report))


def test_running_smoke_counts_only_receipt_names_and_stays_blinded(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _contract(root, "smoke")
    updated = subject.build_status(
        _status(),
        root=root,
        stage="smoke",
        counts=_counts("smoke", 7),
        now_ms=10_000,
        started_ms=0,
    )
    assert updated["phase"] == "r2-map-strength-blinded-smoke"
    assert updated["benchmark"]["pairs_completed"] == 7
    assert updated["benchmark"]["focal"] is None
    assert updated["benchmark"]["paired_delta"] is None
    assert updated["benchmark"]["eta_seconds"] == 19
    assert updated["benchmark"]["throughput_games_per_second"] == 1.4
    assert updated["benchmark"]["scheduler_work_items"]["completed"] == 7
    assert updated["benchmark"]["scheduler_work_items"]["total"] == 20


def test_running_dashboard_uses_scheduler_completions_before_atomic_import(
    tmp_path: Path,
) -> None:
    root = tmp_path / "smoke"
    _contract(root, "smoke")
    observations = {
        work_item: {
            "state": "completed" if index < 5 else "pending_admission",
            "attempts": 1 if index < 5 else 0,
        }
        for index, work_item in enumerate(subject.expected_work_items("smoke"))
    }
    updated = subject.build_status(
        _status(),
        root=root,
        stage="smoke",
        counts=_counts("smoke", 0),
        cluster_observation=observations,
        now_ms=10_000,
        started_ms=0,
    )
    assert updated["benchmark"]["pairs_completed"] == 5
    assert updated["benchmark"]["scheduler_work_items"]["completed"] == 5
    assert updated["benchmark"]["scheduler_work_items"]["states"] == {
        "completed": 5,
        "pending_admission": 15,
    }
    assert all(
        updated["hosts"][name]["intent"] == "benchmark"
        for name in ("john1", "john2", "john3")
    )
    assert all(
        "placement=bacalhau-managed" in updated["hosts"][name]["detail"]
        for name in ("john1", "john2", "john3")
    )


def test_complete_smoke_advances_only_when_integrity_report_passes(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _contract(root, "smoke")
    (root / "reports").mkdir()
    (root / "projections").mkdir()
    (root / "projections/dashboard-benchmark.json").write_text(
        json.dumps(
            {
                "active": False,
                "pairs_completed": 20,
                "pairs_total": 20,
                "focal": None,
                "paired_delta": None,
                "classification": "pending",
            }
        )
    )
    (root / "reports/focal-benchmark.json").write_text(
        json.dumps(
            {
                "result": {
                    "kind": "strength-blinded-smoke",
                    "statistics": {
                        "strength_outputs_blinded": True,
                        "pairs": 20,
                        "physical_games": 40,
                        "all_clean_shutdowns": True,
                        "all_pinecone_conservation_checks_passed": True,
                        "peak_rss_bytes": 1024,
                        "maximum_swap_delta_bytes": 0,
                    },
                }
            }
        )
    )
    _scheduler_report(root, "smoke")
    updated = subject.build_status(
        _status(), root=root, stage="smoke", counts=_counts("smoke", 20), now_ms=20
    )
    assert updated["legal_next_transitions"] == ["materialize-fixed-250-protected-domain"]
    assert updated["benchmark"]["focal"] is None
    scheduler = updated["benchmark"]["scheduler_work_items"]
    assert scheduler["retry_attempts"] == 2
    assert scheduler["utilization"]["cpu_allocated_peak"] == 28.0
    assert updated["stale_after_seconds"] == subject.TERMINAL_STALE_AFTER_SECONDS


def test_managed_request_projects_items_waiting_for_capacity_without_job_ids(
    tmp_path: Path, monkeypatch
) -> None:
    import cascadia_cluster.bacalhau_api as api_module

    state_dir = tmp_path / "state"
    request_dir = state_dir / "requests"
    request_dir.mkdir(parents=True)
    items = [
        {
            "key": f"pair-{index:04}",
            "index": index,
            "spec_sha256": "a" * 64,
            "job_payload": {},
            "bacalhau_job_id": "job-0" if index == 0 else None,
        }
        for index in range(20)
    ]
    state = {
        "schema_id": "cascadia.cluster.managed-request-state.v2",
        "request_id": "req-smoke",
        "image_digest": "registry/worker@sha256:" + "b" * 64,
        "experiment_id": "smoke",
        "admission": {
            "kind": "scheduler-capacity-backpressure",
            "maximum_outstanding": 6,
            "closed": False,
        },
        "items": items,
    }
    state["state_sha256"] = canonical_sha256(state)
    (request_dir / "req-smoke.json").write_text(json.dumps(state))

    class FakeAPI:
        def __init__(self, _endpoint: str) -> None:
            pass

        def get_job(self, job_id: str) -> dict:
            assert job_id == "job-0"
            return {"Job": {"State": {"StateType": "Running", "Message": ""}}}

        def executions(self, job_id: str) -> list[dict]:
            assert job_id == "job-0"
            return []

    monkeypatch.setattr(api_module, "BacalhauAPI", FakeAPI)
    observation = subject.request_observation(
        "req-smoke", state_directory=state_dir, endpoint="http://scheduler"
    )
    assert observation["pair-0000"]["state"] == "running"
    assert observation["pair-0001"] == {
        "job_id": None,
        "state": "pending_admission",
        "message": "waiting for scheduler-capacity admission",
        "attempts": 0,
    }


def test_complete_smoke_accepts_swap_decrease_as_no_swap_increase(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _contract(root, "smoke")
    (root / "reports").mkdir()
    (root / "projections").mkdir()
    (root / "projections/dashboard-benchmark.json").write_text(
        json.dumps(
            {
                "active": False,
                "pairs_completed": 20,
                "pairs_total": 20,
                "focal": None,
                "paired_delta": None,
                "classification": "pending",
            }
        )
    )
    (root / "reports/focal-benchmark.json").write_text(
        json.dumps(
            {
                "result": {
                    "kind": "strength-blinded-smoke",
                    "statistics": {
                        "strength_outputs_blinded": True,
                        "pairs": 20,
                        "physical_games": 40,
                        "all_clean_shutdowns": True,
                        "all_pinecone_conservation_checks_passed": True,
                        "peak_rss_bytes": 1024,
                        "maximum_swap_delta_bytes": -4096,
                    },
                }
            }
        )
    )
    _scheduler_report(root, "smoke")
    updated = subject.build_status(
        _status(), root=root, stage="smoke", counts=_counts("smoke", 20), now_ms=20
    )
    assert updated["legal_next_transitions"] == ["materialize-fixed-250-protected-domain"]


def test_complete_smoke_rejects_memory_over_four_gib(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _contract(root, "smoke")
    (root / "reports").mkdir()
    (root / "projections").mkdir()
    (root / "projections/dashboard-benchmark.json").write_text(
        json.dumps(
            {
                "active": False,
                "pairs_completed": 20,
                "pairs_total": 20,
                "focal": None,
                "paired_delta": None,
                "classification": "pending",
            }
        )
    )
    (root / "reports/focal-benchmark.json").write_text(
        json.dumps(
            {
                "result": {
                    "kind": "strength-blinded-smoke",
                    "statistics": {
                        "strength_outputs_blinded": True,
                        "pairs": 20,
                        "physical_games": 40,
                        "all_clean_shutdowns": True,
                        "all_pinecone_conservation_checks_passed": True,
                        "peak_rss_bytes": subject.MAX_RSS_BYTES + 1,
                        "maximum_swap_delta_bytes": 0,
                    },
                }
            }
        )
    )
    _scheduler_report(root, "smoke")
    updated = subject.build_status(
        _status(), root=root, stage="smoke", counts=_counts("smoke", 20), now_ms=20
    )
    assert updated["benchmark"]["classification"] == "invalid"
    assert updated["legal_next_transitions"] == ["stop-invalid-cross-architecture-smoke"]


def test_fixed_250_projection_is_published_only_after_complete_report(tmp_path: Path) -> None:
    root = tmp_path / "development"
    _contract(root, "development")
    (root / "reports").mkdir()
    (root / "projections").mkdir()
    projection = {
        "active": False,
        "stage": "fixed-250-development",
        "pairs_completed": 250,
        "pairs_total": 250,
        "focal": {"base_total": {"mean": 96.0}},
        "paired_delta": {"mean": 0.3, "confidence_95": [0.1, 0.5]},
        "classification": "promote",
    }
    (root / "projections/dashboard-benchmark.json").write_text(json.dumps(projection))
    (root / "reports/focal-benchmark.json").write_text(
        json.dumps({"result": {"kind": "development", "statistics": {}}})
    )
    _scheduler_report(root, "development")
    updated = subject.build_status(
        _status(),
        root=root,
        stage="development",
        counts=_counts("development", 250),
        now_ms=30,
    )
    assert updated["benchmark"]["paired_delta"]["mean"] == 0.3
    assert updated["phase"] == "r2-map-fixed-250-comparison-complete"
    assert updated["legal_next_transitions"] == []


def test_request_observation_uses_durable_jobs_without_node_affinity(
    tmp_path: Path, monkeypatch
) -> None:
    state_directory = tmp_path / "state"
    request = {
        "schema_id": "cascadia.cluster.request-state.v1",
        "request_id": "req-r2",
        "image_digest": "registry/cascadia@sha256:" + "a" * 64,
        "experiment_id": "r2-smoke",
        "items": [
            {
                "key": work_item,
                "index": index,
                "spec_sha256": str(index) * 64,
                "bacalhau_job_id": f"job-{work_item}",
            }
            for index, work_item in enumerate(subject.expected_work_items("smoke"), start=1)
        ],
    }
    request["state_sha256"] = canonical_sha256(request)
    path = state_directory / "requests/req-r2.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(request))

    class FakeAPI:
        def __init__(self, endpoint: str) -> None:
            assert endpoint == "http://scheduler"

        def get_job(self, job_id: str) -> dict:
            return {"Job": {"State": {"StateType": "Running", "Message": job_id}}}

        def executions(self, job_id: str) -> list[dict]:
            return [
                {
                    "NodeID": "node-a",
                    "ComputeState": {"StateType": "Failed"},
                },
                {
                    "NodeID": "node-b",
                    "ComputeState": {"StateType": "Completed"},
                },
            ]

    monkeypatch.setattr("cascadia_cluster.bacalhau_api.BacalhauAPI", FakeAPI)
    observation = subject.request_observation(
        "req-r2", state_directory=state_directory, endpoint="http://scheduler"
    )
    assert set(observation) == set(subject.expected_work_items("smoke"))
    assert observation["pair-0000"]["state"] == "running"
    assert observation["pair-0000"]["attempts"] == 2
