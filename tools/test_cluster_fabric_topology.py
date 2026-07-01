from __future__ import annotations

from pathlib import Path

import cluster_fabric_health
import r2_map_bacalhau_gate

REPOSITORY = Path(__file__).resolve().parents[1]


def test_fabric_advertises_the_registered_resource_pool() -> None:
    orchestrator = (REPOSITORY / "infra/bacalhau/orchestrator.yaml").read_text()
    compute = (REPOSITORY / "infra/bacalhau/compute.yaml.in").read_text()
    assert "CPU: 9000m" in orchestrator
    assert "Memory: 12Gi" in orchestrator
    assert "Disk: 80Gi" in orchestrator
    assert "CPU: 10000m" in compute
    assert "Memory: 15Gi" in compute
    assert "Disk: 80Gi" in compute
    assert cluster_fabric_health.EXPECTED_CPU_CAPACITY == {
        "john1": 9,
        "john2": 10,
        "john3": 10,
        "john4": 10,
    }
    assert sum(cluster_fabric_health.EXPECTED_CPU_CAPACITY.values()) == 39
    assert cluster_fabric_health.EXPECTED_MEMORY_CAPACITY_BYTES == {
        "john1": 12 * 1024**3,
        "john2": 15 * 1024**3,
        "john3": 15 * 1024**3,
        "john4": 15 * 1024**3,
    }
    assert sum(cluster_fabric_health.EXPECTED_MEMORY_CAPACITY_BYTES.values()) == 57 * 1024**3
    assert cluster_fabric_health.EXPECTED_DISK_CAPACITY_BYTES == {
        "john1": 80 * 1024**3,
        "john2": 80 * 1024**3,
        "john3": 80 * 1024**3,
        "john4": 80 * 1024**3,
    }
    assert sum(cluster_fabric_health.EXPECTED_DISK_CAPACITY_BYTES.values()) == 320 * 1024**3


def test_orchestrator_protects_transient_scheduler_evaluations() -> None:
    orchestrator = (REPOSITORY / "infra/bacalhau/orchestrator.yaml").read_text()
    assert "QueueBackoff: 10s" in orchestrator
    assert "MaxRetryCount: 1000" in orchestrator


def test_focal_gate_exposes_only_independent_pair_work_items() -> None:
    assert r2_map_bacalhau_gate.expected_work_items("smoke") == tuple(
        f"pair-{index:04}" for index in range(20)
    )
    assert len(r2_map_bacalhau_gate.expected_work_items("development")) == 250
    source = (REPOSITORY / "tools/r2_map_bacalhau_gate.py").read_text()
    assert "canonical-even-odd" not in source
    assert '"executor_shard"' in source  # fail-closed topology-field rejection
    assert "--work-item" in r2_map_bacalhau_gate.WORK_ITEM_SCRIPT
