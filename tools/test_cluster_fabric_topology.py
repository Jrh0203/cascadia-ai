from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_fabric_templates_advertise_useful_resources() -> None:
    orchestrator = (REPOSITORY / "infra/bacalhau/orchestrator.yaml").read_text()
    compute = (REPOSITORY / "infra/bacalhau/compute.yaml.in").read_text()
    assert "CPU: 9000m" in orchestrator
    assert "Memory: 12Gi" in orchestrator
    assert "Disk: 80Gi" in orchestrator
    assert "CPU: 10000m" in compute
    assert "Memory: 15Gi" in compute
    assert "Disk: 80Gi" in compute


def test_orchestrator_protects_transient_scheduler_evaluations() -> None:
    orchestrator = (REPOSITORY / "infra/bacalhau/orchestrator.yaml").read_text()
    assert "QueueBackoff: 10s" in orchestrator
    assert "MaxRetryCount: 1000" in orchestrator
