from __future__ import annotations

from pathlib import Path

import cluster_experiment_ledger as ledger
import pytest


def _experiment(experiment_id: str = "experiment-v1") -> dict[str, object]:
    return {
        "id": experiment_id,
        "title": "Experiment",
        "hypothesis": "The treatment works.",
        "summary": "The treatment is running.",
        "status": "running",
        "outcome": "pending",
        "verdict": None,
        "plan_section": "P2",
        "started_unix_ms": 1,
        "completed_unix_ms": None,
        "updated_unix_ms": 1,
        "hosts": ["john1"],
        "tags": ["representation"],
        "task_ids": ["origin"],
        "metrics": [{"label": "Recall", "value": "pending", "tone": "neutral"}],
        "criteria": [
            {
                "label": "Recall at least 95%",
                "passed": None,
                "observed": None,
            }
        ],
        "notes": ["Validation will be opened once."],
        "artifacts": [
            {
                "label": "Preregistration",
                "path": "docs/v3/TRAINING_PIPELINE.md",
            }
        ],
    }


def test_upsert_replaces_existing_experiment() -> None:
    value = ledger.empty_ledger(now_ms=1)
    ledger.upsert(value, _experiment())
    replacement = _experiment()
    replacement["summary"] = "Updated"
    ledger.upsert(value, replacement)
    assert len(value["experiments"]) == 1
    assert value["experiments"][0]["summary"] == "Updated"


def test_completed_experiment_requires_nonpending_outcome() -> None:
    experiment = _experiment()
    experiment["status"] = "completed"
    experiment["completed_unix_ms"] = 2
    with pytest.raises(ledger.LedgerError, match="outcome"):
        ledger.validate_experiment(experiment)


def test_artifacts_must_remain_inside_repository() -> None:
    experiment = _experiment()
    experiment["artifacts"][0]["path"] = "../secret"
    with pytest.raises(ledger.LedgerError, match="repository"):
        ledger.validate_experiment(experiment)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    value = ledger.empty_ledger(now_ms=1)
    ledger.upsert(value, _experiment())
    ledger.write_ledger(path, value)
    assert ledger.read_ledger(path) == value
