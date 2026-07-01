from __future__ import annotations

import json
from pathlib import Path

import pytest
import v3_bootstrap_pipeline as pipeline


def _receipt() -> dict:
    return {
        "schema_id": "cascadia-v3-bootstrap-collection-completion-v1",
        "passed": True,
        "work_items": 250,
        "games": 500_000,
        "failures": [],
        "request_id": pipeline.COLLECTION_REQUEST,
        "approved_readiness_sha256": pipeline.READINESS,
    }


def test_collection_barrier_accepts_the_monitor_completion_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    completion = tmp_path / "phase2/bootstrap/collection/completion-receipt.json"
    completion.parent.mkdir(parents=True)
    completion.write_text(json.dumps(_receipt()))
    assert pipeline._wait_for_collection(1) == completion


def test_collection_barrier_rejects_incomplete_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    completion = tmp_path / "phase2/bootstrap/collection/completion-receipt.json"
    completion.parent.mkdir(parents=True)
    value = _receipt()
    value["work_items"] = 249
    completion.write_text(json.dumps(value))
    with pytest.raises(pipeline.BootstrapError):
        pipeline._wait_for_collection(1)
