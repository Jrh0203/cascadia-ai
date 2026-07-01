from __future__ import annotations

from pathlib import Path

import cluster_legacy_freeze as subject
import cluster_research_queue as queue
import pytest


def test_idle_queue_freeze_is_content_bound_and_write_gated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "artifacts/cluster"
    root.mkdir(parents=True)
    queue_path = root / "research-queue-v1.json"
    queue._atomic_write(queue_path, queue.empty_queue("historical", now_ms=10))
    (root / "experiment-specs").mkdir()
    (root / "experiment-specs/one.json").write_text("{}\n")
    manifest = subject.freeze_legacy_queue(queue_path, freeze_id="cutover-v1", observed_unix_ms=20)
    assert manifest["summary"]["ready"] == 0
    assert manifest["summary"]["running"] == 0
    assert manifest["queue_sha256"] == manifest["snapshot_sha256"]
    assert queue.main(["--queue", str(queue_path), "status"]) == 0
    assert queue.main(["--queue", str(queue_path), "init", "--campaign-id", "new", "--force"]) == 2
    assert "legacy queue is frozen" in capsys.readouterr().err


def test_freeze_refuses_active_queue(tmp_path: Path) -> None:
    root = tmp_path / "artifacts/cluster"
    root.mkdir(parents=True)
    queue_path = root / "research-queue-v1.json"
    state = queue.empty_queue("active", now_ms=10)
    task = {
        "id": "ready",
        "title": "ready",
        "experiment_id": "active",
        "decision": "run",
        "artifact_path": "x",
        "stop_rule": "stop",
        "workload_class": "independent-experiment",
        "status": "ready",
        "priority": 1,
        "decision_value": 1.0,
        "expected_runtime_seconds": 1,
        "critical_path": False,
        "decision_terminal": False,
        "compatible_hosts": ["john1"],
        "dependencies": [],
        "command": ["true"],
        "resources": {"cpu_cores": 1, "memory_gib": 1, "uses_mlx": False},
        "claim": None,
        "attempts": [],
        "result": None,
        "created_unix_ms": 10,
    }
    queue.add_task(state, task)
    queue._atomic_write(queue_path, state)
    with pytest.raises(subject.LegacyFreezeError, match="zero ready"):
        subject.freeze_legacy_queue(queue_path, freeze_id="bad", observed_unix_ms=20)
