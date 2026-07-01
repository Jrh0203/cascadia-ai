from __future__ import annotations

import json
from pathlib import Path

import v3_checkpoint_lifecycle as lifecycle


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    for name in ("old", "latest"):
        checkpoint = run / "checkpoints" / name
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.json").write_text(json.dumps({"checkpoint": name}))
        (checkpoint / "weights.bin").write_bytes(name.encode() * 100)
    (run / "swa").mkdir()
    (run / "swa/snapshot.bin").write_bytes(b"swa" * 100)
    (run / "serving").mkdir()
    (run / "serving/model.json").write_text("{}")
    (run / "training-report.json").write_text(json.dumps({"passed": True}))
    (run / "evaluation.json").write_text(json.dumps({"passed": True}))
    (run / "latest.json").write_text(json.dumps({"checkpoint": "latest"}))
    return run


def test_compaction_retains_only_latest_exact_resume_checkpoint(tmp_path: Path) -> None:
    run = _run(tmp_path)
    receipt = lifecycle.compact_completed_run(run)
    assert receipt["bytes_reclaimed"] > 0
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == ["latest"]
    assert not (run / "swa").exists()
    assert (run / "latest.json").is_file()


def test_retirement_preserves_reports_and_serving_but_removes_optimizer_state(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    receipt = lifecycle.retire_completed_run(run, reason="candidate-not-selected")
    assert receipt["scientific_outputs_preserved"] is True
    assert not (run / "checkpoints").exists()
    assert not (run / "latest.json").exists()
    assert (run / "training-report.json").is_file()
    assert (run / "evaluation.json").is_file()
    assert (run / "serving/model.json").is_file()


def test_rejected_retirement_preserves_failure_evidence(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "evaluation.json").unlink()
    (run / "evaluation-failure.json").write_text(json.dumps({"passed": False}))

    receipt = lifecycle.retire_rejected_run(run, reason="serving-overflow")

    assert receipt["passed"] is True
    assert receipt["evaluation_passed"] is False
    assert not (run / "checkpoints").exists()
    assert not (run / "latest.json").exists()
    assert (run / "evaluation-failure.json").is_file()
    assert (run / "serving/model.json").is_file()
