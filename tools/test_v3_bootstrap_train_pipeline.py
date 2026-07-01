from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import v3_bootstrap_train_pipeline as pipeline


def test_paired_nonregression_accepts_uniform_improvement() -> None:
    result = pipeline.paired_nonregression([11, 12, 13], [10, 10, 10])
    assert result["passed"] is True
    assert result["mean_delta"] == 2.0


def test_paired_nonregression_rejects_clear_regression() -> None:
    result = pipeline.paired_nonregression([1, 1, 1], [10, 10, 10])
    assert result["passed"] is False


def test_paired_nonregression_requires_exact_pairs() -> None:
    with pytest.raises(pipeline.BootstrapTrainingError):
        pipeline.paired_nonregression([1, 2], [1])


def test_selection_uses_loss_only_among_open_nonregressing_candidates() -> None:
    candidates = [
        {
            "label": "open-best",
            "open_game_mean": 10.0,
            "open_scores": [10, 10, 10],
            "quantized_validation_loss": 2.0,
        },
        {
            "label": "loss-best-but-regressed",
            "open_game_mean": 0.0,
            "open_scores": [0, 0, 0],
            "quantized_validation_loss": 0.1,
        },
    ]
    assert pipeline._select(candidates, "test")["label"] == "open-best"


def test_selection_excludes_checkpoint_evaluation_failure() -> None:
    candidates = [
        {
            "label": "overflowed",
            "evaluation_passed": False,
        },
        {
            "label": "valid",
            "evaluation_passed": True,
            "open_game_mean": 10.0,
            "open_scores": [10, 10, 10],
            "quantized_validation_loss": 0.2,
        },
    ]
    assert pipeline._select(candidates, "test")["label"] == "valid"
    assert candidates[0]["eligible"] is False


def test_tolerated_evaluation_failure_is_durable_and_reused(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    calls = 0

    def fail(command: list[str], log: Path) -> None:
        nonlocal calls
        calls += 1
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Error: AccumulatorOverflow\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline, "_run", fail)
    first = pipeline._evaluate(
        run_dir=run,
        origin="calibration-overflow",
        validation=[],
        games=2,
        tolerate_failure=True,
    )
    second = pipeline._evaluate(
        run_dir=run,
        origin="calibration-overflow",
        validation=[],
        games=2,
        tolerate_failure=True,
    )
    assert first == second
    assert first["passed"] is False
    assert first["log_blake3"] == pipeline._digest(run / "evaluation.log")
    assert calls == 1


def test_rejected_candidate_uses_rejected_retirement(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, Path, str]] = []
    monkeypatch.setattr(
        pipeline,
        "retire_rejected_run",
        lambda run, *, reason: calls.append(("rejected", run, reason)) or {"passed": True},
    )
    monkeypatch.setattr(
        pipeline,
        "retire_completed_run",
        lambda run, *, reason: calls.append(("completed", run, reason)) or {"passed": True},
    )
    candidate = {"run_dir": str(tmp_path / "run"), "evaluation_passed": False}

    pipeline._retire_candidate(candidate, reason="calibration-complete")

    assert calls == [("rejected", tmp_path / "run", "calibration-complete")]


def test_controller_lock_rejects_duplicate_launch(tmp_path: Path) -> None:
    path = tmp_path / "controller.lock"
    first = pipeline._acquire_controller_lock(path)
    assert first is not None
    try:
        assert pipeline._acquire_controller_lock(path) is None
    finally:
        first.close()
    replacement = pipeline._acquire_controller_lock(path)
    assert replacement is not None
    replacement.close()


def test_final_worker_publication_is_reused_only_for_exact_source(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = tmp_path / "bootstrap.json"
    evidence.write_text('{"passed": true}')
    publication = tmp_path / "publication.json"
    health = tmp_path / "health.json"
    handoff = tmp_path / "handoff.json"
    source = {"workspace_blake3": "source", "files": 3}
    image = "registry/cascadia/v3-worker@sha256:" + "a" * 64
    publication.write_text(
        json.dumps(
            {
                "schema_id": "cascadia.cluster.image-publication.v1",
                "source_identity": source,
                "image_digest": image,
            }
        )
    )
    health.write_text(json.dumps({"passed": True, "image_digest": image}))
    monkeypatch.setattr(pipeline, "FINAL_IMAGE_RECEIPT", publication)
    monkeypatch.setattr(pipeline, "FINAL_IMAGE_HEALTH", health)
    monkeypatch.setattr(pipeline, "FINAL_HANDOFF", handoff)
    monkeypatch.setattr(pipeline, "_workspace_source_identity", lambda _path: source)
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("exact publication should be reused"),
    )

    selected, output = pipeline._publish_final_worker(evidence)

    assert selected == image
    assert output == handoff
    value = json.loads(handoff.read_text())
    assert value["passed"] is True
    assert value["bootstrap_evidence_blake3"] == pipeline._digest(evidence)


def test_training_command_binds_exact_calibration_stop(tmp_path: Path) -> None:
    command = pipeline._training_command(
        run_dir=tmp_path / "run",
        origin="calibration-1",
        seed=83_000,
        learning_rate=5e-4,
        broad=[tmp_path / "broad.v3g"],
        teacher=[tmp_path / "teacher.v3l"],
        planned_stop=4_000_000,
    )
    assert command[command.index("--planned-stop-after-examples") + 1] == "4000000"
    assert command[command.index("--broad-dataset") + 1].endswith("broad.v3g")
    assert command[command.index("--teacher-dataset") + 1].endswith("teacher.v3l")
