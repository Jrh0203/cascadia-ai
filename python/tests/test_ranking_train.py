from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from cascadia_mlx.checkpoint import TrainerState
from cascadia_mlx.ranking_train import _completed_resume_report, _top1_metrics


def test_top1_value_recall_accepts_an_equally_valued_teacher_action() -> None:
    strict, value_recall, regret = _top1_metrics(
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        np.array([9.0, 10.0, 10.0], dtype=np.float32),
    )

    assert strict == 0
    assert value_recall == 1
    assert regret == 0.0


def test_top1_metrics_report_teacher_regret() -> None:
    strict, value_recall, regret = _top1_metrics(
        np.array([2.0, 1.0], dtype=np.float32),
        np.array([4.5, 6.0], dtype=np.float32),
    )

    assert strict == 0
    assert value_recall == 0
    assert regret == 1.5


def test_completed_resume_preserves_authoritative_final_report(tmp_path) -> None:
    report = {
        "epochs": 11,
        "global_step": 3520,
        "best_ranking_loss": 1.5,
        "best_top1_accuracy": 0.25,
        "elapsed_seconds": 74.3,
    }
    report_path = tmp_path / "final-report.json"
    report_path.write_text(json.dumps(report))
    config = SimpleNamespace(
        resume=True,
        validation_patience=5,
        epochs=20,
        run_dir=tmp_path,
    )
    state = TrainerState(
        epoch=11,
        global_step=3520,
        best_ranking_loss=1.5,
        best_top1_accuracy=0.25,
        ranking_epochs_without_improvement=5,
    )

    assert _completed_resume_report(config, state) == report
    assert json.loads(report_path.read_text())["elapsed_seconds"] == 74.3
