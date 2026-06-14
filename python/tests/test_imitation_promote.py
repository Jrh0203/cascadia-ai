from __future__ import annotations

import json
from pathlib import Path

import mlx.optimizers as optim
import pytest
from cascadia_mlx.checkpoint import (
    TrainerState,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.imitation_model import ImitationModelConfig, SharedStateActionRanker
from cascadia_mlx.imitation_promote import (
    load_promoted_imitation_model,
    promote_imitation,
)
from cascadia_mlx.promote import PromotionError


def _qualified_run(root: Path, *, passed: bool = True) -> Path:
    run = root / "run"
    run.mkdir()
    model = SharedStateActionRanker(
        ImitationModelConfig(
            hidden_dim=32,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    checkpoint = save_checkpoint(
        run,
        model,
        optim.AdamW(1e-3),
        TrainerState(global_step=3, best_ranking_loss=1.25),
    )
    validation = {"groups": 10, "listwise_loss": 1.25, "top1_accuracy": 0.5}
    set_checkpoint_pointer(
        run,
        "best",
        checkpoint,
        {"selection_loss": 1.25, "validation": validation},
    )
    (run / "run.json").write_text(
        json.dumps({"schema_version": 1, "kind": "canonical-action-imitation"})
    )
    (run / "final-report.json").write_text(
        json.dumps(
            {
                "best_ranking_loss": 1.25,
                "validation": validation,
                "initial_validation": {"selection_loss": 1.5},
            }
        )
    )
    (run / "test-report.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "passed": passed,
                "metrics": {"top1_accuracy": 0.5},
            }
        )
    )
    return run


def test_imitation_promotion_requires_and_packages_test_qualification(
    tmp_path: Path,
) -> None:
    run = _qualified_run(tmp_path)
    output = promote_imitation(run, tmp_path / "model")
    loaded = load_promoted_imitation_model(output)
    manifest = json.loads((output / "model.json").read_text())

    assert manifest["kind"] == "canonical-action-imitation"
    assert manifest["test"]["passed"] is True
    assert loaded.config.architecture == "shared-state-action-imitation-v1"


def test_imitation_promotion_rejects_failed_test_gates(tmp_path: Path) -> None:
    run = _qualified_run(tmp_path, passed=False)
    with pytest.raises(PromotionError, match="did not pass"):
        promote_imitation(run, tmp_path / "model")
