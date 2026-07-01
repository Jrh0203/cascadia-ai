from __future__ import annotations

import json
from pathlib import Path

import mlx.optimizers as optim
import pytest
from cascadia_mlx.checkpoint import TrainerState, save_checkpoint, set_checkpoint_pointer
from cascadia_mlx.promote import PromotionError
from cascadia_mlx.public_beam_value_model import (
    PublicBeamValueModel,
    PublicBeamValueModelConfig,
)
from cascadia_mlx.public_beam_value_promote import (
    load_promoted_public_beam_value_model,
    promote_public_beam_value,
)


def _qualified_run(root: Path, *, passed: bool = True) -> Path:
    run = root / "run"
    run.mkdir()
    model = PublicBeamValueModel(
        PublicBeamValueModelConfig(
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
        TrainerState(global_step=3, best_ranking_loss=0.5),
    )
    validation = {"validation_objective": 0.5, "top_action_agreement": 0.6}
    set_checkpoint_pointer(
        run,
        "best",
        checkpoint,
        {"selection_loss": 0.5, "validation": validation},
    )
    (run / "run.json").write_text(json.dumps({"schema_version": 1, "kind": "public-beam-value"}))
    (run / "final-report.json").write_text(
        json.dumps(
            {
                "best_ranking_loss": 0.5,
                "validation": validation,
                "initial_validation": {"selection_loss": 1.0},
            }
        )
    )
    (run / "test-report.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "passed": passed,
                "metrics": {"mean_top_action_regret": 0.2},
            }
        )
    )
    return run


def test_public_beam_value_promotion_packages_qualified_checkpoint(tmp_path: Path) -> None:
    run = _qualified_run(tmp_path)
    output = promote_public_beam_value(run, tmp_path / "model")
    loaded = load_promoted_public_beam_value_model(output)
    manifest = json.loads((output / "model.json").read_text())

    assert manifest["kind"] == "public-beam-value"
    assert manifest["test"]["passed"] is True
    assert loaded.config.architecture == "mlx-public-beam-value-v1"


def test_public_beam_value_promotion_rejects_failed_test(tmp_path: Path) -> None:
    run = _qualified_run(tmp_path, passed=False)
    with pytest.raises(PromotionError, match="did not pass"):
        promote_public_beam_value(run, tmp_path / "model")
