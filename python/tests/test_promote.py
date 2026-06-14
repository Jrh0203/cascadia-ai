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
from cascadia_mlx.model import EntitySetValueModel, ModelConfig
from cascadia_mlx.promote import PromotionError, load_promoted_model, promote


def _complete_run(root: Path) -> None:
    model = EntitySetValueModel(
        ModelConfig(hidden_dim=32, attention_heads=4, board_blocks=0, market_blocks=0)
    )
    optimizer = optim.AdamW(1e-3)
    checkpoint = save_checkpoint(root, model, optimizer, TrainerState(global_step=3))
    set_checkpoint_pointer(
        root,
        "best",
        checkpoint,
        {"validation": {"samples": 10, "total_mae": 4.25}},
    )
    (root / "run.json").write_text(
        json.dumps({"schema_version": 1, "source": {"v2_source_blake3": "test"}})
    )
    (root / "final-report.json").write_text(
        json.dumps(
            {
                "best_validation_mae": 4.25,
                "validation": {"samples": 10, "total_mae": 4.25},
            }
        )
    )


def test_promotion_packages_best_checkpoint_atomically(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _complete_run(run)

    output = promote(run, tmp_path / "models" / "candidate")
    manifest = json.loads((output / "model.json").read_text())

    assert manifest["status"] == "promoted"
    assert manifest["best_validation_mae"] == 4.25
    assert (output / "model.safetensors").is_file()
    loaded = load_promoted_model(output)
    assert loaded.config.architecture == "entity-set-value-v1"


def test_promotion_refuses_to_overwrite(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _complete_run(run)
    output = tmp_path / "model"
    promote(run, output)

    with pytest.raises(PromotionError, match="already exists"):
        promote(run, output)
