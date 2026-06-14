from __future__ import annotations

import json
from pathlib import Path

import mlx.optimizers as optim
from cascadia_mlx.checkpoint import TrainerState, save_checkpoint, set_checkpoint_pointer
from cascadia_mlx.ranking_model import EntitySetRanker, RankingModelConfig
from cascadia_mlx.ranking_promote import load_promoted_ranking_model, promote_ranking


def test_ranking_promotion_packages_best_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    model = EntitySetRanker(
        RankingModelConfig(
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
    validation = {"groups": 10, "listwise_loss": 1.25, "top1_accuracy": 0.4}
    set_checkpoint_pointer(run, "best", checkpoint, {"validation": validation})
    (run / "run.json").write_text(json.dumps({"schema_version": 1}))
    (run / "final-report.json").write_text(
        json.dumps({"best_ranking_loss": 1.25, "validation": validation})
    )

    output = promote_ranking(run, tmp_path / "model")
    loaded = load_promoted_ranking_model(output)
    manifest = json.loads((output / "model.json").read_text())

    assert manifest["kind"] == "action-ranking"
    assert manifest["best_ranking_loss"] == 1.25
    assert loaded.config.architecture == "entity-set-ranker-v1"
