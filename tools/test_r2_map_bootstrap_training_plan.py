from __future__ import annotations

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = Path("/Users/johnherrick/cascadia-bench/r2-map-v1")
PLAN = CAMPAIGN_ROOT / "control/work-packets/r2-map-bootstrap-training-plan-v1.json"
LAUNCHER = CAMPAIGN_ROOT / "control/work-packets/run-r2-map-bootstrap-training-native-v1.zsh"


def test_plan_uses_bounded_streaming_and_exact_packing_steps() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["runtime"] == "native-Apple-Silicon-MLX"
    assert plan["data_gate"] == {
        "games": 100000,
        "primary_examples": 8000000,
        "worker_datasets": 30,
        "global_game_indices": "0-99999-exactly-once",
        "worker_validation_required": True,
        "physical_shard_blake3_required": True,
    }
    streaming = plan["streaming_input"]
    assert streaming["persistent_expanded_stream"] is False
    assert streaming["maximum_prefetch_windows"] == 1
    assert streaming["maximum_simultaneous_expanded_window_bytes"] == 2 * (1 << 30)
    assert streaming["mlx_cache_limit_bytes"] == 1 << 30
    assert streaming["process_rss_stop_bytes"] == 4 * (1 << 30)
    assert plan["packing_gate"]["schedule_steps"].startswith("exact selected plan")


def test_launcher_binds_recovery_validation_and_resume_schedule() -> None:
    script = LAUNCHER.read_text()
    for required in (
        "--compact-index",
        "--compact-shard-root",
        "--maximum-window-bytes 1073741824",
        "--maximum-prefetch-windows 1",
        "--validated-aggregate-receipt",
        "--validated-packing-receipt",
        "--checkpoint-every 1000",
        "--checkpoint-seconds 300",
        "--validate-every 0",
        "--loss-event-every 20",
        "--resume-pointer last_verified",
    ):
        assert required in script
    assert "--allow-reference-expanded-streams" not in script
    assert "schedule_steps=\"$(\"$JQ\" -r '.totals.steps'" in script
    assert "for path in" not in script


def test_native_trainer_samples_resources_and_decouples_recovery_validation() -> None:
    source = (REPOSITORY / "python/cascadia_mlx/r2_map_train.py").read_text()
    assert "R2MapTrainingResourceMonitor.start()" in source
    assert "trainer.epoch > last_validated_epoch" in source
    assert "trainer.validation_metrics() if validation_due else None" in source
    assert "if validation is None:\n            continue" in source
