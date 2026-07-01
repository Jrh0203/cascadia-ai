from __future__ import annotations

import json
from pathlib import Path

import pytest
import v3_campaign as campaign


def _state(root: Path) -> dict:
    return json.loads((root / "control" / "campaign-state.json").read_text())


def test_storage_guard_remeasures_after_sparse_worker_trim(tmp_path: Path, monkeypatch) -> None:
    below = {
        "campaign_bytes": 1,
        "campaign_limit_bytes": 2,
        "free_bytes": 49,
        "minimum_free_bytes": 50,
        "within_campaign_limit": True,
        "free_space_preserved": False,
    }
    recovered = below | {"free_bytes": 51, "free_space_preserved": True}
    observations = iter((below, recovered))
    trims: list[bool] = []
    monkeypatch.setattr(campaign, "_storage", lambda _root: next(observations))
    monkeypatch.setattr(campaign, "_trim_sparse_worker_disk", lambda: trims.append(True) or True)

    assert campaign._assert_storage(tmp_path) == recovered
    assert trims == [True]


def test_storage_guard_still_refuses_real_low_space_after_trim(
    tmp_path: Path, monkeypatch
) -> None:
    below = {
        "campaign_bytes": 1,
        "campaign_limit_bytes": 2,
        "free_bytes": 49,
        "minimum_free_bytes": 50,
        "within_campaign_limit": True,
        "free_space_preserved": False,
    }
    monkeypatch.setattr(campaign, "_storage", lambda _root: below)
    monkeypatch.setattr(campaign, "_trim_sparse_worker_disk", lambda: True)

    with pytest.raises(campaign.CampaignError, match="storage guard refused mutation"):
        campaign._assert_storage(tmp_path)


def test_phase2_is_impossible_before_checksum_bound_approval(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "v3-nnue"
    dashboard = tmp_path / "dashboard.json"
    monkeypatch.setattr(campaign, "DEFAULT_ROOT", root)
    monkeypatch.setattr(campaign, "MIN_FREE_BYTES", 0)
    campaign.initialize(root, dashboard)
    assert _state(root)["phase"] == "part1_engineering"
    with pytest.raises(campaign.CampaignError, match="sealed"):
        campaign.phase2_plan(root)


def test_dashboard_names_all_four_hosts_and_excludes_john4(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "v3-nnue"
    dashboard = tmp_path / "dashboard.json"
    monkeypatch.setattr(campaign, "DEFAULT_ROOT", root)
    monkeypatch.setattr(campaign, "MIN_FREE_BYTES", 0)
    state = campaign.initialize(root, dashboard)
    value = campaign._dashboard(state)
    assert set(value["hosts"]) == {"john1", "john2", "john3", "john4"}
    assert "excluded" in value["hosts"]["john4"]["detail"].lower()
    assert state["topology"]["cpu_workers"] == {"john1": 9, "john2": 10, "john3": 10}
    transitions = sorted((root / "control" / "transitions").glob("*.json"))
    assert len(transitions) == 1
    assert json.loads(transitions[0].read_text())["state_sha256"] == state["state_sha256"]


def test_live_collection_progress_overlays_without_mutating_state(tmp_path: Path) -> None:
    root = tmp_path / "v3-nnue"
    progress = root / "phase2/bootstrap/collection/progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps(
            {
                "work_items": 250,
                "terminal_items": 75,
                "elapsed_seconds": 1500.0,
                "status_counts": {"succeeded": 75, "running": 18, "queued": 157},
            }
        )
    )
    status = campaign._dashboard(
        {
            "phase": "bootstrap_collecting",
            "detail": "bootstrap collecting",
            "legal_next_transitions": ["bootstrap_labeling"],
        }
    )
    campaign._overlay_live_progress(root, status)
    completed = sum(
        status["hosts"][name]["generation_games_completed"]
        for name in ("john1", "john2", "john3")
    )
    assert completed == 150_000
    assert status["benchmark"]["scheduler_work_items"]["completed"] == 75
    assert status["hosts"]["john4"]["generation_games_completed"] == 0


def test_promotion_overlay_includes_completed_prior_increments(tmp_path: Path) -> None:
    root = tmp_path / "v3-nnue"
    progress = root / "phase2/cycles/cycle-01/promotion/progress-300-400.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps(
            {
                "work_items": 80,
                "terminal_items": 4,
                "elapsed_seconds": 10.0,
                "status_counts": {"succeeded": 4, "running": 28, "queued": 48},
            }
        )
    )
    status = campaign._dashboard(
        {
            "phase": "cycle-01-promotion",
            "detail": "cycle 01 promotion",
            "legal_next_transitions": ["cycle-02-collecting"],
        }
    )
    campaign._overlay_live_progress(root, status)
    hosts = status["hosts"]
    completed = sum(
        hosts[name]["generation_games_completed"]
        for name in ("john1", "john2", "john3")
    )
    assert completed == 1_220
    assert sum(
        hosts[name]["generation_games_target"]
        for name in ("john1", "john2", "john3")
    ) == 1_600
    assert hosts["john1"]["throughput_games_per_second"] == 2.0


def test_training_overlay_emits_dashboard_contract_model_and_loss_shapes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3-nnue"
    run = root / "phase2/bootstrap/training/calibration/calibration-1-lr-0.0005"
    checkpoint = "step-000000002-epoch-0000-batch-000000"
    checkpoint_dir = run / "checkpoints" / checkpoint
    checkpoint_dir.mkdir(parents=True)
    (run / "training-report.json").write_text(
        json.dumps({"examples_seen": 16_384, "elapsed_seconds": 4.0})
    )
    (run / "loss.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"examples": 8192, "loss": 0.2},
                    {"examples": 16_384, "loss": 0.1},
                ]
            }
        )
    )
    (run / "evaluation.json").write_text(
        json.dumps({"passed": True, "validation": {"quantized_power_loss": 0.08}})
    )
    (run / "latest.json").write_text(json.dumps({"checkpoint": checkpoint}))
    (checkpoint_dir / "state.json").write_text(
        json.dumps({"examples_seen": 16_384, "elapsed_seconds": 4.0})
    )
    digest = "a" * 64
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {"files": {"model.safetensors": {"blake3": digest, "bytes": 1}}}
        )
    )
    status = campaign._dashboard(
        {
            "phase": "bootstrap_training",
            "detail": "bootstrap training",
            "legal_next_transitions": ["cycle-01-collecting"],
            "runtime": {"host_intents": {"john1": "train"}},
        }
    )
    campaign._overlay_training_progress(root, status, "bootstrap_training")
    assert status["training"]["latest_verified_checkpoint"] == {
        "id": f"calibration-1-lr-0.0005/{checkpoint}",
        "blake3": digest,
    }
    assert status["training"]["loss_samples"] == [
        {"step": 8192, "train_total": 0.2, "validation_total": None},
        {"step": 16_384, "train_total": 0.1, "validation_total": 0.08},
    ]


def test_training_overlay_does_not_publish_failed_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "v3-nnue"
    calibration_root = root / "phase2/bootstrap/training/calibration"
    valid = calibration_root / "calibration-1-lr-0.0005"
    failed = calibration_root / "calibration-2-lr-0.001"
    checkpoint = "step-000000001-epoch-0000-batch-000000"
    for run, passed, digest in ((valid, True, "a" * 64), (failed, False, "b" * 64)):
        checkpoint_dir = run / "checkpoints" / checkpoint
        checkpoint_dir.mkdir(parents=True)
        (run / "latest.json").write_text(json.dumps({"checkpoint": checkpoint}))
        (run / "evaluation.json").write_text(json.dumps({"passed": passed}))
        (checkpoint_dir / "state.json").write_text(
            json.dumps({"examples_seen": 8_192, "elapsed_seconds": 2.0})
        )
        (checkpoint_dir / "checkpoint.json").write_text(
            json.dumps({"files": {"model.safetensors": {"blake3": digest}}})
        )
    status = campaign._dashboard(
        {
            "phase": "bootstrap_training",
            "detail": "bootstrap training",
            "legal_next_transitions": ["cycle-01-collecting"],
            "runtime": {"host_intents": {"john1": "train"}},
        }
    )
    campaign._overlay_training_progress(root, status, "bootstrap_training")
    assert status["training"]["latest_verified_checkpoint"] == {
        "id": f"calibration-1-lr-0.0005/{checkpoint}",
        "blake3": "a" * 64,
    }


def test_training_overlay_publishes_checkpoint_after_serving_integrity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3-nnue"
    run = root / "phase2/bootstrap/training/origins/bootstrap-origin-1"
    checkpoint = "step-000000610-epoch-0000-batch-000000"
    checkpoint_dir = run / "checkpoints" / checkpoint
    checkpoint_dir.mkdir(parents=True)
    examples = 5_000_000
    (run / "latest.json").write_text(json.dumps({"checkpoint": checkpoint}))
    (checkpoint_dir / "state.json").write_text(
        json.dumps({"examples_seen": examples, "elapsed_seconds": 1_000.0})
    )
    digest = "c" * 64
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps({"files": {"model.safetensors": {"blake3": digest}}})
    )
    integrity = run / "checkpoint-integrity"
    integrity.mkdir()
    (integrity / f"{examples:012d}.json").write_text(json.dumps({"passed": True}))
    status = campaign._dashboard(
        {
            "phase": "bootstrap_training",
            "detail": "bootstrap training",
            "legal_next_transitions": ["cycle-01-collecting"],
            "runtime": {"host_intents": {"john1": "train"}},
        }
    )

    campaign._overlay_training_progress(root, status, "bootstrap_training")

    assert status["training"]["latest_verified_checkpoint"] == {
        "id": f"bootstrap-origin-1/{checkpoint}",
        "blake3": digest,
    }


def _atomic_cycle_checkpoint(run: Path, *, model_bytes: int = 5) -> tuple[str, str]:
    checkpoint = "step-000000050-epoch-0000-batch-000000"
    directory = run / "checkpoints" / checkpoint
    directory.mkdir(parents=True)
    (run / "latest.json").write_text(json.dumps({"checkpoint": checkpoint}))
    run_digest = "d" * 64
    (run / "run-manifest.json").write_text(
        json.dumps({"canonical_blake3": run_digest})
    )
    state = {
        "examples_seen": 400_000,
        "elapsed_seconds": 2_000.0,
        "schedule_block": 1,
        "batch_in_block": 0,
        "batch_in_epoch": 0,
    }
    payloads = {
        "model.safetensors": b"m" * model_bytes,
        "optimizer.safetensors": b"optimizer",
        "state.json": json.dumps(state).encode(),
    }
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    model_digest = "a" * 64
    (directory / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_id": checkpoint,
                "files": {
                    name: {
                        "bytes": len(payload),
                        "blake3": model_digest if name == "model.safetensors" else "b" * 64,
                    }
                    for name, payload in payloads.items()
                },
                "metadata": {
                    "examples_seen": 400_000,
                    "completed_pass": 1,
                    "run_manifest_blake3": run_digest,
                },
            }
        )
    )
    (run / "loss.json").write_text(
        json.dumps({"samples": [{"examples": 400_000, "loss": 0.001}]})
    )
    return checkpoint, model_digest


def test_training_overlay_publishes_structurally_verified_atomic_cycle_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3-nnue"
    run = root / "phase2/cycles/cycle-04/training/origin-1"
    checkpoint, digest = _atomic_cycle_checkpoint(run)
    status = campaign._dashboard(
        {
            "phase": "cycle-04-training",
            "detail": "cycle 04 training",
            "legal_next_transitions": ["cycle-04-promotion"],
            "runtime": {"host_intents": {"john1": "train"}},
        }
    )

    campaign._overlay_training_progress(root, status, "cycle-04-training")

    assert status["training"]["latest_verified_checkpoint"] == {
        "id": f"origin-1/{checkpoint}",
        "blake3": digest,
    }
    assert status["training"]["examples_per_second"] == 200.0
    assert status["training"]["eta_seconds"] == 10_000.0


def test_training_overlay_refuses_atomic_cycle_checkpoint_with_size_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3-nnue"
    run = root / "phase2/cycles/cycle-04/training/origin-1"
    checkpoint, _ = _atomic_cycle_checkpoint(run)
    (run / "checkpoints" / checkpoint / "model.safetensors").write_bytes(b"corrupt")
    status = campaign._dashboard(
        {
            "phase": "cycle-04-training",
            "detail": "cycle 04 training",
            "legal_next_transitions": ["cycle-04-promotion"],
            "runtime": {"host_intents": {"john1": "train"}},
        }
    )

    campaign._overlay_training_progress(root, status, "cycle-04-training")

    assert status["training"]["latest_verified_checkpoint"] is None


def test_readiness_checksum_cannot_be_forged(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "v3-nnue"
    dashboard = tmp_path / "dashboard.json"
    monkeypatch.setattr(campaign, "DEFAULT_ROOT", root)
    monkeypatch.setattr(campaign, "MIN_FREE_BYTES", 0)
    campaign.initialize(root, dashboard)
    state = _state(root)
    readiness = {
        "schema_id": campaign.READINESS_SCHEMA,
        "campaign_id": campaign.CAMPAIGN_ID,
        "status": "green",
    }
    readiness["readiness_sha256"] = campaign._sha256(readiness)
    path = root / "reports" / "part1-readiness.json"
    campaign._write_json_atomic(path, readiness)
    state.update(
        {
            "phase": "awaiting_phase2_approval",
            "readiness_path": str(path),
            "readiness_sha256": readiness["readiness_sha256"],
        }
    )
    campaign._write_json_atomic(campaign._state_path(root), state)
    with pytest.raises(campaign.CampaignError, match="checksum"):
        campaign.authorize_phase2(root, dashboard, "0" * 64, "John", False)


def test_projected_write_is_refused_before_crossing_campaign_limit(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "v3-nnue"
    root.mkdir()
    monkeypatch.setattr(campaign, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(campaign, "MAX_BYTES", 1024)
    (root / "existing").write_bytes(b"x" * 900)
    with pytest.raises(campaign.CampaignError, match="projected write"):
        campaign.assert_capacity_for_write(root, 125)
    assert not (root / "new-artifact").exists()


def test_passing_part1_evidence_seals_at_the_human_gate(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "v3-nnue"
    dashboard = tmp_path / "dashboard.json"
    monkeypatch.setattr(campaign, "DEFAULT_ROOT", root)
    monkeypatch.setattr(campaign, "MIN_FREE_BYTES", 0)
    campaign.initialize(root, dashboard)

    values = {
        "feature_manifest": {},
        "model_manifest": {},
        "mlx_profile_1": {"examples_per_second": 1.0},
        "mlx_profile_2": {
            "examples_per_second": 2.0,
            "peak_memory_bytes": 6,
            "physical_memory_bytes": 10,
            "swap_delta_bytes": 0,
        },
        "game_profile_1": {"decisions_per_second": 1.0},
        "game_profile_2": {"decisions_per_second": 2.0},
        "game_profile_2_late": {"radius7_hot_path": [True]},
        "game_profile_2_overflow": {
            "radius7_hot_path": [False],
            "overflow_entities": [1],
        },
        "engineering_corpus": {
            "games": 2_000,
            "records": 160_000,
            "scientific_eligible": False,
            "hot_path_fraction": 0.999,
        },
        "training_smoke": {
            "training_config": {"examples": 160_000},
            "state": {"epoch": 1},
            "metrics": {"interrupted": False},
        },
        "direct_smoke": {"games": 100, "swap_delta_bytes": 0},
        "r600_smoke": {
            "games": 8,
            "r600_seconds_per_game": 44.0,
            "swap_delta_bytes": 0,
        },
        "parity": {
            "overflow_exact": True,
            "rust_mlx_quantized_bit_identical": True,
            "float_quantized_top32_agreement": 0.999,
        },
        "docker_receipt": {"passed": True},
        "recovery_receipt": {"checkpoint_exact_continuation": True},
        "infrastructure_receipt": {
            "bacalhau_worker_retry": True,
            "john1_trainer_restart": True,
            "dashboard_hosts": ["john1", "john2", "john3", "john4"],
            "disk_limit": True,
            "memory_limit": True,
            "clean_shutdown": True,
            "scientific_data_guard": True,
        },
        "capacity_projection": {
            "active_wall_seconds": 6 * 86_400,
            "projected_campaign_bytes": 20 * 1024**3,
            "protected_seed_values_opened": False,
        },
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    readiness = campaign.qualify(root, dashboard, **paths)
    state = _state(root)
    assert readiness["status"] == "green"
    assert state["phase"] == "awaiting_phase2_approval"
    assert state["scientific_training_started"] is False
    assert state["protected_seed_values_opened"] is False
    assert state["legal_next_transitions"] == ["authorize_phase2"]
    dashboard_status = json.loads(json.loads(dashboard.read_text())["canonical_payload"])
    assert dashboard_status["hosts"]["john1"]["detail"] == (
        "Part 1 complete — awaiting John"
    )
