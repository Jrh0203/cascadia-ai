from __future__ import annotations

import json
from pathlib import Path

import r2_map_bootstrap_dashboard_watch as subject


def _status() -> dict:
    return {
        "phase": "bootstrap-generation",
        "hosts": {
            "john1": {"generation_games_completed": 256},
            "john2": {"generation_games_completed": 512},
            "john3": {"generation_games_completed": 768},
        },
        "benchmark": {},
        "legal_next_transitions": [],
    }


def test_read_completed_games_validates_lease_and_targets(tmp_path: Path) -> None:
    for worker, target in enumerate(subject.WORKER_TARGETS[:2]):
        directory = tmp_path / f"worker-{worker}"
        directory.mkdir()
        (directory / "dataset.json").write_text(
            json.dumps(
                {
                    "completed_games": 256,
                    "lease": {"host_id": "john1", "game_count": target},
                }
            )
        )
    completed, readable, rate = subject.read_completed_games(tmp_path)
    assert completed == 512
    assert readable == 2
    assert rate is None


def test_update_status_is_monotonic_and_sums_hosts(monkeypatch) -> None:
    monkeypatch.setattr(subject, "aggregate_is_complete", lambda: False)
    monkeypatch.setattr(subject, "read_returned_host", lambda _host: (0, 0))
    updated = subject.update_status(
        _status(),
        observed_completed=128,
        readable_manifests=10,
        states={"running": 10, "successful": 0, "failed": 0, "unknown": 0},
        now_ms=123,
    )
    assert updated["hosts"]["john1"]["generation_games_completed"] == 256
    assert updated["benchmark"]["pairs_completed"] == 1536
    assert updated["phase"] == "bootstrap-generation"
    assert updated["updated_unix_ms"] == 123


def test_update_status_marks_complete(monkeypatch) -> None:
    monkeypatch.setattr(subject, "aggregate_is_complete", lambda: False)
    status = _status()
    status["hosts"]["john2"]["generation_games_completed"] = 33333
    status["hosts"]["john3"]["generation_games_completed"] = 33333
    updated = subject.update_status(
        status,
        observed_completed=33334,
        readable_manifests=10,
        states={"running": 0, "successful": 10, "failed": 0, "unknown": 0},
        now_ms=456,
    )
    assert updated["benchmark"]["pairs_completed"] == 100000
    assert updated["benchmark"]["active"] is False
    assert updated["phase"] == "bootstrap-generation-complete"


def test_memory_unit_conversion() -> None:
    assert subject._memory_to_bytes("20.5MiB") == 20.5 * (1 << 20)
    assert subject._memory_to_bytes("1GiB") == 1 << 30
    assert subject._memory_to_bytes("bad") == 0


def test_bootstrap_watcher_yields_after_focal_campaign_owns_dashboard() -> None:
    assert subject.focal_campaign_owns_status(
        {"phase": "r2-map-strength-blinded-smoke"}
    )
    assert subject.focal_campaign_owns_status(
        {"phase": "r2-map-fixed-250-comparison-complete"}
    )
    assert not subject.focal_campaign_owns_status({"phase": "bootstrap-training"})


def test_read_training_state_surfaces_loss_and_verified_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    losses = tmp_path / "losses"
    losses.mkdir()
    (losses / "loss-stream.jsonl").write_text(
        json.dumps(
            {
                "global_step": 20,
                "metrics": {
                    "total_loss": 3.5,
                    "primary_score_to_go_loss": 2.5,
                },
            }
        )
        + "\n"
    )
    (tmp_path / "last_verified.json").write_text(
        json.dumps({"checkpoint": "r2-map-bootstrap-iteration0-v1.main.step-000001000"})
    )
    (tmp_path / "resource-baseline.json").write_text(json.dumps({"system_swap_bytes": 1000}))
    monkeypatch.setattr(subject, "TRAINING_STEPS", 100)
    monkeypatch.setattr(subject, "TRAINING_GROUP_BATCH_SIZE", 256)
    monkeypatch.setattr(
        subject,
        "_training_process_metrics",
        lambda: {"pid": 1, "rss_bytes": 4096, "elapsed_seconds": 10},
    )
    monkeypatch.setattr(subject, "_system_swap_bytes", lambda: 1000)
    state = subject.read_training_state(tmp_path, process_active=True)
    assert state["active"] is True
    assert state["current_step"] == 20
    assert state["latest_verified_checkpoint"] == {
        "id": "r2-map-bootstrap-iteration0-v1.main.step-000001000",
        "blake3": None,
    }
    assert state["loss_samples"] == [{"step": 20, "train_total": 3.5, "validation_total": None}]
    assert state["rss_bytes"] == 4096
    assert state["swap_delta_bytes"] == 0
    assert state["examples_per_second"] == 512.0
    assert state["eta_seconds"] == 40.0


def test_training_state_compacts_overlapping_resume_branch_steps(tmp_path: Path) -> None:
    losses = tmp_path / "losses"
    losses.mkdir()
    (losses / "loss-stream.jsonl").write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"global_step": 20, "branch_id": "main", "metrics": {"total_loss": 3.5}},
                {"global_step": 40, "branch_id": "main", "metrics": {"total_loss": 3.0}},
                {
                    "global_step": 40,
                    "branch_id": "resume-step-20-1",
                    "metrics": {"total_loss": 2.9},
                },
                {
                    "global_step": 60,
                    "branch_id": "resume-step-20-1",
                    "metrics": {"total_loss": 2.5},
                },
            )
        )
        + "\n"
    )
    state = subject.read_training_state(tmp_path, process_active=False)
    assert state["loss_samples"] == [
        {"step": 20, "train_total": 3.5, "validation_total": None},
        {"step": 40, "train_total": 2.9, "validation_total": None},
        {"step": 60, "train_total": 2.5, "validation_total": None},
    ]


def test_completed_training_uses_receipt_final_step_not_sparse_loss_step(
    tmp_path: Path, monkeypatch
) -> None:
    losses = tmp_path / "losses"
    losses.mkdir()
    (losses / "loss-stream.jsonl").write_text(
        json.dumps({"global_step": 80, "metrics": {"total_loss": 1.0}}) + "\n"
    )
    (tmp_path / "training-command-receipt.json").write_text(
        json.dumps(
            {
                "schema_id": "r2-map-training-command-receipt-v1",
                "run_id": subject.TRAINING_RUN_ID,
                "final_step": 100,
            }
        )
    )
    monkeypatch.setattr(subject, "TRAINING_STEPS", 100)
    state = subject.read_training_state(tmp_path, process_active=False)
    assert state["complete"] is True
    assert state["current_step"] == 100


def test_training_eta_uses_only_steps_completed_by_the_active_resume_process(
    tmp_path: Path, monkeypatch
) -> None:
    losses = tmp_path / "losses"
    losses.mkdir()
    (losses / "loss-stream.jsonl").write_text(
        json.dumps(
            {
                "global_step": 60,
                "branch_id": "resume-step-20-1",
                "metrics": {"total_loss": 2.5},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(subject, "TRAINING_STEPS", 100)
    monkeypatch.setattr(subject, "TRAINING_GROUP_BATCH_SIZE", 256)
    monkeypatch.setattr(
        subject,
        "_training_process_metrics",
        lambda: {"pid": 1, "rss_bytes": 4096, "elapsed_seconds": 10},
    )
    monkeypatch.setattr(subject, "_system_swap_bytes", lambda: 0)
    state = subject.read_training_state(tmp_path, process_active=True)
    assert state["examples_per_second"] == 1024.0
    assert state["eta_seconds"] == 10.0


def test_active_recovery_branch_hides_unverified_failed_branch_tail(
    tmp_path: Path, monkeypatch
) -> None:
    losses = tmp_path / "losses"
    losses.mkdir()
    (losses / "loss-stream.jsonl").write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {
                    "global_step": 2820,
                    "branch_id": "resume-step-000002000-1",
                    "metrics": {"total_loss": 1.0},
                },
                {
                    "global_step": 2860,
                    "branch_id": "resume-step-000002000-1",
                    "metrics": {"total_loss": 0.9},
                },
            )
        )
        + "\n"
    )
    (tmp_path / "last_verified.json").write_text(
        json.dumps({"checkpoint": "run.resume-step-000002000-1.step-000002826"})
    )
    monkeypatch.setattr(subject, "TRAINING_STEPS", 7235)
    monkeypatch.setattr(
        subject,
        "_training_process_metrics",
        lambda: {
            "pid": 1,
            "rss_bytes": 4096,
            "elapsed_seconds": 30,
            "branch_id": "resume-step-000002826-1",
        },
    )
    monkeypatch.setattr(subject, "_system_swap_bytes", lambda: 0)
    state = subject.read_training_state(tmp_path, process_active=True)
    assert state["current_step"] == 2826
    assert state["loss_samples"] == [
        {"step": 2820, "train_total": 1.0, "validation_total": None}
    ]
    assert state["examples_per_second"] is None
    assert state["eta_seconds"] is None


def test_dashboard_resume_swap_baseline_is_branch_local_and_immutable(tmp_path: Path) -> None:
    first = subject._dashboard_resume_swap_baseline(
        tmp_path, "resume-step-000002826-1", 3_408_500_000
    )
    second = subject._dashboard_resume_swap_baseline(
        tmp_path, "resume-step-000002826-1", 3_500_000_000
    )
    assert first == second == 3_408_500_000
    value = json.loads(
        (
            tmp_path
            / "dashboard-resource-baselines/resume-step-000002826-1.json"
        ).read_text()
    )
    assert value["authority"].startswith("dashboard-only")


def test_elapsed_seconds_parses_ps_formats() -> None:
    assert subject._elapsed_seconds("01:02") == 62
    assert subject._elapsed_seconds("01:02:03") == 3723
    assert subject._elapsed_seconds("2-01:02:03") == 176523
    assert subject._elapsed_seconds("bad") is None


def test_active_training_never_labels_baseline_as_model_promotion(
    monkeypatch, tmp_path: Path
) -> None:
    status = _status()
    status["training"] = {}
    status["hosts"]["john1"].update({"intent": "idle"})
    status["hosts"]["john2"].update({"intent": "idle"})
    status["hosts"]["john3"].update({"intent": "idle"})
    monkeypatch.setattr(subject, "packing_is_complete", lambda: True)
    monkeypatch.setattr(
        subject,
        "read_training_state",
        lambda: {
            "active": True,
            "complete": False,
            "current_step": 100,
            "latest_verified_checkpoint": {"id": "checkpoint", "blake3": "aa"},
            "loss_samples": [],
            "run_exists": True,
            "rss_bytes": 1024,
            "swap_delta_bytes": 0,
            "examples_per_second": 60.0,
            "eta_seconds": 10.0,
        },
    )
    lagged = tmp_path / "lagged.json"
    lagged.write_text(
        json.dumps(
            {
                "schema_id": "cascadia.r2-map.lagged-greedy-benchmark-aggregate.v1",
                "games": 5_000,
                "seed_coverage": "contiguous-exactly-once",
                "aggregate_games_per_second": 1000.0,
            }
        )
    )
    monkeypatch.setattr(subject, "LAGGED_BENCHMARK_AGGREGATE", lagged)
    subject.apply_post_bootstrap_phase(status)
    assert status["benchmark"]["classification"] == "pending"
    assert status["benchmark"]["stage"] == "cross-architecture-smoke-awaiting-candidate"
    assert status["benchmark"]["pairs_completed"] == 0
    assert status["benchmark"]["pairs_total"] == 20
    assert status["training"]["eta_seconds"] == 10.0
    assert status["hosts"]["john1"]["eta_seconds"] == 10.0


def test_recovery_required_uses_versioned_validate_intent(monkeypatch) -> None:
    status = _status()
    status["training"] = {}
    for host in ("john1", "john2", "john3"):
        status["hosts"][host]["intent"] = "idle"
    monkeypatch.setattr(subject, "packing_is_complete", lambda: True)
    monkeypatch.setattr(
        subject,
        "read_training_state",
        lambda: {
            "active": False,
            "complete": False,
            "current_step": 2826,
            "latest_verified_checkpoint": {"id": "checkpoint-2826", "blake3": "aa"},
            "loss_samples": [],
            "run_exists": True,
            "rss_bytes": None,
            "swap_delta_bytes": 0,
            "examples_per_second": None,
            "eta_seconds": None,
        },
    )
    subject.apply_post_bootstrap_phase(status)
    assert status["phase"] == "bootstrap-training-recovery-required"
    assert status["hosts"]["john1"]["intent"] == "validate"
    assert status["legal_next_transitions"] == ["resume-from-last-verified-checkpoint"]
