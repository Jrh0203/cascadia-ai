from __future__ import annotations

from pathlib import Path

import blake3
import pytest
from cascadia_mlx.r3_action_edit_mlx_cache import open_data_verification_id
from r3_action_edit_mlx_checkpoint_replay import build_replay_report


def _digest(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    open_data = {
        "cache_id": "1" * 64,
        "cache_manifest_blake3": "2" * 64,
        "s1_cache_id": "3" * 64,
        "s1_cache_manifest_blake3": "4" * 64,
        "datasets": {
            "train": {
                "dataset_id": "train",
                "manifest_blake3": "5" * 64,
            },
            "validation": {
                "dataset_id": "validation",
                "manifest_blake3": "6" * 64,
            },
        },
    }
    proof_id = open_data_verification_id(open_data)
    origin_performance = {
        "complete_decisions": {
            "action_scores_per_second": 20_000.0,
            "latency_milliseconds": {"p99": 100.0},
        },
        "fixed_chunk": {
            "action_scores_per_second": 40_000.0,
        },
        "memory": {
            "peak_active_bytes": 400,
            "peak_process_rss_bytes": 800,
        },
    }
    origin = {
        "experiment_id": "r3-action-edit-mlx-comparison-v1",
        "protocol_id": "r3-action-edit-mlx-matched-comparison-v1",
        "adr": "0150",
        "arm": "t1-r3-radius3-global",
        "mode": "production",
        "host": "john2",
        "report_id": "7" * 64,
        "checkpoint": {
            "manifest_blake3": _digest(checkpoint / "checkpoint.json"),
            "model_blake3": _digest(checkpoint / "model.safetensors"),
        },
        "optimization": {"global_step": 3000},
        "performance": origin_performance,
    }
    authorization = {
        "experiment_id": origin["experiment_id"],
        "protocol_id": origin["protocol_id"],
        "adr": origin["adr"],
        "approved": True,
        "authorization_id": "8" * 64,
        "identity": {
            "open_data_verification": open_data,
            "open_data_verification_id": proof_id,
        },
    }
    replay = {
        "complete_decisions": {
            "action_scores_per_second": 30_000.0,
            "latency_milliseconds": {"p99": 80.0},
        },
        "fixed_chunk": {
            "action_scores_per_second": 50_000.0,
        },
        "memory": {
            "peak_active_bytes": 300,
            "peak_process_rss_bytes": 600,
            "process_swaps": 0,
        },
        "measurement": {
            "isolated_process": True,
            "checkpoint_model_blake3": origin["checkpoint"]["model_blake3"],
            "open_data_verification_id": proof_id,
            "verification_source": "cluster-preflight",
            "worker_runtime": {"host": "john4"},
        },
    }
    return origin, authorization, replay, checkpoint


def test_build_replay_report_binds_bytes_proof_and_host(tmp_path: Path) -> None:
    origin, authorization, replay, checkpoint = _fixture(tmp_path)
    report = build_replay_report(
        origin=origin,
        authorization=authorization,
        performance=replay,
        checkpoint=checkpoint,
        replay_host="John4.local",
    )

    assert report["operational_pass"] is True
    assert report["classifier_eligible"] is False
    assert report["scientific_identity"]["assertions"]["different_host"] is True
    assert report["host_performance_comparison"] == {
        "complete_action_throughput_ratio": 1.5,
        "complete_p99_latency_ratio": 0.8,
        "fixed_chunk_throughput_ratio": 1.25,
        "peak_active_memory_ratio": 0.75,
        "peak_process_rss_ratio": 0.75,
    }


def test_build_replay_report_rejects_changed_checkpoint(tmp_path: Path) -> None:
    origin, authorization, replay, checkpoint = _fixture(tmp_path)
    (checkpoint / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        build_replay_report(
            origin=origin,
            authorization=authorization,
            performance=replay,
            checkpoint=checkpoint,
            replay_host="john4",
        )
