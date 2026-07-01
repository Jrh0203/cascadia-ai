from __future__ import annotations

from frontier_expected_rank_report import normalize_host, render_markdown, replay_payload


def _metrics() -> dict[str, object]:
    return {
        "expected_rank_target_positive_recall": 0.75,
        "expected_rank_target_set_exact_fraction": 0.10,
        "top64_r4800_winner_recall": 0.99,
        "top64_confidence_set_coverage_95": 1.0,
        "mean_top64_retained_r4800_regret": 0.01,
    }


def test_replay_payload_omits_host_specific_performance() -> None:
    scientific = {
        "checkpoint": "step-1",
        "checkpoint_manifest_blake3": "a",
        "model_blake3": "b",
        "train_dataset_id": "train",
        "train_manifest_blake3": "c",
        "train_cache_identity": "d",
        "validation_dataset_id": "validation",
        "validation_manifest_blake3": "e",
        "validation_cache_identity": "f",
        "train": _metrics(),
        "validation": _metrics(),
        "performance": {"elapsed_seconds": 1.0},
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    payload = replay_payload({"scientific": scientific})
    assert "performance" not in payload
    assert payload["model_blake3"] == "b"


def test_render_reports_failure_and_normalizes_john1() -> None:
    report = {
        "classification": "expected_rank_optimization_underfit",
        "selected_model": {
            "scientific": {
                "train": _metrics(),
                "validation": _metrics(),
            },
            "replay_bit_identical": True,
        },
        "baseline": {
            "validation": {
                **_metrics(),
                "expected_rank_target_positive_recall": 0.20,
                "expected_rank_target_set_exact_fraction": 0.0,
            }
        },
        "cache": {"passed": True},
        "gradient": {"passed": True},
        "execution": {"campaign_wall_seconds": 100.0},
        "gates": {"pilot_passed": False, "train_fit": False},
    }
    rendered = render_markdown(report)
    assert "expected_rank_optimization_underfit" in rendered
    assert "`pilot_passed`" in rendered
    assert normalize_host("Johns-Mac-mini.local") == "john1"
