from __future__ import annotations

from frontier_hierarchical_retrieval_report import (
    EXPECTED_ORACLE,
    cache_audit_summary,
    compare_stage_replay,
)


def test_stage_replay_projection_ignores_execution_host() -> None:
    origin = {
        "host": "john1",
        "config": {"stage": "tile"},
        "weights_blake3": "weights",
        "train_cache_payload_blake3": "train",
        "validation_cache_payload_blake3": "validation",
        "train": {"target_factor_recall": 0.9},
        "validation": {"target_factor_recall": 0.8},
        "test_split_opened": False,
    }
    replay = {
        "host": "john4",
        "scientific": {
            "stage": "tile",
            "weights_blake3": "weights",
            "train_cache_payload_blake3": "train",
            "validation_cache_payload_blake3": "validation",
            "train": {"target_factor_recall": 0.9},
            "validation": {"target_factor_recall": 0.8},
            "test_split_opened": False,
        },
        "scientific_blake3": "replay",
    }
    comparison = compare_stage_replay(origin, replay)
    assert comparison["cross_host"]
    assert comparison["scientific_payload_identical"]


def test_cache_audit_requires_frozen_oracle_metrics() -> None:
    scientific = {
        "train": {
            **EXPECTED_ORACLE["train"],
            "all_factor_target_labels_exact": True,
        },
        "validation": {
            **EXPECTED_ORACLE["validation"],
            "all_factor_target_labels_exact": True,
        },
    }
    origin = {"host": "john1", "scientific": scientific}
    replay = {"host": "john4", "scientific": scientific}
    manifest = {
        "all_factor_bijections": True,
        "all_prefix_invariants": True,
    }
    summary = cache_audit_summary(
        origin,
        replay,
        manifest,
        manifest,
    )
    assert summary["scientific_payload_identical"]
    assert summary["oracle_reconstruction_passed"]
