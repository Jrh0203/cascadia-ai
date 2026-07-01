from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import o1_opponent_intent_reuse_audit_report as report
import pytest


def _dataset(split: str) -> dict:
    games = 2
    positions = games * 80
    candidates = positions * 64
    windows = games * 76
    return {
        "split": split,
        "games": games,
        "positions": positions,
        "candidates": candidates,
        "exact_checks": {
            "exact_turn_order": positions,
            "exact_active_seat": positions,
            "exact_position_bytes": positions,
            "exact_candidate_action_hashes": candidates,
            "exactly_one_selected_action": positions,
            "exact_state_transitions": positions,
            "terminal_games": games,
        },
        "identity_recovery": {
            "positions_with_four_unique_tile_ids": positions,
        },
        "survival_windows": {
            "focal_post_action_windows": windows,
            "market_tile_labels": windows * 4,
        },
    }


def _report(hostname: str) -> dict:
    return {
        "schema_version": report.SCHEMA_VERSION,
        "experiment_id": report.EXPERIMENT_ID,
        "status": "complete",
        "classification": report.EXPECTED_CLASSIFICATION,
        "scientific_blake3": "a" * 64,
        "datasets": [_dataset("train"), _dataset("validation")],
        "cross_dataset_overlaps": [
            {
                "group_id_overlap": 0,
                "position_record_overlap": 0,
                "public_state_overlap": 0,
                "initial_hidden_state_overlap": 0,
            }
        ],
        "recoverability": {
            "exact_sequential_replay": True,
            "exact_candidate_action_reconstruction": True,
            "exact_selected_action_labels": True,
            "exact_unique_tile_identity": True,
            "exact_post_action_tile_survival": True,
            "exact_next_pick_slots_and_species": True,
            "exact_nature_token_action": True,
            "public_recent_draft_history": True,
            "wildlife_token_physical_identity": False,
        },
        "claim_boundary": {
            "foundation_reuse_authorized": True,
            "final_o1_training_corpus_authorized": False,
            "policy_held_out_evaluation_available": False,
            "checkpoint_identity_shortcut_testable": False,
            "strategy_switch_target_available": False,
        },
        "provenance": {
            "hostname": hostname,
            "executable_blake3": "b" * 64,
            "dataset_roots": [
                f"/Users/{hostname}/cascadia-bench/train",
                f"/Users/{hostname}/cascadia-bench/validation",
            ],
        },
    }


def test_classifier_accepts_exact_distinct_host_replay() -> None:
    result = report.classify(
        _report("john4"),
        _report("john2"),
        primary_path=Path("primary.json"),
        replay_path=Path("replay.json"),
    )
    assert result["foundation_reuse_authorized"] is True
    assert result["final_o1_training_corpus_authorized"] is False
    assert result["classification"] == report.EXPECTED_CLASSIFICATION
    assert (
        _report("john4")["provenance"]["dataset_roots"]
        != _report("john2")["provenance"]["dataset_roots"]
    )


def test_classifier_rejects_any_exact_check_loss() -> None:
    replay = deepcopy(_report("john2"))
    replay["datasets"][0]["exact_checks"]["exact_position_bytes"] -= 1
    with pytest.raises(report.ClassificationError, match="exact_position_bytes"):
        report.classify(
            _report("john4"),
            replay,
            primary_path=Path("primary.json"),
            replay_path=Path("replay.json"),
        )


def test_classifier_rejects_cross_split_overlap() -> None:
    primary = _report("john4")
    primary["cross_dataset_overlaps"][0]["public_state_overlap"] = 1
    with pytest.raises(report.ClassificationError, match="public_state_overlap"):
        report.validate_report(primary, role="primary")


def test_classifier_rejects_policy_holdout_overclaim() -> None:
    primary = _report("john4")
    primary["claim_boundary"]["final_o1_training_corpus_authorized"] = True
    with pytest.raises(report.ClassificationError, match="claim boundary"):
        report.validate_report(primary, role="primary")
