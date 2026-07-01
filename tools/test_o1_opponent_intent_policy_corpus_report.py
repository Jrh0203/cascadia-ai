from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import o1_opponent_intent_policy_corpus_report as report
import pytest


def _dataset(role: str, records: int) -> dict:
    return {
        "role": role,
        "records": records,
        "unique_model_inputs": records,
        "duplicate_model_inputs": 0,
        "identity_exclusion_checks": records,
        "model_input_bytes": 1189,
    }


def _report(hostname: str) -> dict:
    scientific = {
        "datasets": [
            _dataset("train-part-0", 38912),
            _dataset("train-part-1", 38912),
            _dataset("validation", 19456),
            _dataset("test", 19456),
            _dataset("final-stress", 9728),
        ],
        "totals": {
            "games": 1664,
            "records": 126464,
            "shards": 104,
            "unique_model_inputs": 126464,
            "duplicate_model_inputs_within_datasets": 0,
            "identity_exclusion_checks": 126464,
        },
        "overlaps": [{"exact_hash_overlap": 0, "sample_hashes": []} for _ in range(10)],
        "action_factor_coverage": [{"passed": True, "missing_from_training": []} for _ in range(9)],
        "survival_coverage": {"passed": True},
        "limitations": [
            {
                "label": "Paid wildlife-wipe intent is unsupported",
                "observed": "0 of 379392 target actions contain a paid wipe",
                "consequence": "scope",
            },
            {
                "label": "Strategy-switch targets are unavailable",
                "observed": "none",
                "consequence": "scope",
            },
            {
                "label": "Policy holdout covers v2 heuristic families only",
                "observed": "heuristics",
                "consequence": "scope",
            },
        ],
        "gates": [{"passed": True} for _ in range(7)],
    }
    return {
        "schema_version": report.SCHEMA_VERSION,
        "experiment_id": report.EXPERIMENT_ID,
        "status": "complete",
        "classification": report.EXPECTED_AUDIT_CLASSIFICATION,
        "scientific": scientific,
        "scientific_blake3": "a" * 64,
        "provenance": {
            "hostname": hostname,
            "executable_blake3": "b" * 64,
            "dataset_roots": {role: f"/Users/{hostname}/{role}" for role in report.EXPECTED_ROLES},
        },
    }


def test_classifier_authorizes_only_supported_learning_scope() -> None:
    result = report.classify(
        _report("john1"),
        _report("john3"),
        primary_path=Path("primary.json"),
        replay_path=Path("replay.json"),
    )
    assert result["classification"] == report.FINAL_CLASSIFICATION
    assert result["authorization"]["market_survival_training"] is True
    assert result["authorization"]["paid_wipe_intent_training"] is False
    assert result["authorization"]["gameplay_promotion"] is False


def test_classifier_rejects_scientific_replay_difference() -> None:
    replay = deepcopy(_report("john3"))
    replay["scientific"]["totals"]["records"] -= 1
    with pytest.raises(report.ClassificationError, match="corpus totals"):
        report.classify(
            _report("john1"),
            replay,
            primary_path=Path("primary.json"),
            replay_path=Path("replay.json"),
        )


def test_classifier_rejects_same_host_replay() -> None:
    with pytest.raises(report.ClassificationError, match="distinct hosts"):
        report.classify(
            _report("john1"),
            _report("john1"),
            primary_path=Path("primary.json"),
            replay_path=Path("replay.json"),
        )


def test_classifier_rejects_hidden_paid_wipe_gap() -> None:
    primary = _report("john1")
    primary["scientific"]["limitations"][0]["observed"] = "support available"
    with pytest.raises(report.ClassificationError, match="paid-wipe"):
        report.validate_report(primary, role="primary")
