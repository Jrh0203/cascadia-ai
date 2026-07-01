from __future__ import annotations

import json
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.opponent_intent_experiment import (
    ARM_ROLES,
    ARMS,
    ROLES,
    build_authorization,
    canonical_blake3,
    classify_reports,
    evaluate_selected,
    evidence_blake3,
    paired_game_bootstrap,
    report_scientific_identity,
    verify_authorization,
)
from test_opponent_intent_dataset import write_opponent_intent_dataset


def _corpus_classification(path: Path) -> Path:
    value = {
        "experiment_id": "o1-opponent-intent-policy-heldout-corpus-v1",
        "classification": "policy_held_out_draft_survival_corpus_passed",
        "classification_blake3": "a" * 64,
        "matched_scientific_blake3": "b" * 64,
        "authorization": {
            "public_state_control_training": True,
            "recent_history_intent_training": True,
            "next_draft_auxiliary_training": True,
            "market_survival_training": True,
            "policy_held_out_calibration": True,
            "paid_wipe_intent_training": False,
            "strategy_switch_training": False,
            "gameplay_promotion": False,
        },
    }
    path.write_text(json.dumps(value))
    return path


def _metrics(brier: float, *, auxiliary_gain: float = 0.10) -> dict:
    return {
        "windows": 8,
        "tile_labels": 32,
        "games": 2,
        "disposition": {
            "negative_log_likelihood": 0.50,
            "multiclass_brier": brier,
            "accuracy": 0.75,
            "macro_f1": 0.70,
            "top_label_ece": 0.02,
            "survival_binary": {
                "brier": 0.10,
                "negative_log_likelihood": 0.30,
                "ece": 0.02,
                "auroc": 0.80,
            },
        },
        "mean_next_draft_relative_nll_gain": auxiliary_gain,
        "prediction_probe_blake3": "c" * 64,
    }


def _evidence(correct_probability: float) -> dict[str, np.ndarray]:
    targets = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.uint8), (8, 1))
    probabilities = np.full((8, 4, 4), (1 - correct_probability) / 3)
    for row in range(8):
        for slot in range(4):
            probabilities[row, slot, targets[row, slot]] = correct_probability
    return {
        "schema_version": np.asarray([1], dtype=np.int16),
        "game_index": np.repeat(np.asarray([10, 11], dtype=np.int64), 4),
        "focal_turn": np.tile(np.arange(4, dtype=np.uint8), 2),
        "disposition_probabilities": probabilities.astype(np.float32),
        "disposition_targets": targets,
    }


def _write_evidence(path: Path, evidence: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **evidence)


def _report(
    role: str,
    *,
    brier: float,
    model_hash: str,
    evidence_path: Path,
    evidence: dict[str, np.ndarray],
    auxiliary_gain: float = 0.10,
) -> dict:
    arm = next(arm for arm, roles in ARM_ROLES.items() if role in roles)
    report = {
        "experiment_id": "o1-opponent-intent-mlx-factorial-v1",
        "adr": "0187",
        "protocol_id": "o1-policy-heldout-matched-mlx-v1",
        "mode": "production",
        "role": role,
        "arm": arm,
        "authorization": {
            "authorization_id": "1" * 64,
            "bundle_id": "2" * 64,
        },
        "protocol": {"training_steps": 5_120},
        "corpus": {"classification": "passed"},
        "datasets": {
            "train": [{"manifest_blake3": "3" * 64}],
            "validation": {"manifest_blake3": "4" * 64},
        },
        "training_priors_blake3": "5" * 64,
        "model": {
            "config": {"arm": arm},
            "parameter_count": 100,
            "parameter_layout_blake3": "6" * 64,
            "initial_parameter_tensor_blake3": "7" * 64,
            "final_parameter_tensor_blake3": blake3.blake3(arm.encode()).hexdigest(),
            "final_model_file_blake3": model_hash,
        },
        "optimization": {
            "global_step": 5_120,
            "training_examples": 1_000,
        },
        "metrics": {
            "validation": _metrics(
                brier,
                auxiliary_gain=auxiliary_gain,
            )
        },
        "evidence": {
            "validation_file_blake3": blake3.blake3(evidence_path.read_bytes()).hexdigest(),
            "validation_array_blake3": evidence_blake3(evidence),
        },
        "integrity": {"all_metrics_finite": True},
        "claims": {"gameplay_strength_measured": False},
    }
    report["scientific_identity"] = report_scientific_identity(report)
    report["report_id"] = canonical_blake3(report["scientific_identity"])
    return report


def test_authorization_binds_corpus_data_and_matched_graph(
    tmp_path: Path,
) -> None:
    train0 = write_opponent_intent_dataset(
        tmp_path / "train0",
        split="train",
        game_index=500,
        cohort_id="train-0",
    )
    train1 = write_opponent_intent_dataset(
        tmp_path / "train1",
        split="train",
        game_index=501,
        cohort_id="train-1",
    )
    validation = write_opponent_intent_dataset(
        tmp_path / "validation",
        split="validation",
        game_index=502,
        cohort_id="validation",
    )
    corpus = _corpus_classification(tmp_path / "corpus.json")
    authorization = build_authorization(
        train_datasets=(train0, train1),
        validation_dataset=validation,
        corpus_classification=corpus,
        bundle_id="d" * 64,
    )
    fingerprints = authorization["identity"]["model"]["arm_fingerprints"]

    assert authorization["approved"] is True
    assert len({item["parameter_count"] for item in fingerprints}) == 1
    assert len({item["initial_parameter_tensor_blake3"] for item in fingerprints}) == 1
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization))
    receipt = verify_authorization(
        path=path,
        train_datasets=(train0, train1),
        validation_dataset=validation,
        corpus_classification=corpus,
        bundle_id="d" * 64,
        role="a0-primary",
    )
    assert receipt["passed"] is True
    assert receipt["optimizer_created"] is False


def test_game_clustered_bootstrap_and_replay_classifier(
    tmp_path: Path,
) -> None:
    reports = {}
    evidence_paths = {}
    model_paths = {}
    for arm, roles in ARM_ROLES.items():
        treatment = arm == ARMS[1]
        arm_evidence = _evidence(0.90 if treatment else 0.70)
        evidence_path = tmp_path / f"{arm}.npz"
        _write_evidence(evidence_path, arm_evidence)
        model_path = tmp_path / f"{arm}.safetensors"
        model_path.write_bytes(arm.encode())
        model_hash = blake3.blake3(model_path.read_bytes()).hexdigest()
        for role in roles:
            report_path = tmp_path / f"{role}.json"
            report_path.write_text(
                json.dumps(
                    _report(
                        role,
                        brier=0.15 if treatment else 0.20,
                        model_hash=model_hash,
                        evidence_path=evidence_path,
                        evidence=arm_evidence,
                    )
                )
            )
            reports[role] = report_path
            evidence_paths[role] = evidence_path
            model_paths[role] = model_path

    comparison = paired_game_bootstrap(
        _evidence(0.70),
        _evidence(0.90),
        samples=1_000,
        seed=17,
    )
    result = classify_reports(reports, evidence_paths, model_paths)

    assert comparison["confidence_95"][1] < 0
    assert set(reports) == set(ROLES)
    assert result["scientific"]["integrity_pass"] is True
    assert result["scientific"]["selected_arm"] == ARMS[1]
    assert result["scientific"]["classification"] == "opponent_intent_validation_arm_selected"


def test_null_validation_does_not_open_sealed_paths(tmp_path: Path) -> None:
    classification = {
        "classification_id": "e" * 64,
        "scientific": {"selected_arm": None},
    }
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(classification))

    result = evaluate_selected(
        classification_path=path,
        reports={},
        models={},
        authorization_path=tmp_path / "missing-authorization.json",
        test_dataset=tmp_path / "missing-test",
        final_stress_dataset=tmp_path / "missing-final",
    )

    assert result["classification"] == "opponent_intent_test_not_opened"
    assert result["test_or_final_opened"] is False
