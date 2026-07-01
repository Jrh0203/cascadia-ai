from __future__ import annotations

import json
from pathlib import Path

import blake3
import numpy as np
from cascadia_mlx.counterfactual_advantage_dataset import (
    CounterfactualAdvantageDataset,
)
from cascadia_mlx.distributional_opportunity_experiment import (
    ARM_ROLES,
    ARMS,
    ROLES,
    build_authorization,
    canonical_blake3,
    classify_reports,
    frozen_homoscedastic_offsets,
    report_scientific_identity,
    target_reliability_audit,
    validate_authorization,
    verify_authorization,
)
from test_counterfactual_advantage_dataset import (
    write_counterfactual_advantage_dataset,
)


def _datasets(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    write_counterfactual_advantage_dataset(
        train,
        split="train",
        game_index=9_996,
    )
    write_counterfactual_advantage_dataset(
        validation,
        split="validation",
        game_index=9_997,
    )
    return train, validation


def _metrics(arm: str) -> dict:
    treatment = arm == "g1-heteroscedastic-gaussian"
    return {
        "centered_mean_absolute_error": 0.50,
        "centered_root_mean_squared_error": 0.65,
        "centered_advantage_correlation": 0.75,
        "top_action_agreement": 0.55,
        "top_value_recall": 0.60,
        "mean_top_action_regret": 0.30,
        "empirical_crps": 0.95 if treatment else 1.0,
        "pairwise_probability": {
            "brier_score": 0.24 if treatment else 0.25,
            "log_loss": 0.60,
        },
        "uncertainty": {
            "absolute_mean_error_correlation": 0.20 if treatment else 0.10,
            "target_stddev_correlation": 0.50,
            "stddev_mean_absolute_error": 0.30,
        },
        "winner_confidence_set": {
            "coverage": 0.95,
            "mean_size": 2.5,
            "singleton_fraction": 0.25,
        },
        "prediction_probe_blake3": blake3.blake3(arm.encode()).hexdigest(),
    }


def _report(role: str, model_hash: str) -> dict:
    arm = next(arm for arm, roles in ARM_ROLES.items() if role in roles)
    report = {
        "experiment_id": "v2-distributional-opportunity-supervision-v1",
        "adr": "0179",
        "protocol_id": "matched-r12-distributional-opportunity-v1",
        "role": role,
        "arm": arm,
        "authorization": {
            "authorization_id": "1" * 64,
            "bundle_id": "2" * 64,
        },
        "protocol": {"training_steps": 3_000},
        "datasets": {
            "train": {"manifest_blake3": "3" * 64},
            "validation": {"manifest_blake3": "4" * 64},
        },
        "target_reliability": {"train": {}, "validation": {}},
        "homoscedastic_offsets": {
            "blake3": "5" * 64,
            "values": [0.0] * 12,
        },
        "model": {
            "parameter_count": 100,
            "parameter_layout_blake3": "6" * 64,
            "initial_parameter_tensor_blake3": "7" * 64,
            "final_parameter_tensor_blake3": blake3.blake3(arm.encode()).hexdigest(),
            "final_model_file_blake3": model_hash,
        },
        "optimization": {
            "global_step": 3_000,
            "training_groups": 96_000,
        },
        "metrics": {"validation": _metrics(arm)},
        "integrity": {"all_metrics_finite": True},
        "claims": {"gameplay_strength_measured": False},
    }
    report["scientific_identity"] = report_scientific_identity(report)
    report["report_id"] = canonical_blake3(report["scientific_identity"])
    return report


def test_train_only_homoscedastic_offsets_are_ordered_and_centered(
    tmp_path: Path,
) -> None:
    train, _validation = _datasets(tmp_path)
    offsets = frozen_homoscedastic_offsets(CounterfactualAdvantageDataset(train))

    assert offsets.shape == (12,)
    assert np.all(np.diff(offsets) >= 0)
    assert abs(float(np.mean(offsets))) <= 1e-7


def test_reliability_audit_and_authorization_bind_exact_data(
    tmp_path: Path,
) -> None:
    train, validation = _datasets(tmp_path)
    audit = target_reliability_audit(CounterfactualAdvantageDataset(train))
    authorization = build_authorization(
        train_dataset=train,
        validation_dataset=validation,
        bundle_id="a" * 64,
    )

    assert audit["groups"] == 4
    assert audit["samples_per_candidate"] == 12
    assert authorization["approved"] is True
    fingerprints = authorization["identity"]["model"]["arm_fingerprints"]
    assert {value["arm"] for value in fingerprints} == set(ARMS)
    assert (
        len(
            {
                (
                    value["parameter_count"],
                    value["parameter_layout_blake3"],
                    value["initial_parameter_tensor_blake3"],
                )
                for value in fingerprints
            }
        )
        == 1
    )

    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization))
    assert (
        validate_authorization(
            authorization_path,
            train_dataset=train,
            validation_dataset=validation,
            bundle_id="a" * 64,
            role="c0-primary",
        )
        == authorization
    )
    receipt = verify_authorization(
        path=authorization_path,
        train_dataset=train,
        validation_dataset=validation,
        bundle_id="a" * 64,
        role="g1-primary",
    )
    assert receipt["passed"] is True
    assert receipt["run_directory_created"] is False
    assert receipt["optimizer_created"] is False


def test_classifier_requires_replay_and_selects_only_eligible_arm(
    tmp_path: Path,
) -> None:
    reports = {}
    models = {}
    for arm, roles in ARM_ROLES.items():
        model_path = tmp_path / f"{arm}.safetensors"
        model_path.write_bytes(arm.encode())
        model_hash = blake3.blake3(model_path.read_bytes()).hexdigest()
        for role in roles:
            report_path = tmp_path / f"{role}.json"
            report_path.write_text(json.dumps(_report(role, model_hash)))
            reports[role] = report_path
            models[role] = model_path

    result = classify_reports(reports, models=models)

    assert set(reports) == set(ROLES)
    assert result["scientific"]["integrity_pass"] is True
    assert result["scientific"]["selected_arm"] == "g1-heteroscedastic-gaussian"
    assert result["scientific"]["classification"] == "distributional_opportunity_arm_selected"

    replay = ARM_ROLES["g1-heteroscedastic-gaussian"][1]
    changed = json.loads(reports[replay].read_text())
    changed["model"]["final_parameter_tensor_blake3"] = "f" * 64
    changed["scientific_identity"] = report_scientific_identity(changed)
    changed["report_id"] = canonical_blake3(changed["scientific_identity"])
    reports[replay].write_text(json.dumps(changed))
    result = classify_reports(reports, models=models)
    assert result["scientific"]["integrity_pass"] is False
    assert result["scientific"]["selected_arm"] is None
