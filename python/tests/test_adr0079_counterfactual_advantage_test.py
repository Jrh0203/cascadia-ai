from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_evaluator():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools/adr0079_counterfactual_advantage_test.py"
    spec = importlib.util.spec_from_file_location("adr0079_test_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator()


def _metrics() -> dict:
    return {
        "decision_objective": 0.45,
        "centered_mean_absolute_error": 0.60,
        "centered_advantage_correlation": 0.70,
        "top_value_recall": 0.55,
        "mean_top_action_regret": 0.30,
        "h6_selected_baseline": {
            "top_value_recall": 0.45,
            "mean_top_action_regret": 0.40,
        },
    }


def _authorization() -> dict:
    return {
        "experiment": evaluator.EXPERIMENT,
        "parent_experiment": evaluator.PARENT_EXPERIMENT,
        "validation_passed": True,
        "validation_report_sha256": "validation-sha",
        "authorized_at_unix_seconds": 100,
        "test_absent_on_nodes": {
            "john1": True,
            "john2": True,
            "john3": True,
        },
    }


def test_frozen_test_gates_pass_only_with_post_validation_authorization() -> None:
    gates = evaluator.evaluate_gates(
        metrics=_metrics(),
        initial_metrics={
            "decision_objective": 0.55,
            "centered_mean_absolute_error": 0.70,
        },
        validation_report={
            "checkpoint": "step-100",
            "checkpoint_manifest_blake3": "checkpoint-hash",
        },
        validation_replay_exact=True,
        checkpoint_name="step-100",
        checkpoint_blake3="checkpoint-hash",
        authorization=_authorization(),
        validation_report_sha256="validation-sha",
        test_created_unix_seconds=101,
        source_matches=True,
    )
    assert all(gates.values())

    premature = evaluator.evaluate_gates(
        metrics=_metrics(),
        initial_metrics={
            "decision_objective": 0.55,
            "centered_mean_absolute_error": 0.70,
        },
        validation_report={
            "checkpoint": "step-100",
            "checkpoint_manifest_blake3": "checkpoint-hash",
        },
        validation_replay_exact=True,
        checkpoint_name="step-100",
        checkpoint_blake3="checkpoint-hash",
        authorization=_authorization(),
        validation_report_sha256="validation-sha",
        test_created_unix_seconds=99,
        source_matches=True,
    )
    assert not premature["validation_passed_before_test_collection"]


def test_checkpoint_or_validation_replay_drift_fails_closed() -> None:
    gates = evaluator.evaluate_gates(
        metrics=_metrics(),
        initial_metrics={
            "decision_objective": 0.55,
            "centered_mean_absolute_error": 0.70,
        },
        validation_report={
            "checkpoint": "step-100",
            "checkpoint_manifest_blake3": "checkpoint-hash",
        },
        validation_replay_exact=False,
        checkpoint_name="step-101",
        checkpoint_blake3="different",
        authorization=_authorization(),
        validation_report_sha256="validation-sha",
        test_created_unix_seconds=101,
        source_matches=True,
    )
    assert not gates["validation_checkpoint_is_unchanged"]
    assert not gates["validation_replay_is_bit_exact"]
