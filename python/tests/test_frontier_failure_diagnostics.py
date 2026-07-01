from __future__ import annotations

from cascadia_mlx.frontier_failure_diagnostics import (
    classify_collision,
    classify_error_anatomy,
    classify_objective_gradient,
    classify_train_fit,
    select_failure_mechanism,
)


def test_train_fit_classifies_low_training_recall_as_underfit() -> None:
    classification = classify_train_fit(
        {
            "target_positive_recall": 0.45,
            "target_set_exact_fraction": 0.0,
        },
        {
            "target_positive_recall": 0.40,
            "target_set_exact_fraction": 0.0,
        },
    )
    assert classification["primary"] == "optimization_or_capacity_underfit"
    assert classification["optimization_or_capacity_underfit"]
    assert not classification["generalization_failure"]


def test_train_fit_only_calls_generalization_after_train_fit_passes() -> None:
    classification = classify_train_fit(
        {
            "target_positive_recall": 0.90,
            "target_set_exact_fraction": 0.50,
        },
        {
            "target_positive_recall": 0.70,
            "target_set_exact_fraction": 0.10,
        },
    )
    assert classification["primary"] == "generalization_failure"


def test_collision_gate_is_strictly_evidence_driven() -> None:
    assert not classify_collision(
        {"conflicting_target_positive_fraction": 0.009}
    )["exact_observable_collision_material"]
    assert classify_collision(
        {"conflicting_target_positive_fraction": 0.01}
    )["exact_observable_collision_material"]


def test_gradient_classifies_conflict_before_domination() -> None:
    classification = classify_objective_gradient(
        {
            "weighted_gradient_norms": {
                "target_set_cross_entropy": 1.0,
                "r1200_listwise": 2.0,
                "screen_only_regularization": 0.0,
            },
            "weighted_auxiliary_gradient_norm": 2.0,
            "target_listwise_gradient_cosine": -0.4,
        }
    )
    assert classification["primary"] == "objective_gradient_conflict"
    assert classification["objective_gradient_conflict"]


def test_error_anatomy_requires_mass_gap_and_minimum_support() -> None:
    classification = classify_error_anatomy(
        0.5,
        {
            "phase": {
                "early": {
                    "target_positives": 100,
                    "recalled": 20,
                    "misses": 80,
                    "recall": 0.2,
                    "miss_share": 0.4,
                },
                "late": {
                    "target_positives": 20,
                    "recalled": 0,
                    "misses": 20,
                    "recall": 0.0,
                    "miss_share": 0.5,
                },
            }
        },
    )
    assert classification["error_concentration_material"]
    assert classification["concentrated_slices"][0]["slice"] == "early"


def test_combined_selection_prefers_collision_over_underfit() -> None:
    reports = {
        "train-fit": {
            "classification": {
                "optimization_or_capacity_underfit": True,
                "generalization_failure": False,
            }
        },
        "observable-collision": {
            "classification": {"exact_observable_collision_material": True}
        },
        "objective-gradient": {
            "classification": {
                "objective_gradient_conflict": False,
                "target_objective_gradient_dominated": False,
            }
        },
        "error-anatomy": {
            "classification": {"error_concentration_material": False}
        },
    }
    selection = select_failure_mechanism(reports)
    assert selection["selected_mechanism"] == "representation_collision"
