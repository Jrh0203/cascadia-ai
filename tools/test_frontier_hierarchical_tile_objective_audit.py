from __future__ import annotations

from frontier_hierarchical_tile_objective_audit import (
    classify_objective_gradient,
)


def _metrics(
    *,
    boundary: float,
    auxiliary: float,
    cosine: float,
) -> dict[str, object]:
    return {
        "mean_gradient_norms": {"boundary": boundary},
        "mean_combined_auxiliary_gradient_norm": auxiliary,
        "mean_boundary_auxiliary_gradient_cosine": cosine,
    }


def test_classifies_gradient_conflict_before_domination() -> None:
    result = classify_objective_gradient(_metrics(boundary=1.0, auxiliary=2.1, cosine=-0.5))
    assert result["primary"] == "objective_gradient_conflict"
    assert result["objective_gradient_conflict"]
    assert result["target_boundary_gradient_dominated"]


def test_classifies_boundary_domination_without_conflict() -> None:
    result = classify_objective_gradient(_metrics(boundary=0.4, auxiliary=1.0, cosine=0.1))
    assert result["primary"] == "target_boundary_gradient_dominated"
    assert not result["objective_gradient_conflict"]
    assert result["target_boundary_gradient_dominated"]


def test_classifies_gradient_pressure_as_non_primary() -> None:
    result = classify_objective_gradient(_metrics(boundary=1.0, auxiliary=1.0, cosine=-0.1))
    assert result["primary"] == "objective_gradient_pressure_not_primary"
    assert not result["objective_gradient_conflict"]
    assert not result["target_boundary_gradient_dominated"]
