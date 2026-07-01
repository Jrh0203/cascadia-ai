from __future__ import annotations

from conditional_tile_capacity_report import classify


def test_pipeline_invalid_has_precedence() -> None:
    assert (
        classify(
            {"pipeline_passed": False},
            "query_relational_representation_insufficient",
        )
        == "conditional_tile_capacity_audit_invalid"
    )


def test_valid_pipeline_preserves_mechanism() -> None:
    assert (
        classify(
            {"pipeline_passed": True},
            "full_data_scale_or_optimization_insufficient",
        )
        == "full_data_scale_or_optimization_insufficient"
    )
