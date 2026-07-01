from __future__ import annotations

from conditional_tile_target_only_report import classify_target_only


def test_pipeline_invalid_has_precedence() -> None:
    assert (
        classify_target_only({"pipeline_passed": False, "treatment_passed": True})
        == "target_only_tile_pipeline_invalid"
    )


def test_failed_treatment_is_insufficient() -> None:
    assert (
        classify_target_only({"pipeline_passed": True, "treatment_passed": False})
        == "target_only_tile_objective_insufficient"
    )


def test_complete_treatment_is_sufficient() -> None:
    assert (
        classify_target_only({"pipeline_passed": True, "treatment_passed": True})
        == "target_only_tile_objective_sufficient"
    )
