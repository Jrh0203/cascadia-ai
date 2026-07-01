from __future__ import annotations

from conditional_tile_local_geometry_dropout_report import (
    classify_dropout,
    dropout_trajectory_matches,
)


def test_pipeline_invalid_has_precedence() -> None:
    assert (
        classify_dropout({"pipeline_passed": False, "treatment_passed": True})
        == "local_geometry_dropout_pipeline_invalid"
    )


def test_valid_failed_treatment_is_insufficient() -> None:
    assert (
        classify_dropout({"pipeline_passed": True, "treatment_passed": False})
        == "local_geometry_dropout_tile_insufficient"
    )


def test_complete_treatment_is_sufficient() -> None:
    assert (
        classify_dropout({"pipeline_passed": True, "treatment_passed": True})
        == "local_geometry_dropout_tile_sufficient"
    )


def test_dropout_trajectory_requires_exact_coverage() -> None:
    events = [
        {
            "dropout_items": 7,
            "dropout_eligible_items": 13,
            "dropout_fraction": 7 / 13,
        }
        for _epoch in range(200)
    ]
    assert dropout_trajectory_matches(
        events,
        expected_selected=7,
        expected_items=13,
    )
    events[-1]["dropout_items"] = 6
    assert not dropout_trajectory_matches(
        events,
        expected_selected=7,
        expected_items=13,
    )
