from __future__ import annotations

from conditional_tile_extended_exposure_report import (
    classify_extended_exposure,
    summarize_trajectory,
)


def test_pipeline_invalid_has_precedence() -> None:
    assert (
        classify_extended_exposure({"pipeline_passed": False, "treatment_passed": True})
        == "extended_exposure_pipeline_invalid"
    )


def test_failed_treatment_is_insufficient() -> None:
    assert (
        classify_extended_exposure({"pipeline_passed": True, "treatment_passed": False})
        == "extended_exposure_tile_insufficient"
    )


def test_complete_treatment_is_sufficient() -> None:
    assert (
        classify_extended_exposure({"pipeline_passed": True, "treatment_passed": True})
        == "extended_exposure_tile_sufficient"
    )


def test_trajectory_summary_requires_every_epoch() -> None:
    events = [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "train": {
                "target_factor_recall": min(1.0, epoch / 100),
                "exact_query_fraction": min(1.0, epoch / 120),
            },
        }
        for epoch in range(1, 201)
    ]
    summary = summarize_trajectory(events)
    assert summary["epochs_complete"] is True
    assert summary["events"] == 200
    assert summary["first_epoch_at_or_above_0.80"] == 80
    assert summary["first_epoch_at_or_above_0.95"] == 95


def test_trajectory_summary_rejects_gap() -> None:
    events = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train": {
                "target_factor_recall": 0.5,
                "exact_query_fraction": 0.2,
            },
        },
        {
            "epoch": 3,
            "train_loss": 0.5,
            "train": {
                "target_factor_recall": 0.6,
                "exact_query_fraction": 0.3,
            },
        },
    ]
    assert summarize_trajectory(events)["epochs_complete"] is False
