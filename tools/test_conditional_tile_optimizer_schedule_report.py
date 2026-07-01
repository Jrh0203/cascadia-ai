from __future__ import annotations

from cascadia_mlx.conditional_tile_optimizer_schedule import (
    EPOCHS,
    late_cosine_learning_rate,
)
from conditional_tile_optimizer_schedule_report import (
    classify_optimizer_schedule,
    schedule_matches,
)


def test_pipeline_invalid_has_precedence() -> None:
    assert (
        classify_optimizer_schedule(
            {"pipeline_passed": False, "treatment_passed": True}
        )
        == "optimizer_schedule_pipeline_invalid"
    )


def test_valid_failed_treatment_is_insufficient() -> None:
    assert (
        classify_optimizer_schedule(
            {"pipeline_passed": True, "treatment_passed": False}
        )
        == "optimizer_schedule_tile_insufficient"
    )


def test_complete_treatment_is_sufficient() -> None:
    assert (
        classify_optimizer_schedule(
            {"pipeline_passed": True, "treatment_passed": True}
        )
        == "optimizer_schedule_tile_sufficient"
    )


def test_schedule_validation_requires_every_exact_epoch() -> None:
    events = [
        {
            "epoch": epoch,
            "learning_rate": late_cosine_learning_rate(epoch, EPOCHS),
        }
        for epoch in range(1, EPOCHS + 1)
    ]
    assert schedule_matches(events)
    events[-1]["learning_rate"] = 1e-4
    assert not schedule_matches(events)
