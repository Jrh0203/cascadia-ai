from __future__ import annotations

from collections import Counter

from conditional_tile_generalization_forensics import (
    EXPERIMENT_ID,
    _js_divergence,
    _width_bin,
    combine,
)


def _arm(classification: str, *, pipeline: bool = True) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "classification": classification,
            "pipeline_passed": pipeline,
        },
    }


def test_width_bins_cover_frozen_boundaries() -> None:
    assert _width_bin(1) == "le32"
    assert _width_bin(32) == "le32"
    assert _width_bin(33) == "33_64"
    assert _width_bin(96) == "65_96"
    assert _width_bin(128) == "97_128"
    assert _width_bin(129) == "ge129"


def test_js_divergence_is_zero_for_identical_histograms() -> None:
    assert _js_divergence(Counter({32: 5, 64: 2}), Counter({32: 5, 64: 2})) == 0


def test_aliasing_has_first_successor_precedence() -> None:
    report = combine(
        _arm("observable_label_aliasing_material"),
        _arm("input_covariate_shift_material"),
        _arm("late_fit_margin_specialization"),
    )
    assert (
        report["scientific"]["mechanical_successor_if_adr0120_fails"]
        == "query_set_aware_tile_scorer"
    )


def test_shift_precedes_margin_when_aliasing_is_not_material() -> None:
    report = combine(
        _arm("observable_label_aliasing_not_material"),
        _arm("input_covariate_shift_material"),
        _arm("late_fit_margin_specialization"),
    )
    assert (
        report["scientific"]["mechanical_successor_if_adr0120_fails"]
        == "distribution_robust_representation"
    )


def test_invalid_arm_invalidates_combined_decision() -> None:
    report = combine(
        _arm("observable_label_aliasing_not_material", pipeline=False),
        _arm("input_covariate_shift_not_material"),
        _arm("late_fit_margin_specialization_not_proven"),
    )
    assert (
        report["scientific"]["mechanical_successor_if_adr0120_fails"]
        == "generalization_forensics_pipeline_invalid"
    )
