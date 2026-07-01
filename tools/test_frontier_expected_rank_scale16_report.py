from __future__ import annotations

from frontier_expected_rank_scale16_report import render_markdown


def _metrics() -> dict[str, float]:
    return {
        "expected_rank_target_positive_recall": 0.55,
        "expected_rank_target_set_exact_fraction": 0.02,
        "top64_r4800_winner_recall": 0.99,
        "top64_confidence_set_coverage_95": 0.99,
        "top64_distinguishable_winner_recall": 0.98,
        "mean_top64_retained_r4800_regret": 0.01,
    }


def test_render_scale16_result_includes_alignment_and_delta() -> None:
    report = {
        "classification": "scale16_alignment_material_but_underfit",
        "selected_model": {
            "scientific": {
                "train": _metrics(),
                "validation": _metrics(),
            },
            "replay_bit_identical": True,
        },
        "baseline": {
            "passed": True,
            "validation": {"baseline": _metrics()},
        },
        "alignment": {
            "train": {
                "probability_mass_in_deployed_target": {"mean": 0.94}
            },
            "validation": {
                "probability_mass_in_deployed_target": {"mean": 0.93},
                "uniform_student_absolute_gradient_fraction_in_deployed_target": {
                    "mean": 0.50
                },
            },
        },
        "cache": {"passed": True},
        "gradient": {"passed": True},
        "comparison": {"train_target_recall_delta": 0.12},
        "execution": {"campaign_wall_seconds": 100.0},
        "gates": {"pilot_passed": False},
    }
    rendered = render_markdown(report)
    assert "scale16_alignment_material_but_underfit" in rendered
    assert "94.00%" in rendered
    assert "+12.00%" in rendered
    assert "`pilot_passed`" in rendered
