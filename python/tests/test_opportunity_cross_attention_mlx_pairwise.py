from __future__ import annotations

from copy import deepcopy

import pytest
from cascadia_mlx.opportunity_cross_attention_mlx_pairwise import (
    compare_decision_panels,
    factorial_effects,
    panel_identity,
)


def _panel(recalls: list[bool], *, squared_error: float) -> list[dict]:
    phases = ("early", "middle", "late", "early", "middle", "late")
    records = []
    for row, recalled in enumerate(recalls):
        records.append(
            {
                "row": row,
                "group_id": 10_000 + row,
                "turn": row * 10,
                "candidates": 80,
                "labeled_candidates": 64,
                "teacher_winner_index": 0,
                "teacher_winner_action_hash": f"{row + 1:064x}",
                "winner_rank": 1 if recalled else 65,
                "top64_recalled": recalled,
                "top64_regret": 0.0 if recalled else 1.0,
                "absolute_error_sum": 32.0,
                "squared_error_sum": squared_error,
                "bias_sum": 0.0,
                "low_supply": row in (4, 5),
                "independent_draft_winner": row in (1, 5),
                "phase": phases[row],
                "opportunities": {
                    "elk": row in (0, 3),
                    "salmon": row in (1, 4),
                    "hawk": row in (2, 5),
                    "bear": row in (0, 5),
                },
                "prediction_blake3": f"{100 + row:064x}",
            }
        )
    return records


def test_pairwise_statistics_use_complete_decisions() -> None:
    control = _panel(
        [True, False, True, False, True, False],
        squared_error=128.0,
    )
    treatment = _panel(
        [True, True, True, True, True, True],
        squared_error=96.0,
    )

    report = compare_decision_panels(
        treatment,
        control,
        bootstrap_replicates=2_000,
        bootstrap_seed=7,
    )

    assert report["groups"] == 6
    assert report["global_top64_recall"]["delta"] == 0.5
    assert (
        report["global_top64_recall"]["probability_favorable"] > 0.95
    )
    assert report["r4800_rmse"]["delta"] < 0
    assert report["protected"]["low_supply"]["groups"] == 2
    assert len(panel_identity(treatment)) == 64


def test_factorial_effects_report_main_effects_and_interaction() -> None:
    c0 = _panel(
        [True, False, True, False, True, False],
        squared_error=128.0,
    )
    t1 = deepcopy(c0)
    t2 = deepcopy(c0)
    t3 = deepcopy(c0)
    t1[1]["top64_recalled"] = True
    t2[3]["top64_recalled"] = True
    t3[1]["top64_recalled"] = True
    t3[3]["top64_recalled"] = True

    effects = factorial_effects(
        {
            "c0-parent-conditioned": c0,
            "t1-supply-query": t1,
            "t2-frontier-query": t2,
            "t3-combined-query": t3,
        }
    )

    assert effects["global_top64_recall"]["supply_main_effect"] == pytest.approx(
        1 / 6
    )
    assert effects["global_top64_recall"][
        "frontier_main_effect"
    ] == pytest.approx(1 / 6)
    assert effects["global_top64_recall"]["interaction"] == pytest.approx(0.0)


def test_panel_alignment_rejects_teacher_identity_drift() -> None:
    control = _panel(
        [True, False, True, False, True, False],
        squared_error=128.0,
    )
    treatment = deepcopy(control)
    treatment[2]["teacher_winner_action_hash"] = "f" * 64

    with pytest.raises(ValueError, match="identical decisions"):
        compare_decision_panels(
            treatment,
            control,
            bootstrap_replicates=100,
        )


def test_panel_identity_reinterprets_signed_group_id_bits() -> None:
    signed = _panel(
        [True, False, True, False, True, False],
        squared_error=128.0,
    )
    signed[0]["group_id"] = -5_482_088_856_184_735_585
    unsigned = deepcopy(signed)
    unsigned[0]["group_id"] &= (1 << 64) - 1

    assert len(panel_identity(signed)) == 64
    report = compare_decision_panels(
        signed,
        unsigned,
        bootstrap_replicates=100,
    )
    assert report["groups"] == 6


@pytest.mark.parametrize("group_id", [-(1 << 63) - 1, 1 << 64, True])
def test_panel_identity_rejects_invalid_group_ids(group_id: object) -> None:
    panel = _panel(
        [True, False, True, False, True, False],
        squared_error=128.0,
    )
    panel[0]["group_id"] = group_id

    with pytest.raises(ValueError, match="group ID"):
        panel_identity(panel)
