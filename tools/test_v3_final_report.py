from __future__ import annotations

import v3_final_report as report


def _anatomy(score: float) -> dict[str, float]:
    return {
        "bear": score / 10,
        "elk": score / 10,
        "salmon": score / 10,
        "hawk": score / 10,
        "fox": score / 10,
        "wildlife_total": score / 2,
        "forest": score / 15,
        "mountain": score / 15,
        "prairie": score / 15,
        "wetland": score / 15,
        "river": score / 15,
        "terrain_total": score / 3,
        "nature_tokens": 2,
        "pinecones": 2,
        "overflow_states": 0,
    }


def test_final_report_requires_lower_bound_to_claim_goal() -> None:
    protected = {
        "pairs": [
            {
                "treatment": {
                    "score": 102 + index % 2,
                    "anatomy": _anatomy(102),
                    "focal_seconds": 4.0,
                },
                "control": {
                    "score": 98 + index % 2,
                    "anatomy": _anatomy(98),
                    "focal_seconds": 5.0,
                },
            }
            for index in range(250)
        ],
        "resource_metrics": {"worker_elapsed_seconds": 2_000.0},
    }
    all_v3 = {
        "games": [
            {
                "seats": [
                    {
                        "score": 101 + (index % 2),
                        "anatomy": _anatomy(101),
                        "decision_seconds": 8.0,
                    }
                    for _ in range(4)
                ]
            }
            for index in range(1_000)
        ],
        "resource_metrics": {"worker_elapsed_seconds": 8_000.0},
    }
    value = report.build_report(protected, all_v3)
    assert value["protected_pairs"]["classification"] == "outperforming"
    assert value["all_v3"]["goal_100_claimed"] is True
    assert value["recommendation"] == "goal-achieved"
    assert value["protected_pairs"]["throughput"]["items_per_worker_second"] == 0.25
    assert value["all_v3"]["throughput"]["items_per_worker_second"] == 0.125
    assert value["all_v3"]["anatomy"]["bear"]["p50"] == 10.1
    assert value["protected_pairs"]["latency"]["treatment_focal_game"]["p90_seconds"] == 4
    assert "# Cascadia V3 Final Campaign Report" in report.render_markdown(value)


def test_final_report_requires_complete_ordered_cycle_history() -> None:
    protected = {
        "pairs": [
            {
                "treatment": {"score": 1, "anatomy": _anatomy(1), "focal_seconds": 1},
                "control": {"score": 0, "anatomy": _anatomy(0), "focal_seconds": 1},
            }
            for _ in range(250)
        ]
    }
    all_v3 = {
        "games": [
            {
                "seats": [
                    {"score": 1, "anatomy": _anatomy(1), "decision_seconds": 1}
                    for _ in range(4)
                ]
            }
            for _ in range(1_000)
        ]
    }
    history = [{"cycle": cycle} for cycle in range(1, 11)]
    value = report.build_report(protected, all_v3, campaign_history=history)
    assert len(value["expert_iteration_learning_curves"]) == 10
