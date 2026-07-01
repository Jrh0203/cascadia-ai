from __future__ import annotations

from s4_candidate_set_mlx_report import (
    ARMS,
    _external_rescue_gates,
    _material_context_gates,
    _performance_rescue_gates,
    _quality_gates,
    _select_arm,
)


def _report(
    *,
    mae: float = 1.40,
    rmse: float = 1.85,
    recall: float = 0.75,
    regret: float = 0.10,
    coverage: float = 1.0,
    low_supply_recall: float = 0.90,
    independent_recall: float = 0.80,
    throughput: float = 30_000.0,
    p99: float = 150.0,
    active: float = 300_000_000.0,
    rss: float = 500_000_000.0,
) -> dict:
    return {
        "metrics": {
            "r4800_value": {"mae": mae, "rmse": rmse},
            "top64_r4800_winner_recall": recall,
            "mean_top64_retained_r4800_regret": regret,
            "top64_confidence_set_coverage_95": coverage,
            "subsets": {
                "low_supply": {
                    "top64_r4800_winner_recall": low_supply_recall,
                },
                "independent_draft_winner": {
                    "top64_r4800_winner_recall": independent_recall,
                },
            },
        },
        "performance": {
            "fixed_chunk": {
                "action_scores_per_second": throughput,
            },
            "complete_decisions": {
                "latency_milliseconds": {"p99": p99},
            },
            "memory": {
                "peak_active_bytes": active,
                "peak_process_rss_bytes": rss,
            },
        },
    }


def test_context_treatment_must_be_nondegraded_and_material() -> None:
    control = _report(coverage=0.97)
    treatment = _report(
        mae=1.33,
        rmse=1.78,
        recall=0.78,
        regret=0.07,
        coverage=1.0,
    )

    assert all(_quality_gates(treatment, control).values())
    assert any(_material_context_gates(treatment, control).values())

    degraded = _report(recall=0.70, coverage=0.98)
    assert not all(_quality_gates(degraded, control).values())


def test_compact_rescue_is_bound_to_full_afterstate_quality_and_speed() -> None:
    r3_control = _report(
        mae=1.35,
        rmse=1.80,
        recall=0.78,
        regret=0.08,
        throughput=25_000.0,
        p99=160.0,
        active=400_000_000.0,
        rss=600_000_000.0,
    )
    rescued = _report(
        mae=1.36,
        rmse=1.82,
        recall=0.78,
        regret=0.08,
        throughput=22_000.0,
        p99=190.0,
        active=450_000_000.0,
        rss=700_000_000.0,
    )

    assert all(_external_rescue_gates(rescued, r3_control).values())
    assert all(_performance_rescue_gates(rescued, r3_control).values())

    too_slow = _report(throughput=6_000.0)
    assert not all(_performance_rescue_gates(too_slow, r3_control).values())


def test_selected_rescue_prefers_ranking_then_regret_then_value() -> None:
    reports = {
        ARMS[1]: _report(recall=0.80, regret=0.08, mae=1.35),
        ARMS[2]: _report(recall=0.82, regret=0.09, mae=1.34),
        ARMS[3]: _report(recall=0.82, regret=0.07, mae=1.36),
    }

    assert _select_arm(list(reports), reports) == ARMS[3]
