from __future__ import annotations

import v3_promotion as promotion


def _records(delta: float, pairs: int = 500) -> list[dict[str, object]]:
    return [
        {
            "tier": tier,
            "pair_index": index,
            "paired_delta": delta,
            "integrity_passed": True,
            "resource_regression": False,
        }
        for tier in promotion.TIERS
        for index in range(pairs)
    ]


def test_strong_candidate_crosses_alternative_boundary() -> None:
    assert promotion.evaluate(_records(0.15))["verdict"] == "promote"


def test_regression_retains_incumbent() -> None:
    assert promotion.evaluate(_records(-0.10))["verdict"] == "retain-incumbent"


def test_resource_regression_blocks_promotion() -> None:
    records = _records(20.0)
    records[0]["resource_regression"] = True
    assert "resource-or-integrity" in promotion.evaluate(records)["verdict"]


def test_decision_estimand_clips_outliers_but_reports_raw_mean() -> None:
    records = _records(0.15)
    records[0]["paired_delta"] = 200.0
    result = promotion.evaluate(records)
    tier = result["tiers"][promotion.TIERS[0]]
    assert tier["mean_delta"] != tier["decision_mean_delta"]
    assert result["registered"]["decision_delta_bounds"] == [-25.0, 25.0]
    assert result["schema_id"] == "cascadia-v3-always-valid-promotion-v2"


def test_cycle_one_rule_remains_exactly_reproducible() -> None:
    import v3_promotion_v1 as frozen

    assert frozen.evaluate(_records(0.15))["verdict"] == "retain-incumbent-inconclusive"
    assert frozen.evaluate(_records(20.0))["verdict"] == "promote"
