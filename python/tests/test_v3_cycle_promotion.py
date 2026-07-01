from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "tools/v3_cycle_promotion.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v3_cycle_promotion_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_tier_matches_rust_clap_value_enum() -> None:
    module = _module()
    assert {tier: module.worker_tier(tier) for tier in module.TIERS} == {
        "direct": "direct",
        "k32-r64": "k32r64",
        "k32-r600": "k32r600",
        "equal-wall-time": "equal-wall-time",
    }


def test_worker_tier_rejects_unregistered_tier() -> None:
    module = _module()
    with pytest.raises(module.CyclePromotionError, match="unknown promotion tier"):
        module.worker_tier("k32-r6")


def test_promotion_worker_contract_pins_search_environment_and_conditioning() -> None:
    module = _module()
    assert module.PROMOTION_SEARCH_ENV == {
        "MCE_LMR": "1",
        "MCE_DIVERSE_PREFILTER": "1",
    }
    assert module.request_id(1, 0, 100) == (
        "v3-cycle-01-promotion-000-100-v3-conditioned-rollout"
    )


def test_promotion_balances_model_staging_against_the_29_cpu_tail(
    tmp_path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.PAIR_INCREMENT == 100
    assert module.PAIRS_PER_ITEM == 4
    assert len(module.TIERS) * module.PAIR_INCREMENT // module.PAIRS_PER_ITEM == 100
    pairs, order = module.request_plan(5, 0, 100)
    assert pairs == 4
    assert order[:8] == (
        ("direct", 0),
        ("k32-r64", 0),
        ("k32-r600", 0),
        ("equal-wall-time", 0),
        ("direct", 4),
        ("k32-r64", 4),
        ("k32-r600", 4),
        ("equal-wall-time", 4),
    )


def test_opened_request_recovers_its_frozen_shard_size(tmp_path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.pairs_per_item_for_request(4, 300, 100) == 4

    request = module.request_id(4, 300, 100)
    path = tmp_path / "phase2/control/cluster-client/requests" / f"{request}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_id": "cascadia.cluster.managed-request-state.v2",
                "request_id": request,
                "items": [
                    {
                        "job_payload": {
                            "Meta": {
                                "cascadia.app.pairs": "5",
                                "cascadia.app.tier": tier,
                                "cascadia.app.first_pair_index": str(first),
                            },
                        }
                    }
                    for tier in module.TIERS
                    for first in range(300, 400, 5)
                ],
            }
        )
    )
    pairs, order = module.request_plan(4, 300, 100)
    assert pairs == 5
    assert order[:2] == (("direct", 300), ("direct", 305))


def test_opened_request_rejects_an_inconsistent_shard_domain(tmp_path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    request = module.request_id(4, 300, 100)
    path = tmp_path / "phase2/control/cluster-client/requests" / f"{request}.json"
    path.parent.mkdir(parents=True)
    items = [
        {
            "job_payload": {
                "Meta": {
                    "cascadia.app.pairs": "5",
                    "cascadia.app.tier": tier,
                    "cascadia.app.first_pair_index": str(first),
                }
            }
        }
        for tier in module.TIERS
        for first in range(300, 400, 5)
    ]
    items[-1]["job_payload"]["Meta"]["cascadia.app.pairs"] = "1"
    path.write_text(
        json.dumps(
            {
                "schema_id": "cascadia.cluster.managed-request-state.v2",
                "request_id": request,
                "items": items,
            }
        )
    )
    with pytest.raises(module.CyclePromotionError, match="shard domain"):
        module.pairs_per_item_for_request(4, 300, 100)


def test_cycle_one_rule_is_frozen_and_future_cycles_use_v2() -> None:
    module = _module()
    assert module.promotion_rule(1).__name__ == "v3_promotion_v1"
    assert module.promotion_rule(2).__name__ == "v3_promotion"
    assert module.promotion_rule(10).__name__ == "v3_promotion"
