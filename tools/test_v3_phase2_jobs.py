from __future__ import annotations

import v3_phase2_jobs as jobs

IMAGE = "registry.example/cascadia/v3@sha256:" + "a" * 64
APPROVAL = "b" * 64


def test_bootstrap_plan_is_exact_and_topology_free() -> None:
    plan = jobs.build_plan(
        {"phase": "bootstrap_collecting", "approved_readiness_sha256": APPROVAL},
        IMAGE,
        100,
    )
    assert plan["games"] == 500_000
    assert plan["work_items"] == 5_000
    assert plan["manual_host_sharding"] is False
    assert all("host" not in str(item).lower() for item in plan["items"])
    assert all(APPROVAL in item["args"] for item in plan["items"])


def test_cycle_plan_rotates_by_game_index_and_uses_registered_epsilon() -> None:
    plan = jobs.build_plan(
        {"phase": "cycle-10-collecting", "approved_readiness_sha256": APPROVAL},
        IMAGE,
        100,
    )
    assert plan["games"] == 10_000
    assert plan["work_items"] == 100
    assert all("0.02" in item["args"] for item in plan["items"])
