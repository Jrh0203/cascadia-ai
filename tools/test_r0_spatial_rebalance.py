from __future__ import annotations

from pathlib import Path

import cluster_research_queue as queue
import pytest
import r0_spatial_campaign as campaign
import r0_spatial_rebalance as rebalance


def _campaign_state() -> dict:
    state = queue.empty_queue("campaign", now_ms=1)
    for index, specification in enumerate(
        campaign.build_task_specs(
            bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        ),
        start=2,
    ):
        queue.add_task(state, specification, now_ms=index)
    claimed = queue.claim_next(state, host="john1", lease_seconds=30, now_ms=100)
    assert claimed is not None
    assert claimed["id"] == "r0f-production-bundle-fanout"
    queue.finish_task(
        state,
        task_id=claimed["id"],
        host="john1",
        token=claimed["claim"]["token"],
        outcome="completed",
        now_ms=101,
    )
    return state


def test_rebalance_specs_use_the_assigned_hosts_and_exact_intervals() -> None:
    plan = rebalance.build_rebalance(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        train_host="john3",
        validation_host="john4",
    )
    tasks = {task["id"]: task for task in plan["replacement_tasks"]}
    train = tasks["r0f-rebalanced-collect-train-part-1"]
    validation = tasks["r0f-rebalanced-collect-validation-part-1"]
    assert train["compatible_hosts"] == ["john3"]
    assert validation["compatible_hosts"] == ["john4"]
    assert train["command"][2].startswith("/Users/john3/")
    assert validation["command"][2].startswith("/Users/john4/")
    assert train["command"][train["command"].index("--first-game-index") + 1] == ("200157")
    assert validation["command"][validation["command"].index("--first-game-index") + 1] == "210032"
    assert plan["dependency_mapping"] == {
        "r0f-fanout-train-part-1": "r0f-rebalanced-fanout-train-part-1",
        "r0f-fanout-validation-part-1": ("r0f-rebalanced-fanout-validation-part-1"),
    }


def test_rebalance_is_atomic_and_rewires_every_benchmark() -> None:
    state = _campaign_state()
    plan = rebalance.build_rebalance(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        train_host="john3",
        validation_host="john4",
    )
    rebalance.apply_rebalance(state, plan)

    by_id = {task["id"]: task for task in state["tasks"]}
    for task_id in plan["cancel_task_ids"]:
        assert by_id[task_id]["status"] == "cancelled"
        assert by_id[task_id]["administrative_cancellation"]["actor"] == ("research-coordinator")
    assert by_id["r0f-rebalanced-collect-train-part-1"]["status"] == "ready"
    assert by_id["r0f-rebalanced-collect-validation-part-1"]["status"] == ("ready")
    for task_id in plan["benchmark_task_ids"]:
        dependencies = by_id[task_id]["dependencies"]
        assert "r0f-fanout-train-part-1" not in dependencies
        assert "r0f-fanout-validation-part-1" not in dependencies
        assert "r0f-rebalanced-fanout-train-part-1" in dependencies
        assert "r0f-rebalanced-fanout-validation-part-1" in dependencies
    queue.validate_queue(state)


def test_rebalance_rejects_duplicate_host_assignment() -> None:
    with pytest.raises(rebalance.RebalanceError, match="parallel"):
        rebalance.build_rebalance(
            bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
            train_host="john3",
            validation_host="john3",
        )


def test_rebalance_rejects_a_benchmark_that_already_started() -> None:
    state = _campaign_state()
    benchmark = next(
        task for task in state["tasks"] if task["id"] == "r0f-benchmark-shard-0-replicate-0"
    )
    benchmark["status"] = "failed"
    benchmark["result"] = {
        "artifact": None,
        "error": "already attempted",
        "completed_unix_ms": 200,
        "host": "john1",
    }
    plan = rebalance.build_rebalance(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        train_host="john3",
        validation_host="john4",
    )
    with pytest.raises(rebalance.RebalanceError, match="already started"):
        rebalance.apply_rebalance(state, plan)
    assert all(task.get("administrative_cancellation") is None for task in state["tasks"])
