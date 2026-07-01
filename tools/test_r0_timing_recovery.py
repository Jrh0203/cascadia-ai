from __future__ import annotations

from pathlib import Path

import cluster_research_queue as queue
import pytest
import r0_spatial_campaign as campaign
import r0_spatial_rebalance as rebalance
import r0_timing_recovery as recovery


def _state_with_contaminated_local_wave() -> dict:
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
    queue.finish_task(
        state,
        task_id=claimed["id"],
        host="john1",
        token=claimed["claim"]["token"],
        outcome="completed",
        now_ms=101,
    )
    rebalance.apply_rebalance(
        state,
        rebalance.build_rebalance(
            bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
            train_host="john3",
            validation_host="john4",
        ),
    )
    by_id = {task["id"]: task for task in state["tasks"]}
    by_id["r0f-benchmark-shard-0-replicate-0"]["status"] = "completed"
    by_id["r0f-benchmark-shard-0-replicate-0"]["result"] = {
        "artifact": "contaminated.json",
        "error": None,
        "completed_unix_ms": 200,
        "host": "john1",
    }
    by_id["r0f-benchmark-shard-0-replicate-1"]["status"] = "failed"
    by_id["r0f-benchmark-shard-0-replicate-1"]["result"] = {
        "artifact": None,
        "error": "terminated after load contamination",
        "completed_unix_ms": 201,
        "host": "john1",
    }
    by_id["r0f-benchmark-shard-0-replicate-2"]["status"] = "cancelled"
    return state


def test_recovery_builds_three_new_local_processes_and_clean_classifier() -> None:
    plan = recovery.build_recovery(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
    )
    by_id = {task["id"]: task for task in plan["replacement_tasks"]}
    for replicate_index in range(3):
        task = by_id[f"r0f-clean-benchmark-shard-0-replicate-{replicate_index}"]
        assert task["compatible_hosts"] == ["john1"]
        assert task["command"][task["command"].index("--replicate-index") + 1] == (
            str(replicate_index)
        )
        assert "source-frozen-clean-shard-0" in task["artifact_path"]
    forward = by_id["r0f-clean-extraction-classification-forward"]
    assert forward["command"].count("--report") == 12
    assert all(
        "john1-source-frozen-clean" in report
        for index, report in enumerate(forward["command"])
        if index > 0 and forward["command"][index - 1] == "--report" and "john1" in report
    )


def test_recovery_preserves_contaminated_artifact_and_replaces_downstream() -> None:
    state = _state_with_contaminated_local_wave()
    plan = recovery.build_recovery(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
    )
    recovery.apply_recovery(state, plan)
    by_id = {task["id"]: task for task in state["tasks"]}
    assert by_id["r0f-benchmark-shard-0-replicate-0"]["status"] == "completed"
    assert by_id["r0f-benchmark-shard-0-replicate-0"]["result"]["artifact"] == ("contaminated.json")
    assert by_id["r0f-benchmark-shard-0-replicate-1"]["status"] == "cancelled"
    assert by_id["r0f-benchmark-report-collection"]["status"] == "cancelled"
    assert by_id["r0f-clean-benchmark-shard-0-replicate-0"]["status"] == ("blocked")
    assert by_id["r0f-clean-extraction-classification-forward"]["status"] == ("blocked")
    queue.validate_queue(state)


def test_recovery_rejects_missing_quarantine_evidence_atomically() -> None:
    state = _state_with_contaminated_local_wave()
    by_id = {task["id"]: task for task in state["tasks"]}
    by_id["r0f-benchmark-shard-0-replicate-0"]["status"] = "ready"
    by_id["r0f-benchmark-shard-0-replicate-0"]["result"] = None
    plan = recovery.build_recovery(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
    )
    with pytest.raises(recovery.RecoveryError, match="quarantined"):
        recovery.apply_recovery(state, plan)
    assert "r0f-clean-benchmark-shard-0-replicate-0" not in by_id
