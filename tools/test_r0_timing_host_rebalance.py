from __future__ import annotations

from pathlib import Path

import cluster_research_queue as queue
import pytest
import r0_spatial_campaign as campaign
import r0_spatial_rebalance as dataset_rebalance
import r0_timing_host_rebalance as timing_rebalance
import r0_timing_recovery as recovery


def _complete(task: dict, host: str) -> None:
    task["status"] = "completed"
    task["claim"] = None
    task["result"] = {
        "artifact": task["artifact_path"],
        "error": None,
        "completed_unix_ms": 10**15,
        "host": host,
    }


def _ready_state() -> dict:
    state = queue.empty_queue("campaign", now_ms=1)
    bundle = Path("artifacts/experiments/r0/bundles/example")
    for index, specification in enumerate(
        campaign.build_task_specs(bundle_relative=bundle),
        start=2,
    ):
        queue.add_task(state, specification, now_ms=index)
    dataset_rebalance.apply_rebalance(
        state,
        dataset_rebalance.build_rebalance(
            bundle_relative=bundle,
            train_host="john3",
            validation_host="john4",
        ),
    )

    by_id = {task["id"]: task for task in state["tasks"]}
    for task in state["tasks"]:
        if task["status"] != "cancelled" and (
            task["id"].startswith("r0f-fanout-")
            or task["id"].startswith("r0f-rebalanced-fanout-")
        ):
            _complete(task, task["compatible_hosts"][0])
    queue.refresh_dependencies(state)

    by_id["r0f-benchmark-shard-0-replicate-0"]["status"] = "completed"
    by_id["r0f-benchmark-shard-0-replicate-0"]["result"] = {
        "artifact": "contaminated.json",
        "error": None,
        "completed_unix_ms": 101,
        "host": "john1",
    }
    by_id["r0f-benchmark-shard-0-replicate-1"]["status"] = "failed"
    by_id["r0f-benchmark-shard-0-replicate-1"]["result"] = {
        "artifact": None,
        "error": "contaminated",
        "completed_unix_ms": 102,
        "host": "john1",
    }
    by_id["r0f-benchmark-shard-0-replicate-2"]["status"] = "cancelled"
    recovery.apply_recovery(state, recovery.build_recovery(bundle_relative=bundle))

    by_id = {task["id"]: task for task in state["tasks"]}
    for replicate_index in range(campaign.REQUIRED_REPLICATES):
        _complete(
            by_id[f"r0f-clean-benchmark-shard-0-replicate-{replicate_index}"],
            "john1",
        )
        _complete(
            by_id[f"r0f-benchmark-shard-2-replicate-{replicate_index}"],
            "john3",
        )
        _complete(
            by_id[f"r0f-benchmark-shard-3-replicate-{replicate_index}"],
            "john4",
        )
    queue.refresh_dependencies(state)
    queue.validate_queue(state)
    return state


def test_build_rebalance_moves_all_three_replicas_to_one_host() -> None:
    plan = timing_rebalance.build_rebalance(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        replacement_host="john3",
    )
    by_id = {task["id"]: task for task in plan["replacement_tasks"]}
    for replicate_index in range(campaign.REQUIRED_REPLICATES):
        task = by_id[
            f"r0f-work-conserving-benchmark-shard-1-replicate-{replicate_index}"
        ]
        assert task["compatible_hosts"] == ["john3"]
        assert task["command"][task["command"].index("--shard-index") + 1] == "1"
        assert task["command"][task["command"].index("--replicate-index") + 1] == str(
            replicate_index
        )
        assert "john3-source-frozen-reassigned-shard-1" in task["artifact_path"]

    forward = by_id["r0f-work-conserving-extraction-classification-forward"]
    assert forward["command"].count("--report") == 12
    shard_one_reports = [
        value
        for index, value in enumerate(forward["command"])
        if index > 0
        and forward["command"][index - 1] == "--report"
        and "reassigned-shard-1" in value
    ]
    assert len(shard_one_reports) == 3


def test_apply_rebalance_preserves_completed_evidence_and_rewires_classifier() -> None:
    state = _ready_state()
    plan = timing_rebalance.build_rebalance(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        replacement_host="john3",
    )
    timing_rebalance.apply_rebalance(state, plan)
    by_id = {task["id"]: task for task in state["tasks"]}

    for replicate_index in range(campaign.REQUIRED_REPLICATES):
        assert by_id[f"r0f-benchmark-shard-1-replicate-{replicate_index}"][
            "status"
        ] == "cancelled"
        assert by_id[f"r0f-clean-benchmark-shard-0-replicate-{replicate_index}"][
            "status"
        ] == "completed"
        assert by_id[f"r0f-benchmark-shard-2-replicate-{replicate_index}"][
            "status"
        ] == "completed"
        assert by_id[f"r0f-benchmark-shard-3-replicate-{replicate_index}"][
            "status"
        ] == "completed"
        assert by_id[
            f"r0f-work-conserving-benchmark-shard-1-replicate-{replicate_index}"
        ]["status"] == "ready"

    assert by_id["r0f-clean-benchmark-report-collection"]["status"] == "cancelled"
    assert by_id["r0f-work-conserving-benchmark-report-collection"]["status"] == (
        "blocked"
    )
    queue.validate_queue(state)


def test_apply_rebalance_fails_before_mutation_if_a_source_task_started() -> None:
    state = _ready_state()
    by_id = {task["id"]: task for task in state["tasks"]}
    by_id["r0f-benchmark-shard-1-replicate-0"]["status"] = "running"
    by_id["r0f-benchmark-shard-1-replicate-0"]["claim"] = {
        "host": "john2",
        "token": "token",
        "claimed_unix_ms": 1,
        "heartbeat_unix_ms": 1,
        "lease_expires_unix_ms": 100,
    }
    plan = timing_rebalance.build_rebalance(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
        replacement_host="john3",
    )

    with pytest.raises(timing_rebalance.TimingRebalanceError, match="safely"):
        timing_rebalance.apply_rebalance(state, plan)
    assert not any(task["id"].startswith("r0f-work-conserving") for task in state["tasks"])


def test_replacement_host_is_restricted_to_free_remote_lanes() -> None:
    with pytest.raises(timing_rebalance.TimingRebalanceError, match="john3 or john4"):
        timing_rebalance.build_rebalance(
            bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
            replacement_host="john2",
        )
