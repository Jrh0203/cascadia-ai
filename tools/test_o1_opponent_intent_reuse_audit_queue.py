from __future__ import annotations

from pathlib import Path

import cluster_research_queue as research_queue
import o1_opponent_intent_reuse_audit_queue as queue


def _specs() -> list[dict]:
    return queue.task_specs(
        bundle_relative=Path(
            "artifacts/experiments/o1-opponent-intent-corpus-reuse-audit-v1/bundles/" + "a" * 64
        ),
        bundle_id="a" * 64,
    )


def _by_id() -> dict[str, dict]:
    return {task["id"]: task for task in _specs()}


def test_campaign_runs_primary_and_replay_on_distinct_free_hosts() -> None:
    by_id = _by_id()
    assert len(by_id) == 7
    primary = by_id["o1reuse-v3-primary"]
    replay = by_id["o1reuse-v3-replay"]
    assert primary["compatible_hosts"] == ["john4"]
    assert replay["compatible_hosts"] == ["john2"]
    assert primary["dependencies"] == replay["dependencies"]
    assert len(primary["dependencies"]) == 3
    assert primary["command"][3].endswith("/bin/opponent_intent_reuse_audit")
    assert replay["command"][3].endswith("/bin/opponent_intent_reuse_audit")
    assert primary["resources"]["cpu_cores"] == 10
    assert replay["resources"]["cpu_cores"] == 10


def test_fanout_tasks_bind_bundle_and_both_datasets() -> None:
    by_id = _by_id()
    bundle = by_id["o1reuse-v3-bundle-fanout"]
    assert "bin/opponent_intent_reuse_audit" in bundle["command"]
    train = by_id["o1reuse-v3-train-fanout"]
    validation = by_id["o1reuse-v3-validation-fanout"]
    for task in (train, validation):
        assert task["compatible_hosts"] == ["john1"]
        assert task["command"].count("--destination") == 2
        assert task["command"].count("--verify-tree") == 1


def test_collection_and_classifier_are_fail_closed() -> None:
    by_id = _by_id()
    collect = by_id["o1reuse-v3-collect"]
    assert collect["dependencies"] == ["o1reuse-v3-primary", "o1reuse-v3-replay"]
    assert collect["command"].count("--artifact") == 2
    classify = by_id["o1reuse-v3-classify"]
    assert classify["dependencies"] == ["o1reuse-v3-collect"]
    assert classify["decision_terminal"] is True
    assert classify["command"][1].endswith("/source/tools/o1_opponent_intent_reuse_audit_report.py")
    assert classify["command"].count("--canonical-output") == 1


def test_every_launch_artifact_is_namespaced_by_bundle_identity() -> None:
    bundle_id = "a" * 64
    launch_fragment = f"/launches/{bundle_id}/"
    for task in _specs():
        assert launch_fragment in f"/{task['artifact_path']}"


def test_campaign_is_valid_under_production_queue_schema() -> None:
    state = research_queue.empty_queue("o1-reuse-test", now_ms=1_000)
    research_queue.add_tasks(state, _specs(), now_ms=1_001)
    research_queue.validate_queue(state)
    assert len(state["tasks"]) == 7
