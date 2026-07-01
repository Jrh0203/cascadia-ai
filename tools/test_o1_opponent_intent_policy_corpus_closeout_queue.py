from __future__ import annotations

from pathlib import Path

import cluster_research_queue as research_queue
import o1_opponent_intent_policy_corpus_closeout_queue as queue


def _specs() -> list[dict]:
    return queue.task_specs(
        bundle_relative=Path(
            "artifacts/experiments/"
            "o1-opponent-intent-policy-heldout-corpus-v1/"
            f"audit-bundles/{'a' * 64}"
        ),
        bundle_id="a" * 64,
    )


def _by_id() -> dict[str, dict]:
    return {task["id"]: task for task in _specs()}


def test_closeout_uses_distinct_hosts_and_exact_tree_fanout() -> None:
    by_id = _by_id()
    assert len(by_id) == 7
    assert by_id["o1corpus-v1-closeout-primary"]["compatible_hosts"] == ["john1"]
    assert by_id["o1corpus-v1-closeout-replay"]["compatible_hosts"] == ["john2"]
    assert "--verify-tree" in by_id["o1corpus-v1-closeout-bundle-fanout"]["command"]
    assert "--verify-tree" in by_id["o1corpus-v1-closeout-dataset-fanout"]["command"]


def test_closeout_dependency_graph_is_complete() -> None:
    by_id = _by_id()
    assert by_id["o1corpus-v1-closeout-collect-trees"]["dependencies"] == [
        queue.COLLECTION_DEPENDENCY
    ]
    assert by_id["o1corpus-v1-closeout-replay"]["dependencies"] == [
        "o1corpus-v1-closeout-bundle-fanout",
        "o1corpus-v1-closeout-dataset-fanout",
    ]
    assert by_id["o1corpus-v1-closeout-classify"]["dependencies"] == [
        "o1corpus-v1-closeout-primary",
        "o1corpus-v1-closeout-collect-replay",
    ]
    assert by_id["o1corpus-v1-closeout-classify"]["decision_terminal"] is True


def test_audit_commands_bind_every_frozen_dataset_role() -> None:
    by_id = _by_id()
    for task_id in (
        "o1corpus-v1-closeout-primary",
        "o1corpus-v1-closeout-replay",
    ):
        command = by_id[task_id]["command"]
        assert command.count("--dataset") == 5
        for role in queue.DATASET_ROLES:
            assert any(
                isinstance(argument, str) and argument.startswith(f"{role}=")
                for argument in command
            )


def test_closeout_is_valid_with_existing_collection_dependency() -> None:
    state = research_queue.empty_queue("o1-closeout-test", now_ms=1_000)
    prerequisite = {
        "id": queue.COLLECTION_DEPENDENCY,
        "title": "Existing collection prerequisite",
        "experiment_id": queue.EXPERIMENT_ID,
        "decision": "Represent the installed completed collection graph",
        "workload_class": "shared-prerequisite",
        "compatible_hosts": ["john1"],
        "dependencies": [],
        "command": ["true"],
        "artifact_path": "artifacts/existing.json",
        "stop_rule": "Existing task placeholder for schema validation.",
        "expected_runtime_seconds": 1,
        "resources": {
            "cpu_cores": 1,
            "memory_gib": 1.0,
            "uses_mlx": False,
        },
    }
    research_queue.add_tasks(state, [prerequisite, *_specs()], now_ms=1_001)
    research_queue.validate_queue(state)
    assert len(state["tasks"]) == 8
