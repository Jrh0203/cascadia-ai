from __future__ import annotations

from pathlib import Path

import cluster_research_queue as research_queue
import o1_opponent_intent_policy_corpus_queue as queue


def _specs() -> list[dict]:
    return queue.task_specs(
        bundle_relative=Path(
            f"artifacts/experiments/o1-opponent-intent-policy-heldout-corpus-v1/bundles/{'a' * 64}"
        ),
        bundle_id="a" * 64,
    )


def _by_id() -> dict[str, dict]:
    return {task["id"]: task for task in _specs()}


def test_initial_collection_occupies_three_free_hosts_without_duplication() -> None:
    by_id = _by_id()
    assert len(by_id) == 7
    assert by_id["o1corpus-v1-collect-train-part-0"]["compatible_hosts"] == ["john2"]
    assert by_id["o1corpus-v1-collect-train-part-1"]["compatible_hosts"] == ["john4"]
    assert by_id["o1corpus-v1-collect-validation"]["compatible_hosts"] == ["john1"]
    outputs = {
        by_id[task_id]["artifact_path"]
        for task_id in (
            "o1corpus-v1-collect-train-part-0",
            "o1corpus-v1-collect-train-part-1",
            "o1corpus-v1-collect-validation",
        )
    }
    assert len(outputs) == 3


def test_held_out_policy_requirements_are_frozen_in_commands() -> None:
    by_id = _by_id()
    validation = by_id["o1corpus-v1-collect-validation"]["command"]
    test = by_id["o1corpus-v1-collect-test"]["command"]
    stress = by_id["o1corpus-v1-collect-final-stress"]["command"]
    assert validation[validation.index("--required-policy") + 1] == "pattern-competition"
    assert test[test.index("--required-policy") + 1] == "pattern-portfolio"
    assert stress[stress.index("--required-policy") + 1] == "random"
    assert (
        "pattern-competition"
        not in by_id["o1corpus-v1-collect-train-part-0"]["command"][
            by_id["o1corpus-v1-collect-train-part-0"]["command"].index("--policy-pool") + 1
        ]
    )


def test_follow_on_work_is_work_conserving_and_manifests_are_collected() -> None:
    by_id = _by_id()
    assert by_id["o1corpus-v1-collect-test"]["dependencies"] == ["o1corpus-v1-collect-train-part-0"]
    assert by_id["o1corpus-v1-collect-final-stress"]["dependencies"] == [
        "o1corpus-v1-collect-train-part-1"
    ]
    collect = by_id["o1corpus-v1-collect-manifests"]
    assert len(collect["dependencies"]) == 5
    assert collect["command"].count("--artifact") == 5


def test_campaign_is_valid_under_production_queue_schema() -> None:
    state = research_queue.empty_queue("o1-policy-corpus-test", now_ms=1_000)
    research_queue.add_tasks(state, _specs(), now_ms=1_001)
    research_queue.validate_queue(state)
    assert len(state["tasks"]) == 7
