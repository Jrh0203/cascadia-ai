from __future__ import annotations

from pathlib import Path

from v1_score_anatomy_queue import (
    EXPERIMENT_ID,
    ROLE_HOSTS,
    campaign_spec,
    task_specs,
)


def test_queue_launches_matched_roles_on_all_four_hosts() -> None:
    bundle_id = "a" * 64
    bundle = Path(
        "artifacts/experiments/v1-score-anatomy-matched-r2-v1/"
        f"bundles/{bundle_id}"
    )
    tasks = task_specs(bundle_relative=bundle, bundle_id=bundle_id)
    by_id = {task["id"]: task for task in tasks}
    assert len(tasks) == 6
    for role, host in ROLE_HOSTS.items():
        task = by_id[f"v1anatomy-v1-run-{role}"]
        assert task["compatible_hosts"] == [host]
        assert task["resources"]["uses_mlx"] is True
        assert task["priority"] == 90
    classifier = by_id["v1anatomy-v1-classify"]
    assert classifier["decision_terminal"] is True
    assert classifier["dependencies"] == ["v1anatomy-v1-collect"]
    campaign = campaign_spec(tasks, bundle_id=bundle_id)
    assert campaign["experiment_id"] == EXPERIMENT_ID
    assert campaign["task_count"] == 6


def test_queue_dependencies_are_closed_and_acyclic() -> None:
    tasks = task_specs(
        bundle_relative=Path("bundles") / ("b" * 64),
        bundle_id="b" * 64,
    )
    by_id = {task["id"]: task for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise AssertionError("cycle")
        visiting.add(task_id)
        for dependency in by_id[task_id]["dependencies"]:
            assert dependency in by_id
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)
    assert visited == set(by_id)
