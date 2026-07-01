from __future__ import annotations

import json
from pathlib import Path

import pytest
from frontier_arbitrary_precision_report import (
    validate_replay_comparisons,
    validate_scheduler_state,
    validate_source_identities,
)


def test_source_identity_requires_four_matching_hosts(tmp_path: Path) -> None:
    paths = []
    for host in ("john1", "john2", "john3", "john4"):
        path = tmp_path / f"{host}.json"
        path.write_text(
            json.dumps(
                {
                    "host": host,
                    "files": 115,
                    "bundle_sha256": "a" * 64,
                }
            )
        )
        paths.append(path)
    assert validate_source_identities(paths)["files"] == 115
    paths[-1].write_text(
        json.dumps(
            {
                "host": "john4",
                "files": 115,
                "bundle_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="not identical"):
        validate_source_identities(paths)


def test_replays_require_24_cross_host_matches(tmp_path: Path) -> None:
    paths = []
    for group_index in range(24):
        path = tmp_path / f"group-{group_index}.json"
        path.write_text(
            json.dumps(
                {
                    "group_index": group_index,
                    "origin_host": "john1",
                    "replay_host": "john2",
                    "scientific_payload_identical": True,
                }
            )
        )
        paths.append(path)
    assert len(validate_replay_comparisons(paths)["reports"]) == 24
    paths[-1].write_text(
        json.dumps(
            {
                "group_index": 23,
                "origin_host": "john1",
                "replay_host": "john1",
                "scientific_payload_identical": True,
            }
        )
    )
    with pytest.raises(ValueError, match="origin host"):
        validate_replay_comparisons(paths)


def test_scheduler_requires_complete_cross_host_task_graph() -> None:
    tasks = {}
    for group_index in range(24):
        tasks[f"origin-{group_index:02d}"] = {
            "kind": "origin",
            "group_index": group_index,
            "status": "done",
            "host": "john1",
        }
        tasks[f"replay-{group_index:02d}"] = {
            "kind": "replay",
            "group_index": group_index,
            "status": "done",
            "host": "john2",
        }
    state = {
        "experiment_id": (
            "complete-action-frontier-arbitrary-precision-control-v1"
        ),
        "tasks": tasks,
    }
    assert validate_scheduler_state(state) is state
    tasks["replay-00"]["host"] = "john1"
    with pytest.raises(ValueError, match="origin host"):
        validate_scheduler_state(state)
