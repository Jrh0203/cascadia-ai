import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.all_wildlife_profile_collect import collect
from tools.all_wildlife_profile_taskset import build_taskset

CASES = [
    "AAAAA:6,1,6,2,5:69",
    "AAAAA:4,2,6,2,6:69",
    "CADAC:0,2,6,6,6:67",
]


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _fixture(tmp_path: Path, *, exact: bool = False) -> tuple[Path, Path, list[Path]]:
    taskset = build_taskset(CASES)
    taskset_path = tmp_path / "taskset.json"
    _write(taskset_path, taskset)

    from tools.all_wildlife_profile_collect import _sha256

    identities = {
        "taskset_sha256": _sha256(taskset_path),
        "rules_source_sha256": _sha256(Path("tools/all_wildlife_rules.py")),
        "exact_source_sha256": _sha256(Path("tools/all_wildlife_exact.py")),
        "exact_support_source_sha256": _sha256(Path("tools/cbddb_wildlife_exact.py")),
        "runner_source_sha256": _sha256(Path("tools/all_wildlife_profile_proof.py")),
    }
    fleet = {
        "schema": "all-wildlife-score-profile-fleet-v1",
        "state": "running",
        "tag": "test",
        "launch_utc": "2026-07-23T00:00:00Z",
        "source_revision": "test",
        "taskset_sha256": identities["taskset_sha256"],
        "rules_source_sha256": identities["rules_source_sha256"],
        "exact_source_sha256": identities["exact_source_sha256"],
        "exact_support_sha256": identities["exact_support_source_sha256"],
        "runner_source_sha256": identities["runner_source_sha256"],
        "configuration": {
            "seconds_per_profile": 30,
            "jobs_per_host": 8,
            "solver_workers": 1,
            "connectivity_required": True,
            "seed": 20260723,
        },
        "shards": [
            {"host": "john1", "task_indices": list(range(0, 37, 2))},
            {"host": "john2", "task_indices": list(range(1, 37, 2))},
        ],
    }
    fleet_path = tmp_path / "fleet.json"
    _write(fleet_path, fleet)
    shard_paths = []
    for assignment in fleet["shards"]:
        results = []
        for index in assignment["task_indices"]:
            task = taskset["tasks"][index]
            status = "INFEASIBLE" if exact else "UNKNOWN"
            results.append(
                {
                    **task,
                    "status": status,
                    "elapsed_seconds": 1.0,
                    "branches": 10,
                    "conflicts": 1,
                    "objective": None,
                    "independent_score_breakdown": None,
                    "tokens": None,
                }
            )
        shard = {
            "schema": "all-wildlife-score-profile-shard-v1",
            "identity": identities,
            "configuration": {
                "seconds_per_profile": 30.0,
                "jobs": 8,
                "solver_workers": 1,
                "connectivity_required": True,
                "seed": 20260723,
            },
            "task_indices": assignment["task_indices"],
            "elapsed_seconds": 2.0,
            "results": results,
        }
        path = tmp_path / f"{assignment['host']}.json"
        _write(path, shard)
        shard_paths.append(path)
    return taskset_path, fleet_path, shard_paths


def test_collection_rejects_incomplete_calibration(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path)
    payload = collect(
        taskset,
        fleet,
        shards,
        known_case_index=0,
        known_max_seconds=10.388,
        hard_case_indices=[1, 2],
    )
    assert payload["totals"] == {
        "profiles": 37,
        "exact_profiles": 0,
        "unknown_profiles": 37,
        "exact_infeasible_profiles": 0,
    }
    assert payload["selection"]["verdict"] == "REJECT"


def test_collection_selects_complete_known_case(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path, exact=True)
    payload = collect(
        taskset,
        fleet,
        shards,
        known_case_index=0,
        known_max_seconds=10.388,
        hard_case_indices=[1, 2],
    )
    assert payload["selection"]["known_case_selected"]
    assert payload["selection"]["complete_hard_cases"] == [1, 2]
    assert payload["selection"]["verdict"] == "SELECT"


def test_collection_fails_closed_on_duplicate_task(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path)
    payload = json.loads(shards[1].read_text())
    duplicate = deepcopy(payload["results"][0])
    duplicate["task_index"] = payload["results"][1]["task_index"]
    payload["results"][0] = duplicate
    _write(shards[1], payload)
    with pytest.raises(ValueError, match="result ordering or coverage mismatch"):
        collect(
            taskset,
            fleet,
            shards,
            known_case_index=0,
            known_max_seconds=10.388,
            hard_case_indices=[1, 2],
        )
