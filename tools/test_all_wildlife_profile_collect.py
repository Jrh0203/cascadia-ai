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
    fleet = {"schema": "all-wildlife-score-profile-fleet-v1", "tag": "test"}
    fleet_path = tmp_path / "fleet.json"
    _write(fleet_path, fleet)

    assignments = [list(range(0, 37, 2)), list(range(1, 37, 2))]
    shard_paths = []
    for shard_index, indices in enumerate(assignments):
        results = []
        for index in indices:
            task = taskset["tasks"][index]
            results.append(
                {
                    **task,
                    "status": "INFEASIBLE" if exact else "UNKNOWN",
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
            "task_indices": indices,
            "elapsed_seconds": 2.0,
            "results": results,
        }
        path = tmp_path / f"shard-{shard_index}.json"
        _write(path, shard)
        shard_paths.append(path)
    return taskset_path, fleet_path, shard_paths


def test_collection_reports_incomplete_profiles_without_rejecting_them(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path)
    payload = collect(taskset, fleet, shards)
    assert payload["coverage"] == {"received": 37, "expected": 37}
    assert payload["totals"]["unknown_profiles"] == 37


def test_collection_reports_complete_cases(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path, exact=True)
    payload = collect(
        taskset,
        fleet,
        shards,
        known_case_index=0,
        known_max_seconds=10.388,
        hard_case_indices=[1, 2],
    )
    assert payload["assessment"]["known_case_complete"]
    assert payload["assessment"]["complete_hard_cases"] == [1, 2]


def test_collection_accepts_partial_shards(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path)
    payload = collect(taskset, fleet, shards[:1])
    assert payload["coverage"] == {"received": 19, "expected": 37}


def test_collection_rejects_semantically_wrong_task(tmp_path: Path) -> None:
    taskset, fleet, shards = _fixture(tmp_path)
    payload = json.loads(shards[0].read_text())
    bad = deepcopy(payload["results"][0])
    bad["counts"] = [0, 0, 0, 0, 20]
    payload["results"][0] = bad
    _write(shards[0], payload)
    with pytest.raises(ValueError, match="counts mismatch"):
        collect(taskset, fleet, shards)
