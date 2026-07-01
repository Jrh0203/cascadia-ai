from __future__ import annotations

import json
from pathlib import Path

import blake3
import pytest
import v3_cycle_collection as collection
import v3_cycle_collection_repair as repair
from cascadia_cluster import ContainerInput

IMAGE = "registry/v3@sha256:" + "a" * 64
ORIGINAL_REQUEST = "cycle-07-original"
REPAIR_REQUEST = "cycle-07-original-memory-repair-v1"
NEWEST = "v3-newest"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _jobs() -> list[ContainerInput]:
    first = 2_000_000_000 + 7 * 10_000
    return [
        ContainerInput(
            key=f"expert-iteration-07-{index:06d}",
            application_metadata={
                "cycle": "7",
                "games": "100",
                "first_game_index": str(first + index * 100),
                "newest_model_id": NEWEST,
                "prior_model_ids": json.dumps(["prior"]),
            },
        )
        for index in range(100)
    ]


def _artifact(directory: Path, job: ContainerInput) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shard = directory / f"{job.key}.v3g"
    shard.write_bytes(f"games:{job.key}".encode())
    first = int(job.application_metadata["first_game_index"])
    v1, prior = collection.expected_policy_seats_for_ranges(7, [(first, 100)])
    _write_json(
        directory / f"{job.key}.receipt.json",
        {
            "schema_id": "cascadia-v3-collection-shard-receipt-v1",
            "scientific_eligible": True,
            "component": "expert-iteration",
            "cycle": 7,
            "games": 100,
            "records": 100,
            "newest_model_seats_per_expert_game": 1,
            "bytes": shard.stat().st_size,
            "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            "approved_readiness_sha256": "b" * 64,
            "policy_seat_games": {NEWEST: 100, collection.V1_POLICY_ID: v1, "prior": prior},
        },
    )
    _write_json(directory / "manifest.json", {"item": job.key})


def _fixture(tmp_path: Path) -> tuple[list[ContainerInput], dict, dict, Path, Path]:
    jobs = _jobs()
    original_root = tmp_path / "original"
    repair_root = tmp_path / "repair"
    rejected = {jobs[30].key, jobs[38].key}
    for job in jobs:
        if job.key in rejected:
            _artifact(repair_root / REPAIR_REQUEST / job.key, job)
        else:
            _artifact(original_root / ORIGINAL_REQUEST / job.key, job)
    request = {
        "schema_id": "cascadia.cluster.managed-request-state.v2",
        "request_id": ORIGINAL_REQUEST,
        "image_digest": IMAGE,
        "items": [
            {
                "key": job.key,
                "job_payload": {
                    "Meta": {
                        f"cascadia.app.{key}": str(value)
                        for key, value in job.application_metadata.items()
                    }
                },
            }
            for job in jobs
        ],
    }
    completion = {
        "schema_id": "cascadia-v3-cluster-stage-completion-v1",
        "passed": True,
        "repair_mode": True,
        "request_id": REPAIR_REQUEST,
        "work_items": 2,
        "succeeded": 2,
        "requested_memory_gib": 2.0,
        "inputs": [{"item": key} for key in sorted(rejected)],
    }
    return jobs, request, completion, original_root, repair_root


def test_reconcile_replaces_only_rejected_collection_shards(tmp_path: Path) -> None:
    jobs, request, completion, original_root, repair_root = _fixture(tmp_path)
    repair._validate_original_request(request, jobs, IMAGE)
    reconciled = tmp_path / "reconciled"
    result = repair.reconcile(
        cycle=7,
        image=IMAGE,
        original_request=request,
        original_root=original_root,
        repair_completion=completion,
        repair_root=repair_root,
        reconciled_root=reconciled,
        jobs=jobs,
    )
    assert result["passed"] is True
    assert result["totals"]["games"] == 10_000
    assert result["totals"]["v1_seat_games"] == 24_000
    assert result["totals"]["prior_v3_seat_games"] == 6_000
    assert result["repair"]["repaired_item_count"] == 2
    assert len([path for path in reconciled.iterdir() if path.is_dir()]) == 100
    assert json.loads((reconciled / jobs[30].key / "lineage.json").read_text())[
        "source_request_id"
    ] == REPAIR_REQUEST
    original_shard = next((original_root / ORIGINAL_REQUEST / jobs[0].key).glob("*.v3g"))
    linked_shard = next((reconciled / jobs[0].key).glob("*.v3g"))
    assert original_shard.stat().st_ino == linked_shard.stat().st_ino


def test_reconcile_rejects_incomplete_repair_domain(tmp_path: Path) -> None:
    jobs, request, completion, original_root, repair_root = _fixture(tmp_path)
    completion["inputs"].pop()
    completion["work_items"] = completion["succeeded"] = 1
    with pytest.raises(repair.CollectionRepairError, match="exactly rejected shards"):
        repair.reconcile(
            cycle=7,
            image=IMAGE,
            original_request=request,
            original_root=original_root,
            repair_completion=completion,
            repair_root=repair_root,
            reconciled_root=tmp_path / "reconciled",
            jobs=jobs,
        )


@pytest.mark.parametrize("matched_layout", [1, None])
def test_original_layout_detection_preserves_request_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, matched_layout: int | None
) -> None:
    def fake_build_jobs(*, prior_bundles_per_shard: int | None, **_: object):
        prior_ids = ["prior-local"] if prior_bundles_per_shard == 1 else ["prior-a", "prior-b"]
        return [
            ContainerInput(
                key="expert-iteration-07-000000",
                application_metadata={
                    "cycle": "7",
                    "games": "100",
                    "first_game_index": "2000070000",
                    "newest_model_id": NEWEST,
                    "prior_model_ids": json.dumps(prior_ids),
                },
            )
        ], NEWEST

    monkeypatch.setattr(collection, "build_jobs", fake_build_jobs)
    expected_jobs, _ = fake_build_jobs(prior_bundles_per_shard=matched_layout)
    request = {
        "schema_id": "cascadia.cluster.managed-request-state.v2",
        "image_digest": IMAGE,
        "items": [
            {
                "key": job.key,
                "job_payload": {
                    "Meta": {
                        f"cascadia.app.{key}": str(value)
                        for key, value in job.application_metadata.items()
                    }
                },
            }
            for job in expected_jobs
        ],
    }

    jobs, observed_layout = repair._match_original_layout(
        request=request,
        plan={},
        store=object(),
        campaign_state=tmp_path / "state.json",
        v1_weights=tmp_path / "v1.bin",
        newest_model=tmp_path / "newest",
        prior_models=[],
        image=IMAGE,
    )

    assert observed_layout == matched_layout
    assert jobs == expected_jobs
