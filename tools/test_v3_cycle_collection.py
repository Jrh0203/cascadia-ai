from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import v3_cycle_collection as collection
from cascadia_cluster import ContainerInput


@dataclass(frozen=True)
class _Reference:
    mounted_path: str
    target: str


class _Store:
    def stage_file(self, path: Path, *, target: str) -> _Reference:
        return _Reference(f"{target}/{path.name}", target)


def _model(directory: Path, name: str) -> Path:
    directory.mkdir()
    weights = directory / "weights.v3q"
    weights.write_bytes(name.encode())
    (directory / "model.json").write_text(
        json.dumps(
            {
                "architecture_id": "cascadia-v3-sfnnv13-r7",
                "checkpoint_id": name,
                "weights_file": weights.name,
                "weights_blake3": collection._digest(weights),
                "serving_compatible": True,
            }
        )
    )
    return directory


def test_cycle_one_uses_only_v1_opponents() -> None:
    assert collection.expected_policy_seats(1) == (30_000, 0)


def test_later_cycles_use_exact_eighty_twenty_opponent_mix() -> None:
    assert collection.expected_policy_seats(2) == (24_000, 6_000)
    assert collection.expected_policy_seats(10) == (24_000, 6_000)


def test_cycle_domain_is_bounded() -> None:
    with pytest.raises(collection.PipelineError):
        collection.expected_policy_seats(0)


def test_collection_rotates_one_prior_bundle_per_shard(tmp_path: Path) -> None:
    newest = _model(tmp_path / "newest", "newest")
    priors = [_model(tmp_path / f"prior-{index}", f"prior-{index}") for index in range(3)]
    state = tmp_path / "campaign-state.json"
    state.write_text("{}")
    v1 = tmp_path / "v1.bin"
    v1.write_bytes(b"v1")
    items = []
    for index in range(6):
        first_game = 2_000_020_000 + index * 100
        items.append(
            {
                "key": f"expert-{index}",
                "args": [
                    "worker",
                    "collect",
                    "--first-game-index",
                    str(first_game),
                    "--games",
                    "100",
                    "--v3-model-dir",
                    "/placeholder/newest",
                ],
                "environment": {},
                "application_metadata": {
                    "cycle": "2",
                    "first_game_index": str(first_game),
                    "games": "100",
                },
            }
        )
    jobs, _ = collection.build_jobs(
        plan={"phase": "cycle-02-collecting", "items": items},
        store=_Store(),
        campaign_state=state,
        v1_weights=v1,
        newest_model=newest,
        prior_models=priors,
    )

    selected = []
    for job in jobs:
        model_dirs = [
            job.args[index + 1]
            for index, value in enumerate(job.args)
            if value == "--v3-model-dir"
        ]
        assert len(model_dirs) == 2
        assert len(job.inputs) == 6  # state, V1, newest manifest/weights, one prior pair
        prior_ids = json.loads(job.application_metadata["prior_model_ids"])
        assert len(prior_ids) == 1
        selected.append(prior_ids[0])
    assert selected == [
        collection.model_identity(priors[index % 3]) for index in range(6)
    ]


def test_collection_deduplicates_prior_equal_to_newest(tmp_path: Path) -> None:
    newest = _model(tmp_path / "newest", "newest")
    state = tmp_path / "campaign-state.json"
    state.write_text("{}")
    v1 = tmp_path / "v1.bin"
    v1.write_bytes(b"v1")
    with pytest.raises(collection.PipelineError, match="at least one checkpoint distinct"):
        collection.build_jobs(
            plan={"phase": "cycle-02-collecting", "items": []},
            store=_Store(),
            campaign_state=state,
            v1_weights=v1,
            newest_model=newest,
            prior_models=[newest],
        )


def test_collection_can_reconstruct_legacy_all_prior_payload(tmp_path: Path) -> None:
    newest = _model(tmp_path / "newest", "newest")
    priors = [_model(tmp_path / f"prior-{index}", f"prior-{index}") for index in range(3)]
    state = tmp_path / "campaign-state.json"
    state.write_text("{}")
    v1 = tmp_path / "v1.bin"
    v1.write_bytes(b"v1")
    item = {
        "key": "expert-0",
        "args": ["worker", "collect", "--v3-model-dir", "/placeholder/newest"],
        "environment": {},
        "application_metadata": {
            "cycle": "7",
            "first_game_index": "2000070000",
            "games": "100",
        },
    }
    jobs, _ = collection.build_jobs(
        plan={"phase": "cycle-07-collecting", "items": [item]},
        store=_Store(),
        campaign_state=state,
        v1_weights=v1,
        newest_model=newest,
        prior_models=priors,
        prior_bundles_per_shard=None,
    )
    model_dirs = [
        jobs[0].args[index + 1]
        for index, value in enumerate(jobs[0].args)
        if value == "--v3-model-dir"
    ]
    assert len(model_dirs) == 4
    assert len(jobs[0].inputs) == 10
    assert len(json.loads(jobs[0].application_metadata["prior_model_ids"])) == 3


def test_collection_rejects_policy_outside_shard_local_domain(tmp_path: Path) -> None:
    shard = tmp_path / "item.v3g"
    shard.write_bytes(b"packed game")
    receipt = {
        "schema_id": "cascadia-v3-collection-shard-receipt-v1",
        "scientific_eligible": True,
        "component": "expert-iteration",
        "cycle": 2,
        "games": 1,
        "records": 1,
        "newest_model_seats_per_expert_game": 1,
        "bytes": shard.stat().st_size,
        "blake3": collection._digest(shard),
        "approved_readiness_sha256": "a" * 64,
        "policy_seat_games": {
            "newest": 1,
            collection.V1_POLICY_ID: 2,
            "undeclared-prior": 1,
        },
    }
    (tmp_path / "item.receipt.json").write_text(json.dumps(receipt))
    job = ContainerInput(
        key="item",
        application_metadata={
            "cycle": "2",
            "games": "1",
            "newest_model_id": "newest",
            "prior_model_ids": json.dumps(["declared-prior"]),
        },
    )

    with pytest.raises(collection.PipelineError, match="receipt is invalid"):
        collection._validate_item(tmp_path, job)
