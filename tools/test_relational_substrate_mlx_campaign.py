from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import relational_substrate_mlx_campaign as campaign


class _FakeTrain:
    def deterministic_training_batch(
        self,
        *,
        step: int,
        seed: int,
        arm: str,
    ) -> SimpleNamespace:
        assert step == 0
        assert seed == campaign.TRAINING_SEED
        candidates = 3
        action_hashes = np.arange(
            candidates * 32,
            dtype=np.uint8,
        ).reshape(1, candidates, 32)
        control = arm == campaign.CONTROL_ARM
        derivative = arm == campaign.S5_ARM
        return SimpleNamespace(
            base=SimpleNamespace(
                group_id=mx.array([17], dtype=mx.uint64),
                candidate_mask=mx.array([[True, True, True]]),
                action_hash=mx.array(action_hashes),
            ),
            parent=SimpleNamespace(
                transform_ids=mx.array([4]),
                r2_token_mask=mx.array(
                    [[[True]]] if control else np.zeros((1, 4, 0))
                ),
                relational_mask=mx.array(
                    np.zeros((1, 4, 0))
                    if control
                    else np.ones((1, 4, 2))
                ),
            ),
            source_candidate_indices=mx.array([[1, 2, 3]]),
            derivative_features=mx.array(
                np.ones((1, candidates, 154), dtype=np.float32)
                if derivative
                else np.zeros(
                    (1, candidates, 154),
                    dtype=np.float32,
                )
            ),
        )


def test_cross_arm_first_batch_keeps_science_equal_and_surfaces_distinct(
) -> None:
    identity = campaign._cross_arm_first_batch_identity(_FakeTrain())
    assert identity["common_candidates"] == 3
    assert len(
        {
            value["scientific_batch_blake3"]
            for value in identity["arms"].values()
        }
    ) == 1
    assert campaign._parent_surface_verified(
        campaign.CONTROL_ARM,
        identity["arms"][campaign.CONTROL_ARM],
    )
    assert campaign._derivative_surface_verified(
        campaign.S5_ARM,
        identity["arms"][campaign.S5_ARM],
    )


def test_task_spec_allocates_four_unique_primary_arms() -> None:
    specification = campaign.build_task_specification(
        relational_cache_relative=Path(
            "artifacts/experiments/relational/cache-id"
        )
    )
    tasks = specification["scientific_identity"]["tasks"]
    preflights = [
        task for task in tasks if task["kind"] == "preflight"
    ]
    training = [
        task
        for task in tasks
        if task["kind"] == "independent-experiment"
    ]
    assert {task["host"] for task in preflights} == set(campaign.HOSTS)
    assert {task["host"] for task in training} == set(campaign.HOSTS)
    assert {task["arm"] for task in training} == set(campaign.ARMS)
    expected_dependencies = {
        f"relmlx-preflight-{host}" for host in campaign.HOSTS
    }
    assert all(
        set(task["dependencies"]) == expected_dependencies
        for task in training
    )


def test_smoke_proof_validation_is_content_addressed(
    tmp_path: Path,
) -> None:
    identity = {
        "experiment_id": campaign.EXPERIMENT_ID,
        "protocol_id": campaign.PROTOCOL_ID,
        "adr": campaign.ADR_ID,
        "arm": campaign.SMOKE_ARM,
        "steps": campaign.SMOKE_STEPS,
        "hosts": ["john1", "john4"],
        "r3_cache_id": "1" * 64,
        "relational_cache_id": "2" * 64,
        "s1_cache_id": "3" * 64,
        "r6_binary_blake3": "4" * 64,
        "checks": {"numeric_parity": True},
    }
    proof = {
        "schema_version": 1,
        "experiment_id": campaign.EXPERIMENT_ID,
        "protocol_id": campaign.PROTOCOL_ID,
        "adr": campaign.ADR_ID,
        "classification": campaign.SMOKE_PASS,
        "proof_id": campaign.canonical_blake3(identity),
        "scientific_identity": identity,
        "claims": {"production_training_started": False},
    }
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    assert (
        campaign.validate_smoke_proof(
            path,
            r3_cache_id="1" * 64,
            relational_cache_id="2" * 64,
            s1_cache_id="3" * 64,
            r6_binary_blake3="4" * 64,
        )
        == proof
    )

    proof["scientific_identity"]["checks"]["numeric_parity"] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(campaign.CampaignError, match="smoke proof"):
        campaign.validate_smoke_proof(
            path,
            r3_cache_id="1" * 64,
            relational_cache_id="2" * 64,
            s1_cache_id="3" * 64,
            r6_binary_blake3="4" * 64,
        )
