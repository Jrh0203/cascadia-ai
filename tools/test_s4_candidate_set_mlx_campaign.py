from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import s4_candidate_set_mlx_campaign as campaign
from cascadia_mlx.s4_candidate_set_mlx_model import S4_ARMS


def test_warm_start_must_match_failed_radius_one_report() -> None:
    warm = {
        "global_step": 3000,
        "model_blake3": "a" * 64,
        "manifest_blake3": "b" * 64,
        "model_config": {"arm": "t3-r3-radius1-global"},
    }
    r3 = {
        "substrate": {
            "checkpoint": {
                "model_blake3": "a" * 64,
                "manifest_blake3": "b" * 64,
            }
        }
    }
    campaign._validate_warm_start_binding(warm, r3)
    warm["model_blake3"] = "c" * 64
    with pytest.raises(campaign.CampaignError, match="failed R3"):
        campaign._validate_warm_start_binding(warm, r3)


def test_smoke_proof_binds_context_and_warm_start(
    tmp_path: Path,
) -> None:
    warm = {"model_blake3": "a" * 64}
    identity = {
        "cache_id": "1" * 64,
        "s1_cache_id": "2" * 64,
        "context_cache_id": "3" * 64,
        "warm_start": warm,
        "checks": {"batch_identity_exact": True},
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
    campaign._write_json_atomic(path, proof)
    assert (
        campaign.validate_smoke_proof(
            path,
            cache_id="1" * 64,
            s1_cache_id="2" * 64,
            context_cache_id="3" * 64,
            warm_start=warm,
        )
        == proof
    )
    with pytest.raises(campaign.CampaignError, match="smoke proof"):
        campaign.validate_smoke_proof(
            path,
            cache_id="1" * 64,
            s1_cache_id="2" * 64,
            context_cache_id="4" * 64,
            warm_start=warm,
        )


def test_context_cache_manifest_is_part_of_authorization_contract() -> None:
    cache = SimpleNamespace(cache_id="a" * 64)
    assert cache.cache_id == "a" * 64


def test_queue_runs_four_nonduplicative_arms_after_all_preflights(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts/bundles/s4"
    cache = repository / "artifacts/cache/r3"
    s1_cache = repository / "artifacts/cache/s1"
    context = repository / "artifacts/cache/s4"
    train = repository / "artifacts/datasets/train"
    validation = repository / "artifacts/datasets/validation"
    control = repository / "artifacts/experiments/s4/control"
    experiment = repository / "artifacts/experiments/s4"
    specs = campaign.build_task_specs(
        repository=repository,
        bundle=bundle,
        cache=cache,
        s1_cache=s1_cache,
        context_cache=context,
        train_dataset=train,
        validation_dataset=validation,
        control=control,
        experiment_root=experiment,
    )
    queue = campaign.queue_specification(specs)
    assert len(specs) == queue["task_count"] == 19
    assert queue["applied"] is False
    by_id = {spec["id"]: spec for spec in specs}
    preflights = {
        f"{campaign.TASK_PREFIX}-preflight-{host}"
        for host in campaign.HOSTS
    }
    for arm, host in campaign.ARM_HOSTS.items():
        task = by_id[
            f"{campaign.TASK_PREFIX}-train-{arm.replace('-', '_')}"
        ]
        assert task["compatible_hosts"] == [host]
        assert task["workload_class"] == "independent-experiment"
        assert set(task["dependencies"]) == preflights
    proof = by_id[f"{campaign.TASK_PREFIX}-classification-order-proof"]
    assert proof["decision_terminal"] is True
    assert set(proof["dependencies"]) == {
        f"{campaign.TASK_PREFIX}-classify-forward",
        f"{campaign.TASK_PREFIX}-classify-reverse",
    }
    for spec in specs:
        if any("python" in item for item in spec["command"]):
            assert "-B" in spec["command"]
        assert "cluster_research_queue.py" not in "\0".join(
            spec["command"]
        )


def test_arm_set_is_frozen() -> None:
    assert S4_ARMS == (
        "c0-independent",
        "t1-inducing-16",
        "t2-exact-relations",
        "t3-combined",
    )
