from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import r4_bounded_parent_mlx_campaign as campaign
from cascadia_mlx.r4_bounded_parent_mlx_cache import ARMS


def test_frozen_initialization_is_exact_across_all_four_arms() -> None:
    initialization = campaign.cross_arm_initialization()
    assert set(initialization["cross_arm_parameter_counts"]) == set(ARMS)
    assert len(set(initialization["cross_arm_parameter_counts"].values())) == 1
    assert len(set(initialization["cross_arm_parameter_layout_blake3"].values())) == 1
    assert len(set(initialization["cross_arm_initial_parameter_tensor_blake3"].values())) == 1


def test_cache_must_be_bound_to_the_immutable_exporter() -> None:
    cache = SimpleNamespace(manifest={"exporter": {"executable_blake3": "a" * 64}})
    campaign._validate_cache_exporter_binding(cache, "a" * 64)
    with pytest.raises(campaign.CampaignError, match="immutable exporter"):
        campaign._validate_cache_exporter_binding(cache, "b" * 64)


def test_cross_host_smoke_proof_is_content_addressed_and_all_checks_must_pass(
    tmp_path: Path,
) -> None:
    identity = {
        "experiment_id": campaign.EXPERIMENT_ID,
        "protocol_id": campaign.PROTOCOL_ID,
        "adr": campaign.ADR_ID,
        "arm": "q3-affordance-parent",
        "r3_cache_id": "0" * 64,
        "parent_cache_id": "2" * 64,
        "s1_cache_id": "1" * 64,
        "checks": {"batch_identity_exact": True, "numerical_parity": True},
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
            r3_cache_id="0" * 64,
            s1_cache_id="1" * 64,
        )
        == proof
    )

    proof["scientific_identity"]["checks"]["numerical_parity"] = False
    proof["proof_id"] = campaign.canonical_blake3(proof["scientific_identity"])
    path.write_text(json.dumps(proof))
    with pytest.raises(campaign.CampaignError, match="smoke proof"):
        campaign.validate_smoke_proof(
            path,
            r3_cache_id="0" * 64,
            s1_cache_id="1" * 64,
        )


def test_inert_queue_runs_four_nonduplicative_arms_and_exact_classification(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts/bundles/r4"
    r3_cache = repository / "artifacts/cache/r3"
    parent_cache = repository / "artifacts/cache/parent"
    s1_cache = repository / "artifacts/cache/s1"
    train = repository / "artifacts/datasets/train"
    validation = repository / "artifacts/datasets/validation"
    control = repository / "artifacts/experiments/r4/control"
    authorization = control / "authorization.json"
    smoke = control / "cross-host-smoke-proof.json"
    experiment = repository / "artifacts/experiments/r4"
    specs = campaign.build_task_specs(
        repository=repository,
        bundle=bundle,
        r3_cache=r3_cache,
        parent_cache=parent_cache,
        s1_cache=s1_cache,
        train_dataset=train,
        validation_dataset=validation,
        authorization=authorization,
        smoke_proof=smoke,
        experiment_root=experiment,
    )
    queue = campaign.queue_specification(specs)
    assert len(specs) == queue["task_count"] == 21
    assert queue["applied"] is False
    by_id = {spec["id"]: spec for spec in specs}
    preflights = {f"{campaign.TASK_PREFIX}-preflight-{host}" for host in campaign.HOSTS}
    for arm, host in campaign.ARM_HOSTS.items():
        task = by_id[f"{campaign.TASK_PREFIX}-train-{arm.replace('-', '_')}"]
        assert task["compatible_hosts"] == [host]
        assert task["workload_class"] == "independent-experiment"
        assert set(task["dependencies"]) == preflights
    control_fanout = by_id[f"{campaign.TASK_PREFIX}-fanout-control-run"]
    assert set(control_fanout["dependencies"]) == {
        f"{campaign.TASK_PREFIX}-train-{arm.replace('-', '_')}" for arm in ARMS
    }
    for arm in ARMS[1:]:
        replay = by_id[f"{campaign.TASK_PREFIX}-paired-c0-{arm.replace('-', '_')}"]
        assert replay["compatible_hosts"] == [campaign.ARM_HOSTS[arm]]
        assert replay["dependencies"] == [control_fanout["id"]]
    classifier = by_id[f"{campaign.TASK_PREFIX}-classify"]
    assert classifier["decision_terminal"] is True
    assert classifier["dependencies"] == [f"{campaign.TASK_PREFIX}-collect"]
    for spec in specs:
        if any("python" in item for item in spec["command"]):
            assert "-B" in spec["command"]
        assert "cluster_research_queue.py" not in "\0".join(spec["command"])
