from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
import s1_exact_supply_mlx_campaign as campaign
from cascadia_mlx.s1_exact_supply_mlx_cache import ARMS
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
    S1ExactSupplyRanker,
)


def test_capacity_and_seeded_initial_weights_are_identical_across_arms() -> None:
    assert set(campaign._parameter_counts().values()) == {FROZEN_PARAMETER_COUNT}
    assert len(set(campaign._parameter_layouts().values())) == 1
    fingerprints = []
    for arm in ARMS:
        mx.random.seed(campaign.TRAINING_SEED)
        fingerprints.append(
            campaign._parameter_fingerprint(
                S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
            )
        )
    assert len(set(fingerprints)) == 1


def test_cache_must_be_produced_by_the_exact_immutable_exporter() -> None:
    cache = SimpleNamespace(
        manifest={"exporter": {"executable_blake3": "a" * 64}}
    )
    campaign._validate_cache_exporter_binding(cache, "a" * 64)
    with pytest.raises(campaign.CampaignError, match="immutable bundle exporter"):
        campaign._validate_cache_exporter_binding(cache, "b" * 64)


def test_inert_queue_uses_all_hosts_once_and_every_python_command_uses_b(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts/bundles/s1"
    cache = repository / "artifacts/cache/s1"
    train = repository / "artifacts/datasets/train"
    validation = repository / "artifacts/datasets/validation"
    authorization = repository / "artifacts/experiments/s1/control/authorization.json"
    experiment = repository / "artifacts/experiments/s1"
    specs = campaign.build_task_specs(
        repository=repository,
        bundle=bundle,
        cache=cache,
        train_dataset=train,
        validation_dataset=validation,
        authorization=authorization,
        experiment_root=experiment,
    )
    queue = campaign.queue_specification(specs)
    assert len(specs) == queue["task_count"] == 17
    assert queue["applied"] is False
    assert queue["installation_supported_by_this_tool"] is False
    by_id = {spec["id"]: spec for spec in specs}
    preflight_ids = {
        f"s1esmlx-preflight-{host}" for host in campaign.HOSTS
    }
    for arm, host in campaign.ARM_HOSTS.items():
        task = by_id[f"s1esmlx-train-{arm.replace('-', '_')}"]
        assert task["compatible_hosts"] == [host]
        assert task["workload_class"] == "independent-experiment"
        assert set(task["dependencies"]) == preflight_ids
    replay = by_id["s1esmlx-replay-control"]
    assert replay["compatible_hosts"] == ["john4"]
    assert set(replay["dependencies"]) == preflight_ids
    for spec in specs:
        if any("python" in item for item in spec["command"]):
            assert "-B" in spec["command"]
        assert "cluster_research_queue.py" not in "\0".join(spec["command"])
