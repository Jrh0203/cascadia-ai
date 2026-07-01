from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("r3_action_edit_campaign.py")
_SPEC = importlib.util.spec_from_file_location("r3_action_edit_campaign", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
campaign = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = campaign
_SPEC.loader.exec_module(campaign)


def _specs(tmp_path: Path) -> list[dict[str, object]]:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts/experiments/r3/bundles/frozen"
    experiment = repository / "artifacts/experiments/r3"
    return campaign.build_task_specs(
        repository=repository,
        bundle=bundle,
        source_bundle_blake3="a" * 64,
        executable_blake3="b" * 64,
        experiment_root=experiment,
    )


def test_queue_uses_four_nonduplicative_shards_and_fail_closed_preflights(
    tmp_path: Path,
) -> None:
    specs = _specs(tmp_path)
    assert len(specs) == 13
    by_id = {spec["id"]: spec for spec in specs}
    preflight_ids = {
        f"{campaign.TASK_PREFIX}-preflight-{host}" for host in campaign.HOSTS
    }
    for host in campaign.HOSTS:
        preflight = by_id[f"{campaign.TASK_PREFIX}-preflight-{host}"]
        assert preflight["compatible_hosts"] == [host]
        assert "--expected-source-bundle-blake3" in preflight["command"]
        assert "--expected-executable-blake3" in preflight["command"]
    for shard_index, host in enumerate(campaign.HOSTS):
        shard = by_id[f"{campaign.TASK_PREFIX}-shard-{shard_index}"]
        assert shard["compatible_hosts"] == [host]
        assert set(shard["dependencies"]) == preflight_ids
        command = shard["command"]
        assert command[command.index("--shard-index") + 1] == str(shard_index)
        assert command[command.index("--shard-count") + 1] == "4"
        assert command[command.index("--train-games") + 1] == "16"
        assert command[command.index("--validation-games") + 1] == "4"


def test_aggregate_orders_are_exact_reversals_and_proof_is_terminal(
    tmp_path: Path,
) -> None:
    specs = _specs(tmp_path)
    by_id = {spec["id"]: spec for spec in specs}
    forward = by_id[f"{campaign.TASK_PREFIX}-aggregate-forward"]["command"]
    reverse = by_id[f"{campaign.TASK_PREFIX}-aggregate-reverse"]["command"]

    def inputs(command: list[str]) -> list[str]:
        return [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--input"
        ]

    assert inputs(reverse) == list(reversed(inputs(forward)))
    proof = by_id[f"{campaign.TASK_PREFIX}-aggregate-order-proof"]
    assert proof["decision_terminal"] is True
    assert set(proof["dependencies"]) == {
        f"{campaign.TASK_PREFIX}-aggregate-forward",
        f"{campaign.TASK_PREFIX}-aggregate-reverse",
    }
    for spec in specs:
        if any("python" in item for item in spec["command"]):
            assert "-B" in spec["command"]


def test_queue_preview_is_deterministic_and_valid(tmp_path: Path) -> None:
    specs = _specs(tmp_path)
    first = campaign.queue_specification(
        specs,
        bundle_id="c" * 64,
        source_bundle_blake3="a" * 64,
        executable_blake3="b" * 64,
    )
    second = campaign.queue_specification(
        specs,
        bundle_id="c" * 64,
        source_bundle_blake3="a" * 64,
        executable_blake3="b" * 64,
    )
    assert first == second
    assert first["task_count"] == 13
    assert first["applied"] is False
    assert len(first["validated_queue_preview"]["tasks"]) == 13
    assert all(
        task["status"] in {"ready", "blocked"}
        for task in first["validated_queue_preview"]["tasks"]
    )


def test_queue_rejects_malformed_runtime_hashes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    with pytest.raises(campaign.CampaignError, match="lowercase"):
        campaign.build_task_specs(
            repository=repository,
            bundle=repository / "bundle",
            source_bundle_blake3="not-a-hash",
            executable_blake3="b" * 64,
        )
