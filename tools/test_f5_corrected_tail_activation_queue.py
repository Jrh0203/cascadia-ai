from __future__ import annotations

from pathlib import Path

import cluster_research_queue as research_queue
import f5_corrected_tail_activation_queue as queue
import pytest


def _specs() -> list[dict]:
    return queue.build_task_specs(
        bundle_relative=Path("artifacts/experiments/f5/bundles/example"),
    )


def _by_id(specs: list[dict]) -> dict[str, dict]:
    return {spec["id"]: spec for spec in specs}


def test_graph_uses_all_four_hosts_on_disjoint_partitions() -> None:
    specs = _specs()
    by_id = _by_id(specs)
    assert len(specs) == 13
    assert len(by_id) == len(specs)
    owned = []
    for shard_index, host in enumerate(queue.HOSTS):
        generation = by_id[f"f5a-generate-shard-{shard_index}"]
        assert generation["compatible_hosts"] == [host]
        assert generation["dependencies"] == ["f5a-bundle-fanout"]
        command = generation["command"]
        assert command[:3] == [
            "/usr/bin/env",
            "-C",
            str(
                queue.REMOTE_ROOTS[host]
                / "artifacts/experiments/f5/bundles/example/source"
            ),
        ]
        assert command[3].endswith(f"/bin/{queue.BINARY_NAME}")
        assert command[4] == "generate-shard"
        assert command[command.index("--shard-index") + 1] == str(shard_index)
        assert command[command.index("--shard-count") + 1] == "4"
        assert command[command.index("--total-games") + 1] == str(queue.TOTAL_GAMES)
        owned.extend(
            index
            for index in range(queue.FIRST_GAME_INDEX, queue.TOTAL_GAMES)
            if index % queue.SHARD_COUNT == shard_index
        )
        census = by_id[f"f5a-census-shard-{shard_index}"]
        assert census["compatible_hosts"] == [host]
        assert census["dependencies"] == [f"f5a-generate-shard-{shard_index}"]
        assert census["command"][4] == "census-shard"
    assert sorted(owned) == list(range(queue.FIRST_GAME_INDEX, queue.TOTAL_GAMES))
    assert len(owned) == len(set(owned))


def test_collection_preserves_remote_corpora_and_reports() -> None:
    collection = _by_id(_specs())["f5a-collect-remote-artifacts"]
    assert collection["dependencies"] == [
        "f5a-census-shard-0",
        "f5a-census-shard-1",
        "f5a-census-shard-2",
        "f5a-census-shard-3",
    ]
    assert collection["command"].count("--artifact") == 9
    destinations = [
        collection["command"][index + 2]
        for index, value in enumerate(collection["command"])
        if value == "--artifact"
    ]
    assert any(value.endswith("/corpus/shard-1/records.jsonl") for value in destinations)
    assert any(value.endswith("/reports/shard-3.json") for value in destinations)


def test_aggregation_is_complete_and_order_independent() -> None:
    by_id = _by_id(_specs())
    forward = by_id["f5a-aggregate-forward"]
    reverse = by_id["f5a-aggregate-reverse"]
    assert forward["dependencies"] == ["f5a-collect-remote-artifacts"]
    assert reverse["dependencies"] == ["f5a-collect-remote-artifacts"]
    forward_reports = [
        forward["command"][index + 1]
        for index, value in enumerate(forward["command"])
        if value == "--report"
    ]
    reverse_reports = [
        reverse["command"][index + 1]
        for index, value in enumerate(reverse["command"])
        if value == "--report"
    ]
    assert len(forward_reports) == queue.SHARD_COUNT
    assert reverse_reports == list(reversed(forward_reports))
    proof = by_id["f5a-aggregate-order-proof"]
    assert proof["dependencies"] == [
        "f5a-aggregate-forward",
        "f5a-aggregate-reverse",
    ]
    assert proof["decision_terminal"] is True
    assert proof["command"][4] == "verify-order"


def test_bundle_validation_requires_every_scientific_source_root() -> None:
    complete_paths = [
        *queue.REQUIRED_SOURCE_FILES,
        *(f"{prefix}placeholder" for prefix in queue.REQUIRED_SOURCE_PREFIXES),
    ]
    manifest = {
        "identity": {
            "experiment_id": queue.EXPERIMENT_ID,
            "source_files": [{"path": path} for path in complete_paths],
        }
    }
    queue.validate_provenance_source_bundle(manifest)
    manifest["identity"]["source_files"] = [
        entry
        for entry in manifest["identity"]["source_files"]
        if entry["path"] != "legacy/crates/cascadia-ai/src/placeholder"
    ]
    with pytest.raises(queue.CampaignError, match="cascadia-ai/src"):
        queue.validate_provenance_source_bundle(manifest)


@pytest.mark.parametrize("total_games", [0, 3, 1_023, 1_025])
def test_graph_rejects_work_that_does_not_partition_across_four_hosts(
    total_games: int,
) -> None:
    with pytest.raises(queue.CampaignError, match="divide evenly"):
        queue.build_task_specs(
            bundle_relative=Path("artifacts/bundle"),
            total_games=total_games,
        )


def test_generated_graph_is_valid_under_the_production_queue_schema() -> None:
    state = research_queue.empty_queue("f5-activation-test", now_ms=1_000)
    for offset, specification in enumerate(_specs()):
        research_queue.add_task(state, specification, now_ms=1_001 + offset)
    research_queue.validate_queue(state)
    assert len(state["tasks"]) == 13
