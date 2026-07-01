from __future__ import annotations

from pathlib import Path

import cluster_research_queue as research_queue
import pytest
import s1_semantic_supply_queue as queue


def _specs() -> list[dict]:
    return queue.build_task_specs(
        bundle_relative=Path("artifacts/experiments/s1/bundles/example"),
    )


def _by_id(specs: list[dict]) -> dict[str, dict]:
    return {spec["id"]: spec for spec in specs}


def test_graph_uses_all_four_hosts_on_disjoint_open_partitions() -> None:
    specs = _specs()
    by_id = _by_id(specs)
    assert len(specs) == 13
    assert len(by_id) == len(specs)
    for split, first_game_index, games in (
        ("train", queue.TRAIN_FIRST_GAME_INDEX, queue.TRAIN_GAMES),
        (
            "validation",
            queue.VALIDATION_FIRST_GAME_INDEX,
            queue.VALIDATION_GAMES,
        ),
    ):
        owned = []
        for shard_index, host in enumerate(queue.HOSTS):
            task = by_id[f"s1ss-{split}-shard-{shard_index}"]
            assert task["compatible_hosts"] == [host]
            assert task["dependencies"] == ["s1ss-bundle-fanout"]
            command = task["command"]
            assert command[:3] == [
                "/usr/bin/env",
                "-C",
                str(queue.REMOTE_ROOTS[host] / "artifacts/experiments/s1/bundles/example/source"),
            ]
            assert command[3].endswith("/bin/exact_semantic_supply_census")
            assert command[command.index("--first-game-index") + 1] == str(first_game_index)
            assert command[command.index("--games") + 1] == str(games)
            assert command[command.index("--shard-index") + 1] == str(shard_index)
            assert command[command.index("--shard-count") + 1] == "4"
            owned.extend(
                first_game_index + offset for offset in range(games) if offset % 4 == shard_index
            )
        assert sorted(owned) == list(range(first_game_index, first_game_index + games))
        assert len(owned) == len(set(owned))


def test_collection_and_merge_require_every_shard_and_are_order_independent() -> None:
    by_id = _by_id(_specs())
    collection = by_id["s1ss-report-collection"]
    assert len(collection["dependencies"]) == 8
    assert collection["command"].count("--artifact") == 6

    forward = by_id["s1ss-merge-forward"]
    reverse = by_id["s1ss-merge-reverse"]
    assert forward["dependencies"] == ["s1ss-report-collection"]
    assert reverse["dependencies"] == ["s1ss-report-collection"]
    assert forward["command"][1].endswith("/source/tools/s1_semantic_supply_merge.py")
    forward_reports = [
        forward["command"][index + 1]
        for index, value in enumerate(forward["command"])
        if value == "--shard"
    ]
    reverse_reports = [
        reverse["command"][index + 1]
        for index, value in enumerate(reverse["command"])
        if value == "--shard"
    ]
    assert len(forward_reports) == 8
    assert reverse_reports == list(reversed(forward_reports))
    proof = by_id["s1ss-merge-order-proof"]
    assert proof["dependencies"] == ["s1ss-merge-forward", "s1ss-merge-reverse"]
    assert proof["decision_terminal"] is True


def test_bundle_validation_requires_all_provenance_roots_and_frozen_merger() -> None:
    complete_paths = [
        *queue.REQUIRED_SOURCE_FILES,
        *(f"{prefix}placeholder" for prefix in queue.REQUIRED_SOURCE_PREFIXES),
    ]
    manifest = {
        "identity": {
            "source_files": [{"path": path} for path in complete_paths],
        }
    }
    queue.validate_provenance_source_bundle(manifest)
    manifest["identity"]["source_files"] = [
        entry
        for entry in manifest["identity"]["source_files"]
        if entry["path"] != "tools/s1_semantic_supply_merge.py"
    ]
    with pytest.raises(queue.CampaignError, match="s1_semantic_supply_merge"):
        queue.validate_provenance_source_bundle(manifest)


@pytest.mark.parametrize(
    ("train_games", "validation_games"),
    [(3, 100), (400, 3)],
)
def test_graph_rejects_splits_that_cannot_keep_every_host_productive(
    train_games: int,
    validation_games: int,
) -> None:
    with pytest.raises(queue.CampaignError, match="at least one game"):
        queue.build_task_specs(
            bundle_relative=Path("artifacts/bundle"),
            train_games=train_games,
            validation_games=validation_games,
        )


def test_generator_is_reviewed_only_unless_apply_is_explicit() -> None:
    assert "--apply" not in _by_id(_specs())["s1ss-bundle-fanout"]["command"]


def test_generated_graph_is_valid_under_the_production_queue_schema() -> None:
    state = research_queue.empty_queue("s1-semantic-supply-test", now_ms=1_000)
    for offset, specification in enumerate(_specs()):
        research_queue.add_task(state, specification, now_ms=1_001 + offset)
    research_queue.validate_queue(state)
    assert len(state["tasks"]) == 13
