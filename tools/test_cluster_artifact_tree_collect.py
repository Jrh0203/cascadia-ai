from __future__ import annotations

from pathlib import Path

import cluster_artifact_tree_collect as collect
import pytest


def _source_tree(root: Path) -> Path:
    source = root / "source"
    (source / "nested").mkdir(parents=True)
    (source / "dataset.json").write_text('{"records": 76}\n')
    (source / "nested" / "shard.o1i").write_bytes(b"exact-records")
    return source


def test_split_remote_requires_absolute_path() -> None:
    assert collect.split_remote("john4:/tmp/dataset/") == (
        "john4",
        "/tmp/dataset/",
    )
    with pytest.raises(collect.TreeCollectError, match="invalid remote tree"):
        collect.split_remote("john4:relative")


def test_tree_hash_uses_stable_relative_paths(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    hashes = collect.tree_sha256(source)
    assert list(hashes) == ["dataset.json", "nested/shard.o1i"]
    assert collect.tree_manifest_sha256(hashes) == collect.tree_manifest_sha256(
        dict(reversed(list(hashes.items())))
    )


def test_collects_local_coordinator_tree_atomically(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    destination = tmp_path / "collected" / "dataset"

    report = collect.collect_trees([(f"john1:{source}/", destination)])

    assert report["all_trees_match"] is True
    assert report["trees"][0]["reused"] is False
    assert collect.tree_sha256(destination) == collect.tree_sha256(source)
    assert not list(destination.parent.glob(".*.collecting-*"))


def test_exact_existing_destination_is_reused(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    destination = tmp_path / "collected" / "dataset"
    first = collect.collect_trees([(f"john1:{source}/", destination)])
    second = collect.collect_trees([(f"john1:{source}/", destination)])

    assert first["trees"][0]["reused"] is False
    assert second["trees"][0]["reused"] is True


def test_mismatched_existing_destination_is_rejected(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    destination = tmp_path / "collected" / "dataset"
    destination.mkdir(parents=True)
    (destination / "dataset.json").write_text('{"different": true}\n')

    with pytest.raises(collect.TreeCollectError, match="different content"):
        collect.collect_trees([(f"john1:{source}/", destination)])


def test_destinations_must_be_unique_and_disjoint(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    destination = tmp_path / "collected"
    with pytest.raises(collect.TreeCollectError, match="unique"):
        collect.collect_trees(
            [
                (f"john1:{source}/", destination),
                (f"john1:{source}/", destination),
            ]
        )
    with pytest.raises(collect.TreeCollectError, match="nested"):
        collect.collect_trees(
            [
                (f"john1:{source}/", destination),
                (f"john1:{source}/", destination / "nested"),
            ]
        )
