from __future__ import annotations

from pathlib import Path

import cluster_artifact_collect as collect
import pytest


def test_split_remote_requires_absolute_path() -> None:
    assert collect.split_remote("john3:/tmp/report.json") == (
        "john3",
        "/tmp/report.json",
    )
    with pytest.raises(collect.CollectError, match="invalid remote path"):
        collect.split_remote("john3:relative")


def test_sha256_streams_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}\n")
    assert (
        collect.sha256(path)
        == "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
    )


def test_collect_requires_unique_destinations(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    with pytest.raises(collect.CollectError, match="unique"):
        collect.collect(
            [
                ("john3:/tmp/a.json", destination),
                ("john4:/tmp/b.json", destination),
            ]
        )


def test_collect_copies_coordinator_local_artifact_without_ssh(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "collected" / "artifact.json"
    source.write_text('{"result":"pass"}\n')

    report = collect.collect([(f"john1:{source}", destination)])

    assert destination.read_bytes() == source.read_bytes()
    assert report["all_artifacts_match"] is True
    assert report["artifacts"][0]["sha256"] == collect.sha256(source)
