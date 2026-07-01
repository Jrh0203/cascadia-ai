from __future__ import annotations

from pathlib import Path

import cluster_artifact_fanout as fanout
import pytest


def test_split_remote_requires_absolute_remote_path() -> None:
    assert fanout.split_remote("john3:/tmp/run/") == ("john3", "/tmp/run/")
    with pytest.raises(fanout.FanoutError, match="invalid remote path"):
        fanout.split_remote("john3:relative")


def test_sha256_streams_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"cascadia")
    assert fanout.sha256(path) == "f1e7e250a6b373d52a2b55eee8dd95192632a2c8c5949c06cc75717e5e6b7b99"


def test_tree_sha256_covers_stable_relative_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "cache.json").write_text("{}\n")
    (tmp_path / "nested" / "batch.bin").write_bytes(b"batch")
    assert fanout.tree_sha256(tmp_path) == {
        "cache.json": ("ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"),
        "nested/batch.bin": ("4bb24efc9641afc5ded1ca77eabb6e2fcf062d2112ccd61bd8bd6acd89180bae"),
    }


def test_fanout_requires_file_or_tree_verification(tmp_path: Path) -> None:
    with pytest.raises(fanout.FanoutError, match="verification"):
        fanout.fanout(
            source="john2:/tmp/run/",
            local_root=tmp_path,
            destinations=["john3:/tmp/run/"],
            required_files=[],
        )


def test_fanout_rejects_escaping_required_path(tmp_path: Path) -> None:
    with pytest.raises(fanout.FanoutError, match="beneath"):
        fanout.fanout(
            source="john2:/tmp/run/",
            local_root=tmp_path,
            destinations=["john3:/tmp/run/"],
            required_files=["../secret"],
        )
