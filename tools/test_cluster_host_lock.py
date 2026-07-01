from __future__ import annotations

import fcntl
import json
from pathlib import Path

import cluster_host_lock as lock
import pytest


def test_host_lock_reports_owner_and_releases_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "host.lock"
    handle = lock.acquire_lock(path, 0.0, 0.01)
    owner = {"pid": 42, "name": "training"}
    lock.write_owner(handle, owner)

    assert lock.lock_status(path) == {"busy": True, "owner": owner}
    with pytest.raises(lock.LockTimeout):
        lock.acquire_lock(path, 0.0, 0.01)

    lock.release_lock(handle)
    assert lock.lock_status(path) == {"busy": False, "owner": None}
    assert path.read_text() == ""


def test_lock_file_is_valid_json_while_held(tmp_path: Path) -> None:
    path = tmp_path / "host.lock"
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    lock.write_owner(handle, {"host": "john1", "command": ["true"]})
    assert json.loads(path.read_text())["host"] == "john1"
    lock.release_lock(handle)
