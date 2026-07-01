from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _module() -> Any:
    path = Path(__file__).resolve().parents[2] / "tools/v3_colima_reclaim.py"
    spec = importlib.util.spec_from_file_location("v3_colima_reclaim_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completion(path: Path, *, succeeded: int = 80) -> Path:
    path.write_text(json.dumps({"passed": True, "work_items": 80, "succeeded": succeeded}) + "\n")
    return path


class FakeRunner:
    def __init__(
        self,
        module: Any,
        containers: list[str] | None = None,
        *,
        fail_first_trim: bool = False,
    ) -> None:
        self.module = module
        self.containers = containers or sorted(module.CONTROL_SERVICES)
        self.commands: list[list[str]] = []
        self.fail_first_trim = fail_first_trim

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        stdout = ""
        if command[:3] == [str(self.module.DOCKER), "ps", "--format"]:
            stdout = "\n".join(self.containers) + "\n"
        elif "fstrim" in command:
            if self.fail_first_trim:
                self.fail_first_trim = False
                raise subprocess.CalledProcessError(1, command)
            stdout = "/mnt/lima-colima-cascadia-r2: 12 GiB trimmed\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_reclaim_restarts_only_after_complete_increment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    colima = tmp_path / "colima"
    colima.touch()
    monkeypatch.setattr(module, "COLIMA", colima)
    docker = tmp_path / "docker"
    docker.touch()
    monkeypatch.setattr(module, "DOCKER", docker)
    runner = FakeRunner(module)
    gib = 1024**3
    usages = iter(
        (
            shutil._ntuple_diskusage(200 * gib, 160 * gib, 40 * gib),
            shutil._ntuple_diskusage(200 * gib, 80 * gib, 120 * gib),
            shutil._ntuple_diskusage(200 * gib, 70 * gib, 130 * gib),
        )
    )
    receipt = tmp_path / "receipt.json"
    value = module.reclaim_completed_increment(
        _completion(tmp_path / "completion.json"),
        receipt,
        runner=runner,
        disk_usage=lambda _: next(usages),
        sleeper=lambda _: None,
    )
    assert value["passed"] is True
    assert value["reclaimed_bytes"] == 90 * gib
    assert any(command[-2:] == ["-p", module.PROFILE] for command in runner.commands)
    assert [command for command in runner.commands if "fstrim" in command]
    assert json.loads(receipt.read_text()) == value


def test_reclaim_is_idempotent(tmp_path: Path) -> None:
    module = _module()
    receipt = tmp_path / "receipt.json"
    expected = {"passed": True, "schema_id": "already-complete"}
    receipt.write_text(json.dumps(expected) + "\n")
    runner = FakeRunner(module)
    assert (
        module.reclaim_completed_increment(
            tmp_path / "missing-completion.json", receipt, runner=runner
        )
        == expected
    )
    assert runner.commands == []


def test_reclaim_environment_is_launchagent_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    environment = module._environment()
    assert environment["PATH"].split(":")[:2] == [
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    assert environment["PATH"].split(":").count("/usr/bin") == 1
    assert environment["DOCKER_HOST"] == f"unix://{module.DOCKER_SOCKET}"
    assert environment["COLIMA_HOME"] == str(module.COLIMA_HOME)


def test_reclaim_rejects_nonterminal_increment(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.ColimaReclaimError, match="not fully reconciled"):
        module.reclaim_completed_increment(
            _completion(tmp_path / "completion.json", succeeded=79),
            tmp_path / "receipt.json",
            runner=FakeRunner(module),
        )


def test_reclaim_rejects_unrelated_live_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    colima = tmp_path / "colima"
    colima.touch()
    monkeypatch.setattr(module, "COLIMA", colima)
    docker = tmp_path / "docker"
    docker.touch()
    monkeypatch.setattr(module, "DOCKER", docker)
    with pytest.raises(module.ColimaReclaimError, match="non-control containers"):
        module.reclaim_completed_increment(
            _completion(tmp_path / "completion.json"),
            tmp_path / "receipt.json",
            runner=FakeRunner(module, [*module.CONTROL_SERVICES, "unexpected-job"]),
        )


def test_reclaim_retries_transient_guest_trim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    colima = tmp_path / "colima"
    colima.touch()
    docker = tmp_path / "docker"
    docker.touch()
    monkeypatch.setattr(module, "COLIMA", colima)
    monkeypatch.setattr(module, "DOCKER", docker)
    runner = FakeRunner(module, fail_first_trim=True)
    gib = 1024**3
    usages = iter(
        (
            shutil._ntuple_diskusage(200 * gib, 140 * gib, 60 * gib),
            shutil._ntuple_diskusage(200 * gib, 70 * gib, 130 * gib),
            shutil._ntuple_diskusage(200 * gib, 70 * gib, 130 * gib),
        )
    )
    module.reclaim_completed_increment(
        _completion(tmp_path / "completion.json"),
        tmp_path / "receipt.json",
        runner=runner,
        disk_usage=lambda _: next(usages),
        sleeper=lambda _: None,
    )
    assert sum("fstrim" in command for command in runner.commands) == 2


class FakeRemoteRunner:
    def __init__(self, free_kib: list[int], *, live_host: str | None = None) -> None:
        self.free_kib = iter(free_kib)
        self.live_host = live_host
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        host = command[-2]
        script = command[-1]
        stdout = ""
        if "docker ps" in script:
            stdout = "live-job\n" if host == self.live_host else ""
        elif "free_kib=" in script:
            stdout = f"free_kib={next(self.free_kib)}\n"
        elif "fstrim" in script:
            stdout = "/mnt/lima-colima: 12 GiB trimmed\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_remote_reclaim_is_serial_evidence_gated_and_fabric_verified(tmp_path: Path) -> None:
    module = _module()
    gib_kib = 1024**2
    runner = FakeRemoteRunner(
        [
            40 * gib_kib,
            120 * gib_kib,
            130 * gib_kib,
            60 * gib_kib,
            140 * gib_kib,
            150 * gib_kib,
        ]
    )
    probes: list[bool] = []
    receipt = tmp_path / "remote.json"
    value = module.reclaim_remote_workers(
        _completion(tmp_path / "completion.json"),
        receipt,
        runner=runner,
        sleeper=lambda _: None,
        fabric_probe=lambda: probes.append(True),
    )
    assert value["passed"] is True
    assert value["reclaimed_bytes"] == 180 * 1024**3
    assert [item["host"] for item in value["workers"]] == ["john2", "john3"]
    assert probes == [True, True]
    assert sum(" fstrim " in command[-1] for command in runner.commands) == 2
    assert sum(command[-1].endswith("colima stop") for command in runner.commands) == 2
    assert sum(command[-1].endswith("colima start") for command in runner.commands) == 2
    assert json.loads(receipt.read_text()) == value


def test_remote_reclaim_refuses_live_worker_container(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.ColimaReclaimError, match="john2 still has running containers"):
        module.reclaim_remote_workers(
            _completion(tmp_path / "completion.json"),
            tmp_path / "remote.json",
            runner=FakeRemoteRunner([], live_host="john2"),
            sleeper=lambda _: None,
            fabric_probe=lambda: None,
        )


def test_remote_reclaim_receipt_is_idempotent(tmp_path: Path) -> None:
    module = _module()
    receipt = tmp_path / "remote.json"
    expected = {"passed": True, "schema_id": "already-complete"}
    receipt.write_text(json.dumps(expected) + "\n")
    runner = FakeRemoteRunner([])
    assert module.reclaim_remote_workers(
        tmp_path / "missing.json",
        receipt,
        runner=runner,
        fabric_probe=lambda: None,
    ) == expected
    assert runner.commands == []
