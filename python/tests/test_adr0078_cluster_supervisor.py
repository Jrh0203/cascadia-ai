from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
runtime = importlib.import_module("adr0078_cluster_runtime")
transport = importlib.import_module("adr0078_cluster_transport")
collection = importlib.import_module("adr0078_collection")
training = importlib.import_module("adr0078_training")
sealed_test = importlib.import_module("adr0079_cluster_handoff")


def test_cluster_orchestration_modules_stay_focused() -> None:
    limits = {
        "adr0078_cluster_supervisor.py": 100,
        "adr0078_cluster_runtime.py": 500,
        "adr0078_cluster_transport.py": 150,
        "adr0078_collection.py": 300,
        "adr0078_training.py": 300,
        "adr0079_cluster_handoff.py": 450,
    }
    for name, limit in limits.items():
        lines = (TOOLS / name).read_text().count("\n") + 1
        assert lines <= limit, f"{name} grew to {lines} lines; split ownership before extending it"


def test_training_action_distinguishes_completion_resume_and_failure() -> None:
    assert (
        training.training_action(
            final_report=True,
            process_running=False,
            status=0,
            run_manifest=True,
            latest_checkpoint=True,
        )
        == "complete"
    )
    assert (
        training.training_action(
            final_report=True,
            process_running=True,
            status=None,
            run_manifest=True,
            latest_checkpoint=True,
        )
        == "monitor"
    )
    assert (
        training.training_action(
            final_report=False,
            process_running=True,
            status=None,
            run_manifest=True,
            latest_checkpoint=True,
        )
        == "monitor"
    )
    assert (
        training.training_action(
            final_report=False,
            process_running=False,
            status=None,
            run_manifest=True,
            latest_checkpoint=True,
        )
        == "resume"
    )
    assert (
        training.training_action(
            final_report=False,
            process_running=False,
            status=1,
            run_manifest=True,
            latest_checkpoint=True,
        )
        == "fail"
    )


def test_training_stall_requires_a_real_old_progress_timestamp() -> None:
    now = 10_000
    assert not training.training_is_stalled(progress_mtime=0, now=now)
    assert not training.training_is_stalled(
        progress_mtime=now - runtime.STALE_TRAINING_SECONDS,
        now=now,
    )
    assert training.training_is_stalled(
        progress_mtime=now - runtime.STALE_TRAINING_SECONDS - 1,
        now=now,
    )


def test_frozen_test_collection_command_cannot_drift() -> None:
    assert runtime.TEST_COLLECT_COMMAND == [
        "./target/release/cascadia-v2",
        "collect-counterfactual-advantage",
        "--output",
        "artifacts/datasets/r12-counterfactual-advantage-v1-test-32",
        "--games",
        "32",
        "--first-game-index",
        "71000",
        "--split",
        "test",
        "--groups-per-game",
        "16",
        "--samples-per-candidate",
        "12",
        "--candidate-selection",
        "stratified",
        "--resume",
    ]


def test_test_authorization_is_written_only_while_all_nodes_are_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "validation.json"
    report.write_text('{"passed": true}\n')
    authorization = tmp_path / "run" / "test-authorization.json"
    monkeypatch.setattr(runtime, "CANONICAL_JSON_REPORT", report)
    monkeypatch.setattr(runtime, "TEST_AUTHORIZATION", authorization)
    monkeypatch.setattr(runtime, "TEST_DATASET", tmp_path / "test")
    monkeypatch.setattr(runtime, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(runtime, "log", lambda _message: None)
    monkeypatch.setattr(sealed_test.time, "time", lambda: 123)
    monkeypatch.setattr(
        runtime,
        "remote_path_exists",
        lambda host, _path: host == "john2",
    )
    validation = {
        "passed": True,
        "failed_gates": [],
        "checkpoint": "step-100",
        "checkpoint_manifest_blake3": "checkpoint-hash",
    }
    with pytest.raises(RuntimeError, match="existed before"):
        sealed_test.authorize_test_collection(validation)
    assert not authorization.exists()

    monkeypatch.setattr(runtime, "remote_path_exists", lambda _host, _path: False)
    value = sealed_test.authorize_test_collection(validation)
    assert json.loads(authorization.read_text()) == value
    assert value["authorized_at_unix_seconds"] == 123
    assert all(value["test_absent_on_nodes"].values())

    value["test_absent_on_nodes"]["john3"] = False
    authorization.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="did not prove"):
        sealed_test.authorize_test_collection(validation)


def test_manifest_contract_rejects_split_and_executable_drift() -> None:
    manifest = {
        "split": "train",
        "requested_games": 128,
        "completed_games": 128,
        "first_game_index": 69_000,
        "provenance": {
            "git_revision": runtime.EXPECTED_REVISION,
            "executable_blake3": runtime.EXPECTED_EXECUTABLE_BLAKE3,
        },
    }
    runtime.validate_manifest_contract(
        manifest,
        runtime.TRAIN_SPEC,
        require_complete=True,
    )

    wrong_split = {**manifest, "split": "validation"}
    with pytest.raises(ValueError, match="split"):
        runtime.validate_manifest_contract(
            wrong_split,
            runtime.TRAIN_SPEC,
            require_complete=True,
        )

    wrong_binary = {
        **manifest,
        "provenance": {
            **manifest["provenance"],
            "executable_blake3": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="executable"):
        runtime.validate_manifest_contract(
            wrong_binary,
            runtime.TRAIN_SPEC,
            require_complete=True,
        )


def test_collection_preserves_progress_through_remote_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = {
        "split": "train",
        "requested_games": 128,
        "completed_games": 128,
        "first_game_index": 69_000,
        "provenance": {
            "git_revision": runtime.EXPECTED_REVISION,
            "executable_blake3": runtime.EXPECTED_EXECUTABLE_BLAKE3,
        },
    }
    validation = {
        "split": "validation",
        "requested_games": 32,
        "completed_games": 32,
        "first_game_index": 70_000,
        "provenance": {
            "git_revision": runtime.EXPECTED_REVISION,
            "executable_blake3": runtime.EXPECTED_EXECUTABLE_BLAKE3,
        },
    }
    validation_reads = iter([runtime.RemoteHostUnavailable("john2 is asleep"), validation])
    states: list[tuple[str, dict[str, object]]] = []

    def load_manifest(spec: runtime.DatasetSpec) -> dict:
        if spec is runtime.TRAIN_SPEC:
            return train
        value = next(validation_reads)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        runtime,
        "load_state",
        lambda: {"train_completed": 127, "validation_completed": 12},
    )
    monkeypatch.setattr(runtime, "load_manifest", load_manifest)
    monkeypatch.setattr(runtime, "collector_running", lambda _spec: True)
    monkeypatch.setattr(
        runtime,
        "update_state",
        lambda stage, **values: states.append((stage, values)),
    )
    monkeypatch.setattr(runtime, "log", lambda _message: None)
    monkeypatch.setattr(collection.time, "sleep", lambda _seconds: None)

    result = collection.wait_for_collections()

    assert result == (train, validation)
    assert not any(values.get("validation_completed") == 0 for _, values in states)


def test_remote_path_probe_never_treats_ssh_failure_as_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = runtime.subprocess.CompletedProcess([], 255, "", "timed out")
    monkeypatch.setattr(runtime, "remote_shell", lambda *_args, **_kwargs: result)

    with pytest.raises(runtime.RemoteHostUnavailable, match="john2"):
        runtime.remote_path_exists("john2", Path("/sealed/test"))


def test_binary_identity_can_defer_only_the_remote_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "sha256_file", lambda _path: runtime.EXPECTED_EXECUTABLE_SHA256)
    local = runtime.subprocess.CompletedProcess(
        [],
        0,
        f"{runtime.EXPECTED_EXECUTABLE_BLAKE3}  binary\n",
        "",
    )
    remote = runtime.subprocess.CompletedProcess([], 255, "", "timed out")
    monkeypatch.setattr(runtime, "run", lambda *_args, **_kwargs: local)
    monkeypatch.setattr(runtime, "remote_shell", lambda *_args, **_kwargs: remote)
    monkeypatch.setattr(runtime, "log", lambda _message: None)

    collection.verify_binary_identity(require_remote=False)
    with pytest.raises(runtime.RemoteHostUnavailable, match="john2"):
        collection.verify_binary_identity()


def test_lock_owner_parses_pid_and_process_probe_handles_current_process(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "supervisor.lock"
    lock.write_text("1234\n")
    assert runtime.lock_owner(lock) == 1234
    assert runtime.process_exists(runtime.os.getpid())
    lock.write_text("not-a-pid\n")
    assert runtime.lock_owner(lock) is None


def test_john2_fallback_is_identity_pinned() -> None:
    primary, fallback = transport.endpoints("john2")
    assert primary.target == "john2"
    assert fallback.target == "john2@192.168.1.238"
    assert "StrictHostKeyChecking=yes" in fallback.options
    assert "HostKeyAlias=100.100.43.38" in fallback.options
    assert str(Path.home() / ".ssh/john2_codex") in fallback.options


def test_remote_retries_unreachable_primary_but_not_remote_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    results = iter(
        [
            runtime.subprocess.CompletedProcess([], 255, "", "timed out"),
            runtime.subprocess.CompletedProcess([], 0, "ok", ""),
        ]
    )

    def run(command: list[str], **_kwargs: object) -> runtime.subprocess.CompletedProcess[str]:
        calls.append(command)
        result = next(results)
        result.args = command
        return result

    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(runtime, "log", lambda _message: None)
    result = runtime.remote("john2", "true")
    assert result.stdout == "ok"
    assert calls[0][-2:] == ["john2", "true"]
    assert calls[1][-2:] == ["john2@192.168.1.238", "true"]

    calls.clear()
    failure = runtime.subprocess.CompletedProcess([], 1, "", "missing")
    failure.args = ["ssh", "john2", "false"]
    monkeypatch.setattr(runtime, "run", lambda command, **_kwargs: failure)
    result = runtime.remote("john2", "false", check=False)
    assert result.returncode == 1


def test_state_update_removes_nullable_status_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"stage":"collecting","unavailable_host":"john2"}\n')
    monkeypatch.setattr(runtime, "STATE_PATH", state_path)
    runtime.update_state("collecting", unavailable_host=None)
    assert "unavailable_host" not in json.loads(state_path.read_text())
