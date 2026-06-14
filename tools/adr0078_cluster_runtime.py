"""Shared constants and fail-closed runtime helpers for ADR 0078/0079."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import adr0078_cluster_transport as transport


class RemoteHostUnavailable(RuntimeError):
    """The remote host could not be reached, so its state is unknown."""


ROOT = Path(__file__).resolve().parents[1]
B3SUM = Path.home() / ".cargo/bin/b3sum"
PGREP = Path("/usr/bin/pgrep")
LOCAL_BINARY = ROOT / "target/release/cascadia-v2"
TRAIN_DATASET = ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-train-128"
VALIDATION_DATASET = ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-validation-32"
TEST_DATASET = ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-test-32"
RUN_DIR = ROOT / "artifacts/runs/r12-counterfactual-advantage-set-ranker-v1"
LOG_DIR = ROOT / "artifacts/logs"
STATE_PATH = LOG_DIR / "adr0078-cluster-supervisor-state.json"
SUPERVISOR_LOG = LOG_DIR / "adr0078-cluster-supervisor.log"
CANONICAL_JSON_REPORT = (
    ROOT / "docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-validation.json"
)
CANONICAL_MARKDOWN_REPORT = (
    ROOT / "docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-validation.md"
)
CANONICAL_TEST_JSON_REPORT = (
    ROOT / "docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-test.json"
)
CANONICAL_TEST_MARKDOWN_REPORT = (
    ROOT / "docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-test.md"
)
TEST_EVALUATOR = ROOT / "tools/adr0079_counterfactual_advantage_test.py"
TEST_AUTHORIZATION = RUN_DIR / "test-authorization.json"

JOHN2_ROOT = Path("/Users/john2/cascadia-r12-adr0078")
JOHN3_ROOT = Path("/Users/john3/cascadia-r12-adr0078")
REMOTE_VALIDATION_DATASET = (
    JOHN2_ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-validation-32"
)
REMOTE_TEST_DATASET = JOHN2_ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-test-32"
REMOTE_TEST_LOG = JOHN2_ROOT / "artifacts/logs/adr0079-test-collection.log"
REMOTE_TEST_STATUS = JOHN2_ROOT / "artifacts/logs/adr0079-test-collection.status"
REMOTE_TEST_PID = JOHN2_ROOT / "artifacts/logs/adr0079-test-collection.pid"
JOHN3_TRAIN_DATASET = JOHN3_ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-train-128"
JOHN3_VALIDATION_DATASET = (
    JOHN3_ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-validation-32"
)
JOHN3_TEST_DATASET = JOHN3_ROOT / "artifacts/datasets/r12-counterfactual-advantage-v1-test-32"
JOHN3_RUN_DIR = JOHN3_ROOT / "artifacts/runs/r12-counterfactual-advantage-set-ranker-v1"
JOHN3_BINARY = JOHN3_ROOT / "target/release/cascadia-v2"
JOHN3_TEST_EVALUATOR = JOHN3_ROOT / "tools/adr0079_counterfactual_advantage_test.py"
JOHN3_TEST_AUTHORIZATION = JOHN3_RUN_DIR / "test-authorization.json"
JOHN3_TRAIN_LOG = JOHN3_ROOT / "artifacts/logs/adr0078-train.log"
JOHN3_TRAIN_STATUS = JOHN3_ROOT / "artifacts/logs/adr0078-train.status"
JOHN3_TRAIN_PID = JOHN3_ROOT / "artifacts/logs/adr0078-train.pid"

EXPECTED_REVISION = "a9918946f66c237a803b23ea299c6a514785ae52"
EXPECTED_JOHN3_SOURCE = "2c761bb49bbc22fc84fbac437242025afe103322cfd5336f4f0950cf183426d4"
EXPECTED_EXECUTABLE_BLAKE3 = "183192792323090bac31de9ba8e4327ae466cb066f844447ef6a8c696fc122d1"
EXPECTED_EXECUTABLE_SHA256 = "6eb4fd471aae9611456ba93e5f17ff9cdd103f7a4579a54b94b1cdf1fb68be52"

POLL_SECONDS = 60
TRAIN_POLL_SECONDS = 30
STALE_PROGRESS_SECONDS = 45 * 60
STALE_TRAINING_SECONDS = 30 * 60
MAX_ABRUPT_RESUMES = 3
MAX_TEST_ABRUPT_RESUMES = 3

TRAIN_COMMAND = [
    "env",
    "PYTHONPATH=python",
    ".venv/bin/python",
    "-m",
    "cascadia_mlx.counterfactual_advantage_train",
    "--train-dataset",
    str(JOHN3_TRAIN_DATASET.relative_to(JOHN3_ROOT)),
    "--validation-dataset",
    str(JOHN3_VALIDATION_DATASET.relative_to(JOHN3_ROOT)),
    "--run-dir",
    str(JOHN3_RUN_DIR.relative_to(JOHN3_ROOT)),
    "--epochs",
    "20",
    "--group-batch-size",
    "32",
    "--learning-rate",
    "0.0001",
    "--weight-decay",
    "0.0001",
    "--seed",
    "20260614",
    "--checkpoint-steps",
    "100",
    "--validation-patience",
    "5",
]

EVALUATE_COMMAND = [
    "env",
    "PYTHONPATH=python",
    ".venv/bin/python",
    "-m",
    "cascadia_mlx.counterfactual_advantage_evaluate",
    "--run-dir",
    str(JOHN3_RUN_DIR.relative_to(JOHN3_ROOT)),
    "--dataset",
    str(JOHN3_VALIDATION_DATASET.relative_to(JOHN3_ROOT)),
    "--output",
    str((JOHN3_RUN_DIR / "validation-report.json").relative_to(JOHN3_ROOT)),
    "--markdown-output",
    str((JOHN3_RUN_DIR / "validation-report.md").relative_to(JOHN3_ROOT)),
    "--group-batch-size",
    "32",
]

TEST_COLLECT_COMMAND = [
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


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    manifest_path: Path
    split: str
    requested_games: int
    first_game_index: int
    process_pattern: str
    host: str | None = None


TRAIN_SPEC = DatasetSpec(
    label="train",
    manifest_path=TRAIN_DATASET / "dataset.json",
    split="train",
    requested_games=128,
    first_game_index=69_000,
    process_pattern=(
        "[c]ollect-counterfactual-advantage.*r12-counterfactual-advantage-v1-train-128"
    ),
)
VALIDATION_SPEC = DatasetSpec(
    label="validation",
    manifest_path=REMOTE_VALIDATION_DATASET / "dataset.json",
    split="validation",
    requested_games=32,
    first_game_index=70_000,
    process_pattern=(
        "[c]ollect-counterfactual-advantage.*r12-counterfactual-advantage-v1-validation-32"
    ),
    host="john2",
)
TEST_SPEC = DatasetSpec(
    label="test",
    manifest_path=REMOTE_TEST_DATASET / "dataset.json",
    split="test",
    requested_games=32,
    first_game_index=71_000,
    process_pattern=("[c]ollect-counterfactual-advantage.*r12-counterfactual-advantage-v1-test-32"),
    host="john2",
)


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    capture: bool = True,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        log(f"run: {shlex.join(command)}")
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=capture,
        text=True,
    )
    if not quiet and capture and result.stdout.strip():
        log(f"stdout: {result.stdout.strip()[-2_000:]}")
    if not quiet and capture and result.stderr.strip():
        log(f"stderr: {result.stderr.strip()[-2_000:]}")
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {shlex.join(command)}")
    return result


def remote(
    host: str,
    command: str,
    *,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    return run_transport_candidates(
        transport.ssh_commands(host, command),
        host=host,
        check=check,
        quiet=quiet,
    )


def remote_shell(
    host: str,
    arguments: list[str],
    *,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    return remote(host, shlex.join(arguments), check=check, quiet=quiet)


def rsync_from_remote(
    host: str,
    remote_path: str,
    local_path: str,
    *,
    delete: bool = False,
) -> subprocess.CompletedProcess[str]:
    return run_transport_candidates(
        transport.rsync_commands(
            host,
            local_path,
            remote_path,
            upload=False,
            delete=delete,
        ),
        host=host,
    )


def rsync_to_remote(
    host: str,
    local_path: str,
    remote_path: str,
    *,
    delete: bool = False,
) -> subprocess.CompletedProcess[str]:
    return run_transport_candidates(
        transport.rsync_commands(
            host,
            local_path,
            remote_path,
            upload=True,
            delete=delete,
        ),
        host=host,
    )


def run_transport_candidates(
    commands: list[list[str]],
    *,
    host: str,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = transport.run_candidates(
        commands,
        lambda command: run(command, check=False, quiet=quiet),
        lambda completed: completed.returncode,
        lambda: log(f"{host} primary transport unavailable; trying verified fallback"),
    )
    if check:
        if result.returncode == 255:
            raise RemoteHostUnavailable(f"{host} is unreachable")
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed with exit {result.returncode}: {shlex.join(result.args)}"
            )
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        value = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def update_state(stage: str, **values: Any) -> None:
    state = load_state()
    if stage != "failed":
        state.pop("error", None)
    for key, value in values.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    state["stage"] = stage
    state["updated_at"] = timestamp()
    atomic_json(STATE_PATH, state)


def load_manifest(spec: DatasetSpec) -> dict[str, Any] | None:
    if spec.host is None:
        if not spec.manifest_path.exists():
            return None
        return json.loads(spec.manifest_path.read_text())
    result = remote_shell(
        spec.host,
        ["cat", str(spec.manifest_path)],
        check=False,
        quiet=True,
    )
    if result.returncode == 1:
        return None
    if result.returncode == 255:
        raise RemoteHostUnavailable(f"{spec.host} is unreachable")
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read {spec.label} manifest on {spec.host}: exit {result.returncode}"
        )
    return json.loads(result.stdout)


def validate_manifest_contract(
    manifest: dict[str, Any],
    spec: DatasetSpec,
    *,
    require_complete: bool,
) -> None:
    expected = {
        "split": spec.split,
        "requested_games": spec.requested_games,
        "first_game_index": spec.first_game_index,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{spec.label} manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    completed = int(manifest.get("completed_games", -1))
    if completed < 0 or completed > spec.requested_games:
        raise ValueError(f"{spec.label} manifest has invalid completed_games={completed}")
    if require_complete and completed != spec.requested_games:
        raise ValueError(f"{spec.label} is incomplete: {completed}/{spec.requested_games} games")
    provenance = manifest.get("provenance", {})
    if provenance.get("git_revision") != EXPECTED_REVISION:
        raise ValueError(f"{spec.label} manifest revision changed")
    if provenance.get("executable_blake3") != EXPECTED_EXECUTABLE_BLAKE3:
        raise ValueError(f"{spec.label} manifest executable changed")


def collector_running(spec: DatasetSpec) -> bool:
    command = [str(PGREP), "-f", spec.process_pattern]
    if spec.host is None:
        return run(command, check=False, quiet=True).returncode == 0
    result = remote_shell(spec.host, command, check=False, quiet=True)
    if result.returncode == 255:
        raise RemoteHostUnavailable(f"{spec.host} is unreachable")
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"could not probe {spec.label} collector on {spec.host}: exit {result.returncode}"
        )
    return result.returncode == 0


def remote_path_exists(host: str, path: Path) -> bool:
    result = remote_shell(host, ["test", "-e", str(path)], check=False, quiet=True)
    if result.returncode == 255:
        raise RemoteHostUnavailable(f"{host} is unreachable")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"could not probe {path} on {host}: exit {result.returncode}")
    return result.returncode == 0


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_lock() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOG_DIR / "adr0078-cluster-supervisor.lock"
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            owner = lock_owner(lock)
            if attempt == 0 and owner is not None and not process_exists(owner):
                log(f"reclaiming stale supervisor lock from PID {owner}")
                lock.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"supervisor lock already exists: {lock}") from error
        os.write(descriptor, f"{os.getpid()}\n".encode())
        return descriptor
    raise RuntimeError(f"could not acquire supervisor lock: {lock}")


def lock_owner(lock: Path) -> int | None:
    try:
        return int(lock.read_text().strip())
    except (OSError, ValueError):
        return None


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def release_lock(descriptor: int) -> None:
    os.close(descriptor)
    (LOG_DIR / "adr0078-cluster-supervisor.lock").unlink(missing_ok=True)
