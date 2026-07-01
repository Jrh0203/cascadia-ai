"""Truthful pre-controller dashboard heartbeat for a RED D0 gate.

The heartbeat is deliberately narrow: it reports no model, game, training, or
benchmark state.  It is only valid while D0 is RED and is replaced by the
campaign controller after qualification.  The file is disposable projection
state; authoritative qualification evidence remains in signed D0 bundles.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from .canonical import CAMPAIGN_ID, canonical_json
from .storage import CANONICAL_ROOT, verify_canonical_storage

STATUS_PATH = CANONICAL_ROOT / "control/dashboard-status.json"
LOCK_PATH = CANONICAL_ROOT / "control/.dashboard-status.heartbeat.lock"
RUNTIME_RECEIPT = Path("/Users/johnherrick/.config/cascadia-r2/runtime-profile-receipt.json")
EVIDENCE_ROOT = (
    CANONICAL_ROOT
    / "reports/infrastructure/d0-runtime-bootstrap-20260618-v1/host-qualification"
)
JOHN2_RUNTIME_RECEIPT = EVIDENCE_ROOT / "john2/runtime-profile-receipt.json"
JOHN2_ARCHIVE_RECEIPT = EVIDENCE_ROOT / "john2/cold-archive-root-receipt.json"
JOHN3_RUNTIME_RECEIPT = EVIDENCE_ROOT / "john3/runtime-profile-receipt.json"
JOHN3_CLEANUP_RECEIPT = EVIDENCE_ROOT / "john3/legacy-cleanup-receipt.json"
JOHN3_ARCHIVE_REOPEN_RECEIPT = (
    CANONICAL_ROOT
    / "reports/infrastructure/d0-runtime-bootstrap-20260618-v1/archive-transactions/"
    "c2ab29624a4a442b8f05976751befad476f2379c60bb82eb70127f341915efbe/"
    "john1-reopen-receipt.json"
)
EXPECTED_EVIDENCE_SHA256 = {
    JOHN2_RUNTIME_RECEIPT: "626a63870d14674de5f6b179e4f632afb25e856ff04f3e17ebb0f15614a26cf8",
    JOHN2_ARCHIVE_RECEIPT: "f147ef654dca77e93fe550b2edaa2841ab14ac3947d1e418cf651670d49758c3",
    JOHN3_RUNTIME_RECEIPT: "0a1902f6298187f783fc13c70b966acae51344d1b1d0f69a41f778e433430aec",
    JOHN3_ARCHIVE_REOPEN_RECEIPT: (
        "4b3e8cae700a44093b918cf9b166b4ea81afd468bc84a4b84247878d5e1573b1"
    ),
    JOHN3_CLEANUP_RECEIPT: (
        "341784ef268960ef65f39d5ac105da58b922f8c27c90dfb175b6157815241a47"
    ),
}
STATUS_SCHEMA = "cascadia.r2-map.dashboard-status.v1"
MAX_STATUS_BYTES = 64 * 1024
STALE_AFTER_SECONDS = 30
LOCAL_COMMAND_TIMEOUT_SECONDS = 10


class LiveDashboardError(RuntimeError):
    """The bounded RED-gate dashboard heartbeat could not be produced safely."""


def _run(argv: list[str], *, extra_env: Mapping[str, str] | None = None) -> tuple[int, str]:
    environment = {
        "LC_ALL": "C",
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        **dict(extra_env or {}),
    }
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=LOCAL_COMMAND_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, type(error).__name__
    output = (completed.stdout + completed.stderr).strip().replace("\n", " ")
    return completed.returncode, output[:160]


def _secure_file_sha256(path: Path, *, maximum: int = 128 * 1024) -> str | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) & 0o077
            or details.st_size > maximum
        ):
            return None
        digest = hashlib.sha256()
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            return None
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _secure_json_document(path: Path, *, maximum: int = 128 * 1024) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) & 0o077
            or details.st_size > maximum
        ):
            return None
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != details.st_size:
            return None
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or bytes(payload) not in {canonical_json(value), canonical_json(value) + b"\n"}
    ):
        return None
    return value


def _john3_role_receipt_valid(path: Path = JOHN3_RUNTIME_RECEIPT) -> bool:
    value = _secure_json_document(path)
    certification = value.get("certification") if isinstance(value, Mapping) else None
    cleanup = value.get("lineage", {}).get("cleanup") if isinstance(value, Mapping) else None
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_id") == "cascadia.r2-map.local-runtime-profile-receipt.v4"
        and value.get("schema_version") == 4
        and value.get("host") == "john3"
        and value.get("role") == "execution-only"
        and value.get("legacy_native_workspace") == "absent"
        and isinstance(certification, Mapping)
        and certification.get("host_role_qualified") is True
        and certification.get("d0_certified") is False
        and certification.get("project_execution_authorized") is False
        and certification.get("blocker") == "Signed D0 topology aggregate is pending."
        and isinstance(cleanup, Mapping)
        and cleanup.get("status") == "pass"
        and cleanup.get("completion_receipt", {}).get("sha256")
        == EXPECTED_EVIDENCE_SHA256[JOHN3_CLEANUP_RECEIPT]
    )


def collect_local_facts() -> dict[str, Any]:
    version_status, version = _run(["/usr/bin/sw_vers", "-productVersion"])
    build_status, build = _run(["/usr/bin/sw_vers", "-buildVersion"])
    darwin_status, darwin = _run(["/usr/bin/uname", "-r"])
    colima_status, colima = _run(
        ["/opt/homebrew/bin/colima", "status", "--profile", "cascadia-r2"],
        extra_env={
            "COLIMA_HOME": "/Users/johnherrick/.local/share/cascadia-r2/colima",
            "DOCKER_CONFIG": "/Users/johnherrick/.config/cascadia-r2/docker",
        },
    )
    receipt_sha256 = _secure_file_sha256(RUNTIME_RECEIPT)
    storage = verify_canonical_storage(measure_size=True)
    evidence = {
        path.name if path.parent.name != "john2" else f"john2-{path.name}": (
            _secure_file_sha256(path) == expected
        )
        for path, expected in EXPECTED_EVIDENCE_SHA256.items()
    }
    evidence[JOHN3_RUNTIME_RECEIPT.name] = (
        evidence.get(JOHN3_RUNTIME_RECEIPT.name) is True
        and _john3_role_receipt_valid()
    )
    return {
        "os_version": version if version_status == 0 else "unavailable",
        "os_build": build if build_status == 0 else "unavailable",
        "darwin": darwin if darwin_status == 0 else "unavailable",
        "runtime_state": "running" if colima_status == 0 else "stopped",
        "runtime_observation": colima,
        "runtime_receipt_sha256": receipt_sha256,
        "storage_receipt_sha256": storage["receipt_sha256"],
        "storage_free_bytes": storage["free_bytes"],
        "storage_campaign_bytes": storage["campaign_apparent_bytes"],
        "evidence": evidence,
    }


def _inactive_work() -> dict[str, Any]:
    return {
        "generation_games_completed": 0,
        "generation_games_target": None,
        "generation_seed_prefix": None,
        "benchmark_pairs_completed": 0,
        "benchmark_pairs_total": None,
        "eta_seconds": None,
        "throughput_games_per_second": None,
        "rss_bytes": None,
        "swap_delta_bytes": None,
    }


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def build_red_status(facts: Mapping[str, Any], *, updated_unix_ms: int) -> dict[str, Any]:
    if (
        not isinstance(updated_unix_ms, int)
        or isinstance(updated_unix_ms, bool)
        or updated_unix_ms <= 0
    ):
        raise LiveDashboardError("dashboard heartbeat timestamp is invalid")
    runtime = facts.get("runtime_state")
    if runtime not in {"running", "stopped"}:
        raise LiveDashboardError("John1 runtime state is invalid")
    receipt = facts.get("runtime_receipt_sha256")
    receipt_text = receipt if isinstance(receipt, str) else "absent"
    john1_detail = (
        "d0=RED; role=control+execution+mlx; "
        f"runtime={runtime}; os={facts.get('os_version')}/{facts.get('os_build')}; "
        f"darwin={facts.get('darwin')}; active_storage=john1-internal-apfs; "
        f"runtime_receipt_sha256={receipt_text}; required=signed-d0-topology-aggregate"
    )
    evidence = facts.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    john2_qualified = (
        evidence.get("john2-runtime-profile-receipt.json") is True
        and evidence.get("john2-cold-archive-root-receipt.json") is True
    )
    john3_qualified = evidence.get("runtime-profile-receipt.json") is True
    john3_archived = evidence.get("john1-reopen-receipt.json") is True
    john3_cleaned = evidence.get("legacy-cleanup-receipt.json") is True
    remote_detail = {
        "john2": (
            "d0=RED; role=sole-builder+execution+cold-archive; "
            f"role_qualification={'pass' if john2_qualified else 'missing'}; "
            "buildx=present-authorized; active_storage_authority=false; "
            "cold_archive_root=qualified"
        ),
        "john3": (
            "d0=RED; role=execution-only; "
            f"role_qualification={'pass' if john3_qualified else 'missing'}; "
            "buildx=absent; active_storage_authority=false; "
            f"legacy_native_workspace={'absent' if john3_cleaned else 'cleanup-unverified'}; "
            f"archive_commit={'verified' if john3_archived else 'missing'}; "
            f"source_cleanup={'verified' if john3_cleaned else 'missing'}"
        ),
    }
    status = {
        "schema_version": 1,
        "schema_id": STATUS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "updated_unix_ms": updated_unix_ms,
        "stale_after_seconds": STALE_AFTER_SECONDS,
        "phase": "d0-blocked",
        "legal_next_transitions": [],
        "round_index": None,
        "models": {"incumbent": None, "candidate": None, "opponent_pool": []},
        "hosts": {
            "john1": {"intent": "control", "detail": john1_detail, **_inactive_work()},
            "john2": {"intent": "idle", "detail": remote_detail["john2"], **_inactive_work()},
            "john3": {"intent": "idle", "detail": remote_detail["john3"], **_inactive_work()},
        },
        "training": {
            "active": False,
            "latest_verified_checkpoint": None,
            "current_step": None,
            "total_steps": None,
            "examples_per_second": None,
            "loss_samples": [],
        },
        "benchmark": {
            "active": False,
            "stage": None,
            "pairs_completed": 0,
            "pairs_total": None,
            "eta_seconds": None,
            "throughput_games_per_second": None,
            "peak_rss_bytes": None,
            "swap_delta_bytes": None,
            "focal": None,
            "paired_delta": None,
            "classification": "pending",
        },
    }
    encoded = canonical_json(status)
    if len(encoded) > MAX_STATUS_BYTES:
        raise LiveDashboardError("dashboard heartbeat exceeds its byte limit")
    if set(status["hosts"]) != {"john1", "john2", "john3"}:
        raise LiveDashboardError("dashboard heartbeat host set differs")
    if "john4" in encoded.decode("ascii").lower():
        raise LiveDashboardError("dashboard heartbeat names the excluded host")
    return status


def publish_status(
    status: Mapping[str, Any],
    *,
    path: Path = STATUS_PATH,
    storage_verifier: Callable[..., Mapping[str, Any]] = verify_canonical_storage,
) -> dict[str, Any]:
    storage = dict(storage_verifier(measure_size=False))
    if storage.get("status") != "pass" or path.parent != CANONICAL_ROOT / "control":
        raise LiveDashboardError("dashboard heartbeat storage boundary differs")
    payload = canonical_json(dict(status))
    if len(payload) > MAX_STATUS_BYTES:
        raise LiveDashboardError("dashboard heartbeat exceeds its byte limit")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    lock_fd = os.open(
        LOCK_PATH.name,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory,
    )
    temporary: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        descriptor, name = tempfile.mkstemp(prefix=".dashboard-status.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o600)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        os.fsync(directory)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()
        os.close(lock_fd)
        os.close(directory)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "state_sha256": _identity(dict(status)),
        "updated_unix_ms": status["updated_unix_ms"],
    }


def publish_red_heartbeat() -> dict[str, Any]:
    facts = collect_local_facts()
    status = build_red_status(facts, updated_unix_ms=time.time_ns() // 1_000_000)
    return publish_status(status)
