"""Typed, CAS-protected pre-controller D0 dashboard diagnostics."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from .canonical import CAMPAIGN_ID, D0_RUN_ID, D0Error, canonical_json, document_sha256
from .inventory import secure_owner_directory
from .storage import (
    CANONICAL_ROOT,
    verify_canonical_commit_boundary,
    verify_canonical_storage,
)

DASHBOARD_STATUS = "control/dashboard-status.json"
DASHBOARD_LOCK = "control/.dashboard-status.json.lock"
DASHBOARD_RECEIPTS = "control/dashboard-diagnostic-receipts"
STATUS_SCHEMA = "cascadia.r2-map.dashboard-status.v1"
SPEC_SCHEMA = "cascadia.r2-map.d0-dashboard-diagnostic-update.v2"
RECEIPT_SCHEMA = "cascadia.r2-map.d0-dashboard-diagnostic-receipt.v1"
MAX_STATUS_BYTES = 65_536
SHA256_LENGTH = 64


class GateStateDocument(TypedDict):
    d0_gate: Literal["red", "green"]
    w0_gate: Literal["red", "green"]
    d0_state_sha256: str
    d0_report_sha256: str
    blocker_codes: list[str]
    host_gates: dict[str, dict[str, Any]]


class DiagnosticSpecDocument(TypedDict):
    schema_id: str
    schema_version: int
    campaign_id: str
    run_id: str
    expected_current_sha256: str
    updated_unix_ms: int
    stale_after_seconds: int
    gate_state: GateStateDocument
    spec_sha256: str


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D0Error(f"{label} is not a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise D0Error(f"{label} fields differ")
    return value


def validate_diagnostic_spec(value: Any) -> DiagnosticSpecDocument:
    spec = _exact_keys(
        value,
        {
            "schema_id",
            "schema_version",
            "campaign_id",
            "run_id",
            "expected_current_sha256",
            "updated_unix_ms",
            "stale_after_seconds",
            "gate_state",
            "spec_sha256",
        },
        "dashboard diagnostic spec",
    )
    expected = spec["expected_current_sha256"]
    if expected != "absent":
        _sha256(expected, "expected dashboard status")
    if (
        spec["schema_id"] != SPEC_SCHEMA
        or spec["schema_version"] != 2
        or spec["campaign_id"] != CAMPAIGN_ID
        or spec["run_id"] != D0_RUN_ID
        or not isinstance(spec["updated_unix_ms"], int)
        or isinstance(spec["updated_unix_ms"], bool)
        or spec["updated_unix_ms"] <= 0
        or not isinstance(spec["stale_after_seconds"], int)
        or isinstance(spec["stale_after_seconds"], bool)
        or not 5 <= spec["stale_after_seconds"] <= 3600
        or spec["spec_sha256"] != document_sha256(spec, "spec_sha256")
    ):
        raise D0Error("dashboard diagnostic spec identity differs")
    gate = _exact_keys(
        spec["gate_state"],
        {
            "d0_gate",
            "w0_gate",
            "d0_state_sha256",
            "d0_report_sha256",
            "blocker_codes",
            "host_gates",
        },
        "dashboard diagnostic gate state",
    )
    _sha256(gate["d0_state_sha256"], "D0 state")
    _sha256(gate["d0_report_sha256"], "D0 report")
    blockers = gate["blocker_codes"]
    host_gates = _exact_keys(
        gate["host_gates"],
        {"john1", "john2", "john3"},
        "dashboard per-host gate state",
    )
    for host, host_gate_value in host_gates.items():
        host_gate = _exact_keys(
            host_gate_value,
            {"status", "state_sha256", "evidence_sha256", "blocker_codes"},
            f"dashboard {host} gate",
        )
        _sha256(host_gate["state_sha256"], f"{host} state")
        _sha256(host_gate["evidence_sha256"], f"{host} evidence")
        host_blockers = host_gate["blocker_codes"]
        if (
            host_gate["status"] not in {"red", "green"}
            or not isinstance(host_blockers, list)
            or host_blockers != sorted(set(host_blockers))
            or len(host_blockers) > 8
            or any(
                not isinstance(item, str)
                or not item
                or len(item) > 32
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                    for character in item
                )
                for item in host_blockers
            )
            or (host_gate["status"] == "green") == bool(host_blockers)
        ):
            raise D0Error(f"dashboard {host} gate state differs")
    all_hosts_green = all(host_gates[host]["status"] == "green" for host in host_gates)
    if (
        gate["d0_gate"] not in {"red", "green"}
        or gate["w0_gate"] not in {"red", "green"}
        or not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or len(blockers) > 8
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > 32
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in item)
            for item in blockers
        )
        or ((gate["d0_gate"], gate["w0_gate"]) == ("green", "green")) == bool(blockers)
        or (gate["d0_gate"] == "green") != all_hosts_green
    ):
        raise D0Error("dashboard diagnostic gate state differs")
    return dict(spec)  # type: ignore[return-value]


def render_diagnostic_spec(value: Mapping[str, Any]) -> bytes:
    document = dict(value)
    document["spec_sha256"] = document_sha256(document, "spec_sha256")
    validate_diagnostic_spec(document)
    return canonical_json(document)




def _inactive_work() -> dict[str, Any]:
    return {
        "benchmark_pairs_completed": 0,
        "benchmark_pairs_total": None,
        "eta_seconds": None,
        "generation_games_completed": 0,
        "generation_games_target": None,
        "generation_seed_prefix": None,
        "rss_bytes": None,
        "swap_delta_bytes": None,
        "throughput_games_per_second": None,
    }


def build_dashboard_diagnostic(spec: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_diagnostic_spec(dict(spec))
    gate = validated["gate_state"]
    ready = (gate["d0_gate"], gate["w0_gate"]) == ("green", "green")
    phase = "contracts-ready" if ready else "d0-blocked"
    digest_binding = (
        f"state_sha256={gate['d0_state_sha256']} report_sha256={gate['d0_report_sha256']}"
    )
    blockers = ",".join(gate["blocker_codes"])
    hosts: dict[str, dict[str, Any]] = {}
    for host in ("john1", "john2", "john3"):
        host_gate = gate["host_gates"][host]
        host_blockers = ",".join(host_gate["blocker_codes"])
        detail = (
            f"d0-runtime:{host_gate['status']} "
            f"host_state_sha256={host_gate['state_sha256']} "
            f"host_evidence_sha256={host_gate['evidence_sha256']} {digest_binding}"
        )
        if host_blockers:
            detail += f" host_blockers={host_blockers}"
        if blockers:
            detail += f" blockers={blockers}"
        hosts[host] = {"intent": "idle", "detail": detail, **_inactive_work()}
    status = {
        "benchmark": {
            "active": False,
            "classification": "pending",
            "eta_seconds": None,
            "focal": None,
            "paired_delta": None,
            "pairs_completed": 0,
            "pairs_total": None,
            "peak_rss_bytes": None,
            "stage": None,
            "swap_delta_bytes": None,
            "throughput_games_per_second": None,
        },
        "campaign_id": CAMPAIGN_ID,
        "hosts": hosts,
        "legal_next_transitions": ["bootstrap-generating"] if ready else [],
        "models": {"candidate": None, "incumbent": None, "opponent_pool": []},
        "phase": phase,
        "round_index": None,
        "schema_id": STATUS_SCHEMA,
        "schema_version": 1,
        "stale_after_seconds": validated["stale_after_seconds"],
        "training": {
            "active": False,
            "current_step": None,
            "examples_per_second": None,
            "latest_verified_checkpoint": None,
            "loss_samples": [],
            "total_steps": None,
        },
        "updated_unix_ms": validated["updated_unix_ms"],
    }
    validate_dashboard_diagnostic(status, spec=validated)
    return status


def validate_dashboard_diagnostic(
    value: Any,
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_dashboard_diagnostic(spec) if value is None else None
    if expected is not None:  # pragma: no cover - defensive API guard
        return expected
    status = _exact_keys(
        value,
        {
            "benchmark",
            "campaign_id",
            "hosts",
            "legal_next_transitions",
            "models",
            "phase",
            "round_index",
            "schema_id",
            "schema_version",
            "stale_after_seconds",
            "training",
            "updated_unix_ms",
        },
        "dashboard status",
    )
    validated = validate_diagnostic_spec(dict(spec))
    gate = validated["gate_state"]
    ready = (gate["d0_gate"], gate["w0_gate"]) == ("green", "green")
    if (
        status["schema_id"] != STATUS_SCHEMA
        or status["schema_version"] != 1
        or status["campaign_id"] != CAMPAIGN_ID
        or status["phase"] != ("contracts-ready" if ready else "d0-blocked")
        or status["legal_next_transitions"] != (["bootstrap-generating"] if ready else [])
        or status["round_index"] is not None
        or status["updated_unix_ms"] != validated["updated_unix_ms"]
        or status["stale_after_seconds"] != validated["stale_after_seconds"]
        or set(status["hosts"]) != {"john1", "john2", "john3"}
        or any(status["hosts"][host]["intent"] != "idle" for host in status["hosts"])
        or any(
            f"d0-runtime:{gate['host_gates'][host]['status']}"
            not in status["hosts"][host]["detail"]
            or gate["host_gates"][host]["state_sha256"]
            not in status["hosts"][host]["detail"]
            or gate["host_gates"][host]["evidence_sha256"]
            not in status["hosts"][host]["detail"]
            for host in ("john1", "john2", "john3")
        )
        or any(
            gate[digest] not in canonical_json(status).decode("ascii")
            for digest in ("d0_state_sha256", "d0_report_sha256")
        )
    ):
        raise D0Error("dashboard status does not match its diagnostic gate state")
    if status["models"] != {"candidate": None, "incumbent": None, "opponent_pool": []}:
        raise D0Error("dashboard diagnostic invents model state")
    training = status["training"]
    benchmark = status["benchmark"]
    if (
        not isinstance(training, dict)
        or training.get("active") is not False
        or training.get("loss_samples") != []
        or not isinstance(benchmark, dict)
        or benchmark.get("active") is not False
        or benchmark.get("classification") != "pending"
    ):
        raise D0Error("dashboard diagnostic invents active research work")
    encoded = canonical_json(status)
    if len(encoded) > MAX_STATUS_BYTES:
        raise D0Error("dashboard diagnostic exceeds its compact byte limit")
    return dict(status)


def _read_regular_at(directory_fd: int, name: str) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or details.st_size > MAX_STATUS_BYTES
        ):
            raise D0Error("dashboard status is not an owner-bound compact regular file")
        value = bytearray()
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            value.extend(chunk)
            if len(value) > MAX_STATUS_BYTES:
                raise D0Error("dashboard status exceeds its compact byte limit")
        return bytes(value)
    finally:
        os.close(descriptor)


def _lock_at(directory_fd: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(Path(DASHBOARD_LOCK).name, flags, 0o600, dir_fd=directory_fd)
    details = os.fstat(descriptor)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.getuid()
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) & 0o022
    ):
        os.close(descriptor)
        raise D0Error("dashboard diagnostic lock identity differs")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise D0Error("another dashboard writer owns the diagnostic lock") from error
    return descriptor


def _stage_atomic_payload_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    except FileExistsError:
        details = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) != mode
        ):
            raise D0Error("dashboard diagnostic partial identity differs") from None
        existing = _read_regular_at(directory_fd, name)
        if existing == payload:
            return
        os.unlink(name, dir_fd=directory_fd)
        descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)




def publish_dashboard_diagnostic(
    spec: Mapping[str, Any],
    *,
    campaign_root: Path = CANONICAL_ROOT,
    storage_verifier: Callable[[], Mapping[str, Any]] = verify_canonical_storage,
    stability_seconds: int = 21,
) -> dict[str, Any]:
    """CAS one diagnostic status after John2 physical storage verification."""

    validated = validate_diagnostic_spec(dict(spec))
    if (
        not isinstance(stability_seconds, int)
        or isinstance(stability_seconds, bool)
        or not 0 <= stability_seconds <= 60
        or (campaign_root == CANONICAL_ROOT and stability_seconds < 21)
    ):
        raise D0Error("dashboard diagnostic stability interval differs")
    if (validated["gate_state"]["d0_gate"], validated["gate_state"]["w0_gate"]) == (
        "green",
        "green",
    ):
        raise D0Error("the pre-controller diagnostic updater cannot declare green gates")
    storage = dict(storage_verifier())
    if storage.get("status") != "pass":
        raise D0Error("John1 active-storage verification did not pass")
    storage_identity = _sha256(
        storage.get("host_identity_sha256"),
        "John1 active-storage identity",
    )
    control = campaign_root / "control"
    secure_owner_directory(control)
    control_fd = os.open(control, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    held_lock = -1
    temporary_name = f".dashboard-status.{validated['spec_sha256']}.partial"
    desired = canonical_json(build_dashboard_diagnostic(validated))
    desired_sha256 = hashlib.sha256(desired).hexdigest()
    try:
        held_lock = _lock_at(control_fd)
        current = _read_regular_at(control_fd, Path(DASHBOARD_STATUS).name)
        current_sha256 = "absent" if current is None else hashlib.sha256(current).hexdigest()
        if current_sha256 == desired_sha256:
            observed_disposition = "already-installed"
        else:
            if current_sha256 != validated["expected_current_sha256"]:
                raise D0Error("dashboard diagnostic compare-and-swap precondition failed")
            _stage_atomic_payload_at(control_fd, temporary_name, desired, mode=0o600)
            observed_again = _read_regular_at(control_fd, Path(DASHBOARD_STATUS).name)
            observed_again_sha256 = (
                "absent" if observed_again is None else hashlib.sha256(observed_again).hexdigest()
            )
            if observed_again_sha256 != current_sha256:
                os.unlink(temporary_name, dir_fd=control_fd)
                raise D0Error("dashboard status changed during its CAS transaction")
            verify_canonical_commit_boundary(control / Path(DASHBOARD_STATUS).name)
            os.replace(
                temporary_name,
                Path(DASHBOARD_STATUS).name,
                src_dir_fd=control_fd,
                dst_dir_fd=control_fd,
            )
            os.fsync(control_fd)
            installed = _read_regular_at(control_fd, Path(DASHBOARD_STATUS).name)
            if installed != desired:
                raise D0Error("dashboard diagnostic atomic install reread differs")
            observed_disposition = "installed"
        os.close(held_lock)
        held_lock = -1
        stability_started = time.monotonic()
        while True:
            elapsed = time.monotonic() - stability_started
            if elapsed >= stability_seconds:
                break
            time.sleep(min(1.0, stability_seconds - elapsed))
            observed = _read_regular_at(control_fd, Path(DASHBOARD_STATUS).name)
            if observed != desired:
                raise D0Error("dashboard diagnostic was replaced during stability verification")
        held_lock = _lock_at(control_fd)
        if _read_regular_at(control_fd, Path(DASHBOARD_STATUS).name) != desired:
            raise D0Error("dashboard diagnostic changed before receipt persistence")
        verified_unix_ms = time.time_ns() // 1_000_000
        if (
            validated["updated_unix_ms"] > verified_unix_ms + 60_000
            or verified_unix_ms
            > validated["updated_unix_ms"] + validated["stale_after_seconds"] * 1000
        ):
            raise D0Error("dashboard diagnostic is not fresh after stability verification")
        receipt: dict[str, Any] = {
            "schema_id": RECEIPT_SCHEMA,
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "run_id": D0_RUN_ID,
            "spec_sha256": validated["spec_sha256"],
            "d0_state_sha256": validated["gate_state"]["d0_state_sha256"],
            "d0_report_sha256": validated["gate_state"]["d0_report_sha256"],
            "previous_sha256": validated["expected_current_sha256"],
            "status_sha256": desired_sha256,
            "phase": build_dashboard_diagnostic(validated)["phase"],
            "legal_next_transitions": build_dashboard_diagnostic(validated)[
                "legal_next_transitions"
            ],
            "path": str(control / Path(DASHBOARD_STATUS).name),
            "lock_path": str(control / Path(DASHBOARD_LOCK).name),
            "storage_identity_sha256": storage_identity,
            "stability_seconds": stability_seconds,
            "disposition": "committed",
            "status": "pass",
        }
        receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
        persistence = _persist_receipt(campaign_root, receipt)
        return {
            "receipt": receipt,
            "observed_disposition": observed_disposition,
            "observed_verified_unix_ms": verified_unix_ms,
            "persistence": persistence,
        }
    finally:
        if held_lock >= 0:
            os.close(held_lock)
        os.close(control_fd)


def _persist_receipt(campaign_root: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    directory = campaign_root / DASHBOARD_RECEIPTS
    secure_owner_directory(directory)
    payload = canonical_json(receipt)
    digest = receipt["receipt_sha256"]
    if digest != document_sha256(receipt, "receipt_sha256"):
        raise D0Error("dashboard diagnostic receipt digest differs")
    path = directory / f"{digest}.json"
    temporary_name = f".{digest}.partial"
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        existing = _read_regular_at(parent, path.name)
        if existing is not None:
            if existing != payload:
                raise D0Error("dashboard diagnostic receipt path contains different bytes")
            return {"relative": str(path.relative_to(campaign_root)), "disposition": "present"}
        _stage_atomic_payload_at(parent, temporary_name, payload, mode=0o400)
        appeared = _read_regular_at(parent, path.name)
        if appeared is not None:
            if appeared != payload:
                raise D0Error("dashboard diagnostic receipt appeared with different bytes")
            os.unlink(temporary_name, dir_fd=parent)
            return {"relative": str(path.relative_to(campaign_root)), "disposition": "present"}
        verify_canonical_commit_boundary(path)
        os.replace(temporary_name, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    return {"relative": str(path.relative_to(campaign_root)), "disposition": "installed"}
