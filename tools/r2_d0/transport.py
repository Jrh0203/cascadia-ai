"""Host-local D0 control transactions.

The root orchestrator is the only cross-host actor.  This module deliberately
contains no SSH client, peer key, worker channel, remote storage worker, or
canonical publication implementation.  John1 sends immutable signed control
envelopes directly to each host and installs returned signed bundles through
``r2_d0.ingress``.
"""

from __future__ import annotations

import base64
import contextlib
import os
import shutil
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .artifacts import atomic_install_bytes
from .bundle import (
    MAX_BUNDLE_BYTES,
    PERSISTENCE_EVIDENCE_NAME,
    PERSISTENCE_MONITOR_NAME,
    PERSISTENCE_RECEIPT_NAME,
    render_persistence_evidence,
    render_persistence_monitor,
    render_persistence_receipt,
    validate_persistence_transaction,
)
from .canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    PATH_CONTRACT,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    primary_operation,
    safe_relative,
    sha256_bytes,
    validate_host_report,
    validate_signature_bundle,
    validate_work_packet,
)
from .inventory import secure_owner_directory
from .signing import normalize_public_key, public_key_fingerprint, verify_stdin

CONTROL_ENVELOPE_SCHEMA = "cascadia.r2-map.d0-control-envelope.v2"
CONTROL_EXECUTION_CLAIM_SCHEMA = "cascadia.r2-map.d0-control-execution-claim.v2"
CONTROL_COMPLETION_SCHEMA = "cascadia.r2-map.d0-control-completion.v2"
CONTROL_INBOX_RECEIPT_SCHEMA = "cascadia.r2-map.d0-control-inbox-receipt.v2"
MAX_CONTROL_ENVELOPE_BYTES = 4 * 1024 * 1024


def ensure_owner_directory(path: Path, *, mode: int = 0o700) -> None:
    """Create an owner-private directory chain without accepting links."""

    secure_owner_directory(path, mode=mode)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    """Write one new owner-private file without following links."""

    parent_details = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_details.st_mode)
        or stat.S_ISLNK(parent_details.st_mode)
        or parent_details.st_uid != os.getuid()
        or path.exists()
        or path.is_symlink()
    ):
        raise D0Error("local transaction destination is unsafe or already present")
    temporary = path.parent / f".{path.name}.partial-{os.getpid()}-{time.time_ns()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise D0Error("local transaction made a short write")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_owner_regular(path: Path, *, maximum: int, label: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or details.st_size > maximum
        ):
            raise D0Error(f"{label} metadata differs")
        result = bytearray()
        while len(result) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        if len(result) != details.st_size:
            raise D0Error(f"{label} changed while reading")
        return bytes(result)
    finally:
        os.close(descriptor)


def render_control_envelope(
    packet_bytes: bytes,
    signature_bytes: bytes,
    *,
    public_key: bytes,
) -> bytes:
    """Bind one signed work packet to one direct orchestrator-to-host edge."""

    packet = load_canonical_json(
        packet_bytes, maximum=1024 * 1024, label="control work packet"
    )
    validate_work_packet(packet)
    signature = load_canonical_json(
        signature_bytes, maximum=1024 * 1024, label="control work packet signature"
    )
    validate_signature_bundle(signature, payload_sha256=sha256_bytes(packet_bytes))
    normalized_key = normalize_public_key(public_key)
    verify_stdin(normalized_key, packet_bytes, signature)
    if packet["public_key_fingerprint"] != public_key_fingerprint(normalized_key):
        raise D0Error("control work packet campaign key differs")
    operation = primary_operation(packet["host"], packet["phase"], packet["allowed_operations"])
    envelope: dict[str, Any] = {
        "schema_id": CONTROL_ENVELOPE_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "source_host": "john1",
        "target_host": packet["host"],
        "cycle_id": packet["cycle_id"],
        "phase": packet["phase"],
        "operation": operation,
        "packet_sha256": packet["packet_sha256"],
        "packet_size": len(packet_bytes),
        "signature_size": len(signature_bytes),
        "packet_base64": base64.b64encode(packet_bytes).decode("ascii"),
        "signature_base64": base64.b64encode(signature_bytes).decode("ascii"),
        "peer_credentials_present": False,
        "protected_seed_values_opened": False,
    }
    envelope["envelope_sha256"] = document_sha256(envelope, "envelope_sha256")
    encoded = canonical_json(envelope)
    if len(encoded) > MAX_CONTROL_ENVELOPE_BYTES:
        raise D0Error("signed control envelope exceeds its byte limit")
    return encoded


def verify_control_envelope(
    envelope_bytes: bytes,
    *,
    public_key: bytes,
    target_host: str | None = None,
    now_unix_ms: int | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    envelope = load_canonical_json(
        envelope_bytes,
        maximum=MAX_CONTROL_ENVELOPE_BYTES,
        label="signed control envelope",
    )
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "source_host",
        "target_host",
        "cycle_id",
        "phase",
        "operation",
        "packet_sha256",
        "packet_size",
        "signature_size",
        "packet_base64",
        "signature_base64",
        "peer_credentials_present",
        "protected_seed_values_opened",
        "envelope_sha256",
    }
    if (
        set(envelope) != required
        or envelope.get("schema_id") != CONTROL_ENVELOPE_SCHEMA
        or envelope.get("schema_version") != 2
        or envelope.get("campaign_id") != CAMPAIGN_ID
        or envelope.get("run_id") != D0_RUN_ID
        or envelope.get("source_host") != "john1"
        or envelope.get("target_host") not in {"john1", "john2", "john3"}
        or envelope.get("peer_credentials_present") is not False
        or envelope.get("protected_seed_values_opened") is not False
        or envelope.get("envelope_sha256")
        != document_sha256(envelope, "envelope_sha256")
        or (target_host is not None and envelope.get("target_host") != target_host)
    ):
        raise D0Error("signed control envelope identity differs")
    try:
        packet_bytes = base64.b64decode(envelope["packet_base64"], validate=True)
        signature_bytes = base64.b64decode(envelope["signature_base64"], validate=True)
    except (TypeError, ValueError) as error:
        raise D0Error("signed control envelope encoding differs") from error
    if (
        len(packet_bytes) != envelope["packet_size"]
        or len(signature_bytes) != envelope["signature_size"]
    ):
        raise D0Error("signed control envelope payload size differs")
    packet = load_canonical_json(
        packet_bytes, maximum=1024 * 1024, label="control work packet"
    )
    validate_work_packet(packet)
    signature = load_canonical_json(
        signature_bytes, maximum=1024 * 1024, label="control work packet signature"
    )
    normalized_key = normalize_public_key(public_key)
    validate_signature_bundle(signature, payload_sha256=sha256_bytes(packet_bytes))
    verify_stdin(normalized_key, packet_bytes, signature)
    operation = primary_operation(packet["host"], packet["phase"], packet["allowed_operations"])
    if any(
        envelope[field] != expected
        for field, expected in (
            ("target_host", packet["host"]),
            ("cycle_id", packet["cycle_id"]),
            ("phase", packet["phase"]),
            ("operation", operation),
            ("packet_sha256", packet["packet_sha256"]),
        )
    ) or packet["public_key_fingerprint"] != public_key_fingerprint(normalized_key):
        raise D0Error("signed control envelope packet binding differs")
    if require_current:
        now = time.time_ns() // 1_000_000 if now_unix_ms is None else now_unix_ms
        if (
            not isinstance(now, int)
            or isinstance(now, bool)
            or not packet["issued_unix_ms"] <= now <= packet["expires_unix_ms"]
        ):
            raise D0Error("signed control envelope is outside its validity window")
    return {
        "envelope": envelope,
        "packet": packet,
        "packet_bytes": packet_bytes,
        "signature": signature,
        "signature_bytes": signature_bytes,
        "envelope_size": len(envelope_bytes),
        "envelope_sha256": sha256_bytes(envelope_bytes),
        "status": "pass",
    }


def control_envelope_path(packet: Mapping[str, Any]) -> Path:
    operation = primary_operation(packet["host"], packet["phase"], packet["allowed_operations"])
    return (
        Path(PATH_CONTRACT[packet["host"]]["control_inbox"])
        / packet["cycle_id"]
        / packet["phase"]
        / operation
        / f"{packet['packet_sha256']}.control.json"
    )


def control_execution_claim_path(packet: Mapping[str, Any]) -> Path:
    return control_envelope_path(packet).with_name(f"{packet['packet_sha256']}.claim.json")


def control_completion_path(packet: Mapping[str, Any]) -> Path:
    return control_envelope_path(packet).with_name(
        f"{packet['packet_sha256']}.completion.json"
    )


def install_control_envelope(
    envelope_bytes: bytes,
    *,
    public_key: bytes,
    target_host: str,
    now_unix_ms: int | None = None,
) -> dict[str, Any]:
    verification = verify_control_envelope(
        envelope_bytes,
        public_key=public_key,
        target_host=target_host,
        now_unix_ms=now_unix_ms,
    )
    destination = control_envelope_path(verification["packet"])
    ensure_owner_directory(destination.parent)
    installed = atomic_install_bytes(destination, envelope_bytes)
    return {
        "schema_id": CONTROL_INBOX_RECEIPT_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "source_host": "john1",
        "target_host": target_host,
        "packet_sha256": verification["packet"]["packet_sha256"],
        "phase": verification["packet"]["phase"],
        "operation": verification["envelope"]["operation"],
        "envelope_size": len(envelope_bytes),
        "envelope_sha256": sha256_bytes(envelope_bytes),
        "path": str(destination),
        "disposition": installed["status"],
        "peer_credentials_present": False,
        "status": "pass",
    }


def _claim_document(verification: Mapping[str, Any]) -> dict[str, Any]:
    packet = verification["packet"]
    claim: dict[str, Any] = {
        "schema_id": CONTROL_EXECUTION_CLAIM_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "target_host": packet["host"],
        "cycle_id": packet["cycle_id"],
        "phase": packet["phase"],
        "operation": verification["envelope"]["operation"],
        "packet_sha256": packet["packet_sha256"],
        "envelope_sha256": verification["envelope_sha256"],
        "status": "claimed",
    }
    claim["claim_sha256"] = document_sha256(claim, "claim_sha256")
    return claim


def _validate_claim(payload: bytes, verification: Mapping[str, Any]) -> dict[str, Any]:
    observed = load_canonical_json(
        payload, maximum=1024 * 1024, label="control execution claim"
    )
    expected = _claim_document(verification)
    if observed != expected:
        raise D0Error("control execution claim differs")
    return observed


def _persistence_monitor(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    persistence = result.get("persistence")
    if not isinstance(persistence, Mapping):
        return None
    transaction = persistence.get("transaction")
    evidence = (
        transaction.get("persistence_evidence")
        if isinstance(transaction, Mapping)
        else persistence.get("persistence_evidence")
    )
    if not isinstance(evidence, Mapping):
        return None
    monitor = evidence.get("monitor")
    return monitor if isinstance(monitor, Mapping) else None


def inspect_control_execution(
    envelope_bytes: bytes,
    *,
    public_key: bytes,
    target_host: str,
    require_current: bool = True,
) -> dict[str, Any]:
    verification = verify_control_envelope(
        envelope_bytes,
        public_key=public_key,
        target_host=target_host,
        require_current=require_current,
    )
    claim_path = control_execution_claim_path(verification["packet"])
    completion_path = control_completion_path(verification["packet"])
    claim_exists = claim_path.exists() or claim_path.is_symlink()
    completion_exists = completion_path.exists() or completion_path.is_symlink()
    if completion_exists and not claim_exists:
        raise D0Error("control completion exists without its execution claim")
    base = {
        "verification": verification,
        "claim_path": str(claim_path),
        "completion_path": str(completion_path),
        "status": "pass",
    }
    if not claim_exists:
        return {**base, "state": "available"}
    claim = _validate_claim(
        _read_owner_regular(claim_path, maximum=1024 * 1024, label="control claim"),
        verification,
    )
    if not completion_exists:
        return {**base, "claim": claim, "state": "claimed-incomplete"}
    completion = load_canonical_json(
        _read_owner_regular(
            completion_path, maximum=MAX_BUNDLE_BYTES, label="control completion"
        ),
        maximum=MAX_BUNDLE_BYTES,
        label="control completion",
    )
    result = completion.get("result")
    report = result.get("host_report") if isinstance(result, Mapping) else None
    monitor = _persistence_monitor(result) if isinstance(result, Mapping) else None
    resources = completion.get("resources")
    expected_fields = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "target_host",
        "cycle_id",
        "phase",
        "operation",
        "packet_sha256",
        "envelope_sha256",
        "claim_sha256",
        "report_sha256",
        "persistence_monitor_sha256",
        "output_sha256",
        "result",
        "resources",
        "execution_status",
        "status",
        "completion_sha256",
    }
    if (
        set(completion) != expected_fields
        or completion.get("schema_id") != CONTROL_COMPLETION_SCHEMA
        or completion.get("schema_version") != 2
        or completion.get("campaign_id") != CAMPAIGN_ID
        or completion.get("run_id") != verification["packet"]["run_id"]
        or completion.get("target_host") != target_host
        or completion.get("cycle_id") != verification["packet"]["cycle_id"]
        or completion.get("phase") != verification["packet"]["phase"]
        or completion.get("operation") != verification["envelope"]["operation"]
        or completion.get("packet_sha256") != verification["packet"]["packet_sha256"]
        or completion.get("envelope_sha256") != verification["envelope_sha256"]
        or completion.get("claim_sha256") != claim["claim_sha256"]
        or not isinstance(report, Mapping)
        or not isinstance(monitor, Mapping)
        or completion.get("report_sha256") != report.get("report_sha256")
        or completion.get("persistence_monitor_sha256") != monitor.get("monitor_sha256")
        or completion.get("output_sha256") != sha256_bytes(canonical_json(result))
        or completion.get("execution_status")
        != ("failed" if report.get("status") == "fail" else "completed")
        or not isinstance(resources, Mapping)
        or resources.get("status") != "pass"
        or resources.get("before", {}).get("swap_used_bytes") != 0
        or resources.get("after", {}).get("swap_used_bytes") != 0
        or resources.get("continuous_swap", {}).get("max_used_bytes") != 0
        or completion.get("status") != "pass"
        or completion.get("completion_sha256")
        != document_sha256(completion, "completion_sha256")
    ):
        raise D0Error("control completion differs")
    validate_host_report(dict(report), packet=verification["packet"])
    return {**base, "claim": claim, "completion": completion, "state": "completed"}


def claim_control_execution(
    envelope_bytes: bytes,
    *,
    public_key: bytes,
    target_host: str,
) -> dict[str, Any]:
    inspected = inspect_control_execution(
        envelope_bytes, public_key=public_key, target_host=target_host
    )
    if inspected["state"] != "available":
        raise D0Error("control envelope was already claimed")
    claim = _claim_document(inspected["verification"])
    path = Path(inspected["claim_path"])
    ensure_owner_directory(path.parent)
    installed = atomic_install_bytes(path, canonical_json(claim))
    if installed["status"] != "installed":
        raise D0Error("control execution claim raced with another executor")
    return {"claim": claim, "installation": installed, "status": "pass"}


def complete_control_execution(
    envelope_bytes: bytes,
    result: Mapping[str, Any],
    *,
    public_key: bytes,
    target_host: str,
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    inspected = inspect_control_execution(
        envelope_bytes, public_key=public_key, target_host=target_host
    )
    if inspected["state"] == "completed":
        if inspected["completion"]["result"] != dict(result):
            raise D0Error("control completion replay result differs")
        return {
            "completion": inspected["completion"],
            "disposition": "already-completed",
            "status": "pass",
        }
    if inspected["state"] != "claimed-incomplete":
        raise D0Error("control envelope was not claimed before completion")
    report = result.get("host_report")
    monitor = _persistence_monitor(result)
    if not isinstance(report, Mapping) or not isinstance(monitor, Mapping):
        raise D0Error("control completion result lacks persisted host evidence")
    validate_host_report(dict(report), packet=inspected["verification"]["packet"])
    if (
        resources.get("status") != "pass"
        or resources.get("before", {}).get("swap_used_bytes") != 0
        or resources.get("after", {}).get("swap_used_bytes") != 0
        or resources.get("continuous_swap", {}).get("max_used_bytes") != 0
    ):
        raise D0Error("control completion resource evidence differs")
    packet = inspected["verification"]["packet"]
    completion: dict[str, Any] = {
        "schema_id": CONTROL_COMPLETION_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "target_host": target_host,
        "cycle_id": packet["cycle_id"],
        "phase": packet["phase"],
        "operation": inspected["verification"]["envelope"]["operation"],
        "packet_sha256": packet["packet_sha256"],
        "envelope_sha256": inspected["verification"]["envelope_sha256"],
        "claim_sha256": inspected["claim"]["claim_sha256"],
        "report_sha256": report["report_sha256"],
        "persistence_monitor_sha256": monitor["monitor_sha256"],
        "output_sha256": sha256_bytes(canonical_json(result)),
        "result": dict(result),
        "resources": dict(resources),
        "execution_status": "failed" if report["status"] == "fail" else "completed",
        "status": "pass",
    }
    completion["completion_sha256"] = document_sha256(
        completion, "completion_sha256"
    )
    installed = atomic_install_bytes(
        Path(inspected["completion_path"]), canonical_json(completion)
    )
    return {
        "completion": completion,
        "installation": installed,
        "disposition": "completed",
        "status": "pass",
    }


def persist_receipt_transaction(
    *,
    output_root: Path,
    packet_bytes: bytes,
    signature_bytes: bytes,
    report: Mapping[str, Any],
    resource_before: Mapping[str, Any],
    swap_monitor: Any,
    resource_snapshot_reader: Callable[[], Mapping[str, Any]],
    collection: str = "pending",
) -> dict[str, Any]:
    """Atomically persist one host-local transaction and terminal monitor journal."""

    validate_host_report(dict(report))
    packet = load_canonical_json(
        packet_bytes, maximum=1024 * 1024, label="persisted work packet"
    )
    validate_work_packet(packet)
    validate_host_report(dict(report), packet=packet)
    if safe_relative(collection, "receipt transaction collection") != "pending":
        raise D0Error("receipt transaction collection differs")
    report_bytes = canonical_json(report)
    digest = report["report_sha256"]
    destination = output_root / collection / digest
    staging = destination.with_name(f".{destination.name}.partial")
    base = {
        "work-packet.json": packet_bytes,
        "work-packet-signature.json": signature_bytes,
        "report.json": report_bytes,
    }
    receipt_bytes = render_persistence_receipt(
        packet_bytes, signature_bytes, report_bytes
    )
    expected_names = {
        *base,
        PERSISTENCE_RECEIPT_NAME,
        PERSISTENCE_EVIDENCE_NAME,
        PERSISTENCE_MONITOR_NAME,
    }
    if resource_before.get("swap_used_bytes") != 0:
        raise D0Error("receipt persistence requires zero swap before mutation")
    ensure_owner_directory(destination.parent)

    def read_transaction(path: Path) -> dict[str, bytes]:
        if path.is_symlink() or not path.is_dir():
            raise D0Error("receipt transaction target is unsafe")
        return {
            entry.name: _read_owner_regular(
                entry, maximum=MAX_BUNDLE_BYTES, label=f"receipt transaction {entry.name}"
            )
            for entry in path.iterdir()
        }

    if destination.exists() or destination.is_symlink():
        observed = read_transaction(destination)
        if set(observed) != expected_names:
            raise D0Error("existing receipt transaction file set differs")
        persisted_receipt, persisted_evidence = validate_persistence_transaction(
            observed, packet=packet, report=report
        )
        if any(observed[name] != payload for name, payload in base.items()):
            raise D0Error("existing receipt transaction bytes differ")
        after = dict(resource_snapshot_reader())
        continuous = swap_monitor.stop()
        if after.get("swap_used_bytes") != 0 or continuous.get("max_used_bytes") != 0:
            raise D0Error("receipt persistence replay used host swap")
        return {
            "destination": str(destination),
            "report_sha256": digest,
            "persistence_receipt": persisted_receipt,
            "persistence_evidence": persisted_evidence,
            "postcommit": after,
            "status": "already-installed",
        }
    if staging.exists() or staging.is_symlink():
        raise D0Error("receipt transaction has unresolved staging state")
    staging.mkdir(mode=0o700)
    os.chmod(staging, 0o700)
    monitor_finalized = False
    evidence_bytes = b""
    monitor_bytes = b""
    final_snapshot: dict[str, Any] = {}
    try:
        for name, payload in {**base, PERSISTENCE_RECEIPT_NAME: receipt_bytes}.items():
            atomic_write(staging / name, payload)
        after_payload = dict(resource_snapshot_reader())
        precommit = dict(resource_snapshot_reader())
        if (
            after_payload.get("swap_used_bytes") != 0
            or precommit.get("swap_used_bytes") != 0
        ):
            raise D0Error("receipt persistence resource evidence differs")
        evidence_bytes = render_persistence_evidence(
            receipt_bytes,
            before=resource_before,
            after_payload_fsync=after_payload,
            precommit=precommit,
        )
        atomic_write(staging / PERSISTENCE_EVIDENCE_NAME, evidence_bytes)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)

        def finalize(continuous: dict[str, Any]) -> dict[str, Any]:
            nonlocal monitor_bytes
            final = dict(resource_snapshot_reader())
            if (
                final.get("swap_used_bytes") != 0
                or continuous.get("status") != "pass"
                or continuous.get("max_used_bytes") != 0
                or continuous.get("nonzero_samples") != 0
            ):
                raise D0Error("receipt persistence terminal sample observed host swap")
            monitor_bytes = render_persistence_monitor(
                receipt_bytes,
                evidence_bytes,
                continuous_swap=continuous,
                final_snapshot=final,
            )
            atomic_write(destination / PERSISTENCE_MONITOR_NAME, monitor_bytes)
            return final

        stop_and_finalize = getattr(swap_monitor, "stop_and_finalize", None)
        if not callable(stop_and_finalize):
            raise D0Error("receipt persistence monitor lacks terminal finalization")
        try:
            _continuous, finalized = stop_and_finalize(finalize)
        finally:
            monitor_finalized = True
        final_snapshot = dict(finalized)
    except BaseException:
        if not monitor_finalized:
            with contextlib.suppress(BaseException):
                swap_monitor.stop()
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    persisted_receipt, persisted_evidence = validate_persistence_transaction(
        {
            **base,
            PERSISTENCE_RECEIPT_NAME: receipt_bytes,
            PERSISTENCE_EVIDENCE_NAME: evidence_bytes,
            PERSISTENCE_MONITOR_NAME: monitor_bytes,
        },
        packet=packet,
        report=report,
    )
    return {
        "destination": str(destination),
        "report_sha256": digest,
        "persistence_receipt": persisted_receipt,
        "persistence_evidence": persisted_evidence,
        "postcommit": final_snapshot,
        "status": "installed",
    }
