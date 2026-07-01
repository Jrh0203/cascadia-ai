"""Signed deterministic D0 result bundles.

The work packet signature authorizes work.  A separate campaign-key signature
over the completed bundle manifest authenticates the resulting report and every
other member.  Unsigned manifests are draft inputs only and are never accepted
as lineage or aggregate evidence.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    safe_relative,
    sha256_bytes,
    validate_host_report,
    validate_signature_bundle,
    validate_work_packet,
)
from .signing import normalize_public_key, public_key_fingerprint, verify_stdin

SIGNED_BUNDLE_MANIFEST_SCHEMA = "cascadia.r2-map.d0-result-bundle-manifest.v3"
MAX_BUNDLE_BYTES = 2 * 1024**3
BUNDLE_RECORD_SIZE = 10 * 1024
MANIFEST_NAME = "bundle-manifest.json"
MANIFEST_SIGNATURE_NAME = "bundle-manifest-signature.json"
DRAFT_TRANSACTION_SCHEMA = "cascadia.r2-map.d0-draft-transaction-export.v1"
DRAFT_MANIFEST_NAME = "draft-transaction-manifest.json"
PERSISTENCE_RECEIPT_NAME = "persistence-receipt.json"
PERSISTENCE_EVIDENCE_NAME = "persistence-evidence.json"
PERSISTENCE_MONITOR_NAME = "persistence-monitor.json"
PERSISTENCE_RECEIPT_SCHEMA = "cascadia.r2-map.d0-persistence-receipt.v1"
PERSISTENCE_EVIDENCE_SCHEMA = "cascadia.r2-map.d0-persistence-evidence.v1"
PERSISTENCE_MONITOR_SCHEMA = "cascadia.r2-map.d0-persistence-monitor.v1"
RESERVED_NAMES = frozenset({MANIFEST_NAME, MANIFEST_SIGNATURE_NAME, DRAFT_MANIFEST_NAME})


def render_persistence_receipt(
    packet_bytes: bytes,
    signature_bytes: bytes,
    report_bytes: bytes,
) -> bytes:
    packet = load_canonical_json(packet_bytes, maximum=1024 * 1024, label="work packet")
    validate_work_packet(packet)
    report = load_canonical_json(report_bytes, maximum=MAX_BUNDLE_BYTES, label="host report")
    validate_host_report(report, packet=packet)
    files = {
        "report.json": report_bytes,
        "work-packet-signature.json": signature_bytes,
        "work-packet.json": packet_bytes,
    }
    receipt: dict[str, Any] = {
        "schema_id": PERSISTENCE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "phase": report["phase"],
        "operation": report["operation"],
        "packet_sha256": packet["packet_sha256"],
        "report_sha256": report["report_sha256"],
        "transaction_relative": f"pending/{report['report_sha256']}",
        "files": [_safe_bundle_file(name, files[name]) for name in sorted(files)],
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    return canonical_json(receipt)


def render_persistence_evidence(
    receipt_bytes: bytes,
    *,
    before: Mapping[str, Any],
    after_payload_fsync: Mapping[str, Any],
    precommit: Mapping[str, Any],
) -> bytes:
    receipt = load_canonical_json(
        receipt_bytes,
        maximum=1024 * 1024,
        label="persistence receipt",
    )
    if (
        receipt.get("schema_id") != PERSISTENCE_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != document_sha256(receipt, "receipt_sha256")
    ):
        raise D0Error("persistence receipt identity differs")
    evidence: dict[str, Any] = {
        "schema_id": PERSISTENCE_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": receipt["run_id"],
        "cycle_id": receipt["cycle_id"],
        "host": receipt["host"],
        "packet_sha256": receipt["packet_sha256"],
        "report_sha256": receipt["report_sha256"],
        "persistence_receipt_sha256": receipt["receipt_sha256"],
        "before": dict(before),
        "after_payload_fsync": dict(after_payload_fsync),
        "continuous_swap_journal": PERSISTENCE_MONITOR_NAME,
        "precommit": dict(precommit),
        "commit": "payload-staged-for-atomic-directory-publication",
        "status": "pass",
    }
    evidence["evidence_sha256"] = document_sha256(evidence, "evidence_sha256")
    return canonical_json(evidence)


def render_persistence_monitor(
    receipt_bytes: bytes,
    evidence_bytes: bytes,
    *,
    continuous_swap: Mapping[str, Any],
    final_snapshot: Mapping[str, Any],
) -> bytes:
    receipt = load_canonical_json(
        receipt_bytes, maximum=1024 * 1024, label="persistence receipt"
    )
    evidence = load_canonical_json(
        evidence_bytes, maximum=4 * 1024 * 1024, label="persistence evidence"
    )
    monitor: dict[str, Any] = {
        "schema_id": PERSISTENCE_MONITOR_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_sha256": receipt["packet_sha256"],
        "report_sha256": receipt["report_sha256"],
        "persistence_receipt_sha256": receipt["receipt_sha256"],
        "persistence_evidence_sha256": evidence["evidence_sha256"],
        "continuous_swap": dict(continuous_swap),
        "final_snapshot": dict(final_snapshot),
        "continuous_coverage": "payload-writes-directory-rename-parent-fsync",
        "terminal_journal": PERSISTENCE_MONITOR_NAME,
        "terminal_journal_sampled": False,
        "status": "committed",
    }
    monitor["monitor_sha256"] = document_sha256(monitor, "monitor_sha256")
    return canonical_json(monitor)


def validate_persistence_transaction(
    files: Mapping[str, bytes],
    *,
    packet: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_canonical_json(
        files.get(PERSISTENCE_RECEIPT_NAME, b""),
        maximum=1024 * 1024,
        label="persistence receipt",
    )
    evidence = load_canonical_json(
        files.get(PERSISTENCE_EVIDENCE_NAME, b""),
        maximum=4 * 1024 * 1024,
        label="persistence evidence",
    )
    monitor = load_canonical_json(
        files.get(PERSISTENCE_MONITOR_NAME, b""),
        maximum=4 * 1024 * 1024,
        label="persistence monitor",
    )
    expected_receipt_fields = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "cycle_id",
        "host",
        "phase",
        "operation",
        "packet_sha256",
        "report_sha256",
        "transaction_relative",
        "files",
        "status",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt["schema_id"] != PERSISTENCE_RECEIPT_SCHEMA
        or receipt["schema_version"] != 1
        or receipt["campaign_id"] != CAMPAIGN_ID
        or receipt["run_id"] != packet["run_id"]
        or receipt["cycle_id"] != packet["cycle_id"]
        or receipt["host"] != packet["host"]
        or receipt["phase"] != report["phase"]
        or receipt["operation"] != report["operation"]
        or receipt["packet_sha256"] != packet["packet_sha256"]
        or receipt["report_sha256"] != report["report_sha256"]
        or receipt["transaction_relative"] != f"pending/{report['report_sha256']}"
        or receipt["status"] != "pass"
        or receipt["receipt_sha256"] != document_sha256(receipt, "receipt_sha256")
    ):
        raise D0Error("persistence receipt binding differs")
    transaction_files = {
        "report.json": files.get("report.json", b""),
        "work-packet-signature.json": files.get("work-packet-signature.json", b""),
        "work-packet.json": files.get("work-packet.json", b""),
    }
    expected_identities = [
        _safe_bundle_file(name, transaction_files[name]) for name in sorted(transaction_files)
    ]
    if receipt["files"] != expected_identities:
        raise D0Error("persistence receipt file identities differ")
    expected_evidence_fields = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "cycle_id",
        "host",
        "packet_sha256",
        "report_sha256",
        "persistence_receipt_sha256",
        "before",
        "after_payload_fsync",
        "continuous_swap_journal",
        "precommit",
        "commit",
        "status",
        "evidence_sha256",
    }
    if (
        set(evidence) != expected_evidence_fields
        or evidence["schema_id"] != PERSISTENCE_EVIDENCE_SCHEMA
        or evidence["schema_version"] != 1
        or evidence["campaign_id"] != CAMPAIGN_ID
        or evidence["run_id"] != packet["run_id"]
        or evidence["cycle_id"] != packet["cycle_id"]
        or evidence["host"] != packet["host"]
        or evidence["packet_sha256"] != packet["packet_sha256"]
        or evidence["report_sha256"] != report["report_sha256"]
        or evidence["persistence_receipt_sha256"] != receipt["receipt_sha256"]
        or evidence["continuous_swap_journal"] != PERSISTENCE_MONITOR_NAME
        or evidence["commit"] != "payload-staged-for-atomic-directory-publication"
        or evidence["status"] != "pass"
        or evidence["evidence_sha256"] != document_sha256(evidence, "evidence_sha256")
        or any(
            not isinstance(evidence.get(boundary), Mapping)
            or evidence[boundary].get("swap_used_bytes") != 0
            for boundary in ("before", "after_payload_fsync", "precommit")
        )
    ):
        raise D0Error("persistence resource evidence differs")
    expected_monitor_fields = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "packet_sha256",
        "report_sha256",
        "persistence_receipt_sha256",
        "persistence_evidence_sha256",
        "continuous_swap",
        "final_snapshot",
        "continuous_coverage",
        "terminal_journal",
        "terminal_journal_sampled",
        "status",
        "monitor_sha256",
    }
    continuous = monitor.get("continuous_swap")
    if (
        set(monitor) != expected_monitor_fields
        or monitor.get("schema_id") != PERSISTENCE_MONITOR_SCHEMA
        or monitor.get("schema_version") != 1
        or monitor.get("campaign_id") != CAMPAIGN_ID
        or monitor.get("packet_sha256") != packet["packet_sha256"]
        or monitor.get("report_sha256") != report["report_sha256"]
        or monitor.get("persistence_receipt_sha256") != receipt["receipt_sha256"]
        or monitor.get("persistence_evidence_sha256") != evidence["evidence_sha256"]
        or monitor.get("continuous_coverage")
        != "payload-writes-directory-rename-parent-fsync"
        or monitor.get("terminal_journal") != PERSISTENCE_MONITOR_NAME
        or monitor.get("terminal_journal_sampled") is not False
        or monitor.get("status") != "committed"
        or monitor.get("monitor_sha256") != document_sha256(monitor, "monitor_sha256")
        or not isinstance(monitor.get("final_snapshot"), Mapping)
        or monitor["final_snapshot"].get("swap_used_bytes") != 0
        or not isinstance(continuous, Mapping)
        or continuous.get("status") != "pass"
        or continuous.get("sample_count", 0) < 1
        or continuous.get("nonzero_samples") != 0
        or continuous.get("max_used_bytes") != 0
    ):
        raise D0Error("persistence monitor journal differs")
    return receipt, {**evidence, "monitor": monitor}


def _safe_bundle_file(path: str, payload: bytes) -> dict[str, Any]:
    safe_relative(path, "bundle member")
    if path in RESERVED_NAMES:
        raise D0Error("bundle payload uses a reserved name")
    if not isinstance(payload, bytes) or len(payload) > MAX_BUNDLE_BYTES:
        raise D0Error("bundle member exceeds its byte limit")
    return {
        "path": path,
        "size": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": "0444",
    }


def _bundle_from_members(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            item.mode = 0o444
            item.uid = 0
            item.gid = 0
            item.mtime = 0
            item.uname = ""
            item.gname = ""
            archive.addfile(item, io.BytesIO(payload))
    value = output.getvalue()
    if len(value) > MAX_BUNDLE_BYTES or len(value) % BUNDLE_RECORD_SIZE:
        raise D0Error("result bundle size or record padding differs")
    return value


def _validated_transaction(
    files: Mapping[str, bytes],
    *,
    run_id: str,
    cycle_id: str,
    host: str,
    role: str,
    packet_sha256: str,
    created_unix_ms: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    required_files = {
        "work-packet.json",
        "work-packet-signature.json",
        "report.json",
        PERSISTENCE_RECEIPT_NAME,
        PERSISTENCE_EVIDENCE_NAME,
        PERSISTENCE_MONITOR_NAME,
    }
    if not required_files.issubset(files) or set(files) & RESERVED_NAMES:
        raise D0Error("result bundle lacks its transaction files or uses a reserved name")
    ordered = sorted(files)
    if ordered != sorted(set(ordered)):
        raise D0Error("result bundle member names are duplicated")
    identities = [_safe_bundle_file(path, files[path]) for path in ordered]
    packet = load_canonical_json(
        files["work-packet.json"],
        maximum=1024 * 1024,
        label="work packet",
    )
    validate_work_packet(packet)
    report = load_canonical_json(
        files["report.json"],
        maximum=MAX_BUNDLE_BYTES,
        label="host report",
    )
    validate_host_report(report, packet=packet)
    persistence_receipt, persistence_evidence = validate_persistence_transaction(
        files,
        packet=packet,
        report=report,
    )
    if (
        not isinstance(created_unix_ms, int)
        or isinstance(created_unix_ms, bool)
        or created_unix_ms <= 0
        or created_unix_ms != report["finished_unix_ms"]
        or run_id != packet["run_id"]
        or cycle_id != packet["cycle_id"]
        or host != packet["host"]
        or role != packet["role"]
        or packet_sha256 != packet["packet_sha256"]
        or len(files["report.json"]) > packet["limits"]["output_max_bytes"]
    ):
        raise D0Error("result bundle transaction bindings or signed limit differ")
    return identities, packet, report, persistence_receipt, persistence_evidence


def render_result_bundle_manifest(
    files: Mapping[str, bytes],
    *,
    run_id: str,
    cycle_id: str,
    host: str,
    role: str,
    packet_sha256: str,
    created_unix_ms: int,
) -> tuple[bytes, dict[str, Any]]:
    """Render the exact manifest bytes John1 must sign."""

    identities, packet, report, persistence_receipt, persistence_evidence = _validated_transaction(
        files,
        run_id=run_id,
        cycle_id=cycle_id,
        host=host,
        role=role,
        packet_sha256=packet_sha256,
        created_unix_ms=created_unix_ms,
    )
    manifest: dict[str, Any] = {
        "schema_id": SIGNED_BUNDLE_MANIFEST_SCHEMA,
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "run_id": run_id,
        "cycle_id": cycle_id,
        "host": host,
        "role": role,
        "packet_sha256": packet_sha256,
        "report_sha256": report["report_sha256"],
        "created_unix_ms": created_unix_ms,
        "files": identities,
        "protected_seed_values_opened": False,
        "project_code_executed": False,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    return canonical_json(manifest), {
        "manifest": manifest,
        "packet": packet,
        "report": report,
        "persistence_receipt": persistence_receipt,
        "persistence_evidence": persistence_evidence,
    }


def seal_result_bundle(
    files: Mapping[str, bytes],
    *,
    manifest_bytes: bytes,
    manifest_signature_bytes: bytes,
    public_key: bytes,
    run_id: str,
    cycle_id: str,
    host: str,
    role: str,
    packet_sha256: str,
    created_unix_ms: int,
) -> tuple[bytes, dict[str, Any]]:
    """Verify John1's manifest signature and assemble the canonical sealed archive."""

    expected_manifest, context = render_result_bundle_manifest(
        files,
        run_id=run_id,
        cycle_id=cycle_id,
        host=host,
        role=role,
        packet_sha256=packet_sha256,
        created_unix_ms=created_unix_ms,
    )
    if manifest_bytes != expected_manifest:
        raise D0Error("signed result-bundle manifest differs from the transaction")
    signature = load_canonical_json(
        manifest_signature_bytes,
        maximum=1024 * 1024,
        label="result bundle manifest signature",
    )
    validate_signature_bundle(signature, payload_sha256=sha256_bytes(manifest_bytes))
    normalized_key = normalize_public_key(public_key)
    verify_stdin(normalized_key, manifest_bytes, signature)
    members = {
        **files,
        MANIFEST_NAME: manifest_bytes,
        MANIFEST_SIGNATURE_NAME: manifest_signature_bytes,
    }
    archive = _bundle_from_members(members)
    if len(archive) > context["packet"]["limits"]["output_max_bytes"]:
        raise D0Error("sealed result bundle exceeds the signed phase output limit")
    return archive, {
        **context,
        "manifest_signature": signature,
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "sealed": True,
        "status": "pass",
    }


def _read_bundle_members(value: bytes) -> dict[str, bytes]:
    if not value or len(value) > MAX_BUNDLE_BYTES:
        raise D0Error("result bundle exceeds its byte limit")
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(value), mode="r:") as archive:
            for item in archive:
                if (
                    not item.isfile()
                    or item.name in members
                    or item.name.startswith("/")
                    or ".." in PurePosixPath(item.name).parts
                    or item.mode != 0o444
                    or item.uid != 0
                    or item.gid != 0
                    or item.mtime != 0
                ):
                    raise D0Error("result bundle has an unsafe member")
                safe_relative(item.name, "result bundle member")
                stream = archive.extractfile(item)
                if stream is None:
                    raise D0Error("result bundle member is unreadable")
                payload = stream.read(MAX_BUNDLE_BYTES + 1)
                if len(payload) != item.size:
                    raise D0Error("result bundle member size differs")
                members[item.name] = payload
    except (OSError, tarfile.TarError) as error:
        raise D0Error("result bundle is not a valid USTAR archive") from error
    return members


def read_result_bundle_members(value: bytes) -> dict[str, bytes]:
    """Return a defensive copy of the strictly parsed archive members."""

    return dict(_read_bundle_members(value))


def render_draft_transaction_export(
    files: Mapping[str, bytes],
    *,
    public_key: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Export an authenticated transaction draft that is never aggregate-eligible.

    The packet signature is verified here. The report is packet-bound and
    self-hashed, but the draft manifest is deliberately unsigned: John1 must
    review and sign the v3 bundle manifest before the transaction can enter a
    canonical result namespace or aggregate.
    """

    if set(files) != {
        "work-packet.json",
        "work-packet-signature.json",
        "report.json",
        PERSISTENCE_RECEIPT_NAME,
        PERSISTENCE_EVIDENCE_NAME,
        PERSISTENCE_MONITOR_NAME,
    }:
        raise D0Error("draft transaction files differ")
    packet_bytes = files["work-packet.json"]
    packet = load_canonical_json(packet_bytes, maximum=1024 * 1024, label="draft work packet")
    validate_work_packet(packet)
    signature = load_canonical_json(
        files["work-packet-signature.json"],
        maximum=1024 * 1024,
        label="draft work packet signature",
    )
    validate_signature_bundle(signature, payload_sha256=sha256_bytes(packet_bytes))
    normalized_key = normalize_public_key(public_key)
    verify_stdin(normalized_key, packet_bytes, signature)
    report = load_canonical_json(
        files["report.json"], maximum=MAX_BUNDLE_BYTES, label="draft host report"
    )
    validate_host_report(report, packet=packet)
    persistence_receipt, persistence_evidence = validate_persistence_transaction(
        files,
        packet=packet,
        report=report,
    )
    identities = [_safe_bundle_file(path, files[path]) for path in sorted(files)]
    manifest: dict[str, Any] = {
        "schema_id": DRAFT_TRANSACTION_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "packet_sha256": packet["packet_sha256"],
        "report_sha256": report["report_sha256"],
        "created_unix_ms": report["finished_unix_ms"],
        "files": identities,
        "sealed": False,
        "canonical_eligible": False,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    manifest_bytes = canonical_json(manifest)
    archive = _bundle_from_members({**files, DRAFT_MANIFEST_NAME: manifest_bytes})
    return archive, {
        "manifest": manifest,
        "packet": packet,
        "report": report,
        "persistence_receipt": persistence_receipt,
        "persistence_evidence": persistence_evidence,
        "files": dict(files),
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "sealed": False,
        "canonical_eligible": False,
        "status": "pass",
    }


def verify_draft_transaction_export(value: bytes, *, public_key: bytes) -> dict[str, Any]:
    """Verify a transport-only draft without promoting it to a sealed bundle."""

    members = _read_bundle_members(value)
    manifest_bytes = members.pop(DRAFT_MANIFEST_NAME, None)
    if manifest_bytes is None or MANIFEST_NAME in members or MANIFEST_SIGNATURE_NAME in members:
        raise D0Error("draft transaction manifest or namespace differs")
    manifest = load_canonical_json(
        manifest_bytes,
        maximum=4 * 1024 * 1024,
        label="draft transaction manifest",
    )
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "cycle_id",
        "host",
        "role",
        "packet_sha256",
        "report_sha256",
        "created_unix_ms",
        "files",
        "sealed",
        "canonical_eligible",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required
        or manifest["schema_id"] != DRAFT_TRANSACTION_SCHEMA
        or manifest["schema_version"] != 1
        or manifest["campaign_id"] != CAMPAIGN_ID
        or manifest["run_id"] != D0_RUN_ID
        or manifest["cycle_id"] not in {"qualification", "final-live"}
        or (manifest["host"], manifest["role"])
        not in {("john1", "worker"), ("john2", "builder-worker"), ("john3", "worker")}
        or manifest["sealed"] is not False
        or manifest["canonical_eligible"] is not False
        or manifest["manifest_sha256"] != document_sha256(manifest, "manifest_sha256")
    ):
        raise D0Error("draft transaction manifest identity differs")
    identities = manifest["files"]
    if not isinstance(identities, list):
        raise D0Error("draft transaction file identities are absent")
    expected: dict[str, bytes] = {}
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {"path", "size", "sha256", "mode"}:
            raise D0Error("draft transaction file identity differs")
        path = safe_relative(identity["path"], "draft transaction file")
        payload = members.get(path)
        if (
            payload is None
            or identity["mode"] != "0444"
            or identity["size"] != len(payload)
            or identity["sha256"] != sha256_bytes(payload)
        ):
            raise D0Error("draft transaction payload identity differs")
        expected[path] = payload
    if [item["path"] for item in identities] != sorted(expected) or set(expected) != set(members):
        raise D0Error("draft transaction files are not exactly manifested")
    if _bundle_from_members({**expected, DRAFT_MANIFEST_NAME: manifest_bytes}) != value:
        raise D0Error("draft transaction is not canonical deterministic USTAR")
    _rebuilt, context = render_draft_transaction_export(expected, public_key=public_key)
    if context["manifest"] != manifest:
        raise D0Error("draft transaction packet/report bindings differ")
    return context


def _validate_manifest(value: Any) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "cycle_id",
        "host",
        "role",
        "packet_sha256",
        "report_sha256",
        "created_unix_ms",
        "files",
        "protected_seed_values_opened",
        "project_code_executed",
        "manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("result bundle manifest fields differ")
    if (
        value["schema_id"] != SIGNED_BUNDLE_MANIFEST_SCHEMA
        or value["schema_version"] != 3
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["cycle_id"] not in {"qualification", "final-live"}
        or (value["host"], value["role"])
        not in {
            ("john1", "worker"),
            ("john2", "builder-worker"),
            ("john3", "worker"),
        }
        or not isinstance(value["created_unix_ms"], int)
        or isinstance(value["created_unix_ms"], bool)
        or value["created_unix_ms"] <= 0
        or value["protected_seed_values_opened"] is not False
        or value["project_code_executed"] is not False
        or value["manifest_sha256"] != document_sha256(value, "manifest_sha256")
    ):
        raise D0Error("result bundle manifest identity differs")
    return value


def verify_result_bundle(value: bytes, *, public_key: bytes) -> dict[str, Any]:
    """Verify both campaign signatures and every deterministic archive binding."""

    members = _read_bundle_members(value)
    manifest_bytes = members.pop(MANIFEST_NAME, None)
    signature_bytes = members.pop(MANIFEST_SIGNATURE_NAME, None)
    if manifest_bytes is None or signature_bytes is None:
        raise D0Error("sealed result bundle manifest or signature is absent")
    manifest = _validate_manifest(
        load_canonical_json(
            manifest_bytes,
            maximum=4 * 1024 * 1024,
            label="result bundle manifest",
        )
    )
    manifest_signature = load_canonical_json(
        signature_bytes,
        maximum=1024 * 1024,
        label="result bundle manifest signature",
    )
    validate_signature_bundle(
        manifest_signature,
        payload_sha256=sha256_bytes(manifest_bytes),
    )
    normalized_key = normalize_public_key(public_key)
    verify_stdin(normalized_key, manifest_bytes, manifest_signature)
    identities = manifest["files"]
    if not isinstance(identities, list):
        raise D0Error("result bundle file identities are absent")
    expected: dict[str, bytes] = {}
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "size",
            "sha256",
            "mode",
        }:
            raise D0Error("result bundle file identity differs")
        path = safe_relative(identity["path"], "result bundle file")
        payload = members.get(path)
        if (
            payload is None
            or identity["mode"] != "0444"
            or identity["size"] != len(payload)
            or identity["sha256"] != sha256_bytes(payload)
        ):
            raise D0Error("result bundle payload identity differs")
        expected[path] = payload
    if [item["path"] for item in identities] != sorted(expected) or set(expected) != set(members):
        raise D0Error("result bundle files are not exactly sorted and manifested")
    canonical_archive = _bundle_from_members(
        {
            **expected,
            MANIFEST_NAME: manifest_bytes,
            MANIFEST_SIGNATURE_NAME: signature_bytes,
        }
    )
    if canonical_archive != value:
        raise D0Error("result bundle is not the canonical deterministic USTAR encoding")
    packet_bytes = expected.get("work-packet.json", b"")
    packet = load_canonical_json(packet_bytes, maximum=1024 * 1024, label="work packet")
    validate_work_packet(packet)
    if packet["public_key_fingerprint"] != public_key_fingerprint(normalized_key):
        raise D0Error("result bundle work packet uses a different campaign key")
    packet_signature = load_canonical_json(
        expected.get("work-packet-signature.json", b""),
        maximum=1024 * 1024,
        label="work packet signature",
    )
    validate_signature_bundle(packet_signature, payload_sha256=sha256_bytes(packet_bytes))
    verify_stdin(normalized_key, packet_bytes, packet_signature)
    report = load_canonical_json(
        expected.get("report.json", b""),
        maximum=MAX_BUNDLE_BYTES,
        label="host report",
    )
    validate_host_report(report, packet=packet)
    persistence_receipt, persistence_evidence = validate_persistence_transaction(
        expected,
        packet=packet,
        report=report,
    )
    if (
        packet["packet_sha256"] != manifest["packet_sha256"]
        or report["report_sha256"] != manifest["report_sha256"]
        or packet["host"] != manifest["host"]
        or packet["role"] != manifest["role"]
        or packet["run_id"] != manifest["run_id"]
        or packet["cycle_id"] != manifest["cycle_id"]
        or manifest["created_unix_ms"] != report["finished_unix_ms"]
        or len(expected["report.json"]) > packet["limits"]["output_max_bytes"]
        or len(value) > packet["limits"]["output_max_bytes"]
    ):
        raise D0Error("result bundle packet/report/manifest bindings differ")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "packet": packet,
        "report": report,
        "persistence_receipt": persistence_receipt,
        "persistence_evidence": persistence_evidence,
        "archive_size": len(value),
        "archive_sha256": sha256_bytes(value),
        "member_count": len(expected) + 2,
        "report_sha256": report["report_sha256"],
        "sealed": True,
        "status": "pass",
    }
