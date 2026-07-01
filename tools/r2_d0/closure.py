"""Authenticated bootstrap and direct materialization records."""

from __future__ import annotations

import re
from collections.abc import Mapping
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
    validate_bootstrap_packet,
    validate_signature_bundle,
)
from .signing import normalize_public_key, public_key_fingerprint, verify_stdin

BOOTSTRAP_RECORD_SCHEMA = "cascadia.r2-map.d0-bootstrap-record.v1"
MATERIALIZATION_RECEIPT_SCHEMA = "cascadia.r2-map.d0-materialization-receipt.v1"
BOOTSTRAP_RECEIPT_SCHEMA = "cascadia.r2-map.d0-bootstrap-receipt.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise D0Error(f"{label} differs")
    return value


def build_bootstrap_record(
    packet_bytes: bytes,
    receipt: Mapping[str, Any],
) -> bytes:
    """Convert one applied bootstrap transaction into John1-signable evidence."""

    packet = load_canonical_json(
        packet_bytes,
        maximum=1024 * 1024,
        label="bootstrap packet",
    )
    validate_bootstrap_packet(packet)
    required_receipt = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "host",
        "packet_content_sha256",
        "packet_sha256",
        "helper_archive_sha256",
        "helper_manifest_sha256",
        "helper_destination",
        "public_key_sha256",
        "public_key_fingerprint",
        "public_key_destination",
        "receipt_destination",
        "installed_unix_ms",
        "runtime_installed",
        "runtime_invoked",
        "project_code_executed",
        "protected_seed_values_opened",
        "status",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required_receipt:
        raise D0Error("bootstrap receipt fields differ")
    if (
        receipt["schema_id"] != BOOTSTRAP_RECEIPT_SCHEMA
        or receipt["schema_version"] != 1
        or receipt["campaign_id"] != CAMPAIGN_ID
        or receipt["run_id"] != D0_RUN_ID
        or receipt["host"] != packet["host"]
        or receipt["packet_content_sha256"] != sha256_bytes(packet_bytes)
        or receipt["packet_sha256"] != packet["packet_sha256"]
        or receipt["helper_archive_sha256"] != packet["helper"]["sha256"]
        or receipt["public_key_sha256"] != packet["public_key"]["openssh_sha256"]
        or receipt["public_key_fingerprint"] != packet["public_key"]["fingerprint"]
        or receipt["helper_destination"] != packet["destinations"]["helper"]
        or receipt["public_key_destination"] != packet["destinations"]["public_key"]
        or receipt["receipt_destination"] != packet["destinations"]["receipt"]
        or receipt["runtime_installed"] is not False
        or receipt["runtime_invoked"] is not False
        or receipt["project_code_executed"] is not False
        or receipt["protected_seed_values_opened"] is not False
        or receipt["status"] != "pass"
        or receipt["receipt_sha256"] != document_sha256(receipt, "receipt_sha256")
    ):
        raise D0Error("bootstrap receipt does not bind its packet and artifacts")
    record: dict[str, Any] = {
        "schema_id": BOOTSTRAP_RECORD_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "host": packet["host"],
        "bootstrap_packet_content_sha256": sha256_bytes(packet_bytes),
        "bootstrap_packet_sha256": packet["packet_sha256"],
        "bootstrap_receipt_sha256": receipt["receipt_sha256"],
        "helper_archive_sha256": receipt["helper_archive_sha256"],
        "helper_manifest_sha256": receipt["helper_manifest_sha256"],
        "public_key_sha256": receipt["public_key_sha256"],
        "public_key_fingerprint": receipt["public_key_fingerprint"],
        "installed_unix_ms": receipt["installed_unix_ms"],
        "runtime_installed": False,
        "runtime_invoked": False,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "pass",
    }
    record["record_sha256"] = document_sha256(record, "record_sha256")
    return canonical_json(record)


def validate_bootstrap_record(value: Any) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "host",
        "bootstrap_packet_content_sha256",
        "bootstrap_packet_sha256",
        "bootstrap_receipt_sha256",
        "helper_archive_sha256",
        "helper_manifest_sha256",
        "public_key_sha256",
        "public_key_fingerprint",
        "installed_unix_ms",
        "runtime_installed",
        "runtime_invoked",
        "project_code_executed",
        "protected_seed_values_opened",
        "status",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("bootstrap record fields differ")
    if (
        value["schema_id"] != BOOTSTRAP_RECORD_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["host"] not in {"john1", "john2", "john3"}
        or not isinstance(value["installed_unix_ms"], int)
        or isinstance(value["installed_unix_ms"], bool)
        or value["installed_unix_ms"] <= 0
        or any(
            value[field] is not False
            for field in (
                "runtime_installed",
                "runtime_invoked",
                "project_code_executed",
                "protected_seed_values_opened",
            )
        )
        or value["status"] != "pass"
        or value["record_sha256"] != document_sha256(value, "record_sha256")
    ):
        raise D0Error("bootstrap record identity differs")
    for field in (
        "bootstrap_packet_content_sha256",
        "bootstrap_packet_sha256",
        "bootstrap_receipt_sha256",
        "helper_archive_sha256",
        "helper_manifest_sha256",
        "public_key_sha256",
    ):
        _sha256(value[field], f"bootstrap record {field}")
    if not isinstance(value["public_key_fingerprint"], str):
        raise D0Error("bootstrap record public-key fingerprint differs")
    return value


def verify_bootstrap_record(
    record_bytes: bytes,
    signature: Mapping[str, Any],
    *,
    public_key: bytes,
) -> dict[str, Any]:
    record = validate_bootstrap_record(
        load_canonical_json(
            record_bytes,
            maximum=1024 * 1024,
            label="bootstrap record",
        )
    )
    normalized = normalize_public_key(public_key)
    validate_signature_bundle(signature, payload_sha256=sha256_bytes(record_bytes))
    verify_stdin(normalized, record_bytes, dict(signature))
    if (
        record["public_key_sha256"] != sha256_bytes(normalized)
        or record["public_key_fingerprint"] != public_key_fingerprint(normalized)
    ):
        raise D0Error("bootstrap record campaign key differs")
    return record


def build_materialization_receipt(
    *,
    source_host: str,
    target_host: str,
    operation: str,
    bundle_sha256: str,
    bundle_size: int,
    manifest_sha256: str,
    packet_sha256: str,
    report_sha256: str,
    destination_relative: str,
    transport_receipt_sha256: str,
    storage_identity_sha256: str,
    persistence_evidence_sha256: str,
    materialized_unix_ms: int,
) -> bytes:
    if (
        source_host not in {"john1", "john2", "john3"}
        or target_host not in {"john1", "john2", "john3"}
        or (
            source_host != target_host
            and "john1" not in {source_host, target_host}
        )
        or not isinstance(operation, str)
        or not operation
        or not isinstance(bundle_size, int)
        or isinstance(bundle_size, bool)
        or bundle_size <= 0
        or not isinstance(materialized_unix_ms, int)
        or isinstance(materialized_unix_ms, bool)
        or materialized_unix_ms <= 0
    ):
        raise D0Error("materialization receipt endpoint or scalar differs")
    relative = safe_relative(destination_relative, "materialization destination")
    receipt: dict[str, Any] = {
        "schema_id": MATERIALIZATION_RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "source_host": source_host,
        "target_host": target_host,
        "operation": operation,
        "bundle_sha256": _sha256(bundle_sha256, "materialized bundle SHA-256"),
        "bundle_size": bundle_size,
        "manifest_sha256": _sha256(manifest_sha256, "materialized manifest SHA-256"),
        "packet_sha256": _sha256(packet_sha256, "materialized packet SHA-256"),
        "report_sha256": _sha256(report_sha256, "materialized report SHA-256"),
        "destination_relative": relative,
        "transport_receipt_sha256": _sha256(
            transport_receipt_sha256,
            "materialization transport receipt SHA-256",
        ),
        "storage_identity_sha256": _sha256(
            storage_identity_sha256,
            "materialization storage identity SHA-256",
        ),
        "persistence_evidence_sha256": _sha256(
            persistence_evidence_sha256,
            "materialization persistence evidence SHA-256",
        ),
        "materialized_unix_ms": materialized_unix_ms,
        "disposition": "installed",
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    return canonical_json(receipt)


def validate_materialization_receipt(value: Any) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "source_host",
        "target_host",
        "operation",
        "bundle_sha256",
        "bundle_size",
        "manifest_sha256",
        "packet_sha256",
        "report_sha256",
        "destination_relative",
        "transport_receipt_sha256",
        "storage_identity_sha256",
        "persistence_evidence_sha256",
        "materialized_unix_ms",
        "disposition",
        "status",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("materialization receipt fields differ")
    if (
        value["schema_id"] != MATERIALIZATION_RECEIPT_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["source_host"] not in {"john1", "john2", "john3"}
        or value["target_host"] not in {"john1", "john2", "john3"}
        or (
            value["source_host"] != value["target_host"]
            and "john1" not in {value["source_host"], value["target_host"]}
        )
        or not isinstance(value["operation"], str)
        or not value["operation"]
        or not isinstance(value["bundle_size"], int)
        or isinstance(value["bundle_size"], bool)
        or value["bundle_size"] <= 0
        or not isinstance(value["materialized_unix_ms"], int)
        or isinstance(value["materialized_unix_ms"], bool)
        or value["materialized_unix_ms"] <= 0
        or value["disposition"] not in {"installed", "already-installed"}
        or value["status"] != "pass"
        or value["receipt_sha256"] != document_sha256(value, "receipt_sha256")
    ):
        raise D0Error("materialization receipt identity differs")
    safe_relative(value["destination_relative"], "materialization destination")
    for field in (
        "bundle_sha256",
        "manifest_sha256",
        "packet_sha256",
        "report_sha256",
        "transport_receipt_sha256",
        "storage_identity_sha256",
        "persistence_evidence_sha256",
    ):
        _sha256(value[field], f"materialization receipt {field}")
    return value

