"""Frozen D0 client contract for the standalone John2 storage worker."""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_SCHEMA = "cascadia.r2-map.remote-protocol.v2"
PROTOCOL_VERSION = 2
CAPACITY_PROOF_SCHEMA = "cascadia.r2-map.remote-capacity-proof.v2"
PUT_RESULT_SCHEMA = "cascadia.r2-map.remote-put-result.v2"
STABLE_MUTATING_OPERATIONS = (
    "put-file",
    "put-stream",
    "publish-status",
    "lock-acquire",
    "lock-renew",
    "lock-release",
    "transaction-begin",
    "transaction-put",
    "transaction-import",
    "transaction-commit",
    "transaction-abort",
    "run-command",
    "run-controller",
    "run-cleanup-commit",
    "failed-run-cleanup-commit",
)

CAPACITY_PROOF_KEYS = (
    "schema_id",
    "schema_version",
    "protocol_sha256",
    "root",
    "root_mode",
    "root_uid",
    "root_gid",
    "root_device",
    "root_inode",
    "host_identity_sha256",
    "filesystem",
    "protocol",
    "internal",
    "removable",
    "solid_state",
    "free_bytes",
    "total_bytes",
    "min_free_bytes",
    "campaign_apparent_bytes",
    "max_campaign_bytes",
    "campaign_data_apparent_bytes",
    "max_data_bytes",
    "receipt_apparent_bytes",
    "receipt_entries",
    "receipt_reservation_bytes",
    "receipt_reservation_entries",
    "data_reservation_apparent_bytes",
    "data_reservation_reserved_bytes",
    "data_reservation_entries",
    "receipt_budget_bytes",
    "max_receipt_bytes",
    "max_receipt_entries",
)

PUT_RESULT_KEYS = (
    "schema_id",
    "schema_version",
    "protocol_sha256",
    "relative",
    "sha256",
    "size",
    "mode",
    "previous_sha256",
    "projected_campaign_bytes",
    "projected_data_bytes",
    "projected_free_bytes",
    "receipt_capacity_reserved_bytes",
    "receipt_reservation_apparent_bytes",
    "data_reservation_apparent_bytes",
    "journal_bytes",
    "backup_bytes",
    "transaction_overhead_bytes",
    "storage_precommit",
    "storage_staged",
    "storage_transaction",
    "payload_size",
    "payload_sha256",
)

PROTOCOL_DOCUMENT: dict[str, Any] = {
    "schema_id": PROTOCOL_SCHEMA,
    "schema_version": PROTOCOL_VERSION,
    "command_schema": "cascadia.r2-map.remote-command.v1",
    "receipt_schema": "cascadia.r2-map.remote-receipt.v2",
    "frame_schema": "cascadia.r2-map.remote-frame.v1",
    "capacity_proof_schema": CAPACITY_PROOF_SCHEMA,
    "capacity_proof_keys": list(CAPACITY_PROOF_KEYS),
    "put_result_schema": PUT_RESULT_SCHEMA,
    "put_result_keys": list(PUT_RESULT_KEYS),
    "stable_mutating_operations": list(STABLE_MUTATING_OPERATIONS),
    "limits": {
        "max_campaign_bytes": 80 * 1024**3,
        "max_data_bytes": 78 * 1024**3,
        "receipt_budget_bytes": 2 * 1024**3,
        "max_receipt_bytes": 64 * 1024,
        "max_receipt_entries": 100_000,
    },
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        PROTOCOL_DOCUMENT,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


def protocol_info() -> dict[str, Any]:
    return {**PROTOCOL_DOCUMENT, "protocol_sha256": PROTOCOL_SHA256}
