"""Receipt-bound immutable identity checks for John1 R2-MAP consumers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import blake3

from cascadia_mlx import r2_map_campaign_controller as controller
from cascadia_mlx import r2_map_remote_worker as worker
from cascadia_mlx.r2_map_local_write_guard import (
    john1_attestation_publication_receipt_relative,
)
from cascadia_mlx.r2_map_remote_storage import (
    REMOTE_HOST_ALIAS,
    REMOTE_IDENTITY_SHA256,
    REMOTE_ROOT,
    RemoteStorageClient,
    canonical_json,
    content_sha256,
    document_sha256,
)
from cascadia_mlx.r2_map_remote_training import (
    RemoteObjectEvidence,
    read_remote_object,
)

MAX_IDENTITY_JSON_BYTES = 64 << 20
SOURCE_MANIFEST_SCHEMA = "cascadia.r2-map.w0-w5-source-manifest.v1"
SOURCE_ARCHIVE_VERIFICATION_SCHEMA = (
    "cascadia.r2-map.source-archive-verification.v1"
)
REFERENCE_MANIFEST_SCHEMA = "cascadia.r2-map.reference-panel-manifest.v1.1"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
MAXIMUM_WIDTH_PANEL_ID = "maximum-width-service"
MAXIMUM_WIDTH_CANDIDATES = 6_372
BOOTSTRAP_GAMES = 100_000
BOOTSTRAP_PHASE_BARRIER_SCHEMA = "cascadia.r2-map.bootstrap-phase-barrier.v1"
BOOTSTRAP_GENERATION_MANIFEST_SCHEMA = "cascadia.r2-map.bootstrap-generation-manifest.v1"
BOOTSTRAP_AGGREGATE_OPERATION = "bootstrap-generation-aggregate"
SOURCE_ARCHIVE_VERIFIER_RELATIVE = "tools/r2_map_source_archive.py"
SOURCE_GATE_ALIASES = {
    "target.mk": "tools/r2_map_rust_w4_target_gate.mk",
    "p1.mk": "tools/r2_map_rust_p1_gate.mk",
    "release.mk": "tools/r2_map_rust_release_gate.mk",
    "python.mk": "tools/r2_map_python_boundary_gate.mk",
    "compile.mk": "tools/r2_map_rust_compile_gate.mk",
    "fixture.mk": "tools/r2_map_python_fixture_gate.mk",
}
SOURCE_TRANSACTION_CONTROL_OBJECTS = frozenset(
    {
        "source-manifest.json",
        "source.tar",
        "source-archive-verification.json",
        "archive-verify.py",
        *SOURCE_GATE_ALIASES,
    }
)
BOOTSTRAP_PHASE_RECEIPT_CONTRACT = (
    ("bootstrap-generate-john1", "generate", "john1"),
    ("bootstrap-generate-john2", "generate", "john2"),
    ("bootstrap-generate-john3", "generate", "john3"),
    (BOOTSTRAP_AGGREGATE_OPERATION, "aggregate", "john1"),
)


class R2MapRemoteIdentityError(ValueError):
    """One immutable John2 object or receipt failed identity verification."""


def _sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _remote_relative(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _source_path_prefix_collision(values: set[str]) -> bool:
    for value in values:
        parts = PurePosixPath(value).parts
        if any(
            PurePosixPath(*parts[:index]).as_posix() in values
            for index in range(1, len(parts))
        ):
            return True
    return False


@dataclass(frozen=True)
class VerifiedRemoteJson:
    value: dict[str, Any]
    payload_sha256: str
    payload_blake3: str
    evidence: RemoteObjectEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_sha256": self.payload_sha256,
            "payload_blake3": self.payload_blake3,
            "evidence": self.evidence.to_dict(),
        }


def load_verified_remote_json(
    client: RemoteStorageClient,
    relative: str,
    *,
    maximum_bytes: int = MAX_IDENTITY_JSON_BYTES,
) -> VerifiedRemoteJson:
    loaded = read_remote_object(client, relative, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(loaded.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2MapRemoteIdentityError(f"remote JSON is invalid: {relative}") from error
    if not isinstance(value, dict):
        raise R2MapRemoteIdentityError(f"remote JSON is not an object: {relative}")
    payload = bytes(loaded.payload)
    return VerifiedRemoteJson(
        value=value,
        payload_sha256=content_sha256(payload),
        payload_blake3=blake3.blake3(payload).hexdigest(),
        evidence=loaded.evidence,
    )


def validate_transaction_manifest(document: VerifiedRemoteJson) -> dict[str, Any]:
    value = document.value
    if (
        set(value)
        != {
            "schema_version",
            "schema_id",
            "transaction_id",
            "target_relative",
            "objects",
            "manifest_sha256",
        }
        or value.get("schema_version") != 1
        or value.get("schema_id") != worker.TRANSACTION_SCHEMA
        or value.get("manifest_sha256") != document_sha256(value, "manifest_sha256")
        or document.payload_sha256 != content_sha256(canonical_json(value))
    ):
        raise R2MapRemoteIdentityError("transaction manifest identity is invalid")
    target = value.get("target_relative")
    objects = value.get("objects")
    if (
        not isinstance(target, str)
        or not _remote_relative(target)
        or not isinstance(value.get("transaction_id"), str)
        or not value["transaction_id"]
        or "/" in value["transaction_id"]
        or not isinstance(objects, list)
        or not objects
        or objects != sorted(objects, key=lambda item: item.get("relative", ""))
    ):
        raise R2MapRemoteIdentityError("transaction manifest paths are invalid")
    seen: set[str] = set()
    for item in objects:
        if (
            not isinstance(item, dict)
            or set(item)
            not in (
                {"relative", "sha256", "size"},
                {"relative", "sha256", "size", "mode"},
            )
            or not _remote_relative(item.get("relative"))
            or item["relative"] in seen
            or item["relative"] == ".r2-map-transaction.json"
            or not _sha256(item.get("sha256"))
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or item["size"] < 0
            or item.get("mode", "0400") not in {"0400", "0500"}
        ):
            raise R2MapRemoteIdentityError("transaction object descriptor is invalid")
        seen.add(item["relative"])
    return value


def transaction_object_descriptor(
    transaction: Mapping[str, Any], object_relative: str
) -> dict[str, Any]:
    target = str(transaction["target_relative"])
    prefix = f"{target}/"
    if not object_relative.startswith(prefix):
        raise R2MapRemoteIdentityError("object is outside its transaction target")
    within = object_relative[len(prefix) :]
    matches = [item for item in transaction["objects"] if item.get("relative") == within]
    if len(matches) != 1:
        raise R2MapRemoteIdentityError("object is not uniquely bound by the transaction")
    return dict(matches[0])


def require_transaction_object(
    transaction: Mapping[str, Any], document: VerifiedRemoteJson
) -> dict[str, Any]:
    descriptor = transaction_object_descriptor(transaction, document.evidence.relative)
    token = document.evidence.object_token
    expected_mode = 0o500 if descriptor.get("mode") == "0500" else 0o400
    if (
        descriptor.get("sha256") != document.payload_sha256
        or descriptor.get("size") != token.get("size")
        or token.get("sha256") != document.payload_sha256
        or token.get("mode") != expected_mode
    ):
        raise R2MapRemoteIdentityError("transaction-bound object token differs")
    return descriptor


def require_open_transaction_object(
    client: RemoteStorageClient,
    transaction: Mapping[str, Any],
    object_relative: str,
) -> dict[str, Any]:
    """Reopen one committed object and bind its live token to the transaction.

    Transaction descriptors and commit receipts prove what John2 accepted.  A
    fresh object token additionally proves that the object currently at that
    path still has those exact bytes, size, and immutable mode.  Keeping this
    check separate from full content reads lets the barrier audit cover every
    shard without materializing the 100,000-game corpus on John1.
    """

    descriptor = transaction_object_descriptor(transaction, object_relative)
    opened = client.open_object_with_receipt(object_relative)
    if not isinstance(opened, dict) or set(opened) != {
        "object_token",
        "storage_receipt_relative",
        "storage_receipt_sha256",
    }:
        raise R2MapRemoteIdentityError("transaction object open evidence differs")
    token = opened.get("object_token")
    token_fields = {
        "schema_version",
        "schema_id",
        "relative",
        "sha256",
        "size",
        "device",
        "inode",
        "mtime_ns",
        "ctime_ns",
        "mode",
        "token_sha256",
    }
    expected_mode = 0o500 if descriptor.get("mode") == "0500" else 0o400
    if (
        not isinstance(token, dict)
        or set(token) != token_fields
        or token.get("schema_version") != 1
        or token.get("schema_id") != worker.OBJECT_TOKEN_SCHEMA
        or token.get("relative") != object_relative
        or token.get("sha256") != descriptor.get("sha256")
        or token.get("size") != descriptor.get("size")
        or token.get("mode") != expected_mode
        or any(
            not isinstance(token.get(name), int)
            or isinstance(token.get(name), bool)
            or token[name] < 0
            for name in ("size", "device", "inode", "mtime_ns", "ctime_ns", "mode")
        )
        or token.get("token_sha256") != document_sha256(token, "token_sha256")
        or not _remote_relative(opened.get("storage_receipt_relative"))
        or PurePosixPath(str(opened["storage_receipt_relative"])).parts[:2]
        != ("control", "receipts")
        or not _sha256(opened.get("storage_receipt_sha256"))
    ):
        raise R2MapRemoteIdentityError("live transaction object differs from commit")
    return dict(opened)


def validate_worker_receipt(document: VerifiedRemoteJson, operation: str) -> dict[str, Any]:
    value = document.value
    expected_keys = {
        "schema_version",
        "schema_id",
        "request_id",
        "command_sha256",
        "operation",
        "status",
        "host",
        "host_identity_sha256",
        "root",
        "completed_unix_ms",
        "result",
        "receipt_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("schema_id") != worker.RECEIPT_SCHEMA
        or value.get("status") != "ok"
        or value.get("operation") != operation
        or value.get("host") != REMOTE_HOST_ALIAS
        or value.get("host_identity_sha256") != REMOTE_IDENTITY_SHA256
        or value.get("root") != str(REMOTE_ROOT)
        or not _sha256(value.get("command_sha256"))
        or not isinstance(value.get("completed_unix_ms"), int)
        or isinstance(value.get("completed_unix_ms"), bool)
        or value["completed_unix_ms"] < 0
        or not isinstance(value.get("result"), dict)
        or value.get("receipt_sha256") != document_sha256(value, "receipt_sha256")
        or document.payload_sha256 != content_sha256(canonical_json(value))
    ):
        raise R2MapRemoteIdentityError("persisted worker receipt identity is invalid")
    expected_relative = f"control/receipts/{value.get('request_id')}.json"
    if document.evidence.relative != expected_relative:
        raise R2MapRemoteIdentityError("persisted worker receipt path differs")
    return value


def validate_transaction_commit(
    document: VerifiedRemoteJson,
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = validate_worker_receipt(document, "transaction-commit")
    result = receipt.get("result")
    if (
        not isinstance(result, dict)
        or set(result)
        != {
            "transaction_id",
            "target_relative",
            "manifest_sha256",
            "object_count",
            "committed",
            "payload_size",
            "payload_sha256",
        }
        or result.get("committed") is not True
        or result.get("transaction_id") != transaction.get("transaction_id")
        or result.get("target_relative") != transaction.get("target_relative")
        or result.get("manifest_sha256") != transaction.get("manifest_sha256")
        or result.get("object_count") != len(transaction.get("objects", ()))
        or result.get("payload_size") != 0
        or result.get("payload_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise R2MapRemoteIdentityError("transaction commit receipt differs from manifest")
    return receipt


def validate_immutable_publication_receipt(
    document: VerifiedRemoteJson,
    *,
    object_relative: str,
    object_sha256: str,
    object_size: int,
) -> dict[str, Any]:
    receipt = validate_worker_receipt(document, "put-file")
    result = receipt.get("result")
    if (
        not isinstance(result, dict)
        or set(result)
        != {
            "relative",
            "sha256",
            "size",
            "mode",
            "previous_sha256",
            "payload_size",
            "payload_sha256",
        }
        or result.get("relative") != object_relative
        or result.get("sha256") != object_sha256
        or result.get("size") != object_size
        or result.get("mode") != "0o400"
        or result.get("previous_sha256") is not None
        or result.get("payload_size") != 0
        or result.get("payload_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise R2MapRemoteIdentityError("immutable publication receipt differs from object")
    return receipt


def _verified_remote_json_token_matches(
    document: VerifiedRemoteJson,
    *,
    expected_relative: str,
    expected_mode: int,
) -> bool:
    token = document.evidence.object_token
    fields = {
        "schema_version",
        "schema_id",
        "relative",
        "sha256",
        "size",
        "device",
        "inode",
        "mtime_ns",
        "ctime_ns",
        "mode",
        "token_sha256",
    }
    return bool(
        document.evidence.relative == expected_relative
        and isinstance(token, dict)
        and set(token) == fields
        and token.get("schema_version") == 1
        and token.get("schema_id") == worker.OBJECT_TOKEN_SCHEMA
        and token.get("relative") == expected_relative
        and token.get("sha256") == document.payload_sha256
        and all(
            isinstance(token.get(name), int)
            and not isinstance(token.get(name), bool)
            and token[name] >= 0
            for name in ("size", "device", "inode", "mtime_ns", "ctime_ns", "mode")
        )
        and token["size"] > 0
        and token.get("mode") == expected_mode
        and token.get("token_sha256") == document_sha256(token, "token_sha256")
    )


def validate_john1_attestation_publication_receipt(
    *,
    attestation_document: VerifiedRemoteJson,
    publication_document: VerifiedRemoteJson,
) -> dict[str, str]:
    """Bind a John1 zero-write attestation to its deterministic direct put."""

    attestation = attestation_document.value
    attestation_sha256 = attestation.get("attestation_sha256")
    try:
        expected_relative = john1_attestation_publication_receipt_relative(
            attestation_sha256
        )
    except ValueError as error:
        raise R2MapRemoteIdentityError("John1 attestation digest is invalid") from error
    if (
        attestation.get("schema_version") != 1
        or attestation.get("schema_id")
        != "cascadia.r2-map.john1-local-write-attestation.v1"
        or attestation_sha256 != document_sha256(attestation, "attestation_sha256")
        or not _remote_relative(attestation_document.evidence.relative)
        or not _verified_remote_json_token_matches(
            attestation_document,
            expected_relative=attestation_document.evidence.relative,
            expected_mode=0o400,
        )
        or not _verified_remote_json_token_matches(
            publication_document,
            expected_relative=expected_relative,
            expected_mode=0o400,
        )
    ):
        raise R2MapRemoteIdentityError("John1 attestation object identity differs")
    receipt = validate_immutable_publication_receipt(
        publication_document,
        object_relative=attestation_document.evidence.relative,
        object_sha256=attestation_document.payload_sha256,
        object_size=attestation_document.evidence.object_token.get("size"),
    )
    expected_request_id = PurePosixPath(expected_relative).stem
    if receipt.get("request_id") != expected_request_id:
        raise R2MapRemoteIdentityError("John1 attestation publication request differs")
    return {
        "relative": expected_relative,
        "object_sha256": publication_document.payload_sha256,
        "object_token_sha256": publication_document.evidence.object_token[
            "token_sha256"
        ],
        "receipt_sha256": receipt["receipt_sha256"],
    }


def bootstrap_phase_barrier_relative(transaction: Mapping[str, Any]) -> str:
    target = transaction.get("target_relative")
    if (
        not isinstance(target, str)
        or PurePosixPath(target).is_absolute()
        or PurePosixPath(target).parts[:1] != ("datasets",)
        or any(part in {"", ".", ".."} for part in PurePosixPath(target).parts)
    ):
        raise R2MapRemoteIdentityError("bootstrap dataset transaction target is invalid")
    return f"{target}.bootstrap-phase-barrier.json"


def validate_bootstrap_phase_barrier_value(
    value: object,
    *,
    dataset_target_relative: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "phase",
        "controller_state_sha256",
        "aggregate_task_id",
        "phase_receipts",
        "receipt_count",
        "dataset_transaction",
        "compact_index",
        "generation_manifest",
        "identity_sha256",
        "publication_receipt_relative",
        "barrier_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise R2MapRemoteIdentityError("bootstrap phase barrier schema differs")
    barrier = dict(value)
    semantic = {
        key: item
        for key, item in barrier.items()
        if key not in {"identity_sha256", "publication_receipt_relative", "barrier_sha256"}
    }
    identity_sha256 = hashlib.sha256(canonical_json(semantic)).hexdigest()
    request_id = f"req-bootstrap-barrier-{identity_sha256[:32]}"
    publication_relative = f"control/receipts/{request_id}.json"
    transaction = barrier.get("dataset_transaction")
    compact_index = barrier.get("compact_index")
    phase_receipts = barrier.get("phase_receipts")
    generation_manifest = barrier.get("generation_manifest")
    transaction_required = {
        "target_relative",
        "manifest_relative",
        "manifest_sha256",
        "commit_receipt_relative",
        "commit_receipt_sha256",
    }
    index_required = {
        "relative",
        "payload_sha256",
        "index_blake3",
        "protocol_id",
        "game_count",
        "collection_kind",
        "dataset_blake3",
        "shard_root_relative",
        "shard_count",
    }
    generation_manifest_required = {
        "relative",
        "bytes",
        "payload_sha256",
        "identity_sha256",
        "publication_receipt_relative",
        "publication_receipt_sha256",
    }
    if (
        barrier["schema_version"] != 1
        or barrier["schema_id"] != BOOTSTRAP_PHASE_BARRIER_SCHEMA
        or barrier["campaign_id"] != CAMPAIGN_ID
        or barrier["phase"] != "bootstrap-generating"
        or barrier["aggregate_task_id"] != BOOTSTRAP_AGGREGATE_OPERATION
        or not _sha256(barrier["controller_state_sha256"])
        or barrier["receipt_count"] != len(BOOTSTRAP_PHASE_RECEIPT_CONTRACT)
        or not isinstance(transaction, dict)
        or set(transaction) != transaction_required
        or not isinstance(compact_index, dict)
        or set(compact_index) != index_required
        or not isinstance(generation_manifest, dict)
        or set(generation_manifest) != generation_manifest_required
        or barrier["identity_sha256"] != identity_sha256
        or barrier["publication_receipt_relative"] != publication_relative
        or barrier["barrier_sha256"] != document_sha256(barrier, "barrier_sha256")
    ):
        raise R2MapRemoteIdentityError("bootstrap phase barrier identity differs")
    expected_manifest_relative = f"{dataset_target_relative}/.r2-map-transaction.json"
    expected_generation_manifest_relative = (
        f"{dataset_target_relative}.generation-manifest.json"
    )
    if (
        transaction["target_relative"] != dataset_target_relative
        or transaction["manifest_relative"] != expected_manifest_relative
        or not _sha256(transaction["manifest_sha256"])
        or not _remote_relative(transaction["commit_receipt_relative"])
        or not _sha256(transaction["commit_receipt_sha256"])
        or not _remote_relative(compact_index["relative"])
        or not compact_index["relative"].startswith(f"{dataset_target_relative}/")
        or not _sha256(compact_index["payload_sha256"])
        or not _sha256(compact_index["index_blake3"])
        or compact_index["protocol_id"] != "r2-map-compact-index-v3"
        or compact_index["game_count"] != BOOTSTRAP_GAMES
        or compact_index["collection_kind"] != "bootstrap"
        or not _sha256(compact_index["dataset_blake3"])
        or not _remote_relative(compact_index["shard_root_relative"])
        or not compact_index["shard_root_relative"].startswith(
            f"{dataset_target_relative}/"
        )
        or not isinstance(compact_index["shard_count"], int)
        or isinstance(compact_index["shard_count"], bool)
        or compact_index["shard_count"] <= 0
        or generation_manifest["relative"] != expected_generation_manifest_relative
        or not isinstance(generation_manifest["bytes"], int)
        or isinstance(generation_manifest["bytes"], bool)
        or generation_manifest["bytes"] <= 0
        or not _sha256(generation_manifest["payload_sha256"])
        or not _sha256(generation_manifest["identity_sha256"])
        or not _remote_relative(generation_manifest["publication_receipt_relative"])
        or not _sha256(generation_manifest["publication_receipt_sha256"])
    ):
        raise R2MapRemoteIdentityError("bootstrap phase barrier dataset binding differs")
    if not isinstance(phase_receipts, list) or len(phase_receipts) != len(
        BOOTSTRAP_PHASE_RECEIPT_CONTRACT
    ):
        raise R2MapRemoteIdentityError("bootstrap phase receipt count differs")
    phase_required = {
        "task_id",
        "task_kind",
        "host",
        "packet_sha256",
        "receipt_relative",
        "receipt_sha256",
    }
    for entry, (operation, task_kind, host) in zip(
        phase_receipts, BOOTSTRAP_PHASE_RECEIPT_CONTRACT, strict=True
    ):
        if (
            not isinstance(entry, dict)
            or set(entry) != phase_required
            or not isinstance(entry["task_id"], str)
            or not entry["task_id"].endswith(f"-{operation}")
            or entry["task_kind"] != task_kind
            or entry["host"] != host
            or not _sha256(entry["packet_sha256"])
            or entry["receipt_relative"] != f"control/receipts/{entry['task_id']}.json"
            or not _sha256(entry["receipt_sha256"])
        ):
            raise R2MapRemoteIdentityError("bootstrap phase receipt binding differs")
    task_ids = [entry["task_id"] for entry in phase_receipts]
    if len(task_ids) != len(set(task_ids)):
        raise R2MapRemoteIdentityError("bootstrap aggregate task identity differs")
    return barrier


def validate_bootstrap_generation_manifest_value(
    value: object,
    *,
    dataset_target_relative: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "phase",
        "controller_state_sha256",
        "aggregate_task_id",
        "generation_receipts",
        "dataset_transaction",
        "compact_index",
        "shard_bindings",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise R2MapRemoteIdentityError("bootstrap generation manifest schema differs")
    manifest = dict(value)
    identity = document_sha256(manifest, "identity_sha256")
    transaction_required = {
        "target_relative",
        "manifest_relative",
        "manifest_sha256",
        "commit_receipt_relative",
        "commit_receipt_sha256",
    }
    index_required = {
        "relative",
        "bytes",
        "sha256",
        "index_blake3",
        "protocol_id",
        "collection_kind",
        "game_count",
        "dataset_blake3",
        "shard_root_relative",
        "shard_count",
    }
    receipt_required = {
        "task_id",
        "host",
        "packet_sha256",
        "receipt_relative",
        "receipt_sha256",
        "used_seed_prefix",
        "artifacts",
    }
    artifact_required = {
        "label",
        "path",
        "bytes",
        "sha256",
        "storage_receipt_relative",
        "storage_receipt_sha256",
    }
    binding_required = {
        "source_task_id",
        "source_artifact_path",
        "source_artifact_sha256",
        "source_artifact_bytes",
        "target_relative",
        "target_sha256",
        "target_blake3",
        "bytes",
        "file_name",
        "first_game_index",
        "next_game_index",
        "game_count",
    }
    transaction = manifest.get("dataset_transaction")
    compact = manifest.get("compact_index")
    receipts = manifest.get("generation_receipts")
    bindings = manifest.get("shard_bindings")
    if (
        manifest["schema_version"] != 1
        or manifest["schema_id"] != BOOTSTRAP_GENERATION_MANIFEST_SCHEMA
        or manifest["campaign_id"] != CAMPAIGN_ID
        or manifest["phase"] != "bootstrap-generating"
        or not _sha256(manifest["controller_state_sha256"])
        or manifest["aggregate_task_id"] != BOOTSTRAP_AGGREGATE_OPERATION
        or manifest["identity_sha256"] != identity
        or not isinstance(transaction, dict)
        or set(transaction) != transaction_required
        or not isinstance(compact, dict)
        or set(compact) != index_required
        or not isinstance(receipts, list)
        or len(receipts) != 3
        or not isinstance(bindings, list)
        or not bindings
    ):
        raise R2MapRemoteIdentityError("bootstrap generation manifest identity differs")
    if (
        transaction["target_relative"] != dataset_target_relative
        or transaction["manifest_relative"]
        != f"{dataset_target_relative}/.r2-map-transaction.json"
        or not _sha256(transaction["manifest_sha256"])
        or not _remote_relative(transaction["commit_receipt_relative"])
        or not _sha256(transaction["commit_receipt_sha256"])
        or not _remote_relative(compact["relative"])
        or not compact["relative"].startswith(f"{dataset_target_relative}/")
        or not isinstance(compact["bytes"], int)
        or isinstance(compact["bytes"], bool)
        or compact["bytes"] <= 0
        or not _sha256(compact["sha256"])
        or not _sha256(compact["index_blake3"])
        or compact["protocol_id"] != "r2-map-compact-index-v3"
        or compact["collection_kind"] != "bootstrap"
        or compact["game_count"] != BOOTSTRAP_GAMES
        or not _sha256(compact["dataset_blake3"])
        or not _remote_relative(compact["shard_root_relative"])
        or not compact["shard_root_relative"].startswith(f"{dataset_target_relative}/")
        or not isinstance(compact["shard_count"], int)
        or isinstance(compact["shard_count"], bool)
        or compact["shard_count"] <= 0
        or len(bindings) != compact["shard_count"]
    ):
        raise R2MapRemoteIdentityError("bootstrap generation dataset binding differs")
    expected_receipt_hosts = ("john1", "john2", "john3")
    observed_task_ids: list[str] = []
    for receipt, host in zip(receipts, expected_receipt_hosts, strict=True):
        if not isinstance(receipt, dict) or set(receipt) != receipt_required:
            raise R2MapRemoteIdentityError("bootstrap generation receipt schema differs")
        prefix = receipt["used_seed_prefix"]
        artifacts = receipt["artifacts"]
        if (
            receipt["host"] != host
            or not isinstance(receipt["task_id"], str)
            or not receipt["task_id"].endswith(f"-bootstrap-generate-{host}")
            or not _sha256(receipt["packet_sha256"])
            or receipt["receipt_relative"] != f"control/receipts/{receipt['task_id']}.json"
            or not _sha256(receipt["receipt_sha256"])
            or not isinstance(prefix, dict)
            or set(prefix) != {"lease_sha256", "used_count", "unused_count", "last_index"}
            or not _sha256(prefix["lease_sha256"])
            or not isinstance(prefix["used_count"], int)
            or isinstance(prefix["used_count"], bool)
            or prefix["used_count"] <= 0
            or not isinstance(prefix["unused_count"], int)
            or isinstance(prefix["unused_count"], bool)
            or prefix["unused_count"] < 0
            or not isinstance(prefix["last_index"], int)
            or isinstance(prefix["last_index"], bool)
            or not isinstance(artifacts, list)
            or not artifacts
            or artifacts
            != sorted(artifacts, key=lambda item: (item.get("label", ""), item.get("path", "")))
        ):
            raise R2MapRemoteIdentityError("bootstrap generation receipt identity differs")
        for artifact in artifacts:
            if (
                not isinstance(artifact, dict)
                or set(artifact) != artifact_required
                or not isinstance(artifact["label"], str)
                or not artifact["label"]
                or not _remote_relative(artifact["path"])
                or not isinstance(artifact["bytes"], int)
                or isinstance(artifact["bytes"], bool)
                or artifact["bytes"] <= 0
                or not _sha256(artifact["sha256"])
                or not _remote_relative(artifact["storage_receipt_relative"])
                or not _sha256(artifact["storage_receipt_sha256"])
            ):
                raise R2MapRemoteIdentityError("bootstrap generation artifact differs")
        observed_task_ids.append(receipt["task_id"])
    if (
        observed_task_ids != sorted(observed_task_ids)
        or len(set(observed_task_ids)) != 3
        or sum(receipt["used_seed_prefix"]["used_count"] for receipt in receipts)
        != BOOTSTRAP_GAMES
    ):
        raise R2MapRemoteIdentityError("bootstrap generation receipts are not canonically ordered")
    if bindings != sorted(bindings, key=lambda item: item.get("file_name", "")):
        raise R2MapRemoteIdentityError("bootstrap shard bindings are not canonically ordered")
    seen_files: set[str] = set()
    seen_source_artifacts: set[tuple[str, str]] = set()
    for binding in bindings:
        if (
            not isinstance(binding, dict)
            or set(binding) != binding_required
            or binding["source_task_id"] not in observed_task_ids
            or not _remote_relative(binding["source_artifact_path"])
            or not _sha256(binding["source_artifact_sha256"])
            or not isinstance(binding["source_artifact_bytes"], int)
            or isinstance(binding["source_artifact_bytes"], bool)
            or binding["source_artifact_bytes"] <= 0
            or not _remote_relative(binding["target_relative"])
            or not binding["target_relative"].startswith(f"{compact['shard_root_relative']}/")
            or not _sha256(binding["target_sha256"])
            or not _sha256(binding["target_blake3"])
            or not isinstance(binding["bytes"], int)
            or isinstance(binding["bytes"], bool)
            or binding["bytes"] <= 0
            or not isinstance(binding["file_name"], str)
            or PurePosixPath(binding["file_name"]).name != binding["file_name"]
            or binding["file_name"] in seen_files
            or (binding["source_task_id"], binding["source_artifact_path"])
            in seen_source_artifacts
            or any(
                not isinstance(binding[name], int)
                or isinstance(binding[name], bool)
                or binding[name] < 0
                for name in ("first_game_index", "next_game_index", "game_count")
            )
            or binding["game_count"] <= 0
            or binding["next_game_index"] - binding["first_game_index"]
            != binding["game_count"]
            or binding["target_relative"]
            != f"{compact['shard_root_relative']}/{binding['file_name']}"
            or binding["bytes"] != binding["source_artifact_bytes"]
            or binding["target_sha256"] != binding["source_artifact_sha256"]
        ):
            raise R2MapRemoteIdentityError("bootstrap shard binding identity differs")
        source_receipt = next(
            receipt for receipt in receipts if receipt["task_id"] == binding["source_task_id"]
        )
        artifact_matches = [
            artifact
            for artifact in source_receipt["artifacts"]
            if artifact["path"] == binding["source_artifact_path"]
        ]
        if (
            len(artifact_matches) != 1
            or artifact_matches[0]["sha256"] != binding["source_artifact_sha256"]
            or artifact_matches[0]["bytes"] != binding["source_artifact_bytes"]
        ):
            raise R2MapRemoteIdentityError("bootstrap source artifact binding differs")
        seen_files.add(binding["file_name"])
        seen_source_artifacts.add(
            (binding["source_task_id"], binding["source_artifact_path"])
        )
    for receipt in receipts:
        if sum(
            binding["game_count"]
            for binding in bindings
            if binding["source_task_id"] == receipt["task_id"]
        ) != receipt["used_seed_prefix"]["used_count"]:
            raise R2MapRemoteIdentityError("bootstrap per-task shard game accounting differs")
    if sum(binding["game_count"] for binding in bindings) != BOOTSTRAP_GAMES:
        raise R2MapRemoteIdentityError("bootstrap shard game accounting differs")
    return manifest


def validate_bootstrap_aggregate_generation_binding(
    *,
    barrier: Mapping[str, Any],
    generation_manifest: Mapping[str, Any],
    packets_by_task: Mapping[str, Mapping[str, Any]],
    receipts_by_task: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Prove the controller aggregate's sole artifact is the bound manifest."""

    phase_entries = barrier["phase_receipts"]
    generation_task_ids = tuple(entry["task_id"] for entry in phase_entries[:-1])
    aggregate_task_id = phase_entries[-1]["task_id"]
    aggregate_packet = packets_by_task.get(aggregate_task_id)
    aggregate_receipt = receipts_by_task.get(aggregate_task_id)
    if not isinstance(aggregate_packet, Mapping) or not isinstance(
        aggregate_receipt, Mapping
    ):
        raise R2MapRemoteIdentityError("bootstrap aggregate packet/receipt is absent")
    generation_receipts = generation_manifest["generation_receipts"]
    generation_receipts_by_task = {
        receipt["task_id"]: receipt for receipt in generation_receipts
    }
    if set(generation_receipts_by_task) != set(generation_task_ids):
        raise R2MapRemoteIdentityError("generation manifest task set differs")
    for task_id in generation_task_ids:
        packet = packets_by_task.get(task_id)
        receipt = receipts_by_task.get(task_id)
        mirrored = generation_receipts_by_task[task_id]
        if (
            not isinstance(packet, Mapping)
            or not isinstance(receipt, Mapping)
            or mirrored["host"] != receipt["host"]
            or mirrored["packet_sha256"] != packet["packet_sha256"]
            or mirrored["receipt_relative"] != f"control/receipts/{task_id}.json"
            or mirrored["receipt_sha256"] != receipt["receipt_sha256"]
            or mirrored["used_seed_prefix"] != receipt["used_seed_prefix"]
            or mirrored["artifacts"] != receipt["artifacts"]
        ):
            raise R2MapRemoteIdentityError("generation manifest receipt mirror differs")
    aggregate_artifacts = aggregate_receipt["artifacts"]
    generation_section = barrier["generation_manifest"]
    if (
        aggregate_packet["dependencies"] != list(generation_task_ids)
        or aggregate_packet["aggregate_kind"] != "generation"
        or not isinstance(aggregate_artifacts, list)
        or len(aggregate_artifacts) != 1
    ):
        raise R2MapRemoteIdentityError("bootstrap aggregate dependency/artifact set differs")
    aggregate_artifact = aggregate_artifacts[0]
    if (
        aggregate_artifact["label"] != "generation-manifest"
        or aggregate_artifact["path"] != generation_section["relative"]
        or aggregate_artifact["bytes"] != generation_section["bytes"]
        or aggregate_artifact["sha256"] != generation_section["payload_sha256"]
        or aggregate_artifact["storage_receipt_relative"]
        != generation_section["publication_receipt_relative"]
        or aggregate_artifact["storage_receipt_sha256"]
        != generation_section["publication_receipt_sha256"]
    ):
        raise R2MapRemoteIdentityError("bootstrap aggregate generation-manifest binding differs")
    return generation_task_ids


def load_verified_bootstrap_phase_barrier(
    client: RemoteStorageClient,
    *,
    dataset_transaction_hint: VerifiedRemoteJson,
) -> tuple[
    dict[str, Any],
    VerifiedRemoteJson,
    VerifiedRemoteJson,
    VerifiedRemoteJson,
]:
    """Reopen and prove the W7 phase barrier and every object it authorizes."""

    hinted_transaction = validate_transaction_manifest(dataset_transaction_hint)
    target = hinted_transaction["target_relative"]
    barrier_relative = bootstrap_phase_barrier_relative(hinted_transaction)
    barrier_document = load_verified_remote_json(client, barrier_relative, maximum_bytes=8 << 20)
    barrier = validate_bootstrap_phase_barrier_value(
        barrier_document.value,
        dataset_target_relative=target,
    )
    if (
        barrier_document.payload_sha256 != content_sha256(canonical_json(barrier))
        or barrier_document.evidence.relative != barrier_relative
        or barrier_document.evidence.object_token.get("mode") != 0o400
    ):
        raise R2MapRemoteIdentityError("bootstrap phase barrier object identity differs")

    publication_document = load_verified_remote_json(
        client,
        barrier["publication_receipt_relative"],
        maximum_bytes=2 << 20,
    )
    publication = validate_worker_receipt(publication_document, "put-file")
    expected_request_id = PurePosixPath(barrier["publication_receipt_relative"]).stem
    result = publication.get("result")
    if publication.get("request_id") != expected_request_id:
        raise R2MapRemoteIdentityError("bootstrap phase barrier request identity differs")
    if (
        not isinstance(result, dict)
        or set(result)
        != {
            "relative",
            "sha256",
            "size",
            "mode",
            "previous_sha256",
            "payload_size",
            "payload_sha256",
        }
        or result["relative"] != barrier_relative
        or result["sha256"] != barrier_document.payload_sha256
        or result["size"] != barrier_document.evidence.object_token.get("size")
        or result["mode"] != "0o400"
        or result["previous_sha256"] is not None
        or result["payload_size"] != 0
        or result["payload_sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise R2MapRemoteIdentityError("bootstrap phase barrier publication differs")

    transaction_section = barrier["dataset_transaction"]
    transaction_document = load_verified_remote_json(
        client,
        transaction_section["manifest_relative"],
    )
    transaction = validate_transaction_manifest(transaction_document)
    if (
        transaction["target_relative"] != target
        or transaction["manifest_sha256"] != transaction_section["manifest_sha256"]
        or transaction_document.evidence.object_token.get("token_sha256")
        != dataset_transaction_hint.evidence.object_token.get("token_sha256")
    ):
        raise R2MapRemoteIdentityError("barrier-bound dataset transaction differs")
    commit_document = load_verified_remote_json(
        client,
        transaction_section["commit_receipt_relative"],
        maximum_bytes=2 << 20,
    )
    commit = validate_transaction_commit(commit_document, transaction)
    if commit["receipt_sha256"] != transaction_section["commit_receipt_sha256"]:
        raise R2MapRemoteIdentityError("barrier-bound dataset commit differs")

    compact_section = barrier["compact_index"]
    index_document = load_verified_remote_json(client, compact_section["relative"])
    require_transaction_object(transaction, index_document)
    from cascadia_mlx.r2_map_dataset import validate_compact_index_value

    index = validate_compact_index_value(index_document.value)
    manifest = index["dataset_manifest"]
    if (
        index_document.payload_sha256 != compact_section["payload_sha256"]
        or index["index_blake3"] != compact_section["index_blake3"]
        or index["protocol_id"] != compact_section["protocol_id"]
        or manifest["game_count"] != compact_section["game_count"]
        or manifest["round"]["collection_kind"] != compact_section["collection_kind"]
        or manifest["dataset_blake3"] != compact_section["dataset_blake3"]
        or len(manifest["sources"]) != compact_section["shard_count"]
    ):
        raise R2MapRemoteIdentityError("barrier-bound compact index differs")
    for source in manifest["sources"]:
        shard_relative = (
            f"{compact_section['shard_root_relative']}/{source['file_name']}"
        )
        descriptor = transaction_object_descriptor(transaction, shard_relative)
        if (
            descriptor.get("size") != source["bytes"]
            or descriptor.get("mode", "0400") != "0400"
        ):
            raise R2MapRemoteIdentityError("barrier-bound compact shard differs")
        require_open_transaction_object(client, transaction, shard_relative)

    generation_section = barrier["generation_manifest"]
    generation_document = load_verified_remote_json(
        client,
        generation_section["relative"],
        maximum_bytes=8 << 20,
    )
    generation = validate_bootstrap_generation_manifest_value(
        generation_document.value,
        dataset_target_relative=target,
    )
    if (
        generation_document.payload_sha256 != content_sha256(canonical_json(generation))
        or generation_document.evidence.object_token.get("mode") != 0o400
        or generation_document.evidence.object_token.get("size") != generation_section["bytes"]
        or generation_document.payload_sha256 != generation_section["payload_sha256"]
        or generation["identity_sha256"] != generation_section["identity_sha256"]
        or generation["controller_state_sha256"] != barrier["controller_state_sha256"]
        or generation["dataset_transaction"] != barrier["dataset_transaction"]
        or generation["compact_index"]["relative"] != compact_section["relative"]
        or generation["compact_index"]["bytes"]
        != index_document.evidence.object_token.get("size")
        or generation["compact_index"]["sha256"] != compact_section["payload_sha256"]
        or generation["compact_index"]["index_blake3"] != compact_section["index_blake3"]
        or generation["compact_index"]["protocol_id"] != compact_section["protocol_id"]
        or generation["compact_index"]["collection_kind"]
        != compact_section["collection_kind"]
        or generation["compact_index"]["game_count"] != compact_section["game_count"]
        or generation["compact_index"]["dataset_blake3"] != compact_section["dataset_blake3"]
        or generation["compact_index"]["shard_root_relative"]
        != compact_section["shard_root_relative"]
        or generation["compact_index"]["shard_count"] != compact_section["shard_count"]
    ):
        raise R2MapRemoteIdentityError("barrier-bound generation manifest differs")
    generation_publication_document = load_verified_remote_json(
        client,
        generation_section["publication_receipt_relative"],
        maximum_bytes=2 << 20,
    )
    generation_publication = validate_worker_receipt(
        generation_publication_document,
        "put-file",
    )
    generation_publication_result = generation_publication.get("result")
    if (
        generation_publication["receipt_sha256"]
        != generation_section["publication_receipt_sha256"]
        or not isinstance(generation_publication_result, dict)
        or set(generation_publication_result)
        != {
            "relative",
            "sha256",
            "size",
            "mode",
            "previous_sha256",
            "payload_size",
            "payload_sha256",
        }
        or generation_publication_result["relative"] != generation_section["relative"]
        or generation_publication_result["sha256"] != generation_section["payload_sha256"]
        or generation_publication_result["size"] != generation_section["bytes"]
        or generation_publication_result["mode"] != "0o400"
        or generation_publication_result["previous_sha256"] is not None
        or generation_publication_result["payload_size"] != 0
        or generation_publication_result["payload_sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise R2MapRemoteIdentityError("bootstrap generation manifest publication differs")

    binding_by_name = {binding["file_name"]: binding for binding in generation["shard_bindings"]}
    if set(binding_by_name) != {source["file_name"] for source in manifest["sources"]}:
        raise R2MapRemoteIdentityError("bootstrap compact source/shard bijection differs")
    for source in manifest["sources"]:
        binding = binding_by_name[source["file_name"]]
        descriptor = transaction_object_descriptor(transaction, binding["target_relative"])
        if (
            binding["target_blake3"] != source["blake3"]
            or binding["bytes"] != source["bytes"]
            or binding["first_game_index"] != source["first_game_index"]
            or binding["next_game_index"] != source["next_game_index"]
            or binding["game_count"] != source["game_count"]
            or descriptor.get("sha256") != binding["target_sha256"]
            or descriptor.get("size") != binding["bytes"]
            or descriptor.get("mode", "0400") != "0400"
        ):
            raise R2MapRemoteIdentityError("bootstrap compact source/shard identity differs")

    total_generated_games = 0
    phase_evidence = []
    packets_by_task: dict[str, dict[str, Any]] = {}
    receipts_by_task: dict[str, dict[str, Any]] = {}
    for entry, (operation, task_kind, host) in zip(
        barrier["phase_receipts"], BOOTSTRAP_PHASE_RECEIPT_CONTRACT, strict=True
    ):
        packet_document = load_verified_remote_json(
            client,
            f"control/work-packets/{entry['task_id']}.json",
            maximum_bytes=2 << 20,
        )
        receipt_document = load_verified_remote_json(
            client,
            entry["receipt_relative"],
            maximum_bytes=8 << 20,
        )
        packet = controller.validate_work_packet(packet_document.value)
        receipt = controller.validate_receipt(receipt_document.value, packet=packet)
        if (
            packet["operation"] != operation
            or packet["task_kind"] != task_kind
            or packet["host"] != host
            or packet["phase"] != barrier["phase"]
            or packet["controller_state_sha256"] != barrier["controller_state_sha256"]
            or packet["packet_sha256"] != entry["packet_sha256"]
            or receipt["receipt_sha256"] != entry["receipt_sha256"]
            or packet_document.payload_sha256 != content_sha256(canonical_json(packet))
            or receipt_document.payload_sha256 != content_sha256(canonical_json(receipt))
        ):
            raise R2MapRemoteIdentityError("bootstrap controller packet/receipt differs")
        _verify_controller_receipt_storage_evidence(client, receipt, packet)
        if task_kind == "generate":
            total_generated_games += receipt["used_seed_prefix"]["used_count"]
            if receipt["metrics"]["games"] != receipt["used_seed_prefix"]["used_count"]:
                raise R2MapRemoteIdentityError("bootstrap generation game metric differs")
        elif receipt["metrics"]["games"] != 0:
            raise R2MapRemoteIdentityError("bootstrap aggregate game metric differs")
        packets_by_task[entry["task_id"]] = packet
        receipts_by_task[entry["task_id"]] = receipt
        phase_evidence.append(
            {
                "task_id": entry["task_id"],
                "packet": packet_document.to_dict(),
                "receipt": receipt_document.to_dict(),
            }
        )
    expected_dependencies = validate_bootstrap_aggregate_generation_binding(
        barrier=barrier,
        generation_manifest=generation,
        packets_by_task=packets_by_task,
        receipts_by_task=receipts_by_task,
    )
    used_seed_indices: set[int] = set()
    for task_id in expected_dependencies:
        packet = packets_by_task[task_id]
        receipt = receipts_by_task[task_id]
        lease = packet["seed_lease"]
        prefix = receipt["used_seed_prefix"]
        if not isinstance(lease, dict):
            raise R2MapRemoteIdentityError("generation packet seed lease is absent")
        indices = set(
            range(
                lease["first_index"],
                lease["first_index"] + prefix["used_count"] * lease["stride"],
                lease["stride"],
            )
        )
        if len(indices) != prefix["used_count"] or used_seed_indices.intersection(indices):
            raise R2MapRemoteIdentityError("bootstrap generation seed prefixes overlap")
        used_seed_indices.update(indices)
    if (
        total_generated_games != BOOTSTRAP_GAMES
        or len(used_seed_indices) != BOOTSTRAP_GAMES
    ):
        raise R2MapRemoteIdentityError("bootstrap phase completion accounting differs")

    identity = {
        "barrier_relative": barrier_relative,
        "identity_sha256": barrier["identity_sha256"],
        "barrier_sha256": barrier["barrier_sha256"],
        "controller_state_sha256": barrier["controller_state_sha256"],
        "phase_receipt_count": len(phase_evidence),
        "generation_manifest_relative": generation_section["relative"],
        "generation_manifest_payload_sha256": generation_section["payload_sha256"],
        "generation_manifest_identity_sha256": generation_section["identity_sha256"],
        "generation_manifest_publication_receipt_relative": generation_section[
            "publication_receipt_relative"
        ],
        "generation_manifest_publication_receipt_sha256": generation_section[
            "publication_receipt_sha256"
        ],
        "dataset_target_relative": target,
        "dataset_transaction_manifest_relative": transaction_document.evidence.relative,
        "dataset_transaction_commit_receipt_relative": commit_document.evidence.relative,
        "compact_index_relative": index_document.evidence.relative,
        "shard_root_relative": compact_section["shard_root_relative"],
        "barrier_document": barrier_document.to_dict(),
        "publication_receipt": publication_document.to_dict(),
        "publication_receipt_sha256": publication["receipt_sha256"],
        "generation_manifest_document": generation_document.to_dict(),
        "generation_manifest_publication_receipt": generation_publication_document.to_dict(),
    }
    return identity, index_document, transaction_document, commit_document


def _verify_controller_receipt_storage_evidence(
    client: RemoteStorageClient,
    receipt: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> None:
    """Remote-object equivalent of the controller's local storage-evidence audit."""

    controller.validate_receipt(receipt, packet=packet)
    for artifact in receipt["artifacts"]:
        publication_document = load_verified_remote_json(
            client,
            artifact["storage_receipt_relative"],
            maximum_bytes=2 << 20,
        )
        operation = publication_document.value.get("operation")
        if operation not in {"put-file", "put-stream", "transaction-commit"}:
            raise R2MapRemoteIdentityError("controller artifact publication operation differs")
        publication = validate_worker_receipt(publication_document, operation)
        if publication["receipt_sha256"] != artifact["storage_receipt_sha256"]:
            raise R2MapRemoteIdentityError("controller artifact publication receipt differs")
        result = publication.get("result")
        if not isinstance(result, dict):
            raise R2MapRemoteIdentityError("controller artifact publication result is absent")
        if operation in {"put-file", "put-stream"}:
            expected_fields = {
                "relative",
                "sha256",
                "size",
                "mode",
                "previous_sha256",
                "payload_size",
                "payload_sha256",
            }
            if operation == "put-stream":
                expected_fields.add("max_bytes")
            if (
                set(result) != expected_fields
                or result["relative"] != artifact["path"]
                or result["sha256"] != artifact["sha256"]
                or result["size"] != artifact["bytes"]
                or result["mode"] != "0o400"
                or result["previous_sha256"] is not None
                or (
                    "max_bytes" in result
                    and (
                        not isinstance(result["max_bytes"], int)
                        or isinstance(result["max_bytes"], bool)
                        or result["max_bytes"] < artifact["bytes"]
                    )
                )
                or result["payload_size"] != 0
                or result["payload_sha256"] != hashlib.sha256(b"").hexdigest()
            ):
                raise R2MapRemoteIdentityError("controller direct artifact publication differs")
            opened = client.open_object_with_receipt(artifact["path"])["object_token"]
            if (
                opened.get("sha256") != artifact["sha256"]
                or opened.get("size") != artifact["bytes"]
                or opened.get("mode") != 0o400
            ):
                raise R2MapRemoteIdentityError("controller direct artifact object differs")
            continue
        required = {
            "transaction_id",
            "target_relative",
            "manifest_sha256",
            "object_count",
            "committed",
            "payload_size",
            "payload_sha256",
        }
        if (
            set(result) != required
            or result["committed"] is not True
            or result["payload_size"] != 0
            or result["payload_sha256"] != hashlib.sha256(b"").hexdigest()
        ):
            raise R2MapRemoteIdentityError("controller transaction publication differs")
        provenance = load_verified_remote_json(
            client,
            f"{result['target_relative']}/.r2-map-transaction.json",
            maximum_bytes=2 << 20,
        )
        transaction = validate_transaction_manifest(provenance)
        validate_transaction_commit(publication_document, transaction)
        descriptor = transaction_object_descriptor(transaction, artifact["path"])
        expected_mode = 0o500 if descriptor.get("mode") == "0500" else 0o400
        if (
            descriptor.get("sha256") != artifact["sha256"]
            or descriptor.get("size") != artifact["bytes"]
            or result["object_count"] != len(transaction["objects"])
        ):
            raise R2MapRemoteIdentityError("controller transaction artifact differs")
        opened = client.open_object_with_receipt(artifact["path"])["object_token"]
        if (
            opened.get("sha256") != artifact["sha256"]
            or opened.get("size") != artifact["bytes"]
            or opened.get("mode") != expected_mode
        ):
            raise R2MapRemoteIdentityError("controller transaction object differs")


def validate_source_identity(
    *,
    source_manifest: VerifiedRemoteJson,
    reference_manifest: VerifiedRemoteJson,
    source_archive_verification: VerifiedRemoteJson,
    transaction_manifest: VerifiedRemoteJson,
    transaction_commit_receipt: VerifiedRemoteJson,
) -> dict[str, Any]:
    transaction = validate_transaction_manifest(transaction_manifest)
    if PurePosixPath(transaction["target_relative"]).parts[:1] != ("source",):
        raise R2MapRemoteIdentityError("W0 source transaction is outside source/")
    require_transaction_object(transaction, source_manifest)
    require_transaction_object(transaction, reference_manifest)
    validate_transaction_commit(transaction_commit_receipt, transaction)
    source = source_manifest.value
    reference = reference_manifest.value
    archive_verification = source_archive_verification.value
    if (
        source.get("schema_id") != SOURCE_MANIFEST_SCHEMA
        or source.get("campaign_id") != CAMPAIGN_ID
        or source.get("document_sha256") != document_sha256(source, "document_sha256")
        or source.get("protected_seed_values_opened") is not False
    ):
        raise R2MapRemoteIdentityError("W0 source manifest identity is invalid")
    files = source.get("files")
    if (
        not isinstance(files, list)
        or not files
        or len(files) != source.get("file_count")
        or files != sorted(files, key=lambda item: item.get("relative", ""))
        or source.get("total_bytes") != sum(
            item.get("size", -1) for item in files if isinstance(item, dict)
        )
    ):
        raise R2MapRemoteIdentityError("W0 source file inventory is invalid")
    seen_source_paths: set[str] = set()
    for item in files:
        try:
            archive_relative = item["relative"].encode("ascii")
        except (KeyError, TypeError, AttributeError, UnicodeEncodeError) as error:
            raise R2MapRemoteIdentityError(
                "W0 source path is outside the strict USTAR contract"
            ) from error
        if (
            not isinstance(item, dict)
            or set(item) != {"relative", "size", "sha256", "mode"}
            or not _remote_relative(item["relative"])
            or item["relative"] in seen_source_paths
            or not isinstance(item["size"], int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
            or not _sha256(item["sha256"])
            or item["mode"] not in {"0400", "0500"}
            or len(archive_relative) > 100
            or any(part.startswith("._") for part in PurePosixPath(item["relative"]).parts)
        ):
            raise R2MapRemoteIdentityError("W0 source file descriptor is invalid")
        seen_source_paths.add(item["relative"])
        descriptor = transaction_object_descriptor(
            transaction,
            f"{transaction['target_relative']}/{item['relative']}",
        )
        if (
            descriptor.get("sha256") != item["sha256"]
            or descriptor.get("size") != item["size"]
            or descriptor.get("mode", "0400") != item["mode"]
        ):
            raise R2MapRemoteIdentityError("W0 source inventory is not transaction-bound")
    expected_transaction_objects = seen_source_paths | SOURCE_TRANSACTION_CONTROL_OBJECTS
    if _source_path_prefix_collision(seen_source_paths):
        raise R2MapRemoteIdentityError(
            "W0 source inventory contains a file/directory prefix collision"
        )
    observed_transaction_objects = {
        item["relative"] for item in transaction["objects"]
    }
    if observed_transaction_objects != expected_transaction_objects:
        raise R2MapRemoteIdentityError(
            "W0 source transaction object set differs from the exact archive closure"
        )

    target = transaction["target_relative"]

    def bound_descriptor(relative: str, *, mode: str) -> dict[str, Any]:
        descriptor = transaction_object_descriptor(transaction, f"{target}/{relative}")
        observed_mode = descriptor.get("mode", "0400")
        if observed_mode != mode:
            raise R2MapRemoteIdentityError(
                f"W0 source transaction mode differs: {relative}"
            )
        return {
            "relative": f"{target}/{relative}",
            "sha256": descriptor["sha256"],
            "size": descriptor["size"],
            "mode": observed_mode,
        }

    file_entries = {item["relative"]: item for item in files}
    archive_descriptor = bound_descriptor("source.tar", mode="0400")
    archive_report_descriptor = bound_descriptor(
        "source-archive-verification.json", mode="0400"
    )
    archive_verifier_descriptor = bound_descriptor("archive-verify.py", mode="0500")
    verifier_source = file_entries.get(SOURCE_ARCHIVE_VERIFIER_RELATIVE)
    if (
        verifier_source is None
        or verifier_source["mode"] != "0500"
        or archive_verifier_descriptor["sha256"] != verifier_source["sha256"]
        or archive_verifier_descriptor["size"] != verifier_source["size"]
    ):
        raise R2MapRemoteIdentityError(
            "transaction archive verifier differs from its reviewed source"
        )
    gate_alias_descriptors: dict[str, dict[str, Any]] = {}
    for alias, source_relative in SOURCE_GATE_ALIASES.items():
        source_entry = file_entries.get(source_relative)
        descriptor = bound_descriptor(alias, mode="0400")
        if (
            source_entry is None
            or source_entry["mode"] != "0400"
            or descriptor["sha256"] != source_entry["sha256"]
            or descriptor["size"] != source_entry["size"]
        ):
            raise R2MapRemoteIdentityError(
                f"transaction gate alias differs from reviewed source: {alias}"
            )
        gate_alias_descriptors[alias] = descriptor

    require_transaction_object(transaction, source_archive_verification)
    if source_archive_verification.evidence.relative != (
        f"{target}/source-archive-verification.json"
    ):
        raise R2MapRemoteIdentityError("source archive verification path differs")
    report_keys = {
        "schema_id",
        "status",
        "document_sha256",
        "archive_sha256",
        "archive_bytes",
        "member_count",
        "member_names_sha256",
        "content_bytes",
        "terminal_zero_bytes",
        "regular_only",
        "pax_or_extended_headers",
        "metadata_normalized",
    }
    member_names_sha256 = hashlib.sha256(
        ("\n".join(item["relative"] for item in files) + "\n").encode("ascii")
    ).hexdigest()
    raw_member_bytes = sum(
        512 + ((item["size"] + 511) // 512) * 512 for item in files
    )
    canonical_archive_bytes = (
        (raw_member_bytes + 2 * 512 + 10_240 - 1) // 10_240
    ) * 10_240
    if (
        set(archive_verification) != report_keys
        or archive_verification.get("schema_id")
        != SOURCE_ARCHIVE_VERIFICATION_SCHEMA
        or archive_verification.get("status") != "valid"
        or archive_verification.get("document_sha256")
        != source.get("document_sha256")
        or archive_verification.get("archive_sha256")
        != archive_descriptor["sha256"]
        or archive_verification.get("archive_bytes")
        != archive_descriptor["size"]
        or archive_descriptor["size"] != canonical_archive_bytes
        or archive_verification.get("member_count") != source.get("file_count")
        or archive_verification.get("member_names_sha256") != member_names_sha256
        or archive_verification.get("content_bytes") != source.get("total_bytes")
        or archive_verification.get("terminal_zero_bytes")
        != canonical_archive_bytes - raw_member_bytes
        or archive_verification.get("regular_only") is not True
        or archive_verification.get("pax_or_extended_headers") is not False
        or archive_verification.get("metadata_normalized") is not True
        or source_archive_verification.payload_sha256
        != content_sha256(canonical_json(archive_verification) + b"\n")
        or archive_report_descriptor["sha256"]
        != source_archive_verification.payload_sha256
        or archive_report_descriptor["size"]
        != source_archive_verification.evidence.object_token.get("size")
    ):
        raise R2MapRemoteIdentityError(
            "source archive verification differs from the source transaction"
        )
    reference_suffix = "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
    reference_entries = [item for item in files if item.get("relative") == reference_suffix]
    if (
        len(reference_entries) != 1
        or reference_entries[0].get("sha256") != reference_manifest.payload_sha256
        or source.get("w0_reference_manifest_sha256") != reference_manifest.payload_sha256
    ):
        raise R2MapRemoteIdentityError("W0 reference manifest is not source-bound")
    if (
        reference.get("schema_id") != REFERENCE_MANIFEST_SCHEMA
        or reference.get("campaign_id") != CAMPAIGN_ID
        or reference.get("contract_revision") != "sequential-public-market-v1.1"
        or reference.get("manifest_sha256") != document_sha256(reference, "manifest_sha256")
    ):
        raise R2MapRemoteIdentityError("W0 v1.1 reference manifest identity is invalid")
    panels = [
        panel
        for panel in reference.get("panels", ())
        if isinstance(panel, dict) and panel.get("panel_id") == MAXIMUM_WIDTH_PANEL_ID
    ]
    if len(panels) != 1:
        raise R2MapRemoteIdentityError("maximum-width panel is not unique")
    panel = panels[0]
    panel_without_digest = dict(panel)
    claimed_panel_sha256 = panel_without_digest.pop("panel_sha256", None)
    definition = panel.get("definition")
    implementation = reference.get("implementation_identity")
    if (
        claimed_panel_sha256 != content_sha256(canonical_json(panel_without_digest))
        or not isinstance(definition, dict)
        or definition.get("reference_candidate_count") != MAXIMUM_WIDTH_CANDIDATES
        or definition.get("expected_action_evaluations") != MAXIMUM_WIDTH_CANDIDATES
        or definition.get("complete_cardinality_required") is not True
        or definition.get("truncation_allowed") is not False
        or not isinstance(implementation, dict)
        or implementation.get("maximum_width_panel_sha256") != claimed_panel_sha256
    ):
        raise R2MapRemoteIdentityError("registered maximum-width panel is invalid")
    logical_source = {
        "files": files,
        "source_archive": archive_descriptor,
        "source_archive_verification": archive_report_descriptor,
        "source_archive_verifier": archive_verifier_descriptor,
        "source_gate_aliases": gate_alias_descriptors,
    }
    source_blake3 = blake3.blake3(canonical_json(logical_source)).hexdigest()
    return {
        "source_blake3": source_blake3,
        "source_manifest": source_manifest.to_dict(),
        "reference_manifest": reference_manifest.to_dict(),
        "source_archive": archive_descriptor,
        "source_archive_verification": source_archive_verification.to_dict(),
        "source_archive_verification_descriptor": archive_report_descriptor,
        "source_archive_verifier": archive_verifier_descriptor,
        "source_gate_aliases": gate_alias_descriptors,
        "transaction_manifest": transaction_manifest.to_dict(),
        "transaction_manifest_sha256": transaction["manifest_sha256"],
        "transaction_commit_receipt": transaction_commit_receipt.to_dict(),
        "transaction_commit_receipt_sha256": transaction_commit_receipt.payload_sha256,
        "maximum_width_panel_sha256": claimed_panel_sha256,
        "maximum_width_candidates": MAXIMUM_WIDTH_CANDIDATES,
    }
