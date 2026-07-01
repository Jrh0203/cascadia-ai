"""Manifest-bound, non-executable cold-archive migration planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .canonical import CAMPAIGN_ID, D0Error, canonical_json, document_sha256

ARCHIVE_PLAN_SCHEMA = "cascadia.r2-map.legacy-cold-archive-plan.v1"
JOHN2_ARCHIVE_ROOT = "/Users/john2/cascadia-bench/r2-map-archive-v1"
JOHN3_LEGACY_ROOT = "/Users/john3/cascadia-bench/r2-map-v1"


def build_john3_legacy_archive_plan(
    manifest: bytes,
    *,
    manifest_file_sha256: str,
    destination_storage_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind a future archive transaction without authorizing payload movement."""

    if hashlib.sha256(manifest).hexdigest() != manifest_file_sha256:
        raise D0Error("legacy archive manifest file hash differs")
    try:
        value = json.loads(manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("legacy archive manifest is invalid JSON") from error
    totals = value.get("totals") if isinstance(value, Mapping) else None
    freeze = value.get("freeze") if isinstance(value, Mapping) else None
    if (
        value.get("schema_id")
        != "cascadia.r2-map.john3-legacy-native-workspace-freeze.v1"
        or value.get("schema_version") != 1
        or value.get("host") != "john3"
        or value.get("root") != JOHN3_LEGACY_ROOT
        or not isinstance(totals, Mapping)
        or not isinstance(freeze, Mapping)
        or freeze.get("all_entries_immutable") is not True
        or freeze.get("two_post_freeze_scans_stable") is not True
        or freeze.get("project_execution_performed") is not False
        or not isinstance(value.get("manifest_sha256"), str)
        or len(value["manifest_sha256"]) != 64
    ):
        raise D0Error("legacy archive manifest identity differs")
    for field in ("entry_count", "unique_regular_bytes", "root_tree_sha256"):
        if field not in totals:
            raise D0Error("legacy archive manifest totals are incomplete")
    if destination_storage_receipt_sha256 is not None and (
        len(destination_storage_receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in destination_storage_receipt_sha256
        )
    ):
        raise D0Error("legacy archive destination receipt hash differs")
    ready = destination_storage_receipt_sha256 is not None
    plan: dict[str, Any] = {
        "schema_id": ARCHIVE_PLAN_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "operation_id": "john3-legacy-native-workspace-to-john2-cold-archive-v1",
        "state": (
            "ready-for-explicit-execution-authorization"
            if ready
            else "awaiting-target-receipt"
        ),
        "source": {
            "host": "john3",
            "root": JOHN3_LEGACY_ROOT,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_sha256": value["manifest_sha256"],
            "root_tree_sha256": totals["root_tree_sha256"],
            "entry_count": totals["entry_count"],
            "unique_regular_bytes": totals["unique_regular_bytes"],
            "all_entries_immutable": True,
        },
        "destination": {
            "host": "john2",
            "root": JOHN2_ARCHIVE_ROOT,
            "storage_receipt_sha256": destination_storage_receipt_sha256,
            "relative": (
                "john3/legacy-native-workspace/"
                f"{value['manifest_sha256']}/john3-r2-map-v1-legacy.tar"
            ),
        },
        "transaction": {
            "deterministic_archive_required": True,
            "preserve_modes_and_hardlinks": True,
            "no_follow_source_walk": True,
            "reject_specials_symlinks_and_path_escapes": True,
            "john2_atomic_commit_required": True,
            "john1_independent_reopen_required": True,
            "source_delete_after_reopen": False,
        },
        "authorization": {
            "payload_transfer_authorized": False,
            "source_mutation_authorized": False,
            "source_delete_authorized": False,
            "external_ssd_authorized": False,
            "john4_authorized": False,
            "authorized_by": None,
            "goal_sha256": None,
            "supersedes_plan_sha256": None,
        },
        "blockers": ([] if ready else ["john2-archive-storage-receipt-missing"]),
    }
    plan["plan_sha256"] = document_sha256(plan, "plan_sha256")
    return validate_legacy_archive_plan(plan)


def validate_legacy_archive_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "operation_id",
        "state",
        "source",
        "destination",
        "transaction",
        "authorization",
        "blockers",
        "plan_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise D0Error("legacy archive plan fields differ")
    source = value["source"]
    destination = value["destination"]
    transaction = value["transaction"]
    authorization = value["authorization"]
    if (
        value["schema_id"] != ARCHIVE_PLAN_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["operation_id"]
        != "john3-legacy-native-workspace-to-john2-cold-archive-v1"
        or not isinstance(source, Mapping)
        or source.get("host") != "john3"
        or source.get("root") != JOHN3_LEGACY_ROOT
        or not isinstance(destination, Mapping)
        or destination.get("host") != "john2"
        or destination.get("root") != JOHN2_ARCHIVE_ROOT
        or not isinstance(transaction, Mapping)
        or transaction.get("source_delete_after_reopen") is not False
        or not isinstance(authorization, Mapping)
        or value["plan_sha256"] != document_sha256(value, "plan_sha256")
    ):
        raise D0Error("legacy archive plan identity differs")
    expected_authorization_fields = {
        "payload_transfer_authorized",
        "source_mutation_authorized",
        "source_delete_authorized",
        "external_ssd_authorized",
        "john4_authorized",
        "authorized_by",
        "goal_sha256",
        "supersedes_plan_sha256",
    }
    if set(authorization) != expected_authorization_fields:
        raise D0Error("legacy archive authorization fields differ")
    receipt = destination.get("storage_receipt_sha256")
    blockers = value["blockers"]
    if receipt is None:
        if value["state"] != "awaiting-target-receipt" or blockers != [
            "john2-archive-storage-receipt-missing"
        ] or any(item is not False and item is not None for item in authorization.values()):
            raise D0Error("legacy archive plan missing-receipt state differs")
    elif value["state"] == "ready-for-explicit-execution-authorization":
        if blockers != [] or any(
            item is not False and item is not None for item in authorization.values()
        ):
            raise D0Error("legacy archive plan ready state differs")
    elif value["state"] == "payload-transfer-authorized":
        for field in (
            "source_mutation_authorized",
            "source_delete_authorized",
            "external_ssd_authorized",
            "john4_authorized",
        ):
            if authorization.get(field) is not False:
                raise D0Error("legacy archive authorization exceeds payload transfer")
        for field in ("goal_sha256", "supersedes_plan_sha256"):
            digest = authorization.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise D0Error("legacy archive authorization digest differs")
        if (
            blockers != []
            or authorization.get("payload_transfer_authorized") is not True
            or authorization.get("authorized_by") != "root-orchestrator"
        ):
            raise D0Error("legacy archive payload authorization differs")
    else:
        raise D0Error("legacy archive plan state differs")
    return dict(value)


def encode_legacy_archive_plan(value: Mapping[str, Any]) -> bytes:
    return canonical_json(validate_legacy_archive_plan(value))


def authorize_legacy_archive_transfer(
    value: Mapping[str, Any],
    *,
    goal_sha256: str,
) -> dict[str, Any]:
    """Authorize only payload transfer; source mutation and deletion stay forbidden."""

    current = validate_legacy_archive_plan(value)
    if current["state"] != "ready-for-explicit-execution-authorization":
        raise D0Error("legacy archive plan is not ready for payload authorization")
    if len(goal_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in goal_sha256
    ):
        raise D0Error("legacy archive goal hash differs")
    authorized = json.loads(canonical_json(current))
    authorized["state"] = "payload-transfer-authorized"
    authorized["authorization"].update(
        {
            "payload_transfer_authorized": True,
            "authorized_by": "root-orchestrator",
            "goal_sha256": goal_sha256,
            "supersedes_plan_sha256": current["plan_sha256"],
        }
    )
    authorized["plan_sha256"] = document_sha256(authorized, "plan_sha256")
    return validate_legacy_archive_plan(authorized)
