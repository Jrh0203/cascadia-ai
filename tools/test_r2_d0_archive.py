from __future__ import annotations

import hashlib
import json

import pytest
from r2_d0.archive import (
    authorize_legacy_archive_transfer,
    build_john3_legacy_archive_plan,
    validate_legacy_archive_plan,
)
from r2_d0.canonical import D0Error


def manifest() -> bytes:
    return json.dumps(
        {
            "schema_id": "cascadia.r2-map.john3-legacy-native-workspace-freeze.v1",
            "schema_version": 1,
            "host": "john3",
            "root": "/Users/john3/cascadia-bench/r2-map-v1",
            "manifest_sha256": "a" * 64,
            "freeze": {
                "all_entries_immutable": True,
                "two_post_freeze_scans_stable": True,
                "project_execution_performed": False,
            },
            "totals": {
                "entry_count": 7052,
                "unique_regular_bytes": 978085576,
                "root_tree_sha256": "b" * 64,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_missing_target_receipt_produces_nonexecutable_plan() -> None:
    payload = manifest()
    plan = build_john3_legacy_archive_plan(
        payload,
        manifest_file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert plan["state"] == "awaiting-target-receipt"
    assert plan["blockers"] == ["john2-archive-storage-receipt-missing"]
    assert all(item is False or item is None for item in plan["authorization"].values())
    assert plan["transaction"]["source_delete_after_reopen"] is False


def test_target_receipt_only_makes_plan_ready_for_separate_authorization() -> None:
    payload = manifest()
    plan = build_john3_legacy_archive_plan(
        payload,
        manifest_file_sha256=hashlib.sha256(payload).hexdigest(),
        destination_storage_receipt_sha256="c" * 64,
    )
    assert plan["state"] == "ready-for-explicit-execution-authorization"
    assert plan["blockers"] == []
    assert plan["authorization"]["payload_transfer_authorized"] is False

    authorized = authorize_legacy_archive_transfer(plan, goal_sha256="d" * 64)
    assert authorized["state"] == "payload-transfer-authorized"
    assert authorized["authorization"]["payload_transfer_authorized"] is True
    assert authorized["authorization"]["source_mutation_authorized"] is False
    assert authorized["authorization"]["source_delete_authorized"] is False
    assert authorized["authorization"]["supersedes_plan_sha256"] == plan["plan_sha256"]


def test_hash_and_authorization_tamper_fail_closed() -> None:
    payload = manifest()
    with pytest.raises(D0Error, match="file hash"):
        build_john3_legacy_archive_plan(payload, manifest_file_sha256="0" * 64)
    plan = build_john3_legacy_archive_plan(
        payload,
        manifest_file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    plan["authorization"]["payload_transfer_authorized"] = True
    with pytest.raises(D0Error, match="identity differs"):
        validate_legacy_archive_plan(plan)
