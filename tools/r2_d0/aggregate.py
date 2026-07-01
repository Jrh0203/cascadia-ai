"""Final signed D0 aggregate over the complete two-cycle transaction graph."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .bundle import verify_result_bundle
from .canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    INSTALL_OPERATIONS_BY_HOST,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    sha256_bytes,
)
from .closure import validate_materialization_receipt, verify_bootstrap_record
from .signing import (
    normalize_public_key,
    public_key_fingerprint,
    verify_stdin,
)

AGGREGATE_SCHEMA = "cascadia.r2-map.d0-final-aggregate.v3"
AGGREGATE_MAX_BYTES = 4 * 1024 * 1024
HELPER_TRANSITION_SCHEMA = "cascadia.r2-map.d0-helper-transition.v1"
FINALIZED_HELPER_TRANSITION_SCHEMA = "cascadia.r2-map.d0-helper-transition-finalization.v1"

REQUIRED_TOPOLOGY_IDENTITIES = {
    ("cascadia.r2-map.local-runtime-profile-receipt.v4", "john1"),
    ("cascadia.r2-map.local-runtime-profile-receipt.v3", "john2"),
    ("cascadia.r2-map.local-runtime-profile-receipt.v4", "john3"),
    ("cascadia.r2-map.cold-archive-root-receipt.v1", "john2"),
    ("cascadia.r2-map.legacy-dashboard-termination-receipt.v1", "john2"),
    ("cascadia.r2-map.john1-cold-archive-reopen.v1", "john1"),
    ("cascadia.r2-map.john2-cold-archive-commit.v1", "john2"),
    ("cascadia.r2-map.john3-legacy-cleanup-receipt.v1", "john3"),
}


def validate_topology_receipts(receipts: Sequence[bytes]) -> list[dict[str, Any]]:
    """Validate the live role, archive, cleanup, and direct-control barrier."""

    indexed: dict[tuple[str, str], tuple[dict[str, Any], bytes]] = {}
    for payload in receipts:
        if len(payload) > 4 * 1024 * 1024:
            raise D0Error("D0 topology receipt exceeds its byte limit")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0Error("D0 topology receipt is not valid JSON") from error
        if not isinstance(value, dict) or payload not in {
            canonical_json(value),
            canonical_json(value) + b"\n",
        }:
            raise D0Error("D0 topology receipt is not canonical JSON")
        schema = value.get("schema_id")
        host = value.get("host")
        if schema == "cascadia.r2-map.john3-legacy-cleanup-receipt.v1":
            host = "john3"
        elif schema == "cascadia.r2-map.john1-cold-archive-reopen.v1":
            host = "john1"
        elif schema == "cascadia.r2-map.john2-cold-archive-commit.v1":
            host = "john2"
        identity = (str(schema), str(host))
        if identity in indexed:
            raise D0Error("D0 topology receipt identity is duplicated")
        indexed[identity] = (value, payload)
    if set(indexed) != REQUIRED_TOPOLOGY_IDENTITIES:
        raise D0Error("D0 topology receipt set has missing or extra identities")
    runtime = {
        "john1": indexed[("cascadia.r2-map.local-runtime-profile-receipt.v4", "john1")][0],
        "john2": indexed[("cascadia.r2-map.local-runtime-profile-receipt.v3", "john2")][0],
        "john3": indexed[("cascadia.r2-map.local-runtime-profile-receipt.v4", "john3")][0],
    }
    for host, receipt in runtime.items():
        certification = receipt.get("certification")
        if (
            receipt.get("host") != host
            or not isinstance(certification, Mapping)
            or certification.get("project_execution_authorized") is not False
            or certification.get("d0_certified") is not False
            or (
                certification.get("campaign_execution_blocked") is True
                and not (
                    certification.get("host_role_qualified") is True
                    and certification.get("blocker") == "Signed D0 topology aggregate is pending."
                )
            )
        ):
            raise D0Error(f"D0 {host} live role receipt is blocked or differs")
    john2_buildx = runtime["john2"].get("buildx")
    if not isinstance(john2_buildx, Mapping) or not john2_buildx.get("version"):
        raise D0Error("D0 John2 sole-buildx role is unproven")
    for host in ("john1", "john3"):
        buildx = runtime[host].get("buildx")
        if not isinstance(buildx, Mapping):
            buildx = runtime[host].get("runtime", {}).get("buildx_absence")
        if not isinstance(buildx, Mapping) or buildx.get("installed") is not False:
            raise D0Error(f"D0 {host} execution-only buildx absence is unproven")
    archive = indexed[("cascadia.r2-map.cold-archive-root-receipt.v1", "john2")][0]
    cleanup = indexed[("cascadia.r2-map.john3-legacy-cleanup-receipt.v1", "john3")][0]
    reopen = indexed[("cascadia.r2-map.john1-cold-archive-reopen.v1", "john1")][0]
    commit = indexed[("cascadia.r2-map.john2-cold-archive-commit.v1", "john2")][0]
    if (
        archive.get("archive_root", {}).get("path")
        != "/Users/john2/cascadia-bench/r2-map-archive-v1"
        or archive.get("authority", {}).get("active_artifact_authority")
        != "john1:/Users/johnherrick/cascadia-bench/r2-map-v1"
        or cleanup.get("status") != "pass"
        or cleanup.get("source_root_absent") is not True
        or cleanup.get("deleted_entry_count") != 7052
        or cleanup.get("deleted_unique_regular_bytes") != 978085576
        or reopen.get("status") != "pass"
        or commit.get("status") != "pass"
        or cleanup.get("john1_reopen_receipt_sha256") != reopen.get("receipt_sha256")
        or cleanup.get("john2_commit_receipt_sha256") != commit.get("receipt_sha256")
        or cleanup.get("archive_sha256") != reopen.get("verification", {}).get("archive_sha256")
        or cleanup.get("archive_sha256") != commit.get("archive_sha256")
    ):
        raise D0Error("D0 cold archive or John3 cleanup closure differs")
    return [
        {
            "schema_id": schema,
            "host": host,
            "payload_size": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "receipt_sha256": value.get("receipt_sha256"),
            "status": "pass",
        }
        for (schema, host), (value, payload) in sorted(indexed.items())
    ]


def _expected_transaction_keys() -> tuple[tuple[str, str, str, str], ...]:
    result: list[tuple[str, str, str, str]] = []
    for cycle in ("qualification", "final-live"):
        for host in ("john1", "john2", "john3"):
            result.extend(((cycle, host, "preflight", "preflight-audit"),))
            result.extend(
                (cycle, host, "install", item) for item in INSTALL_OPERATIONS_BY_HOST[host]
            )
            result.extend(
                (
                    (cycle, host, "start", "start-runtime"),
                    (cycle, host, "verify", "verify-runtime"),
                )
            )
            if cycle == "qualification":
                result.extend(
                    (
                        (cycle, host, "rollback", "rollback-runtime"),
                        (cycle, host, "postflight", "postflight-audit"),
                    )
                )
    return tuple(result)


EXPECTED_TRANSACTION_KEYS = frozenset(_expected_transaction_keys())


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise D0Error(f"D0 helper transition {label} differs")
    try:
        int(value, 16)
    except ValueError as error:
        raise D0Error(f"D0 helper transition {label} differs") from error
    return value


def _validate_base_helper_transition(value: Any) -> dict[str, Any]:
    """Validate one signed helper transition's exact, ordered closure."""

    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "chain_index",
        "from_plan_sha256",
        "from_plan_file_sha256",
        "from_helper_sha256",
        "to_plan_sha256",
        "to_plan_file_sha256",
        "to_helper_sha256",
        "collision_incident_sha256",
        "collision_incident_file_sha256",
        "accepted_transactions",
        "migration_authorizations",
        "migration_receipts",
        "old_bootstraps",
        "new_bootstraps",
        "project_code_executed",
        "protected_seed_values_opened",
        "transition_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("D0 helper transition fields differ")
    if (
        value["schema_id"] != HELPER_TRANSITION_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or not isinstance(value["chain_index"], int)
        or isinstance(value["chain_index"], bool)
        or value["chain_index"] <= 0
        or value["project_code_executed"] is not False
        or value["protected_seed_values_opened"] is not False
        or value["transition_sha256"] != document_sha256(value, "transition_sha256")
    ):
        raise D0Error("D0 helper transition identity differs")
    for field in (
        "from_plan_sha256",
        "from_plan_file_sha256",
        "from_helper_sha256",
        "to_plan_sha256",
        "to_plan_file_sha256",
        "to_helper_sha256",
        "collision_incident_sha256",
        "collision_incident_file_sha256",
    ):
        _sha256(value[field], field)
    transaction_fields = {
        "sequence",
        "cycle_id",
        "host",
        "phase",
        "operation",
        "packet_sha256",
        "report_sha256",
        "bundle_sha256",
        "bundle_size",
        "manifest_sha256",
        "finished_unix_ms",
    }
    accepted = value["accepted_transactions"]
    if not isinstance(accepted, list):
        raise D0Error("D0 helper transition accepted transaction set differs")
    for index, item in enumerate(accepted, 1):
        if (
            not isinstance(item, dict)
            or set(item) != transaction_fields
            or item["sequence"] != index
            or item["cycle_id"] not in {"qualification", "final-live"}
            or item["host"] not in {"john1", "john2", "john3"}
            or not isinstance(item["phase"], str)
            or not isinstance(item["operation"], str)
            or not isinstance(item["bundle_size"], int)
            or item["bundle_size"] <= 0
            or not isinstance(item["finished_unix_ms"], int)
            or item["finished_unix_ms"] <= 0
        ):
            raise D0Error("D0 helper transition accepted transaction order differs")
        for field in ("packet_sha256", "report_sha256", "bundle_sha256", "manifest_sha256"):
            _sha256(item[field], f"accepted transaction {field}")
    authorization_fields = {
        "host",
        "authorization_sha256",
        "authorization_file_sha256",
        "signature_file_sha256",
    }
    receipt_fields = {
        "host",
        "receipt_sha256",
        "receipt_file_sha256",
        "authorization_sha256",
        "old_helper_sha256",
        "new_helper_sha256",
        "old_bootstrap_receipt_sha256",
        "new_bootstrap_receipt_sha256",
        "finished_unix_ms",
    }
    bootstrap_fields = {
        "host",
        "record_sha256",
        "record_payload_sha256",
        "record_signature_bundle_sha256",
        "bootstrap_receipt_sha256",
        "helper_archive_sha256",
        "installed_unix_ms",
    }

    def host_rows(name: str, fields: set[str]) -> list[dict[str, Any]]:
        rows = value[name]
        if (
            not isinstance(rows, list)
            or [item.get("host") for item in rows if isinstance(item, dict)]
            != ["john1", "john2", "john3"]
            or any(not isinstance(item, dict) or set(item) != fields for item in rows)
        ):
            raise D0Error(f"D0 helper transition {name} host order differs")
        return rows

    authorizations = host_rows("migration_authorizations", authorization_fields)
    receipts = host_rows("migration_receipts", receipt_fields)
    old_bootstraps = host_rows("old_bootstraps", bootstrap_fields)
    new_bootstraps = host_rows("new_bootstraps", bootstrap_fields)
    for rows in (authorizations, receipts, old_bootstraps, new_bootstraps):
        for item in rows:
            for field, scalar in item.items():
                if field == "host" or field == "finished_unix_ms" or field == "installed_unix_ms":
                    continue
                _sha256(scalar, f"{field}")
    for authorization, receipt, old, new in zip(  # noqa: B905 -- Python 3.9.
        authorizations, receipts, old_bootstraps, new_bootstraps
    ):
        if (
            receipt["authorization_sha256"] != authorization["authorization_sha256"]
            or receipt["old_helper_sha256"] != value["from_helper_sha256"]
            or receipt["new_helper_sha256"] != value["to_helper_sha256"]
            or receipt["old_bootstrap_receipt_sha256"] != old["bootstrap_receipt_sha256"]
            or receipt["new_bootstrap_receipt_sha256"] != new["bootstrap_receipt_sha256"]
            or old["helper_archive_sha256"] != value["from_helper_sha256"]
            or new["helper_archive_sha256"] != value["to_helper_sha256"]
            or not isinstance(receipt["finished_unix_ms"], int)
            or not (
                old["installed_unix_ms"] < new["installed_unix_ms"] <= receipt["finished_unix_ms"]
            )
        ):
            raise D0Error("D0 helper transition migration/bootstrap binding differs")
    return value


def _validate_finalized_helper_transition(value: Any) -> dict[str, Any]:
    """Validate the one terminal, signed closure of a provisional transition.

    The migration lineage is deliberately a provisional lower-bound snapshot.
    A finalization can add exact old-helper transactions that were omitted from
    that snapshot, but only inside the source host's old-bootstrap-to-rotation
    cutoff.  There is no amendment array: the finalization is terminal.
    """

    base_fields = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "chain_index",
        "from_plan_sha256",
        "from_plan_file_sha256",
        "from_helper_sha256",
        "to_plan_sha256",
        "to_plan_file_sha256",
        "to_helper_sha256",
        "collision_incident_sha256",
        "collision_incident_file_sha256",
        "accepted_transactions",
        "migration_authorizations",
        "migration_receipts",
        "old_bootstraps",
        "new_bootstraps",
        "project_code_executed",
        "protected_seed_values_opened",
        "transition_sha256",
    }
    required = base_fields | {
        "base_transition_sha256",
        "base_transition_file_sha256",
        "base_transition_signature_file_sha256",
        "previous_transition_sha256",
        "base_accepted_transaction_count",
        "tail_transaction_count",
        "tail_transactions",
        "collision_incident_signature_file_sha256",
        "migration_receipt_cutoffs",
        "terminal",
        "finalized_unix_ms",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("D0 finalized helper transition fields differ")
    if (
        value["schema_id"] != FINALIZED_HELPER_TRANSITION_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["terminal"] is not True
        or value["project_code_executed"] is not False
        or value["protected_seed_values_opened"] is not False
        or value["transition_sha256"] != document_sha256(value, "transition_sha256")
        or not isinstance(value["finalized_unix_ms"], int)
        or isinstance(value["finalized_unix_ms"], bool)
        or value["finalized_unix_ms"] <= 0
    ):
        raise D0Error("D0 finalized helper transition identity differs")
    for field in (
        "base_transition_sha256",
        "base_transition_file_sha256",
        "base_transition_signature_file_sha256",
        "previous_transition_sha256",
        "collision_incident_signature_file_sha256",
    ):
        _sha256(value[field], field)

    # Reuse the frozen v1 structural contract for the effective transition.
    projection = {field: value[field] for field in base_fields}
    projection["schema_id"] = HELPER_TRANSITION_SCHEMA
    projection["schema_version"] = 1
    projection["transition_sha256"] = document_sha256(projection, "transition_sha256")
    _validate_base_helper_transition(projection)

    accepted = value["accepted_transactions"]
    base_count = value["base_accepted_transaction_count"]
    tail_count = value["tail_transaction_count"]
    if (
        not isinstance(base_count, int)
        or isinstance(base_count, bool)
        or base_count < 0
        or not isinstance(tail_count, int)
        or isinstance(tail_count, bool)
        or tail_count < 0
        or base_count + tail_count != len(accepted)
    ):
        raise D0Error("D0 finalized helper transition transaction counts differ")

    tail_fields = {
        "sequence",
        "cycle_id",
        "host",
        "phase",
        "operation",
        "packet_sha256",
        "report_sha256",
        "bundle_sha256",
        "bundle_size",
        "manifest_sha256",
        "finished_unix_ms",
        "canonical_receipt_sha256",
        "canonical_receipt_file_sha256",
        "target_host",
        "target_receipt_sha256",
        "target_receipt_file_sha256",
        "source_helper_sha256",
    }
    tail = value["tail_transactions"]
    if not isinstance(tail, list) or len(tail) != tail_count:
        raise D0Error("D0 finalized helper transition tail count differs")
    transaction_fields = {
        "sequence",
        "cycle_id",
        "host",
        "phase",
        "operation",
        "packet_sha256",
        "report_sha256",
        "bundle_sha256",
        "bundle_size",
        "manifest_sha256",
        "finished_unix_ms",
    }
    seen_keys: set[tuple[str, str, str, str]] = set()
    seen_identities: set[tuple[str, str, str]] = set()
    for item in accepted[:base_count]:
        seen_keys.add((item["cycle_id"], item["host"], item["phase"], item["operation"]))
        seen_identities.add((item["packet_sha256"], item["report_sha256"], item["bundle_sha256"]))
    for offset, item in enumerate(tail, base_count + 1):
        if (
            not isinstance(item, dict)
            or set(item) != tail_fields
            or item["sequence"] != offset
            or item["target_host"] != item["host"]
            or item["source_helper_sha256"] != value["from_helper_sha256"]
        ):
            raise D0Error("D0 finalized helper transition tail order differs")
        for field in (
            "canonical_receipt_sha256",
            "canonical_receipt_file_sha256",
            "target_receipt_sha256",
            "target_receipt_file_sha256",
            "source_helper_sha256",
        ):
            _sha256(item[field], f"tail transaction {field}")
        projected = {field: item[field] for field in transaction_fields}
        if projected != accepted[offset - 1]:
            raise D0Error("D0 finalized helper transition tail projection differs")
        key = (item["cycle_id"], item["host"], item["phase"], item["operation"])
        identity = (item["packet_sha256"], item["report_sha256"], item["bundle_sha256"])
        if key in seen_keys or identity in seen_identities:
            raise D0Error("D0 finalized helper transition tail is duplicated")
        seen_keys.add(key)
        seen_identities.add(identity)

    cutoff_fields = {
        "host",
        "migration_receipt_sha256",
        "migration_receipt_file_sha256",
        "old_bootstrap_installed_unix_ms",
        "new_bootstrap_installed_unix_ms",
        "rotation_finished_unix_ms",
    }
    cutoffs = value["migration_receipt_cutoffs"]
    if (
        not isinstance(cutoffs, list)
        or [item.get("host") for item in cutoffs if isinstance(item, dict)]
        != ["john1", "john2", "john3"]
        or any(not isinstance(item, dict) or set(item) != cutoff_fields for item in cutoffs)
    ):
        raise D0Error("D0 finalized helper transition cutoff host order differs")
    receipts = {item["host"]: item for item in value["migration_receipts"]}
    old_bootstraps = {item["host"]: item for item in value["old_bootstraps"]}
    new_bootstraps = {item["host"]: item for item in value["new_bootstraps"]}
    by_host: dict[str, dict[str, Any]] = {}
    for item in cutoffs:
        for field in ("migration_receipt_sha256", "migration_receipt_file_sha256"):
            _sha256(item[field], f"cutoff {field}")
        receipt = receipts[item["host"]]
        old = old_bootstraps[item["host"]]
        new = new_bootstraps[item["host"]]
        if (
            item["migration_receipt_sha256"] != receipt["receipt_sha256"]
            or item["migration_receipt_file_sha256"] != receipt["receipt_file_sha256"]
            or item["old_bootstrap_installed_unix_ms"] != old["installed_unix_ms"]
            or item["new_bootstrap_installed_unix_ms"] != new["installed_unix_ms"]
            or item["rotation_finished_unix_ms"] != receipt["finished_unix_ms"]
            or not (
                item["old_bootstrap_installed_unix_ms"]
                < item["new_bootstrap_installed_unix_ms"]
                <= item["rotation_finished_unix_ms"]
                <= value["finalized_unix_ms"]
            )
        ):
            raise D0Error("D0 finalized helper transition cutoff binding differs")
        by_host[item["host"]] = item
    for item in accepted:
        cutoff = by_host[item["host"]]
        if not (
            cutoff["old_bootstrap_installed_unix_ms"]
            < item["finished_unix_ms"]
            <= cutoff["new_bootstrap_installed_unix_ms"]
        ):
            raise D0Error("D0 finalized helper transition transaction is outside cutoff")
    return value


def validate_helper_transition(value: Any) -> dict[str, Any]:
    """Validate a provisional v1 transition or its terminal finalization."""

    if isinstance(value, dict) and value.get("schema_id") == FINALIZED_HELPER_TRANSITION_SCHEMA:
        return _validate_finalized_helper_transition(value)
    return _validate_base_helper_transition(value)


def verify_helper_transitions(
    transitions: Sequence[tuple[bytes, Mapping[str, Any]]],
    *,
    public_key: bytes,
) -> list[dict[str, Any]]:
    """Verify signatures and exact ordering for the helper-transition chain."""

    verified: list[dict[str, Any]] = []
    for payload, signature in transitions:
        value = validate_helper_transition(
            load_canonical_json(
                payload,
                maximum=AGGREGATE_MAX_BYTES,
                label="D0 helper transition",
            )
        )
        verify_stdin(public_key, payload, dict(signature))
        verified.append(value)
    if [item["chain_index"] for item in verified] != list(range(1, len(verified) + 1)):
        raise D0Error("D0 helper transition chain order differs")
    for index in range(len(verified) - 1):
        before, after = verified[index], verified[index + 1]
        if (
            before["to_helper_sha256"] != after["from_helper_sha256"]
            or before["to_plan_sha256"] != after["from_plan_sha256"]
            or before["to_plan_file_sha256"] != after["from_plan_file_sha256"]
            or (
                after["schema_id"] == FINALIZED_HELPER_TRANSITION_SCHEMA
                and after["previous_transition_sha256"] != before["transition_sha256"]
            )
        ):
            raise D0Error("D0 helper transition chain continuity differs")
    return verified


def _validate_helper_history(
    transactions: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    bootstraps: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Admit mixed helpers only through an exact signed transition chain."""

    helpers = {packet["helper_sha256"] for packet, _report, _record in transactions}
    current_helpers = {item["helper_archive_sha256"] for item in bootstraps}
    if len(current_helpers) != 1:
        raise D0Error("D0 aggregate current bootstrap helper differs by host")
    current_helper = next(iter(current_helpers))
    if not transitions:
        if helpers != {current_helper}:
            raise D0Error("D0 aggregate contains an unlisted mixed-helper packet")
        return []
    if transitions[-1]["to_helper_sha256"] != current_helper:
        raise D0Error("D0 aggregate helper transition does not reach current bootstrap")
    allowed = {transitions[0]["from_helper_sha256"]} | {
        item["to_helper_sha256"] for item in transitions
    }
    if helpers - allowed:
        raise D0Error("D0 aggregate contains an unlisted mixed-helper packet")
    actual_by_helper: dict[str, list[dict[str, Any]]] = {}
    for packet, report, record in transactions:
        actual_by_helper.setdefault(packet["helper_sha256"], []).append(
            {
                "cycle_id": packet["cycle_id"],
                "host": packet["host"],
                "phase": report["phase"],
                "operation": report["operation"],
                "packet_sha256": packet["packet_sha256"],
                "report_sha256": report["report_sha256"],
                "bundle_sha256": record["archive_sha256"],
                "bundle_size": record["archive_size"],
                "manifest_sha256": record["manifest_sha256"],
                "finished_unix_ms": record["finished_unix_ms"],
            }
        )
    summaries: list[dict[str, Any]] = []
    for transition in transitions:
        expected = [
            {key: value for key, value in item.items() if key != "sequence"}
            for item in transition["accepted_transactions"]
        ]
        actual = actual_by_helper.get(transition["from_helper_sha256"], [])
        actual.sort(key=lambda item: expected.index(item) if item in expected else len(expected))
        if actual != expected:
            raise D0Error("D0 aggregate accepted old-helper transaction set differs")
        old_bootstrap_by_host = {item["host"]: item for item in transition["old_bootstraps"]}
        for packet, _report, record in transactions:
            if (
                packet["helper_sha256"] == transition["from_helper_sha256"]
                and old_bootstrap_by_host[packet["host"]]["installed_unix_ms"]
                > record["started_unix_ms"]
            ):
                raise D0Error("D0 aggregate old bootstrap was installed after work began")
        current_bootstrap_projection = [
            {
                key: item[key]
                for key in (
                    "host",
                    "record_sha256",
                    "record_payload_sha256",
                    "record_signature_bundle_sha256",
                    "bootstrap_receipt_sha256",
                    "helper_archive_sha256",
                    "installed_unix_ms",
                )
            }
            for item in sorted(bootstraps, key=lambda row: row["host"])
        ]
        if (
            transition is transitions[-1]
            and transition["new_bootstraps"] != current_bootstrap_projection
        ):
            raise D0Error("D0 aggregate current bootstrap records differ from transition")
        for receipt in transition["migration_receipts"]:
            first_new = min(
                (
                    record["started_unix_ms"]
                    for packet, _report, record in transactions
                    if packet["host"] == receipt["host"]
                    and packet["helper_sha256"] == transition["to_helper_sha256"]
                ),
                default=None,
            )
            if first_new is not None and receipt["finished_unix_ms"] > first_new:
                raise D0Error("D0 aggregate helper migration finished after new-helper work")
        summaries.append(
            {
                "chain_index": transition["chain_index"],
                "transition_sha256": transition["transition_sha256"],
                "from_helper_sha256": transition["from_helper_sha256"],
                "to_helper_sha256": transition["to_helper_sha256"],
                "accepted_transaction_count": len(expected),
            }
        )
    if any(
        helper != current_helper
        and helper not in {item["from_helper_sha256"] for item in transitions}
        for helper in helpers
    ):
        raise D0Error("D0 aggregate helper transition history is incomplete")
    return summaries


def _key(packet: Mapping[str, Any], report: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(packet.get("cycle_id")),
        str(packet.get("host")),
        str(report.get("phase")),
        str(report.get("operation")),
    )


def _pass(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("status") not in {"pass", "installed"}:
        raise D0Error(f"D0 aggregate {label} evidence did not pass")
    return value


def validate_operation_evidence(packet: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Recompute critical pass semantics instead of trusting a report status bit."""

    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise D0Error("D0 aggregate report evidence is absent")
    phase_resources = evidence.get("phase_resources")
    if (
        not isinstance(phase_resources, Mapping)
        or phase_resources.get("status") != "pass"
        or phase_resources.get("zero_swap_entire_phase") is not True
        or any(
            not isinstance(phase_resources.get(boundary), Mapping)
            or phase_resources[boundary].get("swap_used_bytes") != 0
            for boundary in ("before", "after")
        )
    ):
        raise D0Error("D0 aggregate phase-wide zero-swap evidence differs")
    continuous_swap = phase_resources.get("continuous_swap")
    if (
        not isinstance(continuous_swap, Mapping)
        or continuous_swap.get("status") != "pass"
        or not isinstance(continuous_swap.get("sample_count"), int)
        or continuous_swap.get("sample_count", 0) < 1
        or continuous_swap.get("nonzero_samples") != 0
        or continuous_swap.get("max_used_bytes") != 0
        or not isinstance(continuous_swap.get("sample_stream_sha256"), str)
    ):
        raise D0Error("D0 aggregate continuous zero-swap evidence differs")
    operation = report["operation"]
    status = report["status"]
    if status == "fail":
        raise D0Error("D0 aggregate contains a failed transaction")
    if operation == "preflight-audit":
        _pass(evidence.get("platform"), "runtime-host platform")
        _pass(evidence.get("runtime_budget_preflight"), "runtime budget")
        if (
            evidence.get("runtime_activity", {}).get("inactive") is not True
            or evidence.get("resources", {}).get("swap_used_bytes") != 0
        ):
            raise D0Error("D0 aggregate runtime-host preflight is not quiescent")
        if packet["host"] == "john1":
            podman = _pass(evidence.get("podman_negative_control"), "Podman negative control")
            semantic = podman.get("semantic")
            if (
                not isinstance(semantic, Mapping)
                or semantic.get("machine_records") != 0
                or semantic.get("machine_disks") != 0
                or semantic.get("socket_entries") != 0
                or semantic.get("storage_payload_files") != 0
            ):
                raise D0Error("D0 aggregate Podman machine/storage state is present")
    elif operation == "acquire-core":
        core = evidence.get("core_image")
        expected = packet["artifacts"]["core_image"]
        if not isinstance(core, Mapping) or any(
            core.get(field) != expected[field] for field in ("size", "sha256")
        ):
            raise D0Error("D0 aggregate Colima-core evidence differs")
    elif operation == "acquire-smoke":
        installed = evidence.get("installed")
        rendered = evidence.get("smoke_oci")
        expected = packet["artifacts"]["smoke_oci"]
        if (
            not isinstance(installed, Mapping)
            or not isinstance(rendered, Mapping)
            or not isinstance(installed.get("size"), int)
            or installed["size"] <= 0
            or not isinstance(installed.get("sha256"), str)
            or len(installed["sha256"]) != 64
            or rendered.get("archive_bytes") != installed["size"]
            or rendered.get("archive_sha256") != installed["sha256"]
            or (
                expected is not None
                and any(installed.get(field) != expected[field] for field in ("size", "sha256"))
            )
        ):
            raise D0Error("D0 aggregate Alpine OCI evidence differs")
    elif operation == "acquire-scanner":
        supply = _pass(evidence.get("scanner_supply"), "scanner supply")
        installed = supply.get("installed", {}).get("oci")
        rendered = supply.get("oci")
        expected = packet["artifacts"]["scanner_oci"]
        if (
            not isinstance(installed, Mapping)
            or not isinstance(rendered, Mapping)
            or not isinstance(installed.get("size"), int)
            or installed["size"] <= 0
            or not isinstance(installed.get("sha256"), str)
            or len(installed["sha256"]) != 64
            or rendered.get("archive_size") != installed["size"]
            or rendered.get("archive_sha256") != installed["sha256"]
            or (
                expected is not None
                and any(installed.get(field) != expected[field] for field in ("size", "sha256"))
            )
        ):
            raise D0Error("D0 aggregate scanner OCI evidence differs")
    elif operation == "acquire-homebrew-artifacts":
        _pass(evidence, "Homebrew acquisition")
        observed_formulae = evidence.get("formulae")
        expected_formulae = [item["name"] for item in packet["artifacts"]["bottles"]]
        if (
            not isinstance(observed_formulae, list)
            or len(observed_formulae) != len(expected_formulae)
            or any(not isinstance(item, str) for item in observed_formulae)
            or len(set(observed_formulae)) != len(observed_formulae)
            or set(observed_formulae) != set(expected_formulae)
        ):
            raise D0Error("D0 aggregate Homebrew closure formulae differ")
    elif operation == "render-runtime-supply":
        supply = _pass(evidence.get("runtime_supply"), "runtime supply render")
        installed = evidence.get("supply_install")
        artifact = packet["artifacts"]["runtime_supply"]
        if (
            not isinstance(installed, Mapping)
            or supply.get("archive_size") != installed.get("size")
            or supply.get("archive_sha256") != installed.get("sha256")
            or (
                artifact is not None
                and (
                    supply.get("archive_size") != artifact["size"]
                    or supply.get("archive_sha256") != artifact["sha256"]
                )
            )
        ):
            raise D0Error("D0 aggregate runtime-supply render identity differs")
        _pass(evidence.get("homebrew_closure"), "runtime-supply Homebrew closure")
    elif operation == "materialize-runtime-supply":
        materialized = _pass(evidence.get("runtime_supply"), "runtime supply materialization")
        verification = materialized.get("verification")
        artifact = packet["artifacts"]["runtime_supply"]
        if (
            artifact is None
            or not isinstance(verification, Mapping)
            or verification.get("archive_size") != artifact["size"]
            or verification.get("archive_sha256") != artifact["sha256"]
            or materialized.get("transaction") not in {"atomically-committed", "replayed-exact"}
        ):
            raise D0Error("D0 aggregate runtime-supply materialization differs")
        ingress = _pass(evidence.get("direct_ingress"), "runtime-supply direct ingress")
        expected_source = "john2" if packet["host"] == "john1" else "john1"
        if (
            ingress.get("source_host") != expected_source
            or ingress.get("target_host") != packet["host"]
            or ingress.get("size") != artifact["size"]
            or ingress.get("sha256") != artifact["sha256"]
            or ingress.get("peer_credentials_present") is not False
        ):
            raise D0Error("D0 aggregate direct runtime-supply ingress differs")
    elif operation == "render-probe-context":
        installed = evidence.get("installed")
        artifact_key = "probe_context"
        expected = packet["artifacts"][artifact_key]
        if (
            not isinstance(installed, Mapping)
            or expected is None
            or any(installed.get(field) != expected[field] for field in ("size", "sha256"))
        ):
            raise D0Error(f"D0 aggregate {operation} evidence differs")
    elif operation == "install-runtime":
        _pass(evidence.get("homebrew"), "runtime install")
        configs = evidence.get("configs")
        if not isinstance(configs, Mapping) or not configs.get("colima_sha256"):
            raise D0Error("D0 aggregate runtime config evidence is absent")
    elif operation == "start-runtime":
        _pass(evidence, "runtime start")
        core = evidence.get("core_image")
        expected = packet["artifacts"]["core_image"]
        if not isinstance(core, Mapping) or any(
            core.get(field) != expected[field] for field in ("size", "sha256")
        ):
            raise D0Error("D0 aggregate start core identity differs")
    elif operation == "verify-runtime":
        _pass(evidence, "runtime verification")
        for field in ("colima", "socket", "engine", "engine_info", "buildkit", "guest"):
            if not isinstance(evidence.get(field), Mapping):
                raise D0Error(f"D0 aggregate runtime {field} evidence is absent")
        _pass(evidence.get("homebrew_comparison"), "Homebrew positive delta")
        _pass(evidence.get("budget"), "runtime footprint budget")
        smoke = evidence.get("smoke_image", {}).get("roundtrip")
        if not isinstance(smoke, Mapping) or smoke.get("cleanup") != "complete":
            raise D0Error("D0 aggregate smoke cleanup is incomplete")
        recovery = evidence.get("stop_start_recovery")
        context = evidence.get("docker_context")
        named_context = context.get("named_context") if isinstance(context, Mapping) else None
        guest = evidence.get("guest")
        if (
            not isinstance(recovery, Mapping)
            or recovery.get("status") != "pass"
            or recovery.get("identical_smoke") is not True
            or not isinstance(context, Mapping)
            or context.get("current") != "default"
            or not isinstance(named_context, Mapping)
            or named_context.get("credentials_absent") is not True
            or named_context.get("tls_storage_absent") is not True
            or not isinstance(guest, Mapping)
            or guest.get("effective_config", {}).get("status") != "pass"
            or not isinstance(guest.get("tcp_listener_allowlist"), list)
        ):
            raise D0Error("D0 aggregate runtime recovery/config/security evidence differs")
        if any(
            evidence.get(field, {}).get("swap_used_bytes") != 0
            for field in ("resources_before", "resources_after")
        ):
            raise D0Error("D0 aggregate runtime verification used host swap")
    elif operation == "rollback-runtime":
        if status != "rolled-back" or evidence.get("status") != "pass":
            raise D0Error("D0 aggregate rollback did not restore its baseline")
    elif operation == "postflight-audit":
        if (
            evidence.get("status") != "pass"
            or evidence.get("selected_runtime_comparison", {}).get("status") != "pass"
            or evidence.get("homebrew_comparison", {}).get("status") != "pass"
            or evidence.get("runtime_activity", {}).get("inactive") is not True
        ):
            raise D0Error("D0 aggregate postflight baseline differs")
        if packet["host"] == "john1" and (
            evidence.get("podman_semantics_stable") is not True
            or not isinstance(evidence.get("podman_after"), Mapping)
            or evidence["podman_after"].get("status") != "pass"
        ):
            raise D0Error("D0 aggregate John1 Podman postflight semantics are unproven")
    else:
        raise D0Error(f"D0 aggregate contains unsupported operation evidence: {operation}")


# Compatibility name retained for existing unit tests and external read-only auditors.
_validate_report_evidence = validate_operation_evidence


def _validate_transaction_set(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    by_report: dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = tuple(record.get("key", ()))
        if len(key) != 4 or key in indexed:
            raise D0Error("D0 aggregate transaction key is invalid or duplicated")
        typed_key = (str(key[0]), str(key[1]), str(key[2]), str(key[3]))
        indexed[typed_key] = record
        report_sha = record.get("report_sha256")
        if not isinstance(report_sha, str) or report_sha in by_report:
            raise D0Error("D0 aggregate report identity is invalid or duplicated")
        by_report[report_sha] = record
    if set(indexed) != EXPECTED_TRANSACTION_KEYS:
        missing = sorted(EXPECTED_TRANSACTION_KEYS - set(indexed))
        extra = sorted(set(indexed) - EXPECTED_TRANSACTION_KEYS)
        raise D0Error(
            f"D0 aggregate transaction graph differs: missing={missing!r} extra={extra!r}"
        )
    return by_report


def build_final_aggregate(
    archives: Sequence[bytes],
    *,
    public_key: bytes,
    created_unix_ms: int,
    bootstrap_records: Sequence[tuple[bytes, Mapping[str, Any]]],
    materialization_receipts: Sequence[bytes],
    topology_receipts: Sequence[bytes],
    helper_transitions: Sequence[tuple[bytes, Mapping[str, Any]]] = (),
) -> bytes:
    """Verify every bundle and render the one complete D0 aggregate document."""

    if (
        not isinstance(created_unix_ms, int)
        or isinstance(created_unix_ms, bool)
        or created_unix_ms <= 0
    ):
        raise D0Error("D0 aggregate creation time differs")
    normalized_key = normalize_public_key(public_key)
    transitions = verify_helper_transitions(helper_transitions, public_key=normalized_key)
    topology = validate_topology_receipts(topology_receipts)
    materializations: dict[str, dict[str, Any]] = {}
    for receipt_bytes in materialization_receipts:
        receipt = validate_materialization_receipt(
            load_canonical_json(
                receipt_bytes,
                maximum=1024 * 1024,
                label="D0 materialization receipt",
            )
        )
        if receipt["receipt_sha256"] in materializations:
            raise D0Error("D0 aggregate materialization receipt is duplicated")
        materializations[receipt["receipt_sha256"]] = receipt
    bootstraps: list[dict[str, Any]] = []
    bootstrap_hosts: set[str] = set()
    for record_bytes, signature in bootstrap_records:
        record = verify_bootstrap_record(
            record_bytes,
            signature,
            public_key=normalized_key,
        )
        if record["host"] in bootstrap_hosts:
            raise D0Error("D0 aggregate bootstrap host is duplicated")
        bootstrap_hosts.add(record["host"])
        bootstraps.append(
            {
                "host": record["host"],
                "record_sha256": record["record_sha256"],
                "record_payload_sha256": sha256_bytes(record_bytes),
                "record_signature_bundle_sha256": signature["bundle_sha256"],
                "bootstrap_packet_sha256": record["bootstrap_packet_sha256"],
                "bootstrap_receipt_sha256": record["bootstrap_receipt_sha256"],
                "helper_archive_sha256": record["helper_archive_sha256"],
                "public_key_sha256": record["public_key_sha256"],
                "installed_unix_ms": record["installed_unix_ms"],
                "status": "pass",
            }
        )
    if bootstrap_hosts != {"john1", "john2", "john3"}:
        raise D0Error("D0 aggregate requires exactly one bootstrap record per participating host")
    records: list[dict[str, Any]] = []
    transactions: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    common: dict[str, Any] | None = None
    for archive in archives:
        verification = verify_result_bundle(archive, public_key=normalized_key)
        packet = verification["packet"]
        report = verification["report"]
        validate_operation_evidence(packet, report)
        expected_status = "rolled-back" if report["phase"] == "rollback" else "pass"
        if report["status"] != expected_status:
            raise D0Error("D0 aggregate transaction status differs")
        identity = {
            "limits": packet["limits"],
            "policy": packet["policy"],
            "public_key_fingerprint": packet["public_key_fingerprint"],
        }
        if common is None:
            common = identity
        elif common != identity:
            raise D0Error("D0 aggregate policy/key/limit identity drifted")
        key = _key(packet, report)
        record = {
            "key": list(key),
            "archive_size": len(archive),
            "archive_sha256": sha256_bytes(archive),
            "manifest_sha256": verification["manifest"]["manifest_sha256"],
            "packet_sha256": packet["packet_sha256"],
            "report_sha256": report["report_sha256"],
            "evidence_sha256": sha256_bytes(canonical_json(report["evidence"])),
            "started_unix_ms": report["started_unix_ms"],
            "finished_unix_ms": report["finished_unix_ms"],
            "status": report["status"],
        }
        records.append(record)
        transactions.append((packet, report, record))
    if common is None:
        raise D0Error("D0 aggregate has no transactions")
    if any(item["public_key_sha256"] != sha256_bytes(normalized_key) for item in bootstraps):
        raise D0Error("D0 aggregate bootstrap campaign key differs")
    helper_history = _validate_helper_history(transactions, bootstraps, transitions)
    current_bootstrap_by_host = {item["host"]: item for item in bootstraps}
    for packet, _report, record in transactions:
        if (
            packet["helper_sha256"] == bootstraps[0]["helper_archive_sha256"]
            and current_bootstrap_by_host[packet["host"]]["installed_unix_ms"]
            > record["started_unix_ms"]
        ):
            raise D0Error("D0 aggregate current bootstrap was installed after work began")
    by_report = _validate_transaction_set(records)
    transaction_index = {
        (packet["cycle_id"], packet["host"], report["operation"]): (packet, report)
        for packet, report, _record in transactions
    }
    for cycle_id in ("qualification", "final-live"):
        _render_packet, render_report = transaction_index[
            (cycle_id, "john2", "render-runtime-supply")
        ]
        rendered_supply = render_report["evidence"].get("runtime_supply")
        if not isinstance(rendered_supply, Mapping):
            raise D0Error("D0 aggregate sealed runtime-supply render postcondition is absent")
        rendered_identity = (
            rendered_supply.get("archive_size"),
            rendered_supply.get("archive_sha256"),
        )
        for host in ("john1", "john3"):
            materialize_packet, materialize_report = transaction_index[
                (cycle_id, host, "materialize-runtime-supply")
            ]
            artifact = materialize_packet["artifacts"].get("runtime_supply")
            verification = (
                materialize_report["evidence"].get("runtime_supply", {}).get("verification", {})
            )
            if (
                not isinstance(artifact, Mapping)
                or (artifact.get("size"), artifact.get("sha256")) != rendered_identity
                or not isinstance(verification, Mapping)
                or (
                    verification.get("archive_size"),
                    verification.get("archive_sha256"),
                )
                != rendered_identity
            ):
                raise D0Error("D0 aggregate worker supply differs from the sealed John2 render")
        john1_report = transaction_index[(cycle_id, "john1", "materialize-runtime-supply")][1]
        john3_report = transaction_index[(cycle_id, "john3", "materialize-runtime-supply")][1]
        if (
            john1_report["evidence"].get("direct_ingress", {}).get("source_host") != "john2"
            or john3_report["evidence"].get("direct_ingress", {}).get("source_host") != "john1"
        ):
            raise D0Error("D0 aggregate runtime supply did not traverse direct John1 edges")
    used_materializations: set[str] = set()
    for packet, _report, _record in transactions:
        for predecessor in packet["predecessors"]:
            bound = by_report.get(predecessor["report_sha256"])
            if bound is None or any(
                bound[field] != predecessor[field]
                for field in ("packet_sha256", "report_sha256", "finished_unix_ms", "status")
            ):
                raise D0Error("D0 aggregate predecessor is absent or differs")
            if (
                bound["archive_sha256"] != predecessor["bundle_sha256"]
                or bound["archive_size"] != predecessor["bundle_size"]
                or bound["manifest_sha256"] != predecessor["manifest_sha256"]
            ):
                raise D0Error("D0 aggregate predecessor bundle identity differs")
            if (
                bound["key"][:3]
                != [
                    predecessor["cycle_id"],
                    predecessor["host"],
                    predecessor["phase"],
                ]
                or bound["key"][3] != predecessor["operation"]
            ):
                raise D0Error("D0 aggregate predecessor key differs")
            materialization = materializations.get(predecessor["materialization_receipt_sha256"])
            if materialization is None or any(
                materialization[field] != expected
                for field, expected in (
                    ("source_host", predecessor["host"]),
                    ("target_host", packet["host"]),
                    ("operation", predecessor["operation"]),
                    ("bundle_sha256", predecessor["bundle_sha256"]),
                    ("bundle_size", predecessor["bundle_size"]),
                    ("manifest_sha256", predecessor["manifest_sha256"]),
                    ("packet_sha256", predecessor["packet_sha256"]),
                    ("report_sha256", predecessor["report_sha256"]),
                    ("destination_relative", predecessor["receipt_relative"]),
                )
            ):
                raise D0Error("D0 aggregate predecessor materialization differs")
            if not (
                predecessor["finished_unix_ms"]
                <= materialization["materialized_unix_ms"]
                <= packet["issued_unix_ms"]
            ):
                raise D0Error("D0 aggregate predecessor materialization time differs")
            used_materializations.add(materialization["receipt_sha256"])
    if used_materializations != set(materializations):
        raise D0Error("D0 aggregate materialization receipt set has missing or extra entries")
    ordered = sorted(records, key=lambda item: tuple(item["key"]))
    document: dict[str, Any] = {
        "schema_id": AGGREGATE_SCHEMA,
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "created_unix_ms": created_unix_ms,
        "public_key_sha256": sha256_bytes(normalized_key),
        "public_key_fingerprint": public_key_fingerprint(normalized_key),
        "helper_sha256": bootstraps[0]["helper_archive_sha256"],
        **common,
        "bootstraps": sorted(bootstraps, key=lambda item: item["host"]),
        "bootstrap_count": len(bootstraps),
        "materialization_receipt_count": len(materializations),
        "materialization_receipt_sha256s": sorted(materializations),
        "topology_receipts": topology,
        "topology_receipt_count": len(topology),
        "helper_transitions": helper_history,
        "helper_transition_count": len(helper_history),
        "transactions": ordered,
        "transaction_count": len(ordered),
        "qualification_complete": True,
        "final_live_runtime_verified": True,
        "canonical_result_host": "john1",
        "direct_control_source": "john1",
        "peer_credentials_present": False,
        "john4_used": False,
        "protected_seed_values_opened": False,
        "project_code_executed": False,
        "status": "pass",
    }
    # common contains the same fingerprint; retaining one exact field is enough.
    document["public_key_fingerprint"] = public_key_fingerprint(normalized_key)
    document["aggregate_sha256"] = document_sha256(document, "aggregate_sha256")
    encoded = canonical_json(document)
    if len(encoded) > AGGREGATE_MAX_BYTES:
        raise D0Error("D0 aggregate exceeds its byte limit")
    return encoded


def verify_final_aggregate(
    aggregate_bytes: bytes,
    signature: Mapping[str, Any],
    *,
    public_key: bytes,
    archives: Sequence[bytes],
    bootstrap_records: Sequence[tuple[bytes, Mapping[str, Any]]],
    materialization_receipts: Sequence[bytes],
    topology_receipts: Sequence[bytes],
    helper_transitions: Sequence[tuple[bytes, Mapping[str, Any]]] = (),
) -> dict[str, Any]:
    """Verify signature and independently rebuild the aggregate from bundles."""

    aggregate = load_canonical_json(
        aggregate_bytes,
        maximum=AGGREGATE_MAX_BYTES,
        label="D0 final aggregate",
    )
    if (
        aggregate.get("schema_id") != AGGREGATE_SCHEMA
        or aggregate.get("schema_version") != 3
        or aggregate.get("campaign_id") != CAMPAIGN_ID
        or aggregate.get("run_id") != D0_RUN_ID
        or aggregate.get("aggregate_sha256") != document_sha256(aggregate, "aggregate_sha256")
        or aggregate.get("status") != "pass"
        or aggregate.get("john4_used") is not False
        or aggregate.get("canonical_result_host") != "john1"
        or aggregate.get("direct_control_source") != "john1"
        or aggregate.get("peer_credentials_present") is not False
    ):
        raise D0Error("D0 final aggregate identity differs")
    verify_stdin(public_key, aggregate_bytes, dict(signature))
    rebuilt = build_final_aggregate(
        archives,
        public_key=public_key,
        created_unix_ms=aggregate["created_unix_ms"],
        bootstrap_records=bootstrap_records,
        materialization_receipts=materialization_receipts,
        topology_receipts=topology_receipts,
        helper_transitions=helper_transitions,
    )
    if rebuilt != aggregate_bytes:
        raise D0Error("D0 final aggregate does not reproduce from its result bundles")
    return {
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "transaction_count": aggregate["transaction_count"],
        "bootstrap_count": aggregate["bootstrap_count"],
        "materialization_receipt_count": aggregate["materialization_receipt_count"],
        "topology_receipt_count": aggregate["topology_receipt_count"],
        "public_key_fingerprint": aggregate["public_key_fingerprint"],
        "status": "pass",
    }
