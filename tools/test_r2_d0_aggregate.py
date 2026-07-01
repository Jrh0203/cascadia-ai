from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import r2_d0.aggregate as aggregate_module
from r2_d0.aggregate import (
    EXPECTED_TRANSACTION_KEYS,
    FINALIZED_HELPER_TRANSITION_SCHEMA,
    _expected_transaction_keys,
    _validate_helper_history,
    _validate_report_evidence,
    _validate_transaction_set,
    build_final_aggregate,
    validate_helper_transition,
    validate_operation_evidence,
    validate_topology_receipts,
    verify_helper_transitions,
)
from r2_d0.canonical import D0Error, canonical_json, document_sha256, sha256_bytes
from r2_d0.signing import public_key_from_private, sign_stdin
from r2_d0_test_support import work_spec


def _transition() -> dict[str, object]:
    old_helper = "1" * 64
    new_helper = "2" * 64
    authorizations = []
    receipts = []
    old_bootstraps = []
    new_bootstraps = []
    for index, host in enumerate(("john1", "john2", "john3"), 1):
        authorization = f"{100 + index:064x}"
        old_receipt = f"{200 + index:064x}"
        new_receipt = f"{300 + index:064x}"
        authorizations.append(
            {
                "host": host,
                "authorization_sha256": authorization,
                "authorization_file_sha256": f"{400 + index:064x}",
                "signature_file_sha256": f"{500 + index:064x}",
            }
        )
        receipts.append(
            {
                "host": host,
                "receipt_sha256": f"{600 + index:064x}",
                "receipt_file_sha256": f"{700 + index:064x}",
                "authorization_sha256": authorization,
                "old_helper_sha256": old_helper,
                "new_helper_sha256": new_helper,
                "old_bootstrap_receipt_sha256": old_receipt,
                "new_bootstrap_receipt_sha256": new_receipt,
                "finished_unix_ms": 20,
            }
        )
        old_bootstraps.append(
            {
                "host": host,
                "record_sha256": f"{800 + index:064x}",
                "record_payload_sha256": f"{900 + index:064x}",
                "record_signature_bundle_sha256": f"{1000 + index:064x}",
                "bootstrap_receipt_sha256": old_receipt,
                "helper_archive_sha256": old_helper,
                "installed_unix_ms": 10,
            }
        )
        new_bootstraps.append(
            {
                "host": host,
                "record_sha256": f"{1100 + index:064x}",
                "record_payload_sha256": f"{1200 + index:064x}",
                "record_signature_bundle_sha256": f"{1300 + index:064x}",
                "bootstrap_receipt_sha256": new_receipt,
                "helper_archive_sha256": new_helper,
                "installed_unix_ms": 19,
            }
        )
    accepted = [
        {
            "sequence": 1,
            "cycle_id": "qualification",
            "host": "john1",
            "phase": "preflight",
            "operation": "preflight-audit",
            "packet_sha256": "a" * 64,
            "report_sha256": "b" * 64,
            "bundle_sha256": "c" * 64,
            "bundle_size": 100,
            "manifest_sha256": "d" * 64,
            "finished_unix_ms": 15,
        }
    ]
    value: dict[str, object] = {
        "schema_id": "cascadia.r2-map.d0-helper-transition.v1",
        "schema_version": 1,
        "campaign_id": "r2-map-expert-iteration-v1",
        "run_id": "d0-runtime-bootstrap-20260618-v1",
        "chain_index": 1,
        "from_plan_sha256": "3" * 64,
        "from_plan_file_sha256": "4" * 64,
        "from_helper_sha256": old_helper,
        "to_plan_sha256": "5" * 64,
        "to_plan_file_sha256": "6" * 64,
        "to_helper_sha256": new_helper,
        "collision_incident_sha256": "7" * 64,
        "collision_incident_file_sha256": "8" * 64,
        "accepted_transactions": accepted,
        "migration_authorizations": authorizations,
        "migration_receipts": receipts,
        "old_bootstraps": old_bootstraps,
        "new_bootstraps": new_bootstraps,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
    }
    value["transition_sha256"] = document_sha256(value, "transition_sha256")
    return value


def _finalized_transition(*, tail: bool = True) -> dict[str, object]:
    base = _transition()
    base_accepted = copy.deepcopy(base["accepted_transactions"])
    accepted = copy.deepcopy(base_accepted)
    tail_rows: list[dict[str, object]] = []
    if tail:
        transaction = {
            "sequence": 2,
            "cycle_id": "qualification",
            "host": "john2",
            "phase": "install",
            "operation": "acquire-scanner",
            "packet_sha256": "e" * 64,
            "report_sha256": "f" * 64,
            "bundle_sha256": "0" * 64,
            "bundle_size": 200,
            "manifest_sha256": "9" * 64,
            "finished_unix_ms": 18,
        }
        accepted.append(transaction)
        tail_rows.append(
            {
                **transaction,
                "canonical_receipt_sha256": "a" * 64,
                "canonical_receipt_file_sha256": "b" * 64,
                "target_host": "john2",
                "target_receipt_sha256": "c" * 64,
                "target_receipt_file_sha256": "d" * 64,
                "source_helper_sha256": base["from_helper_sha256"],
            }
        )
    value = copy.deepcopy(base)
    value.pop("transition_sha256")
    value.update(
        {
            "schema_id": FINALIZED_HELPER_TRANSITION_SCHEMA,
            "accepted_transactions": accepted,
            "base_transition_sha256": base["transition_sha256"],
            "base_transition_file_sha256": "a" * 64,
            "base_transition_signature_file_sha256": "b" * 64,
            "previous_transition_sha256": "c" * 64,
            "base_accepted_transaction_count": len(base_accepted),
            "tail_transaction_count": len(tail_rows),
            "tail_transactions": tail_rows,
            "collision_incident_signature_file_sha256": "d" * 64,
            "migration_receipt_cutoffs": [
                {
                    "host": receipt["host"],
                    "migration_receipt_sha256": receipt["receipt_sha256"],
                    "migration_receipt_file_sha256": receipt["receipt_file_sha256"],
                    "old_bootstrap_installed_unix_ms": old["installed_unix_ms"],
                    "new_bootstrap_installed_unix_ms": new["installed_unix_ms"],
                    "rotation_finished_unix_ms": receipt["finished_unix_ms"],
                }
                for receipt, old, new in zip(  # noqa: B905 -- Apple system Python is 3.9.
                    value["migration_receipts"],
                    value["old_bootstraps"],
                    value["new_bootstraps"],
                )
            ],
            "terminal": True,
            "finalized_unix_ms": 21,
        }
    )
    value["transition_sha256"] = document_sha256(value, "transition_sha256")
    return value


@pytest.fixture
def transition_signing_key(tmp_path: Path) -> Path:
    key = tmp_path / "transition-ed25519"
    subprocess.run(
        ["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
    )
    key.chmod(0o600)
    return key


def test_signed_helper_transition_is_ordered_and_tamper_evident(
    transition_signing_key: Path,
) -> None:
    transition = _transition()
    payload = canonical_json(transition)
    signature = sign_stdin(transition_signing_key, payload)
    assert (
        verify_helper_transitions(
            [(payload, signature)], public_key=public_key_from_private(transition_signing_key)
        )[0]["transition_sha256"]
        == transition["transition_sha256"]
    )

    tampered = copy.deepcopy(transition)
    tampered["accepted_transactions"][0]["report_sha256"] = "e" * 64
    tampered["transition_sha256"] = document_sha256(tampered, "transition_sha256")
    with pytest.raises(D0Error):
        verify_helper_transitions(
            [(canonical_json(tampered), signature)],
            public_key=public_key_from_private(transition_signing_key),
        )

    missing = copy.deepcopy(transition)
    missing.pop("migration_receipts")
    with pytest.raises(D0Error, match="fields differ"):
        validate_helper_transition(missing)

    reordered = copy.deepcopy(transition)
    reordered["new_bootstraps"].reverse()
    reordered["transition_sha256"] = document_sha256(reordered, "transition_sha256")
    with pytest.raises(D0Error, match="host order"):
        validate_helper_transition(reordered)

    boundary = copy.deepcopy(transition)
    for bootstrap in boundary["new_bootstraps"]:
        bootstrap["installed_unix_ms"] = 20
    boundary["transition_sha256"] = document_sha256(boundary, "transition_sha256")
    validate_helper_transition(boundary)

    after_finish = copy.deepcopy(transition)
    after_finish["new_bootstraps"][0]["installed_unix_ms"] = 21
    after_finish["transition_sha256"] = document_sha256(after_finish, "transition_sha256")
    with pytest.raises(D0Error, match="migration/bootstrap"):
        validate_helper_transition(after_finish)

    before_old = copy.deepcopy(transition)
    before_old["new_bootstraps"][0]["installed_unix_ms"] = 10
    before_old["transition_sha256"] = document_sha256(before_old, "transition_sha256")
    with pytest.raises(D0Error, match="migration/bootstrap"):
        validate_helper_transition(before_old)

    no_work = copy.deepcopy(transition)
    no_work["accepted_transactions"] = []
    no_work["transition_sha256"] = document_sha256(no_work, "transition_sha256")
    validate_helper_transition(no_work)


def test_terminal_transition_finalization_is_exact_cutoff_bound_and_non_amendable() -> None:
    final = _finalized_transition()
    assert validate_helper_transition(final)["tail_transaction_count"] == 1

    boundary = copy.deepcopy(final)
    boundary["accepted_transactions"][1]["finished_unix_ms"] = 19
    boundary["tail_transactions"][0]["finished_unix_ms"] = 19
    boundary["transition_sha256"] = document_sha256(boundary, "transition_sha256")
    validate_helper_transition(boundary)

    after_cutoff = copy.deepcopy(final)
    after_cutoff["accepted_transactions"][1]["finished_unix_ms"] = 20
    after_cutoff["tail_transactions"][0]["finished_unix_ms"] = 20
    after_cutoff["transition_sha256"] = document_sha256(after_cutoff, "transition_sha256")
    with pytest.raises(D0Error, match="outside cutoff"):
        validate_helper_transition(after_cutoff)

    new_helper = copy.deepcopy(final)
    new_helper["tail_transactions"][0]["source_helper_sha256"] = final["to_helper_sha256"]
    new_helper["transition_sha256"] = document_sha256(new_helper, "transition_sha256")
    with pytest.raises(D0Error, match="tail order"):
        validate_helper_transition(new_helper)

    duplicate = copy.deepcopy(final)
    duplicate["tail_transactions"][0].update(
        {
            "cycle_id": duplicate["accepted_transactions"][0]["cycle_id"],
            "host": duplicate["accepted_transactions"][0]["host"],
            "phase": duplicate["accepted_transactions"][0]["phase"],
            "operation": duplicate["accepted_transactions"][0]["operation"],
        }
    )
    duplicate["accepted_transactions"][1].update(
        {
            "cycle_id": duplicate["accepted_transactions"][0]["cycle_id"],
            "host": duplicate["accepted_transactions"][0]["host"],
            "phase": duplicate["accepted_transactions"][0]["phase"],
            "operation": duplicate["accepted_transactions"][0]["operation"],
        }
    )
    duplicate["tail_transactions"][0]["target_host"] = duplicate["tail_transactions"][0]["host"]
    duplicate["transition_sha256"] = document_sha256(duplicate, "transition_sha256")
    with pytest.raises(D0Error, match="duplicated"):
        validate_helper_transition(duplicate)

    reordered = copy.deepcopy(final)
    reordered["tail_transactions"][0]["sequence"] = 1
    reordered["transition_sha256"] = document_sha256(reordered, "transition_sha256")
    with pytest.raises(D0Error, match="tail order"):
        validate_helper_transition(reordered)

    missing = copy.deepcopy(final)
    missing["tail_transactions"] = []
    missing["transition_sha256"] = document_sha256(missing, "transition_sha256")
    with pytest.raises(D0Error, match="tail count"):
        validate_helper_transition(missing)

    amendment = copy.deepcopy(final)
    amendment["tail_finalizations"] = []
    amendment["transition_sha256"] = document_sha256(amendment, "transition_sha256")
    with pytest.raises(D0Error, match="fields differ"):
        validate_helper_transition(amendment)


def test_zero_tail_transition_finalization_is_terminal() -> None:
    final = _finalized_transition(tail=False)
    validated = validate_helper_transition(final)
    assert validated["tail_transaction_count"] == 0
    assert validated["accepted_transactions"] == _transition()["accepted_transactions"]


def test_mixed_helper_history_requires_exact_transition_transaction_set() -> None:
    transition = validate_helper_transition(_transition())
    accepted = transition["accepted_transactions"][0]
    old_packet = {
        "helper_sha256": transition["from_helper_sha256"],
        "cycle_id": accepted["cycle_id"],
        "host": accepted["host"],
        "packet_sha256": accepted["packet_sha256"],
    }
    old_report = {
        "phase": accepted["phase"],
        "operation": accepted["operation"],
        "report_sha256": accepted["report_sha256"],
    }
    old_record = {
        "archive_sha256": accepted["bundle_sha256"],
        "archive_size": accepted["bundle_size"],
        "manifest_sha256": accepted["manifest_sha256"],
        "finished_unix_ms": accepted["finished_unix_ms"],
        "started_unix_ms": 14,
    }
    new_packet = {
        "helper_sha256": transition["to_helper_sha256"],
        "cycle_id": "qualification",
        "host": "john2",
        "packet_sha256": "f" * 64,
    }
    new_report = {
        "phase": "install",
        "operation": "acquire-scanner",
        "report_sha256": "9" * 64,
    }
    new_record = {
        "archive_sha256": "8" * 64,
        "archive_size": 200,
        "manifest_sha256": "7" * 64,
        "finished_unix_ms": 31,
        "started_unix_ms": 30,
    }
    transactions = [(old_packet, old_report, old_record), (new_packet, new_report, new_record)]
    bootstraps = transition["new_bootstraps"]
    assert (
        _validate_helper_history(transactions, bootstraps, [transition])[0][
            "accepted_transaction_count"
        ]
        == 1
    )
    with pytest.raises(D0Error, match="unlisted mixed-helper"):
        _validate_helper_history(transactions, bootstraps, [])
    altered = copy.deepcopy(transition)
    altered["accepted_transactions"][0]["report_sha256"] = "0" * 64
    altered["transition_sha256"] = document_sha256(altered, "transition_sha256")
    with pytest.raises(D0Error, match="transaction set differs"):
        _validate_helper_history(transactions, bootstraps, [validate_helper_transition(altered)])


def test_full_synthetic_final_aggregate_closes_two_helper_transitions(
    monkeypatch: pytest.MonkeyPatch,
    transition_signing_key: Path,
) -> None:
    """Exercise the complete 46-node aggregate with v9c -> v9d -> v9e."""

    old_helper, intermediate_helper, final_helper = "1" * 64, "2" * 64, "3" * 64
    old_keys = [
        ("qualification", "john1", "preflight", "preflight-audit"),
        ("qualification", "john2", "preflight", "preflight-audit"),
        ("qualification", "john3", "preflight", "preflight-audit"),
        ("qualification", "john2", "install", "acquire-core"),
        ("qualification", "john2", "install", "acquire-homebrew-artifacts"),
    ]
    supply = {"size": 400, "sha256": "4" * 64}
    verified: dict[bytes, dict[str, object]] = {}
    accepted: list[dict[str, object]] = []
    for index, key in enumerate(_expected_transaction_keys(), 1):
        cycle, host, phase, operation = key
        archive = f"synthetic-archive-{index}".encode()
        helper = old_helper if key in old_keys else final_helper
        packet = {
            "helper_sha256": helper,
            "limits": {"frozen": 1},
            "policy": {"goal": "same"},
            "public_key_fingerprint": "synthetic-fingerprint",
            "cycle_id": cycle,
            "host": host,
            "packet_sha256": f"{10000 + index:064x}",
            "predecessors": [],
            "artifacts": {"runtime_supply": supply},
            "issued_unix_ms": 80 + index,
        }
        evidence: dict[str, object] = {}
        if host == "john2" and operation == "render-runtime-supply":
            evidence["runtime_supply"] = {
                "archive_size": supply["size"],
                "archive_sha256": supply["sha256"],
            }
        if host in {"john1", "john3"} and operation == "materialize-runtime-supply":
            evidence["runtime_supply"] = {
                "verification": {
                    "archive_size": supply["size"],
                    "archive_sha256": supply["sha256"],
                }
            }
            evidence["direct_ingress"] = {"source_host": "john2" if host == "john1" else "john1"}
        report = {
            "phase": phase,
            "operation": operation,
            "status": "rolled-back" if phase == "rollback" else "pass",
            "report_sha256": f"{20000 + index:064x}",
            "started_unix_ms": 50 + index,
            "finished_unix_ms": 51 + index,
            "evidence": evidence,
        }
        manifest = {"manifest_sha256": f"{30000 + index:064x}"}
        verified[archive] = {"packet": packet, "report": report, "manifest": manifest}
        if key in old_keys:
            accepted.append(
                {
                    "sequence": old_keys.index(key) + 1,
                    "cycle_id": cycle,
                    "host": host,
                    "phase": phase,
                    "operation": operation,
                    "packet_sha256": packet["packet_sha256"],
                    "report_sha256": report["report_sha256"],
                    "bundle_sha256": sha256_bytes(archive),
                    "bundle_size": len(archive),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "finished_unix_ms": report["finished_unix_ms"],
                }
            )

    public_key = public_key_from_private(transition_signing_key)
    accepted.sort(key=lambda item: item["sequence"])

    def bootstrap_rows(helper: str, installed: int, offset: int) -> list[dict[str, object]]:
        return [
            {
                "host": host,
                "record_sha256": f"{offset + index:064x}",
                "record_payload_sha256": f"{offset + 10 + index:064x}",
                "record_signature_bundle_sha256": f"{offset + 20 + index:064x}",
                "bootstrap_receipt_sha256": f"{offset + 30 + index:064x}",
                "helper_archive_sha256": helper,
                "installed_unix_ms": installed,
            }
            for index, host in enumerate(("john1", "john2", "john3"), 1)
        ]

    old_bootstraps = bootstrap_rows(old_helper, 10, 40000)
    intermediate_bootstraps = bootstrap_rows(intermediate_helper, 30, 50000)
    final_bootstraps = bootstrap_rows(final_helper, 40, 60000)
    for row in final_bootstraps:
        row["record_payload_sha256"] = sha256_bytes(row["host"].encode())

    def transition(
        index: int,
        before: str,
        after: str,
        old_rows: list[dict[str, object]],
        new_rows: list[dict[str, object]],
        accepted_rows: list[dict[str, object]],
        finish: int,
    ) -> dict[str, object]:
        authorizations = [
            {
                "host": host,
                "authorization_sha256": f"{70000 + index * 100 + row:064x}",
                "authorization_file_sha256": f"{71000 + index * 100 + row:064x}",
                "signature_file_sha256": f"{72000 + index * 100 + row:064x}",
            }
            for row, host in enumerate(("john1", "john2", "john3"), 1)
        ]
        receipts = [
            {
                "host": auth["host"],
                "receipt_sha256": f"{73000 + index * 100 + row:064x}",
                "receipt_file_sha256": f"{74000 + index * 100 + row:064x}",
                "authorization_sha256": auth["authorization_sha256"],
                "old_helper_sha256": before,
                "new_helper_sha256": after,
                "old_bootstrap_receipt_sha256": old_rows[row - 1]["bootstrap_receipt_sha256"],
                "new_bootstrap_receipt_sha256": new_rows[row - 1]["bootstrap_receipt_sha256"],
                "finished_unix_ms": finish,
            }
            for row, auth in enumerate(authorizations, 1)
        ]
        value: dict[str, object] = {
            "schema_id": "cascadia.r2-map.d0-helper-transition.v1",
            "schema_version": 1,
            "campaign_id": "r2-map-expert-iteration-v1",
            "run_id": "d0-runtime-bootstrap-20260618-v1",
            "chain_index": index,
            "from_plan_sha256": f"{80000 + index:064x}",
            "from_plan_file_sha256": f"{81000 + index:064x}",
            "from_helper_sha256": before,
            "to_plan_sha256": f"{80001 + index:064x}",
            "to_plan_file_sha256": f"{81001 + index:064x}",
            "to_helper_sha256": after,
            "collision_incident_sha256": "8" * 64,
            "collision_incident_file_sha256": "9" * 64,
            "accepted_transactions": accepted_rows,
            "migration_authorizations": authorizations,
            "migration_receipts": receipts,
            "old_bootstraps": old_rows,
            "new_bootstraps": new_rows,
            "project_code_executed": False,
            "protected_seed_values_opened": False,
        }
        value["transition_sha256"] = document_sha256(value, "transition_sha256")
        return value

    first = transition(
        1, old_helper, intermediate_helper, old_bootstraps, intermediate_bootstraps, accepted, 31
    )
    second = transition(
        2, intermediate_helper, final_helper, intermediate_bootstraps, final_bootstraps, [], 41
    )
    # Close plan continuity across the independently signed transition records.
    second["from_plan_sha256"] = first["to_plan_sha256"]
    second["from_plan_file_sha256"] = first["to_plan_file_sha256"]
    second["transition_sha256"] = document_sha256(second, "transition_sha256")
    transition_inputs = []
    for value in (first, second):
        payload = canonical_json(value)
        transition_inputs.append((payload, sign_stdin(transition_signing_key, payload)))

    monkeypatch.setattr(
        aggregate_module,
        "verify_result_bundle",
        lambda archive, public_key: verified[archive],
    )
    monkeypatch.setattr(aggregate_module, "validate_operation_evidence", lambda *_: None)
    monkeypatch.setattr(aggregate_module, "validate_topology_receipts", lambda _receipts: [])
    bootstrap_by_host = {item["host"]: item for item in final_bootstraps}

    def fake_bootstrap(record_bytes, signature, *, public_key):
        row = bootstrap_by_host[record_bytes.decode()]
        return {
            **row,
            "bootstrap_packet_sha256": "a" * 64,
            "public_key_sha256": sha256_bytes(public_key),
            "status": "pass",
        }

    monkeypatch.setattr(aggregate_module, "verify_bootstrap_record", fake_bootstrap)
    encoded = build_final_aggregate(
        list(verified),
        public_key=public_key,
        created_unix_ms=100,
        bootstrap_records=[
            (
                host.encode(),
                {
                    "bundle_sha256": next(
                        row["record_signature_bundle_sha256"]
                        for row in final_bootstraps
                        if row["host"] == host
                    )
                },
            )
            for host in ("john1", "john2", "john3")
        ],
        materialization_receipts=[],
        topology_receipts=[],
        helper_transitions=transition_inputs,
    )
    aggregate = __import__("json").loads(encoded)
    assert aggregate["transaction_count"] == 46
    assert aggregate["helper_transition_count"] == 2
    assert aggregate["helper_sha256"] == final_helper


def _record(key: tuple[str, str, str, str], index: int) -> dict[str, object]:
    return {
        "key": list(key),
        "packet_sha256": f"{index + 1:064x}",
        "report_sha256": f"{index + 1000:064x}",
        "finished_unix_ms": index + 1,
        "status": "rolled-back" if key[2] == "rollback" else "pass",
    }


def test_final_aggregate_requires_the_exact_qualification_and_live_graph() -> None:
    ordered = _expected_transaction_keys()
    assert len(ordered) == 46
    assert len(EXPECTED_TRANSACTION_KEYS) == 46
    assert not any(key[0] == "final-live" and key[2] == "rollback" for key in ordered)
    records = [_record(key, index) for index, key in enumerate(ordered)]
    assert len(_validate_transaction_set(records)) == 46

    with pytest.raises(D0Error, match="transaction graph differs"):
        _validate_transaction_set(records[:-1])
    duplicate = copy.deepcopy(records)
    duplicate[-1]["key"] = duplicate[0]["key"]
    with pytest.raises(D0Error, match="duplicated"):
        _validate_transaction_set(duplicate)


def test_aggregate_recomputes_podman_negative_control_semantics() -> None:
    packet = work_spec("john1", "preflight")
    report = {
        "phase": "preflight",
        "operation": "preflight-audit",
        "status": "pass",
        "evidence": {
            "phase_resources": {
                "before": {"swap_used_bytes": 0},
                "after": {"swap_used_bytes": 0},
                "continuous_swap": {
                    "sample_count": 2,
                    "nonzero_samples": 0,
                    "max_used_bytes": 0,
                    "sample_stream_sha256": "0" * 64,
                    "status": "pass",
                },
                "zero_swap_entire_phase": True,
                "status": "pass",
            },
            "platform": {"status": "pass"},
            "runtime_activity": {"inactive": True},
            "resources": {"swap_used_bytes": 0},
            "runtime_budget_preflight": {"status": "pass"},
            "podman_negative_control": {
                "status": "pass",
                "semantic": {
                    "machine_records": 0,
                    "machine_disks": 0,
                    "socket_entries": 0,
                    "storage_payload_files": 0,
                },
            },
        },
    }
    _validate_report_evidence(packet, report)
    report["evidence"]["podman_negative_control"]["semantic"]["machine_disks"] = 1
    with pytest.raises(D0Error, match="machine/storage"):
        _validate_report_evidence(packet, report)


def test_homebrew_evidence_accepts_dependency_order_but_rejects_set_drift() -> None:
    packet = work_spec("john2", "install", operations=["acquire-homebrew-artifacts"])
    report = {
        "phase": "install",
        "operation": "acquire-homebrew-artifacts",
        "status": "pass",
        "evidence": {
            "status": "pass",
            "formulae": ["lima", "colima", "docker", "docker-buildx"],
            "phase_resources": {
                "before": {"swap_used_bytes": 0},
                "after": {"swap_used_bytes": 0},
                "continuous_swap": {
                    "sample_count": 2,
                    "nonzero_samples": 0,
                    "max_used_bytes": 0,
                    "sample_stream_sha256": "0" * 64,
                    "status": "pass",
                },
                "zero_swap_entire_phase": True,
                "status": "pass",
            },
        },
    }
    _validate_report_evidence(packet, report)

    report["evidence"]["formulae"][-1] = "docker"
    with pytest.raises(D0Error, match="formulae differ"):
        _validate_report_evidence(packet, report)


def test_verify_runtime_aggregate_uses_emitted_context_isolation_evidence() -> None:
    packet = work_spec("john2", "verify", operations=["verify-runtime"])
    report = {
        "operation": "verify-runtime",
        "status": "pass",
        "evidence": {
            "status": "pass",
            "phase_resources": {
                "status": "pass",
                "zero_swap_entire_phase": True,
                "before": {"swap_used_bytes": 0},
                "after": {"swap_used_bytes": 0},
                "continuous_swap": {
                    "status": "pass",
                    "sample_count": 1,
                    "nonzero_samples": 0,
                    "max_used_bytes": 0,
                    "sample_stream_sha256": "0" * 64,
                },
            },
            "colima": {},
            "socket": {},
            "engine": {},
            "engine_info": {},
            "buildkit": {},
            "guest": {
                "effective_config": {"status": "pass"},
                "tcp_listener_allowlist": [],
            },
            "homebrew_comparison": {"status": "pass"},
            "budget": {"status": "pass"},
            "smoke_image": {"roundtrip": {"cleanup": "complete"}},
            "stop_start_recovery": {"status": "pass", "identical_smoke": True},
            "docker_context": {
                "current": "default",
                "named_context": {
                    "credentials_absent": True,
                    "tls_storage_absent": True,
                },
            },
            "resources_before": {"swap_used_bytes": 0},
            "resources_after": {"swap_used_bytes": 0},
        },
    }

    validate_operation_evidence(packet, report)

    report["evidence"]["docker_context"]["named_context"]["tls_storage_absent"] = False
    with pytest.raises(D0Error, match="runtime recovery/config/security evidence differs"):
        validate_operation_evidence(packet, report)


@pytest.mark.parametrize(
    ("operation", "artifact_key", "rendered_key", "size_key", "sha_key"),
    [
        ("acquire-smoke", "smoke_oci", "smoke_oci", "archive_bytes", "archive_sha256"),
        ("acquire-scanner", "scanner_oci", "oci", "archive_size", "archive_sha256"),
    ],
)
def test_derived_acquisition_output_self_binds_before_successor_projection(
    operation: str,
    artifact_key: str,
    rendered_key: str,
    size_key: str,
    sha_key: str,
) -> None:
    packet = work_spec("john2", "install", operations=[operation])
    assert packet["artifacts"][artifact_key] is None
    installed = {"size": 1234, "sha256": "a" * 64, "status": "installed"}
    rendered = {size_key: 1234, sha_key: "a" * 64, "status": "pass"}
    operation_evidence = (
        {"installed": installed, "smoke_oci": rendered}
        if operation == "acquire-smoke"
        else {
            "scanner_supply": {
                "status": "pass",
                "installed": {"oci": installed},
                "oci": rendered,
            }
        }
    )
    report = {
        "operation": operation,
        "status": "pass",
        "evidence": {
            **operation_evidence,
            "phase_resources": {
                "before": {"swap_used_bytes": 0},
                "after": {"swap_used_bytes": 0},
                "continuous_swap": {
                    "sample_count": 2,
                    "nonzero_samples": 0,
                    "max_used_bytes": 0,
                    "sample_stream_sha256": "0" * 64,
                    "status": "pass",
                },
                "zero_swap_entire_phase": True,
                "status": "pass",
            },
        },
    }
    _validate_report_evidence(packet, report)
    rendered[sha_key] = "b" * 64
    with pytest.raises(D0Error, match="OCI evidence differs"):
        _validate_report_evidence(packet, report)


def _topology_receipts(*, john3_blocked: bool = False) -> list[bytes]:
    runtime = []
    for host in ("john1", "john2", "john3"):
        certification = {
            "project_execution_authorized": False,
            "d0_certified": False,
        }
        if host == "john3" and john3_blocked:
            certification["campaign_execution_blocked"] = True
        runtime.append(
            canonical_json(
                {
                    "schema_id": (
                        "cascadia.r2-map.local-runtime-profile-receipt.v4"
                        if host in {"john1", "john3"}
                        else "cascadia.r2-map.local-runtime-profile-receipt.v3"
                    ),
                    "host": host,
                    "certification": certification,
                    "buildx": ({"version": "0.35.0"} if host == "john2" else {"installed": False}),
                    "receipt_sha256": host[0] * 64,
                }
            )
        )
    archive_sha = "a" * 64
    commit_sha = "b" * 64
    reopen_sha = "c" * 64
    return [
        *runtime,
        canonical_json(
            {
                "schema_id": "cascadia.r2-map.cold-archive-root-receipt.v1",
                "host": "john2",
                "archive_root": {"path": "/Users/john2/cascadia-bench/r2-map-archive-v1"},
                "authority": {
                    "active_artifact_authority": (
                        "john1:/Users/johnherrick/cascadia-bench/r2-map-v1"
                    )
                },
                "receipt_sha256": "d" * 64,
            }
        ),
        canonical_json(
            {
                "schema_id": "cascadia.r2-map.legacy-dashboard-termination-receipt.v1",
                "host": "john2",
                "receipt_sha256": "e" * 64,
            }
        ),
        canonical_json(
            {
                "schema_id": "cascadia.r2-map.john1-cold-archive-reopen.v1",
                "status": "pass",
                "verification": {"archive_sha256": archive_sha},
                "receipt_sha256": reopen_sha,
            }
        ),
        canonical_json(
            {
                "schema_id": "cascadia.r2-map.john2-cold-archive-commit.v1",
                "status": "pass",
                "archive_sha256": archive_sha,
                "receipt_sha256": commit_sha,
            }
        ),
        canonical_json(
            {
                "schema_id": "cascadia.r2-map.john3-legacy-cleanup-receipt.v1",
                "status": "pass",
                "source_root_absent": True,
                "deleted_entry_count": 7052,
                "deleted_unique_regular_bytes": 978085576,
                "archive_sha256": archive_sha,
                "john1_reopen_receipt_sha256": reopen_sha,
                "john2_commit_receipt_sha256": commit_sha,
                "receipt_sha256": "f" * 64,
            }
        ),
    ]


def test_topology_receipts_require_cleanup_and_unblocked_superseding_role() -> None:
    verified = validate_topology_receipts(_topology_receipts())
    assert len(verified) == 8
    with pytest.raises(D0Error, match="john3 live role receipt"):
        validate_topology_receipts(_topology_receipts(john3_blocked=True))
    with pytest.raises(D0Error, match="missing or extra"):
        validate_topology_receipts(_topology_receipts()[:-1])
