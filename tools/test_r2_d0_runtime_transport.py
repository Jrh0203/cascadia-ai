from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
import r2_d0.authorization as authorization_module
import r2_d0.cli as cli_module
from r2_d0.bundle import render_result_bundle_manifest, seal_result_bundle
from r2_d0.canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    LEGACY_WORK_PACKET_SCHEMA,
    D0Error,
    canonical_json,
    document_sha256,
    render_document,
)
from r2_d0.closure import build_materialization_receipt
from r2_d0.runtime import host_report
from r2_d0.signing import (
    public_key_fingerprint,
    public_key_from_private,
    sign_stdin,
    signature_bytes,
)
from r2_d0.transport import (
    atomic_write,
    claim_control_execution,
    complete_control_execution,
    control_envelope_path,
    inspect_control_execution,
    install_control_envelope,
    render_control_envelope,
    verify_control_envelope,
)
from r2_d0_test_support import persisted_transaction_files, work_spec


@pytest.fixture
def signing_key(tmp_path: Path) -> Path:
    key = tmp_path / "test-ed25519"
    subprocess.run(
        ["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    os.chmod(key, 0o600)
    return key


def envelope_fixture(
    tmp_path: Path,
    signing_key: Path,
    *,
    host: str = "john3",
) -> tuple[bytes, bytes, dict[str, object]]:
    public_key = public_key_from_private(signing_key)
    now = time.time_ns() // 1_000_000
    packet_bytes = render_document(
        work_spec(
            host,
            "preflight",
            now=now,
            temporary_root=tmp_path / "runtime",
            fingerprint=public_key_fingerprint(public_key),
        ),
        kind="work",
    )
    signature_bytes = canonical_json(sign_stdin(signing_key, packet_bytes))
    envelope = render_control_envelope(
        packet_bytes,
        signature_bytes,
        public_key=public_key,
    )
    return envelope, public_key, json.loads(packet_bytes)


def test_direct_control_envelope_is_john1_sourced_and_has_no_peer_credentials(
    tmp_path: Path, signing_key: Path
) -> None:
    envelope, public_key, packet = envelope_fixture(tmp_path, signing_key)
    verified = verify_control_envelope(
        envelope,
        public_key=public_key,
        target_host="john3",
    )
    assert verified["packet"] == packet
    assert verified["envelope"]["source_host"] == "john1"
    assert verified["envelope"]["target_host"] == "john3"
    assert verified["envelope"]["peer_credentials_present"] is False


def test_control_envelope_rejects_wrong_target_and_semantic_tamper(
    tmp_path: Path, signing_key: Path
) -> None:
    envelope, public_key, _packet = envelope_fixture(tmp_path, signing_key)
    with pytest.raises(D0Error, match="identity"):
        verify_control_envelope(envelope, public_key=public_key, target_host="john2")
    changed = json.loads(envelope)
    changed["source_host"] = "john2"
    changed["envelope_sha256"] = document_sha256(changed, "envelope_sha256")
    with pytest.raises(D0Error, match="identity"):
        verify_control_envelope(canonical_json(changed), public_key=public_key)


def test_control_install_is_target_local_atomic_and_idempotent(
    tmp_path: Path, signing_key: Path
) -> None:
    envelope, public_key, packet = envelope_fixture(tmp_path, signing_key)
    first = install_control_envelope(
        envelope,
        public_key=public_key,
        target_host="john3",
    )
    second = install_control_envelope(
        envelope,
        public_key=public_key,
        target_host="john3",
    )
    destination = control_envelope_path(packet)
    assert first["disposition"] == "installed"
    assert second["disposition"] == "already-installed"
    assert destination.read_bytes() == envelope
    assert first["source_host"] == "john1"
    assert first["peer_credentials_present"] is False


def _host_report(packet: dict[str, object]) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_id": "cascadia.r2-map.d0-host-report.v4",
        "schema_version": 4,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "phase": packet["phase"],
        "operation": "preflight-audit",
        "packet_sha256": packet["packet_sha256"],
        "started_unix_ms": packet["issued_unix_ms"],
        "finished_unix_ms": int(packet["issued_unix_ms"]) + 1,
        "status": "pass",
        "evidence": {"bounded": True},
        "protected_seed_values_opened": False,
        "project_code_executed": False,
    }
    report["report_sha256"] = document_sha256(report, "report_sha256")
    return report


def test_control_execution_claim_and_completion_are_replay_safe(
    tmp_path: Path, signing_key: Path
) -> None:
    envelope, public_key, packet = envelope_fixture(tmp_path, signing_key)
    install_control_envelope(envelope, public_key=public_key, target_host="john3")
    claimed = claim_control_execution(
        envelope,
        public_key=public_key,
        target_host="john3",
    )
    assert claimed["status"] == "pass"
    with pytest.raises(D0Error, match="already claimed"):
        claim_control_execution(
            envelope,
            public_key=public_key,
            target_host="john3",
        )
    report = _host_report(packet)
    result = {
        "host_report": report,
        "persistence": {"persistence_evidence": {"monitor": {"monitor_sha256": "a" * 64}}},
    }
    resources = {
        "before": {"swap_used_bytes": 0},
        "after": {"swap_used_bytes": 0},
        "continuous_swap": {"max_used_bytes": 0},
        "status": "pass",
    }
    completed = complete_control_execution(
        envelope,
        result,
        public_key=public_key,
        target_host="john3",
        resources=resources,
    )
    replay = complete_control_execution(
        envelope,
        result,
        public_key=public_key,
        target_host="john3",
        resources=resources,
    )
    assert completed["disposition"] == "completed"
    assert replay["disposition"] == "already-completed"
    assert (
        inspect_control_execution(
            envelope,
            public_key=public_key,
            target_host="john3",
        )["state"]
        == "completed"
    )


def test_atomic_write_is_owner_private_and_refuses_links(tmp_path: Path) -> None:
    parent = tmp_path / "safe"
    parent.mkdir(mode=0o700)
    destination = parent / "receipt.json"
    atomic_write(destination, b"receipt")
    assert destination.read_bytes() == b"receipt"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    with pytest.raises(D0Error, match="already present"):
        atomic_write(destination, b"other")

    unsafe = tmp_path / "unsafe"
    unsafe.symlink_to(parent, target_is_directory=True)
    with pytest.raises(D0Error, match="unsafe"):
        atomic_write(unsafe / "receipt.json", b"other")


def test_transport_source_contains_no_peer_or_remote_publication_surface() -> None:
    source = (Path(__file__).parent / "r2_d0/transport.py").read_text()
    banned = (
        "JOHN2_JOHN3",
        "authorized_keys",
        "worker_channel",
        "publish_to_john2",
        "pull_draft_from_john2",
        "remote_storage_worker",
    )
    assert not [token for token in banned if token in source]


def _transition_document(
    *,
    chain_index: int,
    old_helper: str,
    new_helper: str,
    old_plan: str,
    new_plan: str,
    accepted: list[dict[str, object]],
) -> dict[str, object]:
    authorizations = []
    receipts = []
    old_bootstraps = []
    new_bootstraps = []
    for index, host in enumerate(("john1", "john2", "john3"), 1):
        authorization = f"{chain_index * 10_000 + 100 + index:064x}"
        old_receipt = f"{chain_index * 10_000 + 200 + index:064x}"
        new_receipt = f"{chain_index * 10_000 + 300 + index:064x}"
        authorizations.append(
            {
                "host": host,
                "authorization_sha256": authorization,
                "authorization_file_sha256": f"{chain_index * 10_000 + 400 + index:064x}",
                "signature_file_sha256": f"{chain_index * 10_000 + 500 + index:064x}",
            }
        )
        receipts.append(
            {
                "host": host,
                "receipt_sha256": f"{chain_index * 10_000 + 600 + index:064x}",
                "receipt_file_sha256": f"{chain_index * 10_000 + 700 + index:064x}",
                "authorization_sha256": authorization,
                "old_helper_sha256": old_helper,
                "new_helper_sha256": new_helper,
                "old_bootstrap_receipt_sha256": old_receipt,
                "new_bootstrap_receipt_sha256": new_receipt,
                "finished_unix_ms": 20 + chain_index * 10,
            }
        )
        old_bootstraps.append(
            {
                "host": host,
                "record_sha256": f"{chain_index * 10_000 + 800 + index:064x}",
                "record_payload_sha256": f"{chain_index * 10_000 + 900 + index:064x}",
                "record_signature_bundle_sha256": f"{chain_index * 10_000 + 1000 + index:064x}",
                "bootstrap_receipt_sha256": old_receipt,
                "helper_archive_sha256": old_helper,
                "installed_unix_ms": 10 + chain_index * 10,
            }
        )
        new_bootstraps.append(
            {
                "host": host,
                "record_sha256": f"{chain_index * 10_000 + 1100 + index:064x}",
                "record_payload_sha256": f"{chain_index * 10_000 + 1200 + index:064x}",
                "record_signature_bundle_sha256": f"{chain_index * 10_000 + 1300 + index:064x}",
                "bootstrap_receipt_sha256": new_receipt,
                "helper_archive_sha256": new_helper,
                "installed_unix_ms": 19 + chain_index * 10,
            }
        )
    value: dict[str, object] = {
        "schema_id": "cascadia.r2-map.d0-helper-transition.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "chain_index": chain_index,
        "from_plan_sha256": old_plan,
        "from_plan_file_sha256": f"{chain_index * 10_000 + 1400:064x}",
        "from_helper_sha256": old_helper,
        "to_plan_sha256": new_plan,
        "to_plan_file_sha256": f"{chain_index * 10_000 + 1500:064x}",
        "to_helper_sha256": new_helper,
        "collision_incident_sha256": f"{chain_index * 10_000 + 1600:064x}",
        "collision_incident_file_sha256": f"{chain_index * 10_000 + 1700:064x}",
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


def _signed_transition(signing_key: Path, document: dict[str, object]) -> dict[str, object]:
    return {
        "document": document,
        "signature": sign_stdin(signing_key, canonical_json(document)),
    }


def _mixed_helper_runtime_fixture(
    tmp_path: Path, signing_key: Path
) -> tuple[Namespace, dict[str, object], list[dict[str, object]], str]:
    public_key = public_key_from_private(signing_key)
    fingerprint = public_key_fingerprint(public_key)
    public_key_path = tmp_path / "campaign-key.pub"
    public_key_path.write_bytes(public_key)
    old_helper, middle_helper, current_helper = "1" * 64, "2" * 64, "3" * 64
    old_plan, middle_plan, current_plan = "4" * 64, "5" * 64, "6" * 64
    old_spec = work_spec(
        "john2",
        "preflight",
        helper_sha256=old_helper,
        fingerprint=fingerprint,
        temporary_root=tmp_path / "runtime",
    )
    old_spec.update({"schema_id": LEGACY_WORK_PACKET_SCHEMA, "schema_version": 9})
    old_spec.pop("helper_transitions")
    old_packet_bytes = render_document(old_spec, kind="work")
    old_packet = json.loads(old_packet_bytes)
    report = host_report(
        old_packet,
        status="pass",
        evidence={
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
            "runtime_budget_preflight": {"status": "pass"},
            "runtime_activity": {"inactive": True},
            "resources": {"swap_used_bytes": 0},
        },
        started_unix_ms=old_packet["issued_unix_ms"],
    )
    files = persisted_transaction_files(
        {
            "work-packet.json": old_packet_bytes,
            "work-packet-signature.json": signature_bytes(
                sign_stdin(signing_key, old_packet_bytes)
            ),
            "report.json": canonical_json(report),
        }
    )
    bundle_arguments = {
        "run_id": old_packet["run_id"],
        "cycle_id": old_packet["cycle_id"],
        "host": old_packet["host"],
        "role": old_packet["role"],
        "packet_sha256": old_packet["packet_sha256"],
        "created_unix_ms": report["finished_unix_ms"],
    }
    manifest_bytes, context = render_result_bundle_manifest(files, **bundle_arguments)
    bundle, sealed = seal_result_bundle(
        files,
        manifest_bytes=manifest_bytes,
        manifest_signature_bytes=signature_bytes(sign_stdin(signing_key, manifest_bytes)),
        public_key=public_key,
        **bundle_arguments,
    )
    accepted = [
        {
            "sequence": 1,
            "cycle_id": report["cycle_id"],
            "host": report["host"],
            "phase": report["phase"],
            "operation": report["operation"],
            "packet_sha256": report["packet_sha256"],
            "report_sha256": report["report_sha256"],
            "bundle_sha256": sealed["archive_sha256"],
            "bundle_size": sealed["archive_size"],
            "manifest_sha256": context["manifest"]["manifest_sha256"],
            "finished_unix_ms": report["finished_unix_ms"],
        }
    ]
    first = _transition_document(
        chain_index=1,
        old_helper=old_helper,
        new_helper=middle_helper,
        old_plan=old_plan,
        new_plan=middle_plan,
        accepted=accepted,
    )
    second = _transition_document(
        chain_index=2,
        old_helper=middle_helper,
        new_helper=current_helper,
        old_plan=middle_plan,
        new_plan=current_plan,
        accepted=[],
    )
    # Continuity includes the signed plan-file identity, not only semantic plans.
    second["from_plan_file_sha256"] = first["to_plan_file_sha256"]
    second["transition_sha256"] = document_sha256(second, "transition_sha256")
    transitions = [_signed_transition(signing_key, first), _signed_transition(signing_key, second)]
    current_issued = report["finished_unix_ms"] + 10
    current_spec = work_spec(
        "john2",
        "install",
        operations=["acquire-core"],
        helper_sha256=current_helper,
        fingerprint=fingerprint,
        now=current_issued,
        temporary_root=tmp_path / "runtime",
    )
    current_spec["helper_transitions"] = transitions
    relative = f"receipts/{report['report_sha256']}"
    receipt_bytes = build_materialization_receipt(
        source_host="john2",
        target_host="john2",
        operation="preflight-audit",
        bundle_sha256=sealed["archive_sha256"],
        bundle_size=sealed["archive_size"],
        manifest_sha256=context["manifest"]["manifest_sha256"],
        packet_sha256=old_packet["packet_sha256"],
        report_sha256=report["report_sha256"],
        destination_relative=relative,
        transport_receipt_sha256="7" * 64,
        storage_identity_sha256="8" * 64,
        persistence_evidence_sha256="9" * 64,
        materialized_unix_ms=report["finished_unix_ms"] + 1,
    )
    receipt = json.loads(receipt_bytes)
    current_spec["predecessors"] = [
        {
            "phase": report["phase"],
            "cycle_id": report["cycle_id"],
            "host": report["host"],
            "operation": report["operation"],
            "status": report["status"],
            "packet_sha256": report["packet_sha256"],
            "report_sha256": report["report_sha256"],
            "bundle_sha256": sealed["archive_sha256"],
            "bundle_size": sealed["archive_size"],
            "manifest_sha256": context["manifest"]["manifest_sha256"],
            "materialization_receipt_sha256": receipt["receipt_sha256"],
            "finished_unix_ms": report["finished_unix_ms"],
            "receipt_relative": relative,
        }
    ]
    destination = Path(current_spec["paths"]["output_root"]) / relative
    destination.mkdir(parents=True)
    (destination / "bundle.tar").write_bytes(bundle)
    (destination / "materialization-receipt.json").write_bytes(receipt_bytes)
    packet_bytes = render_document(current_spec, kind="work")
    packet_path = tmp_path / "current-work-packet.json"
    signature_path = tmp_path / "current-work-packet-signature.json"
    packet_path.write_bytes(packet_bytes)
    signature_path.write_bytes(signature_bytes(sign_stdin(signing_key, packet_bytes)))
    args = Namespace(
        packet=packet_path,
        signature=signature_path,
        public_key=public_key_path,
        control_envelope=None,
        execute=False,
    )
    return args, json.loads(packet_bytes), transitions, current_helper


def test_runtime_authorizes_signed_mixed_helper_successor_and_rejects_attacks(
    tmp_path: Path, signing_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, packet, transitions, current_helper = _mixed_helper_runtime_fixture(tmp_path, signing_key)
    monkeypatch.setattr(
        cli_module,
        "_active_helper_identity",
        lambda: {"archive_sha256": current_helper},
    )
    monkeypatch.setitem(authorization_module.HOST_USERS, "john2", "johnherrick")
    authorized = cli_module._authorized(args, phase="install", operation="acquire-core")
    assert authorized["packet_sha256"] == packet["packet_sha256"]
    assert args._predecessor_reports[0]["operation"] == "preflight-audit"

    def attempt(changed: dict[str, object], match: str) -> None:
        changed.pop("packet_sha256", None)
        payload = render_document(changed, kind="work")
        args.packet.write_bytes(payload)
        args.signature.write_bytes(signature_bytes(sign_stdin(signing_key, payload)))
        with pytest.raises(D0Error, match=match):
            cli_module._authorized(args, phase="install", operation="acquire-core")

    missing = deepcopy(packet)
    missing["helper_transitions"] = []
    attempt(missing, "authorization lineage")

    tampered = deepcopy(packet)
    tampered["helper_transitions"][0]["document"]["accepted_transactions"][0]["report_sha256"] = (
        "a" * 64
    )
    tampered["helper_transitions"][0]["document"]["transition_sha256"] = document_sha256(
        tampered["helper_transitions"][0]["document"], "transition_sha256"
    )
    with pytest.raises(D0Error, match="signature bundle identity"):
        attempt(tampered, "signature bundle identity")

    reordered = deepcopy(packet)
    reordered["helper_transitions"].reverse()
    with pytest.raises(D0Error, match="chain order"):
        attempt(reordered, "chain order")

    gap = deepcopy(packet)
    gap["helper_transitions"].pop(0)
    with pytest.raises(D0Error, match="chain order"):
        attempt(gap, "chain order")

    quarantined = deepcopy(packet)
    first = deepcopy(transitions[0]["document"])
    first["accepted_transactions"] = []
    first["transition_sha256"] = document_sha256(first, "transition_sha256")
    quarantined["helper_transitions"][0] = _signed_transition(signing_key, first)
    attempt(quarantined, "authorization lineage")
