from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from r2_d0 import bundle as bundle_module
from r2_d0.bundle import (
    DRAFT_MANIFEST_NAME,
    MANIFEST_NAME,
    MANIFEST_SIGNATURE_NAME,
    render_draft_transaction_export,
    render_result_bundle_manifest,
    seal_result_bundle,
    verify_draft_transaction_export,
    verify_result_bundle,
)
from r2_d0.canonical import D0Error, canonical_json, document_sha256, sha256_bytes
from r2_d0.runtime import host_report
from r2_d0.signing import (
    public_key_fingerprint,
    public_key_from_private,
    sign_stdin,
    signature_bytes,
)
from r2_d0_test_support import persisted_transaction_files, rendered_work


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


def _transaction(signing_key: Path, tmp_path: Path) -> tuple[dict, dict[str, bytes], bytes]:
    public_key = public_key_from_private(signing_key)
    packet_bytes = rendered_work(
        "john2",
        "preflight",
        fingerprint=public_key_fingerprint(public_key),
        temporary_root=tmp_path,
    )
    packet = json.loads(packet_bytes)
    report = host_report(
        packet,
        status="pass",
        evidence={"fixture": "signed-manifest"},
        started_unix_ms=packet["issued_unix_ms"],
    )
    files = persisted_transaction_files({
        "work-packet.json": packet_bytes,
        "work-packet-signature.json": signature_bytes(sign_stdin(signing_key, packet_bytes)),
        "report.json": canonical_json(report),
    })
    return packet, files, public_key


def _transaction_for_host(
    signing_key: Path, tmp_path: Path, host: str
) -> tuple[dict, dict[str, bytes], bytes]:
    public_key = public_key_from_private(signing_key)
    packet_bytes = rendered_work(
        host,
        "preflight",
        fingerprint=public_key_fingerprint(public_key),
        temporary_root=tmp_path,
    )
    packet = json.loads(packet_bytes)
    report = host_report(
        packet,
        status="pass",
        evidence={"fixture": f"{host}-draft"},
        started_unix_ms=packet["issued_unix_ms"],
    )
    return packet, persisted_transaction_files({
        "work-packet.json": packet_bytes,
        "work-packet-signature.json": signature_bytes(sign_stdin(signing_key, packet_bytes)),
        "report.json": canonical_json(report),
    }), public_key


def test_result_bundle_requires_a_campaign_signed_manifest(
    signing_key: Path,
    tmp_path: Path,
) -> None:
    packet, files, public_key = _transaction(signing_key, tmp_path)
    arguments = {
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "packet_sha256": packet["packet_sha256"],
        "created_unix_ms": json.loads(files["report.json"])["finished_unix_ms"],
    }
    manifest_bytes, context = render_result_bundle_manifest(files, **arguments)
    manifest_signature = signature_bytes(sign_stdin(signing_key, manifest_bytes))
    archive, sealed = seal_result_bundle(
        files,
        manifest_bytes=manifest_bytes,
        manifest_signature_bytes=manifest_signature,
        public_key=public_key,
        **arguments,
    )
    verified = verify_result_bundle(archive, public_key=public_key)
    assert context["manifest"]["report_sha256"] == json.loads(files["report.json"])[
        "report_sha256"
    ]
    assert sealed["sealed"] is True
    assert verified["sealed"] is True
    assert verified["manifest_signature"]["payload_sha256"] == sha256_bytes(manifest_bytes)

    unsigned = bundle_module._bundle_from_members(
        {**files, MANIFEST_NAME: manifest_bytes}
    )
    with pytest.raises(D0Error, match="manifest or signature is absent"):
        verify_result_bundle(unsigned, public_key=public_key)


def test_report_and_manifest_rewrite_cannot_reuse_the_old_signature(
    signing_key: Path,
    tmp_path: Path,
) -> None:
    packet, files, public_key = _transaction(signing_key, tmp_path)
    report = json.loads(files["report.json"])
    arguments = {
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "packet_sha256": packet["packet_sha256"],
        "created_unix_ms": report["finished_unix_ms"],
    }
    manifest_bytes, _ = render_result_bundle_manifest(files, **arguments)
    old_signature = signature_bytes(sign_stdin(signing_key, manifest_bytes))

    report["evidence"] = {"fixture": "rewritten"}
    report["report_sha256"] = document_sha256(report, "report_sha256")
    changed_files = persisted_transaction_files(
        {
            "work-packet.json": files["work-packet.json"],
            "work-packet-signature.json": files["work-packet-signature.json"],
            "report.json": canonical_json(report),
        }
    )
    changed_manifest, _ = render_result_bundle_manifest(changed_files, **arguments)
    forged = bundle_module._bundle_from_members(
        {
            **changed_files,
            MANIFEST_NAME: changed_manifest,
            MANIFEST_SIGNATURE_NAME: old_signature,
        }
    )
    with pytest.raises(D0Error):
        verify_result_bundle(forged, public_key=public_key)


def test_bundle_refuses_missing_or_tampered_persistence_proof(
    signing_key: Path,
    tmp_path: Path,
) -> None:
    packet, files, _public_key = _transaction(signing_key, tmp_path)
    report = json.loads(files["report.json"])
    arguments = {
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "packet_sha256": packet["packet_sha256"],
        "created_unix_ms": report["finished_unix_ms"],
    }
    missing = dict(files)
    missing.pop("persistence-evidence.json")
    with pytest.raises(D0Error, match="transaction files"):
        render_result_bundle_manifest(missing, **arguments)

    evidence = json.loads(files["persistence-evidence.json"])
    evidence["before"]["swap_used_bytes"] = 1
    evidence["evidence_sha256"] = document_sha256(evidence, "evidence_sha256")
    swapped = {**files, "persistence-evidence.json": canonical_json(evidence)}
    with pytest.raises(D0Error, match="resource evidence"):
        render_result_bundle_manifest(swapped, **arguments)

    receipt = json.loads(files["persistence-receipt.json"])
    receipt["report_sha256"] = "f" * 64
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    evidence = json.loads(files["persistence-evidence.json"])
    evidence["persistence_receipt_sha256"] = receipt["receipt_sha256"]
    evidence["evidence_sha256"] = document_sha256(evidence, "evidence_sha256")
    mismatched = {
        **files,
        "persistence-receipt.json": canonical_json(receipt),
        "persistence-evidence.json": canonical_json(evidence),
    }
    with pytest.raises(D0Error, match="receipt binding"):
        render_result_bundle_manifest(mismatched, **arguments)


def test_john3_draft_is_not_canonical_until_john1_signs_v3_manifest(
    signing_key: Path,
    tmp_path: Path,
) -> None:
    packet, files, public_key = _transaction_for_host(signing_key, tmp_path, "john3")
    draft, draft_context = render_draft_transaction_export(files, public_key=public_key)
    verified_draft = verify_draft_transaction_export(draft, public_key=public_key)
    assert verified_draft["sealed"] is False
    assert verified_draft["canonical_eligible"] is False
    assert DRAFT_MANIFEST_NAME not in verified_draft["files"]
    with pytest.raises(D0Error, match="manifest or signature is absent"):
        verify_result_bundle(draft, public_key=public_key)

    report = json.loads(files["report.json"])
    arguments = {
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "packet_sha256": packet["packet_sha256"],
        "created_unix_ms": report["finished_unix_ms"],
    }
    manifest, _ = render_result_bundle_manifest(draft_context["files"], **arguments)
    sealed, _ = seal_result_bundle(
        draft_context["files"],
        manifest_bytes=manifest,
        manifest_signature_bytes=signature_bytes(sign_stdin(signing_key, manifest)),
        public_key=public_key,
        **arguments,
    )
    verified = verify_result_bundle(sealed, public_key=public_key)
    assert verified["sealed"] is True
    assert verified["manifest"]["schema_version"] == 3
    assert verified["manifest"]["host"] == "john3"


def test_legacy_unsigned_v2_and_negative_control_role_are_rejected(
    signing_key: Path,
    tmp_path: Path,
) -> None:
    packet, files, public_key = _transaction_for_host(signing_key, tmp_path, "john1")
    report = json.loads(files["report.json"])
    legacy_manifest = {
        "schema_id": "cascadia.r2-map.d0-result-bundle-manifest.v2",
        "schema_version": 2,
        "campaign_id": packet["campaign_id"],
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": "john1",
        "role": "negative-control",
        "packet_sha256": packet["packet_sha256"],
        "created_unix_ms": report["finished_unix_ms"],
        "files": [],
        "protected_seed_values_opened": False,
        "project_code_executed": False,
    }
    legacy_manifest["manifest_sha256"] = document_sha256(
        legacy_manifest, "manifest_sha256"
    )
    legacy = bundle_module._bundle_from_members(
        {**files, MANIFEST_NAME: canonical_json(legacy_manifest)}
    )
    with pytest.raises(D0Error, match="manifest or signature is absent"):
        verify_result_bundle(legacy, public_key=public_key)
