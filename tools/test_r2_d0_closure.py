from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from r2_d0.canonical import CAMPAIGN_ID, D0_RUN_ID, D0Error, document_sha256
from r2_d0.closure import (
    BOOTSTRAP_RECORD_SCHEMA,
    build_bootstrap_record,
    build_materialization_receipt,
    validate_materialization_receipt,
    verify_bootstrap_record,
)
from r2_d0.signing import (
    public_key_fingerprint,
    public_key_from_private,
    sign_stdin,
)
from r2_d0_test_support import rendered_bootstrap


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


def test_bootstrap_record_is_exactly_packet_receipt_and_key_bound(
    signing_key: Path,
) -> None:
    public_key = public_key_from_private(signing_key)
    now = time.time_ns() // 1_000_000
    packet_bytes = rendered_bootstrap(
        host="john2",
        helper_sha256="1" * 64,
        helper_size=1024,
        public_key_sha256=hashlib.sha256(public_key).hexdigest(),
        fingerprint=public_key_fingerprint(public_key),
        now=now,
    )
    packet = json.loads(packet_bytes)
    receipt = {
        "schema_id": "cascadia.r2-map.d0-bootstrap-receipt.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "host": "john2",
        "packet_content_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "packet_sha256": packet["packet_sha256"],
        "helper_archive_sha256": "1" * 64,
        "helper_manifest_sha256": "2" * 64,
        "helper_destination": packet["destinations"]["helper"],
        "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
        "public_key_fingerprint": public_key_fingerprint(public_key),
        "public_key_destination": packet["destinations"]["public_key"],
        "receipt_destination": packet["destinations"]["receipt"],
        "installed_unix_ms": now,
        "runtime_installed": False,
        "runtime_invoked": False,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    record_bytes = build_bootstrap_record(packet_bytes, receipt)
    record = json.loads(record_bytes)
    assert record["schema_id"] == BOOTSTRAP_RECORD_SCHEMA
    verified = verify_bootstrap_record(
        record_bytes,
        sign_stdin(signing_key, record_bytes),
        public_key=public_key,
    )
    assert verified["host"] == "john2"
    receipt["helper_manifest_sha256"] = "3" * 64
    with pytest.raises(D0Error, match="does not bind"):
        build_bootstrap_record(packet_bytes, receipt)


def test_materialization_receipt_binds_exact_bundle_destination_and_transport() -> None:
    encoded = build_materialization_receipt(
        source_host="john1",
        target_host="john3",
        operation="stage-worker-core",
        bundle_sha256="1" * 64,
        bundle_size=4096,
        manifest_sha256="2" * 64,
        packet_sha256="3" * 64,
        report_sha256="4" * 64,
        destination_relative="dependencies/john1/" + "4" * 64,
        transport_receipt_sha256="5" * 64,
        storage_identity_sha256="6" * 64,
        persistence_evidence_sha256="7" * 64,
        materialized_unix_ms=1_781_755_200_000,
    )
    receipt = validate_materialization_receipt(json.loads(encoded))
    assert receipt["bundle_size"] == 4096
    receipt["destination_relative"] = "../escape"
    with pytest.raises(D0Error):
        validate_materialization_receipt(receipt)


def test_materialization_receipt_rejects_worker_to_worker_peer_edge() -> None:
    with pytest.raises(D0Error, match="endpoint"):
        build_materialization_receipt(
            source_host="john2",
            target_host="john3",
            operation="peer-transfer",
            bundle_sha256="1" * 64,
            bundle_size=1,
            manifest_sha256="2" * 64,
            packet_sha256="3" * 64,
            report_sha256="4" * 64,
            destination_relative="dependencies/john2/" + "4" * 64,
            transport_receipt_sha256="5" * 64,
            storage_identity_sha256="6" * 64,
            persistence_evidence_sha256="7" * 64,
            materialized_unix_ms=1,
        )
