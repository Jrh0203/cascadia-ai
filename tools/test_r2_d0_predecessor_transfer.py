from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from r2_d0.canonical import D0_RUN_ID, D0Error, document_sha256
from r2_d0.closure import build_materialization_receipt
from r2_d0_predecessor_transfer import (
    SCHEMA,
    _destination_relative,
    _local_storage_identity,
    _validate_authorization,
)
from r2_d0_render_successor import _validate_transfer_provenance


def authorization() -> dict[str, object]:
    report = "2" * 64
    root = f"/Users/john2/.local/share/cascadia-r2/results/{D0_RUN_ID}"
    now = time.time_ns() // 1_000_000
    value: dict[str, object] = {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "campaign_id": "r2-map-expert-iteration-v1",
        "run_id": D0_RUN_ID,
        "source_control_host": "john1",
        "source_host": "john2",
        "target_host": "john2",
        "cycle_id": "qualification",
        "phase": "preflight",
        "operation": "preflight-audit",
        "packet_sha256": "1" * 64,
        "report_sha256": report,
        "bundle_sha256": "3" * 64,
        "bundle_size": 10240,
        "manifest_sha256": "4" * 64,
        "source_materialization_receipt_sha256": "5" * 64,
        "source_materialization_receipt_file_sha256": "6" * 64,
        "installer_sha256": "7" * 64,
        "target_output_root": root,
        "destination_relative": f"receipts/{report}",
        "destination": f"{root}/receipts/{report}",
        "transport": "direct-john1-control-edge",
        "peer_credentials_present": False,
        "issued_unix_ms": now - 1000,
        "expires_unix_ms": now + 60_000,
        "protected_seed_values_opened": False,
    }
    value["authorization_sha256"] = document_sha256(value, "authorization_sha256")
    return value


def test_transfer_authorization_binds_exact_target_receipt_path() -> None:
    value = authorization()
    assert _validate_authorization(value) == value
    value["destination"] = str(value["destination"]) + "-other"
    value["authorization_sha256"] = document_sha256(value, "authorization_sha256")
    with pytest.raises(D0Error, match="destination"):
        _validate_authorization(value)


def test_cross_host_transfer_uses_dependency_namespace() -> None:
    value = authorization()
    value["target_host"] = "john3"
    root = f"/Users/john3/.local/share/cascadia-r2/results/{D0_RUN_ID}"
    relative = f"dependencies/john2/{value['report_sha256']}"
    value["target_output_root"] = root
    value["destination_relative"] = relative
    value["destination"] = f"{root}/{relative}"
    value["authorization_sha256"] = document_sha256(
        value, "authorization_sha256"
    )
    assert _validate_authorization(value) == value
    assert (
        _destination_relative("john2", "john3", str(value["report_sha256"]))
        == relative
    )

    value["destination_relative"] = f"receipts/{value['report_sha256']}"
    value["destination"] = f"{root}/{value['destination_relative']}"
    value["authorization_sha256"] = document_sha256(
        value, "authorization_sha256"
    )
    with pytest.raises(D0Error, match="destination"):
        _validate_authorization(value)


def test_transfer_authorization_rejects_peer_transport_and_expiry() -> None:
    peer = authorization()
    peer["peer_credentials_present"] = True
    peer["authorization_sha256"] = document_sha256(peer, "authorization_sha256")
    with pytest.raises(D0Error, match="identity"):
        _validate_authorization(peer)

    expired = authorization()
    expired["issued_unix_ms"] = 1
    expired["expires_unix_ms"] = 2
    expired["authorization_sha256"] = document_sha256(expired, "authorization_sha256")
    with pytest.raises(D0Error, match="validity"):
        _validate_authorization(expired)


def test_target_storage_identity_is_stable_and_rejects_symlink(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    first = _local_storage_identity(tmp_path, "john2")
    second = _local_storage_identity(tmp_path, "john2")
    assert first == second
    assert len(first) == 64

    link = tmp_path.parent / f"{tmp_path.name}-link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(D0Error, match="unsafe"):
        _local_storage_identity(link, "john2")


def test_successor_binding_preserves_canonical_acceptance_and_worker_transfer() -> None:
    packet_sha256 = "1" * 64
    report_sha256 = "2" * 64
    bundle_sha256 = "3" * 64
    manifest_sha256 = "4" * 64
    bundle_size = 10240
    canonical_bytes = build_materialization_receipt(
        source_host="john2",
        target_host="john1",
        operation="preflight-audit",
        bundle_sha256=bundle_sha256,
        bundle_size=bundle_size,
        manifest_sha256=manifest_sha256,
        packet_sha256=packet_sha256,
        report_sha256=report_sha256,
        destination_relative=f"receipts/{report_sha256}",
        transport_receipt_sha256="8" * 64,
        storage_identity_sha256="9" * 64,
        persistence_evidence_sha256="a" * 64,
        materialized_unix_ms=time.time_ns() // 1_000_000 - 2000,
    )
    canonical_receipt = json.loads(canonical_bytes)
    transfer = authorization()
    transfer.update(
        {
            "packet_sha256": packet_sha256,
            "report_sha256": report_sha256,
            "bundle_sha256": bundle_sha256,
            "bundle_size": bundle_size,
            "manifest_sha256": manifest_sha256,
            "source_materialization_receipt_sha256": canonical_receipt[
                "receipt_sha256"
            ],
            "source_materialization_receipt_file_sha256": hashlib.sha256(
                canonical_bytes
            ).hexdigest(),
        }
    )
    transfer["authorization_sha256"] = document_sha256(
        transfer, "authorization_sha256"
    )
    target_receipt = json.loads(
        build_materialization_receipt(
            source_host="john2",
            target_host="john2",
            operation="preflight-audit",
            bundle_sha256=bundle_sha256,
            bundle_size=bundle_size,
            manifest_sha256=manifest_sha256,
            packet_sha256=packet_sha256,
            report_sha256=report_sha256,
            destination_relative=f"receipts/{report_sha256}",
            transport_receipt_sha256=transfer["authorization_sha256"],
            storage_identity_sha256="b" * 64,
            persistence_evidence_sha256="a" * 64,
            materialized_unix_ms=time.time_ns() // 1_000_000 - 1000,
        )
    )
    arguments = {
        "canonical_acceptance_bytes": canonical_bytes,
        "canonical_acceptance": canonical_receipt,
        "target_receipt": target_receipt,
        "authorization": transfer,
        "packet": {"host": "john2", "packet_sha256": packet_sha256},
        "report": {"report_sha256": report_sha256},
        "manifest_sha256": manifest_sha256,
        "bundle_sha256": bundle_sha256,
        "bundle_size": bundle_size,
        "target_host": "john2",
    }
    _validate_transfer_provenance(**arguments)

    target_receipt["transport_receipt_sha256"] = "c" * 64
    with pytest.raises(D0Error, match="provenance"):
        _validate_transfer_provenance(**arguments)


def test_successor_binding_requires_cross_host_dependency_namespace() -> None:
    packet_sha256 = "1" * 64
    report_sha256 = "2" * 64
    bundle_sha256 = "3" * 64
    manifest_sha256 = "4" * 64
    bundle_size = 10240
    canonical_bytes = build_materialization_receipt(
        source_host="john1",
        target_host="john1",
        operation="materialize-runtime-supply",
        bundle_sha256=bundle_sha256,
        bundle_size=bundle_size,
        manifest_sha256=manifest_sha256,
        packet_sha256=packet_sha256,
        report_sha256=report_sha256,
        destination_relative=f"receipts/{report_sha256}",
        transport_receipt_sha256="8" * 64,
        storage_identity_sha256="9" * 64,
        persistence_evidence_sha256="a" * 64,
        materialized_unix_ms=time.time_ns() // 1_000_000 - 2000,
    )
    canonical_receipt = json.loads(canonical_bytes)
    transfer = authorization()
    relative = f"dependencies/john1/{report_sha256}"
    target_root = f"/Users/john3/.local/share/cascadia-r2/results/{D0_RUN_ID}"
    transfer.update(
        {
            "source_host": "john1",
            "target_host": "john3",
            "target_output_root": target_root,
            "destination_relative": relative,
            "destination": f"{target_root}/{relative}",
            "packet_sha256": packet_sha256,
            "report_sha256": report_sha256,
            "bundle_sha256": bundle_sha256,
            "bundle_size": bundle_size,
            "manifest_sha256": manifest_sha256,
            "source_materialization_receipt_sha256": canonical_receipt[
                "receipt_sha256"
            ],
            "source_materialization_receipt_file_sha256": hashlib.sha256(
                canonical_bytes
            ).hexdigest(),
        }
    )
    transfer["authorization_sha256"] = document_sha256(
        transfer, "authorization_sha256"
    )
    target_receipt = json.loads(
        build_materialization_receipt(
            source_host="john1",
            target_host="john3",
            operation="materialize-runtime-supply",
            bundle_sha256=bundle_sha256,
            bundle_size=bundle_size,
            manifest_sha256=manifest_sha256,
            packet_sha256=packet_sha256,
            report_sha256=report_sha256,
            destination_relative=relative,
            transport_receipt_sha256=transfer["authorization_sha256"],
            storage_identity_sha256="b" * 64,
            persistence_evidence_sha256="a" * 64,
            materialized_unix_ms=time.time_ns() // 1_000_000 - 1000,
        )
    )
    arguments = {
        "canonical_acceptance_bytes": canonical_bytes,
        "canonical_acceptance": canonical_receipt,
        "target_receipt": target_receipt,
        "authorization": transfer,
        "packet": {"host": "john1", "packet_sha256": packet_sha256},
        "report": {"report_sha256": report_sha256},
        "manifest_sha256": manifest_sha256,
        "bundle_sha256": bundle_sha256,
        "bundle_size": bundle_size,
        "target_host": "john3",
    }
    _validate_transfer_provenance(**arguments)

    target_receipt["destination_relative"] = f"receipts/{report_sha256}"
    with pytest.raises(D0Error, match="provenance"):
        _validate_transfer_provenance(**arguments)
