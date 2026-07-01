from __future__ import annotations

import json
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest
from r2_d0.canonical import D0Error, document_sha256, sha256_bytes
from r2_d0.signing import load_public_key
from r2_d0_helper_migration import _render_retry_lineage

CAMPAIGN = Path("/Users/johnherrick/cascadia-bench/r2-map-v1/control/d0-v8-campaign")
FAILURE = CAMPAIGN / "qualification-ready/john2/verify-runtime-v11-full"
DIAGNOSTIC = CAMPAIGN / "recovery/qualification/john2/v38-readonly-post-canonical-egress-failure"
PUBLIC_KEY = CAMPAIGN / "plan/campaign-public-key"
OLD_HELPER = CAMPAIGN / (
    "plan-v11-full-superseded-"
    "d93a4ddae9a866fb0009711e587d5874f4b7b4e1a8bd2de40cac98ad1447d484/helper.tar"
)
CURRENT_FAILURE = CAMPAIGN / "qualification-ready/john2/verify-runtime-v12-shared-policy-retry-01"
PATH_CHAIN_DIAGNOSTIC = CAMPAIGN / "recovery/qualification/john2/v39-readonly-path-chain-inventory"
NETWORK_FAILURE = CAMPAIGN / "qualification-ready/john2/verify-runtime-v13-managed-link-retry-02"
NETWORK_DIAGNOSTIC = (
    CAMPAIGN / "recovery/qualification/john2/v42-path-corrected-post-retry02-network-inventory"
)


def _arguments() -> Namespace:
    return Namespace(
        retry_failure_packet=FAILURE / "work-packet.json",
        retry_failure_packet_signature=FAILURE / "work-packet-signature.json",
        retry_failure_report=FAILURE / "report.json",
        retry_failure_completion=FAILURE / "control-completion.json",
        retry_failure_persistence_receipt=FAILURE / "persistence-receipt.json",
        retry_failure_persistence_evidence=FAILURE / "persistence-evidence.json",
        retry_failure_persistence_monitor=FAILURE / "persistence-monitor.json",
        retry_diagnostic_authorization=DIAGNOSTIC / "authorization.json",
        retry_diagnostic_authorization_signature=(DIAGNOSTIC / "authorization-signature.json"),
        retry_diagnostic_package_manifest=DIAGNOSTIC / "package-manifest.json",
        retry_diagnostic_package_manifest_signature=(
            DIAGNOSTIC / "package-manifest-signature.json"
        ),
        retry_diagnostic_result=DIAGNOSTIC / "result.json",
        retry_diagnostic_evidence=DIAGNOSTIC / "evidence.json",
        retry_diagnostic_receipt=DIAGNOSTIC / "receipt.json",
    )


def _path_chain_arguments() -> Namespace:
    return Namespace(
        retry_failure_packet=CURRENT_FAILURE / "work-packet.json",
        retry_failure_packet_signature=CURRENT_FAILURE / "work-packet-signature.json",
        retry_failure_report=CURRENT_FAILURE / "result-finalization/report.json",
        retry_failure_completion=CURRENT_FAILURE / "control-completion.json",
        retry_failure_persistence_receipt=CURRENT_FAILURE / "persistence-receipt.json",
        retry_failure_persistence_evidence=CURRENT_FAILURE / "persistence-evidence.json",
        retry_failure_persistence_monitor=CURRENT_FAILURE / "persistence-monitor.json",
        retry_diagnostic_authorization=PATH_CHAIN_DIAGNOSTIC / "authorization.json",
        retry_diagnostic_authorization_signature=(
            PATH_CHAIN_DIAGNOSTIC / "authorization-signature.json"
        ),
        retry_diagnostic_package_manifest=PATH_CHAIN_DIAGNOSTIC / "package-manifest.json",
        retry_diagnostic_package_manifest_signature=(
            PATH_CHAIN_DIAGNOSTIC / "package-manifest-signature.json"
        ),
        retry_diagnostic_result=PATH_CHAIN_DIAGNOSTIC / "result.json",
        retry_diagnostic_evidence=PATH_CHAIN_DIAGNOSTIC / "evidence.json",
        retry_diagnostic_evidence_signature=PATH_CHAIN_DIAGNOSTIC / "evidence-signature.json",
        retry_diagnostic_receipt=PATH_CHAIN_DIAGNOSTIC / "receipt.json",
        retry_diagnostic_receipt_signature=PATH_CHAIN_DIAGNOSTIC / "receipt-signature.json",
    )


def _network_report(tmp_path: Path) -> Path:
    bundle = next((NETWORK_FAILURE / "result-finalization").glob("*.tar"))
    report = tmp_path / "report.json"
    with tarfile.open(bundle, "r:") as archive:
        source = archive.extractfile("report.json")
        assert source is not None
        report.write_bytes(source.read())
    return report


def _network_arguments(report: Path) -> Namespace:
    return Namespace(
        retry_failure_packet=NETWORK_FAILURE / "work-packet.json",
        retry_failure_packet_signature=NETWORK_FAILURE / "work-packet-signature.json",
        retry_failure_report=report,
        retry_failure_completion=NETWORK_FAILURE / "control-completion.json",
        retry_failure_persistence_receipt=NETWORK_FAILURE / "persistence-receipt.json",
        retry_failure_persistence_evidence=NETWORK_FAILURE / "persistence-evidence.json",
        retry_failure_persistence_monitor=NETWORK_FAILURE / "persistence-monitor.json",
        retry_diagnostic_authorization=NETWORK_DIAGNOSTIC / "authorization.json",
        retry_diagnostic_authorization_signature=(
            NETWORK_DIAGNOSTIC / "authorization-signature.json"
        ),
        retry_diagnostic_package_manifest=NETWORK_DIAGNOSTIC / "package-manifest.json",
        retry_diagnostic_package_manifest_signature=(
            NETWORK_DIAGNOSTIC / "package-manifest-signature.json"
        ),
        retry_diagnostic_result=NETWORK_DIAGNOSTIC / "result.json",
        retry_diagnostic_evidence=NETWORK_DIAGNOSTIC / "evidence.json",
        retry_diagnostic_evidence_signature=NETWORK_DIAGNOSTIC / "evidence-signature.json",
        retry_diagnostic_receipt=NETWORK_DIAGNOSTIC / "receipt.json",
        retry_diagnostic_receipt_signature=NETWORK_DIAGNOSTIC / "receipt-signature.json",
    )


def test_live_retry_lineage_binds_failure_and_clean_read_only_proof() -> None:
    lineage = _render_retry_lineage(
        _arguments(),
        public_key=load_public_key(PUBLIC_KEY),
        old_helper_sha256=sha256_bytes(OLD_HELPER.read_bytes()),
    )
    assert lineage is not None
    assert lineage["disposition"] == {
        "failed_packet_replay_allowed": False,
        "fresh_operation_identity_required": True,
        "qualification_accepted": False,
        "read_only_diagnostic_only": True,
    }
    assert (
        lineage["failed_qualification"]["report_sha256"]
        == "7511c9d0632b7d62515c90d456783b951ededb8093ba360d81dbaa6bf1397866"
    )
    assert (
        lineage["post_failure_read_only_proof"]["receipt_sha256"]
        == "4e6ce8746956877e34c34d4eaa1b3fb4133fcbaae25e736eaa85979fecccef18"
    )


def test_live_path_chain_retry_lineage_binds_signed_nonmutation_proof() -> None:
    failed_helper_sha256 = json.loads((CURRENT_FAILURE / "work-packet.json").read_bytes())[
        "helper_sha256"
    ]
    lineage = _render_retry_lineage(
        _path_chain_arguments(),
        public_key=load_public_key(PUBLIC_KEY),
        old_helper_sha256=failed_helper_sha256,
    )
    assert lineage is not None
    assert (
        lineage["failed_qualification"]["report_sha256"]
        == "99239295eee1a59bf9fe044faa6e99f527ba65cd6f9d3512942192ae439ac6c1"
    )
    proof = lineage["post_failure_read_only_proof"]
    assert (
        proof["receipt_sha256"]
        == "45b859b4444272cdd223ca4d88448c8f94744332aac6452b417a01bb5bc6b021"
    )
    assert "evidence_signature_file_sha256" in proof
    assert "receipt_signature_file_sha256" in proof
    assert (
        proof["result_sha256"] == "7b35fc692c6be08726f18ece215a6659390d0e4b8765d2d4955e7c0d6900d3bf"
    )


def test_live_network_retry_lineage_binds_signed_docker_lifecycle_proof(
    tmp_path: Path,
) -> None:
    failed_helper_sha256 = json.loads((NETWORK_FAILURE / "work-packet.json").read_bytes())[
        "helper_sha256"
    ]
    lineage = _render_retry_lineage(
        _network_arguments(_network_report(tmp_path)),
        public_key=load_public_key(PUBLIC_KEY),
        old_helper_sha256=failed_helper_sha256,
    )
    assert lineage is not None
    assert (
        lineage["failed_qualification"]["report_sha256"]
        == "2bf1f7151a306497a0ed742a3c95a4bfe853effb6f2e07e245b1e48a6f21ded3"
    )
    proof = lineage["post_failure_read_only_proof"]
    assert (
        proof["receipt_sha256"]
        == "f16353df22c211cd3d5191001dfa7f18cd1ddc8436b08620a0778db6f3f80137"
    )
    assert proof["network_change_classification"] == (
        "docker-owned-lazy-default-bridge-and-firewall-lifecycle"
    )
    assert proof["prior_diagnostic_failure"] == {
        "evidence_sha256": ("5433db9ba6658f742f551b33a7320e8d1c47e8f22dcb5e2389690c1fc75743c2"),
        "receipt_sha256": ("079977dc6b7f9e6cac3d5eebfe994e4ef45a35fcee469d97a5f0ec1f22a8eb81"),
        "replay_forbidden": True,
    }


def test_network_retry_lineage_rejects_classification_drift(tmp_path: Path) -> None:
    arguments = _network_arguments(_network_report(tmp_path))
    evidence = json.loads((NETWORK_DIAGNOSTIC / "evidence.json").read_bytes())
    evidence["classification"]["causal_scope"] = "unclassified"
    evidence["evidence_sha256"] = document_sha256(evidence, "evidence_sha256")
    tampered = tmp_path / "evidence.json"
    tampered.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    arguments.retry_diagnostic_evidence = tampered
    with pytest.raises(D0Error):
        _render_retry_lineage(
            arguments,
            public_key=load_public_key(PUBLIC_KEY),
            old_helper_sha256=json.loads((NETWORK_FAILURE / "work-packet.json").read_bytes())[
                "helper_sha256"
            ],
        )


def test_retry_lineage_rejects_incomplete_input_set() -> None:
    arguments = _arguments()
    arguments.retry_diagnostic_receipt = None
    with pytest.raises(D0Error, match="retry lineage input set is incomplete"):
        _render_retry_lineage(
            arguments,
            public_key=load_public_key(PUBLIC_KEY),
            old_helper_sha256=sha256_bytes(OLD_HELPER.read_bytes()),
        )


def test_retry_lineage_rejects_tampered_diagnostic_evidence(tmp_path: Path) -> None:
    arguments = _arguments()
    evidence = json.loads((DIAGNOSTIC / "evidence.json").read_bytes())
    evidence["status"] = "tampered"
    tampered = tmp_path / "evidence.json"
    tampered.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
    arguments.retry_diagnostic_evidence = tampered
    with pytest.raises(D0Error, match="read-only retry diagnostic lineage differs"):
        _render_retry_lineage(
            arguments,
            public_key=load_public_key(PUBLIC_KEY),
            old_helper_sha256=sha256_bytes(OLD_HELPER.read_bytes()),
        )
