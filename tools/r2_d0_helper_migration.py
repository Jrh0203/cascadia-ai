#!/usr/bin/env python3
"""Signed cross-helper migration preserving authenticated D0 predecessor lineage."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import stat
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Optional

SOURCE_ROOT = Path(__file__).resolve().parent
HELPER_ROOT = Path.home() / ".local/libexec/cascadia-r2-d0/v1"
sys.path.insert(0, str(SOURCE_ROOT if (SOURCE_ROOT / "r2_d0").is_dir() else HELPER_ROOT))

from r2_d0.aggregate import validate_helper_transition, validate_operation_evidence  # noqa: E402
from r2_d0.bootstrap import apply_bootstrap, verify_helper_archive  # noqa: E402
from r2_d0.bundle import validate_persistence_transaction, verify_result_bundle  # noqa: E402
from r2_d0.canonical import (  # noqa: E402
    CAMPAIGN_ID,
    D0_RUN_ID,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    sha256_bytes,
    validate_bootstrap_packet,
    validate_host_report,
    validate_work_packet,
)
from r2_d0.closure import (  # noqa: E402
    validate_materialization_receipt,
    verify_bootstrap_record,
)
from r2_d0.signing import load_public_key, verify_stdin  # noqa: E402

LEGACY_LINEAGE_SCHEMA = "cascadia.r2-map.d0-cross-helper-lineage.v1"
LINEAGE_SCHEMA = "cascadia.r2-map.d0-cross-helper-lineage.v2"
AUTH_SCHEMA = "cascadia.r2-map.d0-helper-migration-authorization.v1"
RECEIPT_SCHEMA = "cascadia.r2-map.d0-helper-migration-receipt.v1"
HELPER_TRANSITION_SCHEMA = "cascadia.r2-map.d0-helper-transition.v1"
FINALIZED_HELPER_TRANSITION_SCHEMA = "cascadia.r2-map.d0-helper-transition-finalization.v1"
MAX_JSON = 4 * 1024 * 1024
MAX_HELPER = 16 * 1024 * 1024
MAX_BUNDLE = 2 * 1024 * 1024 * 1024
HOST_USERS = {"john1": "johnherrick", "john2": "john2", "john3": "john3"}
HOST_HOMES = {host: f"/Users/{user}" for host, user in HOST_USERS.items()}


def _read(path: Path, maximum: int, label: str, *, owner: bool = True) -> bytes:
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot inspect {label}") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (owner and observed.st_uid != os.getuid())
        or observed.st_mode & 0o022
        or observed.st_size > maximum
    ):
        raise D0Error(f"{label} metadata is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload = b""
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)
    if len(payload) != observed.st_size:
        raise D0Error(f"{label} changed while reading")
    return payload


def _write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read(path, MAX_JSON, label)
    value = load_canonical_json(payload, maximum=MAX_JSON, label=label)
    if not isinstance(value, dict):
        raise D0Error(f"{label} is not an object")
    return payload, value


def _json_relaxed(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    """Load captured command JSON whose exact stdout may include one newline."""

    payload = _read(path, MAX_JSON, label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise D0Error(f"{label} is not an object")
    return payload, value


def _plan(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload, value = _json(path, "D0 execution plan")
    if (
        value.get("schema_id") != "cascadia.r2-map.d0-execution-plan.v1"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("run_id") != D0_RUN_ID
        or value.get("d0_status") != "red"
        or value.get("plan_sha256") != document_sha256(value, "plan_sha256")
    ):
        raise D0Error("D0 execution plan identity differs")
    return payload, value


def _bootstrap_receipt(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "host",
        "packet_content_sha256",
        "packet_sha256",
        "helper_archive_sha256",
        "helper_manifest_sha256",
        "helper_destination",
        "public_key_sha256",
        "public_key_fingerprint",
        "public_key_destination",
        "receipt_destination",
        "installed_unix_ms",
        "runtime_installed",
        "runtime_invoked",
        "project_code_executed",
        "protected_seed_values_opened",
        "status",
        "receipt_sha256",
    }
    if (
        set(value) != required
        or value["schema_id"] != "cascadia.r2-map.d0-bootstrap-receipt.v1"
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["host"] not in HOST_USERS
        or value["runtime_installed"] is not False
        or value["runtime_invoked"] is not False
        or value["project_code_executed"] is not False
        or value["protected_seed_values_opened"] is not False
        or value["status"] != "pass"
        or value["receipt_sha256"] != document_sha256(value, "receipt_sha256")
    ):
        raise D0Error("bootstrap receipt identity differs")
    return value


def _render_retry_lineage(
    args: argparse.Namespace,
    *,
    public_key: bytes,
    old_helper_sha256: str,
) -> Optional[dict[str, Any]]:  # noqa: UP045 -- remote Apple Python is 3.9.
    """Authenticate one failed canonical attempt and its read-only retry proof.

    Failed reports must never enter ``accepted_predecessors``.  They still need
    durable, signed provenance when a helper correction is specifically caused
    by that failure.  The v2 lineage therefore binds the failed transaction and
    the independently authorized post-failure clean-state proof as retry-only
    evidence, while explicitly forbidding packet replay.
    """

    names = (
        "retry_failure_packet",
        "retry_failure_packet_signature",
        "retry_failure_report",
        "retry_failure_completion",
        "retry_failure_persistence_receipt",
        "retry_failure_persistence_evidence",
        "retry_failure_persistence_monitor",
        "retry_diagnostic_authorization",
        "retry_diagnostic_authorization_signature",
        "retry_diagnostic_package_manifest",
        "retry_diagnostic_package_manifest_signature",
        "retry_diagnostic_result",
        "retry_diagnostic_evidence",
        "retry_diagnostic_receipt",
    )
    paths = {name: getattr(args, name, None) for name in names}
    present = {name for name, path in paths.items() if path is not None}
    if not present:
        return None
    if present != set(names):
        raise D0Error("retry lineage input set is incomplete")

    failure_packet_bytes, failure_packet = _json(paths["retry_failure_packet"], "failed packet")
    validate_work_packet(failure_packet)
    failure_signature_bytes, failure_signature = _json(
        paths["retry_failure_packet_signature"], "failed packet signature"
    )
    verify_stdin(public_key, failure_packet_bytes, failure_signature)
    failure_report_bytes, failure_report = _json(paths["retry_failure_report"], "failed report")
    validate_host_report(failure_report, packet=failure_packet)
    completion_bytes, completion = _json(
        paths["retry_failure_completion"], "failed control completion"
    )
    persistence_receipt_bytes, _persistence_receipt = _json(
        paths["retry_failure_persistence_receipt"], "failed persistence receipt"
    )
    persistence_evidence_bytes, _persistence_evidence = _json(
        paths["retry_failure_persistence_evidence"], "failed persistence evidence"
    )
    persistence_monitor_bytes, _persistence_monitor = _json(
        paths["retry_failure_persistence_monitor"], "failed persistence monitor"
    )
    persistence_files = {
        "work-packet.json": failure_packet_bytes,
        "work-packet-signature.json": failure_signature_bytes,
        "report.json": failure_report_bytes,
        "persistence-receipt.json": persistence_receipt_bytes,
        "persistence-evidence.json": persistence_evidence_bytes,
        "persistence-monitor.json": persistence_monitor_bytes,
    }
    persistence_receipt, persistence_evidence = validate_persistence_transaction(
        persistence_files,
        packet=failure_packet,
        report=failure_report,
    )
    persistence_monitor = persistence_evidence["monitor"]
    completion_result = completion.get("result")
    if (
        failure_packet["host"] != "john2"
        or failure_packet["cycle_id"] != "qualification"
        or failure_packet["phase"] != "verify"
        or failure_packet["helper_sha256"] != old_helper_sha256
        or failure_report["operation"] != "verify-runtime"
        or failure_report["status"] != "fail"
        or completion.get("schema_id") != "cascadia.r2-map.d0-control-completion.v2"
        or completion.get("completion_sha256") != document_sha256(completion, "completion_sha256")
        or completion.get("packet_sha256") != failure_packet["packet_sha256"]
        or completion.get("report_sha256") != failure_report["report_sha256"]
        or completion.get("persistence_monitor_sha256") != persistence_monitor["monitor_sha256"]
        or completion.get("execution_status") != "failed"
        or completion.get("status") != "pass"
        or not isinstance(completion_result, dict)
        or completion_result.get("host_report") != failure_report
    ):
        raise D0Error("failed canonical retry lineage differs")

    authorization_bytes, authorization = _json_relaxed(
        paths["retry_diagnostic_authorization"], "retry diagnostic authorization"
    )
    authorization_signature_bytes, authorization_signature = _json(
        paths["retry_diagnostic_authorization_signature"],
        "retry diagnostic authorization signature",
    )
    verify_stdin(public_key, authorization_bytes, authorization_signature)
    package_manifest_bytes, package_manifest = _json_relaxed(
        paths["retry_diagnostic_package_manifest"], "retry diagnostic package manifest"
    )
    package_signature_bytes, package_signature = _json(
        paths["retry_diagnostic_package_manifest_signature"],
        "retry diagnostic package manifest signature",
    )
    verify_stdin(public_key, package_manifest_bytes, package_signature)
    result_bytes, result = _json_relaxed(
        paths["retry_diagnostic_result"], "retry diagnostic result"
    )
    diagnostic_evidence_bytes, diagnostic_evidence = _json_relaxed(
        paths["retry_diagnostic_evidence"], "retry diagnostic evidence"
    )
    diagnostic_receipt_bytes, diagnostic_receipt = _json_relaxed(
        paths["retry_diagnostic_receipt"], "retry diagnostic receipt"
    )

    diagnostic_schema = authorization.get("schema_id")
    is_path_chain_diagnostic = (
        diagnostic_schema == "cascadia.r2-map.d0-path-chain-inventory-authorization.v1"
    )
    is_network_diagnostic = (
        diagnostic_schema == "cascadia.r2-map.d0-post-failure-network-authorization.v1"
    )
    evidence_signature_bytes: Optional[bytes] = None  # noqa: UP045 -- Apple Python 3.9.
    receipt_signature_bytes: Optional[bytes] = None  # noqa: UP045 -- Apple Python 3.9.
    if is_path_chain_diagnostic or is_network_diagnostic:
        evidence_signature_path = getattr(args, "retry_diagnostic_evidence_signature", None)
        receipt_signature_path = getattr(args, "retry_diagnostic_receipt_signature", None)
        if evidence_signature_path is None or receipt_signature_path is None:
            raise D0Error("signed path-chain retry evidence is incomplete")
        evidence_signature_bytes, evidence_signature = _json(
            evidence_signature_path, "retry diagnostic evidence signature"
        )
        receipt_signature_bytes, receipt_signature = _json(
            receipt_signature_path, "retry diagnostic receipt signature"
        )
        verify_stdin(public_key, diagnostic_evidence_bytes, evidence_signature)
        verify_stdin(public_key, diagnostic_receipt_bytes, receipt_signature)
        if is_path_chain_diagnostic and (
            authorization.get("authorization_sha256")
            != document_sha256(authorization, "authorization_sha256")
            or authorization.get("host") != "john2"
            or authorization.get("helper_sha256") != old_helper_sha256
            or authorization.get("failure_packet_sha256") != failure_packet["packet_sha256"]
            or authorization.get("failure_report_sha256") != failure_report["report_sha256"]
            or authorization.get("failure_completion_sha256") != completion["completion_sha256"]
            or authorization.get("status") != "authorized-once"
            or authorization.get("read_only") is not True
            or authorization.get("project_code_executed") is not False
            or authorization.get("protected_seed_values_opened") is not False
            or authorization.get("scanner_executed") is not False
            or authorization.get("qualification_claimed") is not False
            or package_manifest.get("schema_id")
            != "cascadia.r2-map.d0-path-chain-inventory-package.v1"
            or package_manifest.get("manifest_sha256")
            != document_sha256(package_manifest, "manifest_sha256")
            or package_manifest.get("authorization_sha256") != authorization["authorization_sha256"]
            or package_manifest.get("authorization_file_sha256")
            != sha256_bytes(authorization_bytes)
            or package_manifest.get("read_only") is not True
            or result.get("schema_id") != "cascadia.r2-map.d0-path-chain-inventory-result.v1"
            or result.get("result_sha256") != document_sha256(result, "result_sha256")
            or result.get("authorization_sha256") != authorization["authorization_sha256"]
            or result.get("failure_report_sha256") != failure_report["report_sha256"]
            or result.get("before") != result.get("after")
            or result.get("nonmutation_proven") is not True
            or result.get("status") != "pass-diagnostic"
            or diagnostic_evidence.get("schema_id")
            != "cascadia.r2-map.d0-path-chain-inventory-evidence.v1"
            or diagnostic_evidence.get("evidence_sha256")
            != document_sha256(diagnostic_evidence, "evidence_sha256")
            or diagnostic_evidence.get("authorization_sha256")
            != authorization["authorization_sha256"]
            or diagnostic_evidence.get("package_manifest_sha256")
            != package_manifest["manifest_sha256"]
            or diagnostic_evidence.get("failure_report_sha256") != failure_report["report_sha256"]
            or diagnostic_evidence.get("failure_completion_sha256")
            != completion["completion_sha256"]
            or diagnostic_evidence.get("result_sha256") != result["result_sha256"]
            or diagnostic_evidence.get("result_file_sha256") != sha256_bytes(result_bytes)
            or diagnostic_evidence.get("nonmutation_proven") is not True
            or diagnostic_evidence.get("status") != "pass-diagnostic"
            or diagnostic_receipt.get("schema_id")
            != "cascadia.r2-map.d0-path-chain-inventory-receipt.v1"
            or diagnostic_receipt.get("receipt_sha256")
            != document_sha256(diagnostic_receipt, "receipt_sha256")
            or diagnostic_receipt.get("authorization_sha256")
            != authorization["authorization_sha256"]
            or diagnostic_receipt.get("failure_report_sha256") != failure_report["report_sha256"]
            or diagnostic_receipt.get("evidence_sha256") != diagnostic_evidence["evidence_sha256"]
            or diagnostic_receipt.get("evidence_file_sha256")
            != sha256_bytes(diagnostic_evidence_bytes)
            or diagnostic_receipt.get("evidence_signature_file_sha256")
            != sha256_bytes(evidence_signature_bytes)
            or diagnostic_receipt.get("result_sha256") != result["result_sha256"]
            or diagnostic_receipt.get("result_file_sha256") != sha256_bytes(result_bytes)
            or diagnostic_receipt.get("nonmutation_proven") is not True
            or diagnostic_receipt.get("status") != "pass-diagnostic"
        ):
            raise D0Error("read-only path-chain retry diagnostic lineage differs")

    if is_network_diagnostic:
        failure_baseline = authorization.get("failure_baseline")
        prior_failure = authorization.get("prior_diagnostic_failure")
        package_prior = package_manifest.get("prior_diagnostic")
        result_ownership = result.get("ownership_and_lifecycle")
        result_docker0 = (
            result_ownership.get("docker0") if isinstance(result_ownership, dict) else None
        )
        result_raw = (
            result_ownership.get("ip_raw_prerouting")
            if isinstance(result_ownership, dict)
            else None
        )
        evidence_authorization = diagnostic_evidence.get("authorization")
        evidence_package = diagnostic_evidence.get("package_manifest")
        evidence_result = diagnostic_evidence.get("result")
        evidence_runtime = diagnostic_evidence.get("runtime_accounting")
        evidence_prior = diagnostic_evidence.get("prior_v41_failure")
        if (
            authorization.get("authorization_sha256")
            != document_sha256(authorization, "authorization_sha256")
            or authorization.get("campaign_id") != CAMPAIGN_ID
            or authorization.get("run_id") != D0_RUN_ID
            or authorization.get("host") != "john2"
            or authorization.get("public_key_sha256") != sha256_bytes(public_key)
            or authorization.get("status") != "authorized-once"
            or authorization.get("read_only") is not True
            or authorization.get("qualification_claimed") is not False
            or authorization.get("project_code_executed") is not False
            or authorization.get("protected_seed_values_opened") is not False
            or not isinstance(failure_baseline, dict)
            or failure_baseline.get("packet_sha256") != failure_packet["packet_sha256"]
            or failure_baseline.get("report_sha256") != failure_report["report_sha256"]
            or not isinstance(prior_failure, dict)
            or prior_failure.get("status") != "sealed-capture-environment-failure"
            or prior_failure.get("replay_forbidden") is not True
            or not isinstance(package_prior, dict)
            or package_prior.get("failure_evidence_sha256")
            != prior_failure.get("failure_evidence_sha256")
            or package_prior.get("failure_receipt_sha256")
            != prior_failure.get("failure_receipt_sha256")
            or package_prior.get("replay_forbidden") is not True
            or package_manifest.get("schema_id")
            != "cascadia.r2-map.d0-post-failure-network-package.v3"
            or package_manifest.get("manifest_sha256")
            != document_sha256(package_manifest, "manifest_sha256")
            or package_manifest.get("authorization_sha256") != authorization["authorization_sha256"]
            or package_manifest.get("host") != "john2"
            or package_manifest.get("read_only") is not True
            or package_manifest.get("qualification_claimed") is not False
            or package_manifest.get("project_code_executed") is not False
            or package_manifest.get("protected_seed_values_opened") is not False
            or result.get("schema_id") != "cascadia.r2-map.d0-post-failure-network-result.v1"
            or result.get("result_sha256") != document_sha256(result, "result_sha256")
            or result.get("authorization_sha256") != authorization["authorization_sha256"]
            or result.get("failure_baseline") != failure_baseline
            or result.get("host") != "john2"
            or result.get("status") != "pass-diagnostic"
            or result.get("read_only") is not True
            or result.get("qualification_claimed") is not False
            or result.get("project_code_executed") is not False
            or result.get("protected_seed_values_opened") is not False
            or not isinstance(result_docker0, dict)
            or result_docker0.get("present") is not True
            or result_docker0.get("docker_default_bridge_present") is not True
            or result_docker0.get("owner") != "docker-daemon-libnetwork"
            or result_docker0.get("lifecycle") != "default-bridge-interface"
            or not isinstance(result_raw, dict)
            or result_raw.get("present") is not True
            or result_raw.get("owner") != "docker-daemon-firewall-backend"
            or result_raw.get("lifecycle") != "daemon-network-firewall-programming"
            or diagnostic_evidence.get("schema_id")
            != "cascadia.r2-map.d0-post-failure-network-evidence.v1"
            or diagnostic_evidence.get("evidence_sha256")
            != document_sha256(diagnostic_evidence, "evidence_sha256")
            or diagnostic_evidence.get("host") != "john2"
            or diagnostic_evidence.get("status") != "pass-diagnostic"
            or diagnostic_evidence.get("read_only") is not True
            or diagnostic_evidence.get("qualification_claimed") is not False
            or diagnostic_evidence.get("project_code_executed") is not False
            or diagnostic_evidence.get("protected_seed_values_opened") is not False
            or not isinstance(evidence_authorization, dict)
            or evidence_authorization.get("semantic_sha256")
            != authorization["authorization_sha256"]
            or evidence_authorization.get("file_sha256") != sha256_bytes(authorization_bytes)
            or evidence_authorization.get("signature_file_sha256")
            != sha256_bytes(authorization_signature_bytes)
            or not isinstance(evidence_package, dict)
            or evidence_package.get("semantic_sha256") != package_manifest["manifest_sha256"]
            or evidence_package.get("file_sha256") != sha256_bytes(package_manifest_bytes)
            or evidence_package.get("signature_file_sha256")
            != sha256_bytes(package_signature_bytes)
            or not isinstance(evidence_result, dict)
            or evidence_result.get("semantic_sha256") != result["result_sha256"]
            or evidence_result.get("file_sha256") != sha256_bytes(result_bytes)
            or not isinstance(evidence_runtime, dict)
            or evidence_runtime.get("containers") != 0
            or evidence_runtime.get("images") != 0
            or evidence_runtime.get("volumes") != 0
            or evidence_runtime.get("buildkit_total_bytes") != 0
            or evidence_runtime.get("buildkit_reclaimable_bytes") != 0
            or evidence_runtime.get("docker_service") != "active"
            or not isinstance(evidence_prior, dict)
            or evidence_prior.get("evidence_sha256") != prior_failure.get("failure_evidence_sha256")
            or evidence_prior.get("receipt_sha256") != prior_failure.get("failure_receipt_sha256")
            or evidence_prior.get("replayed") is not False
            or diagnostic_evidence.get("capture", {}).get("swap_zero_throughout") is not True
            or diagnostic_evidence.get("classification", {}).get("causal_scope")
            != (
                "Docker-owned lazy default-bridge and firewall lifecycle; "
                "no D0-owned residue observed"
            )
            or diagnostic_receipt.get("schema_id")
            != "cascadia.r2-map.d0-post-failure-network-receipt.v1"
            or diagnostic_receipt.get("receipt_sha256")
            != document_sha256(diagnostic_receipt, "receipt_sha256")
            or diagnostic_receipt.get("authorization_sha256")
            != authorization["authorization_sha256"]
            or diagnostic_receipt.get("package_manifest_sha256")
            != package_manifest["manifest_sha256"]
            or diagnostic_receipt.get("evidence_sha256") != diagnostic_evidence["evidence_sha256"]
            or diagnostic_receipt.get("evidence_file_sha256")
            != sha256_bytes(diagnostic_evidence_bytes)
            or diagnostic_receipt.get("evidence_signature_file_sha256")
            != sha256_bytes(evidence_signature_bytes)
            or diagnostic_receipt.get("result_sha256") != result["result_sha256"]
            or diagnostic_receipt.get("result_file_sha256") != sha256_bytes(result_bytes)
            or diagnostic_receipt.get("invocation_count") != 1
            or diagnostic_receipt.get("network_change_classification")
            != "docker-owned-lazy-default-bridge-and-firewall-lifecycle"
            or diagnostic_receipt.get("swap_zero_throughout") is not True
            or diagnostic_receipt.get("status") != "pass-diagnostic"
            or diagnostic_receipt.get("read_only") is not True
            or diagnostic_receipt.get("qualification_claimed") is not False
            or diagnostic_receipt.get("project_code_executed") is not False
            or diagnostic_receipt.get("protected_seed_values_opened") is not False
        ):
            raise D0Error("read-only network retry diagnostic lineage differs")

    failure_binding = authorization.get("canonical_failure")
    package_authorization = package_manifest.get("authorization")
    package_failure = package_manifest.get("canonical_failure")
    result_semantic_sha256 = (
        result["result_sha256"]
        if is_path_chain_diagnostic or is_network_diagnostic
        else sha256_bytes(canonical_json(result))
    )
    if (
        not is_path_chain_diagnostic
        and not is_network_diagnostic
        and (
            authorization.get("schema_id")
            != "cascadia.r2-map.d0-v38-readonly-post-canonical-egress-failure.v1"
            or authorization.get("authorization_sha256")
            != document_sha256(authorization, "authorization_sha256")
            or authorization.get("host") != "john2"
            or authorization.get("candidate_helper_sha256") != old_helper_sha256
            or authorization.get("status") != "authorized-once"
            or authorization.get("project_code_executed") is not False
            or authorization.get("protected_seed_values_opened") is not False
            or not isinstance(failure_binding, dict)
            or failure_binding.get("packet_sha256") != failure_packet["packet_sha256"]
            or failure_binding.get("packet_file_sha256") != sha256_bytes(failure_packet_bytes)
            or failure_binding.get("report_sha256") != failure_report["report_sha256"]
            or failure_binding.get("report_file_sha256") != sha256_bytes(failure_report_bytes)
            or failure_binding.get("completion_sha256") != completion["completion_sha256"]
            or failure_binding.get("completion_file_sha256") != sha256_bytes(completion_bytes)
            or failure_binding.get("persistence_receipt_sha256")
            != persistence_receipt["receipt_sha256"]
            or failure_binding.get("persistence_receipt_file_sha256")
            != sha256_bytes(persistence_receipt_bytes)
            or failure_binding.get("persistence_evidence_sha256")
            != persistence_evidence["evidence_sha256"]
            or failure_binding.get("persistence_evidence_file_sha256")
            != sha256_bytes(persistence_evidence_bytes)
            or failure_binding.get("persistence_monitor_sha256")
            != persistence_monitor["monitor_sha256"]
            or failure_binding.get("persistence_monitor_file_sha256")
            != sha256_bytes(persistence_monitor_bytes)
            or package_manifest.get("schema_id")
            != "cascadia.r2-map.d0-v38-dispatch-package-manifest.v1"
            or package_manifest.get("manifest_sha256")
            != document_sha256(package_manifest, "manifest_sha256")
            or package_manifest.get("host") != "john2"
            or package_manifest.get("helper", {}).get("sha256") != old_helper_sha256
            or package_manifest.get("qualification_claimed") is not False
            or package_failure != failure_binding
            or not isinstance(package_authorization, dict)
            or package_authorization.get("semantic_sha256") != authorization["authorization_sha256"]
            or package_authorization.get("file_sha256") != sha256_bytes(authorization_bytes)
            or package_authorization.get("signature_file_sha256")
            != sha256_bytes(authorization_signature_bytes)
            or result.get("mode") != "read-only-egress-socket-inventory"
            or result.get("packet_sha256") != failure_packet["packet_sha256"]
            or result.get("status") != "pass"
            or result.get("project_code_executed") is not False
            or result.get("protected_seed_values_opened") is not False
            or result.get("scanner_executed") is not False
            or result.get("result", {}).get("status") != "diagnostic-pass"
            or result.get("result", {}).get("read_only") is not True
            or diagnostic_evidence.get("schema_id")
            != "cascadia.r2-map.d0-v38-readonly-post-canonical-egress-failure-evidence.v1"
            or diagnostic_evidence.get("evidence_sha256")
            != document_sha256(diagnostic_evidence, "evidence_sha256")
            or diagnostic_evidence.get("authorization_sha256")
            != authorization["authorization_sha256"]
            or diagnostic_evidence.get("authorization_file_sha256")
            != sha256_bytes(authorization_bytes)
            or diagnostic_evidence.get("authorization_signature_file_sha256")
            != sha256_bytes(authorization_signature_bytes)
            or diagnostic_evidence.get("canonical_failure", {}).get("report_sha256")
            != failure_report["report_sha256"]
            or diagnostic_evidence.get("canonical_failure", {}).get("completion_sha256")
            != completion["completion_sha256"]
            or diagnostic_evidence.get("package_manifest", {}).get("semantic_sha256")
            != package_manifest["manifest_sha256"]
            or diagnostic_evidence.get("package_manifest", {}).get("file_sha256")
            != sha256_bytes(package_manifest_bytes)
            or diagnostic_evidence.get("package_manifest", {}).get("signature_file_sha256")
            != sha256_bytes(package_signature_bytes)
            or diagnostic_evidence.get("result", {}).get("semantic_sha256")
            != result_semantic_sha256
            or diagnostic_evidence.get("result", {}).get("file_sha256")
            != sha256_bytes(result_bytes)
            or diagnostic_evidence.get("status") != "pass-diagnostic"
            or diagnostic_evidence.get("qualification_claimed") is not False
            or diagnostic_receipt.get("schema_id")
            != "cascadia.r2-map.d0-v38-readonly-post-canonical-egress-failure-receipt.v1"
            or diagnostic_receipt.get("receipt_sha256")
            != document_sha256(diagnostic_receipt, "receipt_sha256")
            or diagnostic_receipt.get("authorization_sha256")
            != authorization["authorization_sha256"]
            or diagnostic_receipt.get("canonical_failure_report_sha256")
            != failure_report["report_sha256"]
            or diagnostic_receipt.get("packet_sha256") != failure_packet["packet_sha256"]
            or diagnostic_receipt.get("helper_sha256") != old_helper_sha256
            or diagnostic_receipt.get("evidence_sha256") != diagnostic_evidence["evidence_sha256"]
            or diagnostic_receipt.get("evidence_file_sha256")
            != sha256_bytes(diagnostic_evidence_bytes)
            or diagnostic_receipt.get("result_sha256") != result_semantic_sha256
            or diagnostic_receipt.get("result_file_sha256") != sha256_bytes(result_bytes)
            or diagnostic_receipt.get("status") != "pass-diagnostic"
            or diagnostic_receipt.get("read_only") is not True
            or diagnostic_receipt.get("qualification_claimed") is not False
            or diagnostic_receipt.get("scanner_executed") is not False
            or diagnostic_receipt.get("daemon_objects_before") != 0
            or diagnostic_receipt.get("daemon_objects_after") != 0
            or diagnostic_receipt.get("buildkit_present_before") is not False
            or diagnostic_receipt.get("buildkit_present_after") is not False
            or diagnostic_receipt.get("socket_match_count") != 0
        )
    ):
        raise D0Error("read-only retry diagnostic lineage differs")

    return {
        "host": "john2",
        "disposition": {
            "failed_packet_replay_allowed": False,
            "fresh_operation_identity_required": True,
            "qualification_accepted": False,
            "read_only_diagnostic_only": True,
        },
        "failed_qualification": {
            "packet_sha256": failure_packet["packet_sha256"],
            "packet_file_sha256": sha256_bytes(failure_packet_bytes),
            "packet_signature_file_sha256": sha256_bytes(failure_signature_bytes),
            "report_sha256": failure_report["report_sha256"],
            "report_file_sha256": sha256_bytes(failure_report_bytes),
            "completion_sha256": completion["completion_sha256"],
            "completion_file_sha256": sha256_bytes(completion_bytes),
            "persistence_receipt_sha256": persistence_receipt["receipt_sha256"],
            "persistence_receipt_file_sha256": sha256_bytes(persistence_receipt_bytes),
            "persistence_evidence_sha256": persistence_evidence["evidence_sha256"],
            "persistence_evidence_file_sha256": sha256_bytes(persistence_evidence_bytes),
            "persistence_monitor_sha256": persistence_monitor["monitor_sha256"],
            "persistence_monitor_file_sha256": sha256_bytes(persistence_monitor_bytes),
            "failure_message": failure_report["evidence"]["error"]["message"],
        },
        "post_failure_read_only_proof": {
            "authorization_sha256": authorization["authorization_sha256"],
            "authorization_file_sha256": sha256_bytes(authorization_bytes),
            "authorization_signature_file_sha256": sha256_bytes(authorization_signature_bytes),
            "package_manifest_sha256": package_manifest["manifest_sha256"],
            "package_manifest_file_sha256": sha256_bytes(package_manifest_bytes),
            "package_manifest_signature_file_sha256": sha256_bytes(package_signature_bytes),
            "result_sha256": result_semantic_sha256,
            "result_file_sha256": sha256_bytes(result_bytes),
            "evidence_sha256": diagnostic_evidence["evidence_sha256"],
            "evidence_file_sha256": sha256_bytes(diagnostic_evidence_bytes),
            "receipt_sha256": diagnostic_receipt["receipt_sha256"],
            "receipt_file_sha256": sha256_bytes(diagnostic_receipt_bytes),
            **(
                {
                    "prior_diagnostic_failure": {
                        "evidence_sha256": authorization["prior_diagnostic_failure"][
                            "failure_evidence_sha256"
                        ],
                        "receipt_sha256": authorization["prior_diagnostic_failure"][
                            "failure_receipt_sha256"
                        ],
                        "replay_forbidden": True,
                    },
                    "network_change_classification": diagnostic_receipt[
                        "network_change_classification"
                    ],
                }
                if is_network_diagnostic
                else {}
            ),
            **(
                {
                    "evidence_signature_file_sha256": sha256_bytes(evidence_signature_bytes),
                    "receipt_signature_file_sha256": sha256_bytes(receipt_signature_bytes),
                }
                if evidence_signature_bytes is not None and receipt_signature_bytes is not None
                else {}
            ),
        },
    }


def render_lineage(args: argparse.Namespace) -> dict[str, Any]:
    cardinalities = (
        len(args.bundle),
        len(args.canonical_receipt),
        len(args.target_receipt),
    )
    if len(set(cardinalities)) != 1:
        raise D0Error("lineage input cardinality differs")
    public_key = load_public_key(args.public_key)
    incident_bytes, incident = _json(args.collision_incident, "collision quarantine incident")
    incident_signature_bytes, incident_signature = _json(
        args.collision_incident_signature, "collision quarantine incident signature"
    )
    verify_stdin(public_key, incident_bytes, incident_signature)
    if (
        incident.get("schema_id") != "cascadia.r2-map.d0-collision-quarantine.v1"
        or incident.get("status") != "quarantined"
        or incident.get("incident_sha256") != document_sha256(incident, "incident_sha256")
    ):
        raise D0Error("collision quarantine incident identity differs")
    old_plan_bytes, old_plan = _plan(args.old_plan)
    new_plan_bytes, new_plan = _plan(args.new_plan)
    old_helper = _read(args.old_helper, MAX_HELPER, "old helper archive")
    new_helper = _read(args.new_helper, MAX_HELPER, "new helper archive")
    verify_helper_archive(old_helper)
    verify_helper_archive(new_helper)
    if old_plan["helper_sha256"] != sha256_bytes(old_helper) or new_plan[
        "helper_sha256"
    ] != sha256_bytes(new_helper):
        raise D0Error("lineage plan/helper binding differs")
    retry_lineage = _render_retry_lineage(
        args,
        public_key=public_key,
        old_helper_sha256=sha256_bytes(old_helper),
    )
    entries: list[dict[str, Any]] = []
    for bundle_path, canonical_path, target_path in zip(  # noqa: B905 -- Python 3.9.
        args.bundle, args.canonical_receipt, args.target_receipt
    ):
        bundle = _read(bundle_path, MAX_BUNDLE, "accepted predecessor bundle")
        verified = verify_result_bundle(bundle, public_key=public_key)
        packet = verified["packet"]
        report = verified["report"]
        canonical_bytes, canonical_value = _json(
            canonical_path, "canonical John1 materialization receipt"
        )
        canonical_receipt = validate_materialization_receipt(canonical_value)
        target_bytes, target_value = _json(target_path, "target materialization receipt")
        target_receipt = validate_materialization_receipt(target_value)
        common = (
            canonical_receipt["bundle_sha256"] == sha256_bytes(bundle)
            and canonical_receipt["bundle_size"] == len(bundle)
            and canonical_receipt["manifest_sha256"] == verified["manifest"]["manifest_sha256"]
            and canonical_receipt["packet_sha256"] == packet["packet_sha256"]
            and canonical_receipt["report_sha256"] == report["report_sha256"]
            and canonical_receipt["target_host"] == "john1"
            and target_receipt["bundle_sha256"] == sha256_bytes(bundle)
            and target_receipt["bundle_size"] == len(bundle)
            and target_receipt["manifest_sha256"] == verified["manifest"]["manifest_sha256"]
            and target_receipt["packet_sha256"] == packet["packet_sha256"]
            and target_receipt["report_sha256"] == report["report_sha256"]
        )
        if not common:
            raise D0Error("accepted predecessor materialization binding differs")
        entries.append(
            {
                "cycle_id": packet["cycle_id"],
                "host": packet["host"],
                "phase": report["phase"],
                "operation": report["operation"],
                "old_helper_sha256": packet["helper_sha256"],
                "packet_sha256": packet["packet_sha256"],
                "report_sha256": report["report_sha256"],
                "manifest_sha256": verified["manifest"]["manifest_sha256"],
                "bundle_sha256": sha256_bytes(bundle),
                "bundle_size": len(bundle),
                "canonical_receipt_sha256": canonical_receipt["receipt_sha256"],
                "canonical_receipt_file_sha256": sha256_bytes(canonical_bytes),
                "target_host": target_receipt["target_host"],
                "target_receipt_sha256": target_receipt["receipt_sha256"],
                "target_receipt_file_sha256": sha256_bytes(target_bytes),
            }
        )
    entries.sort(key=lambda item: (item["host"], item["phase"], item["operation"]))
    lineage: dict[str, Any] = {
        "schema_id": LINEAGE_SCHEMA if retry_lineage is not None else LEGACY_LINEAGE_SCHEMA,
        "schema_version": 2 if retry_lineage is not None else 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "old_plan_sha256": old_plan["plan_sha256"],
        "old_plan_file_sha256": sha256_bytes(old_plan_bytes),
        "old_helper_sha256": sha256_bytes(old_helper),
        "new_plan_sha256": new_plan["plan_sha256"],
        "new_plan_file_sha256": sha256_bytes(new_plan_bytes),
        "new_helper_sha256": sha256_bytes(new_helper),
        "collision_incident_sha256": incident["incident_sha256"],
        "collision_incident_file_sha256": sha256_bytes(incident_bytes),
        "collision_incident_signature_file_sha256": sha256_bytes(incident_signature_bytes),
        "accepted_predecessors": entries,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
    }
    if retry_lineage is not None:
        lineage["retry_lineage"] = retry_lineage
    lineage["lineage_sha256"] = document_sha256(lineage, "lineage_sha256")
    _write_new(args.out, canonical_json(lineage))
    return lineage


def _validate_lineage(value: dict[str, Any]) -> dict[str, Any]:
    version = value.get("schema_version")
    schema = value.get("schema_id")
    retry = value.get("retry_lineage")
    if (
        (schema, version) not in {(LEGACY_LINEAGE_SCHEMA, 1), (LINEAGE_SCHEMA, 2)}
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("run_id") != D0_RUN_ID
        or value.get("lineage_sha256") != document_sha256(value, "lineage_sha256")
        or value.get("project_code_executed") is not False
        or value.get("protected_seed_values_opened") is not False
        or not isinstance(value.get("collision_incident_sha256"), str)
        or not isinstance(value.get("collision_incident_file_sha256"), str)
        or not isinstance(value.get("collision_incident_signature_file_sha256"), str)
        or not isinstance(value.get("accepted_predecessors"), list)
        or (version == 1 and retry is not None)
        or (version == 2 and not isinstance(retry, dict))
    ):
        raise D0Error("cross-helper lineage identity differs")
    if version == 2:
        disposition = retry.get("disposition")
        failure = retry.get("failed_qualification")
        proof = retry.get("post_failure_read_only_proof")
        if (
            retry.get("host") != "john2"
            or disposition
            != {
                "failed_packet_replay_allowed": False,
                "fresh_operation_identity_required": True,
                "qualification_accepted": False,
                "read_only_diagnostic_only": True,
            }
            or not isinstance(failure, dict)
            or not isinstance(proof, dict)
        ):
            raise D0Error("cross-helper retry lineage identity differs")
    return value


def _current_epoch_predecessors(lineage: dict[str, Any]) -> list[dict[str, Any]]:
    """Project dependency lineage to work completed by the helper being retired.

    Migration authorization may need the complete accepted dependency prefix,
    including reports already closed by earlier helper transitions.  A new
    transition must not repeat those prior epochs: terminal chain validation
    requires each transaction to appear exactly once.
    """

    old_helper = lineage["old_helper_sha256"]
    entries = lineage["accepted_predecessors"]
    if any(
        not isinstance(item, dict) or not isinstance(item.get("old_helper_sha256"), str)
        for item in entries
    ):
        raise D0Error("cross-helper lineage predecessor helper identity differs")
    current = [item for item in entries if item["old_helper_sha256"] == old_helper]
    keys = [
        (item.get("cycle_id"), item.get("host"), item.get("phase"), item.get("operation"))
        for item in current
    ]
    identities = [
        (item.get("packet_sha256"), item.get("report_sha256"), item.get("bundle_sha256"))
        for item in current
    ]
    if len(set(keys)) != len(keys) or len(set(identities)) != len(identities):
        raise D0Error("current helper epoch lineage is duplicated")
    return current


def render_authorization(args: argparse.Namespace) -> dict[str, Any]:
    lineage_bytes, lineage_value = _json(args.lineage, "cross-helper lineage")
    lineage = _validate_lineage(lineage_value)
    old_helper = _read(args.old_helper, MAX_HELPER, "old helper archive")
    new_helper = _read(args.new_helper, MAX_HELPER, "new helper archive")
    old_verification = verify_helper_archive(old_helper)
    new_verification = verify_helper_archive(new_helper)
    bootstrap_bytes, bootstrap = _json(args.bootstrap_packet, "new bootstrap packet")
    validate_bootstrap_packet(bootstrap)
    current_bytes, current_value = _json(
        args.current_bootstrap_receipt, "current bootstrap receipt"
    )
    current = _bootstrap_receipt(current_value)
    public_key = load_public_key(args.public_key)
    _new_plan_bytes, new_plan = _plan(args.new_plan)
    if (
        bootstrap["host"] != args.host
        or bootstrap["helper"]["sha256"] != sha256_bytes(new_helper)
        or bootstrap["public_key"]["openssh_sha256"] != sha256_bytes(public_key)
        or current["host"] != args.host
        or current["helper_archive_sha256"] != sha256_bytes(old_helper)
        or lineage["old_helper_sha256"] != sha256_bytes(old_helper)
        or lineage["new_helper_sha256"] != sha256_bytes(new_helper)
        or lineage["new_plan_sha256"] != new_plan["plan_sha256"]
    ):
        raise D0Error("helper migration source or target binding differs")
    issued = time.time_ns() // 1_000_000
    home = Path(HOST_HOMES[args.host])
    destination = (
        home
        / ".config/cascadia-r2-d0/superseded"
        / (f"helper-{sha256_bytes(old_helper)[:16]}-to-{sha256_bytes(new_helper)[:16]}")
    )
    authorization: dict[str, Any] = {
        "schema_id": AUTH_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "host": args.host,
        "old_helper_sha256": sha256_bytes(old_helper),
        "old_helper_manifest_sha256": old_verification["manifest_sha256"],
        "new_helper_sha256": sha256_bytes(new_helper),
        "new_helper_manifest_sha256": new_verification["manifest_sha256"],
        "new_plan_sha256": new_plan["plan_sha256"],
        "lineage_sha256": lineage["lineage_sha256"],
        "lineage_file_sha256": sha256_bytes(lineage_bytes),
        "current_bootstrap_receipt_sha256": current["receipt_sha256"],
        "current_bootstrap_receipt_file_sha256": sha256_bytes(current_bytes),
        "new_bootstrap_packet_sha256": bootstrap["packet_sha256"],
        "new_bootstrap_packet_file_sha256": sha256_bytes(bootstrap_bytes),
        "public_key_sha256": sha256_bytes(public_key),
        "installer_sha256": sha256_bytes(
            _read(Path(__file__).resolve(), MAX_JSON, "migration installer")
        ),
        "superseded_destination": str(destination),
        "issued_unix_ms": issued,
        "expires_unix_ms": issued + 24 * 60 * 60 * 1000,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
    }
    authorization["authorization_sha256"] = document_sha256(authorization, "authorization_sha256")
    _write_new(args.out, canonical_json(authorization))
    return authorization


def _validate_authorization(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema_id") != AUTH_SCHEMA
        or value.get("schema_version") != 1
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("run_id") != D0_RUN_ID
        or value.get("host") not in HOST_USERS
        or value.get("authorization_sha256") != document_sha256(value, "authorization_sha256")
        or value.get("project_code_executed") is not False
        or value.get("protected_seed_values_opened") is not False
    ):
        raise D0Error("helper migration authorization identity differs")
    now = time.time_ns() // 1_000_000
    if not value["issued_unix_ms"] <= now <= value["expires_unix_ms"]:
        raise D0Error("helper migration authorization is outside its validity window")
    return value


def _verify_installed_helper(root: Path, archive: bytes) -> None:
    expected: dict[str, tuple[bytes, int]] = {}
    import io

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source:
            if member.name == "helper-source-manifest.json":
                continue
            stream = source.extractfile(member)
            if stream is None:
                raise D0Error("helper archive member is unreadable")
            expected[member.name] = (stream.read(), member.mode)
    for relative, (payload, mode) in expected.items():
        path = root / relative
        observed = path.lstat()
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.getuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != mode
            or _read(path, MAX_HELPER, f"installed helper {relative}") != payload
        ):
            raise D0Error("installed helper differs from the authorized old archive")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if str(relative) in expected or path.is_dir():
            continue
        if "__pycache__" not in relative.parts or path.suffix != ".pyc" or path.is_symlink():
            raise D0Error("installed helper contains an unexpected artifact")


def rotate(args: argparse.Namespace) -> dict[str, Any]:
    authorization_bytes, authorization_value = _json(args.authorization, "migration authorization")
    authorization = _validate_authorization(authorization_value)
    signature_bytes, signature = _json(args.signature, "migration authorization signature")
    public_key = load_public_key(args.public_key)
    verify_stdin(public_key, authorization_bytes, signature)
    lineage_bytes, lineage_value = _json(args.lineage, "cross-helper lineage")
    lineage = _validate_lineage(lineage_value)
    old_helper = _read(args.old_helper, MAX_HELPER, "old helper archive")
    new_helper = _read(args.new_helper, MAX_HELPER, "new helper archive")
    bootstrap_bytes, bootstrap = _json(args.bootstrap_packet, "new bootstrap packet")
    validate_bootstrap_packet(bootstrap)
    current_bytes, current_value = _json(
        args.current_bootstrap_receipt, "current bootstrap receipt"
    )
    current = _bootstrap_receipt(current_value)
    host = authorization["host"]
    if pwd.getpwuid(os.getuid()).pw_name != HOST_USERS[host]:
        raise D0Error("helper migration is running as the wrong kernel owner")
    if (
        authorization["installer_sha256"]
        != sha256_bytes(_read(Path(__file__).resolve(), MAX_JSON, "migration installer"))
        or authorization["old_helper_sha256"] != sha256_bytes(old_helper)
        or authorization["new_helper_sha256"] != sha256_bytes(new_helper)
        or authorization["lineage_sha256"] != lineage["lineage_sha256"]
        or authorization["lineage_file_sha256"] != sha256_bytes(lineage_bytes)
        or authorization["current_bootstrap_receipt_sha256"] != current["receipt_sha256"]
        or authorization["current_bootstrap_receipt_file_sha256"] != sha256_bytes(current_bytes)
        or authorization["new_bootstrap_packet_sha256"] != bootstrap["packet_sha256"]
        or authorization["new_bootstrap_packet_file_sha256"] != sha256_bytes(bootstrap_bytes)
        or authorization["public_key_sha256"] != sha256_bytes(public_key)
    ):
        raise D0Error("helper migration input binding differs")
    verify_helper_archive(old_helper)
    verify_helper_archive(new_helper)
    home = Path(HOST_HOMES[host])
    helper_root = home / ".local/libexec/cascadia-r2-d0/v1"
    key_path = home / ".config/cascadia-r2-d0/public-key"
    receipt_path = home / ".config/cascadia-r2-d0/bootstrap-receipt.json"
    destination = Path(authorization["superseded_destination"])
    if destination.exists() or destination.is_symlink():
        raise D0Error("helper migration superseded destination already exists")
    if _read(key_path, MAX_JSON, "installed campaign key") != public_key:
        raise D0Error("installed campaign key differs")
    if _read(receipt_path, MAX_JSON, "installed bootstrap receipt") != current_bytes:
        raise D0Error("installed bootstrap receipt differs")
    _verify_installed_helper(helper_root, old_helper)
    destination.mkdir(parents=True, mode=0o700)
    moved: list[tuple[Path, Path]] = []
    try:
        for source, name in (
            (helper_root, "helper-v1"),
            (key_path, "public-key"),
            (receipt_path, "bootstrap-receipt.json"),
        ):
            target = destination / name
            os.rename(source, target)
            moved.append((source, target))
        _fsync(destination)
        bootstrap_result = apply_bootstrap(
            bootstrap_bytes,
            authorized_packet_sha256=sha256_bytes(bootstrap_bytes),
            helper_archive=new_helper,
            public_key=public_key,
        )
    except BaseException:
        if helper_root.exists():
            shutil.rmtree(helper_root)
        key_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                os.rename(target, source)
        with __import__("contextlib").suppress(OSError):
            destination.rmdir()
        raise
    finished = time.time_ns() // 1_000_000
    receipt: dict[str, Any] = {
        "schema_id": RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "host": host,
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": sha256_bytes(authorization_bytes),
        "signature_file_sha256": sha256_bytes(signature_bytes),
        "lineage_sha256": lineage["lineage_sha256"],
        "old_helper_sha256": sha256_bytes(old_helper),
        "new_helper_sha256": sha256_bytes(new_helper),
        "old_bootstrap_receipt_sha256": current["receipt_sha256"],
        "new_bootstrap_receipt_sha256": bootstrap_result["receipt_sha256"],
        "superseded_destination": str(destination),
        "finished_unix_ms": finished,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "status": "pass",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    migration_receipt_path = destination / "migration-receipt.json"
    try:
        _write_new(migration_receipt_path, canonical_json(receipt))
        _fsync(destination)
    except BaseException:
        migration_receipt_path.unlink(missing_ok=True)
        if helper_root.exists():
            shutil.rmtree(helper_root)
        key_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                os.rename(target, source)
        with __import__("contextlib").suppress(OSError):
            destination.rmdir()
        raise
    return receipt


def render_transition(args: argparse.Namespace) -> dict[str, Any]:
    """Close the signed rotation receipts over the exact accepted v9c prefix."""

    if not (
        len(args.migration_authorization)
        == len(args.migration_authorization_signature)
        == len(args.migration_receipt)
        == 3
        and len(args.old_bootstrap_record) == len(args.old_bootstrap_signature) == 3
        and len(args.new_bootstrap_record) == len(args.new_bootstrap_signature) == 3
    ):
        raise D0Error("helper transition input cardinality differs")
    public_key = load_public_key(args.public_key)
    _lineage_bytes, lineage_value = _json(args.lineage, "cross-helper lineage")
    lineage = _validate_lineage(lineage_value)
    incident_bytes, incident = _json(args.collision_incident, "collision incident")
    incident_signature_bytes, incident_signature = _json(
        args.collision_incident_signature, "collision incident signature"
    )
    verify_stdin(public_key, incident_bytes, incident_signature)
    if (
        incident.get("incident_sha256") != lineage["collision_incident_sha256"]
        or sha256_bytes(incident_bytes) != lineage["collision_incident_file_sha256"]
        or sha256_bytes(incident_signature_bytes)
        != lineage["collision_incident_signature_file_sha256"]
    ):
        raise D0Error("helper transition collision incident binding differs")
    old_plan_bytes, old_plan = _plan(args.old_plan)
    new_plan_bytes, new_plan = _plan(args.new_plan)
    if (
        lineage["old_plan_sha256"] != old_plan["plan_sha256"]
        or lineage["old_plan_file_sha256"] != sha256_bytes(old_plan_bytes)
        or lineage["new_plan_sha256"] != new_plan["plan_sha256"]
        or lineage["new_plan_file_sha256"] != sha256_bytes(new_plan_bytes)
    ):
        raise D0Error("helper transition plan binding differs")
    positions = {item["key"]: item["sequence"] for item in old_plan["transactions"]}
    report_finished: dict[str, int] = {}
    for bundle_path in args.accepted_bundle:
        bundle = _read(bundle_path, MAX_BUNDLE, "accepted transition bundle")
        verification = verify_result_bundle(bundle, public_key=public_key)
        report = verification["report"]
        report_finished[report["report_sha256"]] = report["finished_unix_ms"]
    current_epoch = _current_epoch_predecessors(lineage)
    accepted: list[dict[str, Any]] = []
    for item in current_epoch:
        key = f"{item['cycle_id']}/{item['host']}/{item['phase']}/{item['operation']}"
        if key not in positions:
            raise D0Error("helper transition accepted transaction is outside old plan")
        accepted.append(
            {
                "plan_sequence": positions[key],
                "cycle_id": item["cycle_id"],
                "host": item["host"],
                "phase": item["phase"],
                "operation": item["operation"],
                "packet_sha256": item["packet_sha256"],
                "report_sha256": item["report_sha256"],
                "bundle_sha256": item["bundle_sha256"],
                "bundle_size": item["bundle_size"],
                "manifest_sha256": item["manifest_sha256"],
                "finished_unix_ms": 0,
            }
        )
    # Sealed reports, rather than the plan, own completion time.
    if set(report_finished) != {item["report_sha256"] for item in current_epoch}:
        raise D0Error("helper transition accepted bundle set differs")
    for item in accepted:
        item["finished_unix_ms"] = report_finished[item["report_sha256"]]
    accepted.sort(key=lambda item: item["plan_sequence"])
    for item in accepted:
        item.pop("plan_sequence")
    accepted_transactions = [dict(item, sequence=index) for index, item in enumerate(accepted, 1)]

    authorizations: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for auth_path, sig_path, receipt_path in zip(  # noqa: B905 -- Python 3.9.
        args.migration_authorization,
        args.migration_authorization_signature,
        args.migration_receipt,
    ):
        auth_bytes, auth_value = _json(auth_path, "migration authorization")
        auth = _validate_authorization(auth_value)
        sig_bytes, sig = _json(sig_path, "migration authorization signature")
        verify_stdin(public_key, auth_bytes, sig)
        receipt_bytes, receipt = _json(receipt_path, "migration receipt")
        if (
            receipt.get("schema_id") != RECEIPT_SCHEMA
            or receipt.get("receipt_sha256") != document_sha256(receipt, "receipt_sha256")
            or receipt.get("authorization_sha256") != auth["authorization_sha256"]
            or receipt.get("old_helper_sha256") != lineage["old_helper_sha256"]
            or receipt.get("new_helper_sha256") != lineage["new_helper_sha256"]
            or receipt.get("status") != "pass"
        ):
            raise D0Error("migration receipt binding differs")
        if auth.get("lineage_sha256") != lineage["lineage_sha256"] or auth.get(
            "lineage_file_sha256"
        ) != sha256_bytes(_lineage_bytes):
            raise D0Error("migration authorization lineage binding differs")
        authorizations.append(
            {
                "host": auth["host"],
                "authorization_sha256": auth["authorization_sha256"],
                "authorization_file_sha256": sha256_bytes(auth_bytes),
                "signature_file_sha256": sha256_bytes(sig_bytes),
            }
        )
        receipts.append(
            {
                "host": receipt["host"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_file_sha256": sha256_bytes(receipt_bytes),
                "authorization_sha256": receipt["authorization_sha256"],
                "old_helper_sha256": receipt["old_helper_sha256"],
                "new_helper_sha256": receipt["new_helper_sha256"],
                "old_bootstrap_receipt_sha256": receipt["old_bootstrap_receipt_sha256"],
                "new_bootstrap_receipt_sha256": receipt["new_bootstrap_receipt_sha256"],
                "finished_unix_ms": receipt["finished_unix_ms"],
            }
        )

    def bootstraps(records: list[Path], signatures: list[Path]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for record_path, signature_path in zip(  # noqa: B905 -- Python 3.9.
            records, signatures
        ):
            record_bytes, _record_value = _json(record_path, "bootstrap record")
            _signature_bytes, signature = _json(signature_path, "bootstrap record signature")
            record = verify_bootstrap_record(record_bytes, signature, public_key=public_key)
            result.append(
                {
                    "host": record["host"],
                    "record_sha256": record["record_sha256"],
                    "record_payload_sha256": sha256_bytes(record_bytes),
                    "record_signature_bundle_sha256": signature["bundle_sha256"],
                    "bootstrap_receipt_sha256": record["bootstrap_receipt_sha256"],
                    "helper_archive_sha256": record["helper_archive_sha256"],
                    "installed_unix_ms": record["installed_unix_ms"],
                }
            )
        return sorted(result, key=lambda item: item["host"])

    transition: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.d0-helper-transition.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "chain_index": args.chain_index,
        "from_plan_sha256": old_plan["plan_sha256"],
        "from_plan_file_sha256": sha256_bytes(old_plan_bytes),
        "from_helper_sha256": lineage["old_helper_sha256"],
        "to_plan_sha256": new_plan["plan_sha256"],
        "to_plan_file_sha256": sha256_bytes(new_plan_bytes),
        "to_helper_sha256": lineage["new_helper_sha256"],
        "collision_incident_sha256": incident["incident_sha256"],
        "collision_incident_file_sha256": sha256_bytes(incident_bytes),
        "accepted_transactions": accepted_transactions,
        "migration_authorizations": sorted(authorizations, key=lambda item: item["host"]),
        "migration_receipts": sorted(receipts, key=lambda item: item["host"]),
        "old_bootstraps": bootstraps(args.old_bootstrap_record, args.old_bootstrap_signature),
        "new_bootstraps": bootstraps(args.new_bootstrap_record, args.new_bootstrap_signature),
        "project_code_executed": False,
        "protected_seed_values_opened": False,
    }
    transition["transition_sha256"] = document_sha256(transition, "transition_sha256")
    validate_helper_transition(transition)
    _write_new(args.out, canonical_json(transition))
    return transition


def render_transition_finalization(args: argparse.Namespace) -> dict[str, Any]:
    """Terminally finalize a provisional transition over its exact old-helper tail."""

    cardinalities = (
        len(args.tail_bundle),
        len(args.canonical_receipt),
        len(args.target_receipt),
    )
    if len(set(cardinalities)) != 1 or len(args.migration_receipt) != 3:
        raise D0Error("helper transition finalization input cardinality differs")
    public_key = load_public_key(args.public_key)

    base_bytes, base_value = _json(args.base_transition, "provisional helper transition")
    base = validate_helper_transition(base_value)
    if base["schema_id"] != HELPER_TRANSITION_SCHEMA:
        raise D0Error("helper transition is already terminally finalized")
    base_signature_bytes, base_signature = _json(
        args.base_transition_signature, "provisional helper transition signature"
    )
    verify_stdin(public_key, base_bytes, base_signature)

    previous_bytes, previous_value = _json(args.previous_transition, "previous helper transition")
    previous = validate_helper_transition(previous_value)
    _previous_signature_bytes, previous_signature = _json(
        args.previous_transition_signature, "previous helper transition signature"
    )
    verify_stdin(public_key, previous_bytes, previous_signature)
    if (
        previous["chain_index"] + 1 != base["chain_index"]
        or previous["to_helper_sha256"] != base["from_helper_sha256"]
        or previous["to_plan_sha256"] != base["from_plan_sha256"]
        or previous["to_plan_file_sha256"] != base["from_plan_file_sha256"]
    ):
        raise D0Error("helper transition finalization previous epoch differs")

    incident_bytes, incident = _json(args.collision_incident, "collision quarantine incident")
    incident_signature_bytes, incident_signature = _json(
        args.collision_incident_signature, "collision quarantine incident signature"
    )
    verify_stdin(public_key, incident_bytes, incident_signature)
    quarantined = incident.get("quarantined_report_sha256s")
    if (
        incident.get("incident_sha256") != base["collision_incident_sha256"]
        or sha256_bytes(incident_bytes) != base["collision_incident_file_sha256"]
        or not isinstance(quarantined, list)
        or any(not isinstance(item, str) for item in quarantined)
    ):
        raise D0Error("helper transition finalization quarantine manifest differs")

    old_plan_bytes, old_plan = _plan(args.old_plan)
    if (
        old_plan["plan_sha256"] != base["from_plan_sha256"]
        or sha256_bytes(old_plan_bytes) != base["from_plan_file_sha256"]
    ):
        raise D0Error("helper transition finalization old plan differs")
    positions = {item["key"]: item["sequence"] for item in old_plan["transactions"]}

    receipt_by_host: dict[str, tuple[bytes, dict[str, Any]]] = {}
    base_receipts = {item["host"]: item for item in base["migration_receipts"]}
    for receipt_path in args.migration_receipt:
        receipt_bytes, receipt = _json(receipt_path, "migration receipt cutoff")
        host = receipt.get("host")
        if (
            host not in HOST_USERS
            or host in receipt_by_host
            or receipt.get("schema_id") != RECEIPT_SCHEMA
            or receipt.get("receipt_sha256") != document_sha256(receipt, "receipt_sha256")
            or receipt.get("status") != "pass"
            or receipt.get("old_helper_sha256") != base["from_helper_sha256"]
            or receipt.get("new_helper_sha256") != base["to_helper_sha256"]
        ):
            raise D0Error("helper transition finalization migration receipt differs")
        expected = base_receipts[host]
        if (
            receipt["receipt_sha256"] != expected["receipt_sha256"]
            or sha256_bytes(receipt_bytes) != expected["receipt_file_sha256"]
            or receipt["finished_unix_ms"] != expected["finished_unix_ms"]
        ):
            raise D0Error("helper transition finalization receipt cutoff binding differs")
        receipt_by_host[host] = (receipt_bytes, receipt)
    if set(receipt_by_host) != set(HOST_USERS):
        raise D0Error("helper transition finalization receipt host set differs")

    old_bootstraps = {item["host"]: item for item in base["old_bootstraps"]}
    new_bootstraps = {item["host"]: item for item in base["new_bootstraps"]}
    cutoffs: list[dict[str, Any]] = []
    for host in sorted(HOST_USERS):
        receipt_bytes, receipt = receipt_by_host[host]
        cutoffs.append(
            {
                "host": host,
                "migration_receipt_sha256": receipt["receipt_sha256"],
                "migration_receipt_file_sha256": sha256_bytes(receipt_bytes),
                "old_bootstrap_installed_unix_ms": old_bootstraps[host]["installed_unix_ms"],
                "new_bootstrap_installed_unix_ms": new_bootstraps[host]["installed_unix_ms"],
                "rotation_finished_unix_ms": receipt["finished_unix_ms"],
            }
        )

    base_accepted = base["accepted_transactions"]
    base_keys = {
        (item["cycle_id"], item["host"], item["phase"], item["operation"]) for item in base_accepted
    }
    base_identities = {
        (item["packet_sha256"], item["report_sha256"], item["bundle_sha256"])
        for item in base_accepted
    }
    last_position = 0
    for item in base_accepted:
        key = f"{item['cycle_id']}/{item['host']}/{item['phase']}/{item['operation']}"
        position = positions.get(key)
        if position is None or position <= last_position:
            raise D0Error("provisional transition transaction plan order differs")
        last_position = position

    tails: list[dict[str, Any]] = []
    for bundle_path, canonical_path, target_path in zip(  # noqa: B905 -- Python 3.9.
        args.tail_bundle, args.canonical_receipt, args.target_receipt
    ):
        bundle = _read(bundle_path, MAX_BUNDLE, "tail transition bundle")
        verified = verify_result_bundle(bundle, public_key=public_key)
        packet = verified["packet"]
        report = verified["report"]
        validate_operation_evidence(packet, report)
        canonical_bytes, canonical_value = _json(
            canonical_path, "tail canonical materialization receipt"
        )
        canonical_receipt = validate_materialization_receipt(canonical_value)
        target_bytes, target_value = _json(target_path, "tail target materialization receipt")
        target_receipt = validate_materialization_receipt(target_value)
        if (
            packet["helper_sha256"] != base["from_helper_sha256"]
            or report["status"] != "pass"
            or report["report_sha256"] in quarantined
            or canonical_receipt["target_host"] != "john1"
            or target_receipt["target_host"] != packet["host"]
            or canonical_receipt["bundle_sha256"] != sha256_bytes(bundle)
            or canonical_receipt["bundle_size"] != len(bundle)
            or canonical_receipt["manifest_sha256"] != verified["manifest"]["manifest_sha256"]
            or canonical_receipt["packet_sha256"] != packet["packet_sha256"]
            or canonical_receipt["report_sha256"] != report["report_sha256"]
            or target_receipt["bundle_sha256"] != sha256_bytes(bundle)
            or target_receipt["bundle_size"] != len(bundle)
            or target_receipt["manifest_sha256"] != verified["manifest"]["manifest_sha256"]
            or target_receipt["packet_sha256"] != packet["packet_sha256"]
            or target_receipt["report_sha256"] != report["report_sha256"]
        ):
            raise D0Error("helper transition finalization tail evidence differs")
        key_tuple = (packet["cycle_id"], packet["host"], report["phase"], report["operation"])
        identity = (packet["packet_sha256"], report["report_sha256"], sha256_bytes(bundle))
        key = "/".join(key_tuple)
        position = positions.get(key)
        cutoff = next(item for item in cutoffs if item["host"] == packet["host"])
        if (
            position is None
            or key_tuple in base_keys
            or identity in base_identities
            or not (
                cutoff["old_bootstrap_installed_unix_ms"]
                < report["finished_unix_ms"]
                <= cutoff["new_bootstrap_installed_unix_ms"]
            )
        ):
            raise D0Error("helper transition finalization tail is duplicated or outside cutoff")
        tails.append(
            {
                "plan_sequence": position,
                "cycle_id": packet["cycle_id"],
                "host": packet["host"],
                "phase": report["phase"],
                "operation": report["operation"],
                "packet_sha256": packet["packet_sha256"],
                "report_sha256": report["report_sha256"],
                "bundle_sha256": sha256_bytes(bundle),
                "bundle_size": len(bundle),
                "manifest_sha256": verified["manifest"]["manifest_sha256"],
                "finished_unix_ms": report["finished_unix_ms"],
                "canonical_receipt_sha256": canonical_receipt["receipt_sha256"],
                "canonical_receipt_file_sha256": sha256_bytes(canonical_bytes),
                "target_host": target_receipt["target_host"],
                "target_receipt_sha256": target_receipt["receipt_sha256"],
                "target_receipt_file_sha256": sha256_bytes(target_bytes),
                "source_helper_sha256": packet["helper_sha256"],
            }
        )
        base_keys.add(key_tuple)
        base_identities.add(identity)
    tails.sort(key=lambda item: item["plan_sequence"])
    if tails and tails[0]["plan_sequence"] <= last_position:
        raise D0Error("helper transition finalization is not append-only in plan order")
    if len({item["plan_sequence"] for item in tails}) != len(tails):
        raise D0Error("helper transition finalization tail plan order is duplicated")

    accepted = [dict(item) for item in base_accepted]
    tail_documents: list[dict[str, Any]] = []
    for sequence, item in enumerate(tails, len(accepted) + 1):
        item = dict(item)
        item.pop("plan_sequence")
        tail_document = dict(item, sequence=sequence)
        tail_documents.append(tail_document)
        accepted.append(
            {
                key: value
                for key, value in tail_document.items()
                if key
                not in {
                    "canonical_receipt_sha256",
                    "canonical_receipt_file_sha256",
                    "target_host",
                    "target_receipt_sha256",
                    "target_receipt_file_sha256",
                    "source_helper_sha256",
                }
            }
        )

    finalization = dict(base)
    finalization.pop("transition_sha256")
    finalization.update(
        {
            "schema_id": FINALIZED_HELPER_TRANSITION_SCHEMA,
            "schema_version": 1,
            "accepted_transactions": accepted,
            "base_transition_sha256": base["transition_sha256"],
            "base_transition_file_sha256": sha256_bytes(base_bytes),
            "base_transition_signature_file_sha256": sha256_bytes(base_signature_bytes),
            "previous_transition_sha256": previous["transition_sha256"],
            "base_accepted_transaction_count": len(base_accepted),
            "tail_transaction_count": len(tail_documents),
            "tail_transactions": tail_documents,
            "collision_incident_signature_file_sha256": sha256_bytes(incident_signature_bytes),
            "migration_receipt_cutoffs": cutoffs,
            "terminal": True,
            "finalized_unix_ms": time.time_ns() // 1_000_000,
        }
    )
    finalization["transition_sha256"] = document_sha256(finalization, "transition_sha256")
    validate_helper_transition(finalization)
    _write_new(args.out, canonical_json(finalization))
    return finalization


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    lineage = commands.add_parser("render-lineage")
    lineage.add_argument("--old-plan", type=Path, required=True)
    lineage.add_argument("--new-plan", type=Path, required=True)
    lineage.add_argument("--old-helper", type=Path, required=True)
    lineage.add_argument("--new-helper", type=Path, required=True)
    lineage.add_argument("--public-key", type=Path, required=True)
    lineage.add_argument("--collision-incident", type=Path, required=True)
    lineage.add_argument("--collision-incident-signature", type=Path, required=True)
    lineage.add_argument("--bundle", type=Path, action="append", default=[])
    lineage.add_argument("--canonical-receipt", type=Path, action="append", default=[])
    lineage.add_argument("--target-receipt", type=Path, action="append", default=[])
    lineage.add_argument("--retry-failure-packet", type=Path)
    lineage.add_argument("--retry-failure-packet-signature", type=Path)
    lineage.add_argument("--retry-failure-report", type=Path)
    lineage.add_argument("--retry-failure-completion", type=Path)
    lineage.add_argument("--retry-failure-persistence-receipt", type=Path)
    lineage.add_argument("--retry-failure-persistence-evidence", type=Path)
    lineage.add_argument("--retry-failure-persistence-monitor", type=Path)
    lineage.add_argument("--retry-diagnostic-authorization", type=Path)
    lineage.add_argument("--retry-diagnostic-authorization-signature", type=Path)
    lineage.add_argument("--retry-diagnostic-package-manifest", type=Path)
    lineage.add_argument("--retry-diagnostic-package-manifest-signature", type=Path)
    lineage.add_argument("--retry-diagnostic-result", type=Path)
    lineage.add_argument("--retry-diagnostic-evidence", type=Path)
    lineage.add_argument("--retry-diagnostic-evidence-signature", type=Path)
    lineage.add_argument("--retry-diagnostic-receipt", type=Path)
    lineage.add_argument("--retry-diagnostic-receipt-signature", type=Path)
    lineage.add_argument("--out", type=Path, required=True)
    authorization = commands.add_parser("render-authorization")
    authorization.add_argument("--host", choices=HOST_USERS, required=True)
    authorization.add_argument("--lineage", type=Path, required=True)
    authorization.add_argument("--old-helper", type=Path, required=True)
    authorization.add_argument("--new-helper", type=Path, required=True)
    authorization.add_argument("--new-plan", type=Path, required=True)
    authorization.add_argument("--bootstrap-packet", type=Path, required=True)
    authorization.add_argument("--current-bootstrap-receipt", type=Path, required=True)
    authorization.add_argument("--public-key", type=Path, required=True)
    authorization.add_argument("--out", type=Path, required=True)
    rotation = commands.add_parser("rotate")
    rotation.add_argument("--authorization", type=Path, required=True)
    rotation.add_argument("--signature", type=Path, required=True)
    rotation.add_argument("--lineage", type=Path, required=True)
    rotation.add_argument("--old-helper", type=Path, required=True)
    rotation.add_argument("--new-helper", type=Path, required=True)
    rotation.add_argument("--bootstrap-packet", type=Path, required=True)
    rotation.add_argument("--current-bootstrap-receipt", type=Path, required=True)
    rotation.add_argument("--public-key", type=Path, required=True)
    transition = commands.add_parser("render-transition")
    transition.add_argument("--lineage", type=Path, required=True)
    transition.add_argument("--chain-index", type=int, required=True)
    transition.add_argument("--collision-incident", type=Path, required=True)
    transition.add_argument("--collision-incident-signature", type=Path, required=True)
    transition.add_argument("--old-plan", type=Path, required=True)
    transition.add_argument("--new-plan", type=Path, required=True)
    transition.add_argument("--public-key", type=Path, required=True)
    transition.add_argument("--accepted-bundle", type=Path, action="append", default=[])
    transition.add_argument("--migration-authorization", type=Path, action="append", required=True)
    transition.add_argument(
        "--migration-authorization-signature", type=Path, action="append", required=True
    )
    transition.add_argument("--migration-receipt", type=Path, action="append", required=True)
    transition.add_argument("--old-bootstrap-record", type=Path, action="append", required=True)
    transition.add_argument("--old-bootstrap-signature", type=Path, action="append", required=True)
    transition.add_argument("--new-bootstrap-record", type=Path, action="append", required=True)
    transition.add_argument("--new-bootstrap-signature", type=Path, action="append", required=True)
    transition.add_argument("--out", type=Path, required=True)
    finalization = commands.add_parser("render-transition-finalization")
    finalization.add_argument("--base-transition", type=Path, required=True)
    finalization.add_argument("--base-transition-signature", type=Path, required=True)
    finalization.add_argument("--previous-transition", type=Path, required=True)
    finalization.add_argument("--previous-transition-signature", type=Path, required=True)
    finalization.add_argument("--old-plan", type=Path, required=True)
    finalization.add_argument("--public-key", type=Path, required=True)
    finalization.add_argument("--collision-incident", type=Path, required=True)
    finalization.add_argument("--collision-incident-signature", type=Path, required=True)
    finalization.add_argument("--migration-receipt", type=Path, action="append", required=True)
    finalization.add_argument("--tail-bundle", type=Path, action="append", default=[])
    finalization.add_argument("--canonical-receipt", type=Path, action="append", default=[])
    finalization.add_argument("--target-receipt", type=Path, action="append", default=[])
    finalization.add_argument("--out", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        result = {
            "render-lineage": render_lineage,
            "render-authorization": render_authorization,
            "rotate": rotate,
            "render-transition": render_transition,
            "render-transition-finalization": render_transition_finalization,
        }[args.command](args)
    except (D0Error, OSError, KeyError, ValueError) as error:
        sys.stderr.write(f"r2-d0-helper-migration: {error}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
