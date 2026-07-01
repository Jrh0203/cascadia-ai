"""Fixed-command CLI for the standalone R2-MAP D0 infrastructure helper."""

from __future__ import annotations

import argparse
import os
import pwd
import stat
import sys
import time
from pathlib import Path
from typing import Any

from .aggregate import (
    build_final_aggregate,
    validate_operation_evidence,
    verify_final_aggregate,
    verify_helper_transitions,
)
from .artifacts import (
    RegistryClient,
    acquire_core,
    acquire_homebrew_artifacts,
    acquire_scanner_artifacts,
    atomic_install_bytes,
    homebrew_closure_archive,
    install_homebrew_closure,
    install_runtime_supply_archive,
    probe_context,
    runtime_supply_archive,
    smoke_oci_archive,
)
from .authorization import authorize_work_packet
from .bootstrap import apply_bootstrap, build_helper_archive, verify_helper_archive
from .bundle import (
    PERSISTENCE_EVIDENCE_NAME,
    PERSISTENCE_MONITOR_NAME,
    PERSISTENCE_RECEIPT_NAME,
    render_result_bundle_manifest,
    seal_result_bundle,
    verify_draft_transaction_export,
    verify_result_bundle,
)
from .canonical import (
    FROZEN_RUNTIME,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    primary_operation,
    render_document,
    sha256_bytes,
    validate_host_report,
    validate_work_packet,
)
from .closure import (
    build_bootstrap_record,
    build_materialization_receipt,
    validate_materialization_receipt,
    verify_bootstrap_record,
)
from .dashboard import (
    SPEC_SCHEMA,
    publish_dashboard_diagnostic,
    render_diagnostic_spec,
)
from .ingress import install_result_ingress
from .runtime import (
    HOST_HOME,
    REQUIRED_OPERATION,
    CommandRunner,
    ContinuousSwapMonitor,
    buildkit_probe,
    complete_plan,
    docker_accounting_cleanup_probe,
    docker_accounting_inventory_probe,
    egress_socket_inventory_probe,
    format_plan,
    formulas_for_host,
    full_policy_buildkit_probe,
    guest_network_stability_probe,
    host_report,
    host_resource_snapshot,
    install_configs,
    install_homebrew,
    nft_schema_inventory_probe,
    postflight_audit,
    preflight_audit,
    prepare_runtime_environment_paths,
    rollback_runtime,
    runtime_environment,
    start_runtime,
    validate_explicit_runtime_environment,
    verify_positive_runtime,
)
from .signing import (
    load_public_key,
    public_key_fingerprint,
    sign_stdin,
    signature_bytes,
    verify_stdin,
)
from .storage import CANONICAL_ROOT, verify_canonical_storage
from .transport import (
    atomic_write,
    claim_control_execution,
    complete_control_execution,
    control_envelope_path,
    ensure_owner_directory,
    inspect_control_execution,
    install_control_envelope,
    persist_receipt_transaction,
    render_control_envelope,
    verify_control_envelope,
)

MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024**3

CONTROL_COMMAND_BY_OPERATION = {
    "preflight-audit": "preflight",
    "acquire-core": "acquire-core",
    "acquire-smoke": "acquire-smoke",
    "acquire-homebrew-artifacts": "acquire-homebrew-artifacts",
    "acquire-scanner": "acquire-scanner",
    "render-runtime-supply": "render-runtime-supply",
    "materialize-runtime-supply": "materialize-runtime-supply",
    "render-probe-context": "probe-context",
    "install-runtime": "install",
    "start-runtime": "start",
    "verify-runtime": "verify-runtime",
    "rollback-runtime": "rollback",
    "postflight-audit": "postflight",
}


def _read(path: Path, *, maximum: int, label: str) -> bytes:
    if not path.is_absolute():
        raise D0Error(f"{label} path is not absolute")
    current = Path("/")
    for component in path.parts[1:-1]:
        current /= component
        try:
            ancestor = current.lstat()
        except OSError as error:
            raise D0Error(f"cannot inspect {label} ancestor") from error
        if stat.S_ISLNK(ancestor.st_mode) or not stat.S_ISDIR(ancestor.st_mode):
            raise D0Error(f"{label} ancestor is unsafe")
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot inspect {label}") from error
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or observed.st_size > maximum:
        raise D0Error(f"{label} metadata or size differs")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    value = b"".join(chunks)
    if len(value) != observed.st_size:
        raise D0Error(f"{label} changed while reading")
    return value


def _json_file(path: Path, *, label: str, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    return load_canonical_json(
        _read(path, maximum=maximum, label=label), maximum=maximum, label=label
    )


def _dashboard_authoritative_input(
    path: Path,
    *,
    label: str,
    expected_schema_id: str,
) -> str:
    try:
        path.relative_to(CANONICAL_ROOT)
    except ValueError as error:
        raise D0Error(f"{label} is outside the authoritative John1 root") from error
    payload = _read(path, maximum=4 * 1024 * 1024, label=label)
    document = load_canonical_json(payload, maximum=4 * 1024 * 1024, label=label)
    if (
        document.get("campaign_id") != "r2-map-expert-iteration-v1"
        or document.get("run_id") != "d0-runtime-bootstrap-20260618-v1"
        or document.get("schema_id") != expected_schema_id
        or document.get("schema_version") != 1
    ):
        raise D0Error(f"{label} identity differs")
    return sha256_bytes(payload)


def _source_root() -> Path:
    return Path(__file__).absolute().parent.parent


def _active_helper_identity() -> dict[str, Any]:
    archive, receipt = build_helper_archive(_source_root())
    return {"archive": archive, **receipt}


def _local_control_host() -> str:
    user = pwd.getpwuid(os.getuid()).pw_name
    try:
        return {"johnherrick": "john1", "john2": "john2", "john3": "john3"}[user]
    except KeyError as error:
        raise D0Error("local account is not a D0 control target") from error


def _control_authority(
    args: argparse.Namespace,
    *,
    require_inbox_path: bool,
) -> dict[str, Any]:
    cached = getattr(args, "_control_verification", None)
    if isinstance(cached, dict):
        return cached
    path = getattr(args, "control_envelope", None)
    if not isinstance(path, Path):
        raise D0Error("signed control envelope is absent")
    public_key = load_public_key(args.public_key)
    envelope_bytes = _read(
        path,
        maximum=4 * 1024 * 1024,
        label="signed control envelope",
    )
    verification = verify_control_envelope(
        envelope_bytes,
        public_key=public_key,
        target_host=_local_control_host(),
    )
    if require_inbox_path and path != control_envelope_path(verification["packet"]):
        raise D0Error("signed control envelope is outside its fixed local inbox path")
    verification["path"] = str(path)
    verification["envelope_bytes"] = envelope_bytes
    args._control_verification = verification
    return verification


def _authority_packet_preview(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "control_envelope", None) is not None:
        return _control_authority(args, require_inbox_path=True)["packet"]
    return _json_file(args.packet, label="work packet", maximum=1024 * 1024)


def _authorized(args: argparse.Namespace, *, phase: str, operation: str) -> dict[str, Any]:
    helper = _active_helper_identity()
    public_key = load_public_key(args.public_key)
    if getattr(args, "control_envelope", None) is not None:
        control = _control_authority(args, require_inbox_path=True)
        packet_bytes = control["packet_bytes"]
        signature = control["signature_bytes"]
    else:
        if getattr(args, "execute", False):
            raise D0Error("mutating commands require a signed control envelope")
        packet_bytes = _read(args.packet, maximum=1024 * 1024, label="work packet")
        signature = _read(args.signature, maximum=1024 * 1024, label="work packet signature")
    packet = authorize_work_packet(
        packet_bytes,
        signature,
        public_key,
        expected_phase=phase,
        required_operation=operation,
        helper_sha256=helper["archive_sha256"],
        require_full_execution_window=bool(getattr(args, "execute", False)),
    )
    if phase != "preflight" or packet["predecessors"]:
        args._predecessor_reports = _bound_predecessors(args, packet)
    if getattr(args, "execute", False):
        control = _control_authority(args, require_inbox_path=True)
        confirmation = getattr(args, "confirm_envelope_sha256", None)
        if confirmation != control["envelope_sha256"]:
            raise D0Error("--execute requires the exact control-envelope SHA-256 confirmation")
        args._authorized_packet = packet
        args._authorized_packet_bytes = packet_bytes
        args._authorized_signature_bytes = signature
        before = _phase_resource_snapshot(packet)
        _require_zero_swap(before, label="phase start")
        monitor = ContinuousSwapMonitor()
        monitor.start()
        if getattr(args, "_control_claim_needed", False):
            control = _control_authority(args, require_inbox_path=True)
            try:
                claim = claim_control_execution(
                    control["envelope_bytes"],
                    public_key=public_key,
                    target_host=packet["host"],
                )
            except BaseException:
                monitor.stop()
                raise
            args._control_claim = claim
        args._phase_resources_before = before
        args._phase_swap_monitor = monitor
        if packet["host"] == "john1":
            args._physical_storage_prewrite = verify_canonical_storage()
    args._authorized_packet = packet
    args._authorized_packet_bytes = packet_bytes
    args._authorized_signature_bytes = signature
    return packet


def _bound_predecessors(
    args: argparse.Namespace, current_packet: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reopen every sealed predecessor bundle and its materialization receipt."""

    helper = _active_helper_identity()
    public_key = load_public_key(args.public_key)
    expected_fingerprint = public_key_fingerprint(public_key)
    embedded = current_packet.get("helper_transitions", [])
    transitions = verify_helper_transitions(
        [(canonical_json(item["document"]), item["signature"]) for item in embedded],
        public_key=public_key,
    )
    if current_packet["helper_sha256"] != helper["archive_sha256"]:
        raise D0Error("current packet helper differs from the active helper")
    if transitions and transitions[-1]["to_helper_sha256"] != helper["archive_sha256"]:
        raise D0Error("current packet helper transition target differs")
    transition_by_old_helper = {item["from_helper_sha256"]: item for item in transitions}
    reports: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []

    def reopen(
        path: Path,
        *,
        maximum: int,
        label: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        value = _read(path, maximum=maximum, label=label)
        if expected_size is not None and len(value) != expected_size:
            raise D0Error(f"{label} size differs")
        if expected_sha256 is not None and sha256_bytes(value) != expected_sha256:
            raise D0Error(f"{label} SHA-256 differs")
        return value, {
            "transport": "direct-john1-control-edge",
            "path": str(path),
            "size": len(value),
            "sha256": sha256_bytes(value),
            "peer_credentials_present": False,
            "status": "pass",
        }

    for binding in current_packet["predecessors"]:
        directory = Path(current_packet["paths"]["output_root"]) / binding["receipt_relative"]
        archive, archive_reopen = reopen(
            directory / "bundle.tar",
            maximum=MAX_ARTIFACT_BYTES,
            label="sealed predecessor bundle",
            expected_sha256=binding["bundle_sha256"],
            expected_size=binding["bundle_size"],
        )
        verification = verify_result_bundle(archive, public_key=public_key)
        historical = verification["packet"]
        report = verification["report"]
        receipt_bytes, receipt_reopen = reopen(
            directory / "materialization-receipt.json",
            maximum=1024 * 1024,
            label="predecessor materialization receipt",
        )
        materialization = validate_materialization_receipt(
            load_canonical_json(
                receipt_bytes,
                maximum=1024 * 1024,
                label="predecessor materialization receipt",
            )
        )
        validate_operation_evidence(historical, report)
        historical_helper = historical["helper_sha256"]
        transaction = {
            "cycle_id": report["cycle_id"],
            "host": report["host"],
            "phase": report["phase"],
            "operation": report["operation"],
            "packet_sha256": report["packet_sha256"],
            "report_sha256": report["report_sha256"],
            "bundle_sha256": verification["archive_sha256"],
            "bundle_size": verification["archive_size"],
            "manifest_sha256": verification["manifest"]["manifest_sha256"],
            "finished_unix_ms": report["finished_unix_ms"],
        }
        if historical_helper == helper["archive_sha256"]:
            helper_authorized = historical["policy"] == current_packet["policy"]
        else:
            transition = transition_by_old_helper.get(historical_helper)
            accepted = (
                []
                if transition is None
                else [
                    {key: value for key, value in item.items() if key != "sequence"}
                    for item in transition["accepted_transactions"]
                ]
            )
            helper_authorized = bool(
                transition is not None
                and transaction in accepted
                and historical["policy"] == current_packet["policy"]
            )
        if (
            verification["archive_sha256"] != binding["bundle_sha256"]
            or verification["archive_size"] != binding["bundle_size"]
            or verification["manifest"]["manifest_sha256"] != binding["manifest_sha256"]
            or materialization["receipt_sha256"] != binding["materialization_receipt_sha256"]
            or materialization["source_host"] != binding["host"]
            or materialization["target_host"] != current_packet["host"]
            or materialization["operation"] != binding["operation"]
            or materialization["bundle_sha256"] != binding["bundle_sha256"]
            or materialization["bundle_size"] != binding["bundle_size"]
            or materialization["manifest_sha256"] != binding["manifest_sha256"]
            or materialization["packet_sha256"] != binding["packet_sha256"]
            or materialization["report_sha256"] != binding["report_sha256"]
            or materialization["destination_relative"] != binding["receipt_relative"]
            or not (
                binding["finished_unix_ms"]
                <= materialization["materialized_unix_ms"]
                <= current_packet["issued_unix_ms"]
            )
            or not helper_authorized
            or historical["public_key_fingerprint"] != expected_fingerprint
            or historical["campaign_id"] != current_packet["campaign_id"]
            or historical["run_id"] != current_packet["run_id"]
            or historical["limits"] != current_packet["limits"]
            or report["started_unix_ms"] < historical["issued_unix_ms"]
            or report["started_unix_ms"] > historical["expires_unix_ms"]
            or report["finished_unix_ms"] - report["started_unix_ms"]
            > historical["limits"]["timeout_seconds"] * 1000
        ):
            raise D0Error("predecessor authorization lineage differs")
        expected = {
            "cycle_id": report["cycle_id"],
            "host": report["host"],
            "phase": report["phase"],
            "operation": report["operation"],
            "status": report["status"],
            "packet_sha256": report["packet_sha256"],
            "report_sha256": report["report_sha256"],
            "bundle_sha256": verification["archive_sha256"],
            "bundle_size": verification["archive_size"],
            "manifest_sha256": verification["manifest"]["manifest_sha256"],
            "materialization_receipt_sha256": materialization["receipt_sha256"],
            "finished_unix_ms": report["finished_unix_ms"],
            "receipt_relative": binding["receipt_relative"],
        }
        if binding != expected:
            raise D0Error("predecessor binding differs from its authenticated transaction")
        if (
            binding["host"] == current_packet["host"]
            and historical["paths"] != current_packet["paths"]
        ):
            raise D0Error("same-host predecessor path contract drifted")
        reports.append(report)
        materializations.append(
            {
                "binding": dict(binding),
                "receipt": materialization,
                "bundle_reopen": archive_reopen,
                "receipt_reopen": receipt_reopen,
            }
        )
    _validate_dependency_evidence(current_packet, reports)
    args._predecessor_materializations = materializations
    return reports


def _validate_dependency_evidence(packet: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    by_operation = {report["operation"]: report for report in reports if report["status"] == "pass"}

    def installed_identity(operation: str, evidence_key: str) -> dict[str, Any]:
        report = by_operation[operation]
        evidence = report["evidence"]
        installed = evidence.get(evidence_key) if isinstance(evidence, dict) else None
        if not isinstance(installed, dict):
            raise D0Error(f"{operation} required evidence is absent")
        return installed

    bindings = (
        ("acquire-core", "core_image", "core_image"),
        ("acquire-smoke", "smoke_oci", "installed"),
    )
    for operation, artifact_key, _evidence_key in bindings:
        if operation not in by_operation:
            continue
        artifact = packet["artifacts"].get(artifact_key)
        installed = installed_identity(operation, _evidence_key)
        if (
            artifact is None
            or installed.get("size") != artifact["size"]
            or installed.get("sha256") != artifact["sha256"]
        ):
            raise D0Error(f"{operation} receipt does not bind the current artifact")
    scanner_report = by_operation.get("acquire-scanner")
    if scanner_report is not None:
        scanner_supply = scanner_report["evidence"].get("scanner_supply", {})
        scanner_installed = scanner_supply.get("installed", {}).get("oci")
        scanner_artifact = packet["artifacts"].get("scanner_oci")
        if (
            not isinstance(scanner_installed, dict)
            or scanner_artifact is None
            or scanner_installed.get("size") != scanner_artifact["size"]
            or scanner_installed.get("sha256") != scanner_artifact["sha256"]
        ):
            raise D0Error("acquire-scanner receipt does not bind the current scanner OCI")
    render_report = by_operation.get("render-runtime-supply")
    if render_report is not None:
        rendered = render_report["evidence"].get("runtime_supply")
        artifact = packet["artifacts"].get("runtime_supply")
        if (
            not isinstance(rendered, dict)
            or artifact is None
            or rendered.get("archive_size") != artifact["size"]
            or rendered.get("archive_sha256") != artifact["sha256"]
        ):
            raise D0Error("render-runtime-supply receipt does not bind the current archive")


def _bound_preflight(args: argparse.Namespace, current_packet: dict[str, Any]) -> dict[str, Any]:
    reports = getattr(args, "_predecessor_reports", None)
    if not isinstance(reports, list):
        reports = _bound_predecessors(args, current_packet)
    expected_operation = "preflight-audit"
    matches = [
        report
        for report in reports
        if report["host"] == current_packet["host"]
        and report["phase"] == "preflight"
        and report["operation"] == expected_operation
        and report["status"] == "pass"
    ]
    if len(matches) != 1:
        raise D0Error("phase chain does not contain one passing authenticated preflight")
    return matches[0]


def _write_new(path: Path, payload: bytes) -> None:
    ensure_owner_directory(path.parent)
    atomic_write(path, payload)


def _emit(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json(value) + b"\n")


def _phase_report(packet: dict[str, Any], evidence: dict[str, Any], started: int) -> dict[str, Any]:
    status = "rolled-back" if packet["phase"] == "rollback" else "pass"
    return host_report(packet, status=status, evidence=evidence, started_unix_ms=started)


def _phase_operation(packet: dict[str, Any]) -> str:
    host = packet.get("host")
    phase = packet.get("phase")
    operations = packet.get("allowed_operations")
    if (
        not isinstance(host, str)
        or phase not in REQUIRED_OPERATION
        or not isinstance(operations, list)
    ):
        raise D0Error("work packet host or phase is absent")
    return primary_operation(host, phase, operations)


def _persist_report(
    args: argparse.Namespace,
    packet: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Persist a host-local draft for direct, immutable return to John1."""

    packet_bytes = getattr(args, "_authorized_packet_bytes", None)
    signature = getattr(args, "_authorized_signature_bytes", None)
    if not isinstance(packet_bytes, bytes) or not isinstance(signature, bytes):
        raise D0Error("authenticated report persistence lacks packet bytes")
    before = _phase_resource_snapshot(packet)
    _require_zero_swap(before, label="persistence start")
    monitor = ContinuousSwapMonitor()
    monitor.start()
    persisted = persist_receipt_transaction(
        output_root=Path(packet["paths"]["pending_root"]),
        packet_bytes=packet_bytes,
        signature_bytes=signature,
        report=report,
        resource_before=before,
        swap_monitor=monitor,
        resource_snapshot_reader=lambda: _phase_resource_snapshot(packet),
    )
    persistence_resources = persisted["persistence_evidence"]
    if time.time_ns() // 1_000_000 > packet["expires_unix_ms"]:
        raise D0Error("report persistence exceeded the signed validity window")
    return {
        "transaction": persisted,
        "persistence_resources": persistence_resources,
        "materialization_receipt_resources": None,
    }


def _complete_control_result(
    args: argparse.Namespace,
    packet: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    control = _control_authority(args, require_inbox_path=True)
    before = _phase_resource_snapshot(packet)
    _require_zero_swap(before, label="control completion start")
    monitor = ContinuousSwapMonitor()
    monitor.start()

    def finalize(continuous: dict[str, Any]) -> dict[str, Any]:
        after = _phase_resource_snapshot(packet)
        _require_zero_swap(after, label="control completion finish")
        resources = {
            "before": before,
            "after": after,
            "continuous_swap": continuous,
            "status": "pass",
        }
        return complete_control_execution(
            control["envelope_bytes"],
            result,
            public_key=load_public_key(args.public_key),
            target_host=packet["host"],
            resources=resources,
        )

    _continuous, completion = monitor.stop_and_finalize(finalize)
    return completion


def _runner(packet: dict[str, Any]) -> CommandRunner:
    limits = packet["limits"]
    if packet["phase"] in {"install", "start", "verify", "rollback"}:
        prepare_runtime_environment_paths(packet)
    environment = runtime_environment(packet)
    validate_explicit_runtime_environment(packet, environment)
    return CommandRunner(
        environment,
        timeout_seconds=_remaining_signed_seconds(packet),
        output_max_bytes=limits["output_max_bytes"],
        cleanup_reserve_seconds=120,
    )


def _remaining_signed_seconds(packet: dict[str, Any]) -> int:
    import time

    remaining_ms = packet["expires_unix_ms"] - time.time_ns() // 1_000_000
    seconds = min(packet["limits"]["timeout_seconds"], remaining_ms // 1000)
    if seconds <= 0:
        raise D0Error("signed execution window expired before phase completion")
    return seconds


def _phase_resource_snapshot(packet: dict[str, Any]) -> dict[str, Any]:
    """Collect a path-independent resource sample for every authorized mutation."""

    limits = packet["limits"]
    runner = CommandRunner(
        {
            "HOME": HOST_HOME[packet["host"]],
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
        },
        timeout_seconds=min(_remaining_signed_seconds(packet), 30),
        output_max_bytes=min(limits["output_max_bytes"], 1024 * 1024),
    )
    return host_resource_snapshot(runner)


def _require_zero_swap(snapshot: dict[str, Any], *, label: str) -> None:
    if snapshot.get("swap_used_bytes") != 0:
        raise D0Error(f"{label} requires zero host swap use")


def _phase_resource_evidence(
    args: argparse.Namespace,
    packet: dict[str, Any],
    *,
    enforce_after: bool,
) -> dict[str, Any]:
    before = getattr(args, "_phase_resources_before", None)
    if not isinstance(before, dict):
        raise D0Error("authorized mutation lacks its phase-start resource sample")
    after = _phase_resource_snapshot(packet)
    monitor = getattr(args, "_phase_swap_monitor", None)
    if not isinstance(monitor, ContinuousSwapMonitor):
        raise D0Error("authorized mutation lacks continuous swap monitoring")
    continuous = monitor.stop()
    if enforce_after:
        _require_zero_swap(after, label="phase finish")
    return {
        "before": before,
        "after": after,
        "zero_swap_entire_phase": (
            before.get("swap_used_bytes") == 0
            and after.get("swap_used_bytes") == 0
            and continuous["max_used_bytes"] == 0
        ),
        "continuous_swap": continuous,
        "status": "pass" if enforce_after else "observed-after-failure",
    }


def _attach_phase_resource_evidence(
    result: dict[str, Any],
    packet: dict[str, Any],
    phase_resources: dict[str, Any],
) -> dict[str, Any]:
    if result.get("schema_id", "").endswith("host-report.v4"):
        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise D0Error("host report evidence is not an object")
        evidence["phase_resources"] = phase_resources
        result["finished_unix_ms"] = max(
            result["finished_unix_ms"],
            phase_resources["after"]["collected_unix_ms"],
        )
        result["report_sha256"] = document_sha256(result, "report_sha256")
        validate_host_report(result, packet=packet)
        return result
    return {"result": result, "phase_resources": phase_resources}


def _signed_arguments(parser: argparse.ArgumentParser, *, mutating: bool = False) -> None:
    parser.add_argument("--public-key", type=Path, required=True)
    if mutating:
        parser.add_argument("--control-envelope", type=Path, required=True)
        parser.add_argument("--execute", action="store_true", required=True)
        parser.add_argument("--confirm-envelope-sha256", required=True)
    else:
        parser.add_argument("--packet", type=Path, required=True)
        parser.add_argument("--signature", type=Path, required=True)


def _run_live_buildkit_probe(
    command: str,
    packet: dict[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """Dispatch live probe variants while sharing the canonical full-policy core."""

    if command == "live-buildkit-policy-egress-trace-probe":
        return full_policy_buildkit_probe(packet, runner)
    resolver_tls_only = command in {
        "live-buildkit-resolver-tls-probe",
        "live-buildkit-egress-trace-probe",
    }
    egress_trace = command in {
        "live-buildkit-egress-trace-probe",
        "live-buildkit-policy-output-inventory-probe",
        "live-buildkit-policy-attestation-inventory-probe",
    }
    return buildkit_probe(
        packet,
        runner,
        resolver_tls_only=resolver_tls_only,
        egress_trace=egress_trace,
        output_inventory_only=command == "live-buildkit-policy-output-inventory-probe",
        attestation_inventory_only=(
            command == "live-buildkit-policy-attestation-inventory-probe"
        ),
    )


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    import time

    started = getattr(args, "_started_unix_ms", time.time_ns() // 1_000_000)
    command = args.command
    if command == "render-packet":
        spec = _json_file(args.spec, label="packet specification")
        payload = render_document(spec, kind=args.kind)
        _write_new(args.out, payload)
        return {"path": str(args.out), "size": len(payload), "sha256": sha256_bytes(payload)}
    if command == "sign":
        payload = _read(args.payload, maximum=64 * 1024 * 1024, label="signed payload")
        bundle = sign_stdin(args.private_key, payload)
        encoded = signature_bytes(bundle)
        _write_new(args.out, encoded)
        return {"path": str(args.out), "size": len(encoded), "sha256": sha256_bytes(encoded)}
    if command == "verify-signature":
        payload = _read(args.payload, maximum=64 * 1024 * 1024, label="signed payload")
        signature = _json_file(args.signature, label="signature bundle", maximum=1024 * 1024)
        verify_stdin(load_public_key(args.public_key), payload, signature)
        return {"payload_sha256": sha256_bytes(payload), "status": "pass"}
    if command == "render-control-envelope":
        envelope = render_control_envelope(
            _read(args.packet, maximum=1024 * 1024, label="control work packet"),
            _read(
                args.signature,
                maximum=1024 * 1024,
                label="control work packet signature",
            ),
            public_key=load_public_key(args.public_key),
        )
        _write_new(args.out, envelope)
        verification = verify_control_envelope(
            envelope,
            public_key=load_public_key(args.public_key),
        )
        return {
            "path": str(args.out),
            "size": len(envelope),
            "sha256": sha256_bytes(envelope),
            "target_host": verification["packet"]["host"],
            "packet_sha256": verification["packet"]["packet_sha256"],
            "operation": verification["envelope"]["operation"],
            "status": "pass",
        }
    if command == "verify-control-envelope":
        envelope = _read(
            args.envelope,
            maximum=4 * 1024 * 1024,
            label="signed control envelope",
        )
        verification = verify_control_envelope(
            envelope,
            public_key=load_public_key(args.public_key),
            target_host=args.target_host,
        )
        return {
            "envelope": verification["envelope"],
            "packet": verification["packet"],
            "signature": verification["signature"],
            "envelope_size": verification["envelope_size"],
            "envelope_sha256": verification["envelope_sha256"],
            "status": "pass",
        }
    if command == "install-control-envelope":
        if args.target_host != _local_control_host():
            raise D0Error("control envelope may only be installed by its target account")
        envelope = _read(
            args.envelope,
            maximum=4 * 1024 * 1024,
            label="signed control envelope",
        )
        if args.confirm_envelope_sha256 != sha256_bytes(envelope):
            raise D0Error("control-envelope installation confirmation differs")
        return install_control_envelope(
            envelope,
            public_key=load_public_key(args.public_key),
            target_host=args.target_host,
        )
    if command == "build-helper":
        archive, receipt = build_helper_archive(args.source_root)
        _write_new(args.out, archive)
        return receipt
    if command in {
        "live-buildkit-probe",
        "live-buildkit-policy-egress-trace-probe",
        "live-buildkit-policy-output-inventory-probe",
        "live-buildkit-policy-attestation-inventory-probe",
        "live-buildkit-resolver-tls-probe",
        "live-buildkit-egress-trace-probe",
    }:
        packet_bytes = _read(args.packet, maximum=MAX_JSON_BYTES, label="live probe packet")
        packet = validate_work_packet(
            load_canonical_json(packet_bytes, maximum=MAX_JSON_BYTES, label="live probe packet")
        )
        signature = _json_file(
            args.signature,
            label="live probe packet signature",
            maximum=1024 * 1024,
        )
        verify_stdin(load_public_key(args.public_key), packet_bytes, signature)
        if packet["host"] != "john2" or "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("live BuildKit probe packet authority differs")
        environment = runtime_environment(packet)
        validate_explicit_runtime_environment(packet, environment)
        runner = CommandRunner(
            environment,
            timeout_seconds=packet["limits"]["timeout_seconds"],
            output_max_bytes=packet["limits"]["output_max_bytes"],
            cleanup_reserve_seconds=300,
        )
        resolver_tls_only = command in {
            "live-buildkit-resolver-tls-probe",
            "live-buildkit-egress-trace-probe",
        }
        egress_trace = command in {
            "live-buildkit-egress-trace-probe",
            "live-buildkit-policy-egress-trace-probe",
            "live-buildkit-policy-output-inventory-probe",
            "live-buildkit-policy-attestation-inventory-probe",
        }
        output_inventory_only = command == "live-buildkit-policy-output-inventory-probe"
        attestation_inventory_only = (
            command == "live-buildkit-policy-attestation-inventory-probe"
        )
        result = _run_live_buildkit_probe(command, packet, runner)
        return {
            "mode": (
                "resolver-egress-trace"
                if resolver_tls_only and egress_trace
                else "full-output-inventory-egress-trace"
                if egress_trace and output_inventory_only
                else "full-attestation-inventory-egress-trace"
                if egress_trace and attestation_inventory_only
                else "full-attestation-egress-trace"
                if egress_trace
                else "resolver-tls-only"
                if resolver_tls_only
                else "full-attestation"
            ),
            "packet_sha256": packet["packet_sha256"],
            "project_code_executed": False,
            "protected_seed_values_opened": False,
            "result": result,
            "status": "pass",
        }
    if command == "live-nft-schema-inventory-probe":
        packet_bytes = _read(args.packet, maximum=MAX_JSON_BYTES, label="live probe packet")
        packet = validate_work_packet(
            load_canonical_json(packet_bytes, maximum=MAX_JSON_BYTES, label="live probe packet")
        )
        signature = _json_file(
            args.signature,
            label="live probe packet signature",
            maximum=1024 * 1024,
        )
        verify_stdin(load_public_key(args.public_key), packet_bytes, signature)
        if packet["host"] != "john2" or "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("live nftables inventory packet authority differs")
        environment = runtime_environment(packet)
        validate_explicit_runtime_environment(packet, environment)
        runner = CommandRunner(
            environment,
            timeout_seconds=packet["limits"]["timeout_seconds"],
            output_max_bytes=packet["limits"]["output_max_bytes"],
            cleanup_reserve_seconds=300,
        )
        return {
            "mode": "raw-nft-schema-inventory",
            "packet_sha256": packet["packet_sha256"],
            "project_code_executed": False,
            "protected_seed_values_opened": False,
            "result": nft_schema_inventory_probe(packet, runner),
            "scanner_executed": False,
            "status": "pass",
        }
    if command == "live-docker-accounting-inventory-probe":
        packet_bytes = _read(args.packet, maximum=MAX_JSON_BYTES, label="live probe packet")
        packet = validate_work_packet(
            load_canonical_json(packet_bytes, maximum=MAX_JSON_BYTES, label="live probe packet")
        )
        signature = _json_file(
            args.signature,
            label="live probe packet signature",
            maximum=1024 * 1024,
        )
        verify_stdin(load_public_key(args.public_key), packet_bytes, signature)
        if packet["host"] != "john2" or "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("live Docker accounting inventory packet authority differs")
        environment = runtime_environment(packet)
        validate_explicit_runtime_environment(packet, environment)
        runner = CommandRunner(
            environment,
            timeout_seconds=packet["limits"]["timeout_seconds"],
            output_max_bytes=packet["limits"]["output_max_bytes"],
            cleanup_reserve_seconds=300,
        )
        return {
            "mode": "read-only-docker-accounting-inventory",
            "packet_sha256": packet["packet_sha256"],
            "project_code_executed": False,
            "protected_seed_values_opened": False,
            "result": docker_accounting_inventory_probe(packet, runner),
            "scanner_executed": False,
            "status": "pass",
        }
    if command == "live-docker-accounting-cleanup-probe":
        packet_bytes = _read(args.packet, maximum=MAX_JSON_BYTES, label="live probe packet")
        packet = validate_work_packet(
            load_canonical_json(packet_bytes, maximum=MAX_JSON_BYTES, label="live probe packet")
        )
        signature = _json_file(
            args.signature,
            label="live probe packet signature",
            maximum=1024 * 1024,
        )
        verify_stdin(load_public_key(args.public_key), packet_bytes, signature)
        if packet["host"] != "john2" or "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("live Docker accounting cleanup packet authority differs")
        environment = runtime_environment(packet)
        validate_explicit_runtime_environment(packet, environment)
        runner = CommandRunner(
            environment,
            timeout_seconds=packet["limits"]["timeout_seconds"],
            output_max_bytes=packet["limits"]["output_max_bytes"],
            cleanup_reserve_seconds=300,
        )
        return {
            "mode": "exact-scanner-attestation-residue-cleanup",
            "packet_sha256": packet["packet_sha256"],
            "project_code_executed": False,
            "protected_seed_values_opened": False,
            "result": docker_accounting_cleanup_probe(packet, runner),
            "scanner_executed": False,
            "status": "pass",
        }
    if command == "live-egress-socket-inventory-probe":
        packet_bytes = _read(args.packet, maximum=MAX_JSON_BYTES, label="live probe packet")
        packet = validate_work_packet(
            load_canonical_json(packet_bytes, maximum=MAX_JSON_BYTES, label="live probe packet")
        )
        signature = _json_file(
            args.signature,
            label="live probe packet signature",
            maximum=1024 * 1024,
        )
        verify_stdin(load_public_key(args.public_key), packet_bytes, signature)
        if packet["host"] != "john2" or "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("live egress socket inventory packet authority differs")
        environment = runtime_environment(packet)
        validate_explicit_runtime_environment(packet, environment)
        runner = CommandRunner(
            environment,
            timeout_seconds=packet["limits"]["timeout_seconds"],
            output_max_bytes=packet["limits"]["output_max_bytes"],
            cleanup_reserve_seconds=300,
        )
        return {
            "mode": "read-only-egress-socket-inventory",
            "packet_sha256": packet["packet_sha256"],
            "project_code_executed": False,
            "protected_seed_values_opened": False,
            "result": egress_socket_inventory_probe(packet, runner),
            "scanner_executed": False,
            "status": "pass",
        }
    if command == "live-network-state-stability-probe":
        packet_bytes = _read(args.packet, maximum=MAX_JSON_BYTES, label="live probe packet")
        packet = validate_work_packet(
            load_canonical_json(packet_bytes, maximum=MAX_JSON_BYTES, label="live probe packet")
        )
        signature = _json_file(
            args.signature,
            label="live probe packet signature",
            maximum=1024 * 1024,
        )
        verify_stdin(load_public_key(args.public_key), packet_bytes, signature)
        if packet["host"] != "john2" or "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("live network stability packet authority differs")
        environment = runtime_environment(packet)
        validate_explicit_runtime_environment(packet, environment)
        runner = CommandRunner(
            environment,
            timeout_seconds=packet["limits"]["timeout_seconds"],
            output_max_bytes=packet["limits"]["output_max_bytes"],
        )
        return {
            "packet_sha256": packet["packet_sha256"],
            "project_code_executed": False,
            "protected_seed_values_opened": False,
            "result": guest_network_stability_probe(runner),
            "status": "pass",
        }
    if command == "verify-helper":
        archive = _read(args.archive, maximum=16 * 1024 * 1024, label="helper archive")
        return verify_helper_archive(archive)
    if command in {"render-result-manifest", "seal-result-bundle"}:
        packet_bytes = _read(args.packet, maximum=1024 * 1024, label="work packet")
        packet = load_canonical_json(
            packet_bytes,
            maximum=1024 * 1024,
            label="work packet",
        )
        validate_work_packet(packet)
        signature_bytes_value = _read(
            args.signature,
            maximum=1024 * 1024,
            label="work packet signature",
        )
        report_bytes = _read(
            args.report,
            maximum=MAX_ARTIFACT_BYTES,
            label="host report",
        )
        report = load_canonical_json(
            report_bytes,
            maximum=MAX_ARTIFACT_BYTES,
            label="host report",
        )
        validate_host_report(report, packet=packet)
        files = {
            "work-packet.json": packet_bytes,
            "work-packet-signature.json": signature_bytes_value,
            "report.json": report_bytes,
            PERSISTENCE_RECEIPT_NAME: _read(
                args.persistence_receipt,
                maximum=1024 * 1024,
                label="persistence receipt",
            ),
            PERSISTENCE_EVIDENCE_NAME: _read(
                args.persistence_evidence,
                maximum=4 * 1024 * 1024,
                label="persistence evidence",
            ),
            PERSISTENCE_MONITOR_NAME: _read(
                args.persistence_monitor,
                maximum=4 * 1024 * 1024,
                label="persistence monitor",
            ),
        }
        bundle_arguments = {
            "run_id": packet["run_id"],
            "cycle_id": packet["cycle_id"],
            "host": packet["host"],
            "role": packet["role"],
            "packet_sha256": packet["packet_sha256"],
            "created_unix_ms": report["finished_unix_ms"],
        }
        manifest_bytes, context = render_result_bundle_manifest(
            files,
            **bundle_arguments,
        )
        if command == "render-result-manifest":
            _write_new(args.out, manifest_bytes)
            return {
                "path": str(args.out),
                "size": len(manifest_bytes),
                "sha256": sha256_bytes(manifest_bytes),
                "manifest": context["manifest"],
            }
        supplied_manifest = _read(
            args.manifest,
            maximum=4 * 1024 * 1024,
            label="result bundle manifest",
        )
        manifest_signature = _read(
            args.manifest_signature,
            maximum=1024 * 1024,
            label="result bundle manifest signature",
        )
        archive, sealed = seal_result_bundle(
            files,
            manifest_bytes=supplied_manifest,
            manifest_signature_bytes=manifest_signature,
            public_key=load_public_key(args.public_key),
            **bundle_arguments,
        )
        if supplied_manifest != manifest_bytes:
            raise D0Error("result bundle manifest differs from the rendered transaction")
        _write_new(args.out, archive)
        return {
            "path": str(args.out),
            "size": len(archive),
            "sha256": sha256_bytes(archive),
            "sealed": sealed,
        }
    if command in {"render-result-manifest-from-draft", "seal-draft-result-bundle"}:
        draft = _read(
            args.draft_archive,
            maximum=MAX_ARTIFACT_BYTES,
            label="noncanonical draft transaction",
        )
        public_key = load_public_key(args.public_key)
        draft_context = verify_draft_transaction_export(draft, public_key=public_key)
        packet = draft_context["packet"]
        report = draft_context["report"]
        bundle_arguments = {
            "run_id": packet["run_id"],
            "cycle_id": packet["cycle_id"],
            "host": packet["host"],
            "role": packet["role"],
            "packet_sha256": packet["packet_sha256"],
            "created_unix_ms": report["finished_unix_ms"],
        }
        manifest_bytes, context = render_result_bundle_manifest(
            draft_context["files"], **bundle_arguments
        )
        if command == "render-result-manifest-from-draft":
            _write_new(args.out, manifest_bytes)
            return {
                "path": str(args.out),
                "size": len(manifest_bytes),
                "sha256": sha256_bytes(manifest_bytes),
                "manifest": context["manifest"],
                "draft_sha256": sha256_bytes(draft),
            }
        supplied_manifest = _read(
            args.manifest,
            maximum=4 * 1024 * 1024,
            label="result bundle manifest",
        )
        if supplied_manifest != manifest_bytes:
            raise D0Error("result bundle manifest differs from the draft transaction")
        sealed, verification = seal_result_bundle(
            draft_context["files"],
            manifest_bytes=supplied_manifest,
            manifest_signature_bytes=_read(
                args.manifest_signature,
                maximum=1024 * 1024,
                label="result bundle manifest signature",
            ),
            public_key=public_key,
            **bundle_arguments,
        )
        _write_new(args.out, sealed)
        return {
            "path": str(args.out),
            "size": len(sealed),
            "sha256": sha256_bytes(sealed),
            "sealed": verification,
            "draft_sha256": sha256_bytes(draft),
        }
    if command == "render-bootstrap-record":
        packet_bytes = _read(
            args.bootstrap_packet,
            maximum=1024 * 1024,
            label="bootstrap packet",
        )
        receipt = _json_file(
            args.bootstrap_receipt,
            label="bootstrap receipt",
            maximum=1024 * 1024,
        )
        record = build_bootstrap_record(packet_bytes, receipt)
        _write_new(args.out, record)
        return {"path": str(args.out), "size": len(record), "sha256": sha256_bytes(record)}
    if command == "verify-bootstrap-record":
        record_bytes = _read(
            args.record,
            maximum=1024 * 1024,
            label="bootstrap record",
        )
        record_signature = _json_file(
            args.record_signature,
            label="bootstrap record signature",
            maximum=1024 * 1024,
        )
        return verify_bootstrap_record(
            record_bytes,
            record_signature,
            public_key=load_public_key(args.public_key),
        )
    if command == "render-materialization-receipt":
        specification = _json_file(
            args.spec,
            label="materialization receipt specification",
            maximum=1024 * 1024,
        )
        encoded = build_materialization_receipt(**specification)
        _write_new(args.out, encoded)
        return {"path": str(args.out), "size": len(encoded), "sha256": sha256_bytes(encoded)}
    if command == "install-result-ingress":
        archive = _read(
            args.archive,
            maximum=MAX_ARTIFACT_BYTES,
            label="sealed result bundle",
        )
        digest = sha256_bytes(archive)
        if args.confirm_archive_sha256 != digest:
            raise D0Error("sealed bundle execution confirmation differs")
        return install_result_ingress(
            archive,
            public_key=load_public_key(args.public_key),
            campaign_root=CANONICAL_ROOT,
        )
    if command in {"build-final-aggregate", "sign-final-aggregate"}:
        public_key = load_public_key(args.public_key)
        if len(args.bootstrap_record) != len(args.bootstrap_signature):
            raise D0Error("bootstrap record/signature argument cardinality differs")
        bootstrap_records = [
            (
                _read(record, maximum=1024 * 1024, label="bootstrap record"),
                _json_file(
                    signature,
                    label="bootstrap record signature",
                    maximum=1024 * 1024,
                ),
            )
            for record, signature in zip(  # noqa: B905 -- Apple system Python is 3.9.
                args.bootstrap_record,
                args.bootstrap_signature,
            )
        ]
        materialization_receipts = [
            _read(
                path,
                maximum=1024 * 1024,
                label="materialization receipt",
            )
            for path in args.materialization_receipt
        ]
        topology_receipts = [
            _read(path, maximum=4 * 1024 * 1024, label="D0 topology receipt")
            for path in args.topology_receipt
        ]
        if len(args.helper_transition) != len(args.helper_transition_signature):
            raise D0Error("helper transition/signature argument cardinality differs")
        helper_transitions = [
            (
                _read(path, maximum=4 * 1024 * 1024, label="D0 helper transition"),
                _json_file(
                    signature,
                    label="D0 helper transition signature",
                    maximum=1024 * 1024,
                ),
            )
            for path, signature in zip(  # noqa: B905 -- Apple system Python is 3.9.
                args.helper_transition,
                args.helper_transition_signature,
            )
        ]
        archives = [
            _read(path, maximum=MAX_ARTIFACT_BYTES, label="D0 result bundle")
            for path in args.bundle
        ]
        aggregate_bytes = build_final_aggregate(
            archives,
            public_key=public_key,
            created_unix_ms=args.created_unix_ms,
            bootstrap_records=bootstrap_records,
            materialization_receipts=materialization_receipts,
            topology_receipts=topology_receipts,
            helper_transitions=helper_transitions,
        )
        aggregate = load_canonical_json(
            aggregate_bytes,
            maximum=4 * 1024 * 1024,
            label="D0 final aggregate",
        )
        if command == "build-final-aggregate":
            return aggregate
        if args.confirm_aggregate_sha256 != aggregate["aggregate_sha256"]:
            raise D0Error("final aggregate signing confirmation differs")
        signature = sign_stdin(args.private_key, aggregate_bytes)
        verification = verify_final_aggregate(
            aggregate_bytes,
            signature,
            public_key=public_key,
            archives=archives,
            bootstrap_records=bootstrap_records,
            materialization_receipts=materialization_receipts,
            topology_receipts=topology_receipts,
            helper_transitions=helper_transitions,
        )
        return {
            "aggregate": aggregate,
            "signature": signature,
            "verification": verification,
        }
    if command == "verify-final-aggregate":
        aggregate_bytes = _read(
            args.aggregate,
            maximum=4 * 1024 * 1024,
            label="D0 final aggregate",
        )
        signature = _json_file(
            args.aggregate_signature,
            label="D0 final aggregate signature",
            maximum=1024 * 1024,
        )
        archives = [
            _read(path, maximum=MAX_ARTIFACT_BYTES, label="D0 result bundle")
            for path in args.bundle
        ]
        if len(args.bootstrap_record) != len(args.bootstrap_signature):
            raise D0Error("bootstrap record/signature argument cardinality differs")
        bootstrap_records = [
            (
                _read(record, maximum=1024 * 1024, label="bootstrap record"),
                _json_file(
                    signature,
                    label="bootstrap record signature",
                    maximum=1024 * 1024,
                ),
            )
            for record, signature in zip(  # noqa: B905 -- Apple system Python is 3.9.
                args.bootstrap_record,
                args.bootstrap_signature,
            )
        ]
        materialization_receipts = [
            _read(path, maximum=1024 * 1024, label="materialization receipt")
            for path in args.materialization_receipt
        ]
        topology_receipts = [
            _read(path, maximum=4 * 1024 * 1024, label="D0 topology receipt")
            for path in args.topology_receipt
        ]
        if len(args.helper_transition) != len(args.helper_transition_signature):
            raise D0Error("helper transition/signature argument cardinality differs")
        helper_transitions = [
            (
                _read(path, maximum=4 * 1024 * 1024, label="D0 helper transition"),
                _json_file(
                    transition_signature,
                    label="D0 helper transition signature",
                    maximum=1024 * 1024,
                ),
            )
            for path, transition_signature in zip(  # noqa: B905 -- Apple system Python 3.9.
                args.helper_transition,
                args.helper_transition_signature,
            )
        ]
        return verify_final_aggregate(
            aggregate_bytes,
            signature,
            public_key=load_public_key(args.public_key),
            archives=archives,
            bootstrap_records=bootstrap_records,
            materialization_receipts=materialization_receipts,
            topology_receipts=topology_receipts,
            helper_transitions=helper_transitions,
        )
    if command == "describe-probe-context":
        _archive, receipt = probe_context()
        return receipt
    if command == "dashboard-diagnostic":
        d0_state_sha256 = _dashboard_authoritative_input(
            args.d0_state_path,
            label="authoritative D0 state",
            expected_schema_id="cascadia.r2-map.d0-state.v1",
        )
        d0_report_sha256 = _dashboard_authoritative_input(
            args.d0_report_path,
            label="authoritative D0 report",
            expected_schema_id="cascadia.r2-map.d0-report.v1",
        )
        document = load_canonical_json(
            render_diagnostic_spec(
                {
                    "schema_id": SPEC_SCHEMA,
                    "schema_version": 2,
                    "campaign_id": "r2-map-expert-iteration-v1",
                    "run_id": "d0-runtime-bootstrap-20260618-v1",
                    "expected_current_sha256": args.expected_current_sha256,
                    "updated_unix_ms": args.updated_unix_ms,
                    "stale_after_seconds": args.stale_after_seconds,
                    "gate_state": {
                        "d0_gate": args.d0_gate,
                        "w0_gate": args.w0_gate,
                        "d0_state_sha256": d0_state_sha256,
                        "d0_report_sha256": d0_report_sha256,
                        "blocker_codes": sorted(args.blocker_code),
                        "host_gates": {
                            host: {
                                "status": getattr(args, f"{host}_gate"),
                                "state_sha256": getattr(args, f"{host}_state_sha256"),
                                "evidence_sha256": getattr(args, f"{host}_evidence_sha256"),
                                "blocker_codes": sorted(getattr(args, f"{host}_blocker_code")),
                            }
                            for host in ("john1", "john2", "john3")
                        },
                    },
                }
            ),
            maximum=64 * 1024,
            label="dashboard diagnostic spec",
        )
        if args.confirm_spec_sha256 != document["spec_sha256"]:
            raise D0Error("dashboard diagnostic execution confirmation differs")
        return publish_dashboard_diagnostic(document)
    if command == "apply-bootstrap":
        packet = _read(args.packet, maximum=1024 * 1024, label="bootstrap packet")
        archive = _read(args.helper_archive, maximum=16 * 1024 * 1024, label="helper archive")
        public_key = _read(args.public_key, maximum=16 * 1024, label="campaign public key")
        return apply_bootstrap(
            packet,
            authorized_packet_sha256=args.authorized_packet_sha256,
            helper_archive=archive,
            public_key=public_key,
        )
    if command == "plan":
        raw = _json_file(args.packet, label="work packet", maximum=1024 * 1024)
        operation = _phase_operation(raw)
        packet = _authorized(args, phase=raw["phase"], operation=operation)
        plan = complete_plan(packet)
        return {"plan": plan, "shell_display_only": format_plan(plan)}
    if command == "preflight":
        raw = _authority_packet_preview(args)
        packet = _authorized(args, phase="preflight", operation=_phase_operation(raw))
        evidence = preflight_audit(
            packet,
            _runner(packet),
            home=Path.home(),
        )
        return _phase_report(packet, evidence, started)
    if command == "acquire-core":
        packet = _authorized(args, phase="install", operation="acquire-core")
        if packet["host"] != "john2":
            raise D0Error("only John2 may acquire the canonical Colima core")
        ensure_owner_directory(Path(packet["paths"]["core_image"]).parent)
        receipt = acquire_core(Path(packet["paths"]["core_image"]))
        expected = packet["artifacts"]["core_image"]
        if receipt["size"] != expected["size"] or receipt["sha256"] != expected["sha256"]:
            raise D0Error("acquired Colima core differs from the signed packet")
        return _phase_report(
            packet,
            {"core_image": receipt},
            started,
        )
    if command == "acquire-smoke":
        packet = _authorized(args, phase="install", operation="acquire-smoke")
        if packet["host"] != "john2":
            raise D0Error("only John2 may acquire the canonical smoke image")
        objects = RegistryClient().acquire()
        archive, receipt = smoke_oci_archive(*objects)
        expected = packet["artifacts"]["smoke_oci"]
        if expected is not None and (
            len(archive) != expected["size"] or sha256_bytes(archive) != expected["sha256"]
        ):
            raise D0Error("rendered smoke OCI differs from the signed packet")
        ensure_owner_directory(Path(packet["paths"]["smoke_oci"]).parent)
        installed = atomic_install_bytes(Path(packet["paths"]["smoke_oci"]), archive)
        return _phase_report(packet, {"smoke_oci": receipt, "installed": installed}, started)
    if command == "acquire-scanner":
        packet = _authorized(args, phase="install", operation="acquire-scanner")
        if packet["host"] != "john2":
            raise D0Error("only John2 may acquire the BuildKit scanner supply chain")
        evidence = acquire_scanner_artifacts(
            oci_destination=Path(packet["paths"]["scanner_oci"]),
            source_destination=Path(packet["paths"]["scanner_source_archive"]),
            license_destination=Path(packet["paths"]["scanner_license"]),
        )
        expected = packet["artifacts"]["scanner_oci"]
        installed = evidence["installed"]["oci"]
        if expected is not None and (
            installed["size"] != expected["size"] or installed["sha256"] != expected["sha256"]
        ):
            raise D0Error("rendered BuildKit scanner OCI differs from the signed packet")
        return _phase_report(packet, {"scanner_supply": evidence}, started)
    if command == "acquire-homebrew-artifacts":
        packet = _authorized(args, phase="install", operation="acquire-homebrew-artifacts")
        if packet["host"] != "john2":
            raise D0Error("only John2 may acquire the Homebrew artifact supply")
        total = sum(
            FROZEN_RUNTIME[name]["bottle_size"] for name in formulas_for_host(packet["host"])
        )
        if total > packet["limits"]["output_max_bytes"]:
            raise D0Error("Homebrew artifact closure exceeds the signed output limit")
        evidence = acquire_homebrew_artifacts(
            Path(packet["paths"]["homebrew_cache"]),
            formulas_for_host(packet["host"]),
        )
        return _phase_report(packet, evidence, started)
    if command == "render-runtime-supply":
        packet = _authorized(args, phase="install", operation="render-runtime-supply")
        if packet["host"] != "john2":
            raise D0Error("only John2 may render the canonical worker runtime supply")
        formulas = formulas_for_host("john1")
        closure, closure_receipt = homebrew_closure_archive(
            Path(packet["paths"]["homebrew_cache"]), formulas
        )
        ensure_owner_directory(Path(packet["paths"]["homebrew_closure"]).parent)
        closure_install = atomic_install_bytes(Path(packet["paths"]["homebrew_closure"]), closure)
        archive, supply_receipt = runtime_supply_archive(
            _read(
                Path(packet["paths"]["core_image"]),
                maximum=MAX_ARTIFACT_BYTES,
                label="canonical Colima core",
            ),
            _read(
                Path(packet["paths"]["smoke_oci"]),
                maximum=MAX_ARTIFACT_BYTES,
                label="canonical smoke OCI",
            ),
            closure,
            formulas,
        )
        ensure_owner_directory(Path(packet["paths"]["runtime_supply"]).parent)
        supply_install = atomic_install_bytes(Path(packet["paths"]["runtime_supply"]), archive)
        expected = packet["artifacts"]["runtime_supply"]
        if expected is not None and (
            expected["size"] != len(archive) or expected["sha256"] != sha256_bytes(archive)
        ):
            raise D0Error("rendered worker runtime supply differs from the signed packet")
        return _phase_report(
            packet,
            {
                "runtime_supply": supply_receipt,
                "supply_install": supply_install,
                "homebrew_closure": closure_receipt,
                "homebrew_closure_install": closure_install,
            },
            started,
        )
    if command == "materialize-runtime-supply":
        packet = _authorized(args, phase="install", operation="materialize-runtime-supply")
        if packet["host"] not in {"john1", "john3"}:
            raise D0Error("only execution hosts materialize the worker runtime supply")
        expected = packet["artifacts"]["runtime_supply"]
        if expected is None:
            raise D0Error("worker runtime-supply identity is absent")
        archive = _read(
            Path(packet["paths"]["runtime_supply_inbox"]),
            maximum=MAX_ARTIFACT_BYTES,
            label="direct John1 runtime-supply ingress",
        )
        if len(archive) != expected["size"] or sha256_bytes(archive) != expected["sha256"]:
            raise D0Error("direct runtime-supply ingress differs from the signed identity")
        materialized = install_runtime_supply_archive(
            archive,
            runtime_supply_path=Path(packet["paths"]["runtime_supply"]),
            core_path=Path(packet["paths"]["core_image"]),
            smoke_path=Path(packet["paths"]["smoke_oci"]),
            homebrew_closure_path=Path(packet["paths"]["homebrew_closure"]),
            formulas=formulas_for_host("john1"),
        )
        return _phase_report(
            packet,
            {
                "runtime_supply": materialized,
                "direct_ingress": {
                    "source_host": "john2" if packet["host"] == "john1" else "john1",
                    "target_host": packet["host"],
                    "path": packet["paths"]["runtime_supply_inbox"],
                    "size": len(archive),
                    "sha256": sha256_bytes(archive),
                    "peer_credentials_present": False,
                    "status": "pass",
                },
            },
            started,
        )
    if command == "probe-context":
        packet = _authorized(args, phase="install", operation="render-probe-context")
        if packet["host"] != "john2":
            raise D0Error("only John2 may stage the BuildKit probe context")
        archive, receipt = probe_context()
        destination = Path(packet["artifacts"]["probe_context"]["source"])
        ensure_owner_directory(destination.parent)
        installed = atomic_install_bytes(destination, archive)
        if (
            installed["size"] != packet["artifacts"]["probe_context"]["size"]
            or installed["sha256"] != packet["artifacts"]["probe_context"]["sha256"]
        ):
            raise D0Error("rendered probe context differs from the signed packet")
        return _phase_report(packet, {"probe_context": receipt, "installed": installed}, started)
    if command == "install":
        packet = _authorized(args, phase="install", operation="install-runtime")
        closure_install = None
        if packet["host"] in {"john1", "john3"}:
            closure_install = install_homebrew_closure(
                _read(
                    Path(packet["paths"]["homebrew_closure"]),
                    maximum=MAX_ARTIFACT_BYTES,
                    label="worker Homebrew closure",
                ),
                Path(packet["paths"]["homebrew_cache"]),
                formulas_for_host(packet["host"]),
            )
        runner = _runner(packet)
        evidence = {
            "configs": install_configs(packet),
            "homebrew_closure": closure_install,
            "homebrew": install_homebrew(packet, runner),
        }
        return _phase_report(packet, evidence, started)
    if command == "start":
        packet = _authorized(args, phase="start", operation="start-runtime")
        evidence = start_runtime(packet, _runner(packet))
        return _phase_report(packet, evidence, started)
    if command == "verify-runtime":
        packet = _authorized(args, phase="verify", operation="verify-runtime")
        preflight = _bound_preflight(args, packet)
        evidence = verify_positive_runtime(
            packet,
            _runner(packet),
            preflight=preflight["evidence"],
        )
        return _phase_report(packet, evidence, started)
    if command == "rollback":
        packet = _authorized(args, phase="rollback", operation="rollback-runtime")
        preflight = _bound_preflight(args, packet)
        evidence = rollback_runtime(
            packet,
            _runner(packet),
            preflight=preflight["evidence"],
        )
        return _phase_report(packet, evidence, started)
    if command == "postflight":
        raw = _authority_packet_preview(args)
        packet = _authorized(args, phase="postflight", operation=_phase_operation(raw))
        preflight_report = _bound_preflight(args, packet)
        evidence = postflight_audit(
            packet,
            home=Path.home(),
            preflight=preflight_report["evidence"],
        )
        rollback_reports = [
            report
            for report in getattr(args, "_predecessor_reports", [])
            if report["host"] == packet["host"]
            and report["phase"] == "rollback"
            and report["operation"] == "rollback-runtime"
            and report["status"] == "rolled-back"
        ]
        if len(rollback_reports) != 1:
            raise D0Error("postflight lacks one authenticated rollback report")
        return _phase_report(packet, evidence, started)
    if command == "verify-result-bundle":
        archive = _read(args.archive, maximum=MAX_ARTIFACT_BYTES, label="result bundle")
        return verify_result_bundle(archive, public_key=load_public_key(args.public_key))
    raise D0Error("unknown D0 command")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="r2-map-d0-runtime")
    subparsers = root.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-packet")
    render.add_argument("--kind", choices=("bootstrap", "work"), required=True)
    render.add_argument("--spec", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    sign = subparsers.add_parser("sign")
    sign.add_argument("--payload", type=Path, required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--out", type=Path, required=True)
    signature = subparsers.add_parser("verify-signature")
    signature.add_argument("--payload", type=Path, required=True)
    signature.add_argument("--signature", type=Path, required=True)
    signature.add_argument("--public-key", type=Path, required=True)
    render_control = subparsers.add_parser("render-control-envelope")
    render_control.add_argument("--packet", type=Path, required=True)
    render_control.add_argument("--signature", type=Path, required=True)
    render_control.add_argument("--public-key", type=Path, required=True)
    render_control.add_argument("--out", type=Path, required=True)
    verify_control = subparsers.add_parser("verify-control-envelope")
    verify_control.add_argument("--envelope", type=Path, required=True)
    verify_control.add_argument("--public-key", type=Path, required=True)
    verify_control.add_argument("--target-host", choices=("john1", "john2", "john3"))
    install_control = subparsers.add_parser("install-control-envelope")
    install_control.add_argument("--envelope", type=Path, required=True)
    install_control.add_argument("--public-key", type=Path, required=True)
    install_control.add_argument(
        "--target-host",
        choices=("john1", "john2", "john3"),
        required=True,
    )
    install_control.add_argument("--execute", action="store_true", required=True)
    install_control.add_argument("--confirm-envelope-sha256", required=True)
    run_control = subparsers.add_parser("run-control-envelope")
    _signed_arguments(run_control, mutating=True)
    helper = subparsers.add_parser("build-helper")
    helper.add_argument("--source-root", type=Path, default=_source_root())
    helper.add_argument("--out", type=Path, required=True)
    live_probe = subparsers.add_parser("live-buildkit-probe")
    live_probe.add_argument("--packet", type=Path, required=True)
    live_probe.add_argument("--signature", type=Path, required=True)
    live_probe.add_argument("--public-key", type=Path, required=True)
    resolver_tls_probe = subparsers.add_parser("live-buildkit-resolver-tls-probe")
    resolver_tls_probe.add_argument("--packet", type=Path, required=True)
    resolver_tls_probe.add_argument("--signature", type=Path, required=True)
    resolver_tls_probe.add_argument("--public-key", type=Path, required=True)
    egress_trace_probe = subparsers.add_parser("live-buildkit-egress-trace-probe")
    egress_trace_probe.add_argument("--packet", type=Path, required=True)
    egress_trace_probe.add_argument("--signature", type=Path, required=True)
    egress_trace_probe.add_argument("--public-key", type=Path, required=True)
    full_policy_trace_probe = subparsers.add_parser(
        "live-buildkit-policy-egress-trace-probe"
    )
    full_policy_trace_probe.add_argument("--packet", type=Path, required=True)
    full_policy_trace_probe.add_argument("--signature", type=Path, required=True)
    full_policy_trace_probe.add_argument("--public-key", type=Path, required=True)
    full_policy_inventory_probe = subparsers.add_parser(
        "live-buildkit-policy-output-inventory-probe"
    )
    full_policy_inventory_probe.add_argument("--packet", type=Path, required=True)
    full_policy_inventory_probe.add_argument("--signature", type=Path, required=True)
    full_policy_inventory_probe.add_argument("--public-key", type=Path, required=True)
    full_policy_attestation_inventory_probe = subparsers.add_parser(
        "live-buildkit-policy-attestation-inventory-probe"
    )
    full_policy_attestation_inventory_probe.add_argument(
        "--packet", type=Path, required=True
    )
    full_policy_attestation_inventory_probe.add_argument(
        "--signature", type=Path, required=True
    )
    full_policy_attestation_inventory_probe.add_argument(
        "--public-key", type=Path, required=True
    )
    nft_inventory_probe = subparsers.add_parser("live-nft-schema-inventory-probe")
    nft_inventory_probe.add_argument("--packet", type=Path, required=True)
    nft_inventory_probe.add_argument("--signature", type=Path, required=True)
    nft_inventory_probe.add_argument("--public-key", type=Path, required=True)
    docker_accounting_probe = subparsers.add_parser(
        "live-docker-accounting-inventory-probe"
    )
    docker_accounting_probe.add_argument("--packet", type=Path, required=True)
    docker_accounting_probe.add_argument("--signature", type=Path, required=True)
    docker_accounting_probe.add_argument("--public-key", type=Path, required=True)
    docker_accounting_cleanup = subparsers.add_parser(
        "live-docker-accounting-cleanup-probe"
    )
    docker_accounting_cleanup.add_argument("--packet", type=Path, required=True)
    docker_accounting_cleanup.add_argument("--signature", type=Path, required=True)
    docker_accounting_cleanup.add_argument("--public-key", type=Path, required=True)
    egress_socket_inventory = subparsers.add_parser(
        "live-egress-socket-inventory-probe"
    )
    egress_socket_inventory.add_argument("--packet", type=Path, required=True)
    egress_socket_inventory.add_argument("--signature", type=Path, required=True)
    egress_socket_inventory.add_argument("--public-key", type=Path, required=True)
    network_stability_probe = subparsers.add_parser(
        "live-network-state-stability-probe"
    )
    network_stability_probe.add_argument("--packet", type=Path, required=True)
    network_stability_probe.add_argument("--signature", type=Path, required=True)
    network_stability_probe.add_argument("--public-key", type=Path, required=True)
    verify_helper = subparsers.add_parser("verify-helper")
    verify_helper.add_argument("--archive", type=Path, required=True)
    manifest = subparsers.add_parser("render-result-manifest")
    manifest.add_argument("--packet", type=Path, required=True)
    manifest.add_argument("--signature", type=Path, required=True)
    manifest.add_argument("--report", type=Path, required=True)
    manifest.add_argument("--persistence-receipt", type=Path, required=True)
    manifest.add_argument("--persistence-evidence", type=Path, required=True)
    manifest.add_argument("--persistence-monitor", type=Path, required=True)
    manifest.add_argument("--out", type=Path, required=True)
    seal = subparsers.add_parser("seal-result-bundle")
    seal.add_argument("--packet", type=Path, required=True)
    seal.add_argument("--signature", type=Path, required=True)
    seal.add_argument("--report", type=Path, required=True)
    seal.add_argument("--persistence-receipt", type=Path, required=True)
    seal.add_argument("--persistence-evidence", type=Path, required=True)
    seal.add_argument("--persistence-monitor", type=Path, required=True)
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--manifest-signature", type=Path, required=True)
    seal.add_argument("--public-key", type=Path, required=True)
    seal.add_argument("--out", type=Path, required=True)
    manifest_from_draft = subparsers.add_parser("render-result-manifest-from-draft")
    manifest_from_draft.add_argument("--draft-archive", type=Path, required=True)
    manifest_from_draft.add_argument("--public-key", type=Path, required=True)
    manifest_from_draft.add_argument("--out", type=Path, required=True)
    seal_draft = subparsers.add_parser("seal-draft-result-bundle")
    seal_draft.add_argument("--draft-archive", type=Path, required=True)
    seal_draft.add_argument("--manifest", type=Path, required=True)
    seal_draft.add_argument("--manifest-signature", type=Path, required=True)
    seal_draft.add_argument("--public-key", type=Path, required=True)
    seal_draft.add_argument("--out", type=Path, required=True)
    bootstrap_record = subparsers.add_parser("render-bootstrap-record")
    bootstrap_record.add_argument("--bootstrap-packet", type=Path, required=True)
    bootstrap_record.add_argument("--bootstrap-receipt", type=Path, required=True)
    bootstrap_record.add_argument("--out", type=Path, required=True)
    verify_bootstrap = subparsers.add_parser("verify-bootstrap-record")
    verify_bootstrap.add_argument("--record", type=Path, required=True)
    verify_bootstrap.add_argument("--record-signature", type=Path, required=True)
    verify_bootstrap.add_argument("--public-key", type=Path, required=True)
    materialization = subparsers.add_parser("render-materialization-receipt")
    materialization.add_argument("--spec", type=Path, required=True)
    materialization.add_argument("--out", type=Path, required=True)
    ingress = subparsers.add_parser("install-result-ingress")
    ingress.add_argument("--archive", type=Path, required=True)
    ingress.add_argument("--public-key", type=Path, required=True)
    ingress.add_argument("--execute", action="store_true", required=True)
    ingress.add_argument("--confirm-archive-sha256", required=True)
    for name in ("build-final-aggregate", "sign-final-aggregate"):
        aggregate = subparsers.add_parser(name)
        aggregate.add_argument("--bundle", type=Path, action="append", required=True)
        aggregate.add_argument("--public-key", type=Path, required=True)
        aggregate.add_argument("--created-unix-ms", type=int, required=True)
        aggregate.add_argument("--bootstrap-record", type=Path, action="append", required=True)
        aggregate.add_argument(
            "--bootstrap-signature",
            type=Path,
            action="append",
            required=True,
        )
        aggregate.add_argument(
            "--materialization-receipt",
            type=Path,
            action="append",
            required=True,
        )
        aggregate.add_argument(
            "--topology-receipt",
            type=Path,
            action="append",
            required=True,
        )
        aggregate.add_argument("--helper-transition", type=Path, action="append", default=[])
        aggregate.add_argument(
            "--helper-transition-signature", type=Path, action="append", default=[]
        )
        if name == "sign-final-aggregate":
            aggregate.add_argument("--private-key", type=Path, required=True)
            aggregate.add_argument("--confirm-aggregate-sha256", required=True)
    verify_aggregate = subparsers.add_parser("verify-final-aggregate")
    verify_aggregate.add_argument("--aggregate", type=Path, required=True)
    verify_aggregate.add_argument("--aggregate-signature", type=Path, required=True)
    verify_aggregate.add_argument("--bundle", type=Path, action="append", required=True)
    verify_aggregate.add_argument("--public-key", type=Path, required=True)
    verify_aggregate.add_argument("--bootstrap-record", type=Path, action="append", required=True)
    verify_aggregate.add_argument(
        "--bootstrap-signature",
        type=Path,
        action="append",
        required=True,
    )
    verify_aggregate.add_argument(
        "--materialization-receipt",
        type=Path,
        action="append",
        required=True,
    )
    verify_aggregate.add_argument(
        "--topology-receipt",
        type=Path,
        action="append",
        required=True,
    )
    verify_aggregate.add_argument("--helper-transition", type=Path, action="append", default=[])
    verify_aggregate.add_argument(
        "--helper-transition-signature", type=Path, action="append", default=[]
    )
    subparsers.add_parser("describe-probe-context")
    dashboard = subparsers.add_parser("dashboard-diagnostic")
    dashboard.add_argument("--d0-gate", choices=("red", "green"), required=True)
    dashboard.add_argument("--w0-gate", choices=("red", "green"), required=True)
    dashboard.add_argument("--d0-state-path", type=Path, required=True)
    dashboard.add_argument("--d0-report-path", type=Path, required=True)
    dashboard.add_argument("--blocker-code", action="append", default=[])
    for host in ("john1", "john2", "john3"):
        dashboard.add_argument(f"--{host}-gate", choices=("red", "green"), required=True)
        dashboard.add_argument(f"--{host}-state-sha256", required=True)
        dashboard.add_argument(f"--{host}-evidence-sha256", required=True)
        dashboard.add_argument(f"--{host}-blocker-code", action="append", default=[])
    dashboard.add_argument("--expected-current-sha256", required=True)
    dashboard.add_argument("--updated-unix-ms", type=int, required=True)
    dashboard.add_argument("--stale-after-seconds", type=int, default=3600)
    dashboard.add_argument("--execute", action="store_true", required=True)
    dashboard.add_argument("--confirm-spec-sha256", required=True)
    bootstrap = subparsers.add_parser("apply-bootstrap")
    bootstrap.add_argument("--packet", type=Path, required=True)
    bootstrap.add_argument("--authorized-packet-sha256", required=True)
    bootstrap.add_argument("--helper-archive", type=Path, required=True)
    bootstrap.add_argument("--public-key", type=Path, required=True)
    plan = subparsers.add_parser("plan")
    _signed_arguments(plan)
    preflight = subparsers.add_parser("preflight")
    _signed_arguments(preflight, mutating=True)
    for name in (
        "acquire-core",
        "acquire-smoke",
        "acquire-homebrew-artifacts",
        "acquire-scanner",
        "render-runtime-supply",
        "materialize-runtime-supply",
        "probe-context",
        "install",
        "start",
        "rollback",
    ):
        selected = subparsers.add_parser(name)
        _signed_arguments(selected, mutating=True)
    verify_runtime = subparsers.add_parser("verify-runtime")
    _signed_arguments(verify_runtime, mutating=True)
    postflight = subparsers.add_parser("postflight")
    _signed_arguments(postflight, mutating=True)
    verify_result = subparsers.add_parser("verify-result-bundle")
    verify_result.add_argument("--archive", type=Path, required=True)
    verify_result.add_argument("--public-key", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        sys.stderr.write("r2-map-d0-runtime: invoke with Python isolation flags -I -S -B\n")
        return 2
    arguments = parser().parse_args(argv)
    if arguments.command == "run-control-envelope":
        try:
            control = _control_authority(arguments, require_inbox_path=True)
            operation = control["envelope"]["operation"]
            try:
                arguments.command = CONTROL_COMMAND_BY_OPERATION[operation]
            except KeyError as error:
                raise D0Error(
                    "control envelope operation has no direct phase dispatcher"
                ) from error
        except D0Error as error:
            sys.stderr.write(f"r2-map-d0-runtime: {error}\n")
            return 2
    if getattr(arguments, "control_envelope", None) is not None and getattr(
        arguments, "execute", False
    ):
        try:
            control = _control_authority(arguments, require_inbox_path=True)
            inspected = inspect_control_execution(
                control["envelope_bytes"],
                public_key=load_public_key(arguments.public_key),
                target_host=control["packet"]["host"],
            )
            if inspected["state"] == "completed":
                replay = inspected["completion"]["result"]
                _emit(replay)
                return 2 if replay["host_report"]["status"] == "fail" else 0
            if inspected["state"] != "available":
                raise D0Error(
                    "control envelope has an incomplete prior execution claim; "
                    "issue a signed recovery or rollback packet"
                )
            arguments._control_claim_needed = True
        except D0Error as error:
            sys.stderr.write(f"r2-map-d0-runtime: {error}\n")
            return 2
    import time

    arguments._started_unix_ms = time.time_ns() // 1_000_000
    try:
        result = _execute(arguments)
        storage = getattr(arguments, "_physical_storage_prewrite", None)
        if storage is not None and result.get("schema_id", "").endswith("host-report.v4"):
            result["evidence"]["physical_storage_prewrite"] = storage
        packet = getattr(arguments, "_authorized_packet", None)
        if packet is not None and getattr(arguments, "execute", False):
            phase_resources = _phase_resource_evidence(
                arguments,
                packet,
                enforce_after=True,
            )
            result = _attach_phase_resource_evidence(result, packet, phase_resources)
        if storage is not None and not result.get("schema_id", "").endswith("host-report.v4"):
            result["physical_storage_prewrite"] = storage
        if packet is not None and result.get("schema_id", "").endswith("host-report.v4"):
            persistence = _persist_report(arguments, packet, result)
            result = {"host_report": result, "persistence": persistence}
            if getattr(arguments, "_control_claim", None) is not None:
                _complete_control_result(arguments, packet, result)
        _emit(result)
    except D0Error as error:
        packet = getattr(arguments, "_authorized_packet", None)
        if packet is not None and getattr(arguments, "execute", False):
            evidence: dict[str, Any] = {
                "error": {"type": "D0Error", "message": str(error)},
            }
            storage = getattr(arguments, "_physical_storage_prewrite", None)
            if storage is not None:
                evidence["physical_storage_prewrite"] = storage
            try:
                evidence["phase_resources"] = _phase_resource_evidence(
                    arguments,
                    packet,
                    enforce_after=False,
                )
            except D0Error as resource_error:
                before = getattr(arguments, "_phase_resources_before", None)
                evidence["phase_resources"] = {
                    "before": before if isinstance(before, dict) else None,
                    "after": None,
                    "after_error": str(resource_error),
                    "zero_swap_entire_phase": False,
                    "status": "resource-snapshot-failed",
                }
            failure = host_report(
                packet,
                status="fail",
                evidence=evidence,
                started_unix_ms=arguments._started_unix_ms,
            )
            try:
                persistence = _persist_report(arguments, packet, failure)
                failure_result = {"host_report": failure, "persistence": persistence}
                if getattr(arguments, "_control_claim", None) is not None:
                    _complete_control_result(arguments, packet, failure_result)
                _emit(failure_result)
            except D0Error as persistence_error:
                sys.stderr.write(
                    f"r2-map-d0-runtime: failure receipt persistence failed: {persistence_error}\n"
                )
        sys.stderr.write(f"r2-map-d0-runtime: {error}\n")
        return 2
    return 0
