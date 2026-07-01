#!/usr/bin/env python3
"""Render one signed D0 successor packet from verified materialized predecessors."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2_d0.aggregate import validate_operation_evidence, verify_helper_transitions
from r2_d0.bundle import verify_result_bundle
from r2_d0.canonical import (
    D0Error,
    canonical_json,
    load_canonical_json,
    primary_operation,
    render_document,
    sha256_bytes,
    validate_work_packet,
)
from r2_d0.closure import validate_materialization_receipt
from r2_d0.signing import load_public_key, sign_stdin, signature_bytes, verify_stdin
from r2_d0.transport import render_control_envelope
from r2_d0_predecessor_transfer import _validate_authorization

MAX_JSON = 4 * 1024 * 1024
MAX_BUNDLE = 2 * 1024 * 1024 * 1024


def _read_regular(path: Path, maximum: int, label: str) -> bytes:
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot inspect {label}") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != os.getuid()
        or observed.st_mode & 0o022
        or observed.st_size > maximum
    ):
        raise D0Error(f"{label} metadata is unsafe")
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
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != observed.st_size:
        raise D0Error(f"{label} changed while reading")
    return payload


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _predecessor(
    bundle_path: Path,
    receipt_path: Path,
    *,
    public_key: bytes,
    target_host: str,
    canonical_acceptance_receipt_path: Path | None = None,
    transfer_authorization_path: Path | None = None,
    transfer_signature_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = _read_regular(bundle_path, MAX_BUNDLE, "predecessor result bundle")
    verification = verify_result_bundle(bundle, public_key=public_key)
    packet = verification["packet"]
    report = verification["report"]
    validate_operation_evidence(packet, report)
    receipt = validate_materialization_receipt(
        load_canonical_json(
            _read_regular(receipt_path, MAX_JSON, "predecessor materialization receipt"),
            maximum=MAX_JSON,
            label="predecessor materialization receipt",
        )
    )
    if (
        receipt["source_host"] != packet["host"]
        or receipt["target_host"] != target_host
        or receipt["bundle_sha256"] != sha256_bytes(bundle)
        or receipt["bundle_size"] != len(bundle)
        or receipt["manifest_sha256"] != verification["manifest"]["manifest_sha256"]
        or receipt["packet_sha256"] != packet["packet_sha256"]
        or receipt["report_sha256"] != report["report_sha256"]
    ):
        raise D0Error("predecessor materialization binding differs")
    if target_host != "john1":
        if (
            canonical_acceptance_receipt_path is None
            or transfer_authorization_path is None
            or transfer_signature_path is None
        ):
            raise D0Error("remote predecessor transfer provenance is absent")
        canonical_acceptance_bytes = _read_regular(
            canonical_acceptance_receipt_path,
            MAX_JSON,
            "canonical predecessor acceptance receipt",
        )
        canonical_acceptance = validate_materialization_receipt(
            load_canonical_json(
                canonical_acceptance_bytes,
                maximum=MAX_JSON,
                label="canonical predecessor acceptance receipt",
            )
        )
        authorization_bytes = _read_regular(
            transfer_authorization_path,
            MAX_JSON,
            "predecessor transfer authorization",
        )
        authorization = _validate_authorization(
            load_canonical_json(
                authorization_bytes,
                maximum=MAX_JSON,
                label="predecessor transfer authorization",
            )
        )
        transfer_signature = load_canonical_json(
            _read_regular(
                transfer_signature_path,
                MAX_JSON,
                "predecessor transfer authorization signature",
            ),
            maximum=MAX_JSON,
            label="predecessor transfer authorization signature",
        )
        verify_stdin(public_key, authorization_bytes, transfer_signature)
        _validate_transfer_provenance(
            canonical_acceptance_bytes=canonical_acceptance_bytes,
            canonical_acceptance=canonical_acceptance,
            target_receipt=receipt,
            authorization=authorization,
            packet=packet,
            report=report,
            manifest_sha256=verification["manifest"]["manifest_sha256"],
            bundle_sha256=sha256_bytes(bundle),
            bundle_size=len(bundle),
            target_host=target_host,
        )
    relative = (
        f"receipts/{report['report_sha256']}"
        if packet["host"] == target_host
        else f"dependencies/{packet['host']}/{report['report_sha256']}"
    )
    binding = {
        "phase": report["phase"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "operation": report["operation"],
        "status": report["status"],
        "packet_sha256": packet["packet_sha256"],
        "report_sha256": report["report_sha256"],
        "bundle_sha256": sha256_bytes(bundle),
        "bundle_size": len(bundle),
        "manifest_sha256": verification["manifest"]["manifest_sha256"],
        "materialization_receipt_sha256": receipt["receipt_sha256"],
        "finished_unix_ms": report["finished_unix_ms"],
        "receipt_relative": relative,
    }
    return binding, packet, report


def _validate_transfer_provenance(
    *,
    canonical_acceptance_bytes: bytes,
    canonical_acceptance: dict[str, Any],
    target_receipt: dict[str, Any],
    authorization: dict[str, Any],
    packet: dict[str, Any],
    report: dict[str, Any],
    manifest_sha256: str,
    bundle_sha256: str,
    bundle_size: int,
    target_host: str,
) -> None:
    """Bind worker materialization to John1's canonical acceptance transaction."""

    expected_relative = (
        f"receipts/{report['report_sha256']}"
        if packet["host"] == target_host
        else f"dependencies/{packet['host']}/{report['report_sha256']}"
    )
    if (
        canonical_acceptance["source_host"] != packet["host"]
        or canonical_acceptance["target_host"] != "john1"
        or canonical_acceptance["bundle_sha256"] != bundle_sha256
        or canonical_acceptance["bundle_size"] != bundle_size
        or canonical_acceptance["manifest_sha256"] != manifest_sha256
        or canonical_acceptance["packet_sha256"] != packet["packet_sha256"]
        or canonical_acceptance["report_sha256"] != report["report_sha256"]
        or authorization["source_control_host"] != "john1"
        or authorization["source_host"] != packet["host"]
        or authorization["target_host"] != target_host
        or authorization["packet_sha256"] != packet["packet_sha256"]
        or authorization["report_sha256"] != report["report_sha256"]
        or authorization["bundle_sha256"] != bundle_sha256
        or authorization["bundle_size"] != bundle_size
        or authorization["manifest_sha256"] != manifest_sha256
        or authorization["source_materialization_receipt_sha256"]
        != canonical_acceptance["receipt_sha256"]
        or authorization["source_materialization_receipt_file_sha256"]
        != sha256_bytes(canonical_acceptance_bytes)
        or authorization["destination_relative"] != expected_relative
        or target_receipt["source_host"] != packet["host"]
        or target_receipt["target_host"] != target_host
        or target_receipt["destination_relative"] != expected_relative
        or target_receipt["transport_receipt_sha256"] != authorization["authorization_sha256"]
        or target_receipt["bundle_sha256"] != bundle_sha256
        or target_receipt["bundle_size"] != bundle_size
        or target_receipt["manifest_sha256"] != manifest_sha256
        or target_receipt["packet_sha256"] != packet["packet_sha256"]
        or target_receipt["report_sha256"] != report["report_sha256"]
    ):
        raise D0Error("remote predecessor transfer provenance differs")


def render(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.predecessor_bundle) != len(args.materialization_receipt):
        raise D0Error("predecessor bundles and materialization receipts are not paired")
    if not args.predecessor_bundle:
        raise D0Error("at least one predecessor is required")
    remote_provenance = (
        args.canonical_acceptance_receipt,
        args.predecessor_transfer_authorization,
        args.predecessor_transfer_signature,
    )
    if any(remote_provenance) and not all(remote_provenance):
        raise D0Error("remote predecessor transfer provenance is incomplete")
    if all(remote_provenance) and not all(
        len(items) == len(args.predecessor_bundle) for items in remote_provenance
    ):
        raise D0Error("remote predecessor transfer provenance count differs")
    public_key = load_public_key(args.public_key)
    if len(args.helper_transition) != len(args.helper_transition_signature):
        raise D0Error("helper transition/signature count differs")
    helper_transition_pairs = [
        (
            _read_regular(path, MAX_JSON, "helper transition"),
            load_canonical_json(
                _read_regular(signature, MAX_JSON, "helper transition signature"),
                maximum=MAX_JSON,
                label="helper transition signature",
            ),
        )
        for path, signature in zip(  # noqa: B905 -- Apple system Python is 3.9.
            args.helper_transition,
            args.helper_transition_signature,
        )
    ]
    transitions = verify_helper_transitions(helper_transition_pairs, public_key=public_key)
    base = validate_work_packet(
        load_canonical_json(
            _read_regular(args.base_packet, MAX_JSON, "base work packet"),
            maximum=MAX_JSON,
            label="base work packet",
        )
    )
    target_helper = args.helper_sha256 or base["helper_sha256"]
    if target_helper != base["helper_sha256"]:
        if (
            not transitions
            or base["helper_sha256"] not in {item["from_helper_sha256"] for item in transitions}
            or transitions[-1]["to_helper_sha256"] != target_helper
        ):
            raise D0Error("successor helper transition does not bind base and target")
    elif transitions:
        raise D0Error("helper transitions were supplied without a helper change")
    predecessors = []
    for index, (bundle, receipt) in enumerate(
        zip(  # noqa: B905 - lengths are checked above; Apple system Python is 3.9.
            args.predecessor_bundle, args.materialization_receipt
        )
    ):
        provenance = (
            (None, None, None)
            if not all(remote_provenance)
            else tuple(items[index] for items in remote_provenance)
        )
        predecessors.append(
            _predecessor(
                bundle,
                receipt,
                public_key=public_key,
                target_host=base["host"],
                canonical_acceptance_receipt_path=provenance[0],
                transfer_authorization_path=provenance[1],
                transfer_signature_path=provenance[2],
            )[0]
        )
    predecessors.sort(key=lambda item: (item["finished_unix_ms"], item["report_sha256"]))
    artifacts = base["artifacts"]
    if args.artifacts is not None:
        artifacts_document = load_canonical_json(
            _read_regular(args.artifacts, MAX_JSON, "successor artifact projection"),
            maximum=MAX_JSON,
            label="successor artifact projection",
        )
        artifacts = artifacts_document.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts_document) != {"artifacts"}:
            raise D0Error("successor artifact projection shape differs")
    issued = time.time_ns() // 1_000_000
    specification = dict(base)
    specification.pop("packet_sha256")
    embedded_transitions = base.get("helper_transitions", [])
    if transitions:
        embedded_transitions = [
            {"document": document, "signature": dict(signature)}
            for document, (_payload, signature) in zip(  # noqa: B905 -- Python 3.9.
                transitions, helper_transition_pairs
            )
        ]
    operations = (
        [args.operation]
        if isinstance(args.operation, str)
        else list(args.operation)
    )
    specification.update(
        {
            "schema_id": "cascadia.r2-map.d0-runtime-work-packet.v10",
            "schema_version": 10,
            "cycle_id": args.cycle,
            "phase": args.phase,
            "issued_unix_ms": issued,
            "expires_unix_ms": issued + 24 * 60 * 60 * 1000,
            "artifacts": artifacts,
            "helper_sha256": target_helper,
            "helper_transitions": embedded_transitions,
            "allowed_operations": operations,
            "predecessors": predecessors,
        }
    )
    packet_bytes = render_document(specification, kind="work")
    packet = validate_work_packet(json.loads(packet_bytes))
    signature = signature_bytes(sign_stdin(args.private_key, packet_bytes))
    envelope = render_control_envelope(packet_bytes, signature, public_key=public_key)
    output = args.output_root
    _write_new(output / "work-packet.json", packet_bytes)
    _write_new(output / "work-packet-signature.json", signature)
    _write_new(output / "control-envelope.json", envelope)
    receipt = {
        "status": "ready",
        "host": packet["host"],
        "cycle_id": packet["cycle_id"],
        "phase": packet["phase"],
        "operation": primary_operation(packet["host"], packet["phase"], operations),
        "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": sha256_bytes(packet_bytes),
        "signature_file_sha256": sha256_bytes(signature),
        "control_envelope_file_sha256": sha256_bytes(envelope),
        "predecessor_report_sha256": [item["report_sha256"] for item in predecessors],
        "helper_transition_sha256": [item["transition_sha256"] for item in transitions],
    }
    _write_new(output / "ready-receipt.json", canonical_json(receipt))
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base-packet", type=Path, required=True)
    value.add_argument("--predecessor-bundle", type=Path, action="append", required=True)
    value.add_argument("--materialization-receipt", type=Path, action="append", required=True)
    value.add_argument("--canonical-acceptance-receipt", type=Path, action="append")
    value.add_argument("--predecessor-transfer-authorization", type=Path, action="append")
    value.add_argument("--predecessor-transfer-signature", type=Path, action="append")
    value.add_argument("--artifacts", type=Path)
    value.add_argument("--cycle", choices=("qualification", "final-live"), required=True)
    value.add_argument(
        "--phase",
        choices=("preflight", "install", "start", "verify", "rollback", "postflight"),
        required=True,
    )
    value.add_argument("--operation", action="append", required=True)
    value.add_argument("--public-key", type=Path, required=True)
    value.add_argument("--private-key", type=Path, required=True)
    value.add_argument("--helper-sha256")
    value.add_argument("--helper-transition", type=Path, action="append", default=[])
    value.add_argument("--helper-transition-signature", type=Path, action="append", default=[])
    value.add_argument("--output-root", type=Path, required=True)
    return value


def main() -> int:
    try:
        result = render(parser().parse_args())
        sys.stdout.buffer.write(canonical_json(result))
        return 0
    except (D0Error, OSError, ValueError) as error:
        sys.stderr.write(f"r2-d0-render-successor: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
