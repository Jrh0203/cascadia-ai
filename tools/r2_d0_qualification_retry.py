#!/usr/bin/env python3
"""Render one fresh, signed D0 qualification retry from a sealed failure."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r2_d0.aggregate import validate_helper_transition, verify_helper_transitions
from r2_d0.canonical import (
    CAMPAIGN_ID,
    D0_RUN_ID,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    render_document,
    sha256_bytes,
    validate_host_report,
    validate_work_packet,
)
from r2_d0.signing import (
    load_public_key,
    sign_stdin,
    signature_bytes,
    verify_stdin,
)
from r2_d0.transport import render_control_envelope, verify_control_envelope

MAX_JSON = 128 * 1024 * 1024


def _read(path: Path, label: str) -> bytes:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != os.getuid()
        or observed.st_mode & 0o022
        or observed.st_size > MAX_JSON
    ):
        raise D0Error(f"{label} metadata is unsafe")
    return path.read_bytes()


def _json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read(path, label)
    value = load_canonical_json(payload, maximum=MAX_JSON, label=label)
    if not isinstance(value, dict):
        raise D0Error(f"{label} is not an object")
    return payload, value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_lineage(
    lineage: dict[str, Any],
    *,
    failed_packet: dict[str, Any],
    failed_report: dict[str, Any],
    target_transition: dict[str, Any],
) -> dict[str, Any]:
    retry = lineage.get("retry_lineage")
    failure = retry.get("failed_qualification") if isinstance(retry, dict) else None
    proof = retry.get("post_failure_read_only_proof") if isinstance(retry, dict) else None
    if (
        lineage.get("schema_id") != "cascadia.r2-map.d0-cross-helper-lineage.v2"
        or lineage.get("schema_version") != 2
        or lineage.get("campaign_id") != CAMPAIGN_ID
        or lineage.get("run_id") != D0_RUN_ID
        or lineage.get("lineage_sha256") != document_sha256(lineage, "lineage_sha256")
        or lineage.get("old_helper_sha256") != failed_packet["helper_sha256"]
        or lineage.get("new_helper_sha256") != target_transition["to_helper_sha256"]
        or lineage.get("new_plan_sha256") != target_transition["to_plan_sha256"]
        or not isinstance(retry, dict)
        or retry.get("host") != "john2"
        or retry.get("disposition")
        != {
            "failed_packet_replay_allowed": False,
            "fresh_operation_identity_required": True,
            "qualification_accepted": False,
            "read_only_diagnostic_only": True,
        }
        or not isinstance(failure, dict)
        or failure.get("packet_sha256") != failed_packet["packet_sha256"]
        or failure.get("report_sha256") != failed_report["report_sha256"]
        or not isinstance(proof, dict)
        or proof.get("result_sha256")
        != "353b1d10e31a9e891f2274a13c378f00b2f08d11c2204086ca8ed9e94517a0a4"
        or proof.get("evidence_sha256")
        != "ae490410796cca00214fbf435531a16b9dbc6f9b81dcf8a71b9e3d607f2dfc79"
        or proof.get("receipt_sha256")
        != "f16353df22c211cd3d5191001dfa7f18cd1ddc8436b08620a0778db6f3f80137"
        or proof.get("network_change_classification")
        != "docker-owned-lazy-default-bridge-and-firewall-lifecycle"
        or proof.get("prior_diagnostic_failure", {}).get("replay_forbidden") is not True
    ):
        raise D0Error("qualification retry lineage binding differs")
    return proof


def render(args: argparse.Namespace) -> dict[str, Any]:
    _failed_packet_bytes, failed_packet = _json(args.failed_packet, "failed work packet")
    validate_work_packet(failed_packet)
    completion_bytes, completion = _json(args.failure_completion, "failure completion")
    failed_report = completion.get("result", {}).get("host_report")
    if (
        failed_packet["host"] != "john2"
        or failed_packet["cycle_id"] != "qualification"
        or failed_packet["phase"] != "verify"
        or not isinstance(failed_report, dict)
        or completion.get("schema_id") != "cascadia.r2-map.d0-control-completion.v2"
        or completion.get("completion_sha256") != document_sha256(completion, "completion_sha256")
        or completion.get("packet_sha256") != failed_packet["packet_sha256"]
        or completion.get("report_sha256") != failed_report.get("report_sha256")
        or completion.get("execution_status") != "failed"
        or completion.get("status") != "pass"
    ):
        raise D0Error("qualification retry failure identity differs")
    validate_host_report(failed_report, packet=failed_packet)

    transition_bytes, transition = _json(args.helper_transition, "helper transition")
    transition = validate_helper_transition(transition)
    transition_signature_bytes, transition_signature = _json(
        args.helper_transition_signature, "helper transition signature"
    )
    public_key = load_public_key(args.public_key)
    verify_stdin(public_key, transition_bytes, transition_signature)
    embedded = list(failed_packet["helper_transitions"])
    chain_inputs = [(canonical_json(item["document"]), item["signature"]) for item in embedded] + [
        (transition_bytes, transition_signature)
    ]
    verified_chain = verify_helper_transitions(chain_inputs, public_key=public_key)
    if (
        transition["terminal"] is not True
        or transition["from_helper_sha256"] != failed_packet["helper_sha256"]
        or transition["chain_index"] != len(embedded) + 1
        or verified_chain[-1] != transition
    ):
        raise D0Error("qualification retry helper transition differs")

    lineage_bytes, lineage = _json(args.lineage, "retry lineage")
    proof = _validate_lineage(
        lineage,
        failed_packet=failed_packet,
        failed_report=failed_report,
        target_transition=transition,
    )
    plan_bytes, plan = _json(args.execution_plan, "execution plan")
    if (
        plan.get("schema_id") != "cascadia.r2-map.d0-execution-plan.v1"
        or plan.get("plan_sha256") != document_sha256(plan, "plan_sha256")
        or plan.get("plan_sha256") != transition["to_plan_sha256"]
        or sha256_bytes(plan_bytes) != transition["to_plan_file_sha256"]
        or plan.get("helper_sha256") != transition["to_helper_sha256"]
        or plan.get("d0_status") != "red"
    ):
        raise D0Error("qualification retry promoted plan differs")

    issued = args.issued_unix_ms or time.time_ns() // 1_000_000
    specification = dict(failed_packet)
    specification.pop("packet_sha256")
    specification.update(
        {
            "issued_unix_ms": issued,
            "expires_unix_ms": issued + 24 * 60 * 60 * 1000,
            "helper_sha256": transition["to_helper_sha256"],
            "helper_transitions": [
                *embedded,
                {"document": transition, "signature": transition_signature},
            ],
        }
    )
    packet_bytes = render_document(specification, kind="work")
    packet = validate_work_packet(json.loads(packet_bytes))
    if packet["packet_sha256"] == failed_packet["packet_sha256"]:
        raise D0Error("qualification retry packet identity was reused")
    packet_signature_bytes = signature_bytes(sign_stdin(args.private_key, packet_bytes))
    envelope_bytes = render_control_envelope(
        packet_bytes, packet_signature_bytes, public_key=public_key
    )
    verified_envelope = verify_control_envelope(envelope_bytes, public_key=public_key)
    envelope = verified_envelope["envelope"]
    ready = {
        "status": "ready",
        "host": packet["host"],
        "cycle_id": packet["cycle_id"],
        "phase": packet["phase"],
        "operation": "verify-runtime",
        "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": sha256_bytes(packet_bytes),
        "signature_file_sha256": sha256_bytes(packet_signature_bytes),
        "control_envelope_file_sha256": sha256_bytes(envelope_bytes),
        "predecessor_report_sha256": [item["report_sha256"] for item in packet["predecessors"]],
        "helper_transition_sha256": [
            item["document"]["transition_sha256"] for item in packet["helper_transitions"]
        ],
    }
    manifest: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.d0-qualification-retry-dispatch-package-manifest.v2",
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "host": "john2",
        "cycle_id": "qualification",
        "phase": "verify",
        "operation": "verify-runtime",
        "status": "prepared-not-dispatched",
        "operation_identity": {
            "failed_packet_sha256": failed_packet["packet_sha256"],
            "failed_packet_replay_allowed": False,
            "retry_packet_sha256": packet["packet_sha256"],
            "identity_is_fresh": True,
            "control_claim_state": "unclaimed-pre-dispatch",
            "expected_claim_envelope_sha256": envelope["envelope_sha256"],
        },
        "packet": {
            "semantic_sha256": packet["packet_sha256"],
            "file_sha256": sha256_bytes(packet_bytes),
            "signature_file_sha256": sha256_bytes(packet_signature_bytes),
            "control_envelope_sha256": envelope["envelope_sha256"],
            "control_envelope_file_sha256": sha256_bytes(envelope_bytes),
            "ready_receipt_file_sha256": sha256_bytes(canonical_json(ready)),
            "helper_sha256": packet["helper_sha256"],
            "helper_transition_count": len(packet["helper_transitions"]),
            "predecessor_count": len(packet["predecessors"]),
        },
        "helper_transition": {
            "chain_index": transition["chain_index"],
            "semantic_sha256": transition["transition_sha256"],
            "file_sha256": sha256_bytes(transition_bytes),
            "signature_file_sha256": sha256_bytes(transition_signature_bytes),
            "from_helper_sha256": transition["from_helper_sha256"],
            "to_helper_sha256": transition["to_helper_sha256"],
            "to_plan_sha256": transition["to_plan_sha256"],
        },
        "retry_lineage": {
            "semantic_sha256": lineage["lineage_sha256"],
            "file_sha256": sha256_bytes(lineage_bytes),
            "failed_packet_sha256": failed_packet["packet_sha256"],
            "failed_report_sha256": failed_report["report_sha256"],
            "fresh_operation_identity_required": True,
            "post_failure_read_only_result_sha256": proof["result_sha256"],
            "post_failure_read_only_evidence_sha256": proof["evidence_sha256"],
            "post_failure_read_only_receipt_sha256": proof["receipt_sha256"],
            "network_change_classification": proof["network_change_classification"],
            "prior_diagnostic_failure": proof["prior_diagnostic_failure"],
        },
        "failure_completion": {
            "semantic_sha256": completion["completion_sha256"],
            "file_sha256": sha256_bytes(completion_bytes),
        },
        "execution_plan": {
            "semantic_sha256": plan["plan_sha256"],
            "file_sha256": sha256_bytes(plan_bytes),
        },
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "qualification_claimed": False,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    manifest_bytes = canonical_json(manifest)
    manifest_signature_bytes = signature_bytes(sign_stdin(args.private_key, manifest_bytes))
    output = args.output_root
    for name, payload in (
        ("work-packet.json", packet_bytes),
        ("work-packet-signature.json", packet_signature_bytes),
        ("control-envelope.json", envelope_bytes),
        ("ready-receipt.json", canonical_json(ready)),
        ("retry-package-manifest.json", manifest_bytes),
        ("retry-package-manifest-signature.json", manifest_signature_bytes),
    ):
        _write_new(output / name, payload)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--failed-packet", type=Path, required=True)
    value.add_argument("--failure-completion", type=Path, required=True)
    value.add_argument("--lineage", type=Path, required=True)
    value.add_argument("--helper-transition", type=Path, required=True)
    value.add_argument("--helper-transition-signature", type=Path, required=True)
    value.add_argument("--execution-plan", type=Path, required=True)
    value.add_argument("--public-key", type=Path, required=True)
    value.add_argument("--private-key", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--issued-unix-ms", type=int)
    return value


def main() -> int:
    try:
        result = render(parser().parse_args())
    except (D0Error, OSError, KeyError, ValueError) as error:
        sys.stderr.write(f"r2-d0-qualification-retry: {error}\n")
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
