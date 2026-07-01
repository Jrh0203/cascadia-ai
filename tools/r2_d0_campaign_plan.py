#!/usr/bin/env python3
"""Render the exact D0 v8 schedule and currently-ready signed packets on John1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from r2_d0.canonical import (
    BOOTSTRAP_PACKET_SCHEMA,
    CAMPAIGN_ID,
    CORE_IMAGE,
    D0_RUN_ID,
    FROZEN_RUNTIME,
    PATH_CONTRACT,
    PROBE_ARCHIVE_SHA256,
    PROBE_ARCHIVE_SIZE,
    PUBLIC_KEY_NAMESPACE,
    SCANNER_IMAGE,
    SMOKE_IMAGE,
    WORK_PACKET_SCHEMA,
    canonical_json,
    document_sha256,
    render_document,
    sha256_bytes,
)
from r2_d0.signing import (
    load_public_key,
    public_key_fingerprint,
    sign_stdin,
    signature_bytes,
)
from r2_d0.transport import render_control_envelope

ACTIVE_HOSTS = ("john1", "john2", "john3")
ROLE = {"john1": "worker", "john2": "builder-worker", "john3": "worker"}
HOST_USER = {"john1": "johnherrick", "john2": "john2", "john3": "john3"}
JOHN2_INSTALL_PREFIX = (
    "acquire-core",
    "acquire-homebrew-artifacts",
    "acquire-scanner",
    "acquire-smoke",
    "render-runtime-supply",
)


def _key(cycle: str, host: str, phase: str, operation: str) -> str:
    return f"{cycle}/{host}/{phase}/{operation}"


def ordered_transactions() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: dict[tuple[str, str], str] = {}

    def add(cycle: str, host: str, phase: str, operation: str, *extra: str) -> None:
        identifier = _key(cycle, host, phase, operation)
        dependencies = list(extra)
        prior = previous.get((cycle, host))
        if prior is not None:
            dependencies.append(prior)
        if cycle == "final-live" and phase == "preflight":
            dependencies.append(_key("qualification", host, "postflight", "postflight-audit"))
        result.append(
            {
                "sequence": len(result) + 1,
                "key": identifier,
                "cycle_id": cycle,
                "host": host,
                "phase": phase,
                "operation": operation,
                "dependencies": sorted(set(dependencies)),
            }
        )
        previous[(cycle, host)] = identifier

    for cycle in ("qualification", "final-live"):
        for host in ACTIVE_HOSTS:
            add(cycle, host, "preflight", "preflight-audit")
        for operation in JOHN2_INSTALL_PREFIX:
            add(cycle, "john2", "install", operation)
        add(
            cycle,
            "john1",
            "install",
            "materialize-runtime-supply",
            _key(cycle, "john2", "install", "render-runtime-supply"),
        )
        add(
            cycle,
            "john3",
            "install",
            "materialize-runtime-supply",
            _key(cycle, "john1", "install", "materialize-runtime-supply"),
        )
        add(cycle, "john2", "install", "render-probe-context")
        for host in ACTIVE_HOSTS:
            add(cycle, host, "install", "install-runtime")
        for host in ACTIVE_HOSTS:
            add(cycle, host, "start", "start-runtime")
        for host in ACTIVE_HOSTS:
            add(cycle, host, "verify", "verify-runtime")
        if cycle == "qualification":
            for host in ACTIVE_HOSTS:
                add(cycle, host, "rollback", "rollback-runtime")
            for host in ACTIVE_HOSTS:
                add(cycle, host, "postflight", "postflight-audit")
    if len(result) != 46 or [item["sequence"] for item in result] != list(range(1, 47)):
        raise RuntimeError("D0 execution graph does not contain exactly 46 ordered transactions")
    positions = {item["key"]: item["sequence"] for item in result}
    if any(
        dependency not in positions or positions[dependency] >= item["sequence"]
        for item in result
        for dependency in item["dependencies"]
    ):
        raise RuntimeError("D0 execution graph is not topologically ordered")
    return result


def bootstrap_spec(
    host: str,
    *,
    helper_sha256: str,
    helper_size: int,
    public_key: bytes,
    issued_unix_ms: int,
) -> dict[str, Any]:
    home = f"/Users/{HOST_USER[host]}"
    return {
        "schema_id": BOOTSTRAP_PACKET_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "host": host,
        "issued_unix_ms": issued_unix_ms,
        "expires_unix_ms": issued_unix_ms + 24 * 60 * 60 * 1000,
        "helper": {
            "sha256": helper_sha256,
            "size": helper_size,
            "entrypoint": "r2_map_d0_runtime.py",
        },
        "public_key": {
            "algorithm": "ssh-ed25519",
            "fingerprint": public_key_fingerprint(public_key),
            "openssh_sha256": sha256_bytes(public_key),
            "namespace": PUBLIC_KEY_NAMESPACE,
        },
        "destinations": {
            "helper": f"{home}/.local/libexec/cascadia-r2-d0/v1",
            "public_key": f"{home}/.config/cascadia-r2-d0/public-key",
            "receipt": f"{home}/.config/cascadia-r2-d0/bootstrap-receipt.json",
        },
        "protected_seed_values_opened": False,
    }


def _artifact_spec(host: str) -> dict[str, Any]:
    paths = PATH_CONTRACT[host]
    bottle_names = {
        "john1": ("colima", "docker", "lima"),
        "john2": ("colima", "docker", "docker-buildx", "lima"),
        "john3": ("colima", "docker", "lima"),
    }[host]
    return {
        "core_image": {
            "name": "colima-core-v0.10.4",
            "size": CORE_IMAGE["size"],
            "sha256": CORE_IMAGE["sha256"],
            "source": CORE_IMAGE["url"],
        },
        "smoke_source": dict(SMOKE_IMAGE),
        "smoke_oci": None,
        "scanner_source": dict(SCANNER_IMAGE),
        "scanner_oci": None,
        "scanner_license": {
            "name": "buildkit-syft-scanner-v1.11.0-license",
            "size": SCANNER_IMAGE["license_size"],
            "sha256": SCANNER_IMAGE["license_sha256"],
            "source": SCANNER_IMAGE["license_url"],
        },
        "scanner_source_archive": {
            "name": "buildkit-syft-scanner-v1.11.0-source",
            "size": SCANNER_IMAGE["source_archive_size"],
            "sha256": SCANNER_IMAGE["source_archive_sha256"],
            "source": SCANNER_IMAGE["source_archive_url"],
        },
        "homebrew_closure": None,
        "runtime_supply": None,
        "probe_context": {
            "name": "d0-buildkit-probe",
            "size": PROBE_ARCHIVE_SIZE,
            "sha256": PROBE_ARCHIVE_SHA256,
            "source": f"{paths['output_root']}/probe-context.tar",
        },
        "bottles": [
            {
                "name": name,
                "size": FROZEN_RUNTIME[name]["bottle_size"],
                "sha256": FROZEN_RUNTIME[name]["bottle_sha256"],
                "source": (
                    f"https://ghcr.io/v2/homebrew/core/{name}/blobs/"
                    f"sha256:{FROZEN_RUNTIME[name]['bottle_sha256']}"
                ),
            }
            for name in bottle_names
        ],
    }


def preflight_spec(
    host: str,
    *,
    helper_sha256: str,
    public_key: bytes,
    policy: dict[str, str],
    issued_unix_ms: int,
) -> dict[str, Any]:
    return {
        "schema_id": WORK_PACKET_SCHEMA,
        "schema_version": 10,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "cycle_id": "qualification",
        "host": host,
        "role": ROLE[host],
        "phase": "preflight",
        "issued_unix_ms": issued_unix_ms,
        "expires_unix_ms": issued_unix_ms + 24 * 60 * 60 * 1000,
        "policy": dict(policy),
        "helper_sha256": helper_sha256,
        "helper_transitions": [],
        "public_key_fingerprint": public_key_fingerprint(public_key),
        "paths": dict(PATH_CONTRACT[host]),
        "limits": {
            "runtime_max_bytes": 20 * 1024**3,
            "runtime_max_free_fraction_ppm": 250_000,
            "vm_cpu": 10,
            "vm_memory_gib": 14,
            "host_reserve_gib": 2,
            "root_disk_gib": 5,
            "data_disk_gib": 13,
            "output_max_bytes": 1024**3,
            "timeout_seconds": 3600,
        },
        "artifacts": _artifact_spec(host),
        "allowed_operations": ["preflight-audit"],
        "predecessors": [],
        "protected_seed_values_opened": False,
    }


def _write(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RuntimeError(f"unsafe policy or helper input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError("campaign plan output already exists")
    helper = args.helper_archive.read_bytes()
    public_key = load_public_key(args.public_key)
    helper_sha = sha256_bytes(helper)
    policy = {
        "goal_sha256": _file_sha256(args.goal),
        "plan_sha256": _file_sha256(args.plan),
        "runbook_sha256": _file_sha256(args.runbook),
    }
    transactions = ordered_transactions()
    plan: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.d0-execution-plan.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "created_unix_ms": args.issued_unix_ms,
        "helper_sha256": helper_sha,
        "helper_size": len(helper),
        "public_key_sha256": sha256_bytes(public_key),
        "public_key_fingerprint": public_key_fingerprint(public_key),
        "policy": policy,
        "transaction_count": len(transactions),
        "transactions": transactions,
        "ready_packet_keys": [
            _key("qualification", host, "preflight", "preflight-audit") for host in ACTIVE_HOSTS
        ],
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "d0_status": "red",
    }
    plan["plan_sha256"] = document_sha256(plan, "plan_sha256")
    _write(output / "execution-plan.json", canonical_json(plan))
    _write(output / "helper.tar", helper)
    _write(output / "campaign-public-key", public_key)
    manifest: list[dict[str, Any]] = []
    for host in ACTIVE_HOSTS:
        bootstrap = render_document(
            bootstrap_spec(
                host,
                helper_sha256=helper_sha,
                helper_size=len(helper),
                public_key=public_key,
                issued_unix_ms=args.issued_unix_ms,
            ),
            kind="bootstrap",
        )
        bootstrap_path = output / "bootstrap" / host / "packet.json"
        _write(bootstrap_path, bootstrap)
        packet = render_document(
            preflight_spec(
                host,
                helper_sha256=helper_sha,
                public_key=public_key,
                policy=policy,
                issued_unix_ms=args.issued_unix_ms,
            ),
            kind="work",
        )
        signature = signature_bytes(sign_stdin(args.private_key, packet))
        envelope = render_control_envelope(packet, signature, public_key=public_key)
        host_root = output / "ready" / host / "qualification-preflight"
        _write(host_root / "work-packet.json", packet)
        _write(host_root / "work-packet-signature.json", signature)
        _write(host_root / "control-envelope.json", envelope)
        manifest.append(
            {
                "host": host,
                "bootstrap_packet_path": str(bootstrap_path),
                "bootstrap_packet_file_sha256": sha256_bytes(bootstrap),
                "bootstrap_packet_sha256": json.loads(bootstrap)["packet_sha256"],
                "work_packet_path": str(host_root / "work-packet.json"),
                "work_packet_file_sha256": sha256_bytes(packet),
                "work_packet_sha256": json.loads(packet)["packet_sha256"],
                "signature_path": str(host_root / "work-packet-signature.json"),
                "signature_file_sha256": sha256_bytes(signature),
                "control_envelope_path": str(host_root / "control-envelope.json"),
                "control_envelope_sha256": sha256_bytes(envelope),
            }
        )
    result = {
        "schema_id": "cascadia.r2-map.d0-ready-packet-manifest.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "execution_plan_sha256": plan["plan_sha256"],
        "packets": manifest,
        "status": "ready-for-bootstrap-and-preflight-only",
    }
    result["manifest_sha256"] = document_sha256(result, "manifest_sha256")
    _write(output / "ready-packet-manifest.json", canonical_json(result))
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--output-root", type=Path, required=True)
    root.add_argument("--helper-archive", type=Path, required=True)
    root.add_argument("--public-key", type=Path, required=True)
    root.add_argument("--private-key", type=Path, required=True)
    root.add_argument("--goal", type=Path, required=True)
    root.add_argument("--plan", type=Path, required=True)
    root.add_argument("--runbook", type=Path, required=True)
    root.add_argument("--issued-unix-ms", type=int, default=time.time_ns() // 1_000_000)
    return root


def main() -> int:
    print(json.dumps(prepare(parser().parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
