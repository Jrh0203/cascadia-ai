"""Shared constructors for D0 unit tests; never included in the helper archive."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path
from typing import Any

from r2_d0.bundle import (
    render_persistence_evidence,
    render_persistence_monitor,
    render_persistence_receipt,
)
from r2_d0.canonical import (
    BOOTSTRAP_PACKET_SCHEMA,
    CAMPAIGN_ID,
    CORE_IMAGE,
    FROZEN_RUNTIME,
    INSTALL_OPERATIONS_BY_HOST,
    PATH_CONTRACT,
    PUBLIC_KEY_NAMESPACE,
    SCANNER_IMAGE,
    SMOKE_IMAGE,
    WORK_PACKET_SCHEMA,
    render_document,
)

_PRODUCTION_PATH_CONTRACT = copy.deepcopy(PATH_CONTRACT)


def persisted_transaction_files(files: dict[str, bytes]) -> dict[str, bytes]:
    """Add deterministic passing persistence proof to a unit-test transaction."""

    receipt = render_persistence_receipt(
        files["work-packet.json"],
        files["work-packet-signature.json"],
        files["report.json"],
    )
    snapshot = {"swap_used_bytes": 0, "collected_unix_ms": 1}
    evidence = render_persistence_evidence(
        receipt,
        before=snapshot,
        after_payload_fsync=snapshot,
        precommit=snapshot,
    )
    monitor = render_persistence_monitor(
        receipt,
        evidence,
        continuous_swap={
            "sample_count": 1,
            "nonzero_samples": 0,
            "max_used_bytes": 0,
            "status": "pass",
        },
        final_snapshot=snapshot,
    )
    return {
        **files,
        "persistence-receipt.json": receipt,
        "persistence-evidence.json": evidence,
        "persistence-monitor.json": monitor,
    }


def host_home(host: str) -> str:
    return {"john1": "/Users/johnherrick", "john2": "/Users/john2", "john3": "/Users/john3"}[host]


def bootstrap_spec(
    host: str,
    *,
    helper_sha256: str,
    helper_size: int,
    public_key_sha256: str,
    fingerprint: str,
    now: int | None = None,
) -> dict[str, Any]:
    issued = now if now is not None else time.time_ns() // 1_000_000
    home = host_home(host)
    return {
        "schema_id": BOOTSTRAP_PACKET_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": "d0-runtime-bootstrap-20260618-v1",
        "host": host,
        "issued_unix_ms": issued,
        "expires_unix_ms": issued + 60_000,
        "helper": {
            "sha256": helper_sha256,
            "size": helper_size,
            "entrypoint": "r2_map_d0_runtime.py",
        },
        "public_key": {
            "algorithm": "ssh-ed25519",
            "fingerprint": fingerprint,
            "openssh_sha256": public_key_sha256,
            "namespace": PUBLIC_KEY_NAMESPACE,
        },
        "destinations": {
            "helper": f"{home}/.local/libexec/cascadia-r2-d0/v1",
            "public_key": f"{home}/.config/cascadia-r2-d0/public-key",
            "receipt": f"{home}/.config/cascadia-r2-d0/bootstrap-receipt.json",
        },
        "protected_seed_values_opened": False,
    }


def rendered_bootstrap(**kwargs: Any) -> bytes:
    return render_document(bootstrap_spec(**kwargs), kind="bootstrap")


def _paths(host: str, temporary_root: Path | None = None) -> dict[str, str]:
    if temporary_root is not None:
        runtime = temporary_root / host
        core = runtime / "bootstrap/core.raw.gz"
        smoke = runtime / "bootstrap/alpine.oci.tar"
        output = runtime / "output"
        value = {
            "campaign_root": "/Users/johnherrick/cascadia-bench/r2-map-v1",
            "colima_home": str(runtime / "colima"),
            "colima_cache_home": str(runtime / "colima-cache"),
            "docker_config": str(runtime / "docker"),
            "homebrew_cache": str(runtime / "homebrew/cache"),
            "homebrew_logs": str(runtime / "homebrew/logs"),
            "homebrew_temp": str(runtime / "homebrew/temp"),
            "core_image": str(core),
            "smoke_oci": str(smoke),
            "scanner_oci": str(runtime / "bootstrap/scanner.oci.tar"),
            "scanner_license": str(runtime / "bootstrap/scanner.LICENSE"),
            "scanner_source_archive": str(runtime / "bootstrap/scanner-source.tar.gz"),
            "homebrew_closure": str(runtime / "bootstrap/homebrew-closure.tar"),
            "runtime_supply": str(runtime / "bootstrap/worker-runtime-supply-v1.tar"),
            "runtime_supply_inbox": str(runtime / "supply-inbox/worker-runtime-supply-v1.tar"),
            "pending_root": str(output),
            "control_inbox": str(runtime / "control-inbox"),
            "output_root": str(output),
        }
        PATH_CONTRACT[host] = value
        return dict(value)
    PATH_CONTRACT[host] = copy.deepcopy(_PRODUCTION_PATH_CONTRACT[host])
    return dict(PATH_CONTRACT[host])


def work_spec(
    host: str,
    phase: str,
    *,
    cycle_id: str = "qualification",
    helper_sha256: str = "1" * 64,
    fingerprint: str = "SHA256:" + "A" * 43,
    operations: list[str] | None = None,
    now: int | None = None,
    temporary_root: Path | None = None,
) -> dict[str, Any]:
    issued = now if now is not None else time.time_ns() // 1_000_000
    role = {"john1": "worker", "john2": "builder-worker", "john3": "worker"}[host]
    default: list[str] = {
        "preflight": ["preflight-audit"],
        "install": ["install-runtime"],
        "start": ["start-runtime"],
        "verify": ["buildkit-probe", "verify-runtime"] if host == "john2" else ["verify-runtime"],
        "rollback": ["rollback-runtime"],
        "postflight": ["postflight-audit"],
    }[phase]
    selected_operations = operations or default
    bottle_names = {
        "john1": ("colima", "docker", "lima"),
        "john2": ("colima", "docker", "docker-buildx", "lima"),
        "john3": ("colima", "docker", "lima"),
    }[host]
    bottles = [
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
    ]
    paths = _paths(host, temporary_root)
    john2_order = INSTALL_OPERATIONS_BY_HOST["john2"]

    def derived_present(field: str) -> bool:
        if phase == "preflight":
            return False
        if host == "john2":
            if phase != "install":
                return True
            operation = selected_operations[0]
            producer = {
                "smoke_oci": "acquire-smoke",
                "scanner_oci": "acquire-scanner",
                "homebrew_closure": "render-runtime-supply",
                "runtime_supply": "render-runtime-supply",
            }[field]
            return john2_order.index(operation) > john2_order.index(producer)
        if field == "scanner_oci":
            return False
        if field == "runtime_supply":
            return True
        return not (phase == "install" and selected_operations == ["materialize-runtime-supply"])

    return {
        "schema_id": WORK_PACKET_SCHEMA,
        "schema_version": 10,
        "campaign_id": CAMPAIGN_ID,
        "run_id": "d0-runtime-bootstrap-20260618-v1",
        "cycle_id": cycle_id,
        "host": host,
        "role": role,
        "phase": phase,
        "issued_unix_ms": issued,
        "expires_unix_ms": issued + 7_200_000,
        "policy": {
            "goal_sha256": "2" * 64,
            "plan_sha256": "3" * 64,
            "runbook_sha256": "4" * 64,
        },
        "helper_sha256": helper_sha256,
        "helper_transitions": [],
        "public_key_fingerprint": fingerprint,
        "paths": paths,
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
        "artifacts": {
            "core_image": {
                "name": "colima-core-v0.10.4",
                "size": CORE_IMAGE["size"],
                "sha256": CORE_IMAGE["sha256"],
                "source": CORE_IMAGE["url"],
            },
            "smoke_source": dict(SMOKE_IMAGE),
            "smoke_oci": {
                "name": "alpine-3.22.1-arm64-oci",
                "size": 4_200_000,
                "sha256": "5" * 64,
                "source": paths["smoke_oci"],
            }
            if derived_present("smoke_oci")
            else None,
            "scanner_source": dict(SCANNER_IMAGE),
            "scanner_oci": {
                "name": "buildkit-syft-scanner-v1.11.0-arm64-oci",
                "size": 44_000_000,
                "sha256": "6" * 64,
                "source": paths["scanner_oci"],
            }
            if derived_present("scanner_oci")
            else None,
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
            "homebrew_closure": (
                {
                    "name": f"homebrew-{host}-arm64-tahoe-v1",
                    "size": 1_000_000,
                    "sha256": "7" * 64,
                    "source": paths["homebrew_closure"],
                }
                if derived_present("homebrew_closure")
                else None
            ),
            "runtime_supply": (
                {
                    "name": "worker-runtime-supply-v1",
                    "size": 400_000_000,
                    "sha256": "a" * 64,
                    "source": paths["runtime_supply"],
                }
                if derived_present("runtime_supply")
                else None
            ),
            "probe_context": {
                "name": "d0-buildkit-probe",
                "size": 10_240,
                "sha256": "bb61a97afa096f1af029226404dc12a6d22d0be5700d66d5401cc8c35c8df5db",
                "source": str(Path(paths["output_root"]) / "probe-context.tar"),
            },
            "bottles": bottles,
        },
        "allowed_operations": sorted(selected_operations),
        "predecessors": _predecessors(
            host,
            phase,
            issued,
            "verify-runtime"
            if "verify-runtime" in selected_operations
            else selected_operations[-1],
            cycle_id,
        ),
        "protected_seed_values_opened": False,
    }


def _predecessors(
    host: str,
    phase: str,
    issued: int,
    current_operation: str,
    cycle_id: str,
) -> list[dict[str, Any]]:
    if cycle_id not in {"qualification", "final-live"}:
        raise ValueError("unknown fixture cycle")
    if phase == "preflight" and cycle_id == "qualification":
        return []
    install_operations = INSTALL_OPERATIONS_BY_HOST[host]
    installs = [(host, "install", operation, "pass") for operation in install_operations]
    order = [
        (host, "preflight", "preflight-audit", "pass"),
        *installs,
        (host, "start", "start-runtime", "pass"),
        (host, "verify", "verify-runtime", "pass"),
        (host, "rollback", "rollback-runtime", "rolled-back"),
    ]
    if phase == "preflight":
        phases = []
    elif phase == "install":
        phases = order[: 1 + install_operations.index(current_operation)]
    else:
        length = {
            "start": 1 + len(installs),
            "verify": 2 + len(installs),
            "rollback": 3 + len(installs),
            "postflight": 4 + len(installs),
        }[phase]
        phases = order[:length]
    qualified: list[tuple[str, str, str, str, str]] = []
    if cycle_id == "final-live":
        qualification_phase = "postflight-audit"
        qualification_predecessors = _predecessors(
            host,
            "postflight",
            issued - 60_000,
            qualification_phase,
            "qualification",
        )
        qualified.extend(
            (
                item["cycle_id"],
                item["host"],
                item["phase"],
                item["operation"],
                item["status"],
            )
            for item in qualification_predecessors
        )
        qualified.append(("qualification", host, "postflight", qualification_phase, "pass"))
    qualified.extend((cycle_id, *item) for item in phases)
    if current_operation == "materialize-runtime-supply":
        source_host, source_operation = {
            "john1": ("john2", "render-runtime-supply"),
            "john3": ("john1", "materialize-runtime-supply"),
        }[host]
        qualified.append((cycle_id, source_host, "install", source_operation, "pass"))
    result: list[dict[str, Any]] = []
    for index, (prior_cycle, prior_host, prior_phase, operation, status) in enumerate(qualified):
        packet_digest = hashlib.sha256(
            f"packet-{prior_cycle}-{prior_host}-{operation}-{index}".encode()
        ).hexdigest()
        report_digest = hashlib.sha256(
            f"report-{prior_cycle}-{prior_host}-{operation}-{index}".encode()
        ).hexdigest()
        result.append(
            {
                "cycle_id": prior_cycle,
                "host": prior_host,
                "phase": prior_phase,
                "operation": operation,
                "status": status,
                "packet_sha256": packet_digest,
                "report_sha256": report_digest,
                "bundle_sha256": hashlib.sha256(
                    f"bundle-{prior_cycle}-{prior_host}-{operation}-{index}".encode()
                ).hexdigest(),
                "bundle_size": 10_240 + index * 10_240,
                "manifest_sha256": hashlib.sha256(
                    f"manifest-{prior_cycle}-{prior_host}-{operation}-{index}".encode()
                ).hexdigest(),
                "materialization_receipt_sha256": hashlib.sha256(
                    f"materialized-{prior_cycle}-{prior_host}-{operation}-{index}".encode()
                ).hexdigest(),
                "finished_unix_ms": issued - (len(qualified) - index) * 1000,
                "receipt_relative": (
                    f"receipts/{report_digest}"
                    if prior_host == host
                    else f"dependencies/{prior_host}/{report_digest}"
                ),
            }
        )
    return result


def rendered_work(host: str, phase: str, **kwargs: Any) -> bytes:
    return render_document(work_spec(host, phase, **kwargs), kind="work")
