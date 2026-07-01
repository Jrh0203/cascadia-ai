#!/usr/bin/env python3
"""Signed read-only inventory for post-failure Docker guest network state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AUTHORIZATION_SCHEMA = "cascadia.r2-map.d0-post-failure-network-authorization.v1"
RESULT_SCHEMA = "cascadia.r2-map.d0-post-failure-network-result.v1"
COLIMA = "/opt/homebrew/bin/colima"
DOCKER = "/opt/homebrew/bin/docker"
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


class InventoryError(RuntimeError):
    pass


class InventoryCommandError(InventoryError):
    def __init__(self, message: str, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_sha256(value: Mapping[str, Any], field: str) -> str:
    return sha256_bytes(canonical_json({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, maximum: int = 1024 * 1024) -> bytes:
    observed = path.lstat()
    if not path.is_file() or observed.st_nlink != 1 or observed.st_mode & 0o022:
        raise InventoryError(f"unsafe inventory input: {path}")
    if observed.st_size > maximum:
        raise InventoryError(f"inventory input exceeds its bound: {path}")
    value = path.read_bytes()
    if len(value) != observed.st_size:
        raise InventoryError(f"inventory input changed while reading: {path}")
    return value


def _completed(argv: Sequence[str], *, check: bool = True) -> dict[str, Any]:
    if not argv or any(not isinstance(item, str) or "\0" in item for item in argv):
        raise InventoryError("inventory command differs")
    completed = subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=120,
        env=dict(os.environ),
    )
    if len(completed.stdout) > MAX_OUTPUT_BYTES or len(completed.stderr) > MAX_OUTPUT_BYTES:
        raise InventoryError("inventory command output exceeds its bound")
    try:
        stdout_text = completed.stdout.decode("utf-8")
        stdout_encoding = "utf-8"
    except UnicodeDecodeError:
        stdout_text = completed.stdout.hex()
        stdout_encoding = "hex"
    try:
        stderr_text = completed.stderr.decode("utf-8")
        stderr_encoding = "utf-8"
    except UnicodeDecodeError:
        stderr_text = completed.stderr.hex()
        stderr_encoding = "hex"
    evidence = {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": stdout_text,
        "stdout_encoding": stdout_encoding,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stdout_size": len(completed.stdout),
        "stderr": stderr_text,
        "stderr_encoding": stderr_encoding,
        "stderr_sha256": sha256_bytes(completed.stderr),
        "stderr_size": len(completed.stderr),
    }
    if check and completed.returncode != 0:
        raise InventoryCommandError("inventory command failed", evidence)
    return evidence


def _guest(argv: Sequence[str], *, check: bool = True) -> dict[str, Any]:
    return _completed([COLIMA, "ssh", "--profile", "cascadia-r2", "--", *argv], check=check)


def _json_evidence(evidence: dict[str, Any], label: str) -> dict[str, Any]:
    if evidence["returncode"] != 0:
        raise InventoryError(f"{label} command failed")
    try:
        parsed = json.loads(evidence["stdout"])
    except json.JSONDecodeError as error:
        raise InventoryError(f"{label} is not JSON") from error
    return {**evidence, "parsed": parsed}


def _json_lines_evidence(evidence: dict[str, Any], label: str) -> dict[str, Any]:
    if evidence["returncode"] != 0:
        raise InventoryError(f"{label} command failed")
    try:
        parsed = [json.loads(line) for line in evidence["stdout"].splitlines() if line]
    except json.JSONDecodeError as error:
        raise InventoryError(f"{label} is not JSON lines") from error
    return {**evidence, "parsed": parsed}


def _network_snapshot() -> dict[str, Any]:
    addresses = _json_evidence(_guest(["/usr/sbin/ip", "-j", "address", "show"]), "guest addresses")
    routes = _json_evidence(
        _guest(["/usr/sbin/ip", "-j", "route", "show", "table", "all"]),
        "guest routes",
    )
    rules = _json_evidence(_guest(["/usr/sbin/ip", "-j", "rule", "show"]), "guest rules")
    nft = _json_evidence(
        _guest(["/usr/bin/sudo", "-n", "/usr/sbin/nft", "-j", "list", "ruleset"]),
        "guest nftables ruleset",
    )
    network_list = _json_lines_evidence(
        _completed(
            [
                DOCKER,
                "network",
                "ls",
                "--no-trunc",
                "--format",
                "{{json .}}",
            ]
        ),
        "Docker network list",
    )
    network_ids = sorted(
        str(item.get("ID"))
        for item in network_list["parsed"]
        if isinstance(item, dict) and item.get("ID")
    )
    network_inspect = (
        _json_evidence(
            _completed([DOCKER, "network", "inspect", *network_ids]),
            "Docker network inspect",
        )
        if network_ids
        else {"parsed": [], "status": "no-networks"}
    )
    parsed = {
        "addresses": addresses["parsed"],
        "routes": routes["parsed"],
        "rules": rules["parsed"],
        "nftables": nft["parsed"],
        "docker_network_list": network_list["parsed"],
        "docker_network_inspect": network_inspect["parsed"],
    }
    return {
        "captured_unix_ms": time.time_ns() // 1_000_000,
        "evidence": {
            "addresses": addresses,
            "routes": routes,
            "rules": rules,
            "nftables": nft,
            "docker_network_list": network_list,
            "docker_network_inspect": network_inspect,
        },
        "parsed": parsed,
        "state_sha256": sha256_bytes(canonical_json(parsed)),
    }


def _runtime_inventory() -> dict[str, Any]:
    docker = {
        "info": _json_evidence(
            _completed([DOCKER, "info", "--format", "{{json .}}"]), "Docker info"
        ),
        "containers": _json_lines_evidence(
            _completed(
                [DOCKER, "container", "ls", "--all", "--no-trunc", "--format", "{{json .}}"]
            ),
            "Docker containers",
        ),
        "images": _json_lines_evidence(
            _completed(
                [
                    DOCKER,
                    "image",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--digests",
                    "--format",
                    "{{json .}}",
                ]
            ),
            "Docker images",
        ),
        "volumes": _json_lines_evidence(
            _completed([DOCKER, "volume", "ls", "--format", "{{json .}}"]),
            "Docker volumes",
        ),
        "system_df": _json_lines_evidence(
            _completed([DOCKER, "system", "df", "--format", "{{json .}}"]),
            "Docker system df",
        ),
        "buildx_ls": _completed([DOCKER, "buildx", "ls"]),
        "buildx_du": _completed([DOCKER, "buildx", "du", "--builder", "default"], check=False),
    }
    guest = {
        "docker_service": _guest(["/usr/bin/systemctl", "is-active", "docker"], check=False),
        "buildkit_tree": _guest(
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/find",
                "/var/lib/docker/buildkit",
                "-xdev",
                "-printf",
                "%y %m %u %g %s %p\\n",
            ],
            check=False,
        ),
        "ctr_moby_containers": _guest(
            ["/usr/bin/sudo", "-n", "/usr/bin/ctr", "--namespace", "moby", "containers", "list"],
            check=False,
        ),
        "ctr_moby_images": _guest(
            ["/usr/bin/sudo", "-n", "/usr/bin/ctr", "--namespace", "moby", "images", "list"],
            check=False,
        ),
        "ctr_buildkit_containers": _guest(
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/ctr",
                "--namespace",
                "buildkit",
                "containers",
                "list",
            ],
            check=False,
        ),
        "ctr_buildkit_images": _guest(
            ["/usr/bin/sudo", "-n", "/usr/bin/ctr", "--namespace", "buildkit", "images", "list"],
            check=False,
        ),
        "swap": _guest(["/usr/bin/cat", "/proc/swaps"]),
    }
    conntrack_binary = _guest(["/usr/bin/test", "-x", "/usr/sbin/conntrack"], check=False)
    if conntrack_binary["returncode"] == 0:
        conntrack = {
            "source": "/usr/sbin/conntrack -L",
            "evidence": _guest(["/usr/bin/sudo", "-n", "/usr/sbin/conntrack", "-L"], check=False),
        }
    else:
        conntrack = {
            "source": "/proc/net/nf_conntrack",
            "evidence": _guest(
                ["/usr/bin/sudo", "-n", "/usr/bin/cat", "/proc/net/nf_conntrack"], check=False
            ),
            "binary_probe": conntrack_binary,
        }
    return {"docker": docker, "guest": guest, "conntrack": conntrack}


def classify_network_ownership(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    parsed = snapshot.get("parsed", {})
    addresses = parsed.get("addresses", []) if isinstance(parsed, dict) else []
    networks = parsed.get("docker_network_inspect", []) if isinstance(parsed, dict) else []
    nftables = parsed.get("nftables", {}).get("nftables", []) if isinstance(parsed, dict) else []
    docker0 = next((item for item in addresses if item.get("ifname") == "docker0"), None)
    bridge = next((item for item in networks if item.get("Name") == "bridge"), None)
    raw_objects = [
        item
        for item in nftables
        if isinstance(item, dict)
        and any(
            isinstance(value, dict)
            and value.get("family") == "ip"
            and value.get("table", value.get("name")) == "raw"
            for value in item.values()
        )
    ]
    return {
        "docker0": {
            "present": docker0 is not None,
            "link_local_ipv6": [
                item
                for item in (docker0 or {}).get("addr_info", [])
                if item.get("family") == "inet6" and item.get("scope") == "link"
            ],
            "docker_default_bridge_present": bridge is not None,
            "owner": "docker-daemon-libnetwork" if bridge is not None else "unclassified",
            "lifecycle": "default-bridge-interface",
        },
        "ip_raw_prerouting": {
            "objects": raw_objects,
            "present": bool(raw_objects),
            "owner": "docker-daemon-firewall-backend"
            if bridge is not None and raw_objects
            else "unclassified",
            "lifecycle": "daemon-network-firewall-programming",
        },
        "classification_basis": [
            "guest ip-link/address identity",
            "Docker network inspect default bridge identity",
            "guest nftables object family/table/hook identity",
        ],
        "classification_scope": "ownership-and-lifecycle-not-causal-proof",
    }


def run(authorization_path: Path) -> dict[str, Any]:
    authorization = json.loads(read_regular(authorization_path))
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_id") != AUTHORIZATION_SCHEMA
        or authorization.get("authorization_sha256")
        != document_sha256(authorization, "authorization_sha256")
        or authorization.get("host") != "john2"
        or authorization.get("status") != "authorized-once"
        or authorization.get("read_only") is not True
        or authorization.get("project_code_executed") is not False
        or authorization.get("protected_seed_values_opened") is not False
        or int(authorization.get("expires_unix_ms", 0)) <= time.time_ns() // 1_000_000
    ):
        raise InventoryError("network inventory authorization differs")
    before = _network_snapshot()
    runtime = _runtime_inventory()
    after = _network_snapshot()
    result: dict[str, Any] = {
        "schema_id": RESULT_SCHEMA,
        "schema_version": 1,
        "authorization_sha256": authorization["authorization_sha256"],
        "campaign_id": authorization["campaign_id"],
        "run_id": authorization["run_id"],
        "host": "john2",
        "failure_baseline": authorization["failure_baseline"],
        "network_before": before,
        "runtime": runtime,
        "network_after": after,
        "network_stable_during_inventory": before["parsed"] == after["parsed"],
        "ownership_and_lifecycle": classify_network_ownership(after),
        "read_only": True,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "qualification_claimed": False,
        "finished_unix_ms": time.time_ns() // 1_000_000,
        "status": "pass-diagnostic",
    }
    result["result_sha256"] = document_sha256(result, "result_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    try:
        print(
            json.dumps(
                run(parser.parse_args().authorization), sort_keys=True, separators=(",", ":")
            )
        )
        return 0
    except InventoryCommandError as error:
        failure: dict[str, Any] = {
            "command_evidence": error.evidence,
            "error": str(error),
            "finished_unix_ms": time.time_ns() // 1_000_000,
            "schema_id": "cascadia.r2-map.d0-post-failure-network-probe-failure.v1",
            "schema_version": 1,
            "status": "fail-diagnostic",
        }
        failure["failure_sha256"] = document_sha256(failure, "failure_sha256")
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")))
        return 2
    except (
        InventoryError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"r2-d0-network-inventory: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
