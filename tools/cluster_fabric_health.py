#!/usr/bin/env python3
"""Emit one authoritative health record for the Bacalhau execution fabric."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from cascadia_cluster.bacalhau_api import BacalhauAPI

ENDPOINT = "http://100.110.109.6:1234"
EXPECTED_NODES = {"john1", "john2", "john3", "john4"}
EXPECTED_CPU_CAPACITY = {"john1": 9, "john2": 10, "john3": 10, "john4": 10}
EXPECTED_MEMORY_CAPACITY_BYTES = {
    "john1": 12 * 1024**3,
    "john2": 15 * 1024**3,
    "john3": 15 * 1024**3,
    "john4": 15 * 1024**3,
}
EXPECTED_DISK_CAPACITY_BYTES = {
    "john1": 80 * 1024**3,
    "john2": 80 * 1024**3,
    "john3": 80 * 1024**3,
    "john4": 80 * 1024**3,
}


def _http_health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            response.read(1)
            return response.status < 400
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-storage-down", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    api = BacalhauAPI(ENDPOINT)
    errors: list[str] = []
    try:
        alive = api.alive()
        raw_nodes = api.nodes()
    except Exception as error:  # structured below for operator use
        alive = False
        raw_nodes = []
        errors.append(str(error))
    nodes = []
    observed_names = set()
    for raw in raw_nodes:
        info = raw.get("Info", {})
        labels = info.get("Labels", {})
        name = labels.get("cascadia_internal_node", "unknown")
        observed_names.add(name)
        compute = info.get("ComputeNodeInfo", {})
        maximum = compute.get("MaxCapacity", {})
        available = compute.get("AvailableCapacity", {})
        nodes.append(
            {
                "name": name,
                "node_id": info.get("NodeID"),
                "connected": raw.get("Connection") == "CONNECTED",
                "version": info.get("BacalhauVersion", {}).get("GitVersion"),
                "docker": "docker" in compute.get("ExecutionEngines", []),
                "cpu_capacity": maximum.get("CPU", 0),
                "cpu_available": available.get("CPU", 0),
                "memory_capacity_bytes": maximum.get("Memory", 0),
                "memory_available_bytes": available.get("Memory", 0),
                "disk_capacity_bytes": maximum.get("Disk", 0),
                "disk_available_bytes": available.get("Disk", 0),
                "running_executions": compute.get("RunningExecutions", 0),
            }
        )
    if observed_names != EXPECTED_NODES:
        errors.append(
            f"compute membership differs: expected {sorted(EXPECTED_NODES)}, "
            f"observed {sorted(observed_names)}"
        )
    for node in nodes:
        if not node["connected"] or not node["docker"] or node["version"] != "v1.9.0":
            errors.append(f"compute node is not ready: {node['name']}")
        expected_cpu = EXPECTED_CPU_CAPACITY.get(node["name"])
        if expected_cpu is not None and node["cpu_capacity"] != expected_cpu:
            errors.append(
                f"compute CPU capacity differs for {node['name']}: "
                f"expected {expected_cpu}, observed {node['cpu_capacity']}"
            )
        expected_memory = EXPECTED_MEMORY_CAPACITY_BYTES.get(node["name"])
        if (
            expected_memory is not None
            and node["memory_capacity_bytes"] != expected_memory
        ):
            errors.append(
                f"compute memory capacity differs for {node['name']}: "
                f"expected {expected_memory}, observed {node['memory_capacity_bytes']}"
            )
        expected_disk = EXPECTED_DISK_CAPACITY_BYTES.get(node["name"])
        if expected_disk is not None and node["disk_capacity_bytes"] != expected_disk:
            errors.append(
                f"compute disk capacity differs for {node['name']}: "
                f"expected {expected_disk}, observed {node['disk_capacity_bytes']}"
            )
    registry = _http_health("http://100.110.109.6:5000/v2/")
    object_store = _http_health("http://100.110.109.6:9000/minio/health/live")
    if not args.allow_storage_down and not registry:
        errors.append("OCI registry is unavailable")
    if not args.allow_storage_down and not object_store:
        errors.append("MinIO is unavailable")
    value = {
        "schema_id": "cascadia.cluster.fabric-health.v1",
        "observed_unix_ms": time.time_ns() // 1_000_000,
        "healthy": alive and not errors,
        "orchestrator_alive": alive,
        "registry_healthy": registry,
        "object_store_healthy": object_store,
        "nodes": sorted(nodes, key=lambda node: node["name"]),
        "cpu_capacity_total": sum(node["cpu_capacity"] for node in nodes),
        "cpu_capacity_expected": sum(EXPECTED_CPU_CAPACITY.values()),
        "memory_capacity_total_bytes": sum(node["memory_capacity_bytes"] for node in nodes),
        "memory_capacity_expected_bytes": sum(EXPECTED_MEMORY_CAPACITY_BYTES.values()),
        "disk_capacity_total_bytes": sum(node["disk_capacity_bytes"] for node in nodes),
        "disk_capacity_expected_bytes": sum(EXPECTED_DISK_CAPACITY_BYTES.values()),
        "errors": errors,
    }
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if value["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
