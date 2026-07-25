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
    for raw in raw_nodes:
        info = raw.get("Info", {})
        labels = info.get("Labels", {})
        name = labels.get("cascadia_internal_node", "unknown")
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
    for node in nodes:
        if not node["connected"] or not node["docker"]:
            errors.append(f"compute node is not ready: {node['name']}")
    if alive and not nodes:
        errors.append("scheduler has no compute nodes")
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
        "memory_capacity_total_bytes": sum(node["memory_capacity_bytes"] for node in nodes),
        "disk_capacity_total_bytes": sum(node["disk_capacity_bytes"] for node in nodes),
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
