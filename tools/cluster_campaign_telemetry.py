#!/usr/bin/env python3
"""Summarize a bounded campaign window from cluster telemetry JSONL."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CORES = {
    "john1": 10,
    "john2": 10,
    "john3": 10,
    "john4": 10,
}


def summarize(
    samples: list[dict[str, Any]],
    *,
    start_unix_ms: int,
    end_unix_ms: int,
    cores: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return node and core-weighted aggregate utilization for one window."""
    if end_unix_ms < start_unix_ms:
        raise ValueError("campaign telemetry end precedes start")
    core_counts = DEFAULT_CORES if cores is None else cores
    window = [
        sample
        for sample in samples
        if start_unix_ms <= int(sample["timestamp_unix_ms"]) <= end_unix_ms
    ]
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_cpu: list[float] = []
    aggregate_memory: list[float] = []
    for sample in window:
        reachable = [node for node in sample["nodes"] if bool(node["reachable"])]
        for node in sample["nodes"]:
            by_node[str(node["node_id"])].append(node)
        total_cores = sum(int(core_counts.get(str(node["node_id"]), 0)) for node in reachable)
        if total_cores:
            aggregate_cpu.append(
                sum(
                    float(node["cpu_percent"]) * int(core_counts.get(str(node["node_id"]), 0))
                    for node in reachable
                )
                / total_cores
            )
            aggregate_memory.append(
                sum(
                    float(node["memory_percent"]) * int(core_counts.get(str(node["node_id"]), 0))
                    for node in reachable
                )
                / total_cores
            )

    node_reports = {}
    for node_id, values in sorted(by_node.items()):
        reachable = [value for value in values if bool(value["reachable"])]
        cpu = [float(value["cpu_percent"]) for value in reachable]
        memory = [float(value["memory_percent"]) for value in reachable]
        node_reports[node_id] = {
            "samples": len(values),
            "reachable_samples": len(reachable),
            "reachable_fraction": len(reachable) / max(len(values), 1),
            "mean_cpu_percent": sum(cpu) / max(len(cpu), 1),
            "peak_cpu_percent": max(cpu, default=0.0),
            "mean_memory_percent": sum(memory) / max(len(memory), 1),
            "peak_memory_percent": max(memory, default=0.0),
        }
    return {
        "schema_version": 1,
        "start_unix_ms": start_unix_ms,
        "end_unix_ms": end_unix_ms,
        "observed_samples": len(window),
        "first_observed_unix_ms": (int(window[0]["timestamp_unix_ms"]) if window else None),
        "last_observed_unix_ms": (int(window[-1]["timestamp_unix_ms"]) if window else None),
        "source_interval_seconds": 30,
        "node_cores": core_counts,
        "nodes": node_reports,
        "mean_core_weighted_cpu_percent": (sum(aggregate_cpu) / max(len(aggregate_cpu), 1)),
        "peak_core_weighted_cpu_percent": max(
            aggregate_cpu,
            default=0.0,
        ),
        "mean_core_weighted_memory_percent": (
            sum(aggregate_memory) / max(len(aggregate_memory), 1)
        ),
        "peak_core_weighted_memory_percent": max(
            aggregate_memory,
            default=0.0,
        ),
        "cpu_interpretation": ("raw host CPU utilization; MLX GPU occupancy is not included"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid cluster telemetry line {line_number}") from error
        if int(sample.get("schema_version", -1)) != 1:
            raise ValueError("unsupported cluster telemetry schema")
        samples.append(sample)
    samples.sort(key=lambda sample: int(sample["timestamp_unix_ms"]))
    return samples


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--start-unix-ms", type=int, required=True)
    parser.add_argument("--end-unix-ms", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        load_jsonl(args.history),
        start_unix_ms=args.start_unix_ms,
        end_unix_ms=args.end_unix_ms,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
