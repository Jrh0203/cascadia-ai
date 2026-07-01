#!/usr/bin/env python3
"""Render and validate the frozen ADR 0104 projected-control repair result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "complete-action-frontier-projected-control-repair-v1"
EXPECTED_HOSTS = {"john1", "john2", "john3", "john4"}
EXPECTED_SHARDS = set(range(4))
PHYSICAL_CLUSTER_CORES = 40
WORKERS_PER_HOST = 6


def _canonical_host(value: str) -> str:
    return "john1" if value.lower().startswith("johns-mac-mini") else value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return value


def validate_source_identities(paths: list[Path]) -> dict[str, Any]:
    reports = [load_json(path) for path in paths]
    hosts = {_canonical_host(str(report["host"])) for report in reports}
    bundles = {str(report["bundle_sha256"]) for report in reports}
    file_counts = {int(report["files"]) for report in reports}
    if hosts != EXPECTED_HOSTS:
        raise ValueError(f"source identity hosts differ: {sorted(hosts)}")
    if len(bundles) != 1 or len(file_counts) != 1:
        raise ValueError("cluster MLX source bundles are not identical")
    return {
        "hosts": sorted(hosts),
        "files": next(iter(file_counts)),
        "bundle_sha256": next(iter(bundles)),
    }


def validate_replay_comparisons(paths: list[Path]) -> dict[str, Any]:
    reports = [load_json(path) for path in paths]
    shards = {int(report["shard_index"]) for report in reports}
    if len(reports) != 4 or shards != EXPECTED_SHARDS:
        raise ValueError(f"replay comparison shards differ: {sorted(shards)}")
    if not all(report.get("scientific_payload_identical") is True for report in reports):
        raise ValueError("one or more ADR 0104 replay payloads differ")
    return {"reports": sorted(reports, key=lambda report: int(report["shard_index"]))}


def summarize_campaign_events(paths: list[Path]) -> dict[str, Any]:
    jobs = []
    for path in paths:
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        started = [record for record in records if record.get("event") == "started"]
        finished = [record for record in records if record.get("event") == "finished"]
        if len(started) != 1 or len(finished) != 1:
            raise ValueError(f"event log must contain one start and finish: {path}")
        start = started[0]
        finish = finished[0]
        if start["name"] != finish["name"] or int(finish["return_code"]) != 0:
            raise ValueError(f"campaign event identity or status differs: {path}")
        jobs.append(
            {
                "name": str(start["name"]),
                "host": _canonical_host(str(start["host"])),
                "start": float(start["started_unix_seconds"]),
                "end": float(finish["ended_unix_seconds"]),
                "elapsed": float(finish["elapsed_seconds"]),
                "replay": "replay" in str(start["name"]),
            }
        )
    origins = [job for job in jobs if not job["replay"]]
    replays = [job for job in jobs if job["replay"]]
    if len(origins) != 4 or len(replays) != 4:
        raise ValueError("ADR 0104 campaign requires four origins and four replays")
    origin_seconds = sum(job["elapsed"] for job in origins)
    replay_seconds = sum(job["elapsed"] for job in replays)
    total = origin_seconds + replay_seconds
    return {
        "origin_makespan_seconds": max(job["end"] for job in origins)
        - min(job["start"] for job in origins),
        "end_to_end_makespan_seconds": max(job["end"] for job in jobs)
        - min(job["start"] for job in jobs),
        "total_job_seconds": total,
        "confirmation_compute_fraction": replay_seconds / total,
    }


def summarize_worker_utilization(
    shard_reports: list[dict[str, Any]],
    replay_summary: dict[str, Any],
    campaign_summary: dict[str, Any],
) -> dict[str, Any]:
    origin_telemetry = [report["telemetry"] for report in shard_reports]
    replay_telemetry = [
        comparison["replay_telemetry"]
        for comparison in replay_summary["reports"]
    ]

    def worker_seconds(values: list[dict[str, Any]]) -> float:
        return sum(
            float(worker["elapsed_seconds"])
            for telemetry in values
            for worker in telemetry["workers"]
        )

    origin_worker_seconds = worker_seconds(origin_telemetry)
    replay_worker_seconds = worker_seconds(replay_telemetry)
    origin_makespan = float(campaign_summary["origin_makespan_seconds"])
    end_to_end = float(campaign_summary["end_to_end_makespan_seconds"])
    all_worker_seconds = origin_worker_seconds + replay_worker_seconds
    return {
        "origin_worker_seconds": origin_worker_seconds,
        "replay_worker_seconds": replay_worker_seconds,
        "origin_physical_core_occupancy": (
            origin_worker_seconds / (PHYSICAL_CLUSTER_CORES * origin_makespan)
        ),
        "origin_allocated_worker_occupancy": (
            origin_worker_seconds
            / (len(EXPECTED_HOSTS) * WORKERS_PER_HOST * origin_makespan)
        ),
        "campaign_physical_core_occupancy": (
            all_worker_seconds / (PHYSICAL_CLUSTER_CORES * end_to_end)
        ),
        "maximum_worker_rss_bytes": max(
            int(telemetry["maximum_worker_rss_bytes"])
            for telemetry in origin_telemetry + replay_telemetry
        ),
        "maximum_parent_rss_bytes": max(
            int(telemetry["peak_process_rss_bytes"])
            for telemetry in origin_telemetry + replay_telemetry
        ),
        "all_process_swaps_zero": all(
            int(telemetry["process_swaps"]) == 0
            and int(telemetry["worker_process_swaps"]) == 0
            for telemetry in origin_telemetry + replay_telemetry
        ),
        "all_system_swap_growth_nonpositive": all(
            telemetry["system_swap_delta_bytes"] is not None
            and int(telemetry["system_swap_delta_bytes"]) <= 0
            for telemetry in origin_telemetry + replay_telemetry
        ),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(
    *,
    combined_report: dict[str, Any],
    shard_reports: list[dict[str, Any]],
    source_identity: dict[str, Any],
    replay_summary: dict[str, Any],
    campaign_summary: dict[str, Any],
) -> str:
    if combined_report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("combined report has the wrong experiment ID")
    combined = combined_report["scientific"]
    shards = sorted(
        shard_reports,
        key=lambda report: int(report["scientific"]["shard_index"]),
    )
    utilization = summarize_worker_utilization(
        shards,
        replay_summary,
        campaign_summary,
    )
    lines = [
        "# Complete-Action Frontier Projected-Control Repair V1 Result",
        "",
        f"Classification: `{combined['classification']}`.",
        "",
        "ADR 0104 changed only the independent projected control's maximum "
        "iteration count from 10,000 to 100,000 on the frozen first 24 ADR "
        "0103 groups. The frozen analytic, free-AdamW, and neural evidence "
        "was not rerun. Sealed test, gameplay, new teacher compute, cloud, "
        "and external compute remained closed.",
        "",
        "## Numerical Result",
        "",
        "| Shard | Host | Groups converged | Recall | Exact sets | Max KKT | Max objective gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for report in shards:
        scientific = report["scientific"]
        groups = scientific["groups"]
        converged = sum(int(bool(group["converged"])) for group in groups)
        lines.append(
            f"| {scientific['shard_index']} | "
            f"{_canonical_host(str(report['telemetry']['host']))} | "
            f"{converged}/{len(groups)} | "
            f"{_percent(scientific['aggregate']['target_positive_recall'])} | "
            f"{_percent(scientific['aggregate']['target_set_exact_fraction'])} | "
            f"{scientific['maximum_kkt_violation']:.3e} | "
            f"{scientific['maximum_objective_gap_from_analytic']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"- Aggregate recall: "
            f"{_percent(combined['aggregate']['target_positive_recall'])}.",
            f"- Aggregate exact-set recovery: "
            f"{_percent(combined['aggregate']['target_set_exact_fraction'])}.",
            f"- Maximum projected KKT violation: "
            f"`{combined['maximum_kkt_violation']:.3e}` against `1e-8`.",
            f"- Maximum absolute objective gap: "
            f"`{combined['maximum_objective_gap_from_analytic']:.3e}` "
            "against `1e-7`.",
            "",
            "## Frozen Gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for name, passed in sorted(combined["gates"].items()):
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "Both decision-tolerance numerical gates passed. The campaign is "
            "still invalid because 12 of 24 groups did not reach the stricter "
            "`1e-9` stopping tolerance and three groups selected a different "
            "target set despite tiny objective gaps. The preregistered "
            "optimizer treatment is therefore not authorized.",
            "",
            "## Cross-Host Replays",
            "",
            "| Shard | Origin | Replay | Scientific BLAKE3 |",
            "|---:|---|---|---|",
        ]
    )
    for comparison in replay_summary["reports"]:
        lines.append(
            f"| {comparison['shard_index']} | "
            f"{_canonical_host(str(comparison['origin_host']))} | "
            f"{_canonical_host(str(comparison['replay_host']))} | "
            f"`{comparison['origin_scientific_blake3']}` |"
        )
    lines.extend(
        [
            "",
            "Every origin/replay scientific payload was bit-identical.",
            "",
            "## Campaign Throughput",
            "",
            f"- Origin critical path: "
            f"{campaign_summary['origin_makespan_seconds']:.2f} seconds.",
            f"- End-to-end origin plus confirmation: "
            f"{campaign_summary['end_to_end_makespan_seconds']:.2f} seconds.",
            f"- Scheduled shard time: "
            f"{campaign_summary['total_job_seconds']:.2f} host-seconds.",
            f"- Confirmation compute fraction: "
            f"{campaign_summary['confirmation_compute_fraction']:.2%}.",
            f"- Origin physical-core occupancy from worker CPU intervals: "
            f"{_percent(utilization['origin_physical_core_occupancy'])}; "
            f"allocated six-worker occupancy was "
            f"{_percent(utilization['origin_allocated_worker_occupancy'])}.",
            f"- End-to-end physical-core occupancy from worker intervals: "
            f"{_percent(utilization['campaign_physical_core_occupancy'])}.",
            "- Duplicate discovery fraction: 0.00%; origins solved disjoint "
            "groups and replay work was explicit confirmation.",
            f"- Peak parent RSS: "
            f"{utilization['maximum_parent_rss_bytes'] / 2**20:.1f} MiB; "
            f"peak worker RSS: "
            f"{utilization['maximum_worker_rss_bytes'] / 2**20:.1f} MiB.",
            f"- Process swaps zero: "
            f"{str(utilization['all_process_swaps_zero']).lower()}; "
            f"attributable system swap growth absent: "
            f"{str(utilization['all_system_swap_growth_nonpositive']).lower()}.",
            f"- Source identity: {source_identity['files']} files, "
            f"`{source_identity['bundle_sha256']}`, identical on "
            f"{', '.join(source_identity['hosts'])}.",
            "",
            "The worker traces show severe runtime skew: several groups "
            "finished in under five seconds while the longest took more than "
            "150 seconds. Static six-group host shards therefore left cores "
            "idle near each barrier. Future independent-group campaigns "
            "should use smaller resumable work units and a shared dynamic "
            "queue across john1-john4.",
            "",
            "## Authorized Successor",
            "",
            "Preregister one independent arbitrary-precision reconstruction "
            "of the frozen analytic optimum and selector. It must use a "
            "separate high-precision derivation, retain the same 24 groups, "
            "and replay across hosts. More projected iterations, threshold "
            "relaxation, and model treatments remain unauthorized.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--source-identity", type=Path, action="append", required=True)
    parser.add_argument("--replay-comparison", type=Path, action="append", required=True)
    parser.add_argument("--event-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shard_reports = [load_json(path) for path in args.shard]
    markdown = render_markdown(
        combined_report=load_json(args.combined),
        shard_reports=shard_reports,
        source_identity=validate_source_identities(args.source_identity),
        replay_summary=validate_replay_comparisons(args.replay_comparison),
        campaign_summary=summarize_campaign_events(args.event_log),
    )
    write_text_atomic(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
