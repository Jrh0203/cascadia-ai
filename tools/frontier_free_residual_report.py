#!/usr/bin/env python3
"""Render and validate the frozen ADR 0103 free-residual result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "complete-action-frontier-free-residual-audit-v1"
EXPECTED_HOSTS = {"john1", "john2", "john3", "john4"}
EXPECTED_REPLAYS = {
    ("analytic-optimum", None),
    ("free-adam", None),
    ("projected-control", None),
    *{("neural-continuation-shard", index) for index in range(4)},
}


def _canonical_host(value: str) -> str:
    return "john1" if value.lower().startswith("johns-mac-mini") else value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return value


def validate_source_identities(paths: list[Path]) -> dict[str, Any]:
    reports = [load_json(path) for path in paths]
    hosts = {str(report["host"]) for report in reports}
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
    identities = {
        (str(report["arm"]), report.get("group_index")) for report in reports
    }
    if len(reports) != len(EXPECTED_REPLAYS) or identities != EXPECTED_REPLAYS:
        raise ValueError(f"replay comparison identities differ: {identities}")
    if not all(report.get("scientific_payload_identical") is True for report in reports):
        raise ValueError("one or more ADR 0103 replay payloads differ")
    return {
        "reports": sorted(
            reports,
            key=lambda report: (
                str(report["arm"]),
                -1 if report.get("group_index") is None else int(report["group_index"]),
            ),
        )
    }


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
    if len(origins) != 7 or len(replays) != 7:
        raise ValueError("ADR 0103 campaign requires seven origins and seven replays")
    origin_seconds = sum(job["elapsed"] for job in origins)
    replay_seconds = sum(job["elapsed"] for job in replays)
    host_seconds: dict[str, float] = {}
    host_jobs: dict[str, int] = {}
    for job in jobs:
        host = str(job["host"])
        host_seconds[host] = host_seconds.get(host, 0.0) + float(job["elapsed"])
        host_jobs[host] = host_jobs.get(host, 0) + 1
    total = origin_seconds + replay_seconds
    return {
        "origin_makespan_seconds": max(job["end"] for job in origins)
        - min(job["start"] for job in origins),
        "end_to_end_makespan_seconds": max(job["end"] for job in jobs)
        - min(job["start"] for job in jobs),
        "total_job_seconds": total,
        "confirmation_compute_fraction": replay_seconds / total,
        "host_seconds": dict(sorted(host_seconds.items())),
        "host_jobs": dict(sorted(host_jobs.items())),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _scientific(report: dict[str, Any], arm: str) -> dict[str, Any]:
    if report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"{arm} report has the wrong experiment ID")
    scientific = report.get("scientific")
    if not isinstance(scientific, dict) or scientific.get("arm") != arm:
        raise ValueError(f"{arm} report has the wrong arm identity")
    return scientific


def _successor(classification: str) -> str:
    values = {
        "free_residual_pipeline_invalid": (
            "Preregister and replay only the failed numerical-control gate. "
            "No model treatment is authorized from this invalid campaign."
        ),
        "scale16_objective_box_misaligned": (
            "Test one selector-aligned objective inside the unchanged box."
        ),
        "frozen_optimizer_hyperparameters_insufficient": (
            "Test one calibrated local optimizer mechanism before changing "
            "representation."
        ),
        "full_model_local_budget_insufficient": (
            "Test the smallest local exposure budget that passed."
        ),
        "public_observable_representation_insufficient": (
            "Test one new public-observable representation mechanism."
        ),
        "local_failure_not_reproduced": (
            "Re-audit ADR 0102's local-fit execution path."
        ),
        "local_mechanism_unresolved": (
            "Run another bounded diagnostic; no full trainer is authorized."
        ),
    }
    return values[classification]


def render_markdown(
    *,
    combined_report: dict[str, Any],
    analytic_report: dict[str, Any],
    free_report: dict[str, Any],
    projected_report: dict[str, Any],
    neural_reports: list[dict[str, Any]],
    source_identity: dict[str, Any],
    replay_summary: dict[str, Any],
    campaign_summary: dict[str, Any],
) -> str:
    combined = _scientific(combined_report, "combined")
    analytic = _scientific(analytic_report, "analytic-optimum")
    free = _scientific(free_report, "free-adam")
    projected = _scientific(projected_report, "projected-control")
    neural = sorted(
        (
            _scientific(report, "neural-continuation-shard")
            for report in neural_reports
        ),
        key=lambda report: int(report["group_index"]),
    )
    classification = str(combined["classification"])
    lines = [
        "# Complete-Action Frontier Free-Residual Audit V1 Result",
        "",
        f"Classification: `{classification}`.",
        "",
        "ADR 0103 separated objective geometry, free-parameter optimization, "
        "and long-horizon neural local fit on the frozen ADR 0102 cohort. "
        "Sealed test, gameplay, new teacher compute, cloud, and external "
        "compute remained unused.",
        "",
        "## Objective And Optimizer",
        "",
        "| Diagnostic | Recall | Exact sets | Mean objective |",
        "|---|---:|---:|---:|",
        f"| analytic box optimum | "
        f"{_percent(analytic['aggregate']['target_positive_recall'])} | "
        f"{_percent(analytic['aggregate']['target_set_exact_fraction'])} | "
        f"{analytic['aggregate']['mean_objective']:.6f} |",
        f"| selector ceiling | "
        f"{_percent(analytic['selector_ceiling']['target_positive_recall'])} | "
        f"{_percent(analytic['selector_ceiling']['target_set_exact_fraction'])} | "
        f"{analytic['selector_ceiling']['mean_objective']:.6f} |",
        f"| free AdamW, 120 updates | "
        f"{_percent(free['aggregate_at_120']['target_positive_recall'])} | "
        f"{_percent(free['aggregate_at_120']['target_set_exact_fraction'])} | "
        f"{free['aggregate_at_120']['mean_objective']:.6f} |",
        f"| free AdamW, 1,200 updates | "
        f"{_percent(free['aggregate']['target_positive_recall'])} | "
        f"{_percent(free['aggregate']['target_set_exact_fraction'])} | "
        f"{free['aggregate']['mean_objective']:.6f} |",
        f"| projected control | "
        f"{_percent(projected['aggregate']['target_positive_recall'])} | "
        f"{_percent(projected['aggregate']['target_set_exact_fraction'])} | "
        f"{projected['aggregate']['mean_objective']:.6f} |",
        "",
        f"- Analytic maximum KKT violation: "
        f"`{analytic['maximum_kkt_violation']:.3e}`.",
        f"- Projected maximum KKT violation: "
        f"`{projected['maximum_kkt_violation']:.3e}` against `1e-8`.",
        f"- Projected maximum objective gap: "
        f"`{projected['maximum_objective_gap_from_analytic']:.3e}` "
        "against `1e-7`.",
        "",
        "## Long-Horizon Neural Fit",
        "",
        "| Group | Host | Recall at 120 | Recall at 1,200 | Exact at 1,200 |",
        "|---:|---|---:|---:|---:|",
    ]
    neural_hosts = {
        int(value["group_index"]): value["host"]
        for value in combined["neural_telemetry"]
    }
    for report in neural:
        at_120 = next(
            event["metrics"]
            for event in report["trajectory"]
            if event["exposures_per_group"] == 120
        )
        final = report["final"]
        lines.append(
            f"| {report['group_index']} | "
            f"{_canonical_host(neural_hosts[int(report['group_index'])])} | "
            f"{_percent(at_120['target_positive_recall'])} | "
            f"{_percent(final['target_positive_recall'])} | "
            f"{_percent(final['target_set_exact_fraction'])} |"
        )
    lines.extend(
        [
            "",
            f"- Four-group aggregate at 120 exposures: "
            f"{_percent(combined['neural_at_120']['target_positive_recall'])} "
            "recall, "
            f"{_percent(combined['neural_at_120']['target_set_exact_fraction'])} "
            "exact sets.",
            f"- Four-group aggregate at 1,200 exposures: "
            f"{_percent(combined['neural_at_1200']['target_positive_recall'])} "
            "recall, "
            f"{_percent(combined['neural_at_1200']['target_set_exact_fraction'])} "
            "exact sets.",
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
            "## Cross-Host Replays",
            "",
            "| Arm | Group | Origin | Replay | Scientific BLAKE3 |",
            "|---|---:|---|---|---|",
        ]
    )
    for comparison in replay_summary["reports"]:
        group = (
            "-"
            if comparison.get("group_index") is None
            else str(comparison["group_index"])
        )
        lines.append(
            f"| {comparison['arm']} | {group} | "
            f"{_canonical_host(str(comparison['origin_host']))} | "
            f"{_canonical_host(str(comparison['replay_host']))} | "
            f"`{comparison['origin_scientific_blake3']}` |"
        )
    lines.extend(
        [
            "",
            "Every origin/replay scientific payload was identical.",
            "",
            "## Campaign Throughput",
            "",
            f"- Origin decision makespan: "
            f"{campaign_summary['origin_makespan_seconds']:.2f} seconds.",
            f"- End-to-end origin plus confirmation makespan: "
            f"{campaign_summary['end_to_end_makespan_seconds']:.2f} seconds.",
            f"- Scheduled scientific job time: "
            f"{campaign_summary['total_job_seconds']:.2f} host-seconds.",
            f"- Confirmation compute fraction: "
            f"{campaign_summary['confirmation_compute_fraction']:.2%}.",
            "- Duplicate discovery fraction: 0.00%; neural continuation used "
            "four disjoint group shards.",
            f"- Source identity: {source_identity['files']} files, "
            f"`{source_identity['bundle_sha256']}`, identical on "
            f"{', '.join(source_identity['hosts'])}.",
            "",
            "| Host | Jobs | Scheduled seconds |",
            "|---|---:|---:|",
        ]
    )
    for host, seconds in campaign_summary["host_seconds"].items():
        lines.append(
            f"| {host} | {campaign_summary['host_jobs'][host]} | "
            f"{seconds:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Authorized Successor",
            "",
            _successor(classification),
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
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--free-adam", type=Path, required=True)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--neural", type=Path, action="append", required=True)
    parser.add_argument("--source-identity", type=Path, action="append", required=True)
    parser.add_argument("--replay-comparison", type=Path, action="append", required=True)
    parser.add_argument("--event-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = render_markdown(
        combined_report=load_json(args.combined),
        analytic_report=load_json(args.analytic),
        free_report=load_json(args.free_adam),
        projected_report=load_json(args.projected),
        neural_reports=[load_json(path) for path in args.neural],
        source_identity=validate_source_identities(args.source_identity),
        replay_summary=validate_replay_comparisons(args.replay_comparison),
        campaign_summary=summarize_campaign_events(args.event_log),
    )
    write_text_atomic(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
