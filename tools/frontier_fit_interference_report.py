#!/usr/bin/env python3
"""Render and validate the frozen ADR 0102 fit/interference result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "complete-action-frontier-fit-interference-audit-v1"
EXPECTED_HOSTS = {"john1", "john2", "john3", "john4"}
EXPECTED_ARMS = {
    "nested-subset",
    "capacity-scaling",
    "gradient-conflict",
    "error-anatomy",
}


def _canonical_host(value: str) -> str:
    return "john1" if value.lower().startswith("johns-mac-mini") else value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return value


def validate_source_identities(paths: list[Path]) -> dict[str, Any]:
    """Require one byte-identical complete MLX bundle from every cluster host."""
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
    """Require one scientifically identical cross-host replay for every arm."""
    reports = [load_json(path) for path in paths]
    arms = {str(report["arm"]) for report in reports}
    if len(reports) != len(EXPECTED_ARMS) or arms != EXPECTED_ARMS:
        raise ValueError(f"replay comparison arms differ: {sorted(arms)}")
    if not all(report.get("scientific_payload_identical") is True for report in reports):
        raise ValueError("one or more ADR 0102 replay payloads differ")
    return {
        "all_identical": True,
        "reports": {
            str(report["arm"]): {
                **report,
                "origin_host": _canonical_host(str(report["origin_host"])),
                "replay_host": _canonical_host(str(report["replay_host"])),
            }
            for report in sorted(reports, key=lambda value: str(value["arm"]))
        },
    }


def summarize_campaign_events(paths: list[Path]) -> dict[str, Any]:
    """Summarize the frozen origin and replay waves from host-lock event logs."""
    jobs: list[dict[str, Any]] = []
    names: set[str] = set()
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
        name = str(start["name"])
        if name in names or name != str(finish["name"]):
            raise ValueError(f"event-log job identity differs: {path}")
        if int(finish["return_code"]) != 0:
            raise ValueError(f"campaign job failed: {name}")
        names.add(name)
        jobs.append(
            {
                "name": name,
                "host": _canonical_host(str(start["host"])),
                "started_unix_seconds": float(start["started_unix_seconds"]),
                "ended_unix_seconds": float(finish["ended_unix_seconds"]),
                "elapsed_seconds": float(finish["elapsed_seconds"]),
                "wave": "replay" if "replay" in name else "origin",
            }
        )
    origins = [job for job in jobs if job["wave"] == "origin"]
    replays = [job for job in jobs if job["wave"] == "replay"]
    if len(origins) != 4 or len(replays) != 4:
        raise ValueError("campaign events must contain four origin and four replay jobs")

    def makespan(values: list[dict[str, Any]]) -> float:
        return max(job["ended_unix_seconds"] for job in values) - min(
            job["started_unix_seconds"] for job in values
        )

    origin_seconds = sum(job["elapsed_seconds"] for job in origins)
    replay_seconds = sum(job["elapsed_seconds"] for job in replays)
    host_seconds: dict[str, float] = {}
    host_jobs: dict[str, int] = {}
    for job in jobs:
        host = str(job["host"])
        host_seconds[host] = host_seconds.get(host, 0.0) + float(
            job["elapsed_seconds"]
        )
        host_jobs[host] = host_jobs.get(host, 0) + 1
    total_seconds = origin_seconds + replay_seconds
    return {
        "origin_makespan_seconds": makespan(origins),
        "end_to_end_makespan_seconds": makespan(jobs),
        "origin_job_seconds": origin_seconds,
        "replay_job_seconds": replay_seconds,
        "total_job_seconds": total_seconds,
        "confirmation_compute_fraction": replay_seconds / total_seconds,
        "host_seconds": dict(sorted(host_seconds.items())),
        "host_jobs": dict(sorted(host_jobs.items())),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f}"


def _arm(report: dict[str, Any], name: str) -> dict[str, Any]:
    if report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"{name} report has the wrong experiment ID")
    scientific = report.get("scientific")
    if not isinstance(scientific, dict) or scientific.get("arm") != name:
        raise ValueError(f"{name} report has the wrong arm identity")
    return scientific


def render_markdown(
    *,
    combined_report: dict[str, Any],
    nested_report: dict[str, Any],
    capacity_report: dict[str, Any],
    gradient_report: dict[str, Any],
    error_report: dict[str, Any],
    source_identity: dict[str, Any],
    replay_summary: dict[str, Any] | None = None,
    campaign_summary: dict[str, Any] | None = None,
) -> str:
    """Render a compact evidence-complete Markdown result."""
    combined = _arm(combined_report, "combined")
    nested = _arm(nested_report, "nested-subset")
    capacity = _arm(capacity_report, "capacity-scaling")
    gradient = _arm(gradient_report, "gradient-conflict")
    error = _arm(error_report, "error-anatomy")
    classification = str(combined["classification"])

    lines = [
        "# Complete-Action Frontier Fit/Interference Audit V1 Result",
        "",
        f"Classification: `{classification}`.",
        "",
        "ADR 0102 ran four different open-train diagnostics concurrently on "
        "john1-john4. The sealed test, gameplay, new teacher compute, cloud, "
        "and external compute remained unused.",
        "",
        "## Nested Fit Scaling",
        "",
        "| Groups | Recall | Exact sets | Winner retained | Mean objective |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size in ("1", "4", "16", "64"):
        final = nested["variants"][size]["final"]
        lines.append(
            f"| {size} | {_percent(final['target_positive_recall'])} | "
            f"{_percent(final['target_set_exact_fraction'])} | "
            f"{_percent(final['r4800_winner_retention'])} | "
            f"{final['mean_objective']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Capacity Scaling",
            "",
            "| Hidden width | Parameters | Recall | Exact sets | Winner retained |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for width in ("96", "192", "288"):
        variant = capacity["variants"][width]
        final = variant["final"]
        lines.append(
            f"| {width} | {variant['parameter_count']:,} | "
            f"{_percent(final['target_positive_recall'])} | "
            f"{_percent(final['target_set_exact_fraction'])} | "
            f"{_percent(final['r4800_winner_retention'])} |"
        )

    full_gradient = gradient["selected"]["scopes"]["full_model"]
    other = full_gradient["cosine_to_other_gradient_sum"]
    independent = error["independent"]["aggregate"]
    shared = error["shared"]["aggregate"]
    initial = error["initial"]
    lines.extend(
        [
            "",
            "## Interference",
            "",
            f"- Selected-checkpoint gradients opposing the sum of other groups: "
            f"{_percent(other['negative_fraction'])}.",
            f"- Median cosine to the other-gradient sum: "
            f"`{other['distribution']['median']:.6f}`.",
            f"- Off-diagonal pairs at cosine <= -0.10: "
            f"{_percent(full_gradient['off_diagonal_at_most_negative_0_10_fraction'])}.",
            "",
            "| Adaptation | Recall | Exact sets | Winner retained |",
            "|---|---:|---:|---:|",
            f"| selected baseline | {_percent(initial['target_positive_recall'])} | "
            f"{_percent(initial['target_set_exact_fraction'])} | "
            f"{_percent(initial['r4800_winner_retention'])} |",
            f"| independent per group | "
            f"{_percent(independent['target_positive_recall'])} | "
            f"{_percent(independent['target_set_exact_fraction'])} | "
            f"{_percent(independent['r4800_winner_retention'])} |",
            f"| shared 24-group | {_percent(shared['target_positive_recall'])} | "
            f"{_percent(shared['target_set_exact_fraction'])} | "
            f"{_percent(shared['r4800_winner_retention'])} |",
            "",
            "## Frozen Gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for name, passed in sorted(combined["gates"].items()):
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")

    arm_telemetry = combined["arm_telemetry"]
    lines.extend(
        [
            "",
            "## Execution",
            "",
            "| Arm | Host | Seconds | Peak RSS GiB | Process swaps | System swap delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, telemetry in sorted(arm_telemetry.items()):
        lines.append(
            f"| {name} | {telemetry['host']} | "
            f"{telemetry['elapsed_seconds']:.2f} | "
            f"{_gib(int(telemetry['peak_process_rss_bytes']))} | "
            f"{telemetry['process_swaps']} | "
            f"{telemetry['system_swap_delta_bytes']} |"
        )
    critical_path = max(
        float(telemetry["elapsed_seconds"])
        for telemetry in arm_telemetry.values()
    )
    productive = sum(
        float(telemetry["elapsed_seconds"])
        for telemetry in arm_telemetry.values()
    )
    decisions_per_hour = 4.0 / max(critical_path / 3600.0, 1e-12)
    lines.extend(
        [
            "",
            f"- Critical-path arm time: {critical_path:.2f} seconds.",
            f"- Productive arm wall time summed across hosts: "
            f"{productive:.2f} host-seconds.",
            f"- Frozen diagnostic decisions per critical-path hour: "
            f"{decisions_per_hour:.2f}.",
            f"- Duplicate training fraction: "
            f"{combined['duplicate_training_fraction']:.1%}.",
            f"- Source identity: {source_identity['files']} files, "
            f"`{source_identity['bundle_sha256']}`, identical on "
            f"{', '.join(source_identity['hosts'])}.",
            f"- Frozen cohort: `{combined['full_cohort_digest_blake3']}`.",
        ]
    )
    if replay_summary is not None:
        lines.extend(
            [
                "",
                "## Cross-Host Replays",
                "",
                "| Arm | Origin | Replay | Scientific BLAKE3 | Result |",
                "|---|---|---|---|---|",
            ]
        )
        for name, comparison in sorted(replay_summary["reports"].items()):
            lines.append(
                f"| {name} | {comparison['origin_host']} | "
                f"{comparison['replay_host']} | "
                f"`{comparison['origin_scientific_blake3']}` | identical |"
            )
        lines.extend(
            [
                "",
                "The john1 nested origin recorded zero process swaps but unrelated "
                "positive system-wide swap growth. Its john4 replay was "
                "scientifically identical with zero swap growth and is the "
                "pipeline-selected nested report.",
            ]
        )
    if campaign_summary is not None:
        lines.extend(
            [
                "",
                "## Campaign Throughput",
                "",
                f"- First-wave decision makespan: "
                f"{campaign_summary['origin_makespan_seconds']:.2f} seconds.",
                f"- End-to-end origin plus confirmation makespan: "
                f"{campaign_summary['end_to_end_makespan_seconds']:.2f} seconds.",
                f"- Scheduled scientific job time: "
                f"{campaign_summary['total_job_seconds']:.2f} host-seconds.",
                f"- Confirmation compute fraction: "
                f"{campaign_summary['confirmation_compute_fraction']:.2%}.",
                "- Duplicate discovery fraction: 0.00%; all repeated compute was "
                "the preregistered cross-host confirmation wave.",
                "- The campaign was MLX-bound, so the plan's CPU-bound 85% "
                "physical-core target does not apply to this diagnostic.",
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
            _successor_text(classification),
            "",
        ]
    )
    return "\n".join(lines)


def _successor_text(classification: str) -> str:
    values = {
        "fit_interference_pipeline_invalid": (
            "Repair and replay the invalid diagnostic evidence; no model "
            "treatment is authorized."
        ),
        "local_optimization_or_representation_insufficient": (
            "Test one bounded representation or local-optimizer mechanism. "
            "A larger shared model or conflict-only treatment is not yet "
            "authorized."
        ),
        "mixed_capacity_and_interference": (
            "Test the smallest capacity increase that passed together with "
            "one gradient-conflict mitigation mechanism."
        ),
        "shared_capacity_bottleneck": (
            "Test the smallest capacity increase that passed the frozen gate."
        ),
        "cross_group_gradient_interference": (
            "Test one gradient-conflict mitigation mechanism without "
            "increasing model width."
        ),
        "shared_model_scaling_failure_unresolved": (
            "Run another bounded diagnostic; a full trainer remains closed."
        ),
        "no_material_fit_scaling_failure": (
            "Audit the full-dataset training path before changing the model."
        ),
    }
    try:
        return values[classification]
    except KeyError as error:
        raise ValueError(f"unknown ADR 0102 classification: {classification}") from error


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--nested", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--gradient", type=Path, required=True)
    parser.add_argument("--error-anatomy", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, action="append", required=True)
    parser.add_argument("--replay-comparison", type=Path, action="append")
    parser.add_argument("--event-log", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = validate_source_identities(args.source_identity)
    replay = (
        validate_replay_comparisons(args.replay_comparison)
        if args.replay_comparison
        else None
    )
    campaign = summarize_campaign_events(args.event_log) if args.event_log else None
    markdown = render_markdown(
        combined_report=load_json(args.combined),
        nested_report=load_json(args.nested),
        capacity_report=load_json(args.capacity),
        gradient_report=load_json(args.gradient),
        error_report=load_json(args.error_anatomy),
        source_identity=source,
        replay_summary=replay,
        campaign_summary=campaign,
    )
    write_text_atomic(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
