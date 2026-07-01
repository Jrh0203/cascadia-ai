#!/usr/bin/env python3
"""Move the unstarted R0 shard-1 timing group to an available host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import r0_spatial_campaign as campaign
from cluster_research_queue import (
    QueueError,
    add_task,
    cancel_pending_tasks,
    locked_queue,
)
from rust_experiment_bundle import BundleError, validate_bundle

DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_OUTPUT = (
    campaign.DEFAULT_EXPERIMENT_ROOT / "queue-shard-1-work-conserving-rebalance-v1.json"
)
PREFIX = "r0f-work-conserving"
SHARD_INDEX = 1
ACTOR = "research-coordinator"
REASON = (
    "Moved the complete unstarted shard-1 timing replicate group away from john2 "
    "so R0 can finish while john2 completes the authorized MLX dropout origin. "
    "All three replacements run on one host and preserve the frozen source, data, "
    "partition, paired-arm, and independent-process contracts."
)


class TimingRebalanceError(RuntimeError):
    """Raised when the timing group cannot be reassigned atomically."""


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: float,
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: str,
    stop_rule: str,
    cpu_cores: int,
    memory_gib: float,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "experiment_id": campaign.EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 1.0,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": True,
        "decision_terminal": decision_terminal,
        "compatible_hosts": compatible_hosts,
        "dependencies": dependencies,
        "command": command,
        "artifact_path": artifact_path,
        "stop_rule": stop_rule,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gib": memory_gib,
            "uses_mlx": False,
        },
    }


def _remote_path(host: str, relative: Path) -> str:
    return str(campaign.REMOTE_ROOTS[host] / relative)


def _repeated_flag(flag: str, values: list[str]) -> list[str]:
    return [item for value in values for item in (flag, value)]


def _accepted_fanout_ids() -> list[str]:
    return [
        (
            f"r0f-rebalanced-fanout-{split}-part-1"
            if part_index == 1
            else f"r0f-fanout-{split}-part-{part_index}"
        )
        for split in ("train", "validation")
        for part_index in range(4)
    ]


def _dataset_roots() -> list[Path]:
    return [part.root for part in campaign.dataset_parts()]


def _clean_shard_zero_reports() -> tuple[list[str], list[Path]]:
    task_ids = []
    reports = []
    for replicate_index in range(campaign.REQUIRED_REPLICATES):
        task_ids.append(f"r0f-clean-benchmark-shard-0-replicate-{replicate_index}")
        reports.append(
            campaign.DEFAULT_EXPERIMENT_ROOT
            / "runs"
            / f"john1-source-frozen-clean-shard-0-replicate-{replicate_index}.json"
        )
    return task_ids, reports


def _existing_remote_shard_reports() -> tuple[list[str], list[Path], list[str]]:
    task_ids = []
    reports = []
    collection_args = []
    for shard_index, host in ((2, "john3"), (3, "john4")):
        for replicate_index in range(campaign.REQUIRED_REPLICATES):
            task_ids.append(
                f"r0f-benchmark-shard-{shard_index}-replicate-{replicate_index}"
            )
            report = (
                campaign.DEFAULT_EXPERIMENT_ROOT
                / "runs"
                / (
                    f"{host}-source-frozen-shard-{shard_index}-"
                    f"replicate-{replicate_index}.json"
                )
            )
            reports.append(report)
            collection_args.extend(
                [
                    "--artifact",
                    f"{host}:{_remote_path(host, report)}",
                    str(report),
                ]
            )
    return task_ids, reports, collection_args


def build_rebalance(*, bundle_relative: Path, replacement_host: str) -> dict[str, Any]:
    if replacement_host not in {"john3", "john4"}:
        raise TimingRebalanceError("replacement host must be john3 or john4")
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise TimingRebalanceError("bundle path must be repository-relative")

    replacement_tasks: list[dict[str, Any]] = []
    replacement_ids: list[str] = []
    replacement_reports: list[Path] = []
    replacement_collection_args: list[str] = []
    dataset_args = _repeated_flag(
        "--dataset-root",
        [_remote_path(replacement_host, root) for root in _dataset_roots()],
    )
    for replicate_index in range(campaign.REQUIRED_REPLICATES):
        task_id = f"{PREFIX}-benchmark-shard-{SHARD_INDEX}-replicate-{replicate_index}"
        report = (
            campaign.DEFAULT_EXPERIMENT_ROOT
            / "runs"
            / (
                f"{replacement_host}-source-frozen-reassigned-shard-{SHARD_INDEX}-"
                f"replicate-{replicate_index}.json"
            )
        )
        replacement_ids.append(task_id)
        replacement_reports.append(report)
        replacement_collection_args.extend(
            [
                "--artifact",
                f"{replacement_host}:{_remote_path(replacement_host, report)}",
                str(report),
            ]
        )
        replacement_tasks.append(
            _task(
                task_id=task_id,
                title=f"Work-conserving R0 shard 1 timing replicate {replicate_index}",
                decision=(
                    "Measure the previously unstarted shard-1 partition on an "
                    "available host without changing any scientific input"
                ),
                workload_class="independent-experiment",
                priority=28 + replicate_index,
                expected_runtime_seconds=900,
                decision_terminal=False,
                compatible_hosts=[replacement_host],
                dependencies=_accepted_fanout_ids(),
                command=[
                    "/usr/bin/env",
                    "-C",
                    _remote_path(replacement_host, bundle_relative / "source"),
                    _remote_path(
                        replacement_host,
                        bundle_relative / "bin/spatial_representation_benchmark",
                    ),
                    *dataset_args,
                    "--shard-index",
                    str(SHARD_INDEX),
                    "--shard-count",
                    str(campaign.SHARD_COUNT),
                    "--records",
                    "0",
                    "--iterations",
                    str(campaign.BENCHMARK_ITERATIONS),
                    "--replicate-index",
                    str(replicate_index),
                    "--output",
                    _remote_path(replacement_host, report),
                ],
                artifact_path=str(report),
                stop_rule=(
                    "Round-trip every shard-1 record and write one immutable report; "
                    "all three replicas must remain on the same replacement host."
                ),
                cpu_cores=10,
                memory_gib=8.0,
            )
        )

    clean_ids, clean_reports = _clean_shard_zero_reports()
    existing_ids, existing_reports, existing_collection_args = (
        _existing_remote_shard_reports()
    )
    collection_id = f"{PREFIX}-benchmark-report-collection"
    collection_report = (
        campaign.DEFAULT_EXPERIMENT_ROOT
        / "reports/source-frozen-work-conserving-benchmark-report-collection.json"
    )
    replacement_tasks.append(
        _task(
            task_id=collection_id,
            title="Collect work-conserving distributed R0 benchmark reports",
            decision="Retrieve all accepted remote reports with checksum proof",
            workload_class="shared-prerequisite",
            priority=40,
            expected_runtime_seconds=120,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[*clean_ids, *replacement_ids, *existing_ids],
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_collect.py",
                *replacement_collection_args,
                *existing_collection_args,
                "--output",
                str(collection_report),
            ],
            artifact_path=str(collection_report),
            stop_rule="All nine remote reports must be retrieved with matching hashes.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    all_reports = [*clean_reports, *replacement_reports, *existing_reports]
    forward = (
        campaign.DEFAULT_EXPERIMENT_ROOT
        / "reports/extraction-source-frozen-work-conserving-aggregate-forward.json"
    )
    reverse = (
        campaign.DEFAULT_EXPERIMENT_ROOT
        / "reports/extraction-source-frozen-work-conserving-aggregate-reverse.json"
    )
    forward_id = f"{PREFIX}-extraction-classification-forward"
    reverse_id = f"{PREFIX}-extraction-classification-reverse"
    replacement_tasks.extend(
        [
            _task(
                task_id=forward_id,
                title="Classify work-conserving R0 extraction evidence",
                decision=(
                    "Apply every frozen R0 gate to the clean host-rebalanced report set"
                ),
                workload_class="shared-prerequisite",
                priority=50,
                expected_runtime_seconds=60,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=[
                    ".venv/bin/python",
                    "tools/spatial_representation_benchmark_report.py",
                    *_repeated_flag("--report", [str(report) for report in all_reports]),
                    "--required-replicates",
                    str(campaign.REQUIRED_REPLICATES),
                    "--output",
                    str(forward),
                ],
                artifact_path=str(forward),
                stop_rule=(
                    "Fail closed on missing replicas, source drift, semantic loss, "
                    "or invalid timing."
                ),
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=reverse_id,
                title="Reclassify host-rebalanced R0 evidence in reverse order",
                decision="Prove the accepted aggregate is independent of report order",
                workload_class="replica",
                priority=51,
                expected_runtime_seconds=60,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[forward_id],
                command=[
                    ".venv/bin/python",
                    "tools/spatial_representation_benchmark_report.py",
                    *_repeated_flag(
                        "--report",
                        [str(report) for report in reversed(all_reports)],
                    ),
                    "--required-replicates",
                    str(campaign.REQUIRED_REPLICATES),
                    "--output",
                    str(reverse),
                ],
                artifact_path=str(reverse),
                stop_rule="The reverse-order aggregate must be independently valid.",
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=f"{PREFIX}-extraction-merge-order-proof",
                title="Verify host-rebalanced R0 merge-order determinism",
                decision="Require forward and reverse aggregates to be byte-identical",
                workload_class="shared-prerequisite",
                priority=52,
                expected_runtime_seconds=10,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[forward_id, reverse_id],
                command=["cmp", "-s", str(forward), str(reverse)],
                artifact_path=str(reverse),
                stop_rule="Byte identity is mandatory for the accepted result.",
                cpu_cores=1,
                memory_gib=0.25,
            ),
        ]
    )

    return {
        "schema_version": 1,
        "experiment_id": campaign.EXPERIMENT_ID,
        "reason": REASON,
        "replacement_host": replacement_host,
        "cancel_task_ids": [
            *[
                f"r0f-benchmark-shard-{SHARD_INDEX}-replicate-{replicate_index}"
                for replicate_index in range(campaign.REQUIRED_REPLICATES)
            ],
            "r0f-clean-benchmark-report-collection",
            "r0f-clean-extraction-classification-forward",
            "r0f-clean-extraction-classification-reverse",
            "r0f-clean-extraction-merge-order-proof",
        ],
        "required_completed_task_ids": [*clean_ids, *existing_ids],
        "replacement_tasks": replacement_tasks,
    }


def apply_rebalance(state: dict[str, Any], plan: dict[str, Any]) -> None:
    by_id = {task["id"]: task for task in state["tasks"]}
    replacement_ids = [task["id"] for task in plan["replacement_tasks"]]
    duplicates = sorted(set(by_id).intersection(replacement_ids))
    if duplicates:
        raise TimingRebalanceError(f"replacement task ids already exist: {duplicates}")
    for task_id in plan["required_completed_task_ids"]:
        if by_id.get(task_id, {}).get("status") != "completed":
            raise TimingRebalanceError(f"required accepted task is not complete: {task_id}")
    for task_id in plan["cancel_task_ids"]:
        task = by_id.get(task_id)
        if task is None:
            raise TimingRebalanceError(f"missing task required for rebalance: {task_id}")
        if task["status"] not in {"ready", "blocked", "failed"} or task.get("claim") is not None:
            raise TimingRebalanceError(f"task is no longer safely reassignable: {task_id}")

    cancel_pending_tasks(
        state,
        task_ids=list(plan["cancel_task_ids"]),
        actor=ACTOR,
        reason=str(plan["reason"]),
    )
    for specification in plan["replacement_tasks"]:
        add_task(state, specification)


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replacement-host", choices=("john3", "john4"), default="john3")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    try:
        bundle_relative = args.bundle.resolve().relative_to(repository)
        manifest = validate_bundle(repository / bundle_relative)
        campaign.validate_provenance_source_bundle(manifest)
        plan = build_rebalance(
            bundle_relative=bundle_relative,
            replacement_host=args.replacement_host,
        )
        payload = {
            **plan,
            "bundle_id": manifest["bundle_id"],
            "bundle": str(bundle_relative),
        }
        if args.apply:
            with locked_queue(args.queue) as state:
                apply_rebalance(state, plan)
        _write_output(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        BundleError,
        OSError,
        QueueError,
        TimingRebalanceError,
        ValueError,
    ) as error:
        print(f"R0 timing host rebalance error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
