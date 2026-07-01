#!/usr/bin/env python3
"""Replace a contaminated R0 shard without mutating accepted timing artifacts."""

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
DEFAULT_OUTPUT = campaign.DEFAULT_EXPERIMENT_ROOT / "queue-john1-clean-timing-recovery-v1.json"
PREFIX = "r0f-clean"
ACTOR = "research-coordinator"
REASON = (
    "Replaced the john1 timing shard because local F5 cargo verification "
    "overlapped the original timing processes. Original artifacts remain "
    "quarantined and cannot enter the classifier."
)


class RecoveryError(RuntimeError):
    """Raised when a clean replacement wave cannot be installed atomically."""


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


def build_recovery(*, bundle_relative: Path) -> dict[str, Any]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise RecoveryError("bundle path must be repository-relative")

    host = "john1"
    clean_reports: list[Path] = []
    replacement_tasks: list[dict[str, Any]] = []
    dataset_args = _repeated_flag(
        "--dataset-root",
        [_remote_path(host, root) for root in _dataset_roots()],
    )
    for replicate_index in range(campaign.REQUIRED_REPLICATES):
        report = (
            campaign.DEFAULT_EXPERIMENT_ROOT
            / "runs"
            / f"john1-source-frozen-clean-shard-0-replicate-{replicate_index}.json"
        )
        clean_reports.append(report)
        replacement_tasks.append(
            _task(
                task_id=f"{PREFIX}-benchmark-shard-0-replicate-{replicate_index}",
                title=f"Clean R0 shard 0 timing replicate {replicate_index}",
                decision=(
                    "Replace a load-contaminated local timing process with an "
                    "isolated release-process invocation"
                ),
                workload_class="independent-experiment",
                priority=25 + replicate_index,
                expected_runtime_seconds=900,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=_accepted_fanout_ids(),
                command=[
                    "/usr/bin/env",
                    "-C",
                    _remote_path(host, bundle_relative / "source"),
                    _remote_path(
                        host,
                        bundle_relative / "bin/spatial_representation_benchmark",
                    ),
                    *dataset_args,
                    "--shard-index",
                    "0",
                    "--shard-count",
                    str(campaign.SHARD_COUNT),
                    "--records",
                    "0",
                    "--iterations",
                    str(campaign.BENCHMARK_ITERATIONS),
                    "--replicate-index",
                    str(replicate_index),
                    "--output",
                    _remote_path(host, report),
                ],
                artifact_path=str(report),
                stop_rule=(
                    "Run after local competing verification has stopped; round-trip "
                    "every record and write one new immutable report."
                ),
                cpu_cores=10,
                memory_gib=8.0,
            )
        )

    remote_ids: list[str] = []
    remote_reports: list[Path] = []
    remote_pairs: list[str] = []
    for shard_index, remote_host in enumerate(campaign.HOSTS[1:], start=1):
        for replicate_index in range(campaign.REQUIRED_REPLICATES):
            task_id = f"r0f-benchmark-shard-{shard_index}-replicate-{replicate_index}"
            report = (
                campaign.DEFAULT_EXPERIMENT_ROOT
                / "runs"
                / (
                    f"{remote_host}-source-frozen-shard-{shard_index}-"
                    f"replicate-{replicate_index}.json"
                )
            )
            remote_ids.append(task_id)
            remote_reports.append(report)
            remote_pairs.extend(
                [
                    "--artifact",
                    f"{remote_host}:{_remote_path(remote_host, report)}",
                    str(report),
                ]
            )

    collection_id = f"{PREFIX}-benchmark-report-collection"
    collection_report = (
        campaign.DEFAULT_EXPERIMENT_ROOT
        / "reports/source-frozen-clean-benchmark-report-collection.json"
    )
    replacement_tasks.append(
        _task(
            task_id=collection_id,
            title="Collect clean distributed R0 benchmark reports",
            decision="Retrieve every accepted remote process report with checksum proof",
            workload_class="shared-prerequisite",
            priority=40,
            expected_runtime_seconds=120,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[
                *[
                    f"{PREFIX}-benchmark-shard-0-replicate-{index}"
                    for index in range(campaign.REQUIRED_REPLICATES)
                ],
                *remote_ids,
            ],
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_collect.py",
                *remote_pairs,
                "--output",
                str(collection_report),
            ],
            artifact_path=str(collection_report),
            stop_rule="All nine accepted remote reports must match source checksums.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    all_reports = [*clean_reports, *remote_reports]
    forward = (
        campaign.DEFAULT_EXPERIMENT_ROOT
        / "reports/extraction-source-frozen-clean-aggregate-forward.json"
    )
    reverse = (
        campaign.DEFAULT_EXPERIMENT_ROOT
        / "reports/extraction-source-frozen-clean-aggregate-reverse.json"
    )
    report_args = _repeated_flag("--report", [str(report) for report in all_reports])
    forward_id = f"{PREFIX}-extraction-classification-forward"
    reverse_id = f"{PREFIX}-extraction-classification-reverse"
    replacement_tasks.extend(
        [
            _task(
                task_id=forward_id,
                title="Classify clean R0 extraction evidence in forward order",
                decision=("Apply every R0 gate while excluding the contaminated local timing wave"),
                workload_class="shared-prerequisite",
                priority=50,
                expected_runtime_seconds=60,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=[
                    ".venv/bin/python",
                    "tools/spatial_representation_benchmark_report.py",
                    *report_args,
                    "--required-replicates",
                    str(campaign.REQUIRED_REPLICATES),
                    "--output",
                    str(forward),
                ],
                artifact_path=str(forward),
                stop_rule=(
                    "Fail closed on missing clean replicas, shard drift, semantic "
                    "loss, or invalid timing."
                ),
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=reverse_id,
                title="Classify clean R0 extraction evidence in reverse order",
                decision="Prove the clean aggregate is independent of report order",
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
                stop_rule="The reverse-order clean aggregate must be independently valid.",
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=f"{PREFIX}-extraction-merge-order-proof",
                title="Verify clean R0 merge-order determinism",
                decision="Require clean forward and reverse reports to be byte-identical",
                workload_class="shared-prerequisite",
                priority=52,
                expected_runtime_seconds=10,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[forward_id, reverse_id],
                command=["cmp", "-s", str(forward), str(reverse)],
                artifact_path=str(reverse),
                stop_rule="Byte identity is mandatory for the accepted clean result.",
                cpu_cores=1,
                memory_gib=0.25,
            ),
        ]
    )

    return {
        "schema_version": 1,
        "experiment_id": campaign.EXPERIMENT_ID,
        "reason": REASON,
        "cancel_task_ids": [
            "r0f-benchmark-shard-0-replicate-1",
            "r0f-benchmark-report-collection",
            "r0f-extraction-classification-forward",
            "r0f-extraction-classification-reverse",
            "r0f-extraction-merge-order-proof",
        ],
        "quarantined_completed_task_ids": [
            "r0f-benchmark-shard-0-replicate-0",
        ],
        "already_cancelled_task_ids": [
            "r0f-benchmark-shard-0-replicate-2",
        ],
        "replacement_tasks": replacement_tasks,
    }


def apply_recovery(state: dict[str, Any], plan: dict[str, Any]) -> None:
    by_id = {task["id"]: task for task in state["tasks"]}
    replacement_ids = [task["id"] for task in plan["replacement_tasks"]]
    duplicates = sorted(set(by_id).intersection(replacement_ids))
    if duplicates:
        raise RecoveryError(f"replacement task ids already exist: {duplicates}")
    for task_id in plan["quarantined_completed_task_ids"]:
        if by_id.get(task_id, {}).get("status") != "completed":
            raise RecoveryError(f"expected completed quarantined task {task_id}")
    for task_id in plan["already_cancelled_task_ids"]:
        if by_id.get(task_id, {}).get("status") != "cancelled":
            raise RecoveryError(f"expected already-cancelled task {task_id}")
    for task_id in plan["cancel_task_ids"]:
        if task_id not in by_id:
            raise RecoveryError(f"missing task required for recovery: {task_id}")

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
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    try:
        bundle_relative = args.bundle.resolve().relative_to(repository)
        manifest = validate_bundle(repository / bundle_relative)
        campaign.validate_provenance_source_bundle(manifest)
        plan = build_recovery(bundle_relative=bundle_relative)
        payload = {
            **plan,
            "bundle_id": manifest["bundle_id"],
            "bundle": str(bundle_relative),
        }
        if args.apply:
            with locked_queue(args.queue) as state:
                apply_recovery(state, plan)
        _write_output(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        BundleError,
        QueueError,
        RecoveryError,
        OSError,
        ValueError,
    ) as error:
        print(f"r0 timing recovery error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
