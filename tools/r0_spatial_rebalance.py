#!/usr/bin/env python3
"""Atomically move unstarted R0 collection parts to available cluster hosts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import r0_spatial_campaign as campaign
from cluster_research_queue import (
    QueueError,
    add_task,
    cancel_pending_tasks,
    locked_queue,
    set_task_dependencies,
)
from rust_experiment_bundle import BundleError, validate_bundle

DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_OUTPUT = campaign.DEFAULT_EXPERIMENT_ROOT / "queue-rebalance-source-frozen-v1.json"
PART_INDEX = 1
REPLACEMENT_PREFIX = "r0f-rebalanced"
ACTOR = "research-coordinator"
REASON = (
    "Rebalanced unstarted source-frozen collection away from john2 so the "
    "three free Macs can open the R0 benchmark wave while john2 completes "
    "the authorized MLX dropout origin."
)


class RebalanceError(RuntimeError):
    """Raised when the R0 queue cannot be rebalanced without ambiguity."""


def _replacement_ids(split: str) -> tuple[str, str]:
    return (
        f"{REPLACEMENT_PREFIX}-collect-{split}-part-{PART_INDEX}",
        f"{REPLACEMENT_PREFIX}-fanout-{split}-part-{PART_INDEX}",
    )


def build_rebalance(
    *,
    bundle_relative: Path,
    train_host: str,
    validation_host: str,
) -> dict[str, Any]:
    if train_host not in campaign.HOSTS or validation_host not in campaign.HOSTS:
        raise RebalanceError("replacement hosts must be registered cluster hosts")
    if train_host == validation_host:
        raise RebalanceError("train and validation replacements must run in parallel")
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise RebalanceError("bundle path must be repository-relative")

    base_parts = {
        part.split: part for part in campaign.dataset_parts() if part.part_index == PART_INDEX
    }
    replacements: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for split, host in (
        ("train", train_host),
        ("validation", validation_host),
    ):
        part = replace(base_parts[split], host=host)
        collection_id, fanout_id = _replacement_ids(split)
        replacements.extend(
            campaign.build_dataset_part_task_specs(
                part=part,
                bundle_relative=bundle_relative,
                collection_task_id=collection_id,
                fanout_task_id=fanout_id,
                fanout_report_name=(
                    f"source-frozen-rebalanced-dataset-{split}-part-{PART_INDEX}-fanout.json"
                ),
            )
        )
        mapping[f"r0f-fanout-{split}-part-{PART_INDEX}"] = fanout_id

    originals = [f"r0f-collect-{split}-part-{PART_INDEX}" for split in ("train", "validation")] + [
        f"r0f-fanout-{split}-part-{PART_INDEX}" for split in ("train", "validation")
    ]
    benchmark_ids = [
        f"r0f-benchmark-shard-{shard_index}-replicate-{replicate_index}"
        for shard_index in range(campaign.SHARD_COUNT)
        for replicate_index in range(campaign.REQUIRED_REPLICATES)
    ]
    return {
        "schema_version": 1,
        "experiment_id": campaign.EXPERIMENT_ID,
        "reason": REASON,
        "cancel_task_ids": originals,
        "replacement_tasks": replacements,
        "dependency_mapping": mapping,
        "benchmark_task_ids": benchmark_ids,
    }


def apply_rebalance(state: dict[str, Any], plan: dict[str, Any]) -> None:
    replacement_ids = [task["id"] for task in plan["replacement_tasks"]]
    existing_ids = {task["id"] for task in state["tasks"]}
    duplicates = sorted(existing_ids.intersection(replacement_ids))
    if duplicates:
        raise RebalanceError(f"replacement task ids already exist: {duplicates}")

    benchmark_tasks = {
        task["id"]: task for task in state["tasks"] if task["id"] in plan["benchmark_task_ids"]
    }
    missing_benchmarks = sorted(set(plan["benchmark_task_ids"]) - set(benchmark_tasks))
    if missing_benchmarks:
        raise RebalanceError(f"missing benchmark tasks: {missing_benchmarks}")
    for task in benchmark_tasks.values():
        if task["status"] not in {"ready", "blocked"}:
            raise RebalanceError(
                f"benchmark task {task['id']} already started with status {task['status']}"
            )

    cancel_pending_tasks(
        state,
        task_ids=list(plan["cancel_task_ids"]),
        actor=ACTOR,
        reason=str(plan["reason"]),
    )
    for specification in plan["replacement_tasks"]:
        add_task(state, specification)
    for task_id in plan["benchmark_task_ids"]:
        dependencies = [
            plan["dependency_mapping"].get(dependency, dependency)
            for dependency in benchmark_tasks[task_id]["dependencies"]
        ]
        set_task_dependencies(
            state,
            task_id=task_id,
            dependencies=dependencies,
        )


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
    parser.add_argument("--train-host", choices=campaign.HOSTS, default="john3")
    parser.add_argument(
        "--validation-host",
        choices=campaign.HOSTS,
        default="john4",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    try:
        bundle_relative = args.bundle.resolve().relative_to(repository)
        manifest = validate_bundle(repository / bundle_relative)
        campaign.validate_provenance_source_bundle(manifest)
        plan = build_rebalance(
            bundle_relative=bundle_relative,
            train_host=args.train_host,
            validation_host=args.validation_host,
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
        QueueError,
        RebalanceError,
        OSError,
        ValueError,
    ) as error:
        print(f"r0 rebalance error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
