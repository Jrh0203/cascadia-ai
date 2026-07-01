#!/usr/bin/env python3
"""Build and optionally install the distributed R4 foundation graph."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from cluster_research_queue import add_task, locked_queue
from rust_experiment_bundle import BundleError, validate_bundle

EXPERIMENT_ID = "r4-adaptive-multires-foundation-v1"
TASK_PREFIX = "r4am"
HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
BINARY_NAME = "r4-adaptive-multires-census"


class CampaignError(RuntimeError):
    """Raised when the R4 graph cannot be frozen without ambiguity."""


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: float,
    critical_path: bool,
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
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 1.0,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": critical_path,
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


def _relative_bundle(repository: Path, bundle: Path) -> Path:
    try:
        return bundle.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise CampaignError("bundle must remain beneath the repository") from error


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _binary_command(
    host: str,
    bundle_relative: Path,
    *,
    environment: list[str] | None = None,
) -> list[str]:
    return [
        "/usr/bin/env",
        "-C",
        _remote(host, bundle_relative / "source"),
        *(environment or []),
        _remote(host, bundle_relative / "bin" / BINARY_NAME),
    ]


def build_task_specs(bundle_relative: Path) -> list[dict[str, Any]]:
    preflight_ids = [f"{TASK_PREFIX}-preflight-{host}" for host in HOSTS]
    specs: list[dict[str, Any]] = []
    for host in HOSTS:
        report = EXPERIMENT_ROOT / "reports" / f"preflight-{host}.json"
        specs.append(
            _task(
                task_id=f"{TASK_PREFIX}-preflight-{host}",
                title=f"Run exact R4 production preflight on {host}",
                decision=(
                    "Require the shipped extractor to retain every frozen "
                    "adversarial distinction"
                ),
                workload_class="shared-prerequisite",
                priority=0,
                expected_runtime_seconds=15,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[],
                command=[
                    *_binary_command(host, bundle_relative),
                    "adversarial",
                    "--output",
                    _remote(host, report),
                    "--require-pass",
                ],
                artifact_path=str(report),
                stop_rule=(
                    "The production seven-pair, two-radius matrix must pass "
                    "and be written."
                ),
                cpu_cores=1,
                memory_gib=0.5,
            )
        )

    census_ids: list[str] = []
    for shard_index, host in enumerate(HOSTS):
        task_id = f"{TASK_PREFIX}-census-{host}"
        census_ids.append(task_id)
        report = EXPERIMENT_ROOT / "shards" / f"shard-{shard_index}.json"
        train = Path(
            "artifacts/datasets/"
            f"r0-spatial-position-corpus-v1-source-frozen-train-part-{shard_index}"
        )
        validation = Path(
            "artifacts/datasets/"
            f"r0-spatial-position-corpus-v1-source-frozen-validation-part-{shard_index}"
        )
        rayon_threads = 6 if host == "john1" else 10
        specs.append(
            _task(
                task_id=task_id,
                title=f"Census unique R4 corpus part {shard_index} on {host}",
                decision=(
                    "Measure exact 61/91-cell adaptive representations on "
                    "nonoverlapping frozen rows"
                ),
                workload_class="divisible-evidence",
                priority=10,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=preflight_ids,
                command=[
                    *_binary_command(
                        host,
                        bundle_relative,
                        environment=[f"RAYON_NUM_THREADS={rayon_threads}"],
                    ),
                    "census",
                    "--dataset-root",
                    _remote(host, train),
                    "--dataset-root",
                    _remote(host, validation),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(len(HOSTS)),
                    "--require-frozen",
                    "--output",
                    _remote(host, report),
                ],
                artifact_path=str(report),
                stop_rule=(
                    "Process exactly the assigned train and validation identities "
                    "with no duplicate source row."
                ),
                cpu_cores=rayon_threads,
                memory_gib=3.0,
            )
        )

    collected_root = EXPERIMENT_ROOT / "collected"
    collection_report = EXPERIMENT_ROOT / "reports" / "collection.json"
    collect_command = [
        "/usr/bin/env",
        "-C",
        _remote("john1", bundle_relative / "source"),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
    for host in HOSTS:
        source = EXPERIMENT_ROOT / "reports" / f"preflight-{host}.json"
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, source)}",
                _remote(
                    "john1",
                    collected_root / f"preflight-{host}.json",
                ),
            ]
        )
    for shard_index, host in enumerate(HOSTS):
        source = EXPERIMENT_ROOT / "shards" / f"shard-{shard_index}.json"
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, source)}",
                _remote(
                    "john1",
                    collected_root / f"shard-{shard_index}.json",
                ),
            ]
        )
    collect_command.extend(["--output", _remote("john1", collection_report)])
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-collect",
            title="Collect all R4 preflights and unique shard reports",
            decision="Require checksum-bound evidence from every producing host",
            workload_class="shared-prerequisite",
            priority=20,
            expected_runtime_seconds=60,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=census_ids,
            command=collect_command,
            artifact_path=str(collection_report),
            stop_rule="Collect eight artifacts and require every source checksum to match.",
            cpu_cores=1,
            memory_gib=0.5,
        )
    )

    parity = EXPERIMENT_ROOT / "reports" / "adversarial-parity.json"
    parity_command = [
        *_binary_command("john1", bundle_relative),
        "verify-adversarial",
    ]
    for host in HOSTS:
        parity_command.extend(
            [
                "--report",
                _remote(
                    "john1",
                    collected_root / f"preflight-{host}.json",
                ),
            ]
        )
    parity_command.extend(
        ["--output", _remote("john1", parity), "--require-pass"]
    )
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-adversarial-parity",
            title="Prove cross-host adversarial parity",
            decision=(
                "Require the production representation to be scientifically "
                "identical on all Macs"
            ),
            workload_class="replica",
            priority=30,
            expected_runtime_seconds=10,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[f"{TASK_PREFIX}-collect"],
            command=parity_command,
            artifact_path=str(parity),
            stop_rule=(
                "All four scientific preflight hashes must be identical and passing."
            ),
            cpu_cores=1,
            memory_gib=0.5,
        )
    )

    forward = EXPERIMENT_ROOT / "aggregate-forward.json"
    reverse = EXPERIMENT_ROOT / "aggregate-reverse.json"
    order_proof = EXPERIMENT_ROOT / "order-proof.json"
    aggregate_command = [
        *_binary_command("john1", bundle_relative),
        "aggregate",
    ]
    for shard_index in range(len(HOSTS)):
        aggregate_command.extend(
            [
                "--report",
                _remote(
                    "john1",
                    collected_root / f"shard-{shard_index}.json",
                ),
            ]
        )
    aggregate_command.extend(
        [
            "--adversarial-report",
            _remote("john1", collected_root / "preflight-john1.json"),
            "--forward-output",
            _remote("john1", forward),
            "--reverse-output",
            _remote("john1", reverse),
            "--order-proof-output",
            _remote("john1", order_proof),
        ]
    )
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-aggregate",
            title="Aggregate and classify the R4 foundation",
            decision=(
                "Apply exactness, adversarial, compactness, corpus, and order gates"
            ),
            workload_class="shared-prerequisite",
            priority=40,
            expected_runtime_seconds=15,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[f"{TASK_PREFIX}-adversarial-parity"],
            command=aggregate_command,
            artifact_path=str(forward),
            stop_rule=(
                "Write byte-identical forward/reverse aggregates and one "
                "deterministic foundation classification."
            ),
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    return specs


def build_postprocess_recovery_specs(
    bundle_relative: Path,
    suffix: str,
) -> list[dict[str, Any]]:
    if not suffix.startswith("-") or not suffix[1:].replace("-", "").isalnum():
        raise CampaignError(
            "postprocess recovery suffix must start with '-' and contain letters, digits, or '-'"
        )
    base = build_task_specs(bundle_relative)
    selected = copy.deepcopy(base[-3:])
    id_map = {task["id"]: f"{task['id']}{suffix}" for task in selected}
    for task in selected:
        task["id"] = id_map[task["id"]]
        task["dependencies"] = [
            id_map.get(dependency, dependency)
            for dependency in task["dependencies"]
        ]
        task["title"] = f"{task['title']} recovery"
    return selected


def install_specs(queue: Path, specs: list[dict[str, Any]]) -> None:
    with locked_queue(queue) as state:
        existing = {task["id"] for task in state["tasks"]}
        requested = [spec["id"] for spec in specs]
        duplicates = sorted(existing.intersection(requested))
        if duplicates:
            raise CampaignError(f"queue already contains R4 task IDs: {duplicates}")
        if len(requested) != len(set(requested)):
            raise CampaignError("generated R4 task IDs are not unique")
        for spec in specs:
            add_task(state, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--postprocess-recovery-suffix")
    args = parser.parse_args(argv)

    try:
        repository = args.repository.resolve()
        bundle_relative = _relative_bundle(repository, args.bundle)
        manifest = validate_bundle(repository / bundle_relative)
        binaries = {
            entry["name"] for entry in manifest["identity"].get("binaries", [])
        }
        if BINARY_NAME not in binaries:
            raise CampaignError(f"bundle lacks required binary {BINARY_NAME}")
        specs = (
            build_postprocess_recovery_specs(
                bundle_relative,
                args.postprocess_recovery_suffix,
            )
            if args.postprocess_recovery_suffix
            else build_task_specs(bundle_relative)
        )
        payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "bundle_id": manifest["bundle_id"],
            "bundle": str(bundle_relative),
            "task_count": len(specs),
            "tasks": specs,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(encoded)
            temporary.replace(args.output)
        if args.apply:
            install_specs(args.queue, specs)
        print(encoded, end="")
        return 0
    except (BundleError, CampaignError, OSError) as error:
        print(f"R4 campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
