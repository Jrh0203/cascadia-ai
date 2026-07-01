#!/usr/bin/env python3
"""Build and optionally install the distributed R4 bounded-quotient graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cluster_research_queue import add_task, locked_queue
from rust_experiment_bundle import BundleError, validate_bundle

EXPERIMENT_ID = "r4-bounded-far-quotient-foundation-v1"
TASK_PREFIX = "r4bq"
HOSTS = ("john1", "john2", "john3", "john4")
HOST_ARMS = {
    "john1": "q1-seat-marginal",
    "john2": "q2-directional",
    "john3": "q3-affordance",
    "john4": "q4-selective-exact",
}
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
RAYON_THREADS = {
    "john1": 6,
    "john2": 10,
    "john3": 10,
    "john4": 10,
}
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
BINARY_NAME = "r4-bounded-far-quotient-census"


class CampaignError(RuntimeError):
    """Raised when the bounded-quotient graph cannot be frozen exactly."""


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


def _dataset_roots(host: str) -> list[str]:
    roots = [
        Path(
            "artifacts/datasets/"
            f"r0-spatial-position-corpus-v1-source-frozen-train-part-{part}"
        )
        for part in range(4)
    ]
    roots.extend(
        Path(
            "artifacts/datasets/"
            f"r0-spatial-position-corpus-v1-source-frozen-validation-part-{part}"
        )
        for part in range(4)
    )
    command: list[str] = []
    for root in roots:
        command.extend(["--dataset-root", _remote(host, root)])
    return command


def _collector_command(bundle_relative: Path) -> list[str]:
    return [
        "/usr/bin/env",
        "-C",
        _remote("john1", bundle_relative / "source"),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/cluster_artifact_collect.py",
    ]


def build_task_specs(bundle_relative: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    preflight_ids: list[str] = []
    for host in HOSTS:
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids.append(task_id)
        report = EXPERIMENT_ROOT / "reports" / f"preflight-{host}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Run all bounded-quotient proofs on {host}",
                decision=(
                    "Require every arm to retain the seven registered "
                    "long-range distinctions with exact accounting"
                ),
                workload_class="replica",
                priority=0,
                expected_runtime_seconds=20,
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
                    "Every mechanical, malformed-envelope, target, and D6 "
                    "check must pass, with at least one arm retaining all "
                    "seven pairs; per-arm failures remain negative evidence."
                ),
                cpu_cores=1,
                memory_gib=0.75,
            )
        )

    collected_root = EXPERIMENT_ROOT / "collected"
    preflight_collection = EXPERIMENT_ROOT / "reports" / "preflight-collection.json"
    preflight_collect_command = _collector_command(bundle_relative)
    for host in HOSTS:
        source = EXPERIMENT_ROOT / "reports" / f"preflight-{host}.json"
        preflight_collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, source)}",
                _remote("john1", collected_root / f"preflight-{host}.json"),
            ]
        )
    preflight_collect_command.extend(
        ["--output", _remote("john1", preflight_collection)]
    )
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-collect-preflight",
            title="Collect bounded-quotient preflight proofs",
            decision="Require checksum-bound preflight evidence from every Mac",
            workload_class="shared-prerequisite",
            priority=10,
            expected_runtime_seconds=45,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=preflight_ids,
            command=preflight_collect_command,
            artifact_path=str(preflight_collection),
            stop_rule="Collect four preflight reports with matching source checksums.",
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
                _remote("john1", collected_root / f"preflight-{host}.json"),
            ]
        )
    parity_command.extend(
        ["--output", _remote("john1", parity), "--require-pass"]
    )
    parity_id = f"{TASK_PREFIX}-adversarial-parity"
    specs.append(
        _task(
            task_id=parity_id,
            title="Prove bounded-quotient cross-host parity",
            decision=(
                "Freeze one byte-identical scientific preflight before "
                "production corpus work begins"
            ),
            workload_class="shared-prerequisite",
            priority=20,
            expected_runtime_seconds=10,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[f"{TASK_PREFIX}-collect-preflight"],
            command=parity_command,
            artifact_path=str(parity),
            stop_rule=(
                "All four preflight scientific hashes must be identical and passing."
            ),
            cpu_cores=1,
            memory_gib=0.5,
        )
    )

    census_ids: list[str] = []
    for host in HOSTS:
        arm = HOST_ARMS[host]
        task_id = f"{TASK_PREFIX}-census-{host}"
        census_ids.append(task_id)
        report = EXPERIMENT_ROOT / "arms" / f"{arm}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Census {arm} over all 60,000 positions on {host}",
                decision=(
                    "Measure one distinct bounded far-field hypothesis over "
                    "the identical accepted corpus"
                ),
                workload_class="independent-experiment",
                priority=30,
                expected_runtime_seconds=240,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[parity_id],
                command=[
                    *_binary_command(
                        host,
                        bundle_relative,
                        environment=[
                            f"RAYON_NUM_THREADS={RAYON_THREADS[host]}"
                        ],
                    ),
                    "census",
                    *_dataset_roots(host),
                    "--arm",
                    arm,
                    "--require-frozen",
                    "--output",
                    _remote(host, report),
                ],
                artifact_path=str(report),
                stop_rule=(
                    "Process every accepted row exactly once for this arm and "
                    "write exactness, size, accounting, and paired timing evidence."
                ),
                cpu_cores=RAYON_THREADS[host],
                memory_gib=4.0,
            )
        )

    arm_collection = EXPERIMENT_ROOT / "reports" / "arm-collection.json"
    arm_collect_command = _collector_command(bundle_relative)
    for host in HOSTS:
        arm = HOST_ARMS[host]
        source = EXPERIMENT_ROOT / "arms" / f"{arm}.json"
        arm_collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, source)}",
                _remote("john1", collected_root / f"{arm}.json"),
            ]
        )
    arm_collect_command.extend(
        ["--output", _remote("john1", arm_collection)]
    )
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-collect-arms",
            title="Collect all four bounded-quotient arm reports",
            decision="Require checksum-bound evidence for each distinct hypothesis",
            workload_class="shared-prerequisite",
            priority=40,
            expected_runtime_seconds=45,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=census_ids,
            command=arm_collect_command,
            artifact_path=str(arm_collection),
            stop_rule="Collect one complete and unique 60,000-row report per arm.",
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
    for host in HOSTS:
        arm = HOST_ARMS[host]
        aggregate_command.extend(
            [
                "--report",
                _remote("john1", collected_root / f"{arm}.json"),
            ]
        )
    aggregate_command.extend(
        [
            "--adversarial-report",
            _remote("john1", collected_root / "preflight-john1.json"),
            "--adversarial-parity-report",
            _remote("john1", parity),
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
            title="Classify and select bounded-quotient successors",
            decision=(
                "Apply exactness, information, token, scalar, byte, runtime, "
                "corpus, parity, and order gates"
            ),
            workload_class="shared-prerequisite",
            priority=50,
            expected_runtime_seconds=15,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[f"{TASK_PREFIX}-collect-arms"],
            command=aggregate_command,
            artifact_path=str(forward),
            stop_rule=(
                "Write byte-identical forward/reverse aggregates, classify "
                "all arms, and name minimal and richest passing successors."
            ),
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    return specs


def install_specs(queue: Path, specs: list[dict[str, Any]]) -> None:
    with locked_queue(queue) as state:
        existing = {task["id"] for task in state["tasks"]}
        requested = [spec["id"] for spec in specs]
        duplicates = sorted(existing.intersection(requested))
        if duplicates:
            raise CampaignError(
                f"queue already contains bounded R4 task IDs: {duplicates}"
            )
        if len(requested) != len(set(requested)):
            raise CampaignError("generated bounded R4 task IDs are not unique")
        for spec in specs:
            add_task(state, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
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
        specs = build_task_specs(bundle_relative)
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
        print(f"R4 bounded campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
