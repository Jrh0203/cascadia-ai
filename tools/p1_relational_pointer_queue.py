#!/usr/bin/env python3
"""Build the immutable bundle and crossed queue graph for ADR 0174."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "p1-relational-hierarchical-pointer-foundation-v1"
TASK_PREFIX = "p1ptr-v1"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
TASK_PREFIX_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
SOURCE_INCLUDES = (
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0174-relational-hierarchical-pointer-foundation.md"),
    Path(
        "docs/v2/reports/"
        "p1-relational-hierarchical-pointer-foundation-v1-preregistration.md"
    ),
    Path("python/cascadia_mlx/d6_contract_metadata.v1.json"),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/p1_relational_pointer_foundation.py"),
    Path("tools/p1_relational_pointer_queue.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/test_p1_relational_pointer_foundation.py"),
    Path("tools/test_p1_relational_pointer_queue.py"),
)
FACTOR_CACHE = Path(
    "artifacts/experiments/"
    "full-legal-hierarchical-factor-retrieval-pilot-v1/cache"
)
R3_CACHE = Path(
    "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/"
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}


class PointerQueueError(RuntimeError):
    """The ADR 0174 bundle or queue graph is invalid."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: int,
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: Path,
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
        "critical_path": True,
        "decision_terminal": decision_terminal,
        "compatible_hosts": compatible_hosts,
        "dependencies": dependencies,
        "command": command,
        "artifact_path": str(artifact_path),
        "stop_rule": stop_rule,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gib": memory_gib,
            "uses_mlx": False,
        },
    }


def _python_prefix(host: str, bundle_relative: Path) -> list[str]:
    root = REMOTE_ROOTS[host]
    source = root / bundle_relative / "source"
    return [
        "/usr/bin/env",
        "-C",
        str(source),
        "PYTHONDONTWRITEBYTECODE=1",
        str(root / ".venv/bin/python"),
        "-B",
    ]


def _audit_command(
    host: str,
    *,
    split: str,
    output: Path,
    bundle_relative: Path,
    bundle_id: str,
) -> list[str]:
    root = REMOTE_ROOTS[host]
    return [
        *_python_prefix(host, bundle_relative),
        "tools/p1_relational_pointer_foundation.py",
        "audit",
        "--split",
        split,
        "--factor-cache",
        str(root / FACTOR_CACHE),
        "--r3-cache",
        str(root / R3_CACHE),
        "--d6-metadata",
        str(
            root
            / bundle_relative
            / "source/python/cascadia_mlx/d6_contract_metadata.v1.json"
        ),
        "--bundle-id",
        bundle_id,
        "--host",
        host,
        "--output",
        str(root / output),
    ]


def task_specs(
    *,
    bundle_relative: Path,
    bundle_id: str,
    task_prefix: str = TASK_PREFIX,
) -> list[dict[str, Any]]:
    """Return the complete crossed audit, collection, and classification graph."""
    if not TASK_PREFIX_PATTERN.fullmatch(task_prefix):
        raise PointerQueueError("task prefix contains unsupported characters")
    if len(bundle_id) != 64:
        raise PointerQueueError("bundle ID must contain 64 characters")

    train_origin_id = f"{task_prefix}-train-john2"
    validation_origin_id = f"{task_prefix}-validation-john4"
    train_replay_id = f"{task_prefix}-train-replay-john4"
    validation_replay_id = f"{task_prefix}-validation-replay-john2"
    reports = {
        train_origin_id: CAMPAIGN_ROOT / "reports/origin-train-john2.json",
        validation_origin_id: CAMPAIGN_ROOT / "reports/origin-validation-john4.json",
        train_replay_id: CAMPAIGN_ROOT / "reports/replay-train-john4.json",
        validation_replay_id: CAMPAIGN_ROOT / "reports/replay-validation-john2.json",
    }
    origins = [train_origin_id, validation_origin_id]
    tasks = [
        _task(
            task_id=train_origin_id,
            title="Audit P1 train pointers on john2",
            decision="Prove complete train selected-prefix pointer semantics",
            workload_class="divisible-evidence",
            priority=13,
            expected_runtime_seconds=180,
            decision_terminal=False,
            compatible_hosts=["john2"],
            dependencies=["r2vec2-validation-john2-vectorized-first"],
            command=_audit_command(
                "john2",
                split="train",
                output=reports[train_origin_id],
                bundle_relative=bundle_relative,
                bundle_id=bundle_id,
            ),
            artifact_path=reports[train_origin_id],
            stop_rule="Audit the complete frozen train split once; no sampling or repair.",
            cpu_cores=10,
            memory_gib=3.0,
        ),
        _task(
            task_id=validation_origin_id,
            title="Audit P1 validation pointers on john4",
            decision="Prove complete validation selected-prefix pointer semantics",
            workload_class="divisible-evidence",
            priority=13,
            expected_runtime_seconds=120,
            decision_terminal=False,
            compatible_hosts=["john4"],
            dependencies=["r2vec2-train-john4"],
            command=_audit_command(
                "john4",
                split="validation",
                output=reports[validation_origin_id],
                bundle_relative=bundle_relative,
                bundle_id=bundle_id,
            ),
            artifact_path=reports[validation_origin_id],
            stop_rule="Audit the complete frozen validation split once; no sampling or repair.",
            cpu_cores=10,
            memory_gib=3.0,
        ),
        _task(
            task_id=train_replay_id,
            title="Replay P1 train pointers on john4",
            decision="Require byte-identical train science on a distinct host",
            workload_class="replica",
            priority=14,
            expected_runtime_seconds=180,
            decision_terminal=False,
            compatible_hosts=["john4"],
            dependencies=origins,
            command=_audit_command(
                "john4",
                split="train",
                output=reports[train_replay_id],
                bundle_relative=bundle_relative,
                bundle_id=bundle_id,
            ),
            artifact_path=reports[train_replay_id],
            stop_rule="Replay the unchanged complete train audit on john4.",
            cpu_cores=10,
            memory_gib=3.0,
        ),
        _task(
            task_id=validation_replay_id,
            title="Replay P1 validation pointers on john2",
            decision="Require byte-identical validation science on a distinct host",
            workload_class="replica",
            priority=14,
            expected_runtime_seconds=120,
            decision_terminal=False,
            compatible_hosts=["john2"],
            dependencies=origins,
            command=_audit_command(
                "john2",
                split="validation",
                output=reports[validation_replay_id],
                bundle_relative=bundle_relative,
                bundle_id=bundle_id,
            ),
            artifact_path=reports[validation_replay_id],
            stop_rule="Replay the unchanged complete validation audit on john2.",
            cpu_cores=10,
            memory_gib=3.0,
        ),
    ]

    collected = {
        task_id: CAMPAIGN_ROOT / "collected" / path.name
        for task_id, path in reports.items()
    }
    collection_path = CAMPAIGN_ROOT / "control/collection.json"
    collection_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for task_id, host in (
        (train_origin_id, "john2"),
        (validation_origin_id, "john4"),
        (train_replay_id, "john4"),
        (validation_replay_id, "john2"),
    ):
        collection_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, reports[task_id])}",
                _remote("john1", collected[task_id]),
            ]
        )
    collection_command.extend(["--output", _remote("john1", collection_path)])
    collect_id = f"{task_prefix}-collect"
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect crossed P1 pointer reports",
            decision="Bind all four reports to coordinator checksums",
            workload_class="shared-prerequisite",
            priority=27,
            expected_runtime_seconds=60,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[
                train_origin_id,
                validation_origin_id,
                train_replay_id,
                validation_replay_id,
            ],
            command=collection_command,
            artifact_path=collection_path,
            stop_rule="Every remote report must match its coordinator copy.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    classification_path = CAMPAIGN_ROOT / "classification.json"
    classify_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/p1_relational_pointer_foundation.py",
        "classify",
    ]
    for task_id in (
        train_origin_id,
        train_replay_id,
        validation_origin_id,
        validation_replay_id,
    ):
        classify_command.extend(
            ["--report", _remote("john1", collected[task_id])]
        )
    classify_command.extend(["--output", _remote("john1", classification_path)])
    tasks.append(
        _task(
            task_id=f"{task_prefix}-classify",
            title="Classify the P1 pointer foundation",
            decision="Authorize or block the matched MLX pointer pilot",
            workload_class="shared-prerequisite",
            priority=28,
            expected_runtime_seconds=30,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collect_id],
            command=classify_command,
            artifact_path=classification_path,
            stop_rule="Apply ADR 0174 mechanically; do not reinterpret a failed gate.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    return tasks


def build_task_specs(
    repository: Path,
    bundle: Path,
    *,
    task_prefix: str = TASK_PREFIX,
) -> list[dict[str, Any]]:
    manifest = validate_bundle(bundle)
    if manifest["identity"].get("experiment_id") != EXPERIMENT_ID:
        raise PointerQueueError("bundle belongs to another experiment")
    try:
        relative = bundle.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise PointerQueueError("bundle must remain beneath the repository") from error
    return task_specs(
        bundle_relative=relative,
        bundle_id=manifest["bundle_id"],
        task_prefix=task_prefix,
    )


def campaign_spec(tasks: list[dict[str, Any]], *, bundle_id: str) -> dict[str, Any]:
    """Wrap reviewed tasks in the queue installer's atomic campaign envelope."""
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "bundle_id": bundle_id,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("build-bundle")
    bundle.add_argument("--repository", type=Path, default=Path.cwd())
    bundle.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE_ROOT)

    specification = subparsers.add_parser("build-spec")
    specification.add_argument("--repository", type=Path, default=Path.cwd())
    specification.add_argument("--bundle", type=Path, required=True)
    specification.add_argument("--task-prefix", default=TASK_PREFIX)
    specification.add_argument("--output", type=Path, default=DEFAULT_SPEC)

    args = parser.parse_args()
    if args.command == "build-bundle":
        path, manifest, reused = build_bundle(
            repository=args.repository,
            experiment_id=EXPERIMENT_ID,
            includes=list(SOURCE_INCLUDES),
            binaries=[],
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "bundle": str(path),
                    "bundle_id": manifest["bundle_id"],
                    "reused": reused,
                },
                sort_keys=True,
            )
        )
        return 0

    tasks = build_task_specs(
        args.repository,
        args.bundle,
        task_prefix=args.task_prefix,
    )
    manifest = validate_bundle(args.bundle)
    _write_json(
        args.output,
        campaign_spec(tasks, bundle_id=manifest["bundle_id"]),
    )
    print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
