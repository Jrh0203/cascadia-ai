#!/usr/bin/env python3
"""Build the immutable four-host queue graph for ADR 0176."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "v1-score-anatomy-matched-r2-v1"
TASK_PREFIX = "v1anatomy-v1"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
DEFAULT_AUTHORIZATION = CAMPAIGN_ROOT / "control/authorization.json"
DEFAULT_CACHE = Path(
    "artifacts/experiments/r2-sparse-mlx-architecture-tournament-v1/"
    "caches/c97ce6b2de1beb4cc7d2d5e31e2fbed9213b28d3bde8a8ab4bdcc90b2edd85f8"
)
DEFAULT_CORPUS_LOCK = Path(
    "artifacts/experiments/r2-sparse-mlx-architecture-tournament-v1/"
    "control/corpus-lock.json"
)
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
ROLE_HOSTS = {
    "scalar-primary": "john2",
    "anatomy-primary": "john3",
    "scalar-replay": "john4",
    "anatomy-replay": "john1",
}
SOURCE_INCLUDES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("python/cascadia_mlx"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0176-v1-score-anatomy-matched-r2.md"),
    Path(
        "docs/v2/reports/"
        "v1-score-anatomy-matched-r2-v1-preregistration.md"
    ),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/v1_score_anatomy_queue.py"),
)


class V1ScoreAnatomyQueueError(RuntimeError):
    """Raised when the ADR 0176 bundle or queue graph is invalid."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _source(host: str, bundle_relative: Path) -> Path:
    return REMOTE_ROOTS[host] / bundle_relative / "source"


def _python_prefix(host: str, bundle_relative: Path) -> list[str]:
    source = _source(host, bundle_relative)
    return [
        "/usr/bin/env",
        "-C",
        str(source),
        f"PYTHONPATH={source / 'python'}",
        "PYTHONDONTWRITEBYTECODE=1",
        str(REMOTE_ROOTS[host] / ".venv/bin/python"),
        "-B",
    ]


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    priority: int,
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: Path,
    stop_rule: str,
    expected_runtime_seconds: int,
    workload_class: str,
    cpu_cores: int,
    memory_gib: float,
    uses_mlx: bool,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 0.8,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": False,
        "decision_terminal": decision_terminal,
        "compatible_hosts": compatible_hosts,
        "dependencies": dependencies,
        "command": command,
        "artifact_path": str(artifact_path),
        "stop_rule": stop_rule,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gib": memory_gib,
            "uses_mlx": uses_mlx,
        },
    }


def task_specs(
    *,
    bundle_relative: Path,
    bundle_id: str,
) -> list[dict[str, Any]]:
    if len(bundle_id) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_id
    ):
        raise V1ScoreAnatomyQueueError("bundle ID is not a lowercase digest")
    tasks = []
    run_ids = []
    for role, host in ROLE_HOSTS.items():
        task_id = f"{TASK_PREFIX}-run-{role}"
        run_ids.append(task_id)
        run_root = CAMPAIGN_ROOT / "runs" / role
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Run V1 score-anatomy role {role}",
                decision=(
                    "Measure scalar-only versus component-anatomy supervision "
                    "under an identical exact sparse R2 parameter graph"
                ),
                priority=90,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.v1_score_anatomy_mlx",
                    "run",
                    "--cache",
                    _remote(host, DEFAULT_CACHE),
                    "--corpus-lock",
                    _remote(host, DEFAULT_CORPUS_LOCK),
                    "--authorization",
                    _remote(host, DEFAULT_AUTHORIZATION),
                    "--run-dir",
                    _remote(host, run_root / "training"),
                    "--output",
                    _remote(host, run_root / "report.json"),
                    "--bundle-id",
                    bundle_id,
                    "--role",
                    role,
                ],
                artifact_path=run_root / "report.json",
                stop_rule=(
                    "Complete exactly 3,000 frozen steps and one full open-"
                    "validation pass; do not inspect test or gameplay data."
                ),
                expected_runtime_seconds=900,
                workload_class=(
                    "replica" if role.endswith("replay") else "independent-experiment"
                ),
                cpu_cores=10,
                memory_gib=5.0,
                uses_mlx=True,
            )
        )

    collection = f"{TASK_PREFIX}-collect"
    collection_output = CAMPAIGN_ROOT / "control/collection.json"
    collect_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for role, host in ROLE_HOSTS.items():
        collect_command.extend(
            [
                "--artifact",
                (
                    f"{host}:"
                    f"{_remote(host, CAMPAIGN_ROOT / 'runs' / role / 'report.json')}"
                ),
                _remote(
                    "john1",
                    CAMPAIGN_ROOT / "reports" / f"{role}.json",
                ),
            ]
        )
    collect_command.extend(
        ["--output", _remote("john1", collection_output)]
    )
    tasks.append(
        _task(
            task_id=collection,
            title="Collect all V1 score-anatomy reports",
            decision="Bind primary and replay evidence by checksum on john1",
            priority=91,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=run_ids,
            command=collect_command,
            artifact_path=collection_output,
            stop_rule="Collect all four report bytes and verify every checksum.",
            expected_runtime_seconds=60,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    classification_path = CAMPAIGN_ROOT / "classification.json"
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-classify",
            title="Classify matched V1 score anatomy",
            decision=(
                "Require exact cross-host replay and apply the frozen "
                "calibration and rank-quality promotion gates"
            ),
            priority=92,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collection],
            command=[
                *_python_prefix("john1", bundle_relative),
                "-m",
                "cascadia_mlx.v1_score_anatomy_mlx",
                "classify",
                "--scalar-primary",
                _remote(
                    "john1",
                    CAMPAIGN_ROOT / "reports/scalar-primary.json",
                ),
                "--anatomy-primary",
                _remote(
                    "john1",
                    CAMPAIGN_ROOT / "reports/anatomy-primary.json",
                ),
                "--scalar-replay",
                _remote(
                    "john1",
                    CAMPAIGN_ROOT / "reports/scalar-replay.json",
                ),
                "--anatomy-replay",
                _remote(
                    "john1",
                    CAMPAIGN_ROOT / "reports/anatomy-replay.json",
                ),
                "--output",
                _remote("john1", classification_path),
            ],
            artifact_path=classification_path,
            stop_rule=(
                "Publish the preregistered verdict without validation-driven "
                "threshold changes or gameplay claims."
            ),
            expected_runtime_seconds=30,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    return tasks


def campaign_spec(
    tasks: list[dict[str, Any]],
    *,
    bundle_id: str,
) -> dict[str, Any]:
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

    manifest = validate_bundle(args.bundle)
    if manifest["identity"].get("experiment_id") != EXPERIMENT_ID:
        raise V1ScoreAnatomyQueueError("bundle belongs to another experiment")
    try:
        bundle_relative = args.bundle.resolve().relative_to(
            args.repository.resolve()
        )
    except ValueError as error:
        raise V1ScoreAnatomyQueueError(
            "bundle must remain beneath the repository"
        ) from error
    tasks = task_specs(
        bundle_relative=bundle_relative,
        bundle_id=manifest["bundle_id"],
    )
    _write_json(
        args.output,
        campaign_spec(tasks, bundle_id=manifest["bundle_id"]),
    )
    print(
        json.dumps(
            {"output": str(args.output), "tasks": len(tasks)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
