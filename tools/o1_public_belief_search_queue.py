#!/usr/bin/env python3
"""Build the immutable four-host ADR 0189 public-belief search campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "o1-public-belief-one-rotation-search-v1"
TASK_PREFIX = "o1pbs-v2"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec-v2.json"
DATASET = Path("artifacts/datasets/complete-action-graded-oracle-v1-validation")
COHORT = Path(
    "artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/"
    "cohort/3856f9c4cf73d34c470357cdf220dbf8314a6ddd2a6340ee686a5e2e16254591"
)
INTENT = Path(
    "artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/"
    "intent/b0de970601ddabcc7b3430397b07203df36656f810a53943337c450b2f3152f4"
)
MODEL = Path("artifacts/models/legacy-nnue-v4opp-mlx-v1")
PANEL = CAMPAIGN_ROOT / "control/high-regret-panel.json"
AUTHORIZATION_PACKAGE = CAMPAIGN_ROOT / "control/authorization-package-v2"
AUTHORIZATION = AUTHORIZATION_PACKAGE / "authorization.json"
RUNS_ROOT = CAMPAIGN_ROOT / "runs-v2"
COLLECTED_ROOT = CAMPAIGN_ROOT / "collected-v2"
PREFLIGHT_ROOT = CAMPAIGN_ROOT / "control/preflights-v2"
COLLECTION_REPORT = CAMPAIGN_ROOT / "control/collection-v2.json"
AGGREGATE = CAMPAIGN_ROOT / "aggregate-v2.json"
BINARY = Path(
    "tools/o1_public_belief_search/target/release/o1-public-belief-search"
)
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
PRIMARY_HOSTS = {
    "c0-primary": "john1",
    "a0-primary": "john2",
    "a2-primary": "john3",
    "s3-primary": "john4",
}
REPLAY_HOSTS = {
    "c0-replay": "john2",
    "a0-replay": "john3",
    "a2-replay": "john4",
    "s3-replay": "john1",
}
SOURCE_INCLUDES = (
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("crates/cascadia-data"),
    Path("crates/cascadia-differential"),
    Path("crates/cascadia-eval"),
    Path("crates/cascadia-game"),
    Path("crates/cascadia-model"),
    Path("crates/cascadia-provenance"),
    Path("crates/cascadia-search"),
    Path("crates/cascadia-sim"),
    Path("legacy/crates/cascadia-ai"),
    Path("legacy/crates/cascadia-core"),
    Path("python/cascadia_mlx"),
    Path("python/tests/test_o1_public_belief_search.py"),
    Path("tools/o1_public_belief_search"),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/o1_public_belief_search_queue.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/test_o1_public_belief_search_queue.py"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0189-o1-public-belief-one-rotation-search.md"),
    Path("docs/v2/decisions/0190-frozen-root-prelude-contingency-boundary.md"),
    Path(
        "docs/v2/reports/"
        "o1-public-belief-one-rotation-search-v1-preregistration.md"
    ),
    Path(
        "docs/v2/reports/"
        "o1-public-belief-one-rotation-search-v1-invalid-launch-1.md"
    ),
    PANEL,
)


class PublicBeliefSearchQueueError(RuntimeError):
    """The ADR 0189 bundle or campaign graph is invalid."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _local(relative: Path) -> str:
    return str(REMOTE_ROOTS["john1"] / relative)


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _source(host: str, bundle_relative: Path) -> Path:
    return REMOTE_ROOTS[host] / bundle_relative / "source"


def _bundled_panel(host: str, bundle_relative: Path) -> str:
    return str(_source(host, bundle_relative) / PANEL)


def _bundled_binary(host: str, bundle_relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / bundle_relative / "bin" / BINARY.name)


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


def _binary_prefix(host: str, bundle_relative: Path) -> list[str]:
    source = _source(host, bundle_relative)
    return [
        "/usr/bin/env",
        "-C",
        str(source),
        f"PYTHONPATH={source / 'python'}",
        "PYTHONDONTWRITEBYTECODE=1",
        "RAYON_NUM_THREADS=10",
        _bundled_binary(host, bundle_relative),
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
        "decision_value": 1.0,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": workload_class in {"independent-experiment", "replica"},
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


def _fanout_task(
    *,
    task_id: str,
    title: str,
    source: Path,
    bundle_relative: Path,
    dependencies: list[str],
    required_file: str,
) -> dict[str, Any]:
    report = CAMPAIGN_ROOT / "control" / f"{task_id}.json"
    command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{_local(source)}/",
        "--local-root",
        _local(source),
    ]
    for host in ("john2", "john3", "john4"):
        command.extend(
            [
                "--destination",
                f"{host}:{_remote(host, source)}/",
            ]
        )
    command.extend(
        [
            "--required-file",
            required_file,
            "--verify-tree",
            "--output",
            _local(report),
        ]
    )
    return _task(
        task_id=task_id,
        title=title,
        decision="Bind every host to byte-identical immutable campaign inputs",
        priority=60 if not dependencies else 61,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=dependencies,
        command=command,
        artifact_path=report,
        stop_rule="Every regular file and checksum must match on all four hosts.",
        expected_runtime_seconds=300,
        workload_class="shared-prerequisite",
        cpu_cores=1,
        memory_gib=1.0,
        uses_mlx=False,
    )


def _run_task(
    *,
    task_id: str,
    role: str,
    host: str,
    bundle_relative: Path,
    bundle_id: str,
    dependencies: list[str],
    priority: int,
    workload_class: str,
) -> dict[str, Any]:
    run_root = RUNS_ROOT / role
    return _task(
        task_id=task_id,
        title=f"Run ADR 0189 role {role}",
        decision=(
            "Measure one matched opponent-belief policy under the exact "
            "one-rotation 640-trajectory search"
        ),
        priority=priority,
        decision_terminal=False,
        compatible_hosts=[host],
        dependencies=dependencies,
        command=[
            *_binary_prefix(host, bundle_relative),
            "--dataset-root",
            _remote(host, DATASET),
            "--cohort-root",
            _remote(host, COHORT),
            "--intent-root",
            _remote(host, INTENT),
            "--panel",
            _bundled_panel(host, bundle_relative),
            "--model-dir",
            _remote(host, MODEL),
            "--python",
            str(REMOTE_ROOTS[host] / ".venv/bin/python"),
            "--authorization",
            _remote(host, AUTHORIZATION),
            "--bundle-id",
            bundle_id,
            "--role",
            role,
            "--host",
            host,
            "--run-dir",
            _remote(host, run_root / "state"),
            "--output",
            _remote(host, run_root / "report.json"),
        ],
        artifact_path=run_root / "report.json",
        stop_rule=(
            "Run all 99 frozen groups with 64 roots and exactly 640 trajectories "
            "per group; keep sealed test and gameplay closed."
        ),
        expected_runtime_seconds=300,
        workload_class=workload_class,
        cpu_cores=10,
        memory_gib=8.0,
        uses_mlx=True,
    )


def task_specs(
    *,
    bundle_relative: Path,
    bundle_id: str,
) -> list[dict[str, Any]]:
    if len(bundle_id) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_id
    ):
        raise PublicBeliefSearchQueueError("bundle ID is not a lowercase digest")

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    tasks = [
        _fanout_task(
            task_id=bundle_fanout_id,
            title="Fan out immutable ADR 0189 bundle",
            source=bundle_relative,
            bundle_relative=bundle_relative,
            dependencies=[],
            required_file="bundle.json",
        )
    ]
    input_fanouts = (
        ("dataset", "Fan out graded-oracle validation data", DATASET, "dataset.json"),
        ("cohort", "Fan out frozen exact-R2 cohort", COHORT, "cache.json"),
        ("intent", "Fan out frozen O1 intent cache", INTENT, "cache.json"),
        ("model", "Fan out qualified MLX leaf model", MODEL, "model.json"),
    )
    input_fanout_ids = []
    for name, title, source, required_file in input_fanouts:
        task_id = f"{TASK_PREFIX}-{name}-fanout"
        input_fanout_ids.append(task_id)
        tasks.append(
            _fanout_task(
                task_id=task_id,
                title=title,
                source=source,
                bundle_relative=bundle_relative,
                dependencies=[bundle_fanout_id],
                required_file=required_file,
            )
        )

    authorize_id = f"{TASK_PREFIX}-authorize"
    tasks.append(
        _task(
            task_id=authorize_id,
            title="Authorize the amended O1 public-belief search",
            decision=(
                "Freeze exact bundle, panel, inputs, protocol, role map, "
                "representation boundary, and claim boundary"
            ),
            priority=62,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[bundle_fanout_id, *input_fanout_ids],
            command=[
                *_python_prefix("john1", bundle_relative),
                "-m",
                "cascadia_mlx.o1_public_belief_search",
                "authorize",
                "--bundle-id",
                bundle_id,
                "--dataset-root",
                _local(DATASET),
                "--cohort-root",
                _local(COHORT),
                "--intent-root",
                _local(INTENT),
                "--panel",
                _bundled_panel("john1", bundle_relative),
                "--model-dir",
                _local(MODEL),
                "--output",
                _local(AUTHORIZATION),
            ],
            artifact_path=AUTHORIZATION,
            stop_rule="No production search may begin before this exact authorization exists.",
            expected_runtime_seconds=30,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    authorization_fanout_id = f"{TASK_PREFIX}-authorization-fanout"
    tasks.append(
        _fanout_task(
            task_id=authorization_fanout_id,
            title="Fan out frozen ADR 0189/0190 authorization",
            source=AUTHORIZATION_PACKAGE,
            bundle_relative=bundle_relative,
            dependencies=[authorize_id],
            required_file="authorization.json",
        )
    )

    preflight_ids = {}
    primary_role_by_host = {
        host: role for role, host in PRIMARY_HOSTS.items()
    }
    for host in REMOTE_ROOTS:
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids[host] = task_id
        report = PREFLIGHT_ROOT / f"{host}.json"
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Verify amended ADR 0189/0190 inputs on {host}",
                decision=(
                    "Independently rebuild the authorization from local bytes "
                    "before any trajectory is generated"
                ),
                priority=63,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[authorization_fanout_id],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.o1_public_belief_search",
                    "verify-authorization",
                    "--authorization",
                    _remote(host, AUTHORIZATION),
                    "--role",
                    primary_role_by_host[host],
                    "--bundle-id",
                    bundle_id,
                    "--dataset-root",
                    _remote(host, DATASET),
                    "--cohort-root",
                    _remote(host, COHORT),
                    "--intent-root",
                    _remote(host, INTENT),
                    "--panel",
                    _bundled_panel(host, bundle_relative),
                    "--model-dir",
                    _remote(host, MODEL),
                    "--output",
                    _remote(host, report),
                ],
                artifact_path=report,
                stop_rule=(
                    "Reject byte drift; do not create a run directory or search result."
                ),
                expected_runtime_seconds=30,
                workload_class="shared-prerequisite",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )

    primary_ids = []
    for role, host in PRIMARY_HOSTS.items():
        task_id = f"{TASK_PREFIX}-run-{role}"
        primary_ids.append(task_id)
        tasks.append(
            _run_task(
                task_id=task_id,
                role=role,
                host=host,
                bundle_relative=bundle_relative,
                bundle_id=bundle_id,
                dependencies=[preflight_ids[host]],
                priority=70,
                workload_class="independent-experiment",
            )
        )

    replay_ids = []
    for role, host in REPLAY_HOSTS.items():
        task_id = f"{TASK_PREFIX}-run-{role}"
        replay_ids.append(task_id)
        primary_role = f"{role.removesuffix('-replay')}-primary"
        tasks.append(
            _run_task(
                task_id=task_id,
                role=role,
                host=host,
                bundle_relative=bundle_relative,
                bundle_id=bundle_id,
                dependencies=[
                    f"{TASK_PREFIX}-run-{primary_role}",
                    preflight_ids[host],
                ],
                priority=71,
                workload_class="replica",
            )
        )

    role_hosts = {**PRIMARY_HOSTS, **REPLAY_HOSTS}
    collection_id = f"{TASK_PREFIX}-collect"
    collection_report = COLLECTION_REPORT
    collect_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for role, host in role_hosts.items():
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, RUNS_ROOT / role / 'report.json')}",
                _local(COLLECTED_ROOT / "reports" / f"{role}.json"),
            ]
        )
    collect_command.extend(["--output", _local(collection_report)])
    tasks.append(
        _task(
            task_id=collection_id,
            title="Collect all eight public-belief search reports",
            decision="Bind every primary and replay report by checksum",
            priority=72,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[*primary_ids, *replay_ids],
            command=collect_command,
            artifact_path=collection_report,
            stop_rule="Collect all eight reports without substitution or repair.",
            expected_runtime_seconds=120,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    aggregate = AGGREGATE
    aggregate_command = [
        *_python_prefix("john1", bundle_relative),
        "-m",
        "cascadia_mlx.o1_public_belief_search",
        "aggregate",
    ]
    for role in role_hosts:
        aggregate_command.extend(
            [
                "--report",
                f"{role}={_local(COLLECTED_ROOT / 'reports' / f'{role}.json')}",
            ]
        )
    aggregate_command.extend(["--output", _local(aggregate)])
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-aggregate",
            title="Classify the O1 public-belief search",
            decision=(
                "Promote only exact-replayed A2 search that clears every "
                "effect-size, uncertainty, recall, and pairwise gate"
            ),
            priority=73,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collection_id],
            command=aggregate_command,
            artifact_path=aggregate,
            stop_rule=(
                "Apply ADR 0189 exactly; do not tune thresholds or open sealed test."
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
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-j",
                "1",
                "--manifest-path",
                "tools/o1_public_belief_search/Cargo.toml",
            ],
            cwd=args.repository,
            check=True,
        )
        path, manifest, reused = build_bundle(
            repository=args.repository,
            experiment_id=EXPERIMENT_ID,
            includes=list(SOURCE_INCLUDES),
            binaries=[BINARY],
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
        raise PublicBeliefSearchQueueError("bundle belongs to another experiment")
    try:
        bundle_relative = args.bundle.resolve().relative_to(args.repository.resolve())
    except ValueError as error:
        raise PublicBeliefSearchQueueError(
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
