#!/usr/bin/env python3
"""Build the immutable four-host ADR 0179 campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "v2-distributional-opportunity-supervision-v1"
TASK_PREFIX = "v2dist-v2"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
TRAIN_DATASET = Path("artifacts/datasets/r12-counterfactual-advantage-v1-train-128")
VALIDATION_DATASET = Path("artifacts/datasets/r12-counterfactual-advantage-v1-validation-32")
AUTHORIZATION_PACKAGE = CAMPAIGN_ROOT / "control/authorization-package"
AUTHORIZATION = AUTHORIZATION_PACKAGE / "authorization.json"
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
PRIMARY_HOSTS = {
    "c0-primary": "john1",
    "g1-primary": "john2",
    "q2-primary": "john3",
    "e3-primary": "john4",
}
REPLAY_HOSTS = {
    "c0-replay": "john2",
    "g1-replay": "john3",
    "q2-replay": "john4",
    "e3-replay": "john1",
}
SOURCE_INCLUDES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("python/cascadia_mlx"),
    Path("python/tests/test_counterfactual_advantage_dataset.py"),
    Path("python/tests/test_distributional_opportunity_model.py"),
    Path("python/tests/test_distributional_opportunity_experiment.py"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0179-matched-r12-distributional-opportunity-supervision.md"),
    Path("docs/v2/decisions/0180-distributional-authorization-json-normalization.md"),
    Path("docs/v2/reports/v2-distributional-opportunity-supervision-v1-preregistration.md"),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/v2_distributional_opportunity_queue.py"),
    Path("tools/test_v2_distributional_opportunity_queue.py"),
)


class DistributionalOpportunityQueueError(RuntimeError):
    """Raised when the ADR 0179 bundle or campaign graph is invalid."""


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
        "decision_value": 0.9,
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
        decision="Bind every host to byte-identical immutable inputs",
        priority=35 if not dependencies else 37,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=dependencies,
        command=command,
        artifact_path=report,
        stop_rule="Every regular file and checksum must match on all hosts.",
        expected_runtime_seconds=180,
        workload_class="shared-prerequisite",
        cpu_cores=1,
        memory_gib=1.0,
        uses_mlx=False,
    )


def task_specs(
    *,
    bundle_relative: Path,
    bundle_id: str,
) -> list[dict[str, Any]]:
    if len(bundle_id) != 64 or any(character not in "0123456789abcdef" for character in bundle_id):
        raise DistributionalOpportunityQueueError("bundle ID is not a lowercase digest")

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    bundle_fanout = _fanout_task(
        task_id=bundle_fanout_id,
        title="Fan out immutable ADR 0179 bundle",
        source=bundle_relative,
        bundle_relative=bundle_relative,
        dependencies=[],
        required_file="bundle.json",
    )
    train_fanout_id = f"{TASK_PREFIX}-train-data-fanout"
    train_fanout = _fanout_task(
        task_id=train_fanout_id,
        title="Fan out qualified R12 train data",
        source=TRAIN_DATASET,
        bundle_relative=bundle_relative,
        dependencies=[bundle_fanout_id],
        required_file="dataset.json",
    )
    validation_fanout_id = f"{TASK_PREFIX}-validation-data-fanout"
    validation_fanout = _fanout_task(
        task_id=validation_fanout_id,
        title="Fan out qualified R12 validation data",
        source=VALIDATION_DATASET,
        bundle_relative=bundle_relative,
        dependencies=[bundle_fanout_id],
        required_file="dataset.json",
    )

    authorize_id = f"{TASK_PREFIX}-authorize"
    authorization = _task(
        task_id=authorize_id,
        title="Authorize the matched distributional factorial",
        decision=(
            "Freeze exact data, reliability audit, graph, initialization, "
            "homoscedastic prior, and claim boundary"
        ),
        priority=36,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=[
            bundle_fanout_id,
            train_fanout_id,
            validation_fanout_id,
        ],
        command=[
            *_python_prefix("john1", bundle_relative),
            "-m",
            "cascadia_mlx.distributional_opportunity_experiment",
            "authorize",
            "--train-dataset",
            _local(TRAIN_DATASET),
            "--validation-dataset",
            _local(VALIDATION_DATASET),
            "--bundle-id",
            bundle_id,
            "--output",
            _local(AUTHORIZATION),
        ],
        artifact_path=AUTHORIZATION,
        stop_rule=(
            "Do not optimize any arm until the complete train-only authorization is frozen."
        ),
        expected_runtime_seconds=60,
        workload_class="shared-prerequisite",
        cpu_cores=4,
        memory_gib=3.0,
        uses_mlx=True,
    )
    auth_fanout_id = f"{TASK_PREFIX}-authorization-fanout"
    auth_fanout = _fanout_task(
        task_id=auth_fanout_id,
        title="Fan out frozen ADR 0179 authorization",
        source=AUTHORIZATION_PACKAGE,
        bundle_relative=bundle_relative,
        dependencies=[authorize_id],
        required_file="authorization.json",
    )

    tasks = [
        bundle_fanout,
        train_fanout,
        validation_fanout,
        authorization,
        auth_fanout,
    ]
    preflight_ids = {}
    for role, host in PRIMARY_HOSTS.items():
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids[host] = task_id
        report = CAMPAIGN_ROOT / "control/preflights" / f"{host}.json"
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Verify ADR 0179 authorization on {host}",
                decision=(
                    "Independently rebuild and accept every persisted "
                    "authorization field before creating an optimizer"
                ),
                priority=38,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[auth_fanout_id],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.distributional_opportunity_experiment",
                    "verify-authorization",
                    "--train-dataset",
                    _remote(host, TRAIN_DATASET),
                    "--validation-dataset",
                    _remote(host, VALIDATION_DATASET),
                    "--authorization",
                    _remote(host, AUTHORIZATION),
                    "--bundle-id",
                    bundle_id,
                    "--role",
                    role,
                    "--output",
                    _remote(host, report),
                ],
                artifact_path=report,
                stop_rule=(
                    "Do not create a run directory, optimizer, checkpoint, or training metric."
                ),
                expected_runtime_seconds=60,
                workload_class="shared-prerequisite",
                cpu_cores=4,
                memory_gib=3.0,
                uses_mlx=True,
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
                priority=43,
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
                priority=44,
                workload_class="replica",
            )
        )

    role_hosts = {**PRIMARY_HOSTS, **REPLAY_HOSTS}
    collection_id = f"{TASK_PREFIX}-collect"
    collection_report = CAMPAIGN_ROOT / "control/collection.json"
    collect_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for role, host in role_hosts.items():
        run_root = CAMPAIGN_ROOT / "runs" / role
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, run_root / 'report.json')}",
                _local(CAMPAIGN_ROOT / "collected/reports" / f"{role}.json"),
                "--artifact",
                (f"{host}:{_remote(host, run_root / 'training/final-model.safetensors')}"),
                _local(CAMPAIGN_ROOT / "collected/models" / f"{role}.safetensors"),
            ]
        )
    collect_command.extend(["--output", _local(collection_report)])
    tasks.append(
        _task(
            task_id=collection_id,
            title="Collect all eight distributional reports and models",
            decision="Bind every role report and serialized model by checksum",
            priority=45,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[*primary_ids, *replay_ids],
            command=collect_command,
            artifact_path=collection_report,
            stop_rule="Collect all 16 artifacts without substitution or repair.",
            expected_runtime_seconds=120,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    classification = CAMPAIGN_ROOT / "classification.json"
    classify_command = [
        *_python_prefix("john1", bundle_relative),
        "-m",
        "cascadia_mlx.distributional_opportunity_experiment",
        "classify",
    ]
    for role in role_hosts:
        classify_command.extend(
            [
                "--report",
                (f"{role}={_local(CAMPAIGN_ROOT / 'collected/reports' / f'{role}.json')}"),
                "--model",
                (f"{role}={_local(CAMPAIGN_ROOT / 'collected/models' / f'{role}.safetensors')}"),
            ]
        )
    classify_command.extend(["--output", _local(classification)])
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-classify",
            title="Classify the matched distributional factorial",
            decision=(
                "Select only a calibrated, mean-noninferior, exactly replayed "
                "uncertainty formulation"
            ),
            priority=46,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collection_id],
            command=classify_command,
            artifact_path=classification,
            stop_rule=("Apply ADR 0179 exactly; do not tune thresholds after seeing validation."),
            expected_runtime_seconds=30,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    return tasks


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
    run_root = CAMPAIGN_ROOT / "runs" / role
    return _task(
        task_id=task_id,
        title=f"Run ADR 0179 role {role}",
        decision=(
            "Measure one parameter-matched uncertainty formulation on the "
            "complete open validation split"
        ),
        priority=priority,
        decision_terminal=False,
        compatible_hosts=[host],
        dependencies=dependencies,
        command=[
            *_python_prefix(host, bundle_relative),
            "-m",
            "cascadia_mlx.distributional_opportunity_experiment",
            "run",
            "--train-dataset",
            _remote(host, TRAIN_DATASET),
            "--validation-dataset",
            _remote(host, VALIDATION_DATASET),
            "--authorization",
            _remote(host, AUTHORIZATION),
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
            "Run exactly 3,000 steps, never validate during training, select "
            "only the fixed final checkpoint, and keep test/gameplay closed."
        ),
        expected_runtime_seconds=900,
        workload_class=workload_class,
        cpu_cores=10,
        memory_gib=6.0,
        uses_mlx=True,
    )


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
    bundle.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
    )
    specification = subparsers.add_parser("build-spec")
    specification.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
    )
    specification.add_argument("--bundle", type=Path, required=True)
    specification.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SPEC,
    )
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
        raise DistributionalOpportunityQueueError("bundle belongs to another experiment")
    try:
        bundle_relative = args.bundle.resolve().relative_to(args.repository.resolve())
    except ValueError as error:
        raise DistributionalOpportunityQueueError(
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
