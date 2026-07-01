#!/usr/bin/env python3
"""Build the immutable four-host ADR 0187 MLX campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "o1-opponent-intent-mlx-factorial-v1"
TASK_PREFIX = "o1mlx-v1"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
AUTHORIZATION_PACKAGE = CAMPAIGN_ROOT / "control/authorization-package"
AUTHORIZATION = AUTHORIZATION_PACKAGE / "authorization.json"
CORPUS_ROOT = Path("artifacts/experiments/o1-opponent-intent-policy-heldout-corpus-v1")
DATA_ROOT = CORPUS_ROOT / "datasets"
CORPUS_CLASSIFICATION = CORPUS_ROOT / "classification.json"
TRAIN_ROLES = ("train-part-0", "train-part-1")
VALIDATION_ROLE = "validation"
TEST_ROLE = "test"
FINAL_ROLE = "final-stress"
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
PRIMARY_HOSTS = {
    "a0-primary": "john1",
    "a1-primary": "john2",
    "a2-primary": "john3",
    "a3-primary": "john4",
}
REPLAY_HOSTS = {
    "a0-replay": "john2",
    "a1-replay": "john3",
    "a2-replay": "john4",
    "a3-replay": "john1",
}
SOURCE_INCLUDES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("python/cascadia_mlx"),
    Path("python/tests/test_opponent_intent_dataset.py"),
    Path("python/tests/test_opponent_intent_model.py"),
    Path("python/tests/test_opponent_intent_experiment.py"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0187-o1-opponent-intent-mlx-factorial.md"),
    Path("docs/v2/reports/o1-opponent-intent-mlx-factorial-v1-preregistration.md"),
    CORPUS_CLASSIFICATION,
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/cluster_artifact_tree_collect.py"),
    Path("tools/o1_opponent_intent_mlx_queue.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/test_o1_opponent_intent_mlx_queue.py"),
)


class O1MlxQueueError(RuntimeError):
    """Raised when the ADR 0187 bundle or queue graph is invalid."""


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
        "decision_value": 0.98,
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
    required_files: list[str],
) -> dict[str, Any]:
    report = CAMPAIGN_ROOT / "control" / f"{task_id}.json"
    command = [
        *_python_prefix("john1", bundle_relative),
        str(_source("john1", bundle_relative) / "tools/cluster_artifact_fanout.py"),
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
    for required_file in required_files:
        command.extend(["--required-file", required_file])
    command.extend(
        [
            "--verify-tree",
            "--output",
            _local(report),
        ]
    )
    return _task(
        task_id=task_id,
        title=title,
        decision="Bind every Mac to byte-identical immutable inputs",
        priority=40,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=dependencies,
        command=command,
        artifact_path=report,
        stop_rule="Every regular file and checksum must match on all four Macs.",
        expected_runtime_seconds=300,
        workload_class="shared-prerequisite",
        cpu_cores=1,
        memory_gib=1.0,
        uses_mlx=False,
    )


def _open_data_arguments(host: str, bundle_relative: Path) -> list[str]:
    return [
        "--train-dataset",
        _remote(host, DATA_ROOT / TRAIN_ROLES[0]),
        "--train-dataset",
        _remote(host, DATA_ROOT / TRAIN_ROLES[1]),
        "--validation-dataset",
        _remote(host, DATA_ROOT / VALIDATION_ROLE),
        "--corpus-classification",
        str(_source(host, bundle_relative) / CORPUS_CLASSIFICATION),
    ]


def task_specs(
    *,
    bundle_relative: Path,
    bundle_id: str,
) -> list[dict[str, Any]]:
    if len(bundle_id) != 64 or any(character not in "0123456789abcdef" for character in bundle_id):
        raise O1MlxQueueError("bundle ID is not a lowercase digest")

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    bundle_fanout = _fanout_task(
        task_id=bundle_fanout_id,
        title="Fan out immutable ADR 0187 source bundle",
        source=bundle_relative,
        bundle_relative=bundle_relative,
        dependencies=[],
        required_files=["bundle.json"],
    )
    data_fanout_id = f"{TASK_PREFIX}-data-fanout"
    data_fanout = _fanout_task(
        task_id=data_fanout_id,
        title="Fan out all five immutable O1 corpus roles",
        source=DATA_ROOT,
        bundle_relative=bundle_relative,
        dependencies=[],
        required_files=[
            f"{role}/dataset.json"
            for role in (*TRAIN_ROLES, VALIDATION_ROLE, TEST_ROLE, FINAL_ROLE)
        ],
    )

    authorize_id = f"{TASK_PREFIX}-authorize"
    authorize = _task(
        task_id=authorize_id,
        title="Freeze O1 MLX authorization and train-only priors",
        decision=(
            "Bind exact train and validation bytes, corpus scope, graph, "
            "initialization, priors, thresholds, and sealed-test rule"
        ),
        priority=42,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=[bundle_fanout_id, data_fanout_id],
        command=[
            *_python_prefix("john1", bundle_relative),
            "-m",
            "cascadia_mlx.opponent_intent_experiment",
            "authorize",
            *_open_data_arguments("john1", bundle_relative),
            "--bundle-id",
            bundle_id,
            "--output",
            _local(AUTHORIZATION),
        ],
        artifact_path=AUTHORIZATION,
        stop_rule="No optimizer may exist before the authorization is frozen.",
        expected_runtime_seconds=60,
        workload_class="shared-prerequisite",
        cpu_cores=4,
        memory_gib=2.0,
        uses_mlx=True,
    )
    auth_fanout_id = f"{TASK_PREFIX}-authorization-fanout"
    auth_fanout = _fanout_task(
        task_id=auth_fanout_id,
        title="Fan out frozen O1 MLX authorization",
        source=AUTHORIZATION_PACKAGE,
        bundle_relative=bundle_relative,
        dependencies=[authorize_id],
        required_files=["authorization.json"],
    )

    tasks = [bundle_fanout, data_fanout, authorize, auth_fanout]
    preflight_ids = {}
    for role, host in PRIMARY_HOSTS.items():
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids[host] = task_id
        output = CAMPAIGN_ROOT / "control/preflights" / f"{host}.json"
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Verify O1 MLX launch controls on {host}",
                decision=(
                    "Rebuild every authorization field before creating a run directory or optimizer"
                ),
                priority=44,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[auth_fanout_id],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.opponent_intent_experiment",
                    "verify-authorization",
                    *_open_data_arguments(host, bundle_relative),
                    "--authorization",
                    _remote(host, AUTHORIZATION),
                    "--bundle-id",
                    bundle_id,
                    "--role",
                    role,
                    "--output",
                    _remote(host, output),
                ],
                artifact_path=output,
                stop_rule=(
                    "The host must reproduce authorization without creating "
                    "an optimizer or run directory."
                ),
                expected_runtime_seconds=90,
                workload_class="shared-prerequisite",
                cpu_cores=4,
                memory_gib=2.0,
                uses_mlx=True,
            )
        )

    primary_ids = []
    for role, host in PRIMARY_HOSTS.items():
        task_id = f"{TASK_PREFIX}-run-{role}"
        primary_ids.append(task_id)
        run_root = CAMPAIGN_ROOT / "remote-runs" / role
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Train O1 MLX primary arm {role}",
                decision=(
                    "Measure one distinct public-state, history, auxiliary, "
                    "or joint-intent information pathway"
                ),
                priority=60,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[preflight_ids[host]],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.opponent_intent_experiment",
                    "run",
                    *_open_data_arguments(host, bundle_relative),
                    "--authorization",
                    _remote(host, AUTHORIZATION),
                    "--bundle-id",
                    bundle_id,
                    "--role",
                    role,
                    "--run-dir",
                    _remote(host, run_root / "training"),
                    "--output",
                    _remote(host, run_root / "report.json"),
                ],
                artifact_path=run_root / "report.json",
                stop_rule=(
                    "Run exactly 5,120 fixed steps and evaluate only open "
                    "PatternCompetition validation."
                ),
                expected_runtime_seconds=480,
                workload_class="independent-experiment",
                cpu_cores=10,
                memory_gib=4.0,
                uses_mlx=True,
            )
        )

    replay_ids = []
    for role, host in REPLAY_HOSTS.items():
        task_id = f"{TASK_PREFIX}-run-{role}"
        replay_ids.append(task_id)
        run_root = CAMPAIGN_ROOT / "remote-runs" / role
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Replay O1 MLX arm {role} on a rotated Mac",
                decision=(
                    "Require exact final tensors, model bytes, predictions, "
                    "metrics, and role-neutral identity on different hardware"
                ),
                priority=70,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=primary_ids,
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.opponent_intent_experiment",
                    "run",
                    *_open_data_arguments(host, bundle_relative),
                    "--authorization",
                    _remote(host, AUTHORIZATION),
                    "--bundle-id",
                    bundle_id,
                    "--role",
                    role,
                    "--run-dir",
                    _remote(host, run_root / "training"),
                    "--output",
                    _remote(host, run_root / "report.json"),
                ],
                artifact_path=run_root / "report.json",
                stop_rule=(
                    "Reproduce the corresponding arm exactly without opening test or final data."
                ),
                expected_runtime_seconds=480,
                workload_class="replica",
                cpu_cores=10,
                memory_gib=4.0,
                uses_mlx=True,
            )
        )

    collect_id = f"{TASK_PREFIX}-collect-runs"
    collect_report = CAMPAIGN_ROOT / "control/run-tree-collection.json"
    collect_command = [
        *_python_prefix("john1", bundle_relative),
        str(_source("john1", bundle_relative) / "tools/cluster_artifact_tree_collect.py"),
    ]
    all_hosts = {**PRIMARY_HOSTS, **REPLAY_HOSTS}
    for role, host in all_hosts.items():
        collect_command.extend(
            [
                "--tree",
                (f"{host}:{_remote(host, CAMPAIGN_ROOT / 'remote-runs' / role)}"),
                _local(CAMPAIGN_ROOT / "collected" / role),
            ]
        )
    collect_command.extend(["--output", _local(collect_report)])
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect all eight O1 MLX run trees",
            decision="Bind reports, models, checkpoints, and evidence by checksum",
            priority=80,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[*primary_ids, *replay_ids],
            command=collect_command,
            artifact_path=collect_report,
            stop_rule="All eight complete immutable run trees must collect exactly.",
            expected_runtime_seconds=180,
            workload_class="shared-prerequisite",
            cpu_cores=1,
            memory_gib=2.0,
            uses_mlx=False,
        )
    )

    classify_id = f"{TASK_PREFIX}-classify-validation"
    classification = CAMPAIGN_ROOT / "validation-classification.json"
    classify_command = [
        *_python_prefix("john1", bundle_relative),
        "-m",
        "cascadia_mlx.opponent_intent_experiment",
        "classify",
    ]
    for role in all_hosts:
        root = CAMPAIGN_ROOT / "collected" / role
        classify_command.extend(
            [
                "--report",
                f"{role}={_local(root / 'report.json')}",
                "--evidence",
                f"{role}={_local(root / 'training/validation-evidence.npz')}",
                "--model",
                f"{role}={_local(root / 'training/final-model.safetensors')}",
            ]
        )
    classify_command.extend(["--output", _local(classification)])
    tasks.append(
        _task(
            task_id=classify_id,
            title="Classify O1 policy-held-out validation",
            decision=(
                "Apply exact replay, calibrated Brier, bootstrap, and "
                "auxiliary-learning gates without opening sealed data"
            ),
            priority=82,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[collect_id],
            command=classify_command,
            artifact_path=classification,
            stop_rule=(
                "Select only an eligible treatment under frozen thresholds; "
                "leave test unopened after a null."
            ),
            expected_runtime_seconds=60,
            workload_class="shared-prerequisite",
            cpu_cores=4,
            memory_gib=3.0,
            uses_mlx=False,
        )
    )

    terminal_id = f"{TASK_PREFIX}-sealed-test"
    terminal = CAMPAIGN_ROOT / "classification.json"
    terminal_command = [
        *_python_prefix("john1", bundle_relative),
        "-m",
        "cascadia_mlx.opponent_intent_experiment",
        "evaluate-selected",
        "--classification",
        _local(classification),
        "--authorization",
        _local(AUTHORIZATION),
        "--test-dataset",
        _local(DATA_ROOT / TEST_ROLE),
        "--final-stress-dataset",
        _local(DATA_ROOT / FINAL_ROLE),
    ]
    for role in PRIMARY_HOSTS:
        root = CAMPAIGN_ROOT / "collected" / role
        terminal_command.extend(
            [
                "--report",
                f"{role}={_local(root / 'report.json')}",
                "--model",
                f"{role}={_local(root / 'training/final-model.safetensors')}",
            ]
        )
    terminal_command.extend(["--output", _local(terminal)])
    tasks.append(
        _task(
            task_id=terminal_id,
            title="Conditionally open O1 sealed policy holdout",
            decision=(
                "Leave test closed after a null or replicate selected history "
                "and intent value once on PatternPortfolio"
            ),
            priority=84,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[classify_id],
            command=terminal_command,
            artifact_path=terminal,
            stop_rule=(
                "PatternPortfolio opens once only after selection; Random is "
                "descriptive and no gameplay or score claim is permitted."
            ),
            expected_runtime_seconds=120,
            workload_class="shared-prerequisite",
            cpu_cores=4,
            memory_gib=3.0,
            uses_mlx=True,
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
        raise O1MlxQueueError("bundle belongs to another experiment")
    try:
        bundle_relative = args.bundle.resolve().relative_to(args.repository.resolve())
    except ValueError as error:
        raise O1MlxQueueError("bundle must remain beneath the repository") from error
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
