#!/usr/bin/env python3
"""Build the immutable crossed-host O1 corpus-reuse audit campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "o1-opponent-intent-corpus-reuse-audit-v1"
TASK_PREFIX = "o1reuse-v3"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec-v3.json"
TRAIN_DATASET = Path("artifacts/datasets/canonical-action-imitation-v1-train")
VALIDATION_DATASET = Path("artifacts/datasets/canonical-action-imitation-v1-validation")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
SOURCE_INCLUDES = (
    Path("CASCADIA_V2_GOAL.txt"),
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("crates/cascadia-data"),
    Path("crates/cascadia-game"),
    Path("crates/cascadia-provenance"),
    Path("crates/cascadia-sim"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0182-o1-opponent-intent-corpus-reuse-audit.md"),
    Path("docs/v2/decisions/0183-portable-imitation-dataset-validation.md"),
    Path("docs/v2/decisions/0184-o1-cross-host-scientific-path-normalization.md"),
    Path("docs/v2/reports/o1-opponent-intent-corpus-reuse-audit-v1-preregistration.md"),
    Path("docs/v2/reports/o1-opponent-intent-corpus-reuse-audit-v1-invalid-launch-1.md"),
    Path("docs/v2/reports/o1-opponent-intent-corpus-reuse-audit-v1-invalid-launch-2.md"),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/o1_opponent_intent_reuse_audit_queue.py"),
    Path("tools/o1_opponent_intent_reuse_audit_report.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/test_o1_opponent_intent_reuse_audit_queue.py"),
    Path("tools/test_o1_opponent_intent_reuse_audit_report.py"),
)
BINARY = Path("target/release/opponent_intent_reuse_audit")


class O1ReuseQueueError(RuntimeError):
    """Raised when the O1 audit bundle or execution graph is invalid."""


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
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: Path,
    stop_rule: str,
    expected_runtime_seconds: int,
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
            "uses_mlx": False,
        },
    }


def _fanout_task(
    *,
    task_id: str,
    title: str,
    source: Path,
    destinations: tuple[str, ...],
    required_files: tuple[str, ...],
    dependencies: list[str],
    report: Path,
) -> dict[str, Any]:
    command = [
        ".venv/bin/python",
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{source}/",
        "--local-root",
        str(source),
    ]
    for host in destinations:
        command.extend(["--destination", f"{host}:{_remote(host, source)}/"])
    for required_file in required_files:
        command.extend(["--required-file", required_file])
    command.extend(["--verify-tree", "--output", str(report)])
    return _task(
        task_id=task_id,
        title=title,
        decision="Bind every execution host to byte-identical immutable inputs",
        workload_class="shared-prerequisite",
        priority=70,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=dependencies,
        command=command,
        artifact_path=report,
        stop_rule="Every regular file and checksum must match on every destination.",
        expected_runtime_seconds=180,
        cpu_cores=1,
        memory_gib=1.0,
    )


def task_specs(*, bundle_relative: Path, bundle_id: str) -> list[dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise O1ReuseQueueError("bundle path must be repository-relative")
    if len(bundle_id) != 64:
        raise O1ReuseQueueError("bundle ID must be a 64-character digest")
    launch_root = CAMPAIGN_ROOT / "launches" / bundle_id

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    train_fanout_id = f"{TASK_PREFIX}-train-fanout"
    validation_fanout_id = f"{TASK_PREFIX}-validation-fanout"
    tasks = [
        _fanout_task(
            task_id=bundle_fanout_id,
            title="Fan out immutable O1 reuse-audit bundle",
            source=bundle_relative,
            destinations=("john2", "john4"),
            required_files=("bundle.json", "bin/opponent_intent_reuse_audit"),
            dependencies=[],
            report=launch_root / "control/bundle-fanout.json",
        ),
        _fanout_task(
            task_id=train_fanout_id,
            title="Fan out canonical imitation train corpus",
            source=TRAIN_DATASET,
            destinations=("john2", "john4"),
            required_files=("dataset.json",),
            dependencies=[],
            report=launch_root / "control/train-fanout.json",
        ),
        _fanout_task(
            task_id=validation_fanout_id,
            title="Fan out canonical imitation validation corpus",
            source=VALIDATION_DATASET,
            destinations=("john2", "john4"),
            required_files=("dataset.json",),
            dependencies=[],
            report=launch_root / "control/validation-fanout.json",
        ),
    ]
    run_ids = []
    for role, host in (("primary", "john4"), ("replay", "john2")):
        task_id = f"{TASK_PREFIX}-{role}"
        run_ids.append(task_id)
        output = launch_root / "runs" / role / "report.json"
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Run O1 corpus reuse audit {role} on {host}",
                decision=(
                    "Prove exact sequential replay, tile identity recovery, "
                    "survival labels, and split isolation"
                ),
                workload_class="replica",
                priority=71,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[
                    bundle_fanout_id,
                    train_fanout_id,
                    validation_fanout_id,
                ],
                command=[
                    "/usr/bin/env",
                    "-C",
                    _remote(host, bundle_relative / "source"),
                    _remote(host, bundle_relative / "bin/opponent_intent_reuse_audit"),
                    "--dataset-root",
                    _remote(host, TRAIN_DATASET),
                    "--dataset-root",
                    _remote(host, VALIDATION_DATASET),
                    "--output",
                    _remote(host, output),
                ],
                artifact_path=output,
                stop_rule="Any state, action, terminal, identity, or split-overlap mismatch fails.",
                expected_runtime_seconds=900,
                cpu_cores=10,
                memory_gib=4.0,
            )
        )

    collect_id = f"{TASK_PREFIX}-collect"
    primary_local = launch_root / "collected/john4-primary.json"
    replay_local = launch_root / "collected/john2-replay.json"
    collect_report = launch_root / "control/collection.json"
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect crossed-host O1 reuse-audit reports",
            decision="Retrieve both exact audit reports with checksum receipts",
            workload_class="shared-prerequisite",
            priority=72,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=run_ids,
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_collect.py",
                "--artifact",
                f"john4:{_remote('john4', launch_root / 'runs/primary/report.json')}",
                str(primary_local),
                "--artifact",
                f"john2:{_remote('john2', launch_root / 'runs/replay/report.json')}",
                str(replay_local),
                "--output",
                str(collect_report),
            ],
            artifact_path=collect_report,
            stop_rule="Both reports must be present locally with verified transfer checksums.",
            expected_runtime_seconds=120,
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    classification = launch_root / "classification.json"
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-classify",
            title="Mechanically classify O1 corpus foundation reuse",
            decision="Authorize foundation reuse only when all exact and replay gates pass",
            workload_class="shared-prerequisite",
            priority=73,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collect_id],
            command=[
                str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
                _remote(
                    "john1",
                    bundle_relative / "source/tools/o1_opponent_intent_reuse_audit_report.py",
                ),
                "--primary",
                str(primary_local),
                "--replay",
                str(replay_local),
                "--output",
                str(classification),
                "--canonical-output",
                str(CAMPAIGN_ROOT / "classification.json"),
            ],
            artifact_path=classification,
            stop_rule=(
                "Primary and replay scientific fields must match exactly; "
                "the classifier may not authorize final O1 training."
            ),
            expected_runtime_seconds=60,
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    return tasks


def campaign_spec(tasks: list[dict[str, Any]], *, bundle_id: str) -> dict[str, Any]:
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
        if not (args.repository / BINARY).is_file():
            raise O1ReuseQueueError(f"build the release audit binary before bundling: {BINARY}")
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
        raise O1ReuseQueueError("bundle belongs to another experiment")
    try:
        relative = args.bundle.resolve().relative_to(args.repository.resolve())
    except ValueError as error:
        raise O1ReuseQueueError("bundle must remain beneath the repository") from error
    tasks = task_specs(bundle_relative=relative, bundle_id=manifest["bundle_id"])
    _write_json(args.output, campaign_spec(tasks, bundle_id=manifest["bundle_id"]))
    print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
