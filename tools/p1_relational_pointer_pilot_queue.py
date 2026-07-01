#!/usr/bin/env python3
"""Build the immutable ADR 0175 cluster campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "p1-relational-selected-prefix-pointer-pilot-v1"
TASK_PREFIX = "p1pilot-v1"
FOUNDATION_TASK = "p1ptr-v1-classify"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
TASK_PREFIX_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
SOURCE_INCLUDES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("python/cascadia_mlx"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0174-relational-hierarchical-pointer-foundation.md"),
    Path("docs/v2/decisions/0175-relational-selected-prefix-pointer-pilot.md"),
    Path(
        "docs/v2/reports/"
        "p1-relational-selected-prefix-pointer-pilot-v1-preregistration.md"
    ),
    Path(
        "artifacts/experiments/relational-substrate-mlx-tournament-v1/"
        "reports/c0_exact_r2.json"
    ),
    Path(
        "artifacts/experiments/relational-substrate-mlx-tournament-v1/"
        "runs/c0_exact_r2/checkpoints/"
        "step-000003000-epoch-0000-batch-003000"
    ),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/p1_relational_pointer_pilot_queue.py"),
    Path("tools/rust_experiment_bundle.py"),
)
FACTOR_CACHE = Path(
    "artifacts/experiments/"
    "full-legal-hierarchical-factor-retrieval-pilot-v1/cache"
)
R3_CACHE = Path(
    "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/"
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
FOUNDATION_CLASSIFICATION = Path(
    "artifacts/experiments/"
    "p1-relational-hierarchical-pointer-foundation-v1/classification.json"
)
AUTHORIZATION_ROOT = CAMPAIGN_ROOT / "authorization"
COLLECTED_ROOT = CAMPAIGN_ROOT / "collected"
REPLAY_INPUT_ROOT = CAMPAIGN_ROOT / "replay-input"
REPLAY_REPORT_ROOT = CAMPAIGN_ROOT / "replays"
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
STAGE_HOSTS = {
    "draft": "john1",
    "tile": "john2",
    "wildlife": "john3",
}
STAGES = tuple(STAGE_HOSTS)
SCHEDULER_WORKLOAD_CLASSES = {
    "independent-experiment",
    "divisible-evidence",
    "shared-prerequisite",
    "replica",
}
REPLAY_HOSTS = {
    "draft": "john4",
    "tile": "john4",
    "wildlife": "john2",
}
SELECTED_FILES = (
    "final-report.json",
    "selected/selection.json",
    "selected/checkpoint.json",
    "selected/model.safetensors",
)


class PointerPilotQueueError(RuntimeError):
    """The ADR 0175 bundle or queue graph is invalid."""


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


def _stage_command(
    *,
    stage: str,
    host: str,
    bundle_relative: Path,
    bundle_id: str,
) -> list[str]:
    source = _source(host, bundle_relative)
    run_dir = CAMPAIGN_ROOT / "runs" / stage
    warm_root = Path(
        "artifacts/experiments/relational-substrate-mlx-tournament-v1"
    )
    return [
        *_python_prefix(host, bundle_relative),
        "-m",
        "cascadia_mlx.p1_relational_pointer_train",
        "--stage",
        stage,
        "--run-dir",
        _remote(host, run_dir),
        "--output",
        _remote(host, run_dir / "final-report.json"),
        "--factor-cache",
        _remote(host, FACTOR_CACHE),
        "--r3-cache",
        _remote(host, R3_CACHE),
        "--warm-start-checkpoint",
        str(
            source
            / warm_root
            / "runs/c0_exact_r2/checkpoints/"
            "step-000003000-epoch-0000-batch-003000"
        ),
        "--warm-start-report",
        str(source / warm_root / "reports/c0_exact_r2.json"),
        "--foundation-classification",
        _remote(host, AUTHORIZATION_ROOT / "classification.json"),
        "--bundle-id",
        bundle_id,
    ]


def _smoke_command(
    *,
    host: str,
    bundle_relative: Path,
) -> list[str]:
    source = _source(host, bundle_relative)
    run_dir = CAMPAIGN_ROOT / "smoke/wildlife-john4"
    warm_root = Path(
        "artifacts/experiments/relational-substrate-mlx-tournament-v1"
    )
    return [
        *_python_prefix(host, bundle_relative),
        "-m",
        "cascadia_mlx.p1_relational_pointer_train",
        "--stage",
        "wildlife",
        "--run-dir",
        _remote(host, run_dir),
        "--output",
        _remote(host, run_dir / "final-report.json"),
        "--factor-cache",
        _remote(host, FACTOR_CACHE),
        "--r3-cache",
        _remote(host, R3_CACHE),
        "--warm-start-checkpoint",
        str(
            source
            / warm_root
            / "runs/c0_exact_r2/checkpoints/"
            "step-000003000-epoch-0000-batch-003000"
        ),
        "--warm-start-report",
        str(source / warm_root / "reports/c0_exact_r2.json"),
        "--smoke-batches",
        "1",
    ]


def _collect_stage_command(
    *,
    stage: str,
    host: str,
    bundle_relative: Path,
) -> list[str]:
    command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    remote_run = CAMPAIGN_ROOT / "runs" / stage
    local_run = COLLECTED_ROOT / stage
    for relative in SELECTED_FILES:
        command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, remote_run / relative)}",
                _remote("john1", local_run / relative),
            ]
        )
    command.extend(
        [
            "--output",
            _remote(
                "john1",
                CAMPAIGN_ROOT / f"control/collect-{stage}.json",
            ),
        ]
    )
    return command


def _fanout_stage_command(
    *,
    stage: str,
    replay_host: str,
    bundle_relative: Path,
) -> list[str]:
    local_root = COLLECTED_ROOT / stage
    remote_root = REPLAY_INPUT_ROOT / stage
    command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{_remote('john1', local_root)}/",
        "--local-root",
        _remote("john1", local_root),
        "--destination",
        f"{replay_host}:{_remote(replay_host, remote_root)}/",
    ]
    for relative in SELECTED_FILES:
        command.extend(["--required-file", relative])
    command.extend(
        [
            "--verify-tree",
            "--output",
            _remote(
                "john1",
                CAMPAIGN_ROOT / f"control/fanout-{stage}.json",
            ),
        ]
    )
    return command


def _replay_command(
    *,
    stage: str,
    host: str,
    bundle_relative: Path,
) -> list[str]:
    return [
        *_python_prefix(host, bundle_relative),
        "-m",
        "cascadia_mlx.p1_relational_pointer_evaluate",
        "replay",
        "--stage",
        stage,
        "--run-dir",
        _remote(host, REPLAY_INPUT_ROOT / stage),
        "--factor-cache",
        _remote(host, FACTOR_CACHE),
        "--r3-cache",
        _remote(host, R3_CACHE),
        "--output",
        _remote(host, REPLAY_REPORT_ROOT / stage / "report.json"),
    ]


def task_specs(
    *,
    bundle_relative: Path,
    bundle_id: str,
    task_prefix: str = TASK_PREFIX,
) -> list[dict[str, Any]]:
    """Return the complete authorization, training, replay, and gate graph."""
    if not TASK_PREFIX_PATTERN.fullmatch(task_prefix):
        raise PointerPilotQueueError("task prefix contains unsupported characters")
    if (
        len(bundle_id) != 64
        or any(character not in "0123456789abcdef" for character in bundle_id)
    ):
        raise PointerPilotQueueError("bundle ID is not a lowercase BLAKE3 digest")

    auth_stage = f"{task_prefix}-auth-stage"
    auth_fanout = f"{task_prefix}-auth-fanout"
    tasks = [
        _task(
            task_id=auth_stage,
            title="Stage ADR 0174 pointer authorization",
            decision="Bind the crossed foundation pass into the ADR 0175 campaign",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=30,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[FOUNDATION_TASK],
            command=[
                *_python_prefix("john1", bundle_relative),
                "tools/cluster_artifact_collect.py",
                "--artifact",
                f"john1:{_remote('john1', FOUNDATION_CLASSIFICATION)}",
                _remote("john1", AUTHORIZATION_ROOT / "classification.json"),
                "--output",
                _remote("john1", AUTHORIZATION_ROOT / "collection.json"),
            ],
            artifact_path=AUTHORIZATION_ROOT / "collection.json",
            stop_rule="The exact ADR 0174 terminal classification must exist.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        ),
        _task(
            task_id=auth_fanout,
            title="Fan out ADR 0175 authorization",
            decision="Give every production host the identical launch control",
            workload_class="shared-prerequisite",
            priority=31,
            expected_runtime_seconds=45,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[auth_stage],
            command=[
                *_python_prefix("john1", bundle_relative),
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{_remote('john1', AUTHORIZATION_ROOT)}/",
                "--local-root",
                _remote("john1", AUTHORIZATION_ROOT),
                "--destination",
                f"john2:{_remote('john2', AUTHORIZATION_ROOT)}/",
                "--destination",
                f"john3:{_remote('john3', AUTHORIZATION_ROOT)}/",
                "--destination",
                f"john4:{_remote('john4', AUTHORIZATION_ROOT)}/",
                "--required-file",
                "classification.json",
                "--verify-tree",
                "--output",
                _remote("john1", AUTHORIZATION_ROOT / "fanout.json"),
            ],
            artifact_path=AUTHORIZATION_ROOT / "fanout.json",
            stop_rule="All three remote authorization trees must match john1.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        ),
    ]

    train_ids: dict[str, str] = {}
    for stage, host in STAGE_HOSTS.items():
        task_id = f"{task_prefix}-train-{stage}-{host}"
        train_ids[stage] = task_id
        runtime = {"draft": 600, "tile": 1800, "wildlife": 3600}[stage]
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Train P1 {stage} pointer on {host}",
                decision=f"Learn the frozen {stage} selected-prefix pointer",
                workload_class="independent-experiment",
                priority=32,
                expected_runtime_seconds=runtime,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[auth_fanout],
                command=_stage_command(
                    stage=stage,
                    host=host,
                    bundle_relative=bundle_relative,
                    bundle_id=bundle_id,
                ),
                artifact_path=CAMPAIGN_ROOT / "runs" / stage / "final-report.json",
                stop_rule=(
                    "Run the frozen stage budget once; select on train only and "
                    "open validation only after selection."
                ),
                cpu_cores=10,
                memory_gib=5.0,
                uses_mlx=True,
            )
        )

    smoke_id = f"{task_prefix}-smoke-wildlife-john4"
    tasks.append(
        _task(
            task_id=smoke_id,
            title="Replay the wildlife implementation smoke on john4",
            decision="Exercise the most complex pointer path on the fourth host",
            workload_class="replica",
            priority=32,
            expected_runtime_seconds=180,
            decision_terminal=False,
            compatible_hosts=["john4"],
            dependencies=[auth_fanout],
            command=_smoke_command(
                host="john4",
                bundle_relative=bundle_relative,
            ),
            artifact_path=CAMPAIGN_ROOT
            / "smoke/wildlife-john4/final-report.json",
            stop_rule="Exactly one bounded batch; no result-dependent tuning.",
            cpu_cores=10,
            memory_gib=5.0,
            uses_mlx=True,
        )
    )

    replay_ids = {}
    previous_john4 = smoke_id
    for stage in STAGES:
        origin_host = STAGE_HOSTS[stage]
        replay_host = REPLAY_HOSTS[stage]
        collect_id = f"{task_prefix}-collect-{stage}"
        fanout_id = f"{task_prefix}-fanout-{stage}"
        replay_id = f"{task_prefix}-replay-{stage}-{replay_host}"
        replay_ids[stage] = replay_id
        tasks.append(
            _task(
                task_id=collect_id,
                title=f"Collect selected P1 {stage} checkpoint",
                decision="Bind the selected stage report and fixed model on john1",
                workload_class="shared-prerequisite",
                priority=40,
                expected_runtime_seconds=60,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[train_ids[stage]],
                command=_collect_stage_command(
                    stage=stage,
                    host=origin_host,
                    bundle_relative=bundle_relative,
                ),
                artifact_path=CAMPAIGN_ROOT / f"control/collect-{stage}.json",
                stop_rule="Collect all four fixed selected-stage files by checksum.",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )
        tasks.append(
            _task(
                task_id=fanout_id,
                title=f"Fan out P1 {stage} checkpoint for replay",
                decision="Install the selected model on a distinct replay host",
                workload_class="shared-prerequisite",
                priority=41,
                expected_runtime_seconds=60,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collect_id],
                command=_fanout_stage_command(
                    stage=stage,
                    replay_host=replay_host,
                    bundle_relative=bundle_relative,
                ),
                artifact_path=CAMPAIGN_ROOT / f"control/fanout-{stage}.json",
                stop_rule="Require whole-tree identity on the replay host.",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )
        replay_dependencies = [fanout_id]
        if replay_host == "john4":
            replay_dependencies.append(previous_john4)
            previous_john4 = replay_id
        tasks.append(
            _task(
                task_id=replay_id,
                title=f"Replay selected P1 {stage} stage on {replay_host}",
                decision="Require exact open-split metrics on a distinct host",
                workload_class="replica",
                priority=42,
                expected_runtime_seconds={
                    "draft": 300,
                    "tile": 900,
                    "wildlife": 1800,
                }[stage],
                decision_terminal=False,
                compatible_hosts=[replay_host],
                dependencies=replay_dependencies,
                command=_replay_command(
                    stage=stage,
                    host=replay_host,
                    bundle_relative=bundle_relative,
                ),
                artifact_path=REPLAY_REPORT_ROOT / stage / "report.json",
                stop_rule="Replay complete train and validation exactly once.",
                cpu_cores=10,
                memory_gib=5.0,
                uses_mlx=True,
            )
        )

    collect_replays = f"{task_prefix}-collect-replays"
    replay_collection = CAMPAIGN_ROOT / "control/collect-replays.json"
    collect_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for stage in STAGES:
        host = REPLAY_HOSTS[stage]
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, REPLAY_REPORT_ROOT / stage / 'report.json')}",
                _remote(
                    "john1",
                    REPLAY_REPORT_ROOT / stage / "report.json",
                ),
            ]
        )
    collect_command.extend(
        ["--output", _remote("john1", replay_collection)]
    )
    tasks.append(
        _task(
            task_id=collect_replays,
            title="Collect all P1 cross-host replays",
            decision="Bind the three exact replay reports on the coordinator",
            workload_class="shared-prerequisite",
            priority=50,
            expected_runtime_seconds=60,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=list(replay_ids.values()),
            command=collect_command,
            artifact_path=replay_collection,
            stop_rule="Every replay report must match its remote checksum.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    integration_id = f"{task_prefix}-integrate"
    integration_path = CAMPAIGN_ROOT / "integration.json"
    integration_command = [
        *_python_prefix("john1", bundle_relative),
        "-m",
        "cascadia_mlx.p1_relational_pointer_evaluate",
        "integrate",
        "--draft-run-dir",
        _remote("john1", COLLECTED_ROOT / "draft"),
        "--tile-run-dir",
        _remote("john1", COLLECTED_ROOT / "tile"),
        "--wildlife-run-dir",
        _remote("john1", COLLECTED_ROOT / "wildlife"),
        "--draft-replay",
        _remote("john1", REPLAY_REPORT_ROOT / "draft/report.json"),
        "--tile-replay",
        _remote("john1", REPLAY_REPORT_ROOT / "tile/report.json"),
        "--wildlife-replay",
        _remote("john1", REPLAY_REPORT_ROOT / "wildlife/report.json"),
        "--factor-cache",
        _remote("john1", FACTOR_CACHE),
        "--r3-cache",
        _remote("john1", R3_CACHE),
        "--output",
        _remote("john1", integration_path),
    ]
    tasks.append(
        _task(
            task_id=integration_id,
            title="Integrate and classify the P1 pointer pilot",
            decision="Apply every ADR 0175 proposal and selector gate",
            workload_class="shared-prerequisite",
            priority=60,
            expected_runtime_seconds=2400,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[
                collect_replays,
                *(f"{task_prefix}-collect-{stage}" for stage in STAGES),
            ],
            command=integration_command,
            artifact_path=integration_path,
            stop_rule=(
                "Apply preregistered gates mechanically; no validation-driven "
                "repair or gameplay launch."
            ),
            cpu_cores=10,
            memory_gib=6.0,
            uses_mlx=True,
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
        raise PointerPilotQueueError("bundle belongs to another experiment")
    try:
        relative = bundle.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise PointerPilotQueueError(
            "bundle must remain beneath the repository"
        ) from error
    return task_specs(
        bundle_relative=relative,
        bundle_id=manifest["bundle_id"],
        task_prefix=task_prefix,
    )


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
