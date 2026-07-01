#!/usr/bin/env python3
"""Build the immutable four-host queue graph for ADR 0166."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.opportunity_cross_attention_mlx_model import ARMS
from cascadia_mlx.opportunity_cross_attention_mlx_protocol import (
    ARM_HOSTS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)
from opportunity_cross_attention_mlx_smoke_compare import SMOKE_STEPS
from relational_substrate_mlx_post_campaign import (
    validate_completed_control,
)
from rust_experiment_bundle import build_bundle, validate_bundle

TASK_PREFIX = "oppquery-v2"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "queue-spec.json"
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
LAUNCH_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
WARM_START_RUN = (
    Path("artifacts/experiments")
    / "relational-substrate-mlx-tournament-v1"
    / "runs"
    / "c0_exact_r2"
)
TRAIN_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-train"
)
VALIDATION_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-validation"
)
R3_CACHE = Path(
    "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/"
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
RELATIONAL_CACHE = (
    Path("artifacts/experiments")
    / "relational-substrate-mlx-tournament-v1"
    / "cache"
    / "d4f8e2eb83db237b136fd478b73802544938c36adf77db0bf40f2b3276181bef"
)
S1_CACHE = Path(
    "artifacts/experiments/exact-semantic-supply-learned-comparison-v1/"
    "cache/2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15"
)
R6_BINARY_NAME = "relational-substrate-r6-replay"
DEFAULT_R6_BINARY = Path(
    "tools/relational_feature_census/target/release/"
    "relational-substrate-r6-replay"
)
SOURCE_INCLUDES = (
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("Makefile"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("python/cascadia_mlx"),
    Path("apps/web/src"),
    Path("legacy/crates/cascadia-core"),
    Path("legacy/crates/cascadia-ai"),
    Path("crates"),
    Path("docs/v2/decisions/0166-exact-r2-opportunity-query-factorial.md"),
    Path(
        "docs/v2/decisions/"
        "0172-opportunity-paired-panel-group-id-normalization.md"
    ),
    Path(
        "docs/v2/decisions/"
        "0173-opportunity-terminal-classifier-semantics-repair.md"
    ),
    Path(
        "docs/v2/reports/"
        "opportunity-cross-attention-mlx-tournament-v1-preregistration.md"
    ),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/opportunity_cross_attention_mlx_campaign.py"),
    Path("tools/opportunity_cross_attention_mlx_queue.py"),
    Path("tools/opportunity_cross_attention_mlx_report.py"),
    Path("tools/opportunity_cross_attention_mlx_smoke_compare.py"),
    Path("tools/relational_substrate_mlx_post_campaign.py"),
    Path("tools/rust_experiment_bundle.py"),
)
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
HOST_PREREQUISITES = {
    "john1": "s6top-run-john1",
    "john2": "relmlx-c0-replay-john2",
    "john3": "relmlx-c0-replay-john3",
    "john4": "relmlx-c0-replay-john4",
}


class QueueSpecError(RuntimeError):
    """The ADR 0166 queue cannot be built from the supplied artifacts."""


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _relative(repository: Path, path: Path, label: str) -> Path:
    try:
        return path.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise QueueSpecError(
            f"ADR 0166 {label} must remain beneath the repository"
        ) from error


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _local(relative: Path) -> str:
    return _remote("john1", relative)


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


def _python_prefix(host: str, bundle_relative: Path) -> list[str]:
    root = REMOTE_ROOTS[host]
    source = root / bundle_relative / "source"
    return [
        "/usr/bin/env",
        "-C",
        str(source),
        "PYTHONPATH=python:tools",
        "PYTHONDONTWRITEBYTECODE=1",
        str(root / ".venv/bin/python"),
        "-B",
    ]


def _common_data_flags(host: str, bundle_relative: Path) -> list[str]:
    root = REMOTE_ROOTS[host]
    return [
        "--train-dataset",
        str(root / TRAIN_DATASET),
        "--validation-dataset",
        str(root / VALIDATION_DATASET),
        "--r3-cache",
        str(root / R3_CACHE),
        "--relational-cache",
        str(root / RELATIONAL_CACHE),
        "--s1-cache",
        str(root / S1_CACHE),
        "--r6-binary",
        str(root / bundle_relative / "bin" / R6_BINARY_NAME),
        "--warm-start-run-dir",
        str(root / WARM_START_RUN),
        "--warm-start-report",
        str(root / WARM_START_RUN / "final-report.json"),
    ]


def _checkpoint_relative(
    host: str,
    *,
    smoke: bool,
    campaign_root: Path = CAMPAIGN_ROOT,
) -> Path:
    step = SMOKE_STEPS if smoke else 2_000
    run = (
        campaign_root / "smoke" / host
        if smoke
        else campaign_root / "runs" / _slug(_arm_for_host(host))
    )
    return (
        run
        / "checkpoints"
        / f"step-{step:09d}-epoch-0000-batch-{step:06d}"
    )


def _arm_for_host(host: str) -> str:
    return next(arm for arm, assigned in ARM_HOSTS.items() if assigned == host)


def _slug(value: str) -> str:
    return value.replace("-", "_")


def launch_root(launch_id: str | None) -> Path:
    """Return the isolated artifact root for one launch."""
    if launch_id is None:
        return CAMPAIGN_ROOT
    if not LAUNCH_ID_PATTERN.fullmatch(launch_id):
        raise QueueSpecError(
            "launch ID must be one lowercase path component containing only "
            "letters, digits, '.', '_', or '-'"
        )
    return CAMPAIGN_ROOT / "launches" / launch_id


def validate_task_prefix(task_prefix: str) -> str:
    """Reject ambiguous or path-like queue task prefixes."""
    if not LAUNCH_ID_PATTERN.fullmatch(task_prefix):
        raise QueueSpecError(
            "task prefix must start with a lowercase letter or digit and "
            "contain only letters, digits, '.', '_', or '-'"
        )
    return task_prefix


def build_task_specs(
    repository: Path,
    bundle: Path,
    *,
    control_identity: dict[str, Any],
    task_prefix: str = TASK_PREFIX,
    campaign_root: Path = CAMPAIGN_ROOT,
) -> list[dict[str, Any]]:
    """Return the complete nonduplicative scheduler graph."""
    del control_identity
    task_prefix = validate_task_prefix(task_prefix)
    manifest = validate_bundle(bundle)
    if manifest["identity"].get("experiment_id") != EXPERIMENT_ID:
        raise QueueSpecError("ADR 0166 bundle belongs to another experiment")
    bundle_relative = _relative(repository, bundle, "bundle")
    binary = bundle / "bin" / R6_BINARY_NAME
    if not binary.is_file():
        raise QueueSpecError("ADR 0166 bundle lacks the R6 replay binary")

    fanout_id = f"{task_prefix}-bundle-fanout"
    fanout_report = campaign_root / "control" / "bundle-fanout.json"
    fanout_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{_local(bundle_relative)}/",
        "--local-root",
        _local(bundle_relative),
    ]
    for host in ("john2", "john3", "john4"):
        fanout_command.extend(
            [
                "--destination",
                f"{host}:{_remote(host, bundle_relative)}/",
            ]
        )
    fanout_command.extend(
        [
            "--required-file",
            "bundle.json",
            "--required-file",
            f"bin/{R6_BINARY_NAME}",
            "--verify-tree",
            "--output",
            _local(fanout_report),
        ]
    )
    tasks = [
        _task(
            task_id=fanout_id,
            title="Fan out immutable ADR 0166 bundle",
            decision="Bind all four hosts to identical source and R6 bytes",
            workload_class="shared-prerequisite",
            priority=0,
            expected_runtime_seconds=180,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=fanout_command,
            artifact_path=fanout_report,
            stop_rule="Every regular bundle file must match on all four hosts.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    ]

    smoke_ids = []
    for host in REMOTE_ROOTS:
        task_id = f"{task_prefix}-smoke-{host}"
        smoke_ids.append(task_id)
        run = campaign_root / "smoke" / host
        report = run / "report.json"
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Run common opportunity-query smoke on {host}",
                decision="Prove the same graph and batches behave consistently",
                workload_class="replica",
                priority=4,
                expected_runtime_seconds=600,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[
                    fanout_id,
                    "relmlx-c0-run-fanout",
                    HOST_PREREQUISITES[host],
                ],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.opportunity_cross_attention_mlx_train",
                    *_common_data_flags(host, bundle_relative),
                    "--run-dir",
                    _remote(host, run),
                    "--output",
                    _remote(host, report),
                    "--arm",
                    ARMS[0],
                    "--smoke-steps",
                    str(SMOKE_STEPS),
                ],
                artifact_path=report,
                stop_rule=(
                    "Run exactly the registered three-step common arm; "
                    "production remains unauthorized."
                ),
                cpu_cores=10,
                memory_gib=8.0,
                uses_mlx=True,
            )
        )

    smoke_collection = campaign_root / "control" / "smoke-collection.json"
    smoke_collect_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for host in REMOTE_ROOTS:
        remote_report = campaign_root / "smoke" / host / "report.json"
        local_report = (
            campaign_root
            / "control"
            / "smoke-collected"
            / f"{host}-report.json"
        )
        remote_model = (
            _checkpoint_relative(
                host,
                smoke=True,
                campaign_root=campaign_root,
            )
            / "model.safetensors"
        )
        local_model = (
            campaign_root
            / "control"
            / "smoke-collected"
            / f"{host}-model.safetensors"
        )
        smoke_collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, remote_report)}",
                _local(local_report),
                "--artifact",
                f"{host}:{_remote(host, remote_model)}",
                _local(local_model),
            ]
        )
    smoke_collect_command.extend(["--output", _local(smoke_collection)])
    smoke_collect_id = f"{task_prefix}-smoke-collect"
    tasks.append(
        _task(
            task_id=smoke_collect_id,
            title="Collect four common-arm smoke runs",
            decision="Retrieve exact reports and model tensors for parity",
            workload_class="shared-prerequisite",
            priority=5,
            expected_runtime_seconds=120,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=smoke_ids,
            command=smoke_collect_command,
            artifact_path=smoke_collection,
            stop_rule="All eight smoke artifacts must be checksum-verified.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    smoke_proof = campaign_root / "control" / "cross-host-smoke-proof.json"
    smoke_compare_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/opportunity_cross_attention_mlx_smoke_compare.py",
    ]
    for host in REMOTE_ROOTS:
        smoke_compare_command.extend(
            [
                "--report",
                _remote(
                    "john1",
                    campaign_root
                    / "control"
                    / "smoke-collected"
                    / f"{host}-report.json",
                ),
                "--checkpoint",
                (
                    f"{host}="
                    + _remote(
                        "john1",
                        campaign_root
                        / "control"
                        / "smoke-collected"
                        / f"{host}-model.safetensors",
                    )
                ),
            ]
        )
    smoke_compare_command.extend(
        ["--output", _remote("john1", smoke_proof)]
    )
    smoke_compare_id = f"{task_prefix}-smoke-compare"
    tasks.append(
        _task(
            task_id=smoke_compare_id,
            title="Prove four-host opportunity-query parity",
            decision="Authorize launch only after bounded numerical parity",
            workload_class="shared-prerequisite",
            priority=6,
            expected_runtime_seconds=120,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[smoke_collect_id],
            command=smoke_compare_command,
            artifact_path=smoke_proof,
            stop_rule="Any identity or numerical tolerance miss blocks production.",
            cpu_cores=1,
            memory_gib=4.0,
            uses_mlx=True,
        )
    )

    authorization = campaign_root / "control" / "authorization.json"
    authorize_id = f"{task_prefix}-authorize"
    tasks.append(
        _task(
            task_id=authorize_id,
            title="Authorize ADR 0166 production",
            decision="Bind the exact C0, caches, source, smoke, and first batch",
            workload_class="shared-prerequisite",
            priority=7,
            expected_runtime_seconds=900,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[smoke_compare_id],
            command=[
                *_python_prefix("john1", bundle_relative),
                "tools/opportunity_cross_attention_mlx_campaign.py",
                "authorize",
                "--repository",
                _remote("john1", bundle_relative / "source"),
                *_common_data_flags("john1", bundle_relative),
                "--smoke-proof",
                _remote("john1", smoke_proof),
                "--approved-by",
                "codex-goal-019eb3d2",
                "--output",
                _remote("john1", authorization),
            ],
            artifact_path=authorization,
            stop_rule="No production task may start without exact authorization.",
            cpu_cores=4,
            memory_gib=12.0,
            uses_mlx=True,
        )
    )

    control_fanout = campaign_root / "reports" / "control-fanout.json"
    control_fanout_id = f"{task_prefix}-control-fanout"
    tasks.append(
        _task(
            task_id=control_fanout_id,
            title="Fan out ADR 0166 launch controls",
            decision="Install identical authorization and smoke proof everywhere",
            workload_class="shared-prerequisite",
            priority=8,
            expected_runtime_seconds=90,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[authorize_id],
            command=[
                *_python_prefix("john1", bundle_relative),
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{_local(campaign_root / 'control')}/",
                "--local-root",
                _local(campaign_root / "control"),
                *[
                    value
                    for host in ("john2", "john3", "john4")
                    for value in (
                        "--destination",
                        f"{host}:{_remote(host, campaign_root / 'control')}/",
                    )
                ],
                "--required-file",
                "authorization.json",
                "--required-file",
                "cross-host-smoke-proof.json",
                "--verify-tree",
                "--output",
                _local(control_fanout),
            ],
            artifact_path=control_fanout,
            stop_rule="Every launch-control byte must match on all four hosts.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    preflight_ids = []
    for arm in ARMS:
        host = ARM_HOSTS[arm]
        task_id = f"{task_prefix}-preflight-{host}"
        preflight_ids.append(task_id)
        output = campaign_root / "reports" / f"preflight-{host}.json"
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Preflight {arm} on {host}",
                decision="Reconstruct every launch identity before optimization",
                workload_class="shared-prerequisite",
                priority=9,
                expected_runtime_seconds=900,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[control_fanout_id],
                command=[
                    *_python_prefix(host, bundle_relative),
                    "tools/opportunity_cross_attention_mlx_campaign.py",
                    "preflight",
                    "--host",
                    host,
                    "--arm",
                    arm,
                    "--repository",
                    _remote(host, bundle_relative / "source"),
                    *_common_data_flags(host, bundle_relative),
                    "--authorization",
                    _remote(host, authorization),
                    "--smoke-proof",
                    _remote(host, smoke_proof),
                    "--output",
                    _remote(host, output),
                ],
                artifact_path=output,
                stop_rule="Any source, warm-start, data, or host drift blocks the arm.",
                cpu_cores=4,
                memory_gib=12.0,
                uses_mlx=True,
            )
        )

    c0_control = campaign_root / "reports" / "untouched-c0-control.json"
    c0_control_id = f"{task_prefix}-c0-control"
    tasks.append(
        _task(
            task_id=c0_control_id,
            title="Build untouched exact-R2 C0 paired panel",
            decision="Preserve an absolute paired control for every adapter arm",
            workload_class="shared-prerequisite",
            priority=10,
            expected_runtime_seconds=1800,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=preflight_ids,
            command=[
                *_python_prefix("john1", bundle_relative),
                "tools/opportunity_cross_attention_mlx_report.py",
                "build-c0-control",
                "--warm-start-run-dir",
                _remote("john1", WARM_START_RUN),
                "--warm-start-report",
                _remote("john1", WARM_START_RUN / "final-report.json"),
                "--validation-dataset",
                _remote("john1", VALIDATION_DATASET),
                "--r3-cache",
                _remote("john1", R3_CACHE),
                "--relational-cache",
                _remote("john1", RELATIONAL_CACHE),
                "--s1-cache",
                _remote("john1", S1_CACHE),
                "--output",
                _remote("john1", c0_control),
            ],
            artifact_path=c0_control,
            stop_rule="Score every validation decision once with the exact C0 bytes.",
            cpu_cores=10,
            memory_gib=12.0,
            uses_mlx=True,
        )
    )

    run_ids = []
    for arm in ARMS:
        host = ARM_HOSTS[arm]
        slug = _slug(arm)
        task_id = f"{task_prefix}-run-{slug}"
        run_ids.append(task_id)
        run = campaign_root / "runs" / slug
        report = campaign_root / "reports" / f"{slug}.json"
        dependencies = list(preflight_ids)
        if host == "john1":
            dependencies.append(c0_control_id)
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Train ADR 0166 arm {arm}",
                decision="Test one unique opportunity-query assignment",
                workload_class="independent-experiment",
                priority=11,
                expected_runtime_seconds=7200,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=dependencies,
                command=[
                    *_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.opportunity_cross_attention_mlx_train",
                    *_common_data_flags(host, bundle_relative),
                    "--run-dir",
                    _remote(host, run),
                    "--output",
                    _remote(host, report),
                    "--authorization",
                    _remote(host, authorization),
                    "--preflight",
                    _remote(
                        host,
                        campaign_root
                        / "reports"
                        / f"preflight-{host}.json",
                    ),
                    "--arm",
                    arm,
                ],
                artifact_path=report,
                stop_rule=(
                    "Complete exactly 2,000 matched adapter-only steps and "
                    "the full validation/serving evidence."
                ),
                cpu_cores=10,
                memory_gib=12.0,
                uses_mlx=True,
            )
        )

    collection = campaign_root / "reports" / "production-collection.json"
    collect_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/cluster_artifact_collect.py",
    ]
    for arm in ARMS:
        host = ARM_HOSTS[arm]
        slug = _slug(arm)
        report = campaign_root / "reports" / f"{slug}.json"
        local_report = (
            campaign_root / "reports" / "collected" / f"{slug}.json"
        )
        remote_checkpoint = _checkpoint_relative(
            host,
            smoke=False,
            campaign_root=campaign_root,
        )
        local_checkpoint = (
            campaign_root / "reports" / "collected-checkpoints" / slug
        )
        for name in ("checkpoint.json", "model.safetensors"):
            collect_command.extend(
                [
                    "--artifact",
                    f"{host}:{_remote(host, remote_checkpoint / name)}",
                    _local(local_checkpoint / name),
                ]
            )
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, report)}",
                _local(local_report),
            ]
        )
    collect_command.extend(["--output", _local(collection)])
    collect_id = f"{task_prefix}-collect"
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect four ADR 0166 production arms",
            decision="Retrieve exact reports and final model checkpoints",
            workload_class="shared-prerequisite",
            priority=20,
            expected_runtime_seconds=180,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=run_ids,
            command=collect_command,
            artifact_path=collection,
            stop_rule="All reports and checkpoint bytes must verify exactly.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    forward = campaign_root / "aggregate-forward.json"
    reverse = campaign_root / "aggregate-reverse.json"
    order_proof = campaign_root / "order-proof.json"
    classify_command = [
        *_python_prefix("john1", bundle_relative),
        "tools/opportunity_cross_attention_mlx_report.py",
        "classify",
    ]
    for arm in ARMS:
        slug = _slug(arm)
        classify_command.extend(
            [
                "--report",
                _remote(
                    "john1",
                    campaign_root / "reports" / "collected" / f"{slug}.json",
                ),
                "--checkpoint",
                (
                    f"{arm}="
                    + _remote(
                        "john1",
                        campaign_root
                        / "reports"
                        / "collected-checkpoints"
                        / slug,
                    )
                ),
            ]
        )
    classify_command.extend(
        [
            "--untouched-c0",
            _remote("john1", c0_control),
            "--forward-output",
            _remote("john1", forward),
            "--reverse-output",
            _remote("john1", reverse),
            "--order-proof-output",
            _remote("john1", order_proof),
        ]
    )
    tasks.append(
        _task(
            task_id=f"{task_prefix}-classify",
            title="Classify the opportunity-query factorial",
            decision="Advance only a treatment passing every frozen gate",
            workload_class="shared-prerequisite",
            priority=21,
            expected_runtime_seconds=300,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collect_id, c0_control_id],
            command=classify_command,
            artifact_path=forward,
            stop_rule=(
                "Report paired bootstrap, factorial effects, protected slices, "
                "and absolute serving even when no arm advances."
            ),
            cpu_cores=4,
            memory_gib=8.0,
            uses_mlx=False,
        )
    )
    return tasks


def build_queue_spec(
    repository: Path,
    bundle: Path,
    *,
    task_prefix: str = TASK_PREFIX,
    campaign_root: Path = CAMPAIGN_ROOT,
) -> dict[str, Any]:
    control_identity = validate_completed_control(repository)
    tasks = build_task_specs(
        repository,
        bundle,
        control_identity=control_identity,
        task_prefix=task_prefix,
        campaign_root=campaign_root,
    )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "bundle_id": bundle.name,
        "task_prefix": task_prefix,
        "artifact_root": str(campaign_root),
        "warm_start_control_identity": control_identity,
        "maximum_concurrent_primary_experiments": 4,
        "task_count": len(tasks),
        "tasks": tasks,
        "task_spec_blake3": canonical_blake3(tasks),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("build-bundle")
    bundle.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    bundle.add_argument(
        "--r6-binary",
        type=Path,
        default=DEFAULT_R6_BINARY,
    )
    bundle.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
    )

    spec = subparsers.add_parser("build-spec")
    spec.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    spec.add_argument("--bundle", type=Path, required=True)
    spec.add_argument("--task-prefix", default=TASK_PREFIX)
    spec.add_argument("--launch-id")
    spec.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build-bundle":
        path, manifest, reused = build_bundle(
            repository=args.repository,
            experiment_id=EXPERIMENT_ID,
            includes=list(SOURCE_INCLUDES),
            binaries=[args.r6_binary],
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "bundle_id": manifest["bundle_id"],
                    "bundle_path": str(path),
                    "source_files": len(
                        manifest["identity"]["source_files"]
                    ),
                    "binaries": len(manifest["identity"]["binaries"]),
                    "reused": reused,
                },
                sort_keys=True,
            )
        )
        return 0
    campaign_root = launch_root(args.launch_id)
    payload = build_queue_spec(
        args.repository,
        args.bundle,
        task_prefix=args.task_prefix,
        campaign_root=campaign_root,
    )
    output = args.output or (
        DEFAULT_OUTPUT
        if args.launch_id is None
        else campaign_root / "queue-spec.json"
    )
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "bundle_id": payload["bundle_id"],
                "warm_start_report_id": payload[
                    "warm_start_control_identity"
                ]["report_id"],
                "task_count": payload["task_count"],
                "task_spec_blake3": payload["task_spec_blake3"],
                "artifact_root": str(campaign_root),
                "task_prefix": args.task_prefix,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
