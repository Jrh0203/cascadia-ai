#!/usr/bin/env python3
"""Freeze and describe the four-host R3 action-edit census campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import blake3
from cluster_research_queue import add_task, empty_queue, validate_queue
from rust_experiment_bundle import BundleError, validate_bundle

EXPERIMENT_ID = "r3-action-edit-foundation-v1"
PROTOCOL_ID = "r3-action-edit-open-corpus-v1"
BINARY_NAME = "r3-action-edit-census"
TASK_PREFIX = "r3aef"
HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
PRODUCTION_TRAIN_FIRST_SEED = 3_300_000
PRODUCTION_TRAIN_GAMES = 16
PRODUCTION_VALIDATION_FIRST_SEED = 3_400_000
PRODUCTION_VALIDATION_GAMES = 4
PRODUCTION_SHARD_COUNT = 4
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "Cargo.lock",
    "Cargo.toml",
    "docs/v2/decisions/0148-r3-exact-action-local-patch-global-edit-foundation.md",
    "docs/v2/reports/r3-action-edit-foundation-v1-invalid-smoke-1.md",
    "docs/v2/reports/r3-action-edit-foundation-v1-preregistration.md",
    "tools/cluster_artifact_collect.py",
    "tools/cluster_artifact_fanout.py",
    "tools/cluster_research_queue.py",
    "tools/r3_action_edit_campaign.py",
    "tools/r3_action_edit_census/Cargo.lock",
    "tools/r3_action_edit_census/Cargo.toml",
    "tools/r3_action_edit_census/README.md",
    "tools/rust_experiment_bundle.py",
    "tools/test_r3_action_edit_campaign.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "crates/cascadia-data/",
    "crates/cascadia-game/",
    "crates/cascadia-provenance/",
    "crates/cascadia-sim/",
    "tools/r2_sparse_entity_census/src/",
    "tools/r3_action_edit_census/r2_public_adapter/",
    "tools/r3_action_edit_census/src/",
    "tools/r3_action_edit_census/tests/",
)


class CampaignError(RuntimeError):
    """The R3 campaign cannot proceed without changing scientific identity."""


def validate_bundle_for_campaign(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = validate_bundle(bundle)
    except BundleError as error:
        raise CampaignError(str(error)) from error
    if manifest.get("identity", {}).get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("R3 immutable bundle names the wrong experiment")
    source_entries = manifest.get("identity", {}).get("source_files", [])
    paths = {
        entry.get("path")
        for entry in source_entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing_files = sorted(REQUIRED_SOURCE_FILES - paths)
    missing_prefixes = sorted(
        prefix
        for prefix in REQUIRED_SOURCE_PREFIXES
        if not any(path.startswith(prefix) for path in paths)
    )
    binaries = {
        entry.get("name"): entry
        for entry in manifest.get("identity", {}).get("binaries", [])
        if isinstance(entry, dict)
    }
    if missing_files or missing_prefixes or BINARY_NAME not in binaries:
        raise CampaignError(
            "R3 immutable bundle is incomplete: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}, "
            f"missing_binary={BINARY_NAME not in binaries}"
        )
    runtime = bundle_runtime_identity(bundle)
    if runtime["executable_blake3"] != binaries[BINARY_NAME].get("blake3"):
        raise CampaignError("R3 runtime executable differs from the immutable bundle manifest")
    return manifest, runtime


def bundle_runtime_identity(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    binary = bundle / "bin" / BINARY_NAME
    source = bundle / "source"
    environment = os.environ.copy()
    environment["R3_SOURCE_ROOT"] = str(source)
    completed = subprocess.run(
        [str(binary), "identity"],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CampaignError(
            "R3 immutable runtime identity failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise CampaignError("R3 immutable runtime identity is not JSON") from error
    source_identity = identity.get("source") if isinstance(identity, dict) else None
    source_blake3 = (
        source_identity.get("source_bundle_blake3")
        if isinstance(source_identity, dict)
        else None
    )
    executable_blake3 = (
        identity.get("executable_blake3") if isinstance(identity, dict) else None
    )
    if not _is_blake3(source_blake3) or not _is_blake3(executable_blake3):
        raise CampaignError("R3 immutable runtime identity is malformed")
    return identity


def build_task_specs(
    *,
    repository: Path,
    bundle: Path,
    source_bundle_blake3: str,
    executable_blake3: str,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> list[dict[str, Any]]:
    if not _is_blake3(source_bundle_blake3) or not _is_blake3(executable_blake3):
        raise CampaignError("R3 queue requires lowercase source and executable BLAKE3 values")
    repository = repository.resolve()
    bundle_relative = _relative(repository, bundle, "bundle")
    experiment_relative = _relative(repository, experiment_root, "experiment root")
    specs: list[dict[str, Any]] = []

    fanout_id = f"{TASK_PREFIX}-fanout-bundle"
    specs.append(
        _task(
            task_id=fanout_id,
            title="Fan out immutable R3 bundle",
            decision="Make the exact R3 source and executable byte-identical on all hosts",
            workload_class="shared-prerequisite",
            priority=1,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=[
                ".venv/bin/python",
                "-B",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{bundle_relative}/",
                "--local-root",
                str(bundle_relative),
                *[
                    item
                    for host in HOSTS[1:]
                    for item in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / bundle_relative}/",
                    )
                ],
                "--required-file",
                "bundle.json",
                "--required-file",
                f"bin/{BINARY_NAME}",
                "--verify-tree",
                "--output",
                str(experiment_relative / "reports/fanout-bundle.json"),
            ],
            artifact_path=str(experiment_relative / "reports/fanout-bundle.json"),
            stop_rule="Every bundled source and executable byte must match on all hosts.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    preflight_ids: list[str] = []
    for host in HOSTS:
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids.append(task_id)
        output = experiment_relative / f"reports/preflight-{host}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Verify immutable R3 runtime on {host}",
                decision="Reject source or executable drift before any corpus work",
                workload_class="shared-prerequisite",
                priority=5,
                expected_runtime_seconds=30,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[fanout_id],
                command=[
                    *_frozen_binary_prefix(host, bundle_relative),
                    "identity",
                    "--expected-source-bundle-blake3",
                    source_bundle_blake3,
                    "--expected-executable-blake3",
                    executable_blake3,
                    "--output",
                    str(REMOTE_ROOTS[host] / output),
                ],
                artifact_path=str(output),
                stop_rule="The host must reproduce both frozen BLAKE3 identities exactly.",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )

    shard_ids: list[str] = []
    for shard_index, host in enumerate(HOSTS):
        task_id = f"{TASK_PREFIX}-shard-{shard_index}"
        shard_ids.append(task_id)
        output = experiment_relative / f"reports/shard-{shard_index}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Run R3 production shard {shard_index} on {host}",
                decision=(
                    f"Verify every exact action edit for modulo-owned shard {shard_index}"
                ),
                workload_class="divisible-evidence",
                priority=10,
                expected_runtime_seconds=14_400,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=preflight_ids,
                command=[
                    *_frozen_binary_prefix(host, bundle_relative, rayon_threads=5),
                    "census",
                    "--train-first-seed",
                    str(PRODUCTION_TRAIN_FIRST_SEED),
                    "--train-games",
                    str(PRODUCTION_TRAIN_GAMES),
                    "--validation-first-seed",
                    str(PRODUCTION_VALIDATION_FIRST_SEED),
                    "--validation-games",
                    str(PRODUCTION_VALIDATION_GAMES),
                    "--paid-wipe-sentinels",
                    "true",
                    "--d6-sentinel-per-position",
                    "true",
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(PRODUCTION_SHARD_COUNT),
                    "--output",
                    str(REMOTE_ROOTS[host] / output),
                ],
                artifact_path=str(output),
                stop_rule=(
                    "All owned seeds, actions, codecs, public successors, supply deltas, "
                    "global edits, and D6 sentinels must pass."
                ),
                cpu_cores=5,
                memory_gib=8.0,
                uses_mlx=False,
            )
        )

    collected = experiment_relative / "reports/collected"
    collection_report = experiment_relative / "reports/collection.json"
    collect_command = [
        ".venv/bin/python",
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
    for shard_index, host in enumerate(HOSTS):
        remote_report = REMOTE_ROOTS[host] / experiment_relative / (
            f"reports/shard-{shard_index}.json"
        )
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{remote_report}",
                str(collected / f"shard-{shard_index}.json"),
            ]
        )
    collect_command.extend(["--output", str(collection_report)])
    collection_id = f"{TASK_PREFIX}-collect"
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect four R3 shard reports",
            decision="Checksum-copy exactly one nonoverlapping shard from every host",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=shard_ids,
            command=collect_command,
            artifact_path=str(collection_report),
            stop_rule="All four local files must match the producing hosts exactly.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    aggregate_ids: list[str] = []
    for order in ("forward", "reverse"):
        task_id = f"{TASK_PREFIX}-aggregate-{order}"
        aggregate_ids.append(task_id)
        shard_order = range(PRODUCTION_SHARD_COUNT)
        if order == "reverse":
            shard_order = reversed(range(PRODUCTION_SHARD_COUNT))
        output = experiment_relative / f"reports/aggregate-{order}.json"
        command = [*_frozen_binary_prefix("john1", bundle_relative), "aggregate"]
        for shard_index in shard_order:
            command.extend(
                [
                    "--input",
                    str(
                        REMOTE_ROOTS["john1"]
                        / collected
                        / f"shard-{shard_index}.json"
                    ),
                ]
            )
        command.extend(["--output", str(REMOTE_ROOTS["john1"] / output)])
        specs.append(
            _task(
                task_id=task_id,
                title=f"Aggregate R3 shards in {order} order",
                decision="Recompute exact corpus distributions and every promotion gate",
                workload_class="replica",
                priority=40,
                expected_runtime_seconds=30,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=command,
                artifact_path=str(output),
                stop_rule="Accept exactly shards 0..3 and emit one deterministic aggregate.",
                cpu_cores=1,
                memory_gib=2.0,
                uses_mlx=False,
            )
        )

    proof = experiment_relative / "reports/aggregate-order-proof.json"
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-aggregate-order-proof",
            title="Prove R3 aggregate order invariance",
            decision="Require byte-identical forward and reverse scientific aggregates",
            workload_class="replica",
            priority=50,
            expected_runtime_seconds=10,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=aggregate_ids,
            command=[
                *_frozen_binary_prefix("john1", bundle_relative),
                "prove-order",
                "--forward",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/aggregate-forward.json"
                ),
                "--reverse",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/aggregate-reverse.json"
                ),
                "--output",
                str(REMOTE_ROOTS["john1"] / proof),
            ],
            artifact_path=str(proof),
            stop_rule="Forward and reverse aggregate files must be byte-identical.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    _validate_task_specs(specs)
    return specs


def queue_specification(
    specs: list[dict[str, Any]],
    *,
    bundle_id: str,
    source_bundle_blake3: str,
    executable_blake3: str,
) -> dict[str, Any]:
    state = empty_queue(EXPERIMENT_ID, now_ms=0)
    for index, specification in enumerate(specs, start=1):
        add_task(state, specification, now_ms=index)
    validate_queue(state)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "bundle_id": bundle_id,
        "source_bundle_blake3": source_bundle_blake3,
        "executable_blake3": executable_blake3,
        "tasks": specs,
    }
    return {
        "schema_version": 1,
        **identity,
        "task_count": len(specs),
        "task_spec_blake3": canonical_blake3(identity),
        "applied": False,
        "installation_supported_by_this_tool": False,
        "live_queue_path": None,
        "validated_queue_preview": state,
    }


def _validate_task_specs(specs: list[dict[str, Any]]) -> None:
    identifiers = [spec["id"] for spec in specs]
    if len(specs) != 13 or len(identifiers) != len(set(identifiers)):
        raise CampaignError("R3 queue graph must contain 13 uniquely named tasks")
    known = set(identifiers)
    for spec in specs:
        unknown = set(spec["dependencies"]) - known
        if unknown:
            raise CampaignError(
                f"R3 task {spec['id']} has unknown dependencies: {sorted(unknown)}"
            )
        if any("python" in item for item in spec["command"]) and "-B" not in spec["command"]:
            raise CampaignError(f"R3 Python task omits -B: {spec['id']}")
    preflights = {f"{TASK_PREFIX}-preflight-{host}" for host in HOSTS}
    for shard_index, host in enumerate(HOSTS):
        task = next(
            spec for spec in specs if spec["id"] == f"{TASK_PREFIX}-shard-{shard_index}"
        )
        if task["compatible_hosts"] != [host] or set(task["dependencies"]) != preflights:
            raise CampaignError(f"R3 shard {shard_index} is not pinned and fully preflighted")
        command = task["command"]
        if command[command.index("--shard-index") + 1] != str(shard_index):
            raise CampaignError(f"R3 shard {shard_index} command has the wrong index")
        if command[command.index("--shard-count") + 1] != str(PRODUCTION_SHARD_COUNT):
            raise CampaignError(f"R3 shard {shard_index} command has the wrong shard count")


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
            "uses_mlx": uses_mlx,
        },
    }


def _frozen_binary_prefix(
    host: str,
    bundle_relative: Path,
    *,
    rayon_threads: int | None = None,
) -> list[str]:
    bundle = REMOTE_ROOTS[host] / bundle_relative
    prefix = [
        "/usr/bin/env",
        f"R3_SOURCE_ROOT={bundle / 'source'}",
    ]
    if rayon_threads is not None:
        prefix.append(f"RAYON_NUM_THREADS={rayon_threads}")
    prefix.append(str(bundle / "bin" / BINARY_NAME))
    return prefix


def _relative(repository: Path, path: Path, label: str) -> Path:
    try:
        relative = path.resolve().relative_to(repository)
    except ValueError as error:
        raise CampaignError(f"R3 {label} must remain beneath the repository") from error
    if not relative.parts:
        raise CampaignError(f"R3 {label} cannot be the repository root")
    return relative


def canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _is_blake3(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    queue = subparsers.add_parser("queue-spec")
    queue.add_argument("--repository", type=Path, default=Path("."))
    queue.add_argument("--bundle", type=Path, required=True)
    queue.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    queue.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, runtime = validate_bundle_for_campaign(args.bundle)
        source_bundle_blake3 = runtime["source"]["source_bundle_blake3"]
        executable_blake3 = runtime["executable_blake3"]
        specs = build_task_specs(
            repository=args.repository,
            bundle=args.bundle,
            source_bundle_blake3=source_bundle_blake3,
            executable_blake3=executable_blake3,
            experiment_root=args.experiment_root,
        )
        report = queue_specification(
            specs,
            bundle_id=manifest["bundle_id"],
            source_bundle_blake3=source_bundle_blake3,
            executable_blake3=executable_blake3,
        )
        _write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (CampaignError, BundleError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
