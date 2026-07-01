#!/usr/bin/env python3
"""Build the crossed-host closeout queue for the O1 policy corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "o1-opponent-intent-policy-heldout-corpus-v1"
TASK_PREFIX = "o1corpus-v1-closeout"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "audit-bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "closeout-queue-spec.json"
BINARY = Path("target/release/opponent_intent_policy_corpus_audit")
COLLECTION_DEPENDENCY = "o1corpus-v1-collect-manifests"
REPLAY_HOST = "john2"
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
}
DATASET_ROLES = (
    "train-part-0",
    "train-part-1",
    "validation",
    "test",
    "final-stress",
)
REMOTE_DATASET_SOURCES = {
    "train-part-0": (
        "john2",
        Path("artifacts/datasets/o1-opponent-intent-v1-train-part-0"),
    ),
    "train-part-1": (
        "john4",
        Path("artifacts/datasets/o1-opponent-intent-v1-train-part-1"),
    ),
    "validation": (
        "john1",
        Path("artifacts/datasets/o1-opponent-intent-v1-validation"),
    ),
    "test": (
        "john2",
        Path("artifacts/datasets/o1-opponent-intent-v1-test"),
    ),
    "final-stress": (
        "john4",
        Path("artifacts/datasets/o1-opponent-intent-v1-final-stress"),
    ),
}
SOURCE_INCLUDES = (
    Path("CASCADIA_V2_GOAL.txt"),
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("crates/cascadia-data"),
    Path("crates/cascadia-game"),
    Path("crates/cascadia-provenance"),
    Path("crates/cascadia-sim"),
    Path("docs/v2/CLI_REFERENCE.md"),
    Path("docs/v2/DATA_FORMAT.md"),
    Path("docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md"),
    Path("docs/v2/decisions/0185-o1-policy-held-out-sequential-corpus.md"),
    Path("docs/v2/decisions/0186-o1-policy-corpus-audit-and-scoped-authorization.md"),
    Path("docs/v2/reports/o1-opponent-intent-policy-heldout-corpus-v1-preregistration.md"),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/cluster_artifact_tree_collect.py"),
    Path("tools/o1_opponent_intent_policy_corpus_closeout_queue.py"),
    Path("tools/o1_opponent_intent_policy_corpus_report.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/test_cluster_artifact_tree_collect.py"),
    Path("tools/test_o1_opponent_intent_policy_corpus_closeout_queue.py"),
    Path("tools/test_o1_opponent_intent_policy_corpus_report.py"),
)


class O1CorpusCloseoutQueueError(RuntimeError):
    """Raised when the O1 closeout campaign is malformed."""


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
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: Path,
    stop_rule: str,
    expected_runtime_seconds: int,
    cpu_cores: int,
    decision_terminal: bool = False,
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
            "memory_gib": 4.0,
            "uses_mlx": False,
        },
    }


def _audit_command(
    *,
    host: str,
    bundle_relative: Path,
    output: Path,
) -> list[str]:
    command = [
        "/usr/bin/env",
        "-C",
        _remote(host, bundle_relative / "source"),
        _remote(host, bundle_relative / "bin/opponent_intent_policy_corpus_audit"),
    ]
    for role in DATASET_ROLES:
        command.extend(
            [
                "--dataset",
                f"{role}={_remote(host, CAMPAIGN_ROOT / 'datasets' / role)}",
            ]
        )
    command.extend(["--output", _remote(host, output)])
    return command


def task_specs(*, bundle_relative: Path, bundle_id: str) -> list[dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise O1CorpusCloseoutQueueError("bundle path must be repository-relative")
    if len(bundle_id) != 64:
        raise O1CorpusCloseoutQueueError("bundle ID must be a 64-character digest")

    launch = CAMPAIGN_ROOT / "closeout-launches" / bundle_id
    tree_report = launch / "control/dataset-tree-collection.json"
    tree_command = [
        ".venv/bin/python",
        str(bundle_relative / "source/tools/cluster_artifact_tree_collect.py"),
    ]
    for role in DATASET_ROLES:
        source_host, source_relative = REMOTE_DATASET_SOURCES[role]
        source_root = (
            REMOTE_ROOTS.get(source_host, Path(f"/Users/{source_host}/cascadia-bench"))
            / source_relative
        )
        tree_command.extend(
            [
                "--tree",
                f"{source_host}:{source_root}/",
                str(CAMPAIGN_ROOT / "datasets" / role),
            ]
        )
    tree_command.extend(["--output", str(tree_report)])
    tree_id = f"{TASK_PREFIX}-collect-trees"

    bundle_fanout_report = launch / "control/audit-bundle-fanout.json"
    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    bundle_fanout_command = [
        ".venv/bin/python",
        str(bundle_relative / "source/tools/cluster_artifact_fanout.py"),
        "--source",
        f"{bundle_relative}/",
        "--local-root",
        str(bundle_relative),
        "--destination",
        f"{REPLAY_HOST}:{_remote(REPLAY_HOST, bundle_relative)}/",
        "--required-file",
        "bundle.json",
        "--required-file",
        "bin/opponent_intent_policy_corpus_audit",
        "--verify-tree",
        "--output",
        str(bundle_fanout_report),
    ]

    dataset_fanout_report = launch / "control/dataset-fanout.json"
    dataset_fanout_id = f"{TASK_PREFIX}-dataset-fanout"
    dataset_fanout_command = [
        ".venv/bin/python",
        str(bundle_relative / "source/tools/cluster_artifact_fanout.py"),
        "--source",
        f"{CAMPAIGN_ROOT}/datasets/",
        "--local-root",
        str(CAMPAIGN_ROOT / "datasets"),
        "--destination",
        f"{REPLAY_HOST}:{_remote(REPLAY_HOST, CAMPAIGN_ROOT / 'datasets')}/",
        "--verify-tree",
        "--output",
        str(dataset_fanout_report),
    ]

    primary_output = launch / "runs/john1-primary.json"
    primary_id = f"{TASK_PREFIX}-primary"
    replay_output = launch / f"runs/{REPLAY_HOST}-replay.json"
    replay_id = f"{TASK_PREFIX}-replay"
    collected_replay = launch / f"collected/{REPLAY_HOST}-replay.json"
    collect_report = launch / "control/replay-collection.json"
    collect_id = f"{TASK_PREFIX}-collect-replay"
    classify_output = launch / "classification.json"
    classify_id = f"{TASK_PREFIX}-classify"

    return [
        _task(
            task_id=tree_id,
            title="Collect all five immutable O1 dataset trees",
            decision="Prove every production shard is present unchanged on john1",
            workload_class="shared-prerequisite",
            priority=84,
            compatible_hosts=["john1"],
            dependencies=[COLLECTION_DEPENDENCY],
            command=tree_command,
            artifact_path=tree_report,
            stop_rule="All five source trees must remain stable and match locally file for file.",
            expected_runtime_seconds=180,
            cpu_cores=1,
        ),
        _task(
            task_id=bundle_fanout_id,
            title="Fan out immutable O1 audit bundle",
            decision="Bind john1 and john2 to one auditor and classifier source",
            workload_class="shared-prerequisite",
            priority=84,
            compatible_hosts=["john1"],
            dependencies=[],
            command=bundle_fanout_command,
            artifact_path=bundle_fanout_report,
            stop_rule="The complete audit bundle must match on john2.",
            expected_runtime_seconds=120,
            cpu_cores=1,
        ),
        _task(
            task_id=dataset_fanout_id,
            title="Fan out canonical O1 corpus to john2",
            decision="Create an exact independent-host replay input",
            workload_class="shared-prerequisite",
            priority=85,
            compatible_hosts=["john1"],
            dependencies=[tree_id],
            command=dataset_fanout_command,
            artifact_path=dataset_fanout_report,
            stop_rule="Every dataset manifest and shard must match on john2.",
            expected_runtime_seconds=240,
            cpu_cores=1,
        ),
        _task(
            task_id=primary_id,
            title="Audit O1 policy corpus on john1",
            decision="Measure exact support, leakage, overlap, and policy boundaries",
            workload_class="independent-experiment",
            priority=85,
            compatible_hosts=["john1"],
            dependencies=[tree_id],
            command=_audit_command(
                host="john1",
                bundle_relative=bundle_relative,
                output=primary_output,
            ),
            artifact_path=primary_output,
            stop_rule="Write one complete five-corpus scientific audit and stop.",
            expected_runtime_seconds=120,
            cpu_cores=4,
        ),
        _task(
            task_id=replay_id,
            title="Replay O1 policy-corpus audit on john2",
            decision="Require a distinct host to reproduce the complete audit",
            workload_class="replica",
            priority=86,
            compatible_hosts=[REPLAY_HOST],
            dependencies=[bundle_fanout_id, dataset_fanout_id],
            command=_audit_command(
                host=REPLAY_HOST,
                bundle_relative=bundle_relative,
                output=replay_output,
            ),
            artifact_path=replay_output,
            stop_rule="Reproduce one complete scientific identity from exact copied inputs.",
            expected_runtime_seconds=120,
            cpu_cores=4,
        ),
        _task(
            task_id=collect_id,
            title="Collect john2 O1 audit replay",
            decision="Bring the replay report under coordinator checksum custody",
            workload_class="shared-prerequisite",
            priority=87,
            compatible_hosts=["john1"],
            dependencies=[replay_id],
            command=[
                ".venv/bin/python",
                str(bundle_relative / "source/tools/cluster_artifact_collect.py"),
                "--artifact",
                f"{REPLAY_HOST}:{_remote(REPLAY_HOST, replay_output)}",
                str(collected_replay),
                "--output",
                str(collect_report),
            ],
            artifact_path=collect_report,
            stop_rule="The replay report must be present locally with a matching SHA-256.",
            expected_runtime_seconds=60,
            cpu_cores=1,
        ),
        _task(
            task_id=classify_id,
            title="Classify O1 policy-held-out corpus",
            decision="Authorize only the exact supported MLX learning scope",
            workload_class="shared-prerequisite",
            priority=88,
            compatible_hosts=["john1"],
            dependencies=[primary_id, collect_id],
            command=[
                ".venv/bin/python",
                str(bundle_relative / "source/tools/o1_opponent_intent_policy_corpus_report.py"),
                "--primary",
                str(primary_output),
                "--replay",
                str(collected_replay),
                "--output",
                str(classify_output),
                "--canonical-output",
                str(CAMPAIGN_ROOT / "classification.json"),
            ],
            artifact_path=classify_output,
            stop_rule=(
                "Primary and replay must match exactly; unsupported paid-wipe, "
                "strategy-switch, champion, and gameplay claims must remain false."
            ),
            expected_runtime_seconds=60,
            cpu_cores=1,
            decision_terminal=True,
        ),
    ]


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
    build = subparsers.add_parser("build-bundle")
    build.add_argument("--repository", type=Path, default=Path.cwd())
    build.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    specification = subparsers.add_parser("build-spec")
    specification.add_argument("--repository", type=Path, default=Path.cwd())
    specification.add_argument("--bundle", type=Path, required=True)
    specification.add_argument("--output", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()

    if args.command == "build-bundle":
        if not (args.repository / BINARY).is_file():
            raise O1CorpusCloseoutQueueError(f"build the release auditor first: {BINARY}")
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
        raise O1CorpusCloseoutQueueError("bundle belongs to another experiment")
    try:
        relative = args.bundle.resolve().relative_to(args.repository.resolve())
    except ValueError as error:
        raise O1CorpusCloseoutQueueError("bundle must remain beneath the repository") from error
    tasks = task_specs(
        bundle_relative=relative,
        bundle_id=manifest["bundle_id"],
    )
    _write_json(
        args.output,
        campaign_spec(tasks, bundle_id=manifest["bundle_id"]),
    )
    print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
