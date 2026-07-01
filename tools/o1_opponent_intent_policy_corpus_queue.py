#!/usr/bin/env python3
"""Build the immutable O1 policy-held-out sequential-corpus campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rust_experiment_bundle import build_bundle, validate_bundle

EXPERIMENT_ID = "o1-opponent-intent-policy-heldout-corpus-v1"
TASK_PREFIX = "o1corpus-v1"
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_BUNDLE_ROOT = CAMPAIGN_ROOT / "bundles"
DEFAULT_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
BINARY = Path("target/release/opponent_intent_collect")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
DATASETS = {
    "train-part-0": Path("artifacts/datasets/o1-opponent-intent-v1-train-part-0"),
    "train-part-1": Path("artifacts/datasets/o1-opponent-intent-v1-train-part-1"),
    "validation": Path("artifacts/datasets/o1-opponent-intent-v1-validation"),
    "test": Path("artifacts/datasets/o1-opponent-intent-v1-test"),
    "final-stress": Path("artifacts/datasets/o1-opponent-intent-v1-final-stress"),
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
    Path("docs/v2/reports/o1-opponent-intent-policy-heldout-corpus-v1-preregistration.md"),
    Path("tools/cluster_artifact_collect.py"),
    Path("tools/cluster_artifact_fanout.py"),
    Path("tools/o1_opponent_intent_policy_corpus_queue.py"),
    Path("tools/rust_experiment_bundle.py"),
    Path("tools/test_o1_opponent_intent_policy_corpus_queue.py"),
)


class O1CorpusQueueError(RuntimeError):
    """Raised when the O1 corpus campaign is malformed."""


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
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 0.95,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": True,
        "decision_terminal": False,
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


def _collection_task(
    *,
    role: str,
    host: str,
    bundle_relative: Path,
    split: str,
    first_game_index: int,
    games: int,
    cohort_id: str,
    policy_pool: str,
    required_policy: str | None,
    dependencies: list[str],
    priority: int,
) -> dict[str, Any]:
    dataset = DATASETS[role]
    command = [
        "/usr/bin/env",
        "-C",
        _remote(host, bundle_relative / "source"),
        _remote(host, bundle_relative / "bin/opponent_intent_collect"),
        "collect",
        "--output",
        _remote(host, dataset),
        "--split",
        split,
        "--first-game-index",
        str(first_game_index),
        "--games",
        str(games),
        "--shard-games",
        "16",
        "--cohort-id",
        cohort_id,
        "--policy-pool",
        policy_pool,
    ]
    if required_policy is not None:
        command.extend(["--required-policy", required_policy])
    return _task(
        task_id=f"{TASK_PREFIX}-collect-{role}",
        title=f"Collect O1 {role} sequential corpus on {host}",
        decision="Create nonoverlapping exact opponent-intent windows",
        workload_class="divisible-evidence",
        priority=priority,
        compatible_hosts=[host],
        dependencies=dependencies,
        command=command,
        artifact_path=dataset / "dataset.json",
        stop_rule=(
            "Every game must produce 76 validated windows; any shard, policy, "
            "history, target, or checksum mismatch fails."
        ),
        expected_runtime_seconds=max(180, games * 2),
        cpu_cores=10,
    )


def task_specs(*, bundle_relative: Path, bundle_id: str) -> list[dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise O1CorpusQueueError("bundle path must be repository-relative")
    if len(bundle_id) != 64:
        raise O1CorpusQueueError("bundle ID must be a 64-character digest")

    fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    fanout_report = CAMPAIGN_ROOT / "control/bundle-fanout.json"
    fanout_command = [
        ".venv/bin/python",
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{bundle_relative}/",
        "--local-root",
        str(bundle_relative),
    ]
    for host in ("john2", "john4"):
        fanout_command.extend(["--destination", f"{host}:{_remote(host, bundle_relative)}/"])
    fanout_command.extend(
        [
            "--required-file",
            "bundle.json",
            "--required-file",
            "bin/opponent_intent_collect",
            "--verify-tree",
            "--output",
            str(fanout_report),
        ]
    )
    tasks = [
        _task(
            task_id=fanout_id,
            title="Fan out immutable O1 policy-corpus bundle",
            decision="Bind john1, john2, and john4 to one collector implementation",
            workload_class="shared-prerequisite",
            priority=80,
            compatible_hosts=["john1"],
            dependencies=[],
            command=fanout_command,
            artifact_path=fanout_report,
            stop_rule="Every regular file must match on both remote hosts.",
            expected_runtime_seconds=180,
            cpu_cores=1,
        )
    ]
    train_pool = "greedy,pattern-aware,pattern-commitment"
    validation_pool = f"{train_pool},pattern-competition"
    test_pool = f"{validation_pool},pattern-portfolio"
    tasks.extend(
        [
            _collection_task(
                role="train-part-0",
                host="john2",
                bundle_relative=bundle_relative,
                split="train",
                first_game_index=0,
                games=512,
                cohort_id="o1-train-mixed-v1",
                policy_pool=train_pool,
                required_policy=None,
                dependencies=[fanout_id],
                priority=81,
            ),
            _collection_task(
                role="train-part-1",
                host="john4",
                bundle_relative=bundle_relative,
                split="train",
                first_game_index=512,
                games=512,
                cohort_id="o1-train-mixed-v1",
                policy_pool=train_pool,
                required_policy=None,
                dependencies=[fanout_id],
                priority=81,
            ),
            _collection_task(
                role="validation",
                host="john1",
                bundle_relative=bundle_relative,
                split="validation",
                first_game_index=100_000,
                games=256,
                cohort_id="o1-validation-heldout-competition-v1",
                policy_pool=validation_pool,
                required_policy="pattern-competition",
                dependencies=[fanout_id],
                priority=81,
            ),
            _collection_task(
                role="test",
                host="john2",
                bundle_relative=bundle_relative,
                split="test",
                first_game_index=200_000,
                games=256,
                cohort_id="o1-test-heldout-portfolio-v1",
                policy_pool=test_pool,
                required_policy="pattern-portfolio",
                dependencies=[f"{TASK_PREFIX}-collect-train-part-0"],
                priority=82,
            ),
            _collection_task(
                role="final-stress",
                host="john4",
                bundle_relative=bundle_relative,
                split="final",
                first_game_index=300_000,
                games=128,
                cohort_id="o1-final-heldout-random-v1",
                policy_pool=f"random,{test_pool}",
                required_policy="random",
                dependencies=[f"{TASK_PREFIX}-collect-train-part-1"],
                priority=82,
            ),
        ]
    )

    manifest_report = CAMPAIGN_ROOT / "control/manifest-collection.json"
    manifest_command = [
        ".venv/bin/python",
        "tools/cluster_artifact_collect.py",
    ]
    sources = (
        ("john2", "train-part-0"),
        ("john4", "train-part-1"),
        ("john1", "validation"),
        ("john2", "test"),
        ("john4", "final-stress"),
    )
    for host, role in sources:
        source = _remote(host, DATASETS[role] / "dataset.json")
        destination = CAMPAIGN_ROOT / "manifests" / f"{role}.json"
        manifest_command.extend(["--artifact", f"{host}:{source}", str(destination)])
    manifest_command.extend(["--output", str(manifest_report)])
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-collect-manifests",
            title="Collect all O1 corpus manifests",
            decision="Bind every nonduplicative corpus shard before audit",
            workload_class="shared-prerequisite",
            priority=83,
            compatible_hosts=["john1"],
            dependencies=[
                f"{TASK_PREFIX}-collect-train-part-0",
                f"{TASK_PREFIX}-collect-train-part-1",
                f"{TASK_PREFIX}-collect-validation",
                f"{TASK_PREFIX}-collect-test",
                f"{TASK_PREFIX}-collect-final-stress",
            ],
            command=manifest_command,
            artifact_path=manifest_report,
            stop_rule="All five manifests must transfer with exact SHA-256 receipts.",
            expected_runtime_seconds=120,
            cpu_cores=1,
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
            raise O1CorpusQueueError(f"build the release collector first: {BINARY}")
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
        raise O1CorpusQueueError("bundle belongs to another experiment")
    try:
        relative = args.bundle.resolve().relative_to(args.repository.resolve())
    except ValueError as error:
        raise O1CorpusQueueError("bundle must remain beneath the repository") from error
    tasks = task_specs(bundle_relative=relative, bundle_id=manifest["bundle_id"])
    _write_json(args.output, campaign_spec(tasks, bundle_id=manifest["bundle_id"]))
    print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
