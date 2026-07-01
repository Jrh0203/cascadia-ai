#!/usr/bin/env python3
"""Build the reviewed-only, source-frozen S1 semantic supply queue specification."""

# ruff: noqa: UP045 - cluster tools must run under macOS system Python 3.9.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from cluster_research_queue import add_task, locked_queue
from rust_experiment_bundle import BundleError, validate_bundle

EXPERIMENT_ID = "exact-semantic-supply-v1"
TASK_PREFIX = "s1ss"
HOSTS = ("john1", "john2", "john3", "john4")
SHARD_COUNT = len(HOSTS)
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
TRAIN_FIRST_GAME_INDEX = 320_000
TRAIN_GAMES = 400
VALIDATION_FIRST_GAME_INDEX = 321_000
VALIDATION_GAMES = 100
STRATEGY = "pattern-aware"
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "Cargo.lock",
    "Cargo.toml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "tools/s1_semantic_supply_merge.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "apps/web/src/",
    "crates/cascadia-api/",
    "crates/cascadia-cli-v2/",
    "crates/cascadia-data/",
    "crates/cascadia-differential/",
    "crates/cascadia-eval/",
    "crates/cascadia-game/",
    "crates/cascadia-model/",
    "crates/cascadia-provenance/",
    "crates/cascadia-search/",
    "crates/cascadia-sim/",
    "legacy/crates/cascadia-ai/",
    "legacy/crates/cascadia-core/",
    "python/cascadia_mlx/",
)


class CampaignError(RuntimeError):
    """Raised when the S1 execution graph is incomplete or ambiguous."""


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
            "uses_mlx": False,
        },
    }


def _relative_bundle_path(repository: Path, bundle: Path) -> Path:
    repository = repository.resolve()
    bundle = bundle.resolve()
    try:
        relative = bundle.relative_to(repository)
    except ValueError as error:
        raise CampaignError("bundle must remain beneath the repository") from error
    if ".." in relative.parts:
        raise CampaignError("bundle path escapes the repository")
    return relative


def _remote_path(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _frozen_binary_command(host: str, bundle_relative: Path) -> list[str]:
    return [
        "/usr/bin/env",
        "-C",
        _remote_path(host, bundle_relative / "source"),
        _remote_path(host, bundle_relative / "bin" / "exact_semantic_supply_census"),
    ]


def _frozen_merge_command(bundle_relative: Path) -> list[str]:
    return [
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        _remote_path(
            "john1",
            bundle_relative / "source/tools/s1_semantic_supply_merge.py",
        ),
    ]


def _repeated_flag(flag: str, values: list[str]) -> list[str]:
    return [item for value in values for item in (flag, value)]


def validate_provenance_source_bundle(manifest: dict[str, Any]) -> None:
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
    if missing_files or missing_prefixes:
        raise CampaignError(
            "bundle cannot reproduce S1 source identity: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}"
        )


def census_report_path(split: str, shard_index: int) -> Path:
    return DEFAULT_EXPERIMENT_ROOT / "runs" / f"{split}-source-frozen-shard-{shard_index}.json"


def build_task_specs(
    *,
    bundle_relative: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    train_first_game_index: int = TRAIN_FIRST_GAME_INDEX,
    train_games: int = TRAIN_GAMES,
    validation_first_game_index: int = VALIDATION_FIRST_GAME_INDEX,
    validation_games: int = VALIDATION_GAMES,
    strategy: str = STRATEGY,
) -> list[dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise CampaignError("bundle path must be repository-relative")
    if train_games < SHARD_COUNT or validation_games < SHARD_COUNT:
        raise CampaignError("each open split must assign at least one game to every host")
    if train_first_game_index < 0 or validation_first_game_index < 0:
        raise CampaignError("game indices must be nonnegative")
    if not strategy:
        raise CampaignError("strategy must not be empty")

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    bundle_report = experiment_root / "reports/source-frozen-bundle-fanout.json"
    destinations = [
        f"{host}:{_remote_path(host, bundle_relative)}/" for host in HOSTS if host != "john1"
    ]
    specs = [
        _task(
            task_id=bundle_fanout_id,
            title="Fan out immutable S1 semantic supply bundle",
            decision="Freeze one exact semantic supply executable and source identity",
            workload_class="shared-prerequisite",
            priority=0,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{bundle_relative}/",
                "--local-root",
                str(bundle_relative),
                *_repeated_flag("--destination", destinations),
                "--required-file",
                "bundle.json",
                "--required-file",
                "bin/exact_semantic_supply_census",
                "--required-file",
                "source/tools/s1_semantic_supply_merge.py",
                "--verify-tree",
                "--output",
                str(bundle_report),
            ],
            artifact_path=str(bundle_report),
            stop_rule="Every bundled source file and executable must be byte-identical.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    ]

    census_task_ids = []
    census_reports = []
    for split, first_game_index, games, priority, runtime in (
        ("train", train_first_game_index, train_games, 10, 1_800),
        ("validation", validation_first_game_index, validation_games, 11, 600),
    ):
        for shard_index, host in enumerate(HOSTS):
            task_id = f"{TASK_PREFIX}-{split}-shard-{shard_index}"
            report = experiment_root / "runs" / f"{split}-source-frozen-shard-{shard_index}.json"
            census_task_ids.append(task_id)
            census_reports.append(report)
            specs.append(
                _task(
                    task_id=task_id,
                    title=f"S1 {split} semantic supply shard {shard_index}",
                    decision=(
                        "Export exact public semantic supply on a disjoint complete-game "
                        "partition with hidden-order and D6 audits"
                    ),
                    workload_class="divisible-evidence",
                    priority=priority,
                    expected_runtime_seconds=runtime,
                    critical_path=True,
                    decision_terminal=False,
                    compatible_hosts=[host],
                    dependencies=[bundle_fanout_id],
                    command=[
                        *_frozen_binary_command(host, bundle_relative),
                        "--output",
                        _remote_path(host, report),
                        "--games",
                        str(games),
                        "--first-game-index",
                        str(first_game_index),
                        "--split",
                        split,
                        "--strategy",
                        strategy,
                        "--shard-index",
                        str(shard_index),
                        "--shard-count",
                        str(SHARD_COUNT),
                    ],
                    artifact_path=str(report),
                    stop_rule=(
                        "Emit exactly 80 audited positions for every modulo-owned game; "
                        "fail on any count, parity, D6, serialization, or hidden-order drift."
                    ),
                    cpu_cores=10,
                    memory_gib=6.0,
                )
            )

    remote_artifacts = []
    for split in ("train", "validation"):
        for shard_index, host in enumerate(HOSTS):
            if host == "john1":
                continue
            report = experiment_root / "runs" / f"{split}-source-frozen-shard-{shard_index}.json"
            remote_artifacts.extend(
                [
                    "--artifact",
                    f"{host}:{_remote_path(host, report)}",
                    str(report),
                ]
            )
    collection_report = experiment_root / "reports/source-frozen-shard-collection.json"
    collection_id = f"{TASK_PREFIX}-report-collection"
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect distributed S1 semantic supply shards",
            decision="Retrieve all six remote shard reports with checksum proof",
            workload_class="shared-prerequisite",
            priority=20,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=census_task_ids,
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_collect.py",
                *remote_artifacts,
                "--output",
                str(collection_report),
            ],
            artifact_path=str(collection_report),
            stop_rule="All eight census reports must be available locally and checksum verified.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    forward = experiment_root / "reports/source-frozen-aggregate-forward.json"
    reverse = experiment_root / "reports/source-frozen-aggregate-reverse.json"
    forward_id = f"{TASK_PREFIX}-merge-forward"
    reverse_id = f"{TASK_PREFIX}-merge-reverse"
    report_strings = [str(report) for report in census_reports]
    specs.extend(
        [
            _task(
                task_id=forward_id,
                title="Merge S1 semantic supply shards in canonical order",
                decision="Validate exact corpus coverage and every serialized supply law",
                workload_class="shared-prerequisite",
                priority=30,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=[
                    *_frozen_merge_command(bundle_relative),
                    *_repeated_flag("--shard", report_strings),
                    "--expected-shard-count",
                    str(SHARD_COUNT),
                    "--output",
                    str(forward),
                ],
                artifact_path=str(forward),
                stop_rule=(
                    "Require complete train and validation intervals, common source identity, "
                    "exact canonical bytes, and collision separation."
                ),
                cpu_cores=1,
                memory_gib=4.0,
            ),
            _task(
                task_id=reverse_id,
                title="Merge S1 semantic supply shards in reverse order",
                decision="Prove S1 aggregation is independent of shard arrival order",
                workload_class="replica",
                priority=31,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=[
                    *_frozen_merge_command(bundle_relative),
                    *_repeated_flag("--shard", list(reversed(report_strings))),
                    "--expected-shard-count",
                    str(SHARD_COUNT),
                    "--output",
                    str(reverse),
                ],
                artifact_path=str(reverse),
                stop_rule="Reverse-order merge must independently validate and serialize.",
                cpu_cores=1,
                memory_gib=4.0,
            ),
            _task(
                task_id=f"{TASK_PREFIX}-merge-order-proof",
                title="Verify S1 merge-order determinism",
                decision="Require byte-identical forward and reverse aggregate reports",
                workload_class="shared-prerequisite",
                priority=32,
                expected_runtime_seconds=10,
                critical_path=True,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[forward_id, reverse_id],
                command=["cmp", "-s", str(forward), str(reverse)],
                artifact_path=str(forward),
                stop_rule="Forward and reverse S1 aggregates must be byte-identical.",
                cpu_cores=1,
                memory_gib=0.25,
            ),
        ]
    )
    return specs


def install_specs(queue: Path, specs: list[dict[str, Any]]) -> None:
    with locked_queue(queue) as state:
        existing = {task["id"] for task in state["tasks"]}
        requested = [spec["id"] for spec in specs]
        duplicates = sorted(existing.intersection(requested))
        if duplicates:
            raise CampaignError(f"queue already contains S1 task IDs: {duplicates}")
        if len(requested) != len(set(requested)):
            raise CampaignError("generated S1 task IDs are not unique")
        for spec in specs:
            add_task(state, spec)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--train-first-game-index", type=int, default=TRAIN_FIRST_GAME_INDEX)
    parser.add_argument("--train-games", type=int, default=TRAIN_GAMES)
    parser.add_argument(
        "--validation-first-game-index",
        type=int,
        default=VALIDATION_FIRST_GAME_INDEX,
    )
    parser.add_argument("--validation-games", type=int, default=VALIDATION_GAMES)
    parser.add_argument("--strategy", default=STRATEGY)
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    try:
        bundle_relative = _relative_bundle_path(repository, args.bundle)
        manifest = validate_bundle(repository / bundle_relative)
        validate_provenance_source_bundle(manifest)
        binary_names = {
            entry["name"]
            for entry in manifest["identity"].get("binaries", [])
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
        if "exact_semantic_supply_census" not in binary_names:
            raise CampaignError("bundle lacks exact_semantic_supply_census")
        specs = build_task_specs(
            bundle_relative=bundle_relative,
            train_first_game_index=args.train_first_game_index,
            train_games=args.train_games,
            validation_first_game_index=args.validation_first_game_index,
            validation_games=args.validation_games,
            strategy=args.strategy,
        )
        payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "review_status": "generated-not-applied" if not args.apply else "applied",
            "bundle_id": manifest["bundle_id"],
            "bundle": str(bundle_relative),
            "task_count": len(specs),
            "tasks": specs,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(encoded)
            temporary.replace(args.output)
        if args.apply:
            install_specs(args.queue, specs)
        print(encoded, end="")
        return 0
    except (BundleError, CampaignError, OSError) as error:
        print(f"S1 semantic supply campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
