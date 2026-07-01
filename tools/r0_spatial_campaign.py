#!/usr/bin/env python3
"""Build and install the distributed R0 production execution graph."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster_research_queue import add_task, locked_queue
from rust_experiment_bundle import BundleError, validate_bundle

EXPERIMENT_ID = "r0-spatial-footprint-screen-v1"
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_DATASET_ROOT = Path("artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen")
TASK_PREFIX = "r0f"
HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
TRAIN_GAMES = (157, 156, 156, 156)
TRAIN_FIRST_GAME_INDEX = (200_000, 200_157, 200_313, 200_469)
VALIDATION_GAMES = (32, 31, 31, 31)
VALIDATION_FIRST_GAME_INDEX = (210_000, 210_032, 210_063, 210_094)
REQUIRED_REPLICATES = 3
SHARD_COUNT = len(HOSTS)
COLLECTION_SHARD_GAMES = 8
BENCHMARK_ITERATIONS = 50
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "Cargo.lock",
    "Cargo.toml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
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
    """Raised when the campaign graph cannot be built without ambiguity."""


@dataclass(frozen=True)
class DatasetPart:
    split: str
    part_index: int
    host: str
    games: int
    first_game_index: int
    root: Path

    @property
    def collection_task_id(self) -> str:
        return f"{TASK_PREFIX}-collect-{self.split}-part-{self.part_index}"

    @property
    def fanout_task_id(self) -> str:
        return f"{TASK_PREFIX}-fanout-{self.split}-part-{self.part_index}"


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


def dataset_parts(dataset_root: Path = DEFAULT_DATASET_ROOT) -> list[DatasetPart]:
    parts: list[DatasetPart] = []
    for split, games, first_indexes in (
        ("train", TRAIN_GAMES, TRAIN_FIRST_GAME_INDEX),
        ("validation", VALIDATION_GAMES, VALIDATION_FIRST_GAME_INDEX),
    ):
        for part_index, host in enumerate(HOSTS):
            parts.append(
                DatasetPart(
                    split=split,
                    part_index=part_index,
                    host=host,
                    games=games[part_index],
                    first_game_index=first_indexes[part_index],
                    root=Path(f"{dataset_root}-{split}-part-{part_index}"),
                )
            )
    return parts


def _relative_bundle_path(repository: Path, bundle: Path) -> Path:
    repository = repository.resolve()
    bundle = bundle.resolve()
    try:
        return bundle.relative_to(repository)
    except ValueError as error:
        raise CampaignError("bundle must remain beneath the repository") from error


def _remote_path(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _frozen_binary_command(
    host: str,
    bundle_relative: Path,
    name: str,
) -> list[str]:
    return [
        "/usr/bin/env",
        "-C",
        _remote_path(host, bundle_relative / "source"),
        _remote_path(host, bundle_relative / "bin" / name),
    ]


def _repeated_flag(flag: str, values: list[str]) -> list[str]:
    return [item for value in values for item in (flag, value)]


def _fanout_destinations(relative: Path) -> list[str]:
    return [f"{host}:{_remote_path(host, relative)}/" for host in HOSTS if host != "john1"]


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
            "bundle cannot satisfy cascadia-provenance source hashing: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}"
        )


def build_dataset_part_task_specs(
    *,
    part: DatasetPart,
    bundle_relative: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    bundle_fanout_id: str | None = None,
    collection_task_id: str | None = None,
    fanout_task_id: str | None = None,
    fanout_report_name: str | None = None,
) -> list[dict[str, Any]]:
    """Build one collection task and its coordinator-owned whole-tree fanout."""
    bundle_fanout_id = bundle_fanout_id or f"{TASK_PREFIX}-production-bundle-fanout"
    collection_task_id = collection_task_id or part.collection_task_id
    fanout_task_id = fanout_task_id or part.fanout_task_id
    fanout_report_name = fanout_report_name or (
        f"source-frozen-dataset-{part.split}-part-{part.part_index}-fanout.json"
    )
    collection = _task(
        task_id=collection_task_id,
        title=f"Collect R0 {part.split} part {part.part_index}",
        decision=(
            f"Generate {part.games * 80:,} open {part.split} positions from a "
            "disjoint pattern-aware game interval"
        ),
        workload_class="divisible-evidence",
        priority=10,
        expected_runtime_seconds=max(300, part.games * 5),
        critical_path=True,
        decision_terminal=False,
        compatible_hosts=[part.host],
        dependencies=[bundle_fanout_id],
        command=[
            *_frozen_binary_command(
                part.host,
                bundle_relative,
                "cascadia-v2",
            ),
            "collect",
            "--output",
            _remote_path(part.host, part.root),
            "--games",
            str(part.games),
            "--first-game-index",
            str(part.first_game_index),
            "--split",
            part.split,
            "--strategy",
            "pattern-aware",
            "--shard-games",
            str(COLLECTION_SHARD_GAMES),
            "--resume",
        ],
        artifact_path=str(part.root / "dataset.json"),
        stop_rule=(
            f"Complete exactly {part.games} games beginning at "
            f"{part.first_game_index} with the frozen pattern-aware collector."
        ),
        cpu_cores=10,
        memory_gib=4.0,
    )

    source = (
        f"{part.root}/"
        if part.host == "john1"
        else f"{part.host}:{_remote_path(part.host, part.root)}/"
    )
    fanout_report = experiment_root / "reports" / fanout_report_name
    fanout = _task(
        task_id=fanout_task_id,
        title=f"Verify and fan out R0 {part.split} part {part.part_index}",
        decision="Make this immutable dataset part byte-identical on every benchmark host",
        workload_class="shared-prerequisite",
        priority=20,
        expected_runtime_seconds=180,
        critical_path=True,
        decision_terminal=False,
        compatible_hosts=["john1"],
        dependencies=[collection_task_id],
        command=[
            ".venv/bin/python",
            "tools/cluster_artifact_fanout.py",
            "--source",
            source,
            "--local-root",
            str(part.root),
            *_repeated_flag(
                "--destination",
                _fanout_destinations(part.root),
            ),
            "--required-file",
            "dataset.json",
            "--verify-tree",
            "--output",
            str(fanout_report),
        ],
        artifact_path=str(fanout_report),
        stop_rule=("The complete manifested dataset tree must match byte for byte on every host."),
        cpu_cores=1,
        memory_gib=1.0,
    )
    return [collection, fanout]


def build_task_specs(
    *,
    bundle_relative: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
    required_replicates: int = REQUIRED_REPLICATES,
) -> list[dict[str, Any]]:
    if benchmark_iterations <= 0:
        raise CampaignError("benchmark iterations must be positive")
    if required_replicates < 3:
        raise CampaignError("R0 requires at least three independent process invocations")
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise CampaignError("bundle path must be repository-relative")

    bundle_fanout_id = f"{TASK_PREFIX}-production-bundle-fanout"
    bundle_report = experiment_root / "reports/source-frozen-production-bundle-fanout.json"
    bundle_destinations = [
        f"{host}:{_remote_path(host, bundle_relative)}/" for host in HOSTS if host != "john1"
    ]
    specs = [
        _task(
            task_id=bundle_fanout_id,
            title="Fan out immutable R0 production bundle",
            decision="Freeze one source and executable identity on all four Macs",
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
                *_repeated_flag("--destination", bundle_destinations),
                "--required-file",
                "bundle.json",
                "--required-file",
                "bin/cascadia-v2",
                "--required-file",
                "bin/spatial_representation_benchmark",
                "--verify-tree",
                "--output",
                str(bundle_report),
            ],
            artifact_path=str(bundle_report),
            stop_rule=(
                "Every bundled source file and executable must be byte-identical on all four hosts."
            ),
            cpu_cores=1,
            memory_gib=1.0,
        )
    ]

    parts = dataset_parts(dataset_root)
    for part in parts:
        specs.extend(
            build_dataset_part_task_specs(
                part=part,
                bundle_relative=bundle_relative,
                experiment_root=experiment_root,
                bundle_fanout_id=bundle_fanout_id,
            )
        )

    all_fanouts = [part.fanout_task_id for part in parts]
    dataset_roots = [part.root for part in parts]
    benchmark_task_ids: list[str] = []
    benchmark_reports: list[Path] = []
    for shard_index, host in enumerate(HOSTS):
        for replicate_index in range(required_replicates):
            task_id = f"{TASK_PREFIX}-benchmark-shard-{shard_index}-replicate-{replicate_index}"
            report = (
                experiment_root
                / "runs"
                / (f"{host}-source-frozen-shard-{shard_index}-replicate-{replicate_index}.json")
            )
            benchmark_task_ids.append(task_id)
            benchmark_reports.append(report)
            dataset_args = _repeated_flag(
                "--dataset-root",
                [_remote_path(host, root) for root in dataset_roots],
            )
            specs.append(
                _task(
                    task_id=task_id,
                    title=f"R0 shard {shard_index} timing replicate {replicate_index}",
                    decision=(
                        "Measure all five lossless representations in one independent "
                        "release-process invocation on a disjoint ordinal partition"
                    ),
                    workload_class="independent-experiment",
                    priority=30 + replicate_index,
                    expected_runtime_seconds=900,
                    critical_path=True,
                    decision_terminal=False,
                    compatible_hosts=[host],
                    dependencies=all_fanouts,
                    command=[
                        *_frozen_binary_command(
                            host,
                            bundle_relative,
                            "spatial_representation_benchmark",
                        ),
                        *dataset_args,
                        "--shard-index",
                        str(shard_index),
                        "--shard-count",
                        str(SHARD_COUNT),
                        "--records",
                        "0",
                        "--iterations",
                        str(benchmark_iterations),
                        "--replicate-index",
                        str(replicate_index),
                        "--output",
                        _remote_path(host, report),
                    ],
                    artifact_path=str(report),
                    stop_rule=(
                        "Round-trip every record and emit one complete all-arm report; "
                        "do not reuse a process across replicate indices."
                    ),
                    cpu_cores=10,
                    memory_gib=8.0,
                )
            )

    remote_report_pairs: list[str] = []
    for shard_index, host in enumerate(HOSTS):
        if host == "john1":
            continue
        for replicate_index in range(required_replicates):
            report = (
                experiment_root
                / "runs"
                / (f"{host}-source-frozen-shard-{shard_index}-replicate-{replicate_index}.json")
            )
            remote_report_pairs.extend(
                [
                    "--artifact",
                    f"{host}:{_remote_path(host, report)}",
                    str(report),
                ]
            )
    report_collection = experiment_root / "reports/source-frozen-benchmark-report-collection.json"
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-benchmark-report-collection",
            title="Collect distributed R0 benchmark reports",
            decision="Retrieve every remote process report with checksum proof",
            workload_class="shared-prerequisite",
            priority=40,
            expected_runtime_seconds=120,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=benchmark_task_ids,
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_collect.py",
                *remote_report_pairs,
                "--output",
                str(report_collection),
            ],
            artifact_path=str(report_collection),
            stop_rule="All nine remote process reports must match their source checksums.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    forward = experiment_root / "reports/extraction-source-frozen-aggregate-forward.json"
    reverse = experiment_root / "reports/extraction-source-frozen-aggregate-reverse.json"
    report_args = _repeated_flag(
        "--report",
        [str(report) for report in benchmark_reports],
    )
    common_classification_dependencies = [
        f"{TASK_PREFIX}-benchmark-report-collection",
        *[
            task_id
            for task_id in benchmark_task_ids
            if task_id.startswith(f"{TASK_PREFIX}-benchmark-shard-0-")
        ],
    ]
    specs.extend(
        [
            _task(
                task_id=f"{TASK_PREFIX}-extraction-classification-forward",
                title="Classify R0 extraction evidence in forward order",
                decision="Apply every mechanical, semantic, replicate, and performance gate",
                workload_class="shared-prerequisite",
                priority=50,
                expected_runtime_seconds=60,
                critical_path=True,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=common_classification_dependencies,
                command=[
                    ".venv/bin/python",
                    "tools/spatial_representation_benchmark_report.py",
                    *report_args,
                    "--required-replicates",
                    str(required_replicates),
                    "--output",
                    str(forward),
                ],
                artifact_path=str(forward),
                stop_rule=(
                    "Fail closed on missing replicas, shard drift, semantic loss, "
                    "or invalid timing."
                ),
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=f"{TASK_PREFIX}-extraction-classification-reverse",
                title="Classify R0 extraction evidence in reverse order",
                decision="Prove aggregate identity is independent of report order",
                workload_class="replica",
                priority=51,
                expected_runtime_seconds=60,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[f"{TASK_PREFIX}-extraction-classification-forward"],
                command=[
                    ".venv/bin/python",
                    "tools/spatial_representation_benchmark_report.py",
                    *_repeated_flag(
                        "--report",
                        [str(report) for report in reversed(benchmark_reports)],
                    ),
                    "--required-replicates",
                    str(required_replicates),
                    "--output",
                    str(reverse),
                ],
                artifact_path=str(reverse),
                stop_rule="The reverse-order aggregate must be independently valid.",
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=f"{TASK_PREFIX}-extraction-merge-order-proof",
                title="Verify R0 merge-order determinism",
                decision="Require forward and reverse terminal reports to be byte-identical",
                workload_class="shared-prerequisite",
                priority=52,
                expected_runtime_seconds=10,
                critical_path=True,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[
                    f"{TASK_PREFIX}-extraction-classification-forward",
                    f"{TASK_PREFIX}-extraction-classification-reverse",
                ],
                command=["cmp", "-s", str(forward), str(reverse)],
                artifact_path=str(reverse),
                stop_rule=(
                    "Byte identity is mandatory; no timestamp or invocation-order drift is allowed."
                ),
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
            raise CampaignError(f"queue already contains R0 task IDs: {duplicates}")
        if len(requested) != len(set(requested)):
            raise CampaignError("generated R0 task IDs are not unique")
        for spec in specs:
            add_task(state, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--required-replicates", type=int, default=REQUIRED_REPLICATES)
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    try:
        bundle_relative = _relative_bundle_path(repository, args.bundle)
        manifest = validate_bundle(repository / bundle_relative)
        validate_provenance_source_bundle(manifest)
        binary_names = {entry["name"] for entry in manifest["identity"].get("binaries", [])}
        required_binaries = {"cascadia-v2", "spatial_representation_benchmark"}
        if not required_binaries.issubset(binary_names):
            raise CampaignError(
                f"bundle lacks required binaries: {sorted(required_binaries - binary_names)}"
            )
        specs = build_task_specs(
            bundle_relative=bundle_relative,
            benchmark_iterations=args.benchmark_iterations,
            required_replicates=args.required_replicates,
        )
        payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
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
        print(f"r0 campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
