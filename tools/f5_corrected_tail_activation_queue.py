#!/usr/bin/env python3
"""Build the reviewed-only, source-frozen F5 activation-census queue graph."""

# ruff: noqa: UP045 - cluster tools must run under macOS system Python 3.9.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from cluster_research_queue import add_task, locked_queue
from rust_experiment_bundle import BundleError, validate_bundle

EXPERIMENT_ID = "corrected-mid-tail-activation-census-v1"
TASK_PREFIX = "f5a"
HOSTS = ("john1", "john2", "john3", "john4")
SHARD_COUNT = len(HOSTS)
FIRST_GAME_INDEX = 0
TOTAL_GAMES = 1_024
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
BINARY_NAME = "f5-corrected-tail-activation-census"
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "docs/v2/decisions/0149-corrected-tail-activation-census.md",
    "docs/v2/reports/corrected-mid-tail-activation-census-v1-preregistration.md",
    "legacy/crates/cascadia-ai/Cargo.toml",
    "legacy/crates/cascadia-core/Cargo.toml",
    "tools/f5_corrected_tail_activation_census/Cargo.lock",
    "tools/f5_corrected_tail_activation_census/Cargo.toml",
    "tools/f5_corrected_tail_activation_census/README.md",
    "tools/f5_corrected_tail_activation_census/build.rs",
    "tools/f5_corrected_tail_activation_queue.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "legacy/crates/cascadia-ai/src/",
    "legacy/crates/cascadia-core/src/",
    "tools/f5_corrected_tail_activation_census/src/",
    "tools/f5_corrected_tail_activation_census/tests/",
)


class CampaignError(RuntimeError):
    """Raised when the F5 activation execution graph is incomplete or ambiguous."""


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
        _remote_path(host, bundle_relative / "bin" / BINARY_NAME),
    ]


def _repeated_flag(flag: str, values: list[str]) -> list[str]:
    return [item for value in values for item in (flag, value)]


def corpus_root(shard_index: int, experiment_root: Path = DEFAULT_EXPERIMENT_ROOT) -> Path:
    return experiment_root / "corpus" / f"shard-{shard_index}"


def report_path(shard_index: int, experiment_root: Path = DEFAULT_EXPERIMENT_ROOT) -> Path:
    return experiment_root / "reports" / f"shard-{shard_index}.json"


def validate_provenance_source_bundle(manifest: dict[str, Any]) -> None:
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("bundle scientific identity has the wrong experiment ID")
    source_entries = identity.get("source_files", [])
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
            "bundle cannot reproduce the F5 activation census: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}"
        )


def build_task_specs(
    *,
    bundle_relative: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    first_game_index: int = FIRST_GAME_INDEX,
    total_games: int = TOTAL_GAMES,
) -> list[dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise CampaignError("bundle path must be repository-relative")
    if first_game_index < 0:
        raise CampaignError("first game index must be nonnegative")
    if total_games < SHARD_COUNT or total_games % SHARD_COUNT != 0:
        raise CampaignError("total games must divide evenly across all four hosts")

    fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    fanout_report = experiment_root / "reports" / "source-frozen-bundle-fanout.json"
    destinations = [
        f"{host}:{_remote_path(host, bundle_relative)}/" for host in HOSTS if host != "john1"
    ]
    specs = [
        _task(
            task_id=fanout_id,
            title="Fan out immutable corrected-tail activation bundle",
            decision="Freeze one exact Rust extractor executable and source identity",
            workload_class="shared-prerequisite",
            priority=50,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=[
                str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
                str(REMOTE_ROOTS["john1"] / "tools/cluster_artifact_fanout.py"),
                "--source",
                f"{_remote_path('john1', bundle_relative)}/",
                "--local-root",
                _remote_path("john1", bundle_relative),
                *_repeated_flag("--destination", destinations),
                "--required-file",
                "bundle.json",
                "--required-file",
                f"bin/{BINARY_NAME}",
                "--required-file",
                "source/CASCADIA_V2_GOAL.txt",
                "--required-file",
                "source/legacy/crates/cascadia-ai/src/nnue.rs",
                "--required-file",
                "source/tools/f5_corrected_tail_activation_census/Cargo.toml",
                "--required-file",
                "source/tools/f5_corrected_tail_activation_census/build.rs",
                "--verify-tree",
                "--output",
                _remote_path("john1", fanout_report),
            ],
            artifact_path=str(fanout_report),
            stop_rule="Every bundled source file and executable must be byte-identical.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    ]

    generation_ids = []
    census_ids = []
    for shard_index, host in enumerate(HOSTS):
        root = corpus_root(shard_index, experiment_root)
        generation_id = f"{TASK_PREFIX}-generate-shard-{shard_index}"
        census_id = f"{TASK_PREFIX}-census-shard-{shard_index}"
        generation_ids.append(generation_id)
        census_ids.append(census_id)
        specs.append(
            _task(
                task_id=generation_id,
                title=f"Generate corrected-tail public-state shard {shard_index}",
                decision=(
                    "Generate one disjoint 256-game public-state partition through "
                    "the exact legacy game and feature path"
                ),
                workload_class="divisible-evidence",
                priority=51,
                expected_runtime_seconds=1_200,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[fanout_id],
                command=[
                    *_frozen_binary_command(host, bundle_relative),
                    "generate-shard",
                    "--output-root",
                    _remote_path(host, root),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(SHARD_COUNT),
                    "--first-game-index",
                    str(first_game_index),
                    "--total-games",
                    str(total_games),
                    "--threads",
                    "0",
                ],
                artifact_path=str(root / "manifest.json"),
                stop_rule=(
                    "Emit exactly 80 replay-valid records for every modulo-owned game; "
                    "fail on source drift, public-state mismatch, or payload corruption."
                ),
                cpu_cores=10,
                memory_gib=6.0,
            )
        )
        report = report_path(shard_index, experiment_root)
        specs.append(
            _task(
                task_id=census_id,
                title=f"Census corrected-tail activation shard {shard_index}",
                decision=(
                    "Replay every frozen record through the actual corrected Rust "
                    "extractor and account for all 301 channels"
                ),
                workload_class="divisible-evidence",
                priority=52,
                expected_runtime_seconds=900,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[generation_id],
                command=[
                    *_frozen_binary_command(host, bundle_relative),
                    "census-shard",
                    "--corpus-root",
                    _remote_path(host, root),
                    "--output",
                    _remote_path(host, report),
                ],
                artifact_path=str(report),
                stop_rule=(
                    "Fail on any malformed record, duplicate JSON key, source mismatch, "
                    "hash drift, replay mismatch, or corrected-channel accounting error."
                ),
                cpu_cores=1,
                memory_gib=4.0,
            )
        )

    remote_artifacts = []
    for shard_index, host in enumerate(HOSTS):
        if host == "john1":
            continue
        root = corpus_root(shard_index, experiment_root)
        for relative in ("manifest.json", "records.jsonl"):
            remote_artifacts.extend(
                [
                    "--artifact",
                    f"{host}:{_remote_path(host, root / relative)}",
                    _remote_path("john1", root / relative),
                ]
            )
        report = report_path(shard_index, experiment_root)
        remote_artifacts.extend(
            [
                "--artifact",
                f"{host}:{_remote_path(host, report)}",
                _remote_path("john1", report),
            ]
        )
    collection_id = f"{TASK_PREFIX}-collect-remote-artifacts"
    collection_report = experiment_root / "reports" / "remote-collection.json"
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect corrected-tail corpora and reports",
            decision="Preserve every remote scientific payload on john1 with checksum proof",
            workload_class="shared-prerequisite",
            priority=53,
            expected_runtime_seconds=600,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=census_ids,
            command=[
                str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
                str(REMOTE_ROOTS["john1"] / "tools/cluster_artifact_collect.py"),
                *remote_artifacts,
                "--output",
                _remote_path("john1", collection_report),
            ],
            artifact_path=str(collection_report),
            stop_rule=(
                "All three remote manifests, record streams, and shard reports must "
                "match their sources byte for byte."
            ),
            cpu_cores=1,
            memory_gib=2.0,
        )
    )

    forward = experiment_root / "reports" / "aggregate-forward.json"
    reverse = experiment_root / "reports" / "aggregate-reverse.json"
    forward_id = f"{TASK_PREFIX}-aggregate-forward"
    reverse_id = f"{TASK_PREFIX}-aggregate-reverse"
    ordered_reports = [report_path(index, experiment_root) for index in range(SHARD_COUNT)]

    def aggregate_command(reports: list[Path], output: Path) -> list[str]:
        return [
            *_frozen_binary_command("john1", bundle_relative),
            "aggregate",
            *_repeated_flag(
                "--report",
                [_remote_path("john1", report) for report in reports],
            ),
            "--require-shards",
            str(SHARD_COUNT),
            "--output",
            _remote_path("john1", output),
        ]

    specs.extend(
        [
            _task(
                task_id=forward_id,
                title="Aggregate corrected-tail activation reports forward",
                decision="Classify exact natural activation coverage over all 81,920 rows",
                workload_class="shared-prerequisite",
                priority=54,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=aggregate_command(ordered_reports, forward),
                artifact_path=str(forward),
                stop_rule=(
                    "Require four source-identical disjoint shards, exact corpus totals, "
                    "all preregistered gates, and one separate overflow witness."
                ),
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=reverse_id,
                title="Aggregate corrected-tail activation reports reverse",
                decision="Independently prove aggregation is insensitive to input order",
                workload_class="replica",
                priority=54,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=aggregate_command(list(reversed(ordered_reports)), reverse),
                artifact_path=str(reverse),
                stop_rule="Reverse-order aggregation must independently validate and serialize.",
                cpu_cores=1,
                memory_gib=2.0,
            ),
            _task(
                task_id=f"{TASK_PREFIX}-aggregate-order-proof",
                title="Verify corrected-tail aggregate order independence",
                decision="Require byte-identical forward and reverse aggregate reports",
                workload_class="shared-prerequisite",
                priority=55,
                expected_runtime_seconds=30,
                critical_path=True,
                decision_terminal=True,
                compatible_hosts=["john1"],
                dependencies=[forward_id, reverse_id],
                command=[
                    *_frozen_binary_command("john1", bundle_relative),
                    "verify-order",
                    "--left",
                    _remote_path("john1", forward),
                    "--right",
                    _remote_path("john1", reverse),
                ],
                artifact_path=str(forward),
                stop_rule="Forward and reverse aggregate bytes must be identical and valid.",
                cpu_cores=1,
                memory_gib=0.5,
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
            raise CampaignError(f"queue already contains F5 activation task IDs: {duplicates}")
        if len(requested) != len(set(requested)):
            raise CampaignError("generated F5 activation task IDs are not unique")
        for spec in specs:
            add_task(state, spec)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
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
        if BINARY_NAME not in binary_names:
            raise CampaignError(f"bundle lacks {BINARY_NAME}")
        specs = build_task_specs(bundle_relative=bundle_relative)
        payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "review_status": "applied" if args.apply else "generated-not-applied",
            "bundle_id": manifest["bundle_id"],
            "bundle": str(bundle_relative),
            "allocation": {
                "host_order": list(HOSTS),
                "rule": "shard_index mod 4",
                "maximum_concurrent_experiments": 4,
            },
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
        print(f"F5 activation campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
