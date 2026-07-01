#!/usr/bin/env python3
"""Generate, but never install, the immutable F5 parity cluster task specification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.corrected_mid_tail_parity import (  # noqa: E402
    CORPUS_MANIFEST_FILE_BLAKE3,
    CORPUS_MANIFEST_SCIENTIFIC_BLAKE3,
    CORPUS_PAYLOAD_BLAKE3,
    CORRECTED_CHECKPOINT_BLAKE3,
    CORRECTED_CHECKPOINT_BYTES,
    CORRECTED_CHECKPOINT_CONTRACT,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_CORRECTED_CHECKPOINT,
    DEFAULT_HISTORICAL_CHECKPOINT,
    EXPERIMENT_ID,
    HISTORICAL_CHECKPOINT_BLAKE3,
    HISTORICAL_CHECKPOINT_BYTES,
    HISTORICAL_CHECKPOINT_CONTRACT,
    PRODUCTION_CORPUS_CONTRACT,
    ParityCampaignError,
    checksum_file,
    implementation_identity,
    scientific_blake3,
    validate_all_corpus_payload_identities,
    validate_checkpoint_identity,
)

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
EXPERIMENT_ROOT = Path("artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1")
REPORT_ROOT = EXPERIMENT_ROOT / "reports"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "queue-spec.json"
TASK_PREFIX = "f5p"
SOURCE_FANOUT_ID = f"{TASK_PREFIX}-source-bundle-fanout"


class QueueSpecError(RuntimeError):
    """Raised when the reviewed parity task graph cannot be generated exactly."""


def assigned_host(shard_index: int) -> str:
    if not 0 <= shard_index < PRODUCTION_CORPUS_CONTRACT.shard_count:
        raise QueueSpecError("shard index is outside the frozen ten-shard corpus")
    return HOSTS[shard_index % len(HOSTS)]


def shard_report(shard_index: int) -> Path:
    return REPORT_ROOT / f"shard-{shard_index:05d}.json"


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def source_bundle_root(implementation_blake3: str) -> Path:
    return EXPERIMENT_ROOT / "source" / "blake3" / implementation_blake3


def prepare_source_bundle(
    repository: Path,
    implementation: dict[str, Any],
) -> Path:
    implementation_blake3 = str(implementation["bundle_blake3"])
    bundle_root = repository / source_bundle_root(implementation_blake3)
    files = implementation.get("files")
    if not isinstance(files, list) or not files:
        raise QueueSpecError("implementation identity does not declare source files")

    expected: dict[str, bytes] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise QueueSpecError("implementation source declaration must be an object")
        relative_path = entry.get("relative_file")
        expected_bytes = entry.get("bytes")
        expected_blake3 = entry.get("blake3")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_blake3, str)
            or len(expected_blake3) != 64
        ):
            raise QueueSpecError("implementation source declaration is malformed")
        source = repository / relative_path
        if source.is_symlink() or not source.is_file():
            raise QueueSpecError(f"implementation source is missing: {source}")
        payload = source.read_bytes()
        if len(payload) != expected_bytes:
            raise QueueSpecError(f"implementation source byte count drifted: {source}")
        if checksum_file(source) != expected_blake3:
            raise QueueSpecError(f"implementation source checksum drifted: {source}")
        expected[relative_path] = payload

    manifest = {
        "schema_version": 1,
        "implementation": implementation,
        "files": sorted(expected),
    }
    expected["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()

    if bundle_root.exists():
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise QueueSpecError("existing source bundle must be a regular directory")
        actual_paths = {
            path.relative_to(bundle_root).as_posix()
            for path in bundle_root.rglob("*")
            if path.is_file()
        }
        if actual_paths != set(expected):
            raise QueueSpecError("existing immutable source bundle file set differs")
        for relative_path, payload in expected.items():
            path = bundle_root / relative_path
            if path.is_symlink() or path.read_bytes() != payload:
                raise QueueSpecError(f"existing immutable source bundle differs: {relative_path}")
        return bundle_root

    temporary = bundle_root.with_name(f".{bundle_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        for relative_path, payload in expected.items():
            path = temporary / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            os.chmod(path, 0o444)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
        bundle_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, bundle_root)
        os.chmod(bundle_root, 0o555)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle_root


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
        "expected_runtime_seconds": 300 if uses_mlx else 30,
        "critical_path": True,
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


def _shard_command(host: str, shard_index: int, implementation_blake3: str) -> list[str]:
    root = REMOTE_ROOTS[host]
    bundle = root / source_bundle_root(implementation_blake3)
    return [
        str(root / ".venv/bin/python"),
        "-B",
        str(bundle / "tools/corrected_mid_tail_parity.py"),
        "shard",
        "--shard-index",
        str(shard_index),
        "--corpus-root",
        str(root / DEFAULT_CORPUS_ROOT),
        "--historical-checkpoint",
        str(root / DEFAULT_HISTORICAL_CHECKPOINT),
        "--corrected-checkpoint",
        str(root / DEFAULT_CORRECTED_CHECKPOINT),
        "--batch-rows",
        "512",
        "--expected-implementation-blake3",
        implementation_blake3,
        "--output",
        str(root / shard_report(shard_index)),
    ]


def build_queue_spec(implementation: dict[str, Any] | None = None) -> dict[str, Any]:
    implementation = implementation or implementation_identity()
    implementation_blake3 = str(implementation["bundle_blake3"])
    shard_task_ids = []
    bundle_relative = source_bundle_root(implementation_blake3)
    bundle_files = [
        str(entry["relative_file"])
        for entry in implementation["files"]
        if isinstance(entry, dict) and isinstance(entry.get("relative_file"), str)
    ]
    bundle_files.append("manifest.json")
    bundle_fanout_report = REPORT_ROOT / "source-bundle-fanout.json"
    bundle_fanout_command = [
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        str(REMOTE_ROOTS["john1"] / "tools/cluster_artifact_fanout.py"),
        "--source",
        f"{_remote('john1', bundle_relative)}/",
        "--local-root",
        _remote("john1", bundle_relative),
    ]
    for host in HOSTS[1:]:
        bundle_fanout_command.extend(["--destination", f"{host}:{_remote(host, bundle_relative)}/"])
    for relative_path in bundle_files:
        bundle_fanout_command.extend(["--required-file", relative_path])
    bundle_fanout_command.extend(
        [
            "--verify-tree",
            "--output",
            _remote("john1", bundle_fanout_report),
        ]
    )
    tasks = [
        _task(
            task_id=SOURCE_FANOUT_ID,
            title="Fan out immutable F5 parity source bundle",
            decision=("Require the exact content-addressed Python implementation on every host"),
            workload_class="shared-prerequisite",
            priority=5,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=bundle_fanout_command,
            artifact_path=str(bundle_fanout_report),
            stop_rule=(
                "Every source file and the complete bundle tree must match on john2, "
                "john3, and john4."
            ),
            cpu_cores=1,
            memory_gib=0.25,
            uses_mlx=False,
        )
    ]
    for shard_index in range(PRODUCTION_CORPUS_CONTRACT.shard_count):
        host = assigned_host(shard_index)
        task_id = f"{TASK_PREFIX}-shard-{shard_index:05d}"
        shard_task_ids.append(task_id)
        tasks.append(
            _task(
                task_id=task_id,
                title=f"F5 C0/T1 parity shard {shard_index}",
                decision=(
                    "Prove byte-identical C0/T1 float32 predictions on one immutable "
                    "20,000-row corpus shard"
                ),
                workload_class="divisible-evidence",
                priority=10,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[SOURCE_FANOUT_ID],
                command=_shard_command(host, shard_index, implementation_blake3),
                artifact_path=str(shard_report(shard_index)),
                stop_rule=(
                    "Fail on any input drift, malformed row, discarded activation, "
                    "non-finite output, or prediction-byte mismatch."
                ),
                cpu_cores=1,
                memory_gib=1.5,
                uses_mlx=True,
            )
        )

    remote_shards = [
        shard_index
        for shard_index in range(PRODUCTION_CORPUS_CONTRACT.shard_count)
        if assigned_host(shard_index) != "john1"
    ]
    collection_id = f"{TASK_PREFIX}-collect-remote-reports"
    collection_receipt = REPORT_ROOT / "remote-collection.json"
    collection_command = [
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        str(REMOTE_ROOTS["john1"] / "tools/cluster_artifact_collect.py"),
    ]
    for shard_index in remote_shards:
        host = assigned_host(shard_index)
        report = shard_report(shard_index)
        collection_command.extend(
            [
                "--artifact",
                f"{host}:{_remote(host, report)}",
                _remote("john1", report),
            ]
        )
    collection_command.extend(["--output", _remote("john1", collection_receipt)])
    tasks.append(
        _task(
            task_id=collection_id,
            title="Collect seven remote F5 parity reports",
            decision="Copy each nonlocal scientific receipt to john1 with checksum proof",
            workload_class="shared-prerequisite",
            priority=20,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=shard_task_ids,
            command=collection_command,
            artifact_path=str(collection_receipt),
            stop_rule="Every remote report must match its source byte for byte.",
            cpu_cores=1,
            memory_gib=0.5,
            uses_mlx=False,
        )
    )

    forward_id = f"{TASK_PREFIX}-aggregate-forward"
    reverse_id = f"{TASK_PREFIX}-aggregate-reverse"
    forward = REPORT_ROOT / "aggregate-forward.json"
    reverse = REPORT_ROOT / "aggregate-reverse.json"
    ordered_reports = [shard_report(index) for index in range(10)]

    def aggregate_command(reports: list[Path], output: Path) -> list[str]:
        bundle = REMOTE_ROOTS["john1"] / bundle_relative
        command = [
            str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
            "-B",
            str(bundle / "tools/corrected_mid_tail_parity.py"),
            "aggregate",
        ]
        for report in reports:
            command.extend(["--report", _remote("john1", report)])
        command.extend(["--output", _remote("john1", output)])
        return command

    for task_id, title, reports, output in (
        (forward_id, "Aggregate F5 parity reports forward", ordered_reports, forward),
        (
            reverse_id,
            "Aggregate F5 parity reports reverse",
            list(reversed(ordered_reports)),
            reverse,
        ),
    ):
        tasks.append(
            _task(
                task_id=task_id,
                title=title,
                decision=("Require all ten disjoint shards and classify exact 200,000-row parity"),
                workload_class="shared-prerequisite",
                priority=30,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=aggregate_command(reports, output),
                artifact_path=str(output),
                stop_rule=(
                    "Aggregation must reject missing, duplicated, overlapping, gapped, "
                    "drifted, incomplete, or nonidentical evidence."
                ),
                cpu_cores=1,
                memory_gib=0.5,
                uses_mlx=False,
            )
        )

    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-aggregate-order-proof",
            title="Prove F5 aggregate order independence",
            decision="Require byte-identical forward and reverse aggregate reports",
            workload_class="shared-prerequisite",
            priority=31,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[forward_id, reverse_id],
            command=["/usr/bin/cmp", "-s", _remote("john1", forward), _remote("john1", reverse)],
            artifact_path=str(forward),
            stop_rule="Forward and reverse aggregate bytes must be identical.",
            cpu_cores=1,
            memory_gib=0.25,
            uses_mlx=False,
        )
    )

    host_shard_counts = Counter(assigned_host(index) for index in range(10))
    spec = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "review_status": "generated-not-applied",
        "shared_queue_mutated": False,
        "implementation": implementation,
        "source_bundle": {
            "root": str(bundle_relative),
            "required_files": bundle_files,
            "fanout_task": SOURCE_FANOUT_ID,
        },
        "input_contract": {
            "corpus": {
                "dataset_id": PRODUCTION_CORPUS_CONTRACT.dataset_id,
                "rows": PRODUCTION_CORPUS_CONTRACT.rows,
                "shards": PRODUCTION_CORPUS_CONTRACT.shard_count,
                "manifest_file_blake3": CORPUS_MANIFEST_FILE_BLAKE3,
                "manifest_scientific_blake3": CORPUS_MANIFEST_SCIENTIFIC_BLAKE3,
                "payload_blake3": CORPUS_PAYLOAD_BLAKE3,
            },
            "C0": {
                "bytes": HISTORICAL_CHECKPOINT_BYTES,
                "blake3": HISTORICAL_CHECKPOINT_BLAKE3,
            },
            "T1": {
                "bytes": CORRECTED_CHECKPOINT_BYTES,
                "blake3": CORRECTED_CHECKPOINT_BLAKE3,
            },
        },
        "allocation": {
            "rule": "shard_index mod 4",
            "host_order": list(HOSTS),
            "shards_by_host": {
                host: [
                    index
                    for index in range(PRODUCTION_CORPUS_CONTRACT.shard_count)
                    if assigned_host(index) == host
                ]
                for host in HOSTS
            },
            "shard_count_by_host": dict(sorted(host_shard_counts.items())),
            "maximum_concurrent_shards": 4,
        },
        "task_count": len(tasks),
        "tasks": tasks,
    }
    spec["scientific_blake3"] = scientific_blake3(spec)
    return spec


def write_immutable_json(path: Path, value: object) -> bool:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise QueueSpecError("existing queue specification must be a regular non-symlink file")
        if path.read_text() != encoded:
            raise QueueSpecError("existing immutable queue specification differs")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(encoded)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output = args.output if args.output.is_absolute() else repository / args.output
    try:
        validate_all_corpus_payload_identities(repository / DEFAULT_CORPUS_ROOT)
        validate_checkpoint_identity(
            repository / DEFAULT_HISTORICAL_CHECKPOINT,
            HISTORICAL_CHECKPOINT_CONTRACT,
        )
        validate_checkpoint_identity(
            repository / DEFAULT_CORRECTED_CHECKPOINT,
            CORRECTED_CHECKPOINT_CONTRACT,
        )
        implementation = implementation_identity()
        bundle_root = prepare_source_bundle(repository, implementation)
        spec = build_queue_spec(implementation)
        if bundle_root != repository / Path(spec["source_bundle"]["root"]):
            raise QueueSpecError("prepared source bundle path disagrees with queue specification")
        reused = write_immutable_json(output, spec)
    except (OSError, ParityCampaignError, QueueSpecError) as error:
        print(f"corrected mid-tail parity queue-spec error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "reused": reused,
                "scientific_blake3": spec["scientific_blake3"],
                "task_count": spec["task_count"],
                "shared_queue_mutated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
