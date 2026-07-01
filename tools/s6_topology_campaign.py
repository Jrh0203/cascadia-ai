#!/usr/bin/env python3
"""Freeze, distribute, and classify the four-host S6 topology campaign."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "s6-topological-spectral-foundation-v2"
PROTOCOL_ID = "s6-exact-topology-activation-census-v2"
TASK_PREFIX = "s6top"
BINARY_NAME = "relational-feature-census"
POSITIONS_PER_GAME = 80
PASSING_CLASSIFICATION = "s6_topological_spectral_foundation_v2_authorized"
VALID_CLASSIFICATIONS = {
    PASSING_CLASSIFICATION,
    "s6_topology_decoder_failed",
    "s6_d6_invariance_failed",
    "s6_adversarial_separation_failed",
    "s6_feature_variation_futile",
    "s6_long_range_coverage_futile",
    "s6_isolated_latency_failed",
    "s6_encoding_compactness_failed",
}
DEFAULT_REPOSITORY = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_QUEUE_SPEC = CAMPAIGN_ROOT / "queue-spec.json"
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}


class CampaignError(RuntimeError):
    """Raised when the S6 campaign violates its frozen contract."""


@dataclass(frozen=True)
class Shard:
    host: str
    first_seed: int
    games: int = 10
    rayon_threads: int = 10

    @property
    def report_relative(self) -> Path:
        return CAMPAIGN_ROOT / "reports" / f"{self.host}-production.json"


SHARDS = (
    Shard("john1", 5_610_000),
    Shard("john2", 5_620_000),
    Shard("john3", 5_630_000),
    Shard("john4", 5_640_000),
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _bundle_manifest(bundle: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((bundle / "bundle.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read immutable bundle {bundle}: {error}") from error
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or len(bundle_id) != 64:
        raise CampaignError("bundle manifest has an invalid bundle ID")
    if bundle.name != bundle_id:
        raise CampaignError("bundle directory name does not match its bundle ID")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise CampaignError("bundle manifest lacks its scientific identity")
    expected_bundle_id = blake3.blake3(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if bundle_id != expected_bundle_id:
        raise CampaignError("bundle identity does not match its content address")
    if identity.get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("bundle belongs to another experiment")
    binaries = [
        entry
        for entry in identity.get("binaries", [])
        if isinstance(entry, dict) and entry.get("name") == BINARY_NAME
    ]
    if len(binaries) != 1:
        raise CampaignError(f"bundle must contain exactly one {BINARY_NAME} binary")
    binary = bundle / "bin" / BINARY_NAME
    if not binary.is_file() or file_blake3(binary) != binaries[0].get("blake3"):
        raise CampaignError("bundle binary is missing or checksum-invalid")
    return manifest


def _relative_bundle(repository: Path, bundle: Path) -> Path:
    try:
        return bundle.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise CampaignError("bundle must remain beneath the repository") from error


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: int,
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


def build_task_specs(repository: Path, bundle: Path) -> list[dict[str, Any]]:
    manifest = _bundle_manifest(bundle)
    bundle_id = manifest["bundle_id"]
    bundle_relative = _relative_bundle(repository, bundle)
    fanout_id = f"{TASK_PREFIX}-fanout"
    fanout_report = CAMPAIGN_ROOT / "control" / "bundle-fanout.json"
    fanout_command = [
        ".venv/bin/python",
        "-B",
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{bundle_relative}/",
        "--local-root",
        str(bundle_relative),
    ]
    for host in ("john2", "john3", "john4"):
        fanout_command.extend(
            ["--destination", f"{host}:{_remote(host, bundle_relative)}/"]
        )
    fanout_command.extend(
        [
            "--required-file",
            "bundle.json",
            "--verify-tree",
            "--output",
            str(fanout_report),
        ]
    )
    tasks = [
        _task(
            task_id=fanout_id,
            title="Fan out immutable S6 topology bundle",
            decision="Bind all four hosts to one exact source and executable",
            workload_class="shared-prerequisite",
            priority=0,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=fanout_command,
            artifact_path=str(fanout_report),
            stop_rule="Every regular bundle file must match on john1 through john4.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    ]

    run_ids = []
    for shard in SHARDS:
        task_id = f"{TASK_PREFIX}-run-{shard.host}"
        run_ids.append(task_id)
        command = [
            "/usr/bin/env",
            "-C",
            _remote(shard.host, bundle_relative / "source"),
            _remote(shard.host, bundle_relative / "bin" / BINARY_NAME),
            "--lane",
            "s6",
            "--first-seed",
            str(shard.first_seed),
            "--games",
            str(shard.games),
            "--source-bundle-id",
            bundle_id,
            "--host",
            shard.host,
            "--rayon-threads",
            str(shard.rayon_threads),
            "--output",
            _remote(shard.host, shard.report_relative),
        ]
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Run S6 topology census on {shard.host}",
                decision="Test one disjoint seed block under the frozen S6 gates",
                workload_class="divisible-evidence",
                priority=1,
                expected_runtime_seconds=1_200,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[shard.host],
                dependencies=[fanout_id],
                command=command,
                artifact_path=str(shard.report_relative),
                stop_rule=(
                    "Preserve the registered seed block and every frozen gate; "
                    "a negative classification remains valid evidence."
                ),
                cpu_cores=shard.rayon_threads,
                memory_gib=4.0,
            )
        )

    collection = CAMPAIGN_ROOT / "control" / "production-collection.json"
    collect_command = [
        "/usr/bin/env",
        "-C",
        _remote("john1", bundle_relative / "source"),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
    for shard in SHARDS:
        collect_command.extend(
            [
                "--artifact",
                f"{shard.host}:{_remote(shard.host, shard.report_relative)}",
                _remote("john1", shard.report_relative),
            ]
        )
    collect_command.extend(["--output", _remote("john1", collection)])
    collect_id = f"{TASK_PREFIX}-collect"
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect four S6 topology reports",
            decision="Require checksum-bound evidence from every seed block",
            workload_class="shared-prerequisite",
            priority=2,
            expected_runtime_seconds=90,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=run_ids,
            command=collect_command,
            artifact_path=str(collection),
            stop_rule="All four reports must be present and SHA-256 verified.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    forward = CAMPAIGN_ROOT / "aggregate-forward.json"
    reverse = CAMPAIGN_ROOT / "aggregate-reverse.json"
    proof = CAMPAIGN_ROOT / "order-proof.json"
    aggregate_command = [
        "/usr/bin/env",
        "-C",
        _remote("john1", bundle_relative / "source"),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/s6_topology_campaign.py",
        "aggregate",
        "--repository",
        str(REMOTE_ROOTS["john1"]),
        "--bundle",
        _remote("john1", bundle_relative),
        "--output",
        _remote("john1", forward),
        "--reverse-output",
        _remote("john1", reverse),
        "--order-proof",
        _remote("john1", proof),
    ]
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-aggregate",
            title="Validate and classify the S6 topology campaign",
            decision="Authorize only exact, novel, portable, bounded-cost S6 features",
            workload_class="shared-prerequisite",
            priority=3,
            expected_runtime_seconds=60,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collect_id],
            command=aggregate_command,
            artifact_path=str(forward),
            stop_rule=(
                "Identity drift invalidates the campaign; scientific gate misses "
                "remain valid negative evidence."
            ),
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    return tasks


def build_queue_spec(repository: Path, bundle: Path) -> dict[str, Any]:
    tasks = build_task_specs(repository, bundle)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "task_count": len(tasks),
        "tasks": tasks,
        "task_spec_blake3": blake3.blake3(canonical_json(tasks)).hexdigest(),
    }


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise CampaignError(f"{label}: expected {expected!r}, found {actual!r}")


def _validate_distribution(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} is not an object")
    required = {
        "count",
        "minimum",
        "mean_milli",
        "median",
        "p90",
        "p99",
        "maximum",
    }
    if set(value) != required or any(
        not isinstance(value[field], int) or value[field] < 0 for field in required
    ):
        raise CampaignError(f"{label} is not a valid distribution")
    if not (
        value["minimum"]
        <= value["median"]
        <= value["p90"]
        <= value["p99"]
        <= value["maximum"]
    ):
        raise CampaignError(f"{label} percentiles are not monotone")
    return value


def validate_shard_report(
    report: dict[str, Any],
    shard: Shard,
    *,
    bundle_id: str,
    executable_blake3: str,
) -> dict[str, Any]:
    scientific = report.get("scientific")
    execution = report.get("execution")
    if not isinstance(scientific, dict) or not isinstance(execution, dict):
        raise CampaignError(f"{shard.host} report envelope is malformed")
    expected_hash = blake3.blake3(canonical_json(scientific)).hexdigest()
    _require_equal(
        report.get("scientific_blake3"),
        expected_hash,
        f"{shard.host} scientific hash",
    )
    _require_equal(scientific.get("schema_version"), 1, f"{shard.host} schema")
    _require_equal(
        scientific.get("artifact_kind"),
        "relational_feature_census_report",
        f"{shard.host} artifact kind",
    )
    _require_equal(
        scientific.get("experiment_id"), EXPERIMENT_ID, f"{shard.host} experiment"
    )
    _require_equal(
        scientific.get("protocol_id"), PROTOCOL_ID, f"{shard.host} protocol"
    )
    _require_equal(
        scientific.get("source_bundle_id"), bundle_id, f"{shard.host} bundle"
    )
    expected_config = {
        "lane": "s6-topology",
        "first_seed": shard.first_seed,
        "games": shard.games,
        "source_bundle_id": bundle_id,
        "host": shard.host,
        "rayon_threads": shard.rayon_threads,
    }
    _require_equal(scientific.get("config"), expected_config, f"{shard.host} config")
    corpus = scientific.get("corpus")
    if not isinstance(corpus, dict):
        raise CampaignError(f"{shard.host} report lacks its corpus")
    _require_equal(corpus.get("first_seed"), shard.first_seed, f"{shard.host} seed")
    _require_equal(corpus.get("games"), shard.games, f"{shard.host} games")
    positions = shard.games * POSITIONS_PER_GAME
    _require_equal(corpus.get("positions"), positions, f"{shard.host} positions")
    classification = scientific.get("classification")
    if classification not in VALID_CLASSIFICATIONS:
        raise CampaignError(
            f"{shard.host} emitted unknown classification {classification!r}"
        )
    passed = scientific.get("passed")
    if not isinstance(passed, bool):
        raise CampaignError(f"{shard.host} passed flag is not boolean")
    _require_equal(
        passed,
        classification == PASSING_CLASSIFICATION,
        f"{shard.host} pass/classification consistency",
    )
    metrics = scientific.get("metrics")
    if not isinstance(metrics, dict):
        raise CampaignError(f"{shard.host} report lacks metrics")
    boards = positions * 4
    _require_equal(metrics.get("positions"), positions, f"{shard.host} metric positions")
    _require_equal(metrics.get("board_encodings"), boards, f"{shard.host} boards")
    _require_equal(
        metrics.get("topology_decoder_checks"),
        boards * 11,
        f"{shard.host} decoder checks",
    )
    _require_equal(
        metrics.get("d6_invariance_checks"),
        positions * 12,
        f"{shard.host} D6 checks",
    )
    _require_equal(metrics.get("adversarial_checks"), 4, f"{shard.host} adversarial")
    encoding_bytes = _validate_distribution(
        metrics.get("encoding_bytes"), f"{shard.host} encoding bytes"
    )
    extraction_ns = _validate_distribution(
        metrics.get("extraction_ns"), f"{shard.host} extraction time"
    )
    isolated_extraction_ns = _validate_distribution(
        metrics.get("isolated_extraction_ns"),
        f"{shard.host} isolated extraction time",
    )
    _require_equal(encoding_bytes.get("count"), positions, f"{shard.host} byte count")
    _require_equal(extraction_ns.get("count"), positions, f"{shard.host} timing count")
    _require_equal(
        isolated_extraction_ns.get("count"),
        POSITIONS_PER_GAME,
        f"{shard.host} isolated timing count",
    )
    gate_fields = (
        "exactness_gate_pass",
        "d6_gate_pass",
        "adversarial_gate_pass",
        "feature_variation_gate_pass",
        "long_range_gate_pass",
        "isolated_latency_gate_pass",
        "compactness_gate_pass",
    )
    if any(not isinstance(metrics.get(field), bool) for field in gate_fields):
        raise CampaignError(f"{shard.host} gate fields are not boolean")
    if passed and not all(metrics[field] for field in gate_fields):
        raise CampaignError(f"{shard.host} passed with a failed gate")
    _require_equal(execution.get("host"), shard.host, f"{shard.host} execution host")
    _require_equal(
        execution.get("executable_blake3"),
        executable_blake3,
        f"{shard.host} executable",
    )
    started = execution.get("started_unix_ms")
    completed = execution.get("completed_unix_ms")
    elapsed = execution.get("elapsed_ms")
    if (
        not isinstance(started, int)
        or not isinstance(completed, int)
        or not isinstance(elapsed, int)
        or completed < started
        or elapsed != completed - started
    ):
        raise CampaignError(f"{shard.host} execution timing is invalid")
    return {
        "host": shard.host,
        "first_seed": shard.first_seed,
        "games": shard.games,
        "positions": positions,
        "passed": passed,
        "classification": classification,
        "scientific_blake3": expected_hash,
        "elapsed_ms": elapsed,
        "metrics": metrics,
    }


def _validate_seed_ranges(shards: Iterable[Shard]) -> None:
    ranges = []
    hosts = set()
    for shard in shards:
        if shard.host in hosts:
            raise CampaignError(f"duplicate S6 host {shard.host}")
        hosts.add(shard.host)
        ranges.append((shard.first_seed, shard.first_seed + shard.games, shard.host))
    ranges.sort()
    for index in range(len(ranges) - 1):
        left = ranges[index]
        right = ranges[index + 1]
        if left[1] > right[0]:
            raise CampaignError(
                f"S6 seed ranges overlap between {left[2]} and {right[2]}"
            )


def aggregate_reports(
    reports: Iterable[tuple[Shard, dict[str, Any]]],
    *,
    bundle_id: str,
    executable_blake3: str,
) -> dict[str, Any]:
    _validate_seed_ranges(SHARDS)
    validated = [
        validate_shard_report(
            report,
            shard,
            bundle_id=bundle_id,
            executable_blake3=executable_blake3,
        )
        for shard, report in reports
    ]
    validated.sort(key=lambda value: value["host"])
    if {value["host"] for value in validated} != {shard.host for shard in SHARDS}:
        raise CampaignError("aggregate does not contain exactly one report per host")
    all_passed = all(value["passed"] for value in validated)
    totals = {
        "games": sum(value["games"] for value in validated),
        "positions": sum(value["positions"] for value in validated),
        "board_encodings": sum(
            value["metrics"]["board_encodings"] for value in validated
        ),
        "topology_decoder_checks": sum(
            value["metrics"]["topology_decoder_checks"] for value in validated
        ),
        "topology_decoder_failures": sum(
            value["metrics"]["topology_decoder_failures"] for value in validated
        ),
        "d6_invariance_checks": sum(
            value["metrics"]["d6_invariance_checks"] for value in validated
        ),
        "d6_invariance_failures": sum(
            value["metrics"]["d6_invariance_failures"] for value in validated
        ),
        "baseline_collision_pairs": sum(
            value["metrics"]["baseline_collision_pairs"] for value in validated
        ),
        "full_encoding_separated_pairs": sum(
            value["metrics"]["full_encoding_separated_pairs"] for value in validated
        ),
        "long_range_separated_pairs": sum(
            value["metrics"]["long_range_separated_pairs"] for value in validated
        ),
        "maximum_host_extraction_p99_ns": max(
            value["metrics"]["extraction_ns"]["p99"] for value in validated
        ),
        "maximum_host_isolated_extraction_p99_ns": max(
            value["metrics"]["isolated_extraction_ns"]["p99"]
            for value in validated
        ),
        "maximum_host_median_encoding_bytes": max(
            value["metrics"]["encoding_bytes"]["median"] for value in validated
        ),
    }
    scientific = {
        "schema_version": 1,
        "artifact_kind": "s6_topology_campaign_aggregate",
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "source_bundle_id": bundle_id,
        "executable_blake3": executable_blake3,
        "all_evidence_valid": True,
        "all_hosts_passed": all_passed,
        "classification": (
            PASSING_CLASSIFICATION
            if all_passed
            else "s6_topological_spectral_foundation_partial"
        ),
        "totals": totals,
        "shards": validated,
    }
    return {
        "scientific": scientific,
        "scientific_blake3": blake3.blake3(canonical_json(scientific)).hexdigest(),
        "generated_unix_ms": time.time_ns() // 1_000_000,
    }


def _load_reports(repository: Path) -> list[tuple[Shard, dict[str, Any]]]:
    loaded = []
    for shard in SHARDS:
        path = repository / shard.report_relative
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(
                f"cannot read {shard.host} report {path}: {error}"
            ) from error
        loaded.append((shard, report))
    return loaded


def run_aggregate(
    repository: Path,
    bundle: Path,
    output: Path,
    reverse_output: Path,
    order_proof: Path,
) -> dict[str, Any]:
    manifest = _bundle_manifest(bundle)
    bundle_id = manifest["bundle_id"]
    executable_blake3 = next(
        entry["blake3"]
        for entry in manifest["identity"]["binaries"]
        if entry["name"] == BINARY_NAME
    )
    reports = _load_reports(repository)
    forward = aggregate_reports(
        reports,
        bundle_id=bundle_id,
        executable_blake3=executable_blake3,
    )
    reverse = aggregate_reports(
        reversed(reports),
        bundle_id=bundle_id,
        executable_blake3=executable_blake3,
    )
    _write_json(output, forward)
    _write_json(reverse_output, reverse)
    proof = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_bundle_id": bundle_id,
        "forward_scientific_blake3": forward["scientific_blake3"],
        "reverse_scientific_blake3": reverse["scientific_blake3"],
        "order_invariant": (
            forward["scientific_blake3"] == reverse["scientific_blake3"]
        ),
    }
    if not proof["order_invariant"]:
        raise CampaignError("S6 aggregate depends on report order")
    _write_json(order_proof, proof)
    return {
        "experiment_id": EXPERIMENT_ID,
        "classification": forward["scientific"]["classification"],
        "all_hosts_passed": forward["scientific"]["all_hosts_passed"],
        "scientific_blake3": forward["scientific_blake3"],
        "output": str(output),
        "order_proof": str(order_proof),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-spec")
    build.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--output", type=Path, default=DEFAULT_QUEUE_SPEC)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    aggregate.add_argument("--bundle", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--reverse-output", type=Path, required=True)
    aggregate.add_argument("--order-proof", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "build-spec":
        payload = build_queue_spec(args.repository, args.bundle)
        _write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "task_count": payload["task_count"],
                    "task_spec_blake3": payload["task_spec_blake3"],
                    "output": str(args.output),
                },
                separators=(",", ":"),
            )
        )
        return 0
    result = run_aggregate(
        args.repository,
        args.bundle,
        args.output,
        args.reverse_output,
        args.order_proof,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
