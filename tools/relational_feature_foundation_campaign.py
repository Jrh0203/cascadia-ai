#!/usr/bin/env python3
"""Freeze, distribute, and classify the R5/R6/S3/S5 foundation wave."""

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

CAMPAIGN_ID = "relational-feature-foundation-wave-v1"
PROTOCOL_ID = "relational-feature-four-lane-foundation-v1"
TASK_PREFIX = "relfeat"
BINARY_NAME = "relational-feature-census"
POSITIONS_PER_GAME = 80
DEFAULT_REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_SPEC = (
    Path("artifacts/experiments") / CAMPAIGN_ID / "queue-spec.json"
)
CAMPAIGN_ROOT = Path("artifacts/experiments") / CAMPAIGN_ID
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}


class CampaignError(RuntimeError):
    """Raised when the foundation wave violates its frozen protocol."""


@dataclass(frozen=True)
class Lane:
    key: str
    cli_lane: str
    experiment_id: str
    protocol_id: str
    host: str
    first_seed: int
    games: int
    rayon_threads: int
    expected_runtime_seconds: int
    passing_classification: str
    valid_classifications: tuple[str, ...]

    @property
    def report_relative(self) -> Path:
        return (
            Path("artifacts/experiments")
            / self.experiment_id
            / "reports"
            / f"{self.host}-production.json"
        )


LANES = (
    Lane(
        key="r5",
        cli_lane="r5",
        experiment_id="r5-component-motif-quotient-foundation-v1",
        protocol_id="r5-exact-decoding-and-compactness-v1",
        host="john1",
        first_seed=5_110_000,
        games=20,
        rayon_threads=6,
        expected_runtime_seconds=900,
        passing_classification="r5_local_geometry_exact_and_quotient_compact",
        valid_classifications=(
            "r5_local_geometry_exact_and_quotient_compact",
            "r5_action_local_exactness_failed",
            "r5_component_motif_score_decoder_failed",
            "r5_component_motif_compactness_failed",
        ),
    ),
    Lane(
        key="r6",
        cli_lane="r6",
        experiment_id="r6-incremental-sparse-accumulator-foundation-v1",
        protocol_id="r6-apply-undo-parity-and-throughput-v1",
        host="john2",
        first_seed=5_210_000,
        games=4,
        rayon_threads=10,
        expected_runtime_seconds=900,
        passing_classification="r6_incremental_apply_undo_promoted",
        valid_classifications=(
            "r6_incremental_apply_undo_promoted",
            "r6_incremental_exactness_failed",
            "r6_incremental_throughput_failed",
        ),
    ),
    Lane(
        key="s3",
        cli_lane="s3",
        experiment_id="s3-component-motif-graph-foundation-v1",
        protocol_id="s3-card-a-semantic-decoder-census-v1",
        host="john3",
        first_seed=5_310_000,
        games=14,
        rayon_threads=10,
        expected_runtime_seconds=900,
        passing_classification="s3_exact_component_motif_graph_promoted",
        valid_classifications=(
            "s3_exact_component_motif_graph_promoted",
            "s3_semantic_decoder_failed",
            "s3_opportunity_coverage_failed",
            "s3_d6_invariance_failed",
        ),
    ),
    Lane(
        key="s5",
        cli_lane="s5",
        experiment_id="s5-opportunity-derivative-foundation-v1",
        protocol_id="s5-exact-counterfactual-derivative-census-v1",
        host="john4",
        first_seed=5_410_000,
        games=20,
        rayon_threads=10,
        expected_runtime_seconds=900,
        passing_classification="s5_exact_opportunity_derivatives_promoted",
        valid_classifications=(
            "s5_exact_opportunity_derivatives_promoted",
            "s5_exact_derivative_replay_failed",
            "s5_derivative_normalization_contract_failed",
        ),
    ),
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
    if blake3.blake3(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest() != bundle_id:
        raise CampaignError("bundle identity does not match its content address")
    if identity.get("experiment_id") != CAMPAIGN_ID:
        raise CampaignError("bundle belongs to another experiment")
    binary_entries = [
        entry
        for entry in identity.get("binaries", [])
        if isinstance(entry, dict) and entry.get("name") == BINARY_NAME
    ]
    if len(binary_entries) != 1:
        raise CampaignError(f"bundle must contain exactly one {BINARY_NAME} binary")
    binary_path = bundle / "bin" / BINARY_NAME
    if not binary_path.is_file():
        raise CampaignError(f"bundle binary is missing: {binary_path}")
    if file_blake3(binary_path) != binary_entries[0].get("blake3"):
        raise CampaignError("bundle binary checksum does not match the manifest")
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
        "experiment_id": CAMPAIGN_ID,
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
    fanout_report = CAMPAIGN_ROOT / "reports" / "bundle-fanout.json"
    fanout_id = f"{TASK_PREFIX}-fanout-bundle"
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
            [
                "--destination",
                f"{host}:{_remote(host, bundle_relative)}/",
            ]
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
            title="Fan out relational-feature foundation bundle",
            decision="Require byte-identical source and executable identity on all hosts",
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

    run_ids: list[str] = []
    for lane in LANES:
        task_id = f"{TASK_PREFIX}-run-{lane.key}-{lane.host}"
        run_ids.append(task_id)
        report = lane.report_relative
        command = [
            "/usr/bin/env",
            "-C",
            _remote(lane.host, bundle_relative / "source"),
            _remote(lane.host, bundle_relative / "bin" / BINARY_NAME),
            "--lane",
            lane.cli_lane,
            "--first-seed",
            str(lane.first_seed),
            "--games",
            str(lane.games),
            "--source-bundle-id",
            bundle_id,
            "--host",
            lane.host,
            "--rayon-threads",
            str(lane.rayon_threads),
            "--output",
            _remote(lane.host, report),
        ]
        tasks.append(
            _task(
                task_id=task_id,
                title=f"Run {lane.key.upper()} foundation on {lane.host}",
                decision=f"Test the frozen {lane.experiment_id} hypothesis",
                workload_class="independent-experiment",
                priority=1,
                expected_runtime_seconds=lane.expected_runtime_seconds,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[lane.host],
                dependencies=[fanout_id],
                command=command,
                artifact_path=str(report),
                stop_rule=(
                    "Preserve the preregistered seed range and gates; a failed "
                    "scientific classification is valid negative evidence."
                ),
                cpu_cores=lane.rayon_threads,
                memory_gib=4.0,
            )
        )

    collection = CAMPAIGN_ROOT / "reports" / "collection.json"
    collect_command = [
        "/usr/bin/env",
        "-C",
        _remote("john1", bundle_relative / "source"),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
    for lane in LANES:
        collect_command.extend(
            [
                "--artifact",
                f"{lane.host}:{_remote(lane.host, lane.report_relative)}",
                _remote("john1", lane.report_relative),
            ]
        )
    collect_command.extend(["--output", _remote("john1", collection)])
    collect_id = f"{TASK_PREFIX}-collect"
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect four relational-feature foundation reports",
            decision="Bind every host result to a coordinator-side checksum",
            workload_class="shared-prerequisite",
            priority=2,
            expected_runtime_seconds=90,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=run_ids,
            command=collect_command,
            artifact_path=str(collection),
            stop_rule="All four reports must be collected with matching SHA-256 digests.",
            cpu_cores=1,
            memory_gib=1.0,
        )
    )

    aggregate = CAMPAIGN_ROOT / "aggregate-forward.json"
    reverse = CAMPAIGN_ROOT / "aggregate-reverse.json"
    order_proof = CAMPAIGN_ROOT / "order-proof.json"
    aggregate_command = [
        "/usr/bin/env",
        "-C",
        _remote("john1", bundle_relative / "source"),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/relational_feature_foundation_campaign.py",
        "aggregate",
        "--repository",
        str(REMOTE_ROOTS["john1"]),
        "--bundle",
        _remote("john1", bundle_relative),
        "--output",
        _remote("john1", aggregate),
        "--reverse-output",
        _remote("john1", reverse),
        "--order-proof",
        _remote("john1", order_proof),
    ]
    tasks.append(
        _task(
            task_id=f"{TASK_PREFIX}-aggregate",
            title="Validate and classify the relational-feature foundation wave",
            decision="Authorize only exact, provenance-valid foundations for learned tests",
            workload_class="shared-prerequisite",
            priority=3,
            expected_runtime_seconds=60,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collect_id],
            command=aggregate_command,
            artifact_path=str(aggregate),
            stop_rule=(
                "Identity or scientific-hash drift invalidates the wave. "
                "Scientific gate misses remain valid negative results."
            ),
            cpu_cores=1,
            memory_gib=1.0,
        )
    )
    return tasks


def build_queue_spec(repository: Path, bundle: Path) -> dict[str, Any]:
    tasks = build_task_specs(repository, bundle)
    payload = {
        "schema_version": 1,
        "experiment_id": CAMPAIGN_ID,
        "protocol_id": PROTOCOL_ID,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    payload["task_spec_blake3"] = blake3.blake3(canonical_json(tasks)).hexdigest()
    return payload


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise CampaignError(f"{label}: expected {expected!r}, found {actual!r}")


def _validate_metric_contract(lane: Lane, scientific: dict[str, Any]) -> None:
    metrics = scientific.get("metrics")
    if not isinstance(metrics, dict):
        raise CampaignError(f"{lane.key} report lacks metrics")
    positions = lane.games * POSITIONS_PER_GAME
    if lane.key == "r5":
        _require_equal(metrics.get("positions"), positions, "R5 positions")
        _require_equal(
            metrics.get("current_score_decoder_checks"),
            positions * 4,
            "R5 board decoder checks",
        )
        complete_actions = metrics.get("complete_actions")
        if not isinstance(complete_actions, int) or complete_actions <= 0:
            raise CampaignError("R5 complete action count must be positive")
        for field in (
            "control_affordance_checks",
            "quotient_affordance_underdetermined",
            "local_affordance_checks",
            "local_score_delta_checks",
        ):
            _require_equal(metrics.get(field), complete_actions, f"R5 {field}")
    elif lane.key == "r6":
        _require_equal(metrics.get("positions"), positions, "R6 positions")
        complete_actions = metrics.get("complete_actions")
        if not isinstance(complete_actions, int) or complete_actions <= 0:
            raise CampaignError("R6 complete action count must be positive")
        for field in ("exact_apply_checks", "exact_undo_checks"):
            _require_equal(metrics.get(field), complete_actions, f"R6 {field}")
    elif lane.key == "s3":
        _require_equal(metrics.get("positions"), positions, "S3 positions")
        _require_equal(
            metrics.get("board_score_decoder_checks"),
            positions * 4,
            "S3 board score checks",
        )
        _require_equal(
            metrics.get("action_delta_decoder_checks"),
            positions,
            "S3 action delta checks",
        )
        _require_equal(
            metrics.get("d6_invariance_checks"),
            positions * 12,
            "S3 D6 checks",
        )
        for field in (
            "boards_with_elk_extensions",
            "boards_with_salmon_continuations",
            "boards_with_hawk_opportunities",
            "boards_with_bear_pair_opportunities",
        ):
            value = metrics.get(field)
            if not isinstance(value, int) or value <= 0:
                raise CampaignError(f"S3 {field} must be positive")
    elif lane.key == "s5":
        _require_equal(metrics.get("positions"), positions, "S5 positions")
        expected_samples = positions * 64
        _require_equal(
            metrics.get("sampled_actions"),
            expected_samples,
            "S5 sampled actions",
        )
        _require_equal(
            metrics.get("exact_replay_checks"),
            expected_samples,
            "S5 replay checks",
        )
        _require_equal(
            metrics.get("score_delta_checks"),
            expected_samples,
            "S5 score checks",
        )
        _require_equal(metrics.get("feature_field_count"), 154, "S5 feature fields")
        feature_scales = metrics.get("feature_scales")
        if not isinstance(feature_scales, dict) or len(feature_scales) != 154:
            raise CampaignError("S5 feature scale map must contain 154 fields")


def validate_lane_report(
    report: dict[str, Any],
    lane: Lane,
    *,
    bundle_id: str,
    executable_blake3: str,
) -> dict[str, Any]:
    scientific = report.get("scientific")
    execution = report.get("execution")
    if not isinstance(scientific, dict) or not isinstance(execution, dict):
        raise CampaignError(f"{lane.key} report envelope is malformed")
    expected_hash = blake3.blake3(canonical_json(scientific)).hexdigest()
    _require_equal(
        report.get("scientific_blake3"),
        expected_hash,
        f"{lane.key} scientific hash",
    )
    _require_equal(scientific.get("schema_version"), 1, f"{lane.key} schema")
    _require_equal(
        scientific.get("artifact_kind"),
        "relational_feature_census_report",
        f"{lane.key} artifact kind",
    )
    _require_equal(
        scientific.get("experiment_id"),
        lane.experiment_id,
        f"{lane.key} experiment",
    )
    _require_equal(
        scientific.get("protocol_id"), lane.protocol_id, f"{lane.key} protocol"
    )
    _require_equal(
        scientific.get("source_bundle_id"), bundle_id, f"{lane.key} bundle"
    )
    config = scientific.get("config")
    if not isinstance(config, dict):
        raise CampaignError(f"{lane.key} report lacks its config")
    expected_config = {
        "lane": {
            "r5": "r5-quotient",
            "r6": "r6-incremental",
            "s3": "s3-component-motif",
            "s5": "s5-derivatives",
        }[lane.key],
        "first_seed": lane.first_seed,
        "games": lane.games,
        "source_bundle_id": bundle_id,
        "host": lane.host,
        "rayon_threads": lane.rayon_threads,
    }
    _require_equal(config, expected_config, f"{lane.key} config")
    corpus = scientific.get("corpus")
    if not isinstance(corpus, dict):
        raise CampaignError(f"{lane.key} report lacks its corpus")
    _require_equal(corpus.get("first_seed"), lane.first_seed, f"{lane.key} seed")
    _require_equal(corpus.get("games"), lane.games, f"{lane.key} games")
    _require_equal(
        corpus.get("positions"),
        lane.games * POSITIONS_PER_GAME,
        f"{lane.key} positions",
    )
    classification = scientific.get("classification")
    if classification not in lane.valid_classifications:
        raise CampaignError(
            f"{lane.key} emitted unknown classification {classification!r}"
        )
    passed = scientific.get("passed")
    if not isinstance(passed, bool):
        raise CampaignError(f"{lane.key} passed flag is not boolean")
    _require_equal(
        passed,
        classification == lane.passing_classification,
        f"{lane.key} pass/classification consistency",
    )
    _validate_metric_contract(lane, scientific)
    _require_equal(execution.get("host"), lane.host, f"{lane.key} execution host")
    _require_equal(
        execution.get("executable_blake3"),
        executable_blake3,
        f"{lane.key} executable",
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
        raise CampaignError(f"{lane.key} execution timing is invalid")
    return {
        "lane": lane.key,
        "experiment_id": lane.experiment_id,
        "protocol_id": lane.protocol_id,
        "host": lane.host,
        "first_seed": lane.first_seed,
        "games": lane.games,
        "positions": lane.games * POSITIONS_PER_GAME,
        "passed": passed,
        "classification": classification,
        "scientific_blake3": expected_hash,
        "elapsed_ms": elapsed,
        "metrics": scientific["metrics"],
    }


def aggregate_reports(
    reports: Iterable[tuple[Lane, dict[str, Any]]],
    *,
    bundle_id: str,
    executable_blake3: str,
) -> dict[str, Any]:
    validated = [
        validate_lane_report(
            report,
            lane,
            bundle_id=bundle_id,
            executable_blake3=executable_blake3,
        )
        for lane, report in reports
    ]
    validated.sort(key=lambda value: value["experiment_id"])
    if {value["experiment_id"] for value in validated} != {
        lane.experiment_id for lane in LANES
    }:
        raise CampaignError("aggregate does not contain exactly one report per lane")
    all_passed = all(value["passed"] for value in validated)
    scientific = {
        "schema_version": 1,
        "artifact_kind": "relational_feature_foundation_aggregate",
        "experiment_id": CAMPAIGN_ID,
        "protocol_id": PROTOCOL_ID,
        "source_bundle_id": bundle_id,
        "all_evidence_valid": True,
        "all_foundations_passed": all_passed,
        "classification": (
            "relational_feature_foundations_authorized"
            if all_passed
            else "relational_feature_foundations_partial"
        ),
        "lanes": validated,
    }
    return {
        "scientific": scientific,
        "scientific_blake3": blake3.blake3(canonical_json(scientific)).hexdigest(),
        "generated_unix_ms": time.time_ns() // 1_000_000,
    }


def _load_reports(repository: Path, lanes: Iterable[Lane]) -> list[tuple[Lane, dict[str, Any]]]:
    loaded = []
    for lane in lanes:
        path = repository / lane.report_relative
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(f"cannot read {lane.key} report {path}: {error}") from error
        loaded.append((lane, report))
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
    reports = _load_reports(repository, LANES)
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
        "experiment_id": CAMPAIGN_ID,
        "source_bundle_id": bundle_id,
        "forward_scientific_blake3": forward["scientific_blake3"],
        "reverse_scientific_blake3": reverse["scientific_blake3"],
        "order_invariant": (
            forward["scientific_blake3"] == reverse["scientific_blake3"]
        ),
    }
    if not proof["order_invariant"]:
        raise CampaignError("aggregate classification depends on report order")
    _write_json(order_proof, proof)
    return {
        "experiment_id": CAMPAIGN_ID,
        "classification": forward["scientific"]["classification"],
        "all_foundations_passed": forward["scientific"]["all_foundations_passed"],
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
        output = args.output
        if not output.is_absolute():
            output = args.repository / output
        _write_json(output, payload)
        print(
            json.dumps(
                {
                    "experiment_id": CAMPAIGN_ID,
                    "task_count": payload["task_count"],
                    "task_spec_blake3": payload["task_spec_blake3"],
                    "output": str(output),
                },
                sort_keys=True,
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
