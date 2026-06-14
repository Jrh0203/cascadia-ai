#!/usr/bin/env python3
"""Run and aggregate resumable final-strength benchmark shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "cascadia-aaaaa-4p-base-v1"
EXPERIMENT_ID = "final-exact-mlx-k32-r600-held-out-v1-20260614"
BASELINE_ID = "late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90"
TREATMENT_ID = "canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude"


@dataclass(frozen=True)
class RunConfig:
    binary: Path
    model_dir: Path
    weights: Path
    output_dir: Path
    first_seed: int
    games: int
    rollouts: int
    uv: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def source_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_fingerprints(config: RunConfig) -> dict[str, str]:
    return {
        "binary_sha256": sha256_file(config.binary),
        "model_manifest_sha256": sha256_file(config.model_dir / "model.json"),
        "model_safetensors_sha256": sha256_file(config.model_dir / "model.safetensors"),
        "weights_sha256": sha256_file(config.weights),
    }


def validate_score(score: dict[str, Any]) -> None:
    if len(score.get("habitat", [])) != 5 or len(score.get("wildlife", [])) != 5:
        raise ValueError("score breakdown must contain five habitat and wildlife values")
    expected = sum(score["habitat"]) + sum(score["wildlife"]) + score["nature_tokens"]
    if score.get("base_total") != expected:
        raise ValueError("score base_total does not match its decomposed values")


def validate_game_report(report: dict[str, Any], seed: int, rollouts: int) -> None:
    if report.get("schema_version") != 1:
        raise ValueError(f"seed {seed}: unsupported report schema")
    if report.get("rollouts") != rollouts:
        raise ValueError(f"seed {seed}: rollout budget drifted")
    if report.get("seed_domain") != "final":
        raise ValueError(f"seed {seed}: report did not use the sealed final seed domain")
    if report.get("status") != "smoke-passed":
        raise ValueError(f"seed {seed}: one-game integrity gate did not pass")
    if not report.get("clean_shutdown"):
        raise ValueError(f"seed {seed}: MLX service did not shut down cleanly")
    if not report.get("gates", {}).get("smoke_passed"):
        raise ValueError(f"seed {seed}: bridge/runtime smoke gate failed")
    comparison = report.get("comparison", {})
    if comparison.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"seed {seed}: protocol drifted")
    if comparison.get("baseline_id") != BASELINE_ID:
        raise ValueError(f"seed {seed}: baseline drifted")
    if comparison.get("treatment_id") != TREATMENT_ID:
        raise ValueError(f"seed {seed}: treatment drifted")
    if comparison.get("games") != 1 or comparison.get("first_seed") != seed:
        raise ValueError(f"seed {seed}: report does not contain exactly the requested game")
    records = report.get("game_records", [])
    if len(records) != 1 or records[0].get("seed") != seed:
        raise ValueError(f"seed {seed}: raw game record is absent or mismatched")
    record = records[0]
    if len(record.get("game_seed", [])) != 32:
        raise ValueError(f"seed {seed}: derived 32-byte game seed is absent")
    for role in ("baseline", "treatment"):
        scores = record.get(f"{role}_scores", [])
        decisions = record.get(f"{role}_decision_seconds", [])
        if len(scores) != 4:
            raise ValueError(f"seed {seed}: {role} must contain four seat scores")
        if len(decisions) != 80:
            raise ValueError(f"seed {seed}: {role} must contain 80 decision timings")
        for score in scores:
            validate_score(score)


def validate_completed_game(
    report_path: Path,
    metadata_path: Path,
    *,
    seed: int,
    rollouts: int,
    fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    validate_game_report(report, seed, rollouts)
    if metadata.get("seed") != seed:
        raise ValueError(f"seed {seed}: metadata seed drifted")
    if metadata.get("fingerprints") != fingerprints:
        raise ValueError(f"seed {seed}: executable or model fingerprints drifted")
    if metadata.get("report_sha256") != sha256_file(report_path):
        raise ValueError(f"seed {seed}: report checksum mismatch")
    return report, metadata


def shard_manifest(
    config: RunConfig,
    fingerprints: dict[str, str],
    completed: list[int],
    root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "source_revision": source_revision(root),
        "first_seed": config.first_seed,
        "games": config.games,
        "last_seed": config.first_seed + config.games - 1,
        "rollouts": config.rollouts,
        "fingerprints": fingerprints,
        "completed_seeds": completed,
        "complete": len(completed) == config.games,
        "updated_at": timestamp(),
    }


def run_shard(config: RunConfig) -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        config.binary,
        config.weights,
        config.model_dir / "model.json",
        config.model_dir / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing benchmark inputs: {missing}")
    if config.games <= 0 or config.rollouts <= 0:
        raise ValueError("games and rollouts must be positive")

    fingerprints = run_fingerprints(config)
    games_dir = config.output_dir / "games"
    logs_dir = config.output_dir / "logs"
    games_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    requested = list(range(config.first_seed, config.first_seed + config.games))
    completed: list[int] = []

    for seed in requested:
        report_path = games_dir / f"{seed}.json"
        metadata_path = games_dir / f"{seed}.meta.json"
        if report_path.is_file() and metadata_path.is_file():
            validate_completed_game(
                report_path,
                metadata_path,
                seed=seed,
                rollouts=config.rollouts,
                fingerprints=fingerprints,
            )
            completed.append(seed)
            print(f"recovered seed {seed} ({len(completed)}/{config.games})", flush=True)
            continue
        report_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

        command = [
            str(config.binary),
            "exact-mlx-productive-token-compare",
            "--server-program",
            config.uv,
            "--model-dir",
            str(config.model_dir),
            "--games",
            "1",
            "--first-seed",
            str(seed),
            "--split",
            "final",
            "--rollouts",
            str(config.rollouts),
            "--weights",
            str(config.weights),
            "--output",
            str(report_path),
        ]
        environment = os.environ.copy()
        environment.update({"MCE_LMR": "1", "MCE_DIVERSE_PREFILTER": "1"})
        started_at = timestamp()
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        (logs_dir / f"{seed}.stdout.log").write_text(result.stdout)
        (logs_dir / f"{seed}.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"seed {seed} failed with exit {result.returncode}; "
                f"see {logs_dir / f'{seed}.stderr.log'}"
            )
        report = json.loads(report_path.read_text())
        validate_game_report(report, seed, config.rollouts)
        metadata = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "seed": seed,
            "host": socket.gethostname(),
            "source_revision": source_revision(root),
            "command": command,
            "environment": {
                "MCE_LMR": "1",
                "MCE_DIVERSE_PREFILTER": "1",
            },
            "fingerprints": fingerprints,
            "started_at": started_at,
            "completed_at": timestamp(),
            "wall_seconds": elapsed,
            "report_sha256": sha256_file(report_path),
        }
        write_json_atomic(metadata_path, metadata)
        completed.append(seed)
        write_json_atomic(
            config.output_dir / "shard.json",
            shard_manifest(config, fingerprints, completed, root),
        )
        treatment_mean = report["comparison"]["treatment_mean"]
        print(
            f"completed seed {seed} ({len(completed)}/{config.games}), "
            f"treatment={treatment_mean:.3f}, wall={elapsed:.1f}s",
            flush=True,
        )

    write_json_atomic(
        config.output_dir / "shard.json",
        shard_manifest(config, fingerprints, completed, root),
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_breakdown(records: list[dict[str, Any]], role: str) -> dict[str, Any]:
    scores = [score for record in records for score in record[f"{role}_scores"]]
    count = len(scores)
    return {
        "habitat": [
            statistics.fmean(score["habitat"][index] for score in scores) for index in range(5)
        ],
        "wildlife": [
            statistics.fmean(score["wildlife"][index] for score in scores) for index in range(5)
        ],
        "nature_tokens": statistics.fmean(score["nature_tokens"] for score in scores),
        "base_total": statistics.fmean(score["base_total"] for score in scores),
        "seat_scores": count,
    }


def strategy_statistics(records: list[dict[str, Any]], role: str) -> dict[str, Any]:
    game_means = [
        statistics.fmean(score["base_total"] for score in record[f"{role}_scores"])
        for record in records
    ]
    seat_scores = [
        float(score["base_total"]) for record in records for score in record[f"{role}_scores"]
    ]
    decision_ms = [
        seconds * 1000.0 for record in records for seconds in record[f"{role}_decision_seconds"]
    ]
    mean = statistics.fmean(game_means)
    game_sd = statistics.stdev(game_means) if len(game_means) > 1 else 0.0
    standard_error = game_sd / math.sqrt(len(game_means))
    return {
        "mean_score": mean,
        "game_mean_stddev": game_sd,
        "seat_score_stddev": statistics.stdev(seat_scores) if len(seat_scores) > 1 else 0.0,
        "standard_error": standard_error,
        "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "percentiles": {
            "p10": percentile(seat_scores, 0.10),
            "p50": percentile(seat_scores, 0.50),
            "p90": percentile(seat_scores, 0.90),
        },
        "min_score": min(seat_scores),
        "max_score": max(seat_scores),
        "mean_breakdown": mean_breakdown(records, role),
        "decision_latency": {
            "decisions": len(decision_ms),
            "mean_milliseconds": statistics.fmean(decision_ms),
            "p50_milliseconds": percentile(decision_ms, 0.50),
            "p90_milliseconds": percentile(decision_ms, 0.90),
            "p99_milliseconds": percentile(decision_ms, 0.99),
            "max_milliseconds": max(decision_ms),
        },
        "elapsed_seconds": sum(record[f"{role}_elapsed_seconds"] for record in records),
    }


def subtract_breakdowns(treatment: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "habitat": [treatment["habitat"][index] - baseline["habitat"][index] for index in range(5)],
        "wildlife": [
            treatment["wildlife"][index] - baseline["wildlife"][index] for index in range(5)
        ],
        "nature_tokens": treatment["nature_tokens"] - baseline["nature_tokens"],
        "base_total": treatment["base_total"] - baseline["base_total"],
    }


def collect_inputs(
    directories: Iterable[Path],
    first_seed: int,
    games: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reports: dict[int, dict[str, Any]] = {}
    metadata: dict[int, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for directory in directories:
        manifest_path = directory / "shard.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            raise ValueError(f"incomplete shard: {manifest_path}")
        if manifest.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"experiment identity drifted: {manifest_path}")
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"protocol identity drifted: {manifest_path}")
        manifests.append(manifest)
        for report_path in sorted((directory / "games").glob("[0-9]*.json")):
            if report_path.name.endswith(".meta.json"):
                continue
            seed = int(report_path.stem)
            if seed in reports:
                raise ValueError(f"duplicate final benchmark seed {seed}")
            metadata_path = report_path.with_name(f"{seed}.meta.json")
            report = json.loads(report_path.read_text())
            meta = json.loads(metadata_path.read_text())
            validate_game_report(report, seed, int(manifest["rollouts"]))
            if meta.get("fingerprints") != manifest.get("fingerprints"):
                raise ValueError(f"seed {seed}: metadata fingerprints differ from its shard")
            if meta.get("report_sha256") != sha256_file(report_path):
                raise ValueError(f"seed {seed}: report checksum mismatch")
            reports[seed] = report
            metadata[seed] = meta

    expected = set(range(first_seed, first_seed + games))
    found = set(reports)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"final seed coverage mismatch; missing={missing[:10]} extra={extra[:10]}")
    ordered_reports = [reports[seed] for seed in sorted(expected)]
    ordered_metadata = [metadata[seed] for seed in sorted(expected)]
    records = [report["game_records"][0] for report in ordered_reports]
    return records, ordered_metadata, manifests


def render_markdown(report: dict[str, Any]) -> str:
    treatment = report["treatment"]
    baseline = report["paired_baseline"]
    delta = report["paired_delta"]
    target = "reached" if report["target"]["reached"] else "not reached"
    lines = [
        "# Final Cascadia V2 Strength Validation",
        "",
        f"- Protocol: `{report['protocol_id']}`",
        f"- Games: {report['games']} ({report['seat_games']} treatment seat scores)",
        f"- Seeds: {report['first_seed']} through {report['last_seed']}",
        f"- Treatment: `{report['treatment_id']}`",
        f"- Mean base score: **{treatment['mean_score']:.3f}**",
        (
            f"- Game-block 95% CI: [{treatment['confidence_95'][0]:.3f}, "
            f"{treatment['confidence_95'][1]:.3f}]"
        ),
        f"- 100-point target: **{target}**",
        "",
        "## Paired Canonical V2 Control",
        "",
        f"- Baseline: `{report['baseline_id']}`",
        f"- Baseline mean: {baseline['mean_score']:.3f}",
        f"- Paired delta: {delta['mean']:+.3f}",
        f"- Paired 95% CI: [{delta['confidence_95'][0]:+.3f}, {delta['confidence_95'][1]:+.3f}]",
        f"- Record: {delta['wins']}-{delta['ties']}-{delta['losses']}",
        "",
        "## Score Distribution",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Game-mean SD | {treatment['game_mean_stddev']:.3f} |",
        f"| Seat-score SD | {treatment['seat_score_stddev']:.3f} |",
        f"| Standard error | {treatment['standard_error']:.3f} |",
        f"| P10 | {treatment['percentiles']['p10']:.1f} |",
        f"| P50 | {treatment['percentiles']['p50']:.1f} |",
        f"| P90 | {treatment['percentiles']['p90']:.1f} |",
        "",
        "## Integrity",
        "",
        f"- Complete held-out suite: `{report['integrity']['complete_seed_suite']}`",
        f"- All one-game smoke gates passed: `{report['integrity']['all_smoke_gates_passed']}`",
        f"- All MLX services shut down cleanly: `{report['integrity']['all_clean_shutdown']}`",
        f"- Distinct hosts: {', '.join(sorted(report['host_game_counts']))}",
        f"- Source revisions: {', '.join(report['source_revisions'])}",
    ]
    if report.get("v1_reference"):
        v1 = report["v1_reference"]
        lines.extend(
            [
                "",
                "## Independent V1 Reference",
                "",
                f"- Reproduced v1 mean: {v1['mean_score']:.3f} over {v1['games']} games",
                f"- Absolute treatment-minus-v1 difference: {v1['absolute_difference']:+.3f}",
                "- This is an absolute cross-engine reference, not a paired canonical comparison.",
            ]
        )
    return "\n".join(lines) + "\n"


def aggregate(
    directories: list[Path],
    first_seed: int,
    games: int,
    output: Path,
    markdown_output: Path,
    v1_reference_path: Path | None,
) -> None:
    records, metadata, manifests = collect_inputs(directories, first_seed, games)
    baseline = strategy_statistics(records, "baseline")
    treatment = strategy_statistics(records, "treatment")
    deltas = [
        statistics.fmean(score["base_total"] for score in record["treatment_scores"])
        - statistics.fmean(score["base_total"] for score in record["baseline_scores"])
        for record in records
    ]
    mean_delta = statistics.fmean(deltas)
    delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    delta_se = delta_sd / math.sqrt(len(deltas))
    reports = [
        json.loads((directory / "games" / f"{seed}.json").read_text())
        for directory in directories
        for seed in json.loads((directory / "shard.json").read_text())["completed_seeds"]
    ]
    host_counts = Counter(meta["host"] for meta in metadata)
    source_revisions = sorted({meta["source_revision"] for meta in metadata})
    fingerprint_sets = {json.dumps(meta["fingerprints"], sort_keys=True) for meta in metadata}
    if len(fingerprint_sets) != 1:
        raise ValueError("final benchmark shards do not share one frozen input fingerprint set")
    if len(source_revisions) != 1 or source_revisions == ["unknown"]:
        raise ValueError("final benchmark shards do not share one known source revision")
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "games": games,
        "seat_games": games * 4,
        "first_seed": first_seed,
        "last_seed": first_seed + games - 1,
        "baseline_id": BASELINE_ID,
        "treatment_id": TREATMENT_ID,
        "paired_baseline": baseline,
        "treatment": treatment,
        "paired_delta": {
            "mean": mean_delta,
            "stddev": delta_sd,
            "standard_error": delta_se,
            "confidence_95": [mean_delta - 1.96 * delta_se, mean_delta + 1.96 * delta_se],
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
        },
        "mean_breakdown_delta": subtract_breakdowns(
            treatment["mean_breakdown"], baseline["mean_breakdown"]
        ),
        "target": {
            "mean": 100.0,
            "reached": treatment["mean_score"] >= 100.0,
            "claim_eligible_sample_size": games >= 1000,
        },
        "integrity": {
            "complete_seed_suite": len(records) == games,
            "all_smoke_gates_passed": all(report["gates"]["smoke_passed"] for report in reports),
            "all_clean_shutdown": all(report["clean_shutdown"] for report in reports),
            "single_input_fingerprint_set": len(fingerprint_sets) == 1,
            "single_source_revision": len(source_revisions) == 1,
        },
        "host_game_counts": dict(sorted(host_counts.items())),
        "source_revisions": source_revisions,
        "fingerprints": json.loads(next(iter(fingerprint_sets))),
        "shards": manifests,
        "generated_at": timestamp(),
    }
    if v1_reference_path is not None:
        v1 = json.loads(v1_reference_path.read_text())
        report["v1_reference"] = {
            "path": str(v1_reference_path),
            "games": v1["games"],
            "seat_games": v1["seat_games"],
            "mean_score": v1["mean_score"],
            "confidence_95": v1["ci95"],
            "absolute_difference": treatment["mean_score"] - v1["mean_score"],
            "sha256": sha256_file(v1_reference_path),
        }
    write_json_atomic(output, report)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report))
    print(
        json.dumps(
            {
                "games": games,
                "mean_score": treatment["mean_score"],
                "confidence_95": treatment["confidence_95"],
                "paired_delta": mean_delta,
                "target_reached": report["target"]["reached"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-shard", help="run one resumable seed shard")
    run_parser.add_argument("--binary", type=Path, default=Path("target/release/legacy-teacher"))
    run_parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/legacy-nnue-v4opp-mlx-v1"),
    )
    run_parser.add_argument(
        "--weights",
        type=Path,
        default=Path("nnue_weights_v4opp_modal_iter3.bin"),
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--first-seed", type=int, required=True)
    run_parser.add_argument("--games", type=int, required=True)
    run_parser.add_argument("--rollouts", type=int, default=600)
    run_parser.add_argument("--uv", default="uv")

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="validate and aggregate complete shards"
    )
    aggregate_parser.add_argument("--input", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--first-seed", type=int, required=True)
    aggregate_parser.add_argument("--games", type=int, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--markdown-output", type=Path, required=True)
    aggregate_parser.add_argument("--v1-reference", type=Path)

    args = parser.parse_args()
    if args.command == "run-shard":
        run_shard(
            RunConfig(
                binary=args.binary.resolve(),
                model_dir=args.model_dir.resolve(),
                weights=args.weights.resolve(),
                output_dir=args.output_dir.resolve(),
                first_seed=args.first_seed,
                games=args.games,
                rollouts=args.rollouts,
                uv=args.uv,
            )
        )
    else:
        aggregate(
            directories=[path.resolve() for path in args.input],
            first_seed=args.first_seed,
            games=args.games,
            output=args.output.resolve(),
            markdown_output=args.markdown_output.resolve(),
            v1_reference_path=args.v1_reference.resolve() if args.v1_reference else None,
        )


if __name__ == "__main__":
    main()
